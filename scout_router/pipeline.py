"""End-to-end runtime pipeline for one prompt sample."""

from __future__ import annotations

import inspect
from typing import Any

from scout_router.config import ScoutConfig
from scout_router.detectors import create_detector
from scout_router.detectors.base import DetectorBase
from scout_router.predictor import create_predictor
from scout_router.retrieval import AnchorRetriever, AnchorStats
from scout_router.router import route_min_agreement_pred_skip
from scout_router.schema import DetectorResult, PromptSample, RuntimeDetectorRow, normalize_label_text


class ScoutPipeline:
    """Run cheap detectors, predictor, router, and optional D6 escalation."""

    def __init__(
        self,
        config: ScoutConfig | None = None,
        *,
        detectors: dict[str, DetectorBase] | None = None,
        predictor: Any | None = None,
        retriever: AnchorRetriever | None = None,
    ):
        self.config = config or ScoutConfig.from_env()
        self.detectors = detectors or {
            name: create_detector(name, config=self.config) for name in self.config.detectors_enabled
        }
        self.predictor = predictor or create_predictor(self.config)
        self.retriever = retriever or AnchorRetriever(self.config)

    def _candidate_detectors(self) -> list[str]:
        names = list(self.config.routing.cheap_pool)
        if self.config.routing.escalation_detector not in names:
            names.append(self.config.routing.escalation_detector)
        return names

    def _fingerprint_maps(self, candidate_detectors: list[str]) -> dict[str, dict]:
        if not hasattr(self.retriever, "fingerprint_map"):
            return {}
        return {
            detector: self.retriever.fingerprint_map(detector)
            for detector in candidate_detectors
        }

    def _anchor_stats(self, detector: str, anchors: list[dict]) -> AnchorStats:
        if hasattr(self.retriever, "stats_for"):
            return self.retriever.stats_for(detector, anchors)
        return AnchorStats(trust=0.5, global_trust=0.5, anchor_lat_ms=0.0)

    def _predict(
        self,
        sample: PromptSample,
        cheap_results: dict[str, DetectorResult],
        anchors: list[dict],
        candidate_detectors: list[str],
        fingerprint_maps: dict[str, dict],
    ):
        signature = inspect.signature(self.predictor.predict)
        accepts_varargs = any(
            param.kind == inspect.Parameter.VAR_POSITIONAL
            for param in signature.parameters.values()
        )
        if accepts_varargs or len(signature.parameters) >= 5:
            return self.predictor.predict(
                sample,
                cheap_results,
                anchors,
                candidate_detectors,
                fingerprint_maps,
            )
        return self.predictor.predict(sample, cheap_results, anchors, candidate_detectors)

    def detect(
        self,
        sample: PromptSample,
        *,
        details: bool = False,
        include_details: bool | None = None,
    ) -> dict[str, Any]:
        if include_details is not None:
            details = include_details
        cheap_results: dict[str, DetectorResult] = {}
        for detector_name in self.config.routing.cheap_pool:
            if detector_name not in self.detectors:
                raise RuntimeError(f"configured cheap detector {detector_name!r} is not enabled")
            cheap_results[detector_name] = self.detectors[detector_name].detect(sample)

        anchors = self.retriever.retrieve(sample)
        candidate_detectors = self._candidate_detectors()
        fingerprint_maps = self._fingerprint_maps(candidate_detectors)
        estimates = self._predict(sample, cheap_results, anchors, candidate_detectors, fingerprint_maps)
        sample_id = sample.id or "sample"

        rows: list[RuntimeDetectorRow] = []
        for detector_name, detector_result in cheap_results.items():
            estimate = estimates.get(detector_name)
            if estimate is None:
                raise RuntimeError(f"predictor omitted {detector_name!r}")
            stats = self._anchor_stats(detector_name, anchors)
            rows.append(
                RuntimeDetectorRow(
                    id=sample_id,
                    detector=detector_name,
                    pred_label=detector_result.label_id,
                    detector_confidence=detector_result.confidence,
                    latency_ms=detector_result.latency_ms,
                    pred_corr=estimate.pred_corr,
                    pred_risk=estimate.pred_risk,
                    pred_lat_ms=estimate.pred_lat_ms,
                    trust=stats.trust,
                    global_trust=stats.global_trust,
                    anchor_lat_ms=stats.anchor_lat_ms,
                )
            )

        d6_name = self.config.routing.escalation_detector
        d6_estimate = estimates.get(d6_name)
        if d6_estimate is None:
            raise RuntimeError(f"predictor omitted escalation detector {d6_name!r}")
        d6_stats = self._anchor_stats(d6_name, anchors)
        rows.append(
            RuntimeDetectorRow(
                id=sample_id,
                detector=d6_name,
                pred_label=0,
                detector_confidence=0.0,
                latency_ms=0.0,
                pred_corr=d6_estimate.pred_corr,
                pred_risk=d6_estimate.pred_risk,
                pred_lat_ms=d6_estimate.pred_lat_ms,
                trust=d6_stats.trust,
                global_trust=d6_stats.global_trust,
                anchor_lat_ms=d6_stats.anchor_lat_ms,
            )
        )

        decision = route_min_agreement_pred_skip(rows, self.config.routing)
        d6_result = None
        if decision.escalate:
            if d6_name not in self.detectors:
                raise RuntimeError(f"D6 escalation selected but {d6_name!r} is not enabled")
            d6_result = self.detectors[d6_name].detect(sample)
            label = d6_result.label_text
        else:
            label = normalize_label_text(decision.label)

        output: dict[str, Any] = {"label": label}
        if sample.id is not None:
            output = {"id": sample.id, **output}
        if details:
            output["details"] = {
                "route": decision.detector,
                "d6_used": decision.escalate,
                "threshold": decision.threshold,
                "agreement": decision.agreement,
                "vote": decision.vote,
                "reason": decision.reason,
                "detector_outputs": {
                    name: {
                        "label": result.label_text,
                        "confidence": result.confidence,
                        "latency_ms": result.latency_ms,
                        "raw": result.raw,
                    }
                    for name, result in cheap_results.items()
                },
                "predictor_outputs": {
                    name: estimate.__dict__ for name, estimate in estimates.items()
                },
                "router_rows": [row.as_dict() for row in rows],
            }
            if d6_result is not None:
                output["details"]["detector_outputs"][d6_name] = {
                    "label": d6_result.label_text,
                    "confidence": d6_result.confidence,
                    "latency_ms": d6_result.latency_ms,
                    "raw": d6_result.raw,
                }
        return output
