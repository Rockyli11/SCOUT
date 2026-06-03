"""End-to-end runtime pipeline for one prompt sample."""

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from scout_router.config import ScoutConfig
from scout_router.detectors import create_detector
from scout_router.detectors.base import DetectorBase
from scout_router.predictor import create_predictor
from scout_router.retrieval import AnchorRetriever, AnchorStats
from scout_router.router import route_min_agreement_pred_skip
from scout_router.schema import (
    DetectorResult,
    PromptSample,
    RouteDecision,
    RuntimeDetectorRow,
    normalize_label_text,
)


class ScoutPipeline:
    """Run predictor-selected cheap detectors, router, and optional D6 escalation."""

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

    def _require_predictor_estimates(self, estimates: dict[str, Any], candidate_detectors: list[str]) -> None:
        for detector_name in candidate_detectors:
            if detector_name not in estimates:
                raise RuntimeError(f"predictor omitted {detector_name!r}")

    def _run_cheap_detectors(
        self,
        sample: PromptSample,
        detector_names: list[str],
    ) -> dict[str, DetectorResult]:
        for detector_name in detector_names:
            if detector_name not in self.detectors:
                raise RuntimeError(f"configured cheap detector {detector_name!r} is not enabled")

        with ThreadPoolExecutor(max_workers=len(detector_names)) as executor:
            futures = {
                detector_name: executor.submit(self.detectors[detector_name].detect, sample)
                for detector_name in detector_names
            }
            return {
                detector_name: futures[detector_name].result()
                for detector_name in detector_names
            }

    def _d6_estimate_row(
        self,
        sample_id: str,
        detector_name: str,
        estimate: Any,
        anchors: list[dict],
    ) -> RuntimeDetectorRow:
        stats = self._anchor_stats(detector_name, anchors)
        return RuntimeDetectorRow(
            id=sample_id,
            detector=detector_name,
            pred_label=0,
            detector_confidence=0.0,
            latency_ms=0.0,
            pred_corr=estimate.pred_corr,
            pred_risk=estimate.pred_risk,
            pred_lat_ms=estimate.pred_lat_ms,
            trust=stats.trust,
            global_trust=stats.global_trust,
            anchor_lat_ms=stats.anchor_lat_ms,
        )

    def detect(
        self,
        sample: PromptSample,
        *,
        details: bool = False,
        include_details: bool | None = None,
    ) -> dict[str, Any]:
        if include_details is not None:
            details = include_details
        anchors = self.retriever.retrieve(sample)
        candidate_detectors = self._candidate_detectors()
        fingerprint_maps = self._fingerprint_maps(candidate_detectors)
        estimates = self._predict(sample, {}, anchors, candidate_detectors, fingerprint_maps)
        self._require_predictor_estimates(estimates, candidate_detectors)

        selected_cheap_detectors = [
            detector_name
            for detector_name in self.config.routing.cheap_pool
            if estimates[detector_name].pred_corr >= self.config.routing.pred_corr_vote_threshold
        ]
        skipped_cheap_detectors = [
            detector_name
            for detector_name in self.config.routing.cheap_pool
            if detector_name not in selected_cheap_detectors
        ]
        cheap_results = self._run_cheap_detectors(sample, selected_cheap_detectors) if selected_cheap_detectors else {}
        sample_id = sample.id or "sample"

        rows: list[RuntimeDetectorRow] = []
        for detector_name in self.config.routing.cheap_pool:
            detector_result = cheap_results.get(detector_name)
            if detector_result is None:
                continue
            estimate = estimates[detector_name]
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
        rows.append(self._d6_estimate_row(sample_id, d6_name, estimates[d6_name], anchors))

        d6_result = None
        if not selected_cheap_detectors:
            decision = RouteDecision(
                detector=d6_name,
                label=None,
                escalate=True,
                agreement=None,
                vote=None,
                threshold=self.config.routing.tau,
                reason="no_selected_cheap_detectors",
            )
            if d6_name not in self.detectors:
                raise RuntimeError(f"D6 escalation selected but {d6_name!r} is not enabled")
            d6_result = self.detectors[d6_name].detect(sample)
            label = d6_result.label_text
        else:
            decision = route_min_agreement_pred_skip(rows, self.config.routing)
        if decision.escalate and d6_result is None:
            if d6_name not in self.detectors:
                raise RuntimeError(f"D6 escalation selected but {d6_name!r} is not enabled")
            d6_result = self.detectors[d6_name].detect(sample)
            label = d6_result.label_text
        elif not decision.escalate:
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
                "selected_cheap_detectors": selected_cheap_detectors,
                "skipped_cheap_detectors": skipped_cheap_detectors,
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
