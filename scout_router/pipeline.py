"""End-to-end runtime pipeline for one prompt sample."""

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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
    PredictorEstimate,
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
        self._detectors = detectors
        self._predictor = predictor
        self.retriever = retriever or AnchorRetriever(self.config)

    @property
    def detectors(self) -> dict[str, DetectorBase]:
        if self._detectors is None:
            self._detectors = {
                name: create_detector(name, config=self.config) for name in self.config.detectors_enabled
            }
        return self._detectors

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = create_predictor(self.config)
        return self._predictor

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
        stats: AnchorStats,
    ) -> RuntimeDetectorRow:
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

    def _stats_from_prediction_record(self, record: dict[str, Any]) -> dict[str, AnchorStats]:
        stats_by_detector = {}
        for detector, stats in record.get("anchor_stats", {}).items():
            stats_by_detector[str(detector)] = AnchorStats(
                trust=float(stats.get("trust", 0.5)),
                global_trust=float(stats.get("global_trust", 0.5)),
                anchor_lat_ms=float(stats.get("anchor_lat_ms", 0.0)),
            )
        return stats_by_detector

    def _sample_from_prediction_record(self, record: dict[str, Any]) -> PromptSample:
        sample_data = record.get("sample", record)
        return PromptSample(
            id=sample_data.get("id"),
            eval_content=str(sample_data["eval_content"]),
            goal_text=str(sample_data.get("goal_text", "")),
            policy_text=str(sample_data.get("policy_text", "")),
        )

    def _estimates_from_prediction_record(self, record: dict[str, Any]) -> dict[str, PredictorEstimate]:
        estimates = {}
        for detector, estimate in record.get("predictor_outputs", {}).items():
            estimates[str(detector)] = PredictorEstimate(
                detector=str(estimate.get("detector", detector)),
                pred_corr=float(estimate.get("pred_corr", 0.0)),
                pred_risk=float(estimate.get("pred_risk", 0.0)),
                pred_lat_ms=float(estimate.get("pred_lat_ms", 0.0)),
            )
        return estimates

    def build_prediction_record(self, sample: PromptSample) -> dict[str, Any]:
        """Run retrieval and predictor for one sample, returning a reusable JSON record."""
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
        return {
            "sample": {
                "id": sample.id,
                "eval_content": sample.eval_content,
                "goal_text": sample.goal_text,
                "policy_text": sample.policy_text,
            },
            "candidate_detectors": candidate_detectors,
            "predictor_outputs": {
                name: asdict(estimate) for name, estimate in estimates.items()
            },
            "selected_cheap_detectors": selected_cheap_detectors,
            "skipped_cheap_detectors": skipped_cheap_detectors,
            "anchor_stats": {
                detector_name: asdict(self._anchor_stats(detector_name, anchors))
                for detector_name in candidate_detectors
            },
        }

    def close_predictor(self) -> None:
        if self._predictor is None:
            return
        close = getattr(self._predictor, "close", None)
        if close is not None:
            close()
        self._predictor = None

    def _detect_with_prediction_state(
        self,
        sample: PromptSample,
        estimates: dict[str, PredictorEstimate],
        selected_cheap_detectors: list[str],
        skipped_cheap_detectors: list[str],
        stats_by_detector: dict[str, AnchorStats],
        *,
        details: bool,
    ) -> dict[str, Any]:
        cheap_results = self._run_cheap_detectors(sample, selected_cheap_detectors) if selected_cheap_detectors else {}
        sample_id = sample.id or "sample"

        rows: list[RuntimeDetectorRow] = []
        for detector_name in self.config.routing.cheap_pool:
            detector_result = cheap_results.get(detector_name)
            if detector_result is None:
                continue
            estimate = estimates[detector_name]
            stats = stats_by_detector.get(detector_name, AnchorStats(trust=0.5, global_trust=0.5, anchor_lat_ms=0.0))
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
        rows.append(
            self._d6_estimate_row(
                sample_id,
                d6_name,
                estimates[d6_name],
                stats_by_detector.get(d6_name, AnchorStats(trust=0.5, global_trust=0.5, anchor_lat_ms=0.0)),
            )
        )

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
                    name: asdict(estimate) for name, estimate in estimates.items()
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

    def detect_from_prediction_record(
        self,
        record: dict[str, Any],
        *,
        details: bool = False,
    ) -> dict[str, Any]:
        sample = self._sample_from_prediction_record(record)
        candidate_detectors = list(record.get("candidate_detectors", self._candidate_detectors()))
        estimates = self._estimates_from_prediction_record(record)
        self._require_predictor_estimates(estimates, candidate_detectors)
        if "selected_cheap_detectors" in record:
            selected_cheap_detectors = list(record["selected_cheap_detectors"])
        else:
            selected_cheap_detectors = [
                detector_name
                for detector_name in self.config.routing.cheap_pool
                if estimates[detector_name].pred_corr >= self.config.routing.pred_corr_vote_threshold
            ]
        if "skipped_cheap_detectors" in record:
            skipped_cheap_detectors = list(record["skipped_cheap_detectors"])
        else:
            skipped_cheap_detectors = [
                detector_name
                for detector_name in self.config.routing.cheap_pool
                if detector_name not in selected_cheap_detectors
            ]
        return self._detect_with_prediction_state(
            sample,
            estimates,
            selected_cheap_detectors,
            skipped_cheap_detectors,
            self._stats_from_prediction_record(record),
            details=details,
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
        record = self.build_prediction_record(sample)
        return self.detect_from_prediction_record(record, details=details)
