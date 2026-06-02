"""Runtime schemas and label helpers for SCOUT detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

LABEL_TO_ID = {"benign": 0, "attack": 1}
ID_TO_LABEL = {0: "benign", 1: "attack"}


def normalize_label_id(label: int | bool | str) -> int:
    """Normalize detector labels to 0=benign, 1=attack."""
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, int):
        if label in (0, 1):
            return label
        raise ValueError(f"label id must be 0 or 1, got {label!r}")
    text = str(label).strip().lower()
    if text in LABEL_TO_ID:
        return LABEL_TO_ID[text]
    if text in {"0", "false", "safe"}:
        return 0
    if text in {"1", "true", "malicious", "injection"}:
        return 1
    raise ValueError(f"unknown label {label!r}")


def normalize_label_text(label: int | bool | str) -> str:
    """Normalize detector labels to script output labels."""
    return ID_TO_LABEL[normalize_label_id(label)]


@dataclass(frozen=True)
class PromptSample:
    """One runtime prompt sample."""

    eval_content: str
    id: str | None = None
    goal_text: str = ""
    policy_text: str = ""


@dataclass
class DetectorResult:
    """Detector output at the runtime boundary."""

    label: int | str
    confidence: float
    latency_ms: float = 0.0
    detector_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def label_id(self) -> int:
        return normalize_label_id(self.label)

    @property
    def label_text(self) -> str:
        return normalize_label_text(self.label)


@dataclass
class RuntimeDetectorRow:
    """Router row using runtime-only fields."""

    id: str
    detector: str
    pred_label: int
    detector_confidence: float
    latency_ms: float
    pred_corr: float
    pred_risk: float
    pred_lat_ms: float
    trust: float = 0.5
    global_trust: float = 0.5
    anchor_lat_ms: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RuntimeDetectorRow":
        forbidden = {"true_label", "real_corr", "real_lat_ms"}
        bad = sorted(forbidden.intersection(data))
        if bad:
            raise ValueError(f"runtime router rows must not include evaluation fields: {bad}")
        return cls(
            id=str(data["id"]),
            detector=str(data["detector"]),
            pred_label=normalize_label_id(data.get("pred_label", 0)),
            detector_confidence=float(data.get("detector_confidence", 0.0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            pred_corr=float(data.get("pred_corr", 0.0)),
            pred_risk=float(data.get("pred_risk", 0.0)),
            pred_lat_ms=float(data.get("pred_lat_ms", 0.0)),
            trust=float(data.get("trust", 0.5)),
            global_trust=float(data.get("global_trust", 0.5)),
            anchor_lat_ms=float(data.get("anchor_lat_ms", 0.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "detector": self.detector,
            "pred_label": self.pred_label,
            "detector_confidence": self.detector_confidence,
            "latency_ms": self.latency_ms,
            "pred_corr": self.pred_corr,
            "pred_risk": self.pred_risk,
            "pred_lat_ms": self.pred_lat_ms,
            "trust": self.trust,
            "global_trust": self.global_trust,
            "anchor_lat_ms": self.anchor_lat_ms,
        }


@dataclass(frozen=True)
class PredictorEstimate:
    """SCOUT predictor estimate for one detector candidate.

    `trust` and `global_trust` are retained for API compatibility only. The
    runtime fills router rows from anchor fingerprint statistics, not from the
    predictor output.
    """

    detector: str
    pred_corr: float
    pred_risk: float = 0.0
    pred_lat_ms: float = 0.0
    trust: float = 0.5
    global_trust: float = 0.5


PredictorOutput = PredictorEstimate


@dataclass
class RouteDecision:
    """Runtime routing decision."""

    detector: str
    label: int | None
    escalate: bool
    agreement: float | None
    vote: float | None
    threshold: float
    reason: str

    @property
    def label_text(self) -> str | None:
        return None if self.label is None else normalize_label_text(self.label)

    @property
    def route(self) -> str:
        return self.detector

    @property
    def use_d6(self) -> bool:
        return self.escalate
