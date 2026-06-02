"""Runtime implementation of min_agreement_pred_skip."""

from __future__ import annotations

from collections.abc import Iterable

from scout_router.config import RoutingConfig
from scout_router.schema import RouteDecision, RuntimeDetectorRow


def effective_trust(row: RuntimeDetectorRow) -> float:
    """Experiment-compatible effective trust."""
    return 0.6 * row.trust + 0.4 * row.global_trust


def route_min_agreement_pred_skip(
    rows: Iterable[RuntimeDetectorRow | dict],
    config: RoutingConfig | None = None,
) -> RouteDecision:
    cfg = config or RoutingConfig()
    if cfg.strategy != "min_agreement_pred_skip":
        raise ValueError(f"v1 only supports min_agreement_pred_skip, got {cfg.strategy!r}")

    runtime_rows = [
        row if isinstance(row, RuntimeDetectorRow) else RuntimeDetectorRow.from_mapping(row)
        for row in rows
    ]
    cheap_rows = [row for row in runtime_rows if row.detector in cfg.cheap_pool]
    d6_rows = [row for row in runtime_rows if row.detector == cfg.escalation_detector]

    qualified = [row for row in cheap_rows if row.pred_corr >= cfg.pred_corr_vote_threshold]
    if not qualified:
        return RouteDecision(
            detector=cfg.escalation_detector,
            label=None,
            escalate=True,
            agreement=None,
            vote=None,
            threshold=cfg.tau,
            reason="no_qualified_cheap_voters",
        )

    weights = [effective_trust(row) for row in qualified]
    denom = sum(weights)
    vote = (
        sum(weight * row.pred_label for weight, row in zip(weights, qualified)) / denom
        if denom > 0.0
        else 0.5
    )
    agreement = max(vote, 1.0 - vote)
    d6_pred_corr = d6_rows[0].pred_corr if d6_rows else 0.0

    if agreement < cfg.tau and d6_pred_corr >= cfg.d6_pred_corr_threshold:
        return RouteDecision(
            detector=cfg.escalation_detector,
            label=None,
            escalate=True,
            agreement=agreement,
            vote=vote,
            threshold=cfg.tau,
            reason="low_agreement_d6_pred_corr_passed",
        )

    return RouteDecision(
        detector="ensemble_cheap",
        label=1 if vote > 0.5 else 0,
        escalate=False,
        agreement=agreement,
        vote=vote,
        threshold=cfg.tau,
        reason="cheap_ensemble",
    )
