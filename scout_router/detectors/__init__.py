"""Built-in SCOUT runtime detectors."""

from scout_router.detectors.registry import REGISTRY, available_detectors, create_detector, register_detector

# Import modules for registration side effects.
from scout_router.detectors import d1_rule_based as _d1_rule_based  # noqa: F401
from scout_router.detectors import d2_embedding_clf as _d2_embedding_clf  # noqa: F401
from scout_router.detectors import d3_deberta as _d3_deberta  # noqa: F401
from scout_router.detectors import d4_attention_tracker as _d4_attention_tracker  # noqa: F401
from scout_router.detectors import d5_align_sentinel as _d5_align_sentinel  # noqa: F401
from scout_router.detectors import d6_llm_judge as _d6_llm_judge  # noqa: F401

__all__ = ["REGISTRY", "available_detectors", "create_detector", "register_detector"]
