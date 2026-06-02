"""SCOUT prompt-injection runtime draft."""

from scout_router.config import ScoutConfig
from scout_router.schema import DetectorResult, PromptSample

__all__ = ["DetectorResult", "PromptSample", "ScoutConfig", "ScoutPipeline"]


def __getattr__(name: str):
    if name == "ScoutPipeline":
        from scout_router.pipeline import ScoutPipeline

        return ScoutPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
