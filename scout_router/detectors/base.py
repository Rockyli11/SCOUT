"""Common detector interface for the SCOUT runtime."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from scout_router.schema import DetectorResult, PromptSample


class DetectorNotReady(RuntimeError):
    """Raised when a detector's runtime assets or credentials are missing."""


class DetectorBase(ABC):
    """Base detector interface."""

    name: str = "base"
    cost_tier: str = "cheap"

    @abstractmethod
    def _detect(self, sample: PromptSample) -> DetectorResult:
        ...

    def detect(self, sample: PromptSample) -> DetectorResult:
        t0 = time.perf_counter()
        result = self._detect(sample)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.detector_name = self.name
        return result

    def detect_batch(self, samples: list[PromptSample]) -> list[DetectorResult]:
        return [self.detect(sample) for sample in samples]
