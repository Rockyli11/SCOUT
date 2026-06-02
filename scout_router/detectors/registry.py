"""Detector registry for runtime construction."""

from __future__ import annotations

import inspect
from typing import Any

from scout_router.detectors.base import DetectorBase

REGISTRY: dict[str, type[DetectorBase]] = {}


def register_detector(name: str):
    def decorator(cls: type[DetectorBase]) -> type[DetectorBase]:
        REGISTRY[name] = cls
        return cls

    return decorator


def create_detector(name: str, **kwargs: Any) -> DetectorBase:
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r}; known detectors: {sorted(REGISTRY)}")
    cls = REGISTRY[name]
    signature = inspect.signature(cls)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return cls(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return cls(**accepted)


def available_detectors() -> list[str]:
    return sorted(REGISTRY)
