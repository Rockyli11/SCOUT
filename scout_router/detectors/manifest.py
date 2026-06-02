"""Local detector artifact manifest loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scout_router.config import ScoutConfig
from scout_router.detectors.base import DetectorNotReady

TRAIN_COMMAND = (
    "python scout_runtime/train_detectors.py --detectors d2,d3,d4,d5 "
    "--train-data data/train_set.jsonl --artifact-dir ~/.cache/scout-router/detectors"
)


@dataclass(frozen=True)
class DetectorManifest:
    path: Path
    detectors: dict[str, dict[str, Any]]
    schema_version: int = 1

    def entry(self, detector_name: str, expected_type: str) -> dict[str, Any]:
        if detector_name not in self.detectors:
            raise DetectorNotReady(
                f"Detector {detector_name!r} is not registered in {self.path}. "
                f"Train local detector artifacts first: {TRAIN_COMMAND}"
            )
        entry = self.detectors[detector_name]
        found_type = entry.get("type")
        if found_type != expected_type:
            raise DetectorNotReady(
                f"Detector {detector_name!r} in {self.path} has type {found_type!r}, "
                f"expected {expected_type!r}."
            )
        return entry

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        return path


def load_detector_manifest(config: ScoutConfig) -> DetectorManifest:
    path = Path(config.detector_manifest).expanduser()
    if not path.exists():
        raise DetectorNotReady(
            f"Detector manifest not found at {path}. Set SCOUT_DETECTOR_MANIFEST or train "
            f"local detector artifacts first: {TRAIN_COMMAND}"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DetectorNotReady(f"Detector manifest is not valid JSON: {path}") from exc
    if int(data.get("schema_version", 0)) != 1:
        raise DetectorNotReady(f"Unsupported detector manifest schema at {path}: {data.get('schema_version')!r}")
    detectors = data.get("detectors")
    if not isinstance(detectors, dict):
        raise DetectorNotReady(f"Detector manifest must contain a detectors object: {path}")
    return DetectorManifest(path=path, detectors=detectors, schema_version=1)


def require_manifest_file(config: ScoutConfig, detector_name: str, expected_type: str, key: str) -> Path:
    manifest = load_detector_manifest(config)
    entry = manifest.entry(detector_name, expected_type)
    if key not in entry:
        raise DetectorNotReady(f"Detector {detector_name!r} manifest entry is missing {key!r}: {manifest.path}")
    path = manifest.resolve_path(entry[key])
    if not path.exists():
        raise DetectorNotReady(
            f"Detector {detector_name!r} artifact is missing: {path}. "
            f"Regenerate it with: {TRAIN_COMMAND}"
        )
    return path


def require_manifest_dir(config: ScoutConfig, detector_name: str, expected_type: str, key: str) -> Path:
    path = require_manifest_file(config, detector_name, expected_type, key)
    if not path.is_dir():
        raise DetectorNotReady(f"Detector {detector_name!r} artifact is not a directory: {path}")
    return path
