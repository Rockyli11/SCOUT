"""Asset locations and inference-only download helpers."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from scout_router.config import DEFAULT_CACHE_DIR

ASSET_REPOS = {
    "predictor_base": "sullivanUCSD/SCOUT-SFT-only",
    "predictor_adapter": "sullivanUCSD/SCOUT",
    "anchor_set": "sullivanUCSD/anchor-400",
    "fingerprints": "sullivanUCSD/fingerprint",
    "test_data": "sullivanUCSD/SCOUT-450",
}

INFERENCE_ALLOW_PATTERNS = {
    "predictor_base": None,
    "predictor_adapter": None,
    "anchor_set": ["*.json", "*.jsonl"],
    "fingerprints": ["*.json", "*.jsonl"],
    "test_data": ["*.json", "*.jsonl", "*.parquet", "*.csv", "README*"],
}

TRAINING_ONLY_FILES = {
    "embeddings_train.npz",
    "train_features.npz",
}


@dataclass(frozen=True)
class AssetSpec:
    name: str
    repo_id: str
    allow_patterns: list[str] | None
    revision: str = "main"
    repo_type: str | None = None


def cache_root(cache_dir: str | Path | None = None) -> Path:
    return Path(cache_dir or os.environ.get("SCOUT_CACHE_DIR", DEFAULT_CACHE_DIR)).expanduser()


def asset_dir(cache_dir: str | Path | None, name: str) -> Path:
    return cache_root(cache_dir) / name


def asset_file(cache_dir: str | Path | None, name: str, filename: str) -> Path:
    return asset_dir(cache_dir, name) / filename


def external_model_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Cache for third-party HF models required by detector inference."""
    path = cache_root(cache_dir) / "hf_models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_asset_specs(*, optional_heavy: bool = False) -> list[AssetSpec]:
    _ = optional_heavy
    names = ["predictor_base", "predictor_adapter", "anchor_set", "fingerprints"]
    return named_asset_specs(names)


def named_asset_specs(names: list[str]) -> list[AssetSpec]:
    specs = []
    for name in names:
        if name not in ASSET_REPOS:
            raise KeyError(f"unknown asset {name!r}; known assets: {sorted(ASSET_REPOS)}")
        repo_type = "dataset" if name in {"anchor_set", "fingerprints", "test_data"} else None
        specs.append(AssetSpec(name, ASSET_REPOS[name], INFERENCE_ALLOW_PATTERNS[name], repo_type=repo_type))
    return specs


def inference_files_for(name: str) -> tuple[str, ...] | None:
    patterns = INFERENCE_ALLOW_PATTERNS[name]
    return None if patterns is None else tuple(patterns)


def verify_sha256(path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected_sha256.lower():
        raise RuntimeError(f"hash mismatch for {path}: expected {expected_sha256}, got {digest}")


def validate_inference_files(name: str, target_dir: Path) -> None:
    training_files = [p for p in target_dir.rglob("*") if p.name in TRAINING_ONLY_FILES]
    if training_files:
        files = ", ".join(str(p.relative_to(target_dir)) for p in training_files)
        raise RuntimeError(f"inference asset {name!r} contains training-only files: {files}")


class AssetDownloader:
    """Hugging Face downloader with v1 inference-file allowlists."""

    def __init__(self, cache_dir: str | Path | None = None, token: str | None = None):
        self.cache_dir = cache_root(cache_dir)
        self.token = token or os.environ.get("HF_TOKEN")

    def download(self, specs: list[AssetSpec]) -> list[Path]:
        from huggingface_hub import snapshot_download

        paths = []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            target = asset_dir(self.cache_dir, spec.name)
            path = Path(
                snapshot_download(
                    repo_id=spec.repo_id,
                    repo_type=spec.repo_type,
                    revision=spec.revision,
                    local_dir=target,
                    token=self.token,
                    allow_patterns=spec.allow_patterns,
                    ignore_patterns=["embeddings_train.npz", "train_features.npz"],
                )
            )
            validate_inference_files(spec.name, path)
            paths.append(path)
        return paths


def ensure_asset(name: str, cache_dir: str | Path | None = None) -> Path:
    path = asset_dir(cache_dir, name)
    if path.exists() and any(path.iterdir()):
        return path
    AssetDownloader(cache_dir).download(named_asset_specs([name]))
    return path
