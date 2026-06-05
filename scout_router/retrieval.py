"""Qwen3-Embedding anchor retrieval and anchor-stat trust estimates."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scout_router.assets import asset_dir, ensure_asset
from scout_router.config import ScoutConfig
from scout_router.schema import PromptSample


CACHE_VERSION = "v2"


@dataclass(frozen=True)
class AnchorStats:
    """Anchor-derived reliability estimates for one detector."""

    trust: float
    global_trust: float
    anchor_lat_ms: float


@dataclass
class PreparedText:
    index: int
    text: str
    total_tokens: int
    chunks: list[str]

    @property
    def is_chunked(self) -> bool:
        return len(self.chunks) > 1


def _normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec)
    return vec if denom == 0 else vec / denom


def _chunk_text(
    text: str,
    tokenizer,
    instruction: str,
    max_seq_length: int,
    chunk_overlap: int,
) -> tuple[list[str], int]:
    prompt_tokens = len(tokenizer(instruction, add_special_tokens=False, truncation=False)["input_ids"])
    special_token_slack = 8
    content_budget = max(1, max_seq_length - prompt_tokens - special_token_slack)
    tokenized = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    total_tokens = prompt_tokens + len(tokenized) + 1
    if total_tokens <= max_seq_length:
        return [text], total_tokens

    overlap = min(chunk_overlap, max(0, content_budget - 1))
    step = max(1, content_budget - overlap)
    chunks: list[str] = []
    for start in range(0, len(tokenized), step):
        chunk_ids = tokenized[start:start + content_budget]
        if not chunk_ids:
            continue
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if start + content_budget >= len(tokenized):
            break
    return chunks, total_tokens


def _prepare_texts(
    texts: list[str],
    tokenizer,
    instruction: str,
    max_seq_length: int,
    chunk_overlap: int,
) -> list[PreparedText]:
    prepared: list[PreparedText] = []
    for index, text in enumerate(texts):
        chunks, total_tokens = _chunk_text(
            text=text,
            tokenizer=tokenizer,
            instruction=instruction,
            max_seq_length=max_seq_length,
            chunk_overlap=chunk_overlap,
        )
        prepared.append(PreparedText(index=index, text=text, total_tokens=total_tokens, chunks=chunks))
    return prepared


def _encode_with_retry(texts: list[str], embedder, instruction: str, batch_size: int) -> np.ndarray:
    import torch

    try:
        return embedder.encode(
            texts,
            prompt=instruction,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except torch.OutOfMemoryError:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if len(texts) <= 1:
            raise
        midpoint = max(1, len(texts) // 2)
        left = _encode_with_retry(texts[:midpoint], embedder, instruction, max(1, min(batch_size, midpoint)))
        right = _encode_with_retry(
            texts[midpoint:],
            embedder,
            instruction,
            max(1, min(batch_size, len(texts) - midpoint)),
        )
        return np.concatenate([left, right], axis=0)


def _batched_prepared_texts(
    prepared: list[PreparedText],
    batch_size: int,
    target_batch_tokens: int,
):
    batch: list[PreparedText] = []
    batch_tokens = 0
    for item in prepared:
        would_exceed_count = len(batch) >= batch_size
        would_exceed_tokens = batch and batch_tokens + item.total_tokens > target_batch_tokens
        if would_exceed_count or would_exceed_tokens:
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(item)
        batch_tokens += item.total_tokens
    if batch:
        yield batch


def embed_texts(
    texts: list[str],
    embedder,
    *,
    instruction: str,
    batch_size: int,
    max_seq_length: int,
    chunk_overlap: int,
    target_batch_tokens: int,
) -> np.ndarray:
    tokenizer = embedder.tokenizer
    prepared = _prepare_texts(
        texts=texts,
        tokenizer=tokenizer,
        instruction=instruction,
        max_seq_length=max_seq_length,
        chunk_overlap=chunk_overlap,
    )
    embeddings = np.zeros((len(texts), embedder.get_embedding_dimension()), dtype=np.float32)
    regular_items = [item for item in prepared if not item.is_chunked]
    chunked_items = [item for item in prepared if item.is_chunked]

    for batch in _batched_prepared_texts(regular_items, batch_size, target_batch_tokens):
        batch_embs = _encode_with_retry(
            [item.text for item in batch],
            embedder,
            instruction,
            batch_size=min(batch_size, len(batch)),
        )
        for item, emb in zip(batch, batch_embs):
            embeddings[item.index] = emb.astype(np.float32, copy=False)

    for item in chunked_items:
        chunk_batch_size = min(batch_size, 4, len(item.chunks))
        chunk_embs = _encode_with_retry(
            item.chunks,
            embedder,
            instruction,
            batch_size=max(1, chunk_batch_size),
        )
        embeddings[item.index] = _normalize(chunk_embs.mean(axis=0).astype(np.float32, copy=False))
    return embeddings


class AnchorRetriever:
    """Retrieve top-k anchor fingerprints with the main-code Qwen3 embedding flow."""

    def __init__(self, config: ScoutConfig | None = None, top_k: int | None = None):
        self.config = config or ScoutConfig.from_env()
        self.top_k = top_k or self.config.retrieval.top_k
        self._fingerprints_by_detector: dict[str, dict[str, dict[str, Any]]] | None = None
        self._fingerprint_entries: dict[str, list[dict[str, Any]]] | None = None
        self._embedder = None
        self._anchor_index: tuple[np.ndarray, list[str]] | None = None

    def _fingerprint_root(self) -> Path:
        root = asset_dir(self.config.cache_dir, "fingerprints")
        if root.exists() and any(root.iterdir()):
            return root
        repo_root = Path(__file__).resolve().parents[2]
        local_root = repo_root / "data" / "model_fingerprint"
        if local_root.exists() and any(local_root.iterdir()):
            return local_root
        return ensure_asset("fingerprints", self.config.cache_dir)

    def _load_fingerprint_entries(self) -> dict[str, list[dict[str, Any]]]:
        if self._fingerprint_entries is not None:
            return self._fingerprint_entries
        root = self._fingerprint_root()
        entries: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(root.rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                detector = path.stem
                entries[detector] = [row for row in data if isinstance(row, dict) and row.get("id")]
        for path in sorted(root.rglob("*.jsonl")):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows:
                entries[path.stem] = [row for row in rows if isinstance(row, dict) and row.get("id")]
        self._fingerprint_entries = entries
        return entries

    def fingerprint_map(self, detector: str) -> dict[str, dict[str, Any]]:
        if self._fingerprints_by_detector is None:
            self._fingerprints_by_detector = {}
        if detector not in self._fingerprints_by_detector:
            entries = self._load_fingerprint_entries().get(detector, [])
            self._fingerprints_by_detector[detector] = {str(row["id"]): row for row in entries}
        return self._fingerprints_by_detector[detector]

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        from sentence_transformers import SentenceTransformer
        import torch

        model_ref = Path(self.config.retrieval.embed_model).expanduser()
        if self.config.retrieval.embed_model == "Qwen/Qwen3-Embedding-0.6B":
            repo_root = Path(__file__).resolve().parents[2]
            local_model = repo_root / "sft" / "model" / "qwen3-embedding-0.6b"
            if local_model.exists():
                model_ref = local_model
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model_arg = str(model_ref) if model_ref.exists() else self.config.retrieval.embed_model
        self._embedder = SentenceTransformer(
            model_arg,
            device=device,
            trust_remote_code=True,
            cache_folder=str(self.config.cache_dir / "hf_models"),
        )
        self._embedder.max_seq_length = self.config.retrieval.max_seq_length
        return self._embedder

    def _cache_name(self, stem: str, suffix: str) -> str:
        cfg = self.config.retrieval
        return f"{stem}_{suffix}_{CACHE_VERSION}_ms{cfg.max_seq_length}_ov{cfg.chunk_overlap}.npz"

    def _build_anchor_index(self) -> tuple[np.ndarray, list[str]]:
        if self._anchor_index is not None:
            return self._anchor_index
        cfg = self.config.retrieval
        entries = self._load_fingerprint_entries().get(cfg.index_detector, [])
        if not entries:
            raise RuntimeError(f"no fingerprints found for retrieval index detector {cfg.index_detector!r}")
        anchor_ids = [str(entry["id"]) for entry in entries]
        index_dir = self.config.cache_dir / "fingerprint_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        cache_path = index_dir / self._cache_name(cfg.index_detector, "anchor_index")
        if cache_path.exists():
            cached = np.load(cache_path)
            self._anchor_index = (cached["embeddings"], anchor_ids)
            return self._anchor_index

        embedder = self._load_embedder()
        texts = [str(entry.get("sample_characteristics", "")) for entry in entries]
        embeddings = embed_texts(
            texts,
            embedder,
            instruction=cfg.document_instruction,
            batch_size=cfg.batch_size,
            max_seq_length=cfg.max_seq_length,
            chunk_overlap=cfg.chunk_overlap,
            target_batch_tokens=cfg.target_batch_tokens,
        )
        np.savez(cache_path, embeddings=embeddings)
        self._anchor_index = (embeddings, anchor_ids)
        return self._anchor_index

    def retrieve(self, sample: PromptSample, top_k: int | None = None) -> list[dict[str, Any]]:
        cfg = self.config.retrieval
        anchor_embs, anchor_ids = self._build_anchor_index()
        embedder = self._load_embedder()
        query_embs = embed_texts(
            [sample.eval_content],
            embedder,
            instruction=cfg.query_instruction,
            batch_size=1,
            max_seq_length=cfg.max_seq_length,
            chunk_overlap=cfg.chunk_overlap,
            target_batch_tokens=cfg.target_batch_tokens,
        )
        similarity = query_embs @ anchor_embs.T
        top_indices = np.argsort(similarity[0])[::-1][: top_k or self.top_k]
        return [{"id": anchor_ids[i], "score": float(similarity[0][i])} for i in top_indices]

    def stats_for(self, detector: str, anchors: list[dict[str, Any]]) -> AnchorStats:
        fp_map = self.fingerprint_map(detector)
        anchor_ids = [str(anchor.get("id")) for anchor in anchors if anchor.get("id")]
        corr_vals = [1.0 if fp_map[anchor_id].get("correct") else 0.0 for anchor_id in anchor_ids if anchor_id in fp_map]
        lat_vals = [
            float(fp_map[anchor_id]["latency_ms"])
            for anchor_id in anchor_ids
            if anchor_id in fp_map and fp_map[anchor_id].get("latency_ms") is not None
        ]
        all_corr = [1.0 if row.get("correct") else 0.0 for row in fp_map.values()]
        return AnchorStats(
            trust=float(np.mean(corr_vals)) if corr_vals else 0.5,
            global_trust=float(np.mean(all_corr)) if all_corr else 0.5,
            anchor_lat_ms=float(np.mean(lat_vals)) if lat_vals else 0.0,
        )
