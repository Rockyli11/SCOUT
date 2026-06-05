"""Configuration loading for the SCOUT runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_CACHE_DIR = "~/.cache/scout-router"
RUNTIME_DIR = Path(__file__).resolve().parents[1]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class RoutingConfig:
    strategy: str = "min_agreement_pred_skip"
    tau: float = 0.875
    pred_corr_vote_threshold: float = 0.5
    d6_pred_corr_threshold: float = 0.5
    cheap_pool: tuple[str, ...] = ("d1_rule_based", "d2_lr", "d3_deberta")
    escalation_detector: str = "d6_llm_judge"


@dataclass(frozen=True)
class D6Config:
    provider: str = "openai"
    model: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "D6Config":
        env = environ or os.environ
        provider = env.get("D6_PROVIDER")
        if provider is None and env.get("OPENROUTER_API_KEY") and not env.get("OPENAI_API_KEY"):
            provider = "openai_compatible"
        provider = provider or "openai"
        if provider == "openai":
            api_key = env.get("OPENAI_API_KEY")
            base_url = None
            model = env.get("D6_MODEL", "gpt-4o")
        else:
            api_key = env.get("D6_API_KEY") or env.get("OPENROUTER_API_KEY") or env.get("OPENAI_API_KEY")
            base_url = env.get("D6_BASE_URL") or (OPENROUTER_BASE_URL if env.get("OPENROUTER_API_KEY") else None)
            model = env.get("D6_MODEL", "openai/gpt-4o")
        return cls(provider=provider, model=model, base_url=base_url, api_key=api_key)


@dataclass(frozen=True)
class PredictorConfig:
    backend: str = "vllm"
    base_model: str = "sullivanUCSD/InstinctSCOPE-4B-SFT-CoT_v2"
    adapter: str = "sullivanUCSD/InstinctSCOPE-ckpt-462"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 1.5
    n_samples: int = 1
    seed: int = 0


@dataclass(frozen=True)
class RetrievalConfig:
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    index_detector: str = "d3_deberta"
    top_k: int = 10
    top_k_use: int = 5
    max_seq_length: int = 2048
    chunk_overlap: int = 256
    batch_size: int = 64
    target_batch_tokens: int = 32768
    query_instruction: str = (
        "Instruct: Given raw external content that a language model receives as input, "
        "retrieve the historical prompt injection detection record whose sample description "
        "best matches this content across the following dimensions: "
        "(1) injection category -- whether it is totally benign, an aligned instruction, "
        "a direct misaligned instruction, a hidden or tricky attack embedded in normal content, "
        "a tool misuse attempt, or an exfiltration attempt; "
        "(2) hiding strategy -- whether any attack is explicit, paraphrased, encoded, "
        "context-switching, or structurally camouflaged within the carrier; "
        "(3) carrier type -- the surface form of the content such as plain text, code, "
        "email, HTML, markdown, tool output, tool description, table, or QA context; "
        "(4) attack mechanism -- the specific technique used to override, hijack, or "
        "manipulate the model's goal, if any. "
        "Disregard the topical subject matter of the carrier content itself. "
        "Focus entirely on the structural and semantic properties of any injection signal.\n"
        "Query: "
    )
    document_instruction: str = (
        "Instruct: Represent this prompt injection detection record so that it can be "
        "retrieved by raw input content sharing the same injection category, hiding strategy, "
        "carrier type, and attack mechanism.\n"
        "Query: "
    )


@dataclass(frozen=True)
class ScoutConfig:
    cache_dir: Path = Path(DEFAULT_CACHE_DIR).expanduser()
    detector_manifest: Path = (Path(DEFAULT_CACHE_DIR).expanduser() / "detectors" / "manifest.json")
    detectors_enabled: tuple[str, ...] = (
        "d1_rule_based",
        "d2_lr",
        "d3_deberta",
        "d6_llm_judge",
    )
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    d6: D6Config = field(default_factory=D6Config)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @classmethod
    def from_env(
        cls,
        *,
        env_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> "ScoutConfig":
        if env_path:
            _load_dotenv_file(Path(env_path))
        else:
            _load_dotenv_file(RUNTIME_DIR / ".env")

        resolved_cache = Path(
            cache_dir or os.environ.get("SCOUT_CACHE_DIR", DEFAULT_CACHE_DIR)
        ).expanduser()
        detector_manifest = Path(
            os.environ.get("SCOUT_DETECTOR_MANIFEST", resolved_cache / "detectors" / "manifest.json")
        ).expanduser()
        predictor = PredictorConfig(
            backend=os.environ.get("SCOUT_PREDICTOR_BACKEND", "vllm"),
            base_model=os.environ.get(
                "SCOUT_PREDICTOR_BASE_MODEL",
                "sullivanUCSD/InstinctSCOPE-4B-SFT-CoT_v2",
            ),
            adapter=os.environ.get("SCOUT_PREDICTOR_ADAPTER", "sullivanUCSD/InstinctSCOPE-ckpt-462"),
            max_new_tokens=int(os.environ.get("SCOUT_PREDICTOR_MAX_TOKENS", "4096")),
            temperature=float(os.environ.get("SCOUT_PREDICTOR_TEMPERATURE", "0.2")),
            top_p=float(os.environ.get("SCOUT_PREDICTOR_TOP_P", "0.8")),
            top_k=int(os.environ.get("SCOUT_PREDICTOR_TOP_K", "20")),
            presence_penalty=float(os.environ.get("SCOUT_PREDICTOR_PRESENCE_PENALTY", "1.5")),
            n_samples=int(os.environ.get("SCOUT_PREDICTOR_N_SAMPLES", "1")),
            seed=int(os.environ.get("SCOUT_PREDICTOR_SEED", "0")),
        )
        retrieval = RetrievalConfig(
            embed_model=os.environ.get("SCOUT_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
            index_detector=os.environ.get("SCOUT_RETRIEVAL_INDEX_DETECTOR", "d3_deberta"),
            top_k=int(os.environ.get("SCOUT_RETRIEVAL_TOP_K", "10")),
            top_k_use=int(os.environ.get("SCOUT_RETRIEVAL_TOP_K_USE", "5")),
        )
        return cls(
            cache_dir=resolved_cache,
            detector_manifest=detector_manifest,
            d6=D6Config.from_env(),
            predictor=predictor,
            retrieval=retrieval,
        )

    def with_overrides(self, *, cache_dir: str | Path | None = None) -> "ScoutConfig":
        if cache_dir is None:
            return self
        return replace(self, cache_dir=Path(cache_dir).expanduser())
