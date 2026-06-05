"""SCOUT predictor prompt builder, output parser, and backends."""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

from scout_router.assets import ASSET_REPOS, ensure_asset
from scout_router.config import ScoutConfig
from scout_router.schema import DetectorResult, PredictorEstimate, PromptSample

REASONED_SYSTEM_PROMPT = """\
You are a performance predictor for prompt injection detectors. Given a detector profile and its historical performance records on similar samples, predict how the detector will perform on the target sample.

First, write an Analysis section reasoning through the evidence: what the detector does and where it tends to fail, what kind of sample this is, and what the historical records reveal about the detector's behavior on similar inputs, and how the detector will perform here.

Then end your response with this exact line:
Predicted Performance: {"correctness": "yes/no", "risk": N, "latency": "XX.X ms"}

Where:
- correctness: "yes" if the detector will correctly classify this sample, "no" otherwise
- risk: integer 0-5 (0=definitely benign, 5=definitely attack), your independent assessment
- latency: predicted processing time in milliseconds, one decimal place
"""

HISTORICAL_RECORD_TEMPLATE = """\
### Record {i}
{sample_characteristics}
Outcome: {prediction_result}\
"""

ANCHOR_USER_TEMPLATE = """\
## Detector
{detector_profile}

## Historical Records (5 most similar samples, ranked by similarity)

{historical_records}

## Target Sample
Goal: {goal_text}
Policy: {policy_text}
Content:
{eval_content}
"""

ANSWER_RE = re.compile(
    r'(?:So the predicted performance is:|Predicted Performance:)\s*(\{[^{}]*"correctness"[^{}]*\})',
    re.DOTALL,
)
ANSWER_RE_FALLBACK = re.compile(r'\{[^{}]*"correctness"[^{}]*\}', re.DOTALL)
LATENCY_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def _normalize_latency_json(raw: str) -> str:
    return re.sub(
        r'"latency"\s*:\s*([^",}\s][^,}]*)',
        lambda match: f'"latency": "{match.group(1).strip()}"',
        raw,
    )


def _parse_latency_ms(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = LATENCY_RE.search(str(value))
    return float(match.group(1)) if match else None


def parse_structured_prediction(text: str) -> dict[str, Any]:
    """Parse the main-code Predicted Performance JSON payload."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        matches = ANSWER_RE_FALLBACK.findall(text)
    parsed: dict[str, Any] = {
        "pred_correctness": None,
        "pred_risk": None,
        "pred_latency_ms": None,
        "parse_failed": True,
    }
    if not matches:
        return parsed

    raw = matches[-1]
    for candidate in (raw, _normalize_latency_json(raw)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        correctness = obj.get("correctness", obj.get("correct"))
        if isinstance(correctness, str):
            pred_correctness = 1 if correctness.strip().lower() in {"yes", "1", "true"} else 0
        elif isinstance(correctness, bool):
            pred_correctness = int(correctness)
        elif isinstance(correctness, int) and correctness in {0, 1}:
            pred_correctness = correctness
        else:
            continue

        risk = obj.get("risk")
        try:
            pred_risk = int(risk)
        except (TypeError, ValueError):
            pred_risk = None
        if pred_risk is not None and not 0 <= pred_risk <= 5:
            pred_risk = None

        return {
            "pred_correctness": pred_correctness,
            "pred_risk": pred_risk,
            "pred_latency_ms": _parse_latency_ms(obj.get("latency")),
            "parse_failed": False,
        }
    return parsed


def parse_predictor_output(text: str, detector: str = "detector") -> dict[str, PredictorEstimate]:
    """Parse predictor text into a single detector estimate.

    The canonical runtime format is the main-code `Predicted Performance` line.
    A legacy JSON object with `detectors` is still accepted so older smoke tests
    fail softly during migration, but prompts generated here never request it.
    """
    parsed = parse_structured_prediction(text)
    if not parsed["parse_failed"]:
        return {
            detector: PredictorEstimate(
                detector=detector,
                pred_corr=float(parsed["pred_correctness"]),
                pred_risk=float(parsed["pred_risk"] or 0.0),
                pred_lat_ms=float(parsed["pred_latency_ms"] or 0.0),
            )
        }

    body = text.strip()
    if body.startswith("```"):
        parts = body.split("```")
        if len(parts) >= 3:
            body = parts[1].strip()
            if body.lower().startswith("json"):
                body = body[4:].strip()
    data = json.loads(body)
    if isinstance(data, dict) and "detectors" in data:
        items = data["detectors"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [{"detector": key, **value} for key, value in data.items() if isinstance(value, dict)]
    else:
        raise ValueError("predictor output must be a Predicted Performance line or JSON object")

    estimates: dict[str, PredictorEstimate] = {}
    for item in items:
        name = str(item.get("detector", detector))
        legacy_latency = _parse_latency_ms(item.get("pred_lat_ms", item.get("latency_ms", item.get("latency", 0.0))))
        estimates[name] = PredictorEstimate(
            detector=name,
            pred_corr=float(item.get("pred_corr", item.get("correctness", 0.0))),
            pred_risk=float(item.get("pred_risk", item.get("risk", 0.0))),
            pred_lat_ms=float(legacy_latency or 0.0),
        )
    if not estimates:
        raise ValueError("predictor output contained no detector estimates")
    return estimates


def build_historical_records(
    anchor_ids: list[str],
    fp_map: dict[str, dict[str, Any]],
    *,
    top_k_use: int = 5,
) -> str:
    records = []
    for i, anchor_id in enumerate(anchor_ids[:top_k_use], 1):
        fp = fp_map.get(anchor_id)
        if fp is None:
            continue
        records.append(
            HISTORICAL_RECORD_TEMPLATE.format(
                i=i,
                sample_characteristics=str(fp.get("sample_characteristics", "")).strip(),
                prediction_result=str(fp.get("prediction_result", "")).strip(),
            )
        )
    return "\n\n".join(records)


def detector_profile_from_fingerprints(anchor_ids: list[str], fp_map: dict[str, dict[str, Any]]) -> str:
    for anchor_id in anchor_ids:
        fp = fp_map.get(anchor_id)
        if fp and fp.get("detector_profile"):
            return str(fp["detector_profile"]).strip()
    for fp in fp_map.values():
        if fp.get("detector_profile"):
            return str(fp["detector_profile"]).strip()
    return ""


def build_prediction_messages(
    sample: PromptSample,
    detector: str,
    anchors: list[dict[str, Any]],
    fp_map: dict[str, dict[str, Any]],
    *,
    top_k_use: int = 5,
    max_eval_chars: int = 3000,
) -> list[dict[str, str]]:
    anchor_ids = [str(anchor.get("id")) for anchor in anchors if anchor.get("id")]
    user = ANCHOR_USER_TEMPLATE.format(
        detector_profile=detector_profile_from_fingerprints(anchor_ids, fp_map),
        historical_records=build_historical_records(anchor_ids, fp_map, top_k_use=top_k_use),
        goal_text=sample.goal_text,
        policy_text=sample.policy_text,
        eval_content=sample.eval_content[:max_eval_chars],
    )
    return [
        {"role": "system", "content": REASONED_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_prediction_prompt(
    sample: PromptSample,
    detector: str,
    anchors: list[dict[str, Any]],
    fp_map: dict[str, dict[str, Any]],
    *,
    top_k_use: int = 5,
) -> str:
    """Return a readable prompt for tests and non-chat fallback callers."""
    messages = build_prediction_messages(sample, detector, anchors, fp_map, top_k_use=top_k_use)
    return "\n\n".join(f"{msg['role'].upper()}:\n{msg['content']}" for msg in messages)


class ScoutPredictor:
    """vLLM backend for the published SCOUT predictor base model and adapter."""

    def __init__(self, config: ScoutConfig):
        self.config = config
        self._llm = None
        self._tokenizer = None
        self._base_model_ref: str | None = None
        self._adapter_ref: str | None = None

    def _resolve_asset_ref(self, asset_name: str, configured_value: str) -> str:
        configured_path = Path(configured_value).expanduser()
        if configured_path.exists():
            return str(configured_path)
        if configured_value == ASSET_REPOS[asset_name]:
            return str(ensure_asset(asset_name, self.config.cache_dir))
        return configured_value

    @property
    def base_model_ref(self) -> str:
        if self._base_model_ref is None:
            self._base_model_ref = self._resolve_asset_ref(
                "predictor_base",
                self.config.predictor.base_model,
            )
        return self._base_model_ref

    @property
    def adapter_ref(self) -> str:
        if self._adapter_ref is None:
            self._adapter_ref = self._resolve_asset_ref(
                "predictor_adapter",
                self.config.predictor.adapter,
            )
        return self._adapter_ref

    def _load_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "SCOUT predictor requires vllm. Install with: pip install vllm"
            ) from exc
        self._llm = LLM(
            model=self.base_model_ref,
            dtype="bfloat16",
            gpu_memory_utilization=0.75,
            max_model_len=4096,
            trust_remote_code=True,
            enable_lora=True,
            max_loras=1,
            max_lora_rank=128,
            download_dir=str(self.config.cache_dir / "hf_models"),
        )
        return self._llm

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_ref,
            trust_remote_code=True,
            cache_dir=str(self.config.cache_dir / "hf_models"),
        )
        return self._tokenizer

    def close(self) -> None:
        """Release object-owned vLLM resources before loading detector models."""
        llm = self._llm
        for owner_name, method_name in (
            (None, "shutdown"),
            (None, "close"),
            ("llm_engine", "shutdown"),
            ("llm_engine", "close"),
        ):
            owner = llm if owner_name is None else getattr(llm, owner_name, None)
            method = getattr(owner, method_name, None)
            if callable(method):
                method()
                break
        self._llm = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def predict(
        self,
        sample: PromptSample,
        detector_results: dict[str, DetectorResult],
        anchors: list[dict[str, Any]],
        candidate_detectors: list[str],
        fingerprint_maps: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, PredictorEstimate]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        fingerprint_maps = fingerprint_maps or {}
        tokenizer = self._load_tokenizer()
        prompts = []
        detectors_for_prompt = []
        for detector in candidate_detectors:
            fp_map = fingerprint_maps.get(detector, {})
            messages = build_prediction_messages(
                sample,
                detector,
                anchors,
                fp_map,
                top_k_use=self.config.retrieval.top_k_use,
            )
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            detectors_for_prompt.append(detector)

        llm = self._load_llm()
        cfg = self.config.predictor
        outputs = llm.generate(
            prompts,
            SamplingParams(
                n=cfg.n_samples,
                seed=cfg.seed,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=0.0,
                presence_penalty=cfg.presence_penalty,
                max_tokens=cfg.max_new_tokens,
            ),
            lora_request=LoRARequest("scout", 1, self.adapter_ref),
        )

        estimates: dict[str, PredictorEstimate] = {}
        for detector, output in zip(detectors_for_prompt, outputs):
            parsed_all = [parse_structured_prediction(item.text.strip()) for item in output.outputs]
            valid = [item for item in parsed_all if not item["parse_failed"] and item["pred_correctness"] is not None]
            if valid:
                confidence = sum(float(item["pred_correctness"]) for item in valid) / len(valid)
                latencies = [float(item["pred_latency_ms"]) for item in valid if item["pred_latency_ms"] is not None]
                risks = [float(item["pred_risk"]) for item in valid if item["pred_risk"] is not None]
                estimates[detector] = PredictorEstimate(
                    detector=detector,
                    pred_corr=1.0 if confidence >= 0.5 else 0.0,
                    pred_risk=sum(risks) / len(risks) if risks else 0.0,
                    pred_lat_ms=sum(latencies) / len(latencies) if latencies else 0.0,
                )
            else:
                estimates[detector] = PredictorEstimate(
                    detector=detector,
                    pred_corr=0.0,
                    pred_risk=0.0,
                    pred_lat_ms=0.0,
                )
        return estimates


def create_predictor(config: ScoutConfig):
    if config.predictor.backend == "vllm":
        return ScoutPredictor(config)
    raise ValueError(f"unsupported predictor backend {config.predictor.backend!r}")
