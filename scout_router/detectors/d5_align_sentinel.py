"""Optional D5 AlignSentinel detector."""

from __future__ import annotations

import hashlib
import json
import os

from pathlib import Path

from scout_router.assets import external_model_cache_dir
from scout_router.config import ScoutConfig
from scout_router.detectors.base import DetectorBase, DetectorNotReady
from scout_router.detectors.manifest import require_manifest_dir
from scout_router.detectors.registry import register_detector
from scout_router.schema import DetectorResult, PromptSample

DEFAULT_BACKBONE = "Qwen/Qwen3-4B"
DEFAULT_THRESHOLD = 0.5


def _build_prompt_parts(sample: PromptSample) -> tuple[str, str]:
    prefix_parts = []
    if sample.policy_text:
        prefix_parts.append(f"[POLICY]\n{sample.policy_text}\n")
    if sample.goal_text:
        prefix_parts.append(f"[GOAL]\n{sample.goal_text}\n")
    return "".join(prefix_parts), f"[CONTENT]\n{sample.eval_content}\n"


def _cache_key(sample: PromptSample) -> str:
    digest = hashlib.sha1()
    digest.update(sample.eval_content.encode("utf-8", errors="ignore"))
    digest.update(b"\n<<GOAL>>\n")
    digest.update(sample.goal_text.encode("utf-8", errors="ignore"))
    digest.update(b"\n<<POLICY>>\n")
    digest.update(sample.policy_text.encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def _extract_interaction_vector(*, model, tokenizer, sample: PromptSample, device: str, max_length: int, cache):
    import torch

    key = _cache_key(sample)
    if key in cache:
        return cache[key].clone()

    prefix, content_part = _build_prompt_parts(sample)
    full_prompt = prefix + content_part
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    goal_end = len(prefix_ids)
    encoded = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    seq_len = encoded["input_ids"].shape[1]
    bos_offset = (
        1
        if tokenizer.bos_token_id is not None and encoded["input_ids"][0, 0].item() == tokenizer.bos_token_id
        else 0
    )
    goal_end = min(goal_end + bos_offset, seq_len)
    content_start = goal_end

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)
    attn = torch.stack([layer.squeeze(0) for layer in outputs.attentions], dim=0)

    if goal_end == 0 or content_start >= seq_len:
        layer_count, head_count = attn.shape[:2]
        vector = torch.zeros(layer_count * head_count, dtype=torch.float32)
    else:
        content_to_goal = attn[:, :, content_start:, :goal_end].float()
        vector = content_to_goal.mean(dim=(-2, -1)).reshape(-1).cpu()
    cache[key] = vector.clone()
    return vector


@register_detector("d5_align_sentinel")
class AlignSentinelDetector(DetectorBase):
    name = "d5_align_sentinel"
    cost_tier = "heavy"

    def __init__(
        self,
        config: ScoutConfig | None = None,
        backbone: str | None = None,
        device: str | None = None,
    ):
        self.config = config or ScoutConfig.from_env()
        self.backbone = backbone or os.environ.get("D5_BACKBONE")
        self.device = device
        self.meta: dict | None = None
        self._probe = None
        self._model = None
        self._tokenizer = None
        self._feature_cache = {}

    def _load_assets(self) -> tuple[dict, object]:
        if self.meta is not None and self._probe is not None:
            return self.meta, self._probe
        model_dir = require_manifest_dir(self.config, self.name, "d5_align_sentinel", "model_dir")
        meta_path = Path(model_dir) / "meta.json"
        probe_path = Path(model_dir) / "mlp_probe.pt"
        if not meta_path.exists() or not probe_path.exists():
            raise DetectorNotReady(f"D5 artifacts missing under {model_dir}")
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise DetectorNotReady("D5 requires torch.") from exc

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        input_dim = int(meta.get("feature_dim", meta.get("input_dim")))
        hidden_dim = int(meta.get("hidden", 128))
        num_classes = int(meta.get("num_classes", 3))

        class MLPProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_classes),
                )

            def forward(self, x):
                return self.net(x)

        probe = MLPProbe()
        probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
        probe.eval()
        self.meta = meta
        self._probe = probe
        return meta, probe

    def _load_backbone(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise DetectorNotReady("D5 requires torch and transformers.") from exc
        meta, _ = self._load_assets()
        backbone = self.backbone or str(meta.get("backbone", DEFAULT_BACKBONE))
        tokenizer_name = str(meta.get("tokenizer", backbone))
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = str(external_model_cache_dir(self.config.cache_dir))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            backbone,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()
        self.backbone = backbone
        self.device = device
        self._tokenizer = tokenizer
        self._model = model

    def _detect(self, sample: PromptSample) -> DetectorResult:
        import torch

        meta, probe = self._load_assets()
        self._load_backbone()
        assert self.device is not None
        assert self._model is not None
        assert self._tokenizer is not None

        max_length = int(meta.get("truncation", 2048))
        threshold = float(meta.get("threshold", DEFAULT_THRESHOLD))
        vector = _extract_interaction_vector(
            model=self._model,
            tokenizer=self._tokenizer,
            sample=sample,
            device=self.device,
            max_length=max_length,
            cache=self._feature_cache,
        )
        with torch.no_grad():
            logits = probe(vector.unsqueeze(0))
            proba = torch.softmax(logits, dim=-1)[0]
        attack_prob = float(proba[-1].item())
        return DetectorResult(
            label=int(attack_prob >= threshold),
            confidence=attack_prob,
            raw={
                "p_benign": float(proba[0]),
                "p_aligned": float(proba[1]) if proba.numel() > 2 else None,
                "p_misaligned": attack_prob,
                "threshold": threshold,
                "backbone": self.backbone,
            },
        )
