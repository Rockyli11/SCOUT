"""Optional D4 attention-tracker detector."""

from __future__ import annotations

import json
import os

from scout_router.assets import external_model_cache_dir
from scout_router.config import ScoutConfig
from scout_router.detectors.base import DetectorBase, DetectorNotReady
from scout_router.detectors.manifest import require_manifest_file
from scout_router.detectors.registry import register_detector
from scout_router.schema import DetectorResult, PromptSample

DEFAULT_BACKBONE = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_THRESHOLD = 0.5
_EPS = 1e-8


def _find_subarray(haystack: list[int], needle: list[int]) -> int:
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return -1


def _get_input_ranges(tokenizer, instruction: str, data: str) -> tuple[tuple[int, int], tuple[int, int]]:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": "Data: " + data},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer.encode(text)
    instr_ids = tokenizer.encode(instruction, add_special_tokens=False)
    data_ids = tokenizer.encode("Data: " + data, add_special_tokens=False)
    instr_start = _find_subarray(full_ids, instr_ids)
    data_start = _find_subarray(full_ids, data_ids)
    if instr_start == -1 or data_start == -1:
        mid = len(full_ids) // 2
        return (0, mid), (mid, len(full_ids))
    return (instr_start, instr_start + len(instr_ids)), (data_start, data_start + len(data_ids))


def _first_token_attention_maps(model, encoded):
    import torch

    with torch.no_grad():
        out = model(**encoded, output_attentions=True)
    return [
        torch.nan_to_num(attn.detach().cpu(), nan=0.0)[:, :, -1, :].unsqueeze(2)
        for attn in out.attentions
    ]


def _compute_heatmap(attn_tuple, instr_range: tuple[int, int], data_range: tuple[int, int]) -> np.ndarray:
    import numpy as np
    import torch

    n_layers = len(attn_tuple)
    n_heads = attn_tuple[0].shape[1]
    heatmap = np.zeros((n_layers, n_heads), dtype=np.float32)
    i0, i1 = instr_range
    d0, d1 = data_range
    for layer_idx in range(n_layers):
        attn_layer = attn_tuple[layer_idx][0].to(torch.float32).cpu().numpy()
        last_row = attn_layer[:, -1, :]
        attn_instr = last_row[:, i0:i1].sum(axis=1)
        attn_data = last_row[:, d0:d1].sum(axis=1)
        heatmap[layer_idx] = attn_instr / (attn_instr + attn_data + _EPS)
    return heatmap


def _focus_score(heatmap: np.ndarray, heads: list[tuple[int, int]]) -> float:
    import numpy as np

    vals = [
        float(heatmap[layer, head])
        for layer, head in heads
        if layer < heatmap.shape[0] and head < heatmap.shape[1]
    ]
    return float(np.mean(vals)) if vals else 0.5


@register_detector("d4_attention_tracker")
class AttentionTrackerDetector(DetectorBase):
    name = "d4_attention_tracker"
    cost_tier = "heavy"

    def __init__(
        self,
        config: ScoutConfig | None = None,
        backbone: str | None = None,
        threshold: float | None = None,
        device: str | None = None,
    ):
        self.config = config or ScoutConfig.from_env()
        self.backbone = backbone or os.environ.get("D4_BACKBONE", DEFAULT_BACKBONE)
        self.threshold = float(threshold or os.environ.get("D4_THRESHOLD", DEFAULT_THRESHOLD))
        self.device = device
        self.important_heads: list[tuple[int, int]] | None = None
        self._model = None
        self._tokenizer = None

    def _load_heads(self) -> None:
        if self.important_heads is not None:
            return
        path = require_manifest_file(self.config, self.name, "d4_attention_tracker", "heads_path")
        data = json.loads(path.read_text(encoding="utf-8"))
        heads = data.get("important_heads", data)
        self.important_heads = [tuple(map(int, head)) for head in heads]

    def _load_backbone(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise DetectorNotReady("D4 requires torch and transformers.") from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = str(external_model_cache_dir(self.config.cache_dir))
        tokenizer = AutoTokenizer.from_pretrained(self.backbone, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            self.backbone,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()
        self.device = device
        self._tokenizer = tokenizer
        self._model = model

    def _detect(self, sample: PromptSample) -> DetectorResult:
        import numpy as np

        self._load_heads()
        self._load_backbone()
        assert self.important_heads is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self.device is not None

        instruction = sample.goal_text or sample.policy_text or "Follow the instructions."
        data = sample.eval_content
        instr_range, data_range = _get_input_ranges(self._tokenizer, instruction, data)
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "Data: " + data},
        ]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        seq_len = encoded["input_ids"].shape[1]
        ir = (instr_range[0], min(instr_range[1], seq_len))
        dr = (data_range[0], min(data_range[1], seq_len))
        if ir[0] >= seq_len or dr[0] >= seq_len:
            return DetectorResult(
                label=0,
                confidence=0.1,
                raw={"note": "content_truncated", "focus_score": 1.0},
            )

        attn_maps = _first_token_attention_maps(self._model, encoded)
        heatmap = _compute_heatmap(attn_maps, ir, dr)
        score = _focus_score(heatmap, self.important_heads)
        label = int(score < self.threshold)
        confidence = (
            (1.0 - score / max(self.threshold, _EPS)) * 0.99
            if label
            else (score / max(1.0 - self.threshold, _EPS)) * 0.49
        )
        return DetectorResult(
            label=label,
            confidence=float(np.clip(confidence, 0.01, 0.99)),
            raw={
                "focus_score": score,
                "threshold": self.threshold,
                "n_important_heads": len(self.important_heads),
                "backbone": self.backbone,
            },
        )
