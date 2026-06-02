"""D3 DeBERTa cross-encoder runtime wrapper."""

from __future__ import annotations

from scout_router.config import ScoutConfig
from scout_router.detectors.base import DetectorBase, DetectorNotReady
from scout_router.detectors.manifest import require_manifest_dir
from scout_router.detectors.registry import register_detector
from scout_router.schema import DetectorResult, PromptSample

SEP = " [SEP] "


@register_detector("d3_deberta")
class DebertaDetector(DetectorBase):
    name = "d3_deberta"
    cost_tier = "cheap"

    def __init__(self, config: ScoutConfig | None = None):
        self.config = config or ScoutConfig.from_env()
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        model_dir = require_manifest_dir(self.config, self.name, "d3_deberta", "model_dir")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise DetectorNotReady("D3 requires torch and transformers.") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.eval()

    def _detect(self, sample: PromptSample) -> DetectorResult:
        self._load()
        parts = [sample.eval_content]
        if sample.goal_text:
            parts.append(sample.goal_text)
        if sample.policy_text:
            parts.append(sample.policy_text)
        text = SEP.join(parts)
        encoded = self._tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with self._torch.no_grad():
            logits = self._model(**encoded).logits[0]
            probs = self._torch.softmax(logits, dim=-1)
        attack_probability = float(probs[1]) if probs.numel() > 1 else float(probs[0])
        return DetectorResult(
            label=int(attack_probability >= 0.5),
            confidence=attack_probability,
            raw={"attack_probability": attack_probability},
        )
