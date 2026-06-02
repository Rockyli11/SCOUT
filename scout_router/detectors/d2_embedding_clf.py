"""D2-LR embedding classifier runtime wrapper."""

from __future__ import annotations

import pickle

from scout_router.assets import external_model_cache_dir
from scout_router.config import ScoutConfig
from scout_router.detectors.base import DetectorBase, DetectorNotReady
from scout_router.detectors.manifest import require_manifest_file
from scout_router.detectors.registry import register_detector
from scout_router.schema import DetectorResult, PromptSample

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MAX_TEXT_CHARS = 2000


def _build_input(sample: PromptSample) -> str:
    parts = [sample.eval_content]
    if sample.goal_text:
        parts.append(f"[GOAL] {sample.goal_text}")
    if sample.policy_text:
        parts.append(f"[POLICY] {sample.policy_text}")
    return " ".join(parts)[:MAX_TEXT_CHARS]


class EmbeddingClfDetector(DetectorBase):
    cost_tier = "cheap"

    def __init__(self, config: ScoutConfig | None = None, clf_type: str = "lr"):
        self.config = config or ScoutConfig.from_env()
        self.clf_type = clf_type
        self.name = f"d2_{clf_type}"
        self._model = None
        self._embedder = None

    def _load(self) -> None:
        if self._model is not None:
            return
        model_path = require_manifest_file(self.config, self.name, "d2_embedding_clf", "model_path")
        with model_path.open("rb") as handle:
            self._model = pickle.load(handle)
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(
            EMBED_MODEL_NAME,
            cache_folder=str(external_model_cache_dir(self.config.cache_dir)),
        )

    def _detect(self, sample: PromptSample) -> DetectorResult:
        try:
            self._load()
        except ImportError as exc:
            raise DetectorNotReady("D2-LR requires sentence-transformers and scikit-learn.") from exc
        text = _build_input(sample)
        vec = self._embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)
        clf = self._model["clf"] if isinstance(self._model, dict) and "clf" in self._model else self._model
        proba = float(clf.predict_proba(vec)[0, 1])
        return DetectorResult(label=int(proba >= 0.5), confidence=proba, raw={"attack_probability": proba})


@register_detector("d2_lr")
class EmbeddingLRDetector(EmbeddingClfDetector):
    def __init__(self, config: ScoutConfig | None = None):
        super().__init__(config=config, clf_type="lr")


@register_detector("d2_svm")
class EmbeddingSVMDetector(EmbeddingClfDetector):
    def __init__(self, config: ScoutConfig | None = None):
        super().__init__(config=config, clf_type="svm")


@register_detector("d2_xgb")
class EmbeddingXGBDetector(EmbeddingClfDetector):
    def __init__(self, config: ScoutConfig | None = None):
        super().__init__(config=config, clf_type="xgb")


@register_detector("d2_rf")
class EmbeddingRFDetector(EmbeddingClfDetector):
    def __init__(self, config: ScoutConfig | None = None):
        super().__init__(config=config, clf_type="rf")


@register_detector("d2_mlp")
class EmbeddingMLPDetector(EmbeddingClfDetector):
    def __init__(self, config: ScoutConfig | None = None):
        super().__init__(config=config, clf_type="mlp")


@register_detector("d2_knn")
class EmbeddingKNNDetector(EmbeddingClfDetector):
    def __init__(self, config: ScoutConfig | None = None):
        super().__init__(config=config, clf_type="knn")
