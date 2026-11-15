from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from cs747_submission3 import BPEWordTokenizer
from evaluation.model_adapters import ModelAdapter, require_probability_vector


class AdityaSubmission3Adapter(ModelAdapter):
    """Run the packaged low-memory Submission #3 adaptation ensemble."""

    def __init__(self, artifact_dir: str | Path | None = None):
        self.root_dir = MODEL_DIR
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else MODEL_DIR / "artifacts"
        self.model_ready = False
        self.unavailable_reason = ""
        self.tokenizer = None
        self.vectorizer = None
        self.sgd_model = None
        self.catboost_model = None
        self.sgd_weight = 0.5
        self.catboost_weight = 0.5
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        artifact_names = (
            "bpe_tokenizer.json",
            "tfidf_vectorizer.joblib",
            "sgd_classifier.joblib",
            "catboost_classifier.cbm",
            "configuration.json",
        )
        missing = [name for name in artifact_names if not (self.artifact_dir / name).is_file()]
        if missing:
            self.unavailable_reason = (
                "Missing fitted adaptation artifacts: " + ", ".join(missing)
            )
            return

        import joblib
        from catboost import CatBoostClassifier
        from tokenizers import Tokenizer

        with (self.artifact_dir / "configuration.json").open(encoding="utf-8") as file:
            configuration = json.load(file)
        ensemble = configuration.get("ensemble", {})
        sgd_weight = ensemble.get("sgd_weight")
        catboost_weight = ensemble.get("catboost_weight")
        if (
            not isinstance(sgd_weight, (int, float))
            or not isinstance(catboost_weight, (int, float))
            or not np.isfinite(sgd_weight)
            or not np.isfinite(catboost_weight)
            or not np.isclose(sgd_weight + catboost_weight, 1.0)
            or not 0 <= sgd_weight <= 1
            or not 0 <= catboost_weight <= 1
        ):
            raise ValueError("The saved ensemble weights must be finite probabilities summing to one.")

        self.tokenizer = Tokenizer.from_file(str(self.artifact_dir / "bpe_tokenizer.json"))
        self.vectorizer = joblib.load(self.artifact_dir / "tfidf_vectorizer.joblib")
        self.vectorizer.tokenizer = BPEWordTokenizer(self.tokenizer)
        self.sgd_model = joblib.load(self.artifact_dir / "sgd_classifier.joblib")
        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model(str(self.artifact_dir / "catboost_classifier.cbm"))
        self.sgd_weight = float(sgd_weight)
        self.catboost_weight = float(catboost_weight)
        self.model_ready = True

    @staticmethod
    def _positive_class_probability(model, features) -> np.ndarray:
        positive_indices = np.flatnonzero(np.asarray(model.classes_) == 1)
        if positive_indices.size != 1:
            raise ValueError("The fitted model must expose exactly one positive class labeled 1.")
        return np.asarray(model.predict_proba(features)[:, int(positive_indices[0])], dtype=float)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.asarray([], dtype=np.float64)
        if not self.model_ready:
            raise FileNotFoundError(self.unavailable_reason)

        features = self.vectorizer.transform(texts)
        sgd_probabilities = self._positive_class_probability(self.sgd_model, features)
        catboost_probabilities = self._positive_class_probability(self.catboost_model, features)
        probabilities = (
            self.sgd_weight * sgd_probabilities
            + self.catboost_weight * catboost_probabilities
        )
        return require_probability_vector(probabilities, len(texts))


if __name__ == "__main__":
    adapter = AdityaSubmission3Adapter()
    print("Model ready:", adapter.model_ready)
