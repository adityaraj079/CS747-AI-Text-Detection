from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer

from evaluation.model_adapters import ModelAdapter, ModelNotAvailableError, require_probability_vector

MODEL_DIR = Path(__file__).resolve().parent
REQUIRED = ("configuration.json", "tokenizer.json", "model.joblib")


def _identity(tokens):
    return tokens


class AakashSeventhPlaceAdapter(ModelAdapter):
    """CPU inference for the 7th-place BPE -> TF-IDF -> MultinomialNB + SGD soft-vote ensemble.

    The fitted sklearn models are stored as plain arrays, so inference is version independent:
    NB = softmax(X @ feature_log_prob.T + class_log_prior), SGD (modified Huber) = (clip(X @ coef + b, -1, 1) + 1) / 2.
    """

    def __init__(self, artifact_dir: str | Path | None = None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else MODEL_DIR / "artifacts"
        self.model_ready = False
        self.unavailable_reason = ""
        missing = [name for name in REQUIRED if not (self.artifact_dir / name).is_file()]
        if missing:
            self.unavailable_reason = "Missing fitted model artifacts: " + ", ".join(missing)
            return
        config = json.loads((self.artifact_dir / "configuration.json").read_text(encoding="utf-8"))
        if config.get("model_id") != "aakash_7th_place":
            raise ValueError("Artifact configuration model_id is not aakash_7th_place")
        model = joblib.load(self.artifact_dir / "model.joblib")
        self.tokenizer = Tokenizer.from_file(str(self.artifact_dir / "tokenizer.json"))
        # Same settings as the original tokenizer_data.py; vocabulary and idf are the fitted ones.
        self.vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer="word",
                                          tokenizer=_identity, preprocessor=_identity, token_pattern=None,
                                          strip_accents="unicode", vocabulary=model["vocab"])
        self.vectorizer.idf_ = model["idf"]
        self.nb_log_prob = model["mnb_feature_log_prob"]
        self.nb_log_prior = model["mnb_class_log_prior"]
        self.sgd_coef = model["sgd_coef"]
        self.sgd_intercept = model["sgd_intercept"]
        self.weights = model["weights"]
        self.model_ready = True

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.asarray([], dtype=np.float64)
        if not self.model_ready:
            raise ModelNotAvailableError(self.unavailable_reason)
        output = []
        for start in range(0, len(texts), 256):
            batch = [str(text or "") for text in texts[start:start + 256]]
            tokens = [item.tokens for item in self.tokenizer.encode_batch(batch, add_special_tokens=False)]
            features = self.vectorizer.transform(tokens)
            p_nb = softmax(features @ self.nb_log_prob.T + self.nb_log_prior, axis=1)[:, 1]
            decision = features @ self.sgd_coef.ravel() + self.sgd_intercept[0]
            p_sgd = (np.clip(decision, -1, 1) + 1) / 2
            output.extend(np.average([p_nb, p_sgd], axis=0, weights=self.weights))
        return require_probability_vector(output, len(texts))
