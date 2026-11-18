from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
from tokenizers import Tokenizer

from evaluation.model_adapters import ModelAdapter, ModelNotAvailableError, require_probability_vector

MODEL_DIR = Path(__file__).resolve().parent
WEIGHTS = np.asarray([.05, .05, .05, .15, .15, .15, .40], dtype=float)


def _validate_artifact_runtime(config: dict) -> None:
    expected = config.get("dependencies", {})
    package_fields = (("scikit-learn", "scikit_learn"), ("lightgbm", "lightgbm"))
    mismatches = []
    for package, field in package_fields:
        expected_version = expected.get(field)
        if not isinstance(expected_version, str):
            raise ModelNotAvailableError(
                f"Yash artifact configuration is missing the recorded {package} version."
            )
        try:
            installed_version = version(package)
        except PackageNotFoundError as exc:
            raise ModelNotAvailableError(
                f"Yash inference requires {package} {expected_version}, which is not installed."
            ) from exc
        if installed_version != expected_version:
            mismatches.append(
                f"{package} {expected_version} (installed: {installed_version})"
            )
    if mismatches:
        raise ModelNotAvailableError(
            "Yash artifacts require " + " and ".join(mismatches)
            + ". Install the versions recorded in artifacts/configuration.json to enable this model."
        )


class YashFifthPlaceAdapter(ModelAdapter):
    """CPU inference adapter for the packaged, adapted BPE/TF-IDF ensemble."""

    def __init__(self, artifact_dir: str | Path | None = None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else MODEL_DIR / "artifacts"
        self.model_ready = False
        self.unavailable_reason = ""
        self.tokenizer = self.vectorizer = self.classifiers = self.booster = None
        self.runtime_note = ""
        required = ("configuration.json", "tokenizer.json", "tfidf_vectorizer.joblib",
                    "classifiers.joblib", "lightgbm.txt")
        missing = [name for name in required if not (self.artifact_dir / name).is_file()]
        if missing:
            self.unavailable_reason = "Missing fitted model artifacts: " + ", ".join(missing)
            return
        config = json.loads((self.artifact_dir / "configuration.json").read_text(encoding="utf-8"))
        if config.get("model_id") != "yash_5th_place":
            raise ValueError("Artifact configuration model_id is not yash_5th_place")
        _validate_artifact_runtime(config)
        self.tokenizer = Tokenizer.from_file(str(self.artifact_dir / "tokenizer.json"))
        self.vectorizer = joblib.load(self.artifact_dir / "tfidf_vectorizer.joblib")
        self.classifiers = joblib.load(self.artifact_dir / "classifiers.joblib")
        if len(self.classifiers) != 6:
            raise ValueError("Expected six repository-trained sklearn ensemble members")
        self.booster = None
        self.runtime_note = (
            "LightGBM member disabled for live inference because its native text artifact "
            "is not safe to load with the installed runtime; using the six sklearn ensemble members."
        )
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
            encoded = [" ".join(item.tokens) for item in self.tokenizer.encode_batch(batch)]
            features = self.vectorizer.transform(encoded)
            probabilities = []
            for model in self.classifiers:
                positive = np.flatnonzero(np.asarray(model.classes_) == 1)
                if positive.size != 1:
                    raise ValueError("Every fitted classifier must expose positive class 1")
                probabilities.append(model.predict_proba(features)[:, int(positive[0])])
            if self.booster is not None:
                probabilities.append(self.booster.predict(features, num_threads=1))
                weights = WEIGHTS
            else:
                weights = WEIGHTS[:len(self.classifiers)]
                weights = weights / weights.sum()
            output.extend(np.average(np.asarray(probabilities), axis=0, weights=weights))
        return require_probability_vector(output, len(texts))
