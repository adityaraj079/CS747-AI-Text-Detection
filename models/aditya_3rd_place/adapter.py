from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from evaluation.model_adapters import ModelAdapter, require_probability_vector


class AdityaSubmission3Adapter(ModelAdapter):
    """Adapter for Aditya Raj's Submission #3 replication.

    The project archive contains the submission notebook and the BPE tokenizer
    artifact, but not the fitted TF-IDF vectorizer/classifier objects. When the
    full trained artifacts are present under this folder's `artifacts/` directory,
    this adapter loads and executes them; otherwise it raises a clear error
    instead of fabricating probabilities.
    """

    def __init__(self, artifact_dir: str | Path | None = None):
        self.root_dir = Path(__file__).resolve().parent
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else self.root_dir / 'artifacts'
        self.tokenizer = None
        self.vectorizer = None
        self.model = None
        self.model_ready = False
        self._load_optional_artifacts()

    def _load_optional_artifacts(self):
        tokenizer_path = self.artifact_dir / 'bpe_tokenizer.json'
        if tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            except Exception:
                self.tokenizer = None

        vectorizer_candidates = [
            self.artifact_dir / 'tfidf_vectorizer.joblib',
            self.artifact_dir / 'tfidf_vectorizer.pkl',
            self.artifact_dir / 'vectorizer.joblib',
            self.artifact_dir / 'vectorizer.pkl',
        ]
        classifier_candidates = [
            self.artifact_dir / 'submission3_model.joblib',
            self.artifact_dir / 'submission3_model.pkl',
            self.artifact_dir / 'sgd_model.joblib',
            self.artifact_dir / 'sgd_model.pkl',
            self.artifact_dir / 'catboost_model.joblib',
            self.artifact_dir / 'catboost_model.pkl',
        ]

        vectorizer_path = next((p for p in vectorizer_candidates if p.exists()), None)
        classifier_path = next((p for p in classifier_candidates if p.exists()), None)
        if vectorizer_path is None or classifier_path is None:
            self.model_ready = False
            return

        try:
            import joblib
            self.vectorizer = joblib.load(vectorizer_path)
            self.model = joblib.load(classifier_path)
            self.model_ready = True
        except Exception:
            self.model_ready = False

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer is None:
            return text.split()
        try:
            from cs747_submission3 import BPEWordTokenizer
            return BPEWordTokenizer(self.tokenizer)(text)
        except Exception:
            return text.split()

    def predict_proba(self, texts: list[str]):
        texts = list(texts)
        if not texts:
            return np.asarray([], dtype=np.float64)
        if not self.model_ready:
            raise FileNotFoundError(
                'No fitted Submission #3 vectorizer/classifier artifacts were found in '
                f'{self.artifact_dir}. Place the trained vectorizer and model files there '
                'before running inference.'
            )

        transformed = self.vectorizer.transform([
            ' '.join(self._tokenize(text)) for text in texts
        ])
        probs = self.model.predict_proba(transformed)
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]
        probs = np.asarray(probs, dtype=float)
        return require_probability_vector(probs, len(texts))


if __name__ == '__main__':
    adapter = AdityaSubmission3Adapter()
    print('Model ready:', adapter.model_ready)
