from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from scipy.special import softmax

from evaluation.model_adapters import ModelAdapter, ModelNotAvailableError, require_probability_vector

MODEL_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


class DanielBertAdapter(ModelAdapter):
    """Adapter for a repo-owned or Hugging Face-hosted BERT classifier.

    The model is intentionally strict: if the fitted artifacts are absent or invalid, it
    raises ModelNotAvailableError rather than fabricating probabilities.
    """

    device = "cpu"
    batch_size = 8

    def __init__(self, artifact_dir: str | Path | None = None, *, device="cpu", batch_size=8):
        self.device = device
        self.batch_size = batch_size
        use_local_only = artifact_dir is not None
        self.artifact_dir = Path(artifact_dir) if artifact_dir else MODEL_DIR / "artifacts"
        self.model_ready = False
        self.unavailable_reason = ""
        self.tokenizer = None
        self.model = None
        self.config = {}

        config_path = MODEL_DIR / "config.json"
        if not config_path.is_file():
            self.unavailable_reason = "Missing Daniel BERT package configuration: config.json"
            return

        try:
            self.config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive path
            self.unavailable_reason = f"Model configuration is unreadable: {exc}"
            return

        if self.config.get("model_id") != "daniel_bert":
            raise ValueError("Artifact configuration model_id is not daniel_bert")

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            self.unavailable_reason = f"Missing dependency for BERT inference: {exc}"
            return

        local_dir = str(self.artifact_dir) if any(
            (self.artifact_dir / name).is_file() for name in MODEL_WEIGHTS
        ) else None
        hf_repo = self.config.get("huggingface_model_id")
        revision = self.config.get("huggingface_revision")
        model_source = local_dir or (None if use_local_only else hf_repo)
        if not model_source:
            self.unavailable_reason = (
                "Missing fitted model checkpoint: no local weights were found in the requested artifact directory."
                if use_local_only
                else "Missing fitted model checkpoint: no local weights or Hugging Face model ID is configured."
            )
            return

        try:
            load_options = {"local_files_only": True} if local_dir else {"revision": revision}
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token and not local_dir:
                load_options["token"] = hf_token
            self.tokenizer = AutoTokenizer.from_pretrained(model_source, **load_options)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_source, **load_options)
            self.model.to(self.device)
        except Exception as exc:  # pragma: no cover - network-dependent path
            self.unavailable_reason = (
                f"Model or tokenizer could not be loaded from {model_source}: {exc}. "
                "Check network access, HF_TOKEN for a private repo, and the pinned model revision."
            )
            return

        self.model.eval()
        self.model_ready = True

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.asarray([], dtype=np.float64)
        if not self.model_ready:
            raise ModelNotAvailableError(self.unavailable_reason)

        batch = [str(text or "") for text in texts]
        try:
            import torch

            inference_context = torch.inference_mode
        except ImportError:  # Allows contract tests to use a lightweight fake model.
            inference_context = nullcontext

        probs = []
        with inference_context():
            for start in range(0, len(batch), self.batch_size):
                features = self.tokenizer(
                    batch[start:start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                if self.device != "cpu":
                    features = features.to(self.device)
                outputs = self.model(**features)
                logits = getattr(outputs, "logits", outputs)

                if hasattr(logits, "detach"):
                    logits = logits.detach().cpu().numpy()
                elif hasattr(logits, "numpy"):
                    logits = logits.numpy()

                logits_array = np.asarray(logits, dtype=float)
                if logits_array.ndim == 1:
                    logits_array = logits_array.reshape(-1, 1)
                if logits_array.shape[-1] == 2:
                    probs.extend(softmax(logits_array, axis=-1)[:, 1])
                else:
                    probs.extend(1.0 / (1.0 + np.exp(-logits_array.ravel())))

        return require_probability_vector(probs, len(texts))
