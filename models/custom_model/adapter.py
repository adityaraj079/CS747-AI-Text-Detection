from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.model_adapters import ModelAdapter, ModelNotAvailableError, require_probability_vector

MODEL_DIR = Path(__file__).resolve().parent


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _load_member(name: str, artifact_dir: Path, config: dict, device: str):
    """Only the 4 trusted team members; artifacts come from this repo or the pinned Hugging Face revision."""
    if name == "aakash_7th_place":
        from models.aakash_7th_place.adapter import AakashSeventhPlaceAdapter
        return AakashSeventhPlaceAdapter(artifact_dir / "seventh")
    if name == "daniel_bert":
        from models.daniel_bert.adapter import DanielBertAdapter
        bert_dir = artifact_dir / "bert"
        if not (bert_dir / "model.safetensors").is_file():
            if not config.get("bert_huggingface_revision"):
                raise ModelNotAvailableError("Retrained BERT weights are not local and no pinned Hugging Face revision is set")
            from huggingface_hub import snapshot_download
            bert_dir = Path(snapshot_download(config["bert_huggingface_model_id"],
                                              revision=config["bert_huggingface_revision"]))
        return DanielBertAdapter(bert_dir, device=device, batch_size=32 if device != "cpu" else 8)
    if name in ("aditya_3rd_place", "yash_5th_place"):
        from app_components.model_registry import load_model_adapter
        return load_model_adapter(name)
    raise ValueError(f"Unknown ensemble member: {name}")


class CustomEnsembleAdapter(ModelAdapter):
    """Stacked ensemble of the 4 team models: P(AI) = sigmoid(w . logit(p_members) + b - threshold).

    Weights, bias and threshold come from artifacts/combiner.json (fit on validation with source-held-out selection),
    so 0.5 on the output corresponds to the validation-selected threshold.
    """

    def __init__(self, artifact_dir: str | Path | None = None, *, device: str | None = None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir else MODEL_DIR / "artifacts"
        self.model_ready = False
        self.unavailable_reason = ""
        combiner_path = self.artifact_dir / "combiner.json"
        if not combiner_path.is_file():
            self.unavailable_reason = "Missing fitted ensemble artifact: combiner.json"
            return
        self.combiner = json.loads(combiner_path.read_text(encoding="utf-8"))
        config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        weights = np.asarray(self.combiner["weights"], dtype=float)
        self.members, self.weights = [], []
        for name, weight in zip(self.combiner["members"], weights):
            if weight == 0:
                continue  # zero-weight members are not loaded
            member = _load_member(name, self.artifact_dir, config, device)
            if not getattr(member, "model_ready", True):
                self.unavailable_reason = f"Ensemble member {name} unavailable: {member.unavailable_reason}"
                return
            self.members.append(member)
            self.weights.append(weight)
        self.weights = np.asarray(self.weights)
        self.offset = float(self.combiner["bias"]) - float(self.combiner["threshold_logit"])
        self.model_ready = True

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.asarray([], dtype=np.float64)
        if not self.model_ready:
            raise ModelNotAvailableError(self.unavailable_reason)
        p = np.column_stack([m.predict_proba(texts) for m in self.members])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return require_probability_vector(_sigmoid(np.log(p / (1 - p)) @ self.weights + self.offset), len(texts))
