from __future__ import annotations

from pathlib import Path

from evaluation.model_adapters import ModelAdapter, ModelNotAvailableError, ModelRegistry


MODEL_IDS = (
    "kaggle_first_place",
    "custom_model",
    "yash_5th_place",
    "aakash_7th_place",
    "aditya_3rd_place",
    "daniel_bert",
)


def model_availability(model_id: str, repo_root: str | Path) -> tuple[bool, str]:
    if model_id == "kaggle_first_place":
        return False, "Not integrated: the original implementation and fitted artifacts are not available."
    if model_id == "aditya_3rd_place":
        artifacts = Path(repo_root) / "models" / model_id / "artifacts"
        required = (
            "bpe_tokenizer.json",
            "tfidf_vectorizer.joblib",
            "sgd_classifier.joblib",
            "catboost_classifier.cbm",
            "configuration.json",
        )
        missing = [name for name in required if not (artifacts / name).is_file()]
        if not missing:
            return True, "Low-memory adaptation artifacts are present; original submission settings are unverified."
        return False, "Adaptation artifacts are incomplete: " + ", ".join(missing)
    if model_id == "custom_model":
        return False, "Under development: no trained model package is registered."
    if model_id == "yash_5th_place":
        artifacts = Path(repo_root) / "models" / model_id / "artifacts"
        required = ("configuration.json", "tokenizer.json", "tfidf_vectorizer.joblib",
                    "classifiers.joblib", "lightgbm.txt")
        missing = [name for name in required if not (artifacts / name).is_file()]
        if not missing:
            return True, "Adapted CPU inference package is available; original competition settings are unverified."
        return False, "Fitted model artifacts are incomplete: " + ", ".join(missing)
    if model_id not in MODEL_IDS:
        return False, "Unknown model ID."
    return False, "Not integrated: this model's adapter and fitted artifacts are not present."


def load_model_adapter(model_id: str) -> ModelAdapter:
    """Load only explicitly registered, repository-owned model adapters."""
    if model_id == "kaggle_first_place":
        from models.kaggle_first_place.adapter import KaggleFirstPlaceAdapter

        ModelRegistry.register(model_id, KaggleFirstPlaceAdapter)
        return ModelRegistry.get(model_id)()
    if model_id == "aditya_3rd_place":
        from models.aditya_3rd_place.adapter import AdityaSubmission3Adapter

        ModelRegistry.register(model_id, AdityaSubmission3Adapter)
        adapter = ModelRegistry.get(model_id)()
        if not getattr(adapter, "model_ready", False):
            raise ModelNotAvailableError(adapter.unavailable_reason)
        return adapter
    if model_id == "yash_5th_place":
        from models.yash_5th_place.adapter import YashFifthPlaceAdapter

        ModelRegistry.register(model_id, YashFifthPlaceAdapter)
        adapter = ModelRegistry.get(model_id)()
        if not getattr(adapter, "model_ready", False):
            raise ModelNotAvailableError(adapter.unavailable_reason)
        return adapter
    if model_id in MODEL_IDS:
        raise ModelNotAvailableError(
            f"No trusted adapter is registered for '{model_id}'. Add the model package and explicitly register its adapter."
        )
    raise ModelNotAvailableError(f"Model ID '{model_id}' is not in the dashboard allowlist.")
