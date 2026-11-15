from __future__ import annotations

from evaluation.model_adapters import ModelAdapter, ModelNotAvailableError


class KaggleFirstPlaceAdapter(ModelAdapter):
    """Adapter scaffold; requires the verified solution implementation and artifacts."""

    def __init__(self) -> None:
        raise ModelNotAvailableError(
            "The main-leaderboard first-place model has not been integrated. "
            "Add its verified inference implementation and fitted artifacts before use."
        )

    def predict_proba(self, texts: list[str]):
        raise ModelNotAvailableError(
            "The Kaggle first-place model is unavailable until its inference pipeline is integrated "
            f"(received {len(texts)} essays)."
        )
