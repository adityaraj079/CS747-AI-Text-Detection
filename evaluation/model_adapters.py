from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import numpy as np


class ModelAdapter(ABC):
    """Base interface for all project-compatible model adapters."""

    @abstractmethod
    def predict_proba(self, texts: list[str]):
        """Return a 1D NumPy array of AI-generation probabilities."""
        raise NotImplementedError


class ModelRegistry:
    _models: Dict[str, Type[ModelAdapter]] = {}

    @classmethod
    def register(cls, model_id: str, adapter_cls: Type[ModelAdapter]):
        if not issubclass(adapter_cls, ModelAdapter):
            raise TypeError('Registered class must inherit from ModelAdapter.')
        cls._models[model_id] = adapter_cls

    @classmethod
    def get(cls, model_id: str):
        return cls._models.get(model_id)

    @classmethod
    def list_registered(cls):
        return sorted(cls._models.keys())


def require_probability_vector(values, n_samples: int):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError('Probability output must be a 1D vector.')
    if arr.shape[0] != n_samples:
        raise ValueError(f'Expected {n_samples} probabilities, got {arr.shape[0]}.')
    if np.any(~np.isfinite(arr)):
        raise ValueError('Probability output contains NaN or infinite values.')
    if np.any((arr < 0) | (arr > 1)):
        raise ValueError('Probability values must lie in [0, 1].')
    return arr
