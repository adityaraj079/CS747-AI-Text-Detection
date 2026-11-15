from __future__ import annotations

import numpy as np

from .metrics import compute_metrics
from .model_adapters import ModelAdapter, require_probability_vector


def evaluate_model(model: ModelAdapter, texts, labels, dataset_id: str, model_id: str, threshold: float = 0.5):
    texts = list(texts)
    labels = np.asarray(labels, dtype=int)
    if len(texts) != len(labels):
        raise ValueError('Texts and labels must be the same length.')
    probs = model.predict_proba(texts)
    probs = require_probability_vector(probs, len(texts))
    metrics = compute_metrics(labels, probs, threshold=threshold)
    return {
        'model_id': model_id,
        'dataset_id': dataset_id,
        'threshold': float(threshold),
        **metrics,
    }
