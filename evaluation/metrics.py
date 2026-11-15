from __future__ import annotations

import json
from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError('Prediction count does not match label count.')
    if y_prob.ndim != 1:
        raise ValueError('Probability vector must be 1D.')
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError('Probability values must be in [0, 1].')
    if np.any(~np.isfinite(y_prob)):
        raise ValueError('Probability values must be finite.')
    if np.unique(y_true).size <= 1:
        roc_auc = float('nan')
    else:
        roc_auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'roc_auc': float(roc_auc),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'threshold': float(threshold),
    }


def serialize_metrics(metrics: Dict[str, Any]):
    return json.loads(json.dumps(metrics, allow_nan=True))
