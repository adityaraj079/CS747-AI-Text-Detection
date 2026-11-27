#!/usr/bin/env python3
"""Score the 4 team models on the frozen splits, then pick and fit the ensemble combiner.

Run from the repo root:  python -m models.custom_model.build_ensemble [scores_dir]
1. Member scores for validation / test / unseen_test are cached as CSVs in scores_dir (re-runs skip scoring).
2. Candidates (each single model, equal-weight mean of logits, logistic regression on logits) are compared by
   source-held-out ROC-AUC on validation: GroupKFold over `source`, so each source is scored by a combiner that
   never saw it. The best candidate is refit on all of validation; its threshold maximises held-out balanced accuracy.
3. Test and unseen_test are scored once with the chosen combiner, for reporting only (no selection on them).
Writes models/custom_model/artifacts/combiner.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from models.yash_5th_place.train import load_splits

HERE = Path(__file__).resolve().parent
MEMBERS = ("aakash_7th_place", "aditya_3rd_place", "yash_5th_place", "daniel_bert")
SPLITS = ("validation", "test", "unseen_test")


def load_member(name: str):
    if name == "aakash_7th_place":
        from models.aakash_7th_place.adapter import AakashSeventhPlaceAdapter
        return AakashSeventhPlaceAdapter(HERE / "artifacts" / "seventh")  # retrained on frozen train
    if name == "daniel_bert":
        import torch
        from models.daniel_bert.adapter import DanielBertAdapter
        return DanielBertAdapter(HERE / "artifacts" / "bert", device="cuda" if torch.cuda.is_available() else "cpu",
                                 batch_size=64)  # retrained on frozen train
    from app_components.model_registry import load_model_adapter
    return load_model_adapter(name)  # aditya / yash: already trained on frozen train


def member_scores(splits: dict, scores_dir: Path) -> dict[str, pd.DataFrame]:
    scores_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in SPLITS:
        path = scores_dir / f"{split}.csv"
        frame = pd.read_csv(path) if path.is_file() else splits[split][["essay_id", "source", "labels"]].copy()
        for name in MEMBERS:
            if name not in frame:
                adapter = load_member(name)
                if not getattr(adapter, "model_ready", True):
                    raise RuntimeError(f"{name}: {adapter.unavailable_reason}")
                print(f"scoring {name} on {split} ({len(frame)})", flush=True)
                frame[name] = adapter.predict_proba(splits[split].text.fillna("").astype(str).tolist())
                frame.to_csv(path, index=False)  # checkpoint after each member
        out[split] = frame
    return out


def logits(frame: pd.DataFrame) -> np.ndarray:
    p = np.clip(frame[list(MEMBERS)].to_numpy(float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> int:
    scores_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "scores"
    splits, manifest = load_splits()
    scores = member_scores(splits, scores_dir)
    val = scores["validation"]
    X, y, groups = logits(val), val.labels.to_numpy(), val.source.to_numpy()

    def fit_lr(Xf, yf):
        return LogisticRegression(C=1.0, max_iter=1000).fit(Xf, yf)

    # Held-out scores per candidate: fixed rules need no fitting; LR is fit on the other folds' sources.
    oof = {name: X[:, i] for i, name in enumerate(MEMBERS)}
    oof["mean_logit"] = X.mean(axis=1)
    oof["logistic"] = np.empty(len(y))
    for fit_idx, held_idx in GroupKFold(n_splits=5).split(X, y, groups):
        oof["logistic"][held_idx] = fit_lr(X[fit_idx], y[fit_idx]).decision_function(X[held_idx])
    held_out_auc = {name: float(roc_auc_score(y, s)) for name, s in oof.items()}
    best = max(held_out_auc, key=held_out_auc.get)
    print("source-held-out validation AUC:", json.dumps(held_out_auc, indent=1), "\nchosen:", best, flush=True)

    if best == "logistic":
        lr = fit_lr(X, y)
        weights, bias = lr.coef_.ravel().tolist(), float(lr.intercept_[0])
    elif best == "mean_logit":
        weights, bias = [1 / len(MEMBERS)] * len(MEMBERS), 0.0
    else:
        weights, bias = [float(n == best) for n in MEMBERS], 0.0
    # Threshold on the combined logit, chosen on held-out scores (balanced accuracy: classes are ~86% AI).
    grid = np.quantile(oof[best], np.linspace(0.01, 0.99, 197))
    # ponytail: for "logistic" the threshold comes from fold models and is reused on the refit; scales are close.
    threshold_logit = float(grid[np.argmax([balanced_accuracy_score(y, oof[best] >= t) for t in grid])])

    report = {}
    for split in SPLITS:
        f = scores[split]
        z = logits(f) @ np.array(weights) + bias
        report[split] = {"auc": float(roc_auc_score(f.labels, z)),
                         "balanced_accuracy": float(balanced_accuracy_score(f.labels, z >= threshold_logit)),
                         **{f"auc_{n}": float(roc_auc_score(f.labels, f[n])) for n in MEMBERS}}
    combiner = {"members": list(MEMBERS), "feature": "logit(clip(p, 1e-6, 1-1e-6))", "rule": best,
                "weights": weights, "bias": bias, "threshold_logit": threshold_logit,
                "probability": "sigmoid(weights . logits + bias - threshold_logit) so 0.5 = chosen threshold",
                "selection": "GroupKFold(5) over validation `source`; test/unseen_test never used for selection",
                "source_held_out_validation_auc": held_out_auc, "report": report,
                "dataset_hash": manifest["dataset_hash"], "split_fingerprints": manifest["split_fingerprints"]}
    (HERE / "artifacts").mkdir(exist_ok=True)
    (HERE / "artifacts" / "combiner.json").write_text(json.dumps(combiner, indent=2), encoding="utf-8")
    print("RESULTS " + json.dumps({"chosen": best, "weights": weights, **{s: report[s]["auc"] for s in SPLITS}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
