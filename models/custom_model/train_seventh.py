#!/usr/bin/env python3
"""Retrain the 7th-place pipeline (BPE -> 3-5 gram TF-IDF -> MultinomialNB + SGD soft vote) on the frozen train split.

Run from the repo root:  python -m models.custom_model.train_seventh
Same model settings as the authors' model_ensemble.py / tokenizer_data.py, but tokenizer, vocabulary and IDF are fit
on frozen `train` only (the original fit them on target text). Validation is used only for the reported AUC and the
threshold. Artifacts use the aakash_7th_place format, so AakashSeventhPlaceAdapter(artifact_dir) loads them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

from models.aakash_7th_place.adapter import AakashSeventhPlaceAdapter
from models.yash_5th_place.train import load_splits  # shared frozen-split count + fingerprint check

OUT = Path(__file__).resolve().parent / "artifacts" / "seventh"
MAX_FEATURES = 500_000  # ponytail: top-N by train frequency; raise if memory allows and val AUC improves
WEIGHTS = [0.43, 0.57]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def identity(tokens):
    return tokens


def main() -> int:
    splits, manifest = load_splits()
    train, val = splits["train"], splits["validation"]
    tr_text = train.text.fillna("").astype(str).tolist()
    va_text = val.text.fillna("").astype(str).tolist()
    y_tr, y_va = train.labels.to_numpy(), val.labels.to_numpy()
    log(f"frozen split ok: train={len(train)} val={len(val)}")

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.Sequence([])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tok.train_from_iterator((tr_text[i:i + 1000] for i in range(0, len(tr_text), 1000)),
                            trainer=trainers.BpeTrainer(vocab_size=50275,
                                                        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]))
    log(f"tokenizer trained on train, vocab={tok.get_vocab_size()}")

    def tokens(texts):
        return [e.tokens for e in tok.encode_batch(texts, add_special_tokens=False)]

    tok_tr, tok_va = tokens(tr_text), tokens(va_text)
    log("tokenized")

    vec = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer="word",
                          tokenizer=identity, preprocessor=identity, token_pattern=None, strip_accents="unicode",
                          min_df=2, max_features=MAX_FEATURES)
    X_tr = vec.fit_transform(tok_tr)
    X_va = vec.transform(tok_va)
    del tok_tr
    log(f"TF-IDF on train: {X_tr.shape}, nnz={X_tr.nnz:,}")

    mnb = MultinomialNB(alpha=0.0225).fit(X_tr, y_tr)
    log("MultinomialNB fit")
    sgd = SGDClassifier(max_iter=27000, tol=1e-4, loss="modified_huber", random_state=6743).fit(X_tr, y_tr)
    log(f"SGD fit, epochs={sgd.n_iter_}")

    p_mnb, p_sgd = mnb.predict_proba(X_va)[:, 1], sgd.predict_proba(X_va)[:, 1]
    p_va = np.average([p_mnb, p_sgd], axis=0, weights=WEIGHTS)
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    thr = float(grid[np.argmax([accuracy_score(y_va, p_va >= t) for t in grid])])

    OUT.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT / "tokenizer.json"))
    joblib.dump({"vocab": vec.vocabulary_, "idf": vec.idf_,
                 "mnb_feature_log_prob": mnb.feature_log_prob_, "mnb_class_log_prior": mnb.class_log_prior_,
                 "sgd_coef": sgd.coef_, "sgd_intercept": sgd.intercept_, "weights": WEIGHTS,
                 "threshold_val": thr}, OUT / "model.joblib", compress=3)
    metrics = {"val_auc_ensemble": roc_auc_score(y_va, p_va), "val_auc_mnb": roc_auc_score(y_va, p_mnb),
               "val_auc_sgd": roc_auc_score(y_va, p_sgd), "threshold_from_val": thr,
               "val_accuracy_at_threshold": accuracy_score(y_va, p_va >= thr)}
    (OUT / "configuration.json").write_text(json.dumps({
        "model_id": "aakash_7th_place",  # format tag read by AakashSeventhPlaceAdapter
        "variant": "retrained on frozen train for models/custom_model",
        "dataset_hash": manifest["dataset_hash"], "split_fingerprints": manifest["split_fingerprints"],
        "tokenizer_fit_partition": "train", "vocabulary_fit_partition": "train", "idf_fit_partition": "train",
        "validation_use": "reported AUC and threshold only", "max_features": MAX_FEATURES, "min_df": 2,
        "mnb_alpha": 0.0225, "sgd": {"loss": "modified_huber", "tol": 1e-4, "random_state": 6743,
                                     "epochs": int(sgd.n_iter_)}, "weights": WEIGHTS, **metrics,
        "libs": {"python": sys.version.split()[0], "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }, indent=2), encoding="utf-8")

    # Parity: the shared adapter must reproduce sklearn's ensemble on these artifacts.
    adapter_p = AakashSeventhPlaceAdapter(OUT).predict_proba(va_text[:500])
    diff = float(np.max(np.abs(adapter_p - p_va[:500])))
    assert diff < 1e-6, f"adapter parity failed: max diff {diff}"
    log(f"adapter parity ok (max diff {diff:.1e})")
    log("RESULTS " + json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
