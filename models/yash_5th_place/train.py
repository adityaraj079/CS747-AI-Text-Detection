#!/usr/bin/env python3
"""Reproduce the lightweight Yash ensemble on the repository's frozen split."""
from __future__ import annotations

import hashlib
import json
import sys
import copy
from collections import Counter
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import ComplementNB
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
ARTIFACTS = MODEL_DIR / "artifacts"
SEED = 747
MAX_FEATURES = 100_000
THREADS = 8


def fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256("".join(frame["essay_id"].sort_values().tolist()).encode()).hexdigest()


def load_splits() -> tuple[dict[str, pd.DataFrame], dict]:
    manifest = json.loads((ROOT / "data/manifests/split_manifest.json").read_text(encoding="utf-8"))
    splits = {name: pd.read_csv(ROOT / "data/processed" / f"{name}.csv")
              for name in ("train", "validation", "test", "unseen_test")}
    for name, frame in splits.items():
        if len(frame) != manifest["split_counts"][name] or fingerprint(frame) != manifest["split_fingerprints"][name]:
            raise ValueError(f"{name} does not match the frozen manifest; refusing to train")
    return splits, manifest


def local_source_hash() -> str | None:
    source = ROOT.parent / "CS747data" / "CS747data.csv"
    if not source.is_file():
        return None
    from evaluation.dataset_utils import compute_dataset_hash

    return compute_dataset_hash(pd.read_csv(source))


def token_stream(tokenizer: Tokenizer, texts: list[str], batch_size: int = 128):
    for start in range(0, len(texts), batch_size):
        encodings = tokenizer.encode_batch(texts[start:start + batch_size])
        for item in encodings:
            yield " ".join(item.tokens)


def build_vocabulary(tokenizer: Tokenizer, texts: list[str]) -> dict[str, int]:
    options = dict(ngram_range=(3, 5), tokenizer=str.split, token_pattern=None,
                   lowercase=False, sublinear_tf=True, dtype=np.float32)
    analyzer = TfidfVectorizer(**options).build_analyzer()
    counts: Counter = Counter()
    for document in token_stream(tokenizer, texts):
        counts.update(analyzer(document))
        if len(counts) > MAX_FEATURES * 3:
            counts = Counter(dict(counts.most_common(MAX_FEATURES * 2)))
    vocab = {term: i for i, (term, _) in enumerate(counts.most_common(MAX_FEATURES))}
    if not vocab:
        raise ValueError("The frozen training partition produced an empty vocabulary")
    return vocab


def score_auc(model, matrix, labels) -> float:
    positive = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
    return float(roc_auc_score(labels, model.predict_proba(matrix)[:, positive]))


def main() -> int:
    splits, manifest = load_splits()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train = splits["train"]
    validation = splits["validation"]
    train_texts = train.text.fillna("").astype(str).tolist()
    y_train = train.labels.to_numpy(dtype=np.int8)
    y_val = validation.labels.to_numpy(dtype=np.int8)

    tokenizer_path = ARTIFACTS / "tokenizer.json"
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.train_from_iterator(
        iter(train_texts),
        trainers.BpeTrainer(vocab_size=30522, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]),
        length=len(train_texts),
    )
    tokenizer.save(str(tokenizer_path))

    vectorizer = TfidfVectorizer(vocabulary=build_vocabulary(tokenizer, train_texts),
        ngram_range=(3, 5), tokenizer=str.split, token_pattern=None,
        lowercase=False, sublinear_tf=True, dtype=np.float32)
    X_train = vectorizer.fit_transform(token_stream(tokenizer, train_texts))
    X_val = vectorizer.transform(token_stream(tokenizer, validation.text.fillna("").astype(str).tolist()))
    joblib.dump(vectorizer, ARTIFACTS / "tfidf_vectorizer.joblib", compress=1)

    members = []
    validation_aucs = []
    for alpha in (.02, .1, 1.0):
        model = ComplementNB(alpha=alpha).fit(X_train, y_train)
        members.append(model)
        validation_aucs.append(score_auc(model, X_val, y_val))

    class_counts = np.bincount(y_train, minlength=2)
    class_weight = {label: len(y_train) / (2 * count) for label, count in enumerate(class_counts)}
    for seed in (42, 71, 22):
        model = SGDClassifier(loss="modified_huber", class_weight=class_weight,
                              random_state=seed, shuffle=False, tol=None)
        best_auc = -1.0
        best_model = None
        stale_epochs = 0
        for epoch in range(3):
            order = np.random.default_rng(SEED + seed * 100 + epoch).permutation(len(y_train))
            model.partial_fit(X_train[order], y_train[order], classes=np.array([0, 1]))
            auc = score_auc(model, X_val, y_val)
            if auc > best_auc + 1e-4:
                best_auc, best_model, stale_epochs = auc, copy.deepcopy(model), 0
            else:
                stale_epochs += 1
                if stale_epochs >= 2:
                    break
        members.append(best_model)
        validation_aucs.append(best_auc)

    params = dict(objective="binary", metric="auc", learning_rate=.02,
                  feature_fraction=.85, lambda_l1=1.0, lambda_l2=3.0,
                  min_data_in_leaf=80, max_depth=11, max_bin=255,
                  device_type="cpu", num_threads=THREADS, verbosity=-1,
                  seed=SEED, deterministic=True, force_col_wise=True)
    booster = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=200,
        valid_sets=[lgb.Dataset(X_val, label=y_val, reference=None)],
        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)])
    booster.save_model(str(ARTIFACTS / "lightgbm.txt"))
    joblib.dump(members, ARTIFACTS / "classifiers.joblib", compress=1)

    configuration = {
        "model_id": "yash_5th_place", "member": "Yash Koli",
        "competition_track": "Efficiency Prize (reported fifth place; original notebook/results not present in this repository)",
        "source_reference": "https://www.kaggle.com/competitions/llm-detect-ai-generated-text/",
        "architecture": "Byte-level BPE + training-only 3-5 token n-gram TF-IDF + 3 ComplementNB + 3 SGD modified-Huber + LightGBM weighted ensemble",
        "adaptation_status": "Adapted reproduction; the original notebook and exact original settings were not present in the repository.",
        "training_seed": SEED, "dataset_hash": manifest["dataset_hash"],
        "local_dataframe_hash": local_source_hash(),
        "dataset_hash_note": "The local pandas CSV serialization hash may differ from the frozen manifest hash; all four split counts and fingerprints are checked before fit.",
        "train_split_fingerprint": manifest["split_fingerprints"]["train"],
        "validation_split_fingerprint": manifest["split_fingerprints"]["validation"],
        "split_counts": manifest["split_counts"],
        "tokenizer_fit_partition": "train", "vocabulary_fit_partition": "train",
        "tfidf_idf_fit_partition": "train", "validation_use": "selection and early stopping only",
        "test_use": "final evaluation only after fitting; no threshold/model tuning",
        "max_features": MAX_FEATURES, "sgd_max_epochs": 3,
        "complement_nb_alpha": [.02, .1, 1.0],
        "sgd_loss": "modified_huber", "sgd_random_states": [42, 71, 22],
        "sgd_early_stopping": "keep validation ROC-AUC best epoch; stop after two stale epochs; min improvement 0.0001",
        "vocabulary_selection": "streaming training-only 3-5 gram counts; trim above 300000 to top 200000; keep top 100000",
        "lightgbm_max_rounds": 200, "lightgbm_best_round": booster.best_iteration,
        "lightgbm_parameters": params,
        "ensemble_weights": [.05, .05, .05, .15, .15, .15, .40],
        "validation_roc_auc_by_member": validation_aucs,
        "dependencies": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "pandas": pd.__version__, "scikit_learn": __import__("sklearn").__version__,
                         "lightgbm": lgb.__version__, "tokenizers": __import__("tokenizers").__version__},
        "artifact_paths": ["artifacts/tokenizer.json", "artifacts/tfidf_vectorizer.joblib",
                           "artifacts/classifiers.joblib", "artifacts/lightgbm.txt"],
        "cpu_compatible": True, "license_and_redistribution": "Model owner states written rights cover all source datasets and model artifacts; documentation retained in README.",
    }
    (ARTIFACTS / "configuration.json").write_text(json.dumps(configuration, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"artifacts": str(ARTIFACTS), "vocabulary_size": len(vectorizer.vocabulary_),
                      "validation_auc": validation_aucs, "best_lightgbm_round": booster.best_iteration}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
