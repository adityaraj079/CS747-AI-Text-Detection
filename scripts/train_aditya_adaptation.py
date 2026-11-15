#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "aditya_3rd_place"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
MANIFEST_PATH = ROOT / "data" / "manifests" / "split_manifest.json"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from cs747_submission3 import BPEWordTokenizer

TRAINING_CONFIG = {
    "configuration_mode": "low_memory",
    "original_submission_settings_verified": False,
    "seed": 747,
    "tokenizer": {
        "vocab_size": 16000,
        "unk_token": "[UNK]",
        "normalizer": "NFC",
        "pre_tokenizer": "ByteLevel",
        "add_prefix_space": False,
        "special_tokens": ["[UNK]"],
    },
    "tfidf": {
        "ngram_range": [3, 4],
        "min_df": 3,
        "max_features": 25000,
        "lowercase": False,
        "strip_accents": "unicode",
        "sublinear_tf": True,
        "dtype": "float32",
        "analyzer": "word",
    },
    "sgd": {
        "loss": "modified_huber",
        "penalty": "l2",
        "alpha": 1e-4,
        "max_iter": 5000,
        "tol": 1e-4,
        "random_state": 747,
    },
    "catboost": {
        "iterations": 100,
        "depth": 4,
        "border_count": 32,
        "learning_rate": 0.1,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 747,
        "thread_count": 2,
        "verbose": False,
        "allow_writing_files": False,
    },
    "ensemble_weight_step": 0.05,
    "decision_threshold": 0.5,
}


def load_partition(name: str, manifest: dict) -> tuple[list[str], np.ndarray]:
    path = ROOT / "data" / "processed" / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required processed partition is missing: {path}")
    frame = pd.read_csv(path, usecols=["text", "labels"])
    if frame["text"].isna().any() or frame["labels"].isna().any():
        raise ValueError(f"{path} contains missing text or labels.")
    labels = pd.to_numeric(frame["labels"], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{path} contains labels outside the expected 0/1 classes.")
    expected_count = manifest["split_counts"][name]
    if len(frame) != expected_count:
        raise ValueError(f"{path} has {len(frame)} rows; manifest expects {expected_count}.")
    expected_classes = manifest["class_distributions"][name]
    actual_classes = {str(label): int(np.count_nonzero(labels == label)) for label in (0, 1)}
    if actual_classes != expected_classes:
        raise ValueError(f"{path} class counts do not match split_manifest.json.")
    return frame["text"].astype(str).tolist(), labels


def positive_probability(estimator, features) -> np.ndarray:
    positive_indices = np.flatnonzero(np.asarray(estimator.classes_) == 1)
    if positive_indices.size != 1:
        raise ValueError("A trained classifier must expose exactly one positive class labeled 1.")
    return np.asarray(estimator.predict_proba(features)[:, int(positive_indices[0])], dtype=float)


def fit_adaptation(force: bool = False) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_names = (
        "bpe_tokenizer.json",
        "tfidf_vectorizer.joblib",
        "sgd_classifier.joblib",
        "catboost_classifier.cbm",
        "configuration.json",
    )
    existing = [name for name in output_names if (ARTIFACT_DIR / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "Adaptation artifacts already exist: "
            + ", ".join(existing)
            + ". Use --force to replace them."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    train_texts, y_train = load_partition("train", manifest)
    validation_texts, y_validation = load_partition("validation", manifest)
    print(f"Fitting only on train: {len(train_texts):,}; validation for blend selection: {len(validation_texts):,}")

    tokenizer_config = TRAINING_CONFIG["tokenizer"]
    tokenizer = Tokenizer(models.BPE(unk_token=tokenizer_config["unk_token"]))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=tokenizer_config["add_prefix_space"]
    )
    tokenizer_trainer = trainers.BpeTrainer(
        vocab_size=tokenizer_config["vocab_size"],
        special_tokens=tokenizer_config["special_tokens"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(train_texts, trainer=tokenizer_trainer)
    print("BPE tokenizer fit complete.")

    tfidf_config = TRAINING_CONFIG["tfidf"]
    vectorizer = TfidfVectorizer(
        tokenizer=BPEWordTokenizer(tokenizer),
        token_pattern=None,
        analyzer=tfidf_config["analyzer"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        lowercase=tfidf_config["lowercase"],
        strip_accents=tfidf_config["strip_accents"],
        min_df=tfidf_config["min_df"],
        max_features=tfidf_config["max_features"],
        sublinear_tf=tfidf_config["sublinear_tf"],
        dtype=np.float32,
    )
    train_features = vectorizer.fit_transform(train_texts)
    del train_texts
    gc.collect()
    validation_features = vectorizer.transform(validation_texts)
    del validation_texts
    gc.collect()
    print(f"TF-IDF fit complete: {train_features.shape[0]:,} rows, {train_features.shape[1]:,} features.")

    sgd_model = SGDClassifier(**TRAINING_CONFIG["sgd"])
    sgd_model.fit(train_features, y_train)
    print("SGDClassifier fit complete.")

    catboost_model = CatBoostClassifier(**TRAINING_CONFIG["catboost"])
    catboost_model.fit(train_features, y_train)
    del train_features
    gc.collect()
    print("CatBoost fit complete.")

    sgd_validation = positive_probability(sgd_model, validation_features)
    catboost_validation = positive_probability(catboost_model, validation_features)
    weight_step = TRAINING_CONFIG["ensemble_weight_step"]
    candidate_weights = [index * weight_step for index in range(round(1 / weight_step) + 1)]
    scored_weights = [
        (
            roc_auc_score(
                y_validation,
                sgd_weight * sgd_validation + (1.0 - sgd_weight) * catboost_validation,
            ),
            sgd_weight,
        )
        for sgd_weight in candidate_weights
    ]
    validation_auc, sgd_weight = max(
        scored_weights,
        key=lambda item: (item[0], -abs(item[1] - 0.5)),
    )
    catboost_weight = 1.0 - sgd_weight
    print(
        f"Validation blend selected: SGD={sgd_weight:.2f}, "
        f"CatBoost={catboost_weight:.2f}, ROC-AUC={validation_auc:.4f}"
    )

    configuration = {
        "model_id": "aditya_3rd_place",
        "name": "Submission #3 low-memory adaptation",
        "label_orientation": "1 = AI-generated",
        "training_scope": "frozen standard train partition only",
        "blend_selection_scope": "frozen validation partition only",
        "held_out_partitions_used_for_fitting": [],
        "dataset_hash": manifest["dataset_hash"],
        "train_split_fingerprint": manifest["split_fingerprints"]["train"],
        "validation_split_fingerprint": manifest["split_fingerprints"]["validation"],
        "train_count": manifest["split_counts"]["train"],
        "validation_count": manifest["split_counts"]["validation"],
        **TRAINING_CONFIG,
        "ensemble": {
            "method": "validation ROC-AUC grid search in 0.05 increments",
            "sgd_weight": sgd_weight,
            "catboost_weight": catboost_weight,
            "validation_roc_auc": float(validation_auc),
            "source_verified": False,
        },
        "software": {
            "python": sys.version.split()[0],
            "scikit_learn": __import__("sklearn").__version__,
            "catboost": __import__("catboost").__version__,
            "tokenizers": __import__("tokenizers").__version__,
        },
    }

    with tempfile.TemporaryDirectory(prefix=".aditya-fit-", dir=ARTIFACT_DIR) as temporary:
        temporary_dir = Path(temporary)
        tokenizer.save(str(temporary_dir / "bpe_tokenizer.json"))
        joblib.dump(vectorizer, temporary_dir / "tfidf_vectorizer.joblib", compress=3)
        joblib.dump(sgd_model, temporary_dir / "sgd_classifier.joblib", compress=3)
        catboost_model.save_model(str(temporary_dir / "catboost_classifier.cbm"))
        for name in output_names[:-1]:
            configuration.setdefault("artifact_sizes_bytes", {})[name] = (
                temporary_dir / name
            ).stat().st_size
        (temporary_dir / "configuration.json").write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )
        for name in output_names:
            os.replace(temporary_dir / name, ARTIFACT_DIR / name)

    return configuration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Aditya's documented low-memory adaptation on the frozen project train split."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing adaptation artifacts after a successful fit.",
    )
    args = parser.parse_args()
    configuration = fit_adaptation(force=args.force)
    print(
        "Saved the low-memory adaptation artifacts to "
        f"{ARTIFACT_DIR} (dataset {configuration['dataset_hash']})."
    )


if __name__ == "__main__":
    main()
