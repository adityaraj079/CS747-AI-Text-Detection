from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "daniel_bert_original_80_10_10"
EXPECTED_ROWS = 165086
RANDOM_STATE = 42


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_row_indices(indices: pd.Index) -> str:
    payload = ",".join(str(int(index)) for index in sorted(indices)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_original_dataset(source_csv: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    source_csv = source_csv.expanduser().resolve(strict=True)
    raw = pd.read_csv(source_csv)
    if len(raw) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows from the original notebook dataset, got {len(raw)}."
        )
    if "source" not in raw.columns or raw.columns[-1] != "source":
        raise ValueError("Expected the original CSV's final column to be 'source'.")

    data = raw.iloc[:, :-1].copy()
    if list(data.columns) != ["text", "labels"]:
        raise ValueError(
            f"Expected columns ['text', 'labels'] after removing source; got {list(data.columns)}."
        )
    if data["text"].isna().any():
        raise ValueError("The original dataset contains missing essay text.")

    labels = pd.to_numeric(data["labels"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("The original dataset must contain both binary labels 0 and 1.")
    data["labels"] = labels.astype(int)

    train, remainder = train_test_split(
        data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["labels"],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=remainder["labels"],
    )
    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    if {name: len(frame) for name, frame in splits.items()} != {
        "train": 132068,
        "validation": 16509,
        "test": 16509,
    }:
        raise ValueError("Generated split sizes do not match those recorded in the original notebook.")

    dataset_hash = hashlib.sha256(data.to_csv(index=False).encode("utf-8")).hexdigest()
    metadata = {
        "model_id": "daniel_bert",
        "split_method": "stratified train_test_split; test_size=0.2, then test_size=0.5",
        "random_state": RANDOM_STATE,
        "source_csv": str(source_csv),
        "source_csv_sha256": _sha256_file(source_csv),
        "dataset_hash": dataset_hash,
        "dataset_hash_method": "SHA-256 of text,labels CSV serialization after dropping source, index excluded",
        "source_row_index_fingerprint_method": (
            "SHA-256 of comma-separated sorted zero-based original row indices"
        ),
        "split_counts": {name: int(len(frame)) for name, frame in splits.items()},
        "class_distributions": {
            name: {
                str(int(label)): int(count)
                for label, count in frame["labels"].value_counts().sort_index().items()
            }
            for name, frame in splits.items()
        },
        "split_fingerprints": {
            name: _fingerprint_row_indices(frame.index)
            for name, frame in splits.items()
        },
        "evaluation_note": (
            "This is Daniel BERT's original 80/10/10 split, not the repository's frozen split. "
            "It must not overwrite data/processed/train.csv, validation.csv, test.csv, "
            "unseen_test.csv, or the repository split manifests."
        ),
    }
    return splits, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate Daniel BERT's original stratified 80/10/10 dataset split."
    )
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = [output_dir / f"{name}.csv" for name in ("train", "validation", "test")]
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in [*output_files, manifest_path] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing Daniel split files: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite to replace them."
        )

    splits, metadata = split_original_dataset(args.source_csv)
    for name, frame in splits.items():
        frame[["text", "labels"]].to_csv(output_dir / f"{name}.csv", index=False)
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
