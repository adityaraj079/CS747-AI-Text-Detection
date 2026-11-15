#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_components.model_registry import MODEL_IDS, load_model_adapter
from evaluation.cpu_benchmark import benchmark_cpu_inference
from evaluation.evaluator import evaluate_model
from evaluation.model_adapters import ModelNotAvailableError


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one explicitly registered model on a frozen CS747 test partition."
    )
    parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
    parser.add_argument(
        "--dataset-type",
        choices=("standard_test", "unseen_test"),
        default="standard_test",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "metrics" / "standardized")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--benchmark-samples", type=int, default=256)
    return parser.parse_args()


def _load_partition(data_dir: Path, dataset_type: str) -> tuple[pd.DataFrame, dict[str, Any], str]:
    split_name = "test" if dataset_type == "standard_test" else dataset_type
    split_path = data_dir / "processed" / f"{split_name}.csv"
    manifest_path = data_dir / "manifests" / "split_manifest.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Frozen evaluation partition not found: {split_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(split_path)
    required = {"essay_id", "text", "labels"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation CSV is missing required columns: {missing}")
    if frame["essay_id"].isna().any() or frame["essay_id"].duplicated().any():
        raise ValueError("Evaluation essay IDs must be present and unique.")
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Evaluation essays must contain non-empty text.")
    labels = pd.to_numeric(frame["labels"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("The evaluation partition must contain both binary labels 0 and 1.")
    split_id = manifest.get("split_fingerprints", {}).get(split_name)
    expected_count = manifest.get("split_counts", {}).get(split_name)
    if not split_id:
        raise ValueError(f"Manifest does not contain a fingerprint for {split_name}.")
    if expected_count is not None and int(expected_count) != len(frame):
        raise ValueError(
            f"Record count mismatch for {split_name}: manifest={expected_count}, file={len(frame)}."
        )
    frame = frame.copy()
    frame["labels"] = labels.astype(int)
    return frame, manifest, str(split_id)


def _artifact_size_mb(model_id: str) -> float | None:
    artifact_dir = REPO_ROOT / "models" / model_id / "artifacts"
    if not artifact_dir.is_dir():
        return None
    files = [path for path in artifact_dir.rglob("*") if path.is_file() and path.name != "README.md"]
    if not files:
        return None
    return sum(path.stat().st_size for path in files) / (1024 * 1024)


def evaluate(args: argparse.Namespace) -> Path:
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1].")
    if args.benchmark_samples < 1:
        raise ValueError("--benchmark-samples must be positive.")

    frame, manifest, split_id = _load_partition(args.data_dir, args.dataset_type)
    model = load_model_adapter(args.model_id)
    evaluation = evaluate_model(
        model,
        frame["text"].tolist(),
        frame["labels"].to_numpy(),
        dataset_id=args.dataset_type,
        model_id=args.model_id,
        threshold=args.threshold,
    )

    benchmark_size = min(args.benchmark_samples, len(frame))
    benchmark_texts = frame["text"].iloc[:benchmark_size].tolist()
    benchmark = benchmark_cpu_inference(
        model,
        benchmark_texts,
        batch_size=benchmark_size,
        warmup_steps=3,
        repeats=5,
    )

    record: dict[str, Any] = {
        "result_category": "standardized",
        "model_id": args.model_id,
        "dataset_hash": manifest.get("dataset_hash"),
        "split_id": split_id,
        "dataset_type": args.dataset_type,
        "record_count": int(len(frame)),
        "threshold": float(args.threshold),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "benchmark": benchmark,
    }
    record.update({key: value for key, value in evaluation.items() if key not in {"model_id", "dataset_id"}})
    record.update(
        {
            "latency_ms_per_essay": benchmark["milliseconds_per_essay"],
            "throughput": benchmark["throughput_essays_per_second"],
            "cpu_model": benchmark["cpu_model"],
            "thread_count": benchmark["thread_count"],
            "batch_size": benchmark["batch_size"],
            "model_size_mb": _artifact_size_mb(args.model_id),
        }
    )
    for key, value in list(record.items()):
        if isinstance(value, float) and not math.isfinite(value):
            record[key] = None

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.results_dir / f"{args.model_id}_{args.dataset_type}_{stamp}.json"
    output_path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return output_path


def main() -> int:
    args = _arguments()
    try:
        output_path = evaluate(args)
    except (FileNotFoundError, ImportError, ModelNotAvailableError, ValueError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Saved standardized result: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
