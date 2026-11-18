from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_components.model_registry import load_model_adapter
from evaluation.cpu_benchmark import benchmark_cpu_inference
from evaluation.metrics import compute_metrics
from evaluation.model_adapters import require_probability_vector

MODEL_ID = "daniel_bert"
SPLIT_DIR = REPO_ROOT / "data" / "processed" / "daniel_bert_original_80_10_10"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "metrics" / "historical"
OUTPUT_NAME = "daniel_bert_original_80_10_10_test.json"
INFERENCE_BATCH_SIZE = 8
PROGRESS_INTERVAL_SECONDS = 10


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate only Daniel BERT on its recreated original 80/10/10 test partition."
    )
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--benchmark-samples", type=int, default=32)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> Path:
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1].")
    if args.benchmark_samples < 1:
        raise ValueError("--benchmark-samples must be positive.")

    split_dir = args.split_dir.expanduser().resolve()
    test_path = split_dir / "test.csv"
    manifest_path = split_dir / "manifest.json"
    if not test_path.is_file():
        raise FileNotFoundError(f"Daniel BERT test split not found: {test_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Daniel BERT split manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(test_path)
    required = {"text", "labels"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Daniel BERT test CSV is missing required columns: {missing}")
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Daniel BERT test essays must contain non-empty text.")
    labels = pd.to_numeric(frame["labels"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("Daniel BERT test split must contain both binary labels 0 and 1.")

    expected_count = manifest.get("split_counts", {}).get("test")
    if expected_count is not None and int(expected_count) != len(frame):
        raise ValueError(
            f"Test count mismatch: manifest={expected_count}, file={len(frame)}."
        )
    split_id = manifest.get("split_fingerprints", {}).get("test")
    if not split_id:
        raise ValueError("Daniel BERT split manifest has no test split fingerprint.")

    labels = labels.astype(int).to_numpy()
    texts = frame["text"].astype(str).tolist()
    model = load_model_adapter(MODEL_ID)
    print(
        f"Evaluating Daniel BERT on {len(texts):,} essays; "
        f"CPU inference batches of {INFERENCE_BATCH_SIZE}.",
        flush=True,
    )
    probabilities: list[float] = []
    inference_start = perf_counter()
    last_progress = inference_start
    for start in range(0, len(texts), INFERENCE_BATCH_SIZE):
        batch_texts = texts[start:start + INFERENCE_BATCH_SIZE]
        batch_probabilities = require_probability_vector(
            model.predict_proba(batch_texts),
            len(batch_texts),
        )
        probabilities.extend(batch_probabilities.tolist())
        completed = len(probabilities)
        now = perf_counter()
        if (
            completed == len(texts)
            or completed == INFERENCE_BATCH_SIZE
            or now - last_progress >= PROGRESS_INTERVAL_SECONDS
        ):
            elapsed = now - inference_start
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (len(texts) - completed) / rate if rate > 0 else 0.0
            percent = 100.0 * completed / len(texts)
            print(
                f"Daniel BERT inference: {completed:,}/{len(texts):,} "
                f"({percent:.1f}%) | elapsed {_format_duration(elapsed)} "
                f"| ETA {_format_duration(eta)}",
                flush=True,
            )
            last_progress = now

    probabilities_array = require_probability_vector(probabilities, len(texts))
    metrics = compute_metrics(labels, probabilities_array, threshold=args.threshold)

    benchmark_size = min(args.benchmark_samples, len(texts))
    print(
        f"Running CPU benchmark: {benchmark_size} essays, "
        "3 warm-ups and 5 timed repeats.",
        flush=True,
    )
    benchmark = benchmark_cpu_inference(
        model,
        texts[:benchmark_size],
        batch_size=benchmark_size,
        warmup_steps=3,
        repeats=5,
    )
    print("CPU benchmark complete; saving result record.", flush=True)
    record: dict[str, Any] = {
        "result_category": "historical",
        "model_id": MODEL_ID,
        "dataset_hash": manifest.get("dataset_hash"),
        "split_id": split_id,
        "dataset_type": "standard_test",
        "evaluation_partition": "daniel_original_80_10_10_test",
        "evaluation_protocol": manifest.get("split_method"),
        "training_split_warning": (
            "The model was trained using this dataset's separate 80/10/10 training partition. "
            "This result is historical and not comparable with the repository's frozen-split results."
        ),
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
    record.update({key: value for key, value in metrics.items() if key not in {"model_id", "dataset_id"}})
    record.update(
        {
            "latency_ms_per_essay": benchmark["milliseconds_per_essay"],
            "throughput": benchmark["throughput_essays_per_second"],
            "cpu_model": benchmark["cpu_model"],
            "thread_count": benchmark["thread_count"],
            "batch_size": benchmark["batch_size"],
            "peak_ram_mb": benchmark["peak_ram_mb"],
            "model_size_mb": None,
        }
    )
    for key, value in list(record.items()):
        if isinstance(value, float) and not math.isfinite(value):
            record[key] = None

    output_path = args.results_dir.expanduser().resolve() / OUTPUT_NAME
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite the existing Daniel BERT result: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return output_path


def main() -> int:
    args = _arguments()
    try:
        output_path = evaluate(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f"Daniel BERT evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Saved Daniel BERT historical evaluation: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
