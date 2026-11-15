from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "latency_ms_per_essay",
    "throughput",
    "model_size_mb",
    "peak_ram_mb",
)

_ALIASES = {
    "latency_ms_per_essay": (
        "latency_ms_per_essay",
        "milliseconds_per_essay",
        "ms_per_essay",
    ),
    "throughput": ("throughput", "throughput_essays_per_second", "essays_per_second"),
    "peak_ram_mb": ("peak_ram_mb", "peak_inference_ram_mb", "peak_rss_mb"),
    "record_count": ("record_count", "n_samples", "test_record_count", "sample_count"),
    "training_time_seconds": ("training_time_seconds", "training_seconds"),
}


def normalize_dataset_type(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"test", "standard", "standard_test", "held_out_test"}:
        return "standard_test"
    if normalized in {"unseen", "unseen_test", "unseen_source_test", "ood", "source_disjoint"}:
        return "unseen_test"
    return normalized or None


def _finite_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_confusion_matrix(value: Any) -> list[list[int]] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in value):
        return None
    try:
        return [[int(item) for item in row] for row in value]
    except (TypeError, ValueError):
        return None


def normalize_result_record(
    payload: dict[str, Any],
    *,
    source_file: str = "",
    inferred_category: str = "standardized",
) -> dict[str, Any]:
    nested_metrics = payload.get("metrics")
    record = dict(payload)
    if isinstance(nested_metrics, dict):
        record.update(nested_metrics)

    for target, aliases in _ALIASES.items():
        if record.get(target) is None:
            record[target] = next((record.get(alias) for alias in aliases if record.get(alias) not in (None, "")), None)
        if target in {"latency_ms_per_essay", "throughput", "peak_ram_mb", "training_time_seconds"}:
            record[target] = _finite_number(record.get(target))
            if record[target] is not None and record[target] < 0:
                record[target] = None

    for metric in ("roc_auc", "accuracy", "precision", "recall", "f1", "model_size_mb"):
        if metric in record:
            record[metric] = _finite_number(record[metric])
            if metric in {"roc_auc", "accuracy", "precision", "recall", "f1"} and record[metric] is not None:
                if not 0.0 <= record[metric] <= 1.0:
                    record[metric] = None
            elif metric == "model_size_mb" and record[metric] is not None and record[metric] < 0:
                record[metric] = None

    record["model_id"] = str(record.get("model_id", "")).strip()
    record["dataset_hash"] = record.get("dataset_hash") or None
    record["split_id"] = record.get("split_id") or None
    record["dataset_type"] = normalize_dataset_type(record.get("dataset_type"))
    explicit_category = str(record.get("result_category", record.get("category", ""))).strip().lower()
    record["result_category"] = (
        "historical"
        if explicit_category == "historical" or inferred_category == "historical"
        else "standardized"
    )
    record["timestamp"] = record.get("timestamp") or record.get("evaluated_at") or record.get("created_at")
    record["confusion_matrix"] = _parse_confusion_matrix(record.get("confusion_matrix"))
    record["source_file"] = source_file

    if "thread_count" in record:
        try:
            record["thread_count"] = int(record["thread_count"])
        except (TypeError, ValueError):
            record["thread_count"] = None
    if "record_count" in record:
        try:
            record["record_count"] = int(record["record_count"])
        except (TypeError, ValueError):
            record["record_count"] = None

    return record


def _is_result_record(record: dict[str, Any]) -> bool:
    return bool(record.get("model_id")) and any(
        record.get(key) is not None for key in METRIC_KEYS
    )


def _records_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        raise ValueError("Result JSON must contain an object or a list of objects.")
    nested_records = payload.get("results")
    if isinstance(nested_records, list):
        return [row for row in nested_records if isinstance(row, dict)]
    return [payload]


def load_result_records(metrics_dir: str | Path) -> list[dict[str, Any]]:
    """Load only small metric result files; model artifacts and essay datasets are never read."""
    root = Path(metrics_dir)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv"}:
            continue
        category = "historical" if "historical" in {part.lower() for part in path.parts} else "standardized"
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_records = _records_from_json(payload)
        else:
            with path.open(newline="", encoding="utf-8") as handle:
                raw_records = list(csv.DictReader(handle))
        for raw_record in raw_records:
            record = normalize_result_record(
                raw_record,
                source_file=str(path.relative_to(root)),
                inferred_category=category,
            )
            if _is_result_record(record):
                records.append(record)
    return records


def load_json_if_present(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {file_path}.")
    return payload


def load_dataset_manifests(data_dir: str | Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    root = Path(data_dir) / "manifests"
    return (
        load_json_if_present(root / "split_manifest.json"),
        load_json_if_present(root / "unseen_test_manifest.json"),
    )


def partition_metadata(
    split_manifest: dict[str, Any] | None,
    dataset_type: str,
) -> dict[str, Any] | None:
    if not split_manifest:
        return None
    normalized = normalize_dataset_type(dataset_type)
    split_key = "unseen_test" if normalized == "unseen_test" else "test"
    count_key = split_key
    return {
        "dataset_hash": split_manifest.get("dataset_hash"),
        "split_id": split_manifest.get("split_fingerprints", {}).get(split_key),
        "record_count": split_manifest.get("split_counts", {}).get(count_key),
        "random_seed": split_manifest.get("random_seed"),
        "creation_time_utc": split_manifest.get("creation_time_utc"),
        "held_out_sources": split_manifest.get("held_out_sources", []) if split_key == "unseen_test" else [],
    }


def latest_result(
    records: list[dict[str, Any]],
    model_id: str,
    dataset_type: str | None = None,
    result_category: str | None = None,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in records
        if row.get("model_id") == model_id
        and (dataset_type is None or row.get("dataset_type") == normalize_dataset_type(dataset_type))
        and (result_category is None or row.get("result_category") == result_category)
    ]
    return max(candidates, key=lambda row: (str(row.get("timestamp") or ""), row.get("source_file", "")), default=None)


def compare_results_compatibility(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    require_cpu_compatibility: bool = False,
) -> tuple[bool, str]:
    if not left or not right:
        return False, "Both model results are required."
    for key in ("dataset_hash", "split_id", "dataset_type"):
        if not left.get(key) or not right.get(key):
            return False, f"Missing {key}; results cannot be verified as comparable."
        if left[key] != right[key]:
            return False, f"Results use different {key} values."
    if left.get("result_category") != right.get("result_category"):
        return False, "Results are from different result categories."
    if require_cpu_compatibility:
        for key in ("cpu_model", "thread_count"):
            if left.get(key) in (None, "") or right.get(key) in (None, ""):
                return False, f"Missing {key}; CPU costs cannot be compared fairly."
            if str(left[key]).strip().lower() != str(right[key]).strip().lower():
                return False, f"Results use different {key} settings."
        for key in ("batch_size", "record_count"):
            if left.get(key) is not None and right.get(key) is not None and left[key] != right[key]:
                return False, f"Results use different {key} settings."
    return True, "Comparable evaluation metadata."


def finite_metric(record: dict[str, Any] | None, metric: str) -> float | None:
    if not record:
        return None
    return _finite_number(record.get(metric))


def format_metric(value: Any, *, digits: int = 3, suffix: str = "") -> str:
    numeric = _finite_number(value)
    if numeric is None:
        return "Not measured"
    return f"{numeric:.{digits}f}{suffix}"
