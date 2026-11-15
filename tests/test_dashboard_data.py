from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_components.data_loader import (
    compare_results_compatibility,
    format_metric,
    latest_result,
    load_result_records,
    normalize_result_record,
)
from app_components.charts import generalization_gap_table, metric_comparison_figure
from app_components.model_registry import MODEL_IDS, load_model_adapter, model_availability
from evaluation.model_adapters import ModelNotAvailableError, require_probability_vector


def test_loads_standardized_and_historical_results_with_cpu_aliases(tmp_path):
    standardized = tmp_path / "standardized"
    historical = tmp_path / "historical"
    standardized.mkdir()
    historical.mkdir()
    (standardized / "aditya.json").write_text(
        json.dumps(
            {
                "model_id": "aditya_3rd_place",
                "dataset_hash": "dataset-a",
                "split_id": "split-a",
                "dataset_type": "test",
                "metrics": {"roc_auc": 0.81, "f1": 0.73},
                "milliseconds_per_essay": 1.25,
                "throughput_essays_per_second": 800.0,
                "cpu_model": "CPU A",
                "thread_count": 4,
                "record_count": 100,
                "timestamp": "2026-11-14T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (historical / "yash.csv").write_text(
        "model_id,dataset_type,roc_auc,accuracy\n"
        "yash_5th_place,standard_test,0.78,0.75\n",
        encoding="utf-8",
    )

    records = load_result_records(tmp_path)

    assert len(records) == 2
    aditya = next(record for record in records if record["model_id"] == "aditya_3rd_place")
    yash = next(record for record in records if record["model_id"] == "yash_5th_place")
    assert aditya["dataset_type"] == "standard_test"
    assert aditya["result_category"] == "standardized"
    assert aditya["latency_ms_per_essay"] == pytest.approx(1.25)
    assert aditya["throughput"] == pytest.approx(800.0)
    assert aditya["record_count"] == 100
    assert yash["result_category"] == "historical"
    assert yash["accuracy"] == pytest.approx(0.75)


def test_result_compatibility_requires_matching_provenance_and_hardware():
    first = {
        "dataset_hash": "hash-1",
        "split_id": "split-1",
        "dataset_type": "standard_test",
        "result_category": "standardized",
        "cpu_model": "cpu",
        "thread_count": 4,
        "batch_size": 32,
        "record_count": 100,
    }
    second = dict(first)
    assert compare_results_compatibility(first, second) == (True, "Comparable evaluation metadata.")
    assert compare_results_compatibility(first, second, require_cpu_compatibility=True)[0]

    second["split_id"] = "different-split"
    compatible, reason = compare_results_compatibility(first, second)
    assert not compatible
    assert "split_id" in reason


def test_cpu_comparison_requires_matching_hardware_metadata():
    first = {
        "dataset_hash": "hash",
        "split_id": "split",
        "dataset_type": "standard_test",
        "result_category": "standardized",
        "cpu_model": "CPU A",
        "thread_count": 4,
    }
    second = dict(first, cpu_model="CPU B")
    compatible, reason = compare_results_compatibility(
        first, second, require_cpu_compatibility=True
    )
    assert not compatible
    assert "cpu_model" in reason


def test_missing_metrics_render_as_not_measured_and_non_finite_values_are_ignored():
    record = normalize_result_record({"model_id": "custom_model", "roc_auc": float("nan")})
    assert record["roc_auc"] is None
    assert format_metric(None) == "Not measured"


def test_latest_result_selects_latest_timestamp():
    rows = [
        {"model_id": "m", "dataset_type": "standard_test", "result_category": "standardized", "timestamp": "2026-01-01"},
        {"model_id": "m", "dataset_type": "standard_test", "result_category": "standardized", "timestamp": "2026-02-01"},
    ]
    assert latest_result(rows, "m", "test", "standardized")["timestamp"] == "2026-02-01"


def test_dashboard_allowlist_has_six_models_and_rejects_unknown_adapter():
    assert len(MODEL_IDS) == 6
    with pytest.raises(ModelNotAvailableError, match="not in the dashboard allowlist"):
        load_model_adapter("untrusted_model")


def test_model_availability_does_not_claim_missing_models_are_available(tmp_path):
    available, message = model_availability("kaggle_first_place", tmp_path)
    assert not available
    assert "not available" in message


def test_adapter_probabilities_are_validated():
    assert require_probability_vector([0.0, 0.75, 1.0], 3).tolist() == [0.0, 0.75, 1.0]
    with pytest.raises(ValueError, match="1D"):
        require_probability_vector([[0.5]], 1)


def test_generalization_gap_requires_matching_dataset_and_reports_absolute_change():
    specs = {"m": {"short_name": "Model"}}
    records = [
        {
            "model_id": "m",
            "dataset_type": "standard_test",
            "result_category": "standardized",
            "dataset_hash": "dataset-1",
            "split_id": "standard-split",
            "roc_auc": 0.90,
            "f1": 0.80,
        },
        {
            "model_id": "m",
            "dataset_type": "unseen_test",
            "result_category": "standardized",
            "dataset_hash": "dataset-1",
            "split_id": "unseen-split",
            "roc_auc": 0.72,
            "f1": 0.63,
        },
    ]
    gaps = generalization_gap_table(
        records,
        specs,
        ("m",),
        result_category="standardized",
        unseen_is_valid=True,
    )
    assert gaps.loc[0, "Absolute ROC-AUC change"] == pytest.approx(0.18)
    assert gaps.loc[0, "Absolute F1 change"] == pytest.approx(0.17)

    records[1]["dataset_hash"] = "different-dataset"
    assert generalization_gap_table(
        records,
        specs,
        ("m",),
        result_category="standardized",
        unseen_is_valid=True,
    ).empty


def test_empty_metric_plot_does_not_create_fake_data():
    assert metric_comparison_figure(
        {"m": None},
        {"m": {"short_name": "Model"}},
        ["roc_auc", "accuracy"],
        template="plotly_white",
    ) is None
