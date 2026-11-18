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
from app_components.inference import threshold_decision
from app_components.model_registry import MODEL_IDS, load_model_adapter, model_availability
from app_components.model_catalog import MODEL_SPECS
from app_pages.team_comparison import (
    TEAM_MODEL_IDS,
    _comparison_frame,
    _comparison_table_results,
    _model_card_result,
    _model_detail_result,
)
from evaluation.model_adapters import ModelNotAvailableError, require_probability_vector


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.5, "AI-generated"), (0.9, "AI-generated"), (0.499, "Not AI-generated")],
)
def test_threshold_decision_names_the_predicted_class(probability, expected):
    assert threshold_decision(probability) == expected


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


def test_daniel_model_card_falls_back_to_historical_result_only_when_no_standardized_result():
    historical = {
        "model_id": "daniel_bert",
        "dataset_type": "standard_test",
        "result_category": "historical",
        "evaluation_partition": "daniel_original_80_10_10_test",
    }
    other_historical = {
        "model_id": "yash_5th_place",
        "dataset_type": "standard_test",
        "result_category": "historical",
    }
    assert _model_card_result([historical, other_historical], "daniel_bert") is historical
    assert _model_card_result([historical, other_historical], "yash_5th_place") is None

    standardized = {
        "model_id": "daniel_bert",
        "dataset_type": "standard_test",
        "result_category": "standardized",
    }
    assert _model_card_result([historical, standardized], "daniel_bert") is standardized


def test_daniel_historical_result_is_visible_in_table_and_model_detail_but_stays_separate():
    historical = {
        "model_id": "daniel_bert",
        "dataset_type": "standard_test",
        "result_category": "historical",
        "roc_auc": 0.966,
        "evaluation_partition": "daniel_original_80_10_10_test",
    }
    results = [historical]
    selected = {model_id: None for model_id in TEAM_MODEL_IDS}

    table_results = _comparison_table_results(
        results, selected, "standard_test", "standardized"
    )
    row = _comparison_frame(table_results, MODEL_SPECS).set_index("Model").loc["Daniel BERT"]
    assert row["Status"] == "Historical"
    assert row["Result type"] == "Historical"
    assert row["ROC-AUC"] == "0.966"
    assert _model_detail_result(results, selected, "daniel_bert", "standard_test") is historical
    assert _comparison_table_results(results, selected, "unseen_test", "standardized")[
        "daniel_bert"
    ] is None


def test_dashboard_allowlist_has_six_models_and_rejects_unknown_adapter():
    assert len(MODEL_IDS) == 6
    with pytest.raises(ModelNotAvailableError, match="not in the dashboard allowlist"):
        load_model_adapter("untrusted_model")


def test_model_availability_does_not_claim_missing_models_are_available(tmp_path):
    available, message = model_availability("kaggle_first_place", tmp_path)
    assert not available
    assert "not available" in message


def test_aakash_availability_requires_model_package_artifacts(tmp_path):
    artifacts = tmp_path / "models" / "aakash_7th_place" / "artifacts"
    artifacts.mkdir(parents=True)

    available, message = model_availability("aakash_7th_place", tmp_path)
    assert not available
    assert "configuration.json" in message

    for name in ("configuration.json", "tokenizer.json", "model.joblib"):
        (artifacts / name).touch()
    available, message = model_availability("aakash_7th_place", tmp_path)
    assert available
    assert "separate training split" in message


def test_aakash_adapter_is_registered_for_live_predictions():
    adapter = load_model_adapter("aakash_7th_place")
    probabilities = adapter.predict_proba(["A short essay for a live model smoke test."])

    assert probabilities.shape == (1,)
    assert 0.0 <= probabilities[0] <= 1.0


def test_aditya_adaptation_availability_requires_complete_artifact_set(tmp_path):
    artifacts = tmp_path / "models" / "aditya_3rd_place" / "artifacts"
    artifacts.mkdir(parents=True)
    required = (
        "bpe_tokenizer.json",
        "tfidf_vectorizer.joblib",
        "sgd_classifier.joblib",
        "catboost_classifier.cbm",
        "configuration.json",
    )
    available, message = model_availability("aditya_3rd_place", tmp_path)
    assert not available
    assert "bpe_tokenizer.json" in message

    for name in required:
        (artifacts / name).touch()
    available, message = model_availability("aditya_3rd_place", tmp_path)
    assert available
    assert "original submission settings are unverified" in message


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
