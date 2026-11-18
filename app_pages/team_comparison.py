from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app_components.charts import (
    confusion_matrix_figure,
    generalization_figure,
    generalization_gap_table,
    latency_quality_figure,
    metric_comparison_figure,
    performance_metric_records,
)
from app_components.data_loader import (
    compare_results_compatibility,
    finite_metric,
    format_metric,
    latest_result,
    partition_metadata,
)
from app_components.inference import show_live_predictions
from app_components.model_cards import render_model_card
from app_components.model_registry import model_availability
from app_components.styling import plotly_template

TEAM_MODEL_IDS = ("yash_5th_place", "aakash_7th_place", "aditya_3rd_place", "daniel_bert")
RESULT_CATEGORIES = {
    "Standardized benchmark": "standardized",
    "Historical results": "historical",
}
DATASET_TYPES = {
    "Standard held-out test": "standard_test",
    "Unseen-source test": "unseen_test",
}
TABLE_METRICS = (
    ("ROC-AUC", "roc_auc", 3),
    ("Accuracy", "accuracy", 3),
    ("Precision", "precision", 3),
    ("Recall", "recall", 3),
    ("F1", "f1", 3),
    ("CPU latency (ms / essay)", "latency_ms_per_essay", 2),
    ("Throughput (essays / second)", "throughput", 2),
    ("Model size (MB)", "model_size_mb", 2),
    ("Peak RAM (MB)", "peak_ram_mb", 2),
    ("Training time (seconds)", "training_time_seconds", 2),
)


def _comparison_frame(
    selected: dict[str, dict[str, Any] | None],
    model_specs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for model_id in TEAM_MODEL_IDS:
        record = selected.get(model_id)
        result_type = "Historical" if record and record.get("result_category") == "historical" else "Standardized"
        row: dict[str, Any] = {
            "Model": model_specs[model_id]["short_name"],
            "Status": result_type if record else "Not evaluated",
            "Result type": result_type if record else "Not evaluated",
        }
        for label, key, digits in TABLE_METRICS:
            row[label] = format_metric(finite_metric(record, key), digits=digits)
        matrix = record.get("confusion_matrix") if record else None
        if matrix:
            tn, fp = matrix[0]
            fn, tp = matrix[1]
            row.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp})
        else:
            row.update({"TN": "Not measured", "FP": "Not measured", "FN": "Not measured", "TP": "Not measured"})
        row["Dataset hash"] = record.get("dataset_hash", "Not recorded") if record else "Not evaluated"
        row["Split ID"] = record.get("split_id", "Not recorded") if record else "Not evaluated"
        row["Test records"] = record.get("record_count", "Not recorded") if record else "Not evaluated"
        row["CPU"] = record.get("cpu_model", "Not recorded") if record else "Not evaluated"
        row["Timestamp"] = record.get("timestamp", "Not recorded") if record else "Not evaluated"
        rows.append(row)
    return pd.DataFrame(rows)


def _all_share_dataset(selected: dict[str, dict[str, Any] | None]) -> tuple[bool, str]:
    available = [record for record in selected.values() if record]
    if len(available) < 2:
        return False, "At least two model results are required for a comparison."
    reference = available[0]
    for record in available[1:]:
        compatible, reason = compare_results_compatibility(reference, record)
        if not compatible:
            return False, reason
    return True, "All available results share compatible dataset provenance."


def _model_card_result(
    results: list[dict[str, Any]],
    model_id: str,
) -> dict[str, Any] | None:
    standardized = latest_result(results, model_id, "standard_test", "standardized")
    if standardized is not None or model_id != "daniel_bert":
        return standardized
    return latest_result(results, model_id, "standard_test", "historical")


def _comparison_table_results(
    results: list[dict[str, Any]],
    selected: dict[str, dict[str, Any] | None],
    dataset_type: str,
    result_category: str,
) -> dict[str, dict[str, Any] | None]:
    table_results = dict(selected)
    if dataset_type == "standard_test" and result_category == "standardized":
        table_results["daniel_bert"] = table_results.get("daniel_bert") or latest_result(
            results,
            "daniel_bert",
            "standard_test",
            "historical",
        )
    return table_results


def _model_detail_result(
    results: list[dict[str, Any]],
    selected: dict[str, dict[str, Any] | None],
    model_id: str,
    dataset_type: str,
) -> dict[str, Any] | None:
    result = selected.get(model_id)
    if result is None and model_id == "daniel_bert" and dataset_type == "standard_test":
        return latest_result(results, model_id, "standard_test", "historical")
    return result


def _render_model_detail(
    spec: dict[str, Any],
    result: dict[str, Any] | None,
) -> None:
    st.markdown(f"#### {spec['name']}")
    st.write(f"**Architecture:** {spec['architecture']}")
    st.write(f"**Version/status:** {spec['version']}")
    st.write(f"**Availability:** {spec['availability']}")
    st.write(f"**Integration note:** {spec['status_detail']}")
    if spec.get("reference"):
        st.markdown(f"[{spec['reference_label']}]({spec['reference']})")
    if result:
        st.write(
            {
                "dataset_type": result.get("dataset_type"),
                "dataset_hash": result.get("dataset_hash"),
                "split_id": result.get("split_id"),
                "record_count": result.get("record_count"),
                "threshold": result.get("threshold"),
                "cpu_model": result.get("cpu_model"),
                "thread_count": result.get("thread_count"),
                "timestamp": result.get("timestamp"),
                "source_file": result.get("source_file"),
            }
        )
        figure = confusion_matrix_figure(result, template=plotly_template())
        if figure is not None:
            st.plotly_chart(
                figure,
                use_container_width=True,
                key=f"team-model-detail-confusion-{spec['name']}",
            )
        else:
            st.info("No actual confusion matrix is present in this result.")
    else:
        st.info("No saved evaluation result is available for this model and dataset.")
    st.caption(
        "No member training notebooks or fitted weights are loaded by this page. "
        "Do not interpret missing metrics as zero."
    )


def render_page(
    repo_root: str | Path,
    results: list[dict[str, Any]],
    model_specs: dict[str, dict[str, Any]],
    split_manifest: dict[str, Any] | None,
    unseen_manifest: dict[str, Any] | None,
    adapter_loader,
) -> None:
    root = Path(repo_root)
    st.markdown('<div class="dashboard-eyebrow">CS747 · Reproducibility and benchmarking</div>', unsafe_allow_html=True)
    st.title("Reproducing and Benchmarking Kaggle Solutions")
    st.markdown(
        '<div class="dashboard-subtitle">Comparative evaluation of four independently implemented AI-generated text detectors.</div>',
        unsafe_allow_html=True,
    )

    card_results = {
        model_id: _model_card_result(results, model_id)
        for model_id in TEAM_MODEL_IDS
    }
    card_columns = st.columns(2)
    for index, model_id in enumerate(TEAM_MODEL_IDS):
        with card_columns[index % 2]:
            render_model_card(
                model_specs[model_id],
                card_results[model_id],
                model_availability(model_id, root),
                compact=True,
            )

    st.subheader("Shared benchmark configuration")
    controls = st.columns(2)
    category_label = controls[0].selectbox(
        "Result collection",
        list(RESULT_CATEGORIES),
        key="team-result-category",
    )
    dataset_label = controls[1].selectbox(
        "Evaluation set",
        list(DATASET_TYPES),
        key="team-dataset-type",
    )
    category = RESULT_CATEGORIES[category_label]
    dataset_type = DATASET_TYPES[dataset_label]
    selected = performance_metric_records(results, TEAM_MODEL_IDS, dataset_type, category)
    metadata = partition_metadata(split_manifest, dataset_type)

    info_columns = st.columns(4)
    if metadata:
        info_columns[0].metric("Frozen records", metadata.get("record_count") or "Not available")
        info_columns[1].metric("Seed", metadata.get("random_seed") or "Not available")
        info_columns[2].markdown("**Dataset hash**")
        info_columns[2].code(str(metadata.get("dataset_hash") or "Not available")[:18] + "…")
        info_columns[3].markdown("**Split ID**")
        info_columns[3].code(str(metadata.get("split_id") or "Not available")[:18] + "…")
    else:
        st.info("Dataset split manifest is unavailable; benchmark provenance cannot be verified.")

    available_records = [row for row in selected.values() if row]
    all_compatible, compatibility_note = _all_share_dataset(selected)
    if len(available_records) < len(TEAM_MODEL_IDS):
        st.info(
            f"{len(available_records)} of {len(TEAM_MODEL_IDS)} team models have saved results "
            f"for this selection. Missing models remain 'Not evaluated'."
        )
    if available_records and not all_compatible:
        st.warning(
            "Selected results do not all share the same dataset hash, split ID, and dataset type. "
            f"Side-by-side visual comparisons are withheld: {compatibility_note}"
        )
    elif available_records and category == "historical":
        st.warning("Historical results can differ in split and hardware; use for progress tracking, not final ranking.")
    elif not available_records:
        st.info("No saved results are available for this selection. Standardized benchmarking is pending.")

    st.subheader("Performance comparison")
    table_results = _comparison_table_results(results, selected, dataset_type, category)
    if table_results.get("daniel_bert") and table_results["daniel_bert"].get("result_category") == "historical":
        st.caption(
            "Daniel BERT's row is its saved historical 80/10/10 evaluation; "
            "it is shown for reference and excluded from frozen-split comparisons."
        )
    st.dataframe(_comparison_frame(table_results, model_specs), use_container_width=True, hide_index=True)

    if all_compatible:
        st.subheader("Detection metrics")
        detection_figure = metric_comparison_figure(
            selected,
            model_specs,
            ["roc_auc", "accuracy", "f1"],
            template=plotly_template(),
        )
        if detection_figure is not None:
            st.plotly_chart(
                detection_figure,
                use_container_width=True,
                key="team-detection-metrics",
            )
        else:
            st.info("Detection-metric chart pending actual saved metrics.")

        st.subheader("Precision, recall, and F1")
        prf_figure = metric_comparison_figure(
            selected,
            model_specs,
            ["precision", "recall", "f1"],
            template=plotly_template(),
        )
        if prf_figure is not None:
            st.plotly_chart(prf_figure, use_container_width=True, key="team-prf-metrics")
        else:
            st.info("Precision/recall chart pending actual saved metrics.")
    else:
        st.info("Detection charts will render once results are available on a shared frozen test split.")

    daniel_result = _model_detail_result(results, selected, "daniel_bert", dataset_type)
    if daniel_result and daniel_result.get("result_category") == "historical":
        st.subheader("Daniel BERT historical evaluation")
        st.warning(
            daniel_result.get(
                "training_split_warning",
                "This original 80/10/10 result is historical and not comparable with frozen-split results.",
            )
        )
        daniel_only_result = {"daniel_bert": daniel_result}
        metric_figure = metric_comparison_figure(
            daniel_only_result,
            {"daniel_bert": model_specs["daniel_bert"]},
            ["roc_auc", "accuracy", "precision", "recall", "f1"],
            template=plotly_template(),
        )
        if metric_figure is not None:
            metric_figure.update_layout(title="Daniel BERT — original 80/10/10 test metrics")
            st.plotly_chart(
                metric_figure,
                use_container_width=True,
                key="daniel-historical-metrics",
            )
        confusion_figure = confusion_matrix_figure(daniel_result, template=plotly_template())
        if confusion_figure is not None:
            st.plotly_chart(
                confusion_figure,
                use_container_width=True,
                key="daniel-historical-confusion",
            )
        latency_figure = latency_quality_figure(
            [daniel_result],
            {"daniel_bert": model_specs["daniel_bert"]},
            template=plotly_template(),
        )
        if latency_figure is not None:
            latency_figure.update_layout(title="Daniel BERT — measured CPU performance")
            st.plotly_chart(
                latency_figure,
                use_container_width=True,
                key="daniel-historical-cpu-performance",
            )

    st.subheader("Generalization")
    unseen_is_valid = bool(
        unseen_manifest
        and unseen_manifest.get("selection_strategy") == "source_family_holdout"
        and unseen_manifest.get("held_out_sources")
    )
    if unseen_is_valid:
        gen_figure = generalization_figure(
            results,
            {model_id: model_specs[model_id] for model_id in TEAM_MODEL_IDS},
            result_category=category,
            unseen_is_valid=True,
            template=plotly_template(),
        )
        if gen_figure is not None:
            st.plotly_chart(gen_figure, use_container_width=True, key="team-generalization")
        else:
            st.info("Standard and unseen-source results for the same model and dataset version are pending.")
        gap_table = generalization_gap_table(
            results,
            {model_id: model_specs[model_id] for model_id in TEAM_MODEL_IDS},
            TEAM_MODEL_IDS,
            result_category=category,
            unseen_is_valid=True,
        )
        if not gap_table.empty:
            st.caption("Absolute change is |unseen-source metric − standard-test metric|.")
            st.dataframe(gap_table, use_container_width=True, hide_index=True)
    else:
        st.warning("No valid source-disjoint evaluation manifest is available; unseen-source comparisons are disabled.")

    st.subheader("Accuracy versus CPU cost")
    cpu_records = []
    if all_compatible and available_records:
        reference = available_records[0]
        cpu_compatible = all(
            compare_results_compatibility(reference, row, require_cpu_compatibility=True)[0]
            for row in available_records[1:]
        )
        if cpu_compatible:
            cpu_records = available_records
        else:
            st.info("CPU cost chart withheld because hardware/thread benchmark settings differ or are missing.")
    cpu_figure = latency_quality_figure(
        cpu_records,
        {model_id: model_specs[model_id] for model_id in TEAM_MODEL_IDS},
        template=plotly_template(),
        highlight_frontier=True,
    )
    if cpu_figure is not None:
        st.plotly_chart(cpu_figure, use_container_width=True, key="team-cpu-cost")
    else:
        st.info("Accuracy-versus-cost chart pending compatible measured latency and ROC-AUC.")

    st.subheader("Model-level detail")
    detail_id = st.selectbox(
        "Select a team model",
        TEAM_MODEL_IDS,
        format_func=lambda model_id: model_specs[model_id]["name"],
        index=TEAM_MODEL_IDS.index("daniel_bert"),
        key="team-model-detail",
    )
    _render_model_detail(
        model_specs[detail_id],
        _model_detail_result(results, selected, detail_id, dataset_type),
    )

    show_live_predictions(
        TEAM_MODEL_IDS,
        model_specs,
        adapter_loader,
        section_title="Live four-model prediction",
    )

    if metadata:
        st.caption(
            f"Dataset manifest created: {metadata.get('creation_time_utc') or 'not recorded'} · "
            "CPU model and software versions are shown from each saved result when available."
        )
