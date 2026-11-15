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

MODEL_IDS = ("kaggle_first_place", "custom_model")
RESULT_CATEGORIES = {
    "Standardized benchmark": "standardized",
    "Historical results": "historical",
}
DATASET_TYPES = {
    "Standard held-out test": "standard_test",
    "Unseen-source test": "unseen_test",
}
DISPLAY_METRICS = (
    ("ROC-AUC", "roc_auc", 3),
    ("Accuracy", "accuracy", 3),
    ("Precision", "precision", 3),
    ("Recall", "recall", 3),
    ("F1", "f1", 3),
    ("CPU latency (ms / essay)", "latency_ms_per_essay", 2),
    ("Throughput (essays / second)", "throughput", 2),
    ("Model size (MB)", "model_size_mb", 2),
    ("Peak inference RAM (MB)", "peak_ram_mb", 2),
)


def _result_table(
    champion: dict[str, Any] | None,
    custom: dict[str, Any] | None,
) -> pd.DataFrame:
    comparable, reason = compare_results_compatibility(champion, custom)
    cpu_comparable, cpu_reason = compare_results_compatibility(
        champion, custom, require_cpu_compatibility=True
    )
    rows = []
    for label, key, digits in DISPLAY_METRICS:
        left = finite_metric(champion, key)
        right = finite_metric(custom, key)
        if key in {"latency_ms_per_essay", "throughput", "peak_ram_mb"}:
            can_compare = cpu_comparable
            difference_reason = cpu_reason
        else:
            can_compare = comparable
            difference_reason = reason
        difference = (
            format_metric(right - left, digits=digits)
            if can_compare and left is not None and right is not None
            else "—"
        )
        rows.append(
            {
                "Metric": label,
                "Kaggle #1": format_metric(left, digits=digits),
                "Our custom model": format_metric(right, digits=digits),
                "Custom − Kaggle #1": difference,
                "Comparison": "Comparable" if can_compare else difference_reason,
            }
        )
    return pd.DataFrame(rows)


def _research_conclusion(
    champion: dict[str, Any] | None,
    custom: dict[str, Any] | None,
    records: list[dict[str, Any]],
    category: str,
    unseen_is_valid: bool,
) -> None:
    st.subheader("Research conclusion")
    comparable, reason = compare_results_compatibility(champion, custom)
    cpu_comparable, cpu_reason = compare_results_compatibility(
        champion, custom, require_cpu_compatibility=True
    )
    if not comparable:
        st.info(
            "A controlled quality comparison cannot yet be made. "
            f"{reason} No winner is declared."
        )
    else:
        champion_auc = finite_metric(champion, "roc_auc")
        custom_auc = finite_metric(custom, "roc_auc")
        if champion_auc is not None and custom_auc is not None:
            higher = "Our custom model" if custom_auc > champion_auc else "Kaggle #1"
            if custom_auc == champion_auc:
                st.write("The two models have equal local ROC-AUC on the selected comparable evaluation set.")
            else:
                st.write(f"{higher} has the higher local ROC-AUC on the selected comparable evaluation set.")
        else:
            st.info("ROC-AUC is missing for at least one model; a quality conclusion is unavailable.")

    if cpu_comparable:
        champion_latency = finite_metric(champion, "latency_ms_per_essay")
        custom_latency = finite_metric(custom, "latency_ms_per_essay")
        if champion_latency is not None and custom_latency is not None:
            faster = "Our custom model" if custom_latency < champion_latency else "Kaggle #1"
            st.write(f"{faster} has lower measured CPU latency under matching hardware settings.")
        else:
            st.info("CPU latency is missing for at least one model.")
    else:
        st.info(f"CPU efficiency comparison pending: {cpu_reason}")

    champion_unseen = latest_result(records, "kaggle_first_place", "unseen_test", category)
    custom_unseen = latest_result(records, "custom_model", "unseen_test", category)
    champion_standard = latest_result(records, "kaggle_first_place", "standard_test", category)
    custom_standard = latest_result(records, "custom_model", "standard_test", category)
    if unseen_is_valid and all((champion_unseen, custom_unseen, champion_standard, custom_standard)):
        st.write(
            "Both models have standard and source-held-out results. Review the generalization "
            "chart; compare changes only if the underlying dataset and split provenance match."
        )
    else:
        st.info(
            "Generalization improvement and the predefined success criterion cannot be assessed "
            "until both models have validated standard and unseen-source results."
        )
    st.caption(
        "No numeric success criterion is configured in this repository yet; final success claims "
        "must be tied to a predeclared validation-based criterion."
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
    st.markdown('<div class="dashboard-eyebrow">CS747 · Controlled model evaluation</div>', unsafe_allow_html=True)
    st.title("Can Our AI Detector Compete with Kaggle's Champion?")
    st.markdown(
        '<div class="dashboard-subtitle">Evaluating detection performance, generalization, and CPU inference efficiency.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Research objective")
    st.write(
        "Compare our custom detector with the **main competition private-leaderboard first-place "
        "solution**. A fair comparison requires compatible local results on the same frozen "
        "evaluation records. The Efficiency Prize winner is a different competition track."
    )
    st.markdown(
        f"[Open the official first-place write-up]({model_specs['kaggle_first_place']['reference']})"
    )

    selector_columns = st.columns(2)
    category_label = selector_columns[0].selectbox(
        "Result collection",
        list(RESULT_CATEGORIES),
        key="champion-result-category",
    )
    dataset_label = selector_columns[1].selectbox(
        "Evaluation set",
        list(DATASET_TYPES),
        key="champion-dataset-type",
    )
    category = RESULT_CATEGORIES[category_label]
    dataset_type = DATASET_TYPES[dataset_label]
    selected = performance_metric_records(results, MODEL_IDS, dataset_type, category)

    if not results or not any(selected.values()):
        if category == "standardized":
            st.info(
                "No standardized benchmark results are available yet. Metrics below remain "
                "'Not evaluated'; the dashboard does not substitute leaderboard scores."
            )
        else:
            st.info("No imported historical result records are available.")
    elif category == "historical":
        st.warning("Historical results may use different test sets or hardware; they are progress records, not a controlled head-to-head.")

    card_columns = st.columns(2)
    for column, model_id in zip(card_columns, MODEL_IDS):
        with column:
            render_model_card(
                model_specs[model_id],
                selected[model_id],
                model_availability(model_id, root),
            )

    st.subheader("Controlled comparison")
    comparison_ok, comparison_reason = compare_results_compatibility(
        selected[MODEL_IDS[0]], selected[MODEL_IDS[1]]
    )
    if not comparison_ok and any(selected.values()):
        st.warning(f"Metric deltas and winner claims are disabled: {comparison_reason}")
    st.dataframe(
        _result_table(selected[MODEL_IDS[0]], selected[MODEL_IDS[1]]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Detection performance")
    if comparison_ok:
        figure = metric_comparison_figure(
            selected,
            model_specs,
            ["roc_auc", "accuracy", "f1"],
            template=plotly_template(),
        )
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Detection metrics will appear after compatible local results are saved.")
    else:
        st.info("Chart withheld until both models have results on the same dataset hash and split.")

    st.subheader("CPU efficiency versus detection quality")
    cpu_ok, cpu_reason = compare_results_compatibility(
        selected[MODEL_IDS[0]], selected[MODEL_IDS[1]], require_cpu_compatibility=True
    )
    if cpu_ok:
        figure = latency_quality_figure(
            [selected[model_id] for model_id in MODEL_IDS if selected[model_id]],
            model_specs,
            template=plotly_template(),
        )
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No compatible CPU latency and ROC-AUC measurements are available.")
    else:
        st.info(f"CPU scatter plot withheld: {cpu_reason}")

    st.subheader("Generalization gap")
    unseen_is_valid = bool(
        unseen_manifest
        and unseen_manifest.get("selection_strategy") == "source_family_holdout"
        and unseen_manifest.get("held_out_sources")
    )
    if unseen_is_valid:
        st.caption(
            "The unseen partition is source-group-held-out per the dataset manifest. "
            "Held-out groups: " + ", ".join(unseen_manifest["held_out_sources"])
        )
        figure = generalization_figure(
            results,
            {model_id: model_specs[model_id] for model_id in MODEL_IDS},
            result_category=category,
            unseen_is_valid=True,
            template=plotly_template(),
        )
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Generalization results are pending for at least one model.")
        gap_table = generalization_gap_table(
            results,
            {model_id: model_specs[model_id] for model_id in MODEL_IDS},
            MODEL_IDS,
            result_category=category,
            unseen_is_valid=True,
        )
        if not gap_table.empty:
            st.caption("Absolute change is |unseen-source metric − standard-test metric|.")
            st.dataframe(gap_table, use_container_width=True, hide_index=True)
    else:
        st.warning("The unseen-source manifest is absent or does not certify a source-family holdout; no OOD claim will be shown.")

    st.subheader("Confusion matrix")
    matrix_model = st.selectbox(
        "Model",
        MODEL_IDS,
        format_func=lambda model_id: model_specs[model_id]["name"],
        key="champion-confusion-model",
    )
    matrix_record = selected[matrix_model]
    matrix_figure = confusion_matrix_figure(matrix_record, template=plotly_template())
    if matrix_figure is None:
        st.info("No actual confusion matrix is saved for this model and evaluation set.")
    else:
        matrix = matrix_record["confusion_matrix"]
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        st.plotly_chart(matrix_figure, use_container_width=True)
        columns = st.columns(4)
        for column, label, value in zip(columns, ("TN", "FP", "FN", "TP"), (tn, fp, fn, tp)):
            column.metric(label, value)

    _research_conclusion(selected[MODEL_IDS[0]], selected[MODEL_IDS[1]], results, category, unseen_is_valid)
    show_live_predictions(MODEL_IDS, model_specs, adapter_loader)

    metadata = partition_metadata(split_manifest, dataset_type)
    if metadata:
        with st.expander("Evaluation dataset provenance"):
            st.write(
                {
                    "dataset_type": dataset_type,
                    "dataset_hash": metadata["dataset_hash"],
                    "split_id": metadata["split_id"],
                    "record_count": metadata["record_count"],
                    "random_seed": metadata["random_seed"],
                }
            )
