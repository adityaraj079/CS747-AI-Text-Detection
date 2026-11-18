from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app_components.charts import confusion_matrix_figure
from app_components.data_loader import (
    finite_metric,
    format_metric,
    latest_result,
    load_json_if_present,
    partition_metadata,
)
from app_components.inference import show_live_predictions
from app_components.model_registry import model_availability
from app_components.styling import plotly_template


CUSTOM_MODEL_ID = "custom_model"
PUBLISHED_RESULT_PATH = Path("results") / "published" / "kaggle_first_place.json"


def _published_result(root: Path) -> dict[str, Any]:
    path = root / PUBLISHED_RESULT_PATH
    try:
        result = load_json_if_present(path)
    except (OSError, ValueError) as exc:
        st.error(f"Could not load the published Kaggle reference: {exc}")
        return {}
    if result is None:
        st.error(f"Published Kaggle reference is missing: {path}")
        return {}
    return result


def _published_auc(result: dict[str, Any]) -> float | None:
    value = result.get("private_leaderboard_roc_auc")
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def _standardized_results(
    results: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    return (
        latest_result(results, CUSTOM_MODEL_ID, "standard_test", "standardized"),
        latest_result(results, CUSTOM_MODEL_ID, "unseen_test", "standardized"),
    )


def _comparison_figure(
    published_auc: float | None,
    custom_result: dict[str, Any] | None,
) -> go.Figure | None:
    bars = []
    if published_auc is not None:
        bars.append(
            (
                "Kaggle private leaderboard",
                published_auc,
                "#d97706",
            )
        )
    custom_auc = finite_metric(custom_result, "roc_auc")
    if custom_auc is not None:
        bars.append(("Our standard test", custom_auc, "#0f766e"))
    if not bars:
        return None

    figure = go.Figure()
    for label, value, color in bars:
        figure.add_trace(
            go.Bar(
                name=label,
                x=[label],
                y=[value],
                text=[f"{value:.6f}"],
                textposition="outside",
                marker_color=color,
                hovertemplate=f"{label}<br>ROC-AUC: %{{y:.6f}}<extra></extra>",
            )
        )
    figure.update_layout(
        template=plotly_template(),
        title="ROC-AUC by evaluation source",
        height=360,
        margin=dict(l=12, r=12, t=52, b=12),
        showlegend=False,
        xaxis_title="",
        yaxis_title="ROC-AUC",
    )
    figure.update_yaxes(range=[0, 1.04])
    return figure


def _generalization_figure(
    standard: dict[str, Any] | None,
    unseen: dict[str, Any] | None,
) -> go.Figure | None:
    rows = []
    for label, record in (
        ("Standard test", standard),
        ("Unseen-source test", unseen),
    ):
        for metric, title in (("roc_auc", "ROC-AUC"), ("f1", "F1")):
            value = finite_metric(record, metric)
            if value is not None:
                rows.append(
                    {
                        "Evaluation set": label,
                        "Metric": title,
                        "Value": value,
                    }
                )
    if not rows:
        return None
    figure = px.bar(
        rows,
        x="Metric",
        y="Value",
        color="Evaluation set",
        barmode="group",
        template=plotly_template(),
        color_discrete_map={
            "Standard test": "#0f766e",
            "Unseen-source test": "#d97706",
        },
        text_auto=".4f",
    )
    figure.update_layout(
        height=370,
        margin=dict(l=12, r=12, t=24, b=12),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Measured score",
    )
    figure.update_yaxes(range=[0, 1])
    return figure


def _cpu_efficiency_figure(record: dict[str, Any] | None) -> go.Figure | None:
    latency = finite_metric(record, "latency_ms_per_essay")
    throughput = finite_metric(record, "throughput")
    if latency is None and throughput is None:
        return None

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "CPU latency (ms / essay)",
            "CPU throughput (essays / second)",
        ),
    )
    if latency is not None:
        figure.add_trace(
            go.Bar(
                x=["Our custom model"],
                y=[latency],
                text=[f"{latency:.3f}"],
                textposition="outside",
                marker_color="#0f766e",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if throughput is not None:
        figure.add_trace(
            go.Bar(
                x=["Our custom model"],
                y=[throughput],
                text=[f"{throughput:.2f}"],
                textposition="outside",
                marker_color="#0f766e",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    figure.update_layout(
        template=plotly_template(),
        height=330,
        margin=dict(l=12, r=12, t=55, b=12),
    )
    return figure


def _render_comparison_cards(
    published: dict[str, Any],
    published_auc: float | None,
    custom_spec: dict[str, Any],
    custom_result: dict[str, Any] | None,
) -> None:
    champion_column, custom_column = st.columns(2)
    with champion_column:
        with st.container(border=True):
            st.markdown("### Kaggle Champion")
            members = published.get("team_members")
            st.caption(
                "Original winning team: "
                + (", ".join(str(member) for member in members) if isinstance(members, list) else "Not verified")
            )
            metrics = st.columns(2)
            rank = published.get("historical_rank")
            metrics[0].metric(
                "Historical rank",
                f"#{rank}" if isinstance(rank, int) and rank > 0 else "Not verified",
            )
            metrics[1].metric(
                "Published private ROC-AUC",
                f"{published_auc:.6f}" if published_auc is not None else "Not verified",
            )
            st.write(published.get("architecture_summary") or "Architecture not verified.")
            st.caption("Evaluation status: published historical leaderboard result; not locally tested.")
            st.caption(
                str(
                    published.get(
                        "availability_status",
                        "Original fitted checkpoints unavailable/unverified",
                    )
                )
            )

    with custom_column:
        with st.container(border=True):
            st.markdown("### Our Custom Model")
            st.caption(custom_spec.get("version", "Version not recorded"))
            metrics = st.columns(3)
            for column, label, key, digits in (
                (metrics[0], "Local ROC-AUC", "roc_auc", 6),
                (metrics[1], "Accuracy", "accuracy", 4),
                (metrics[2], "F1", "f1", 4),
            ):
                column.metric(
                    label,
                    format_metric(finite_metric(custom_result, key), digits=digits),
                )
            metrics = st.columns(2)
            metrics[0].metric(
                "CPU latency (ms / essay)",
                format_metric(
                    finite_metric(custom_result, "latency_ms_per_essay"),
                    digits=3,
                ),
            )
            metrics[1].metric(
                "Model size (MB)",
                format_metric(finite_metric(custom_result, "model_size_mb"), digits=2),
            )
            status = (
                "Standardized local test result loaded"
                if custom_result
                else "Not evaluated on the standard test"
            )
            st.caption(f"Evaluation status: {status}")
            st.caption(f"Model: {custom_spec.get('name', 'Our custom model')}")


def render_page(
    repo_root: str | Path,
    results: list[dict[str, Any]],
    model_specs: dict[str, dict[str, Any]],
    split_manifest: dict[str, Any] | None,
    unseen_manifest: dict[str, Any] | None,
    adapter_loader,
) -> None:
    root = Path(repo_root)
    published = _published_result(root)
    published_auc = _published_auc(published)
    standard, unseen = _standardized_results(results)
    if not standard and not unseen and not model_availability(CUSTOM_MODEL_ID, root)[0]:
        st.title("Kaggle champion vs. our custom model")
        champion, custom = st.columns(2)
        with champion.container(border=True):
            st.subheader("Kaggle champion")
            st.metric("Private leaderboard ROC-AUC",
                      f"{published_auc:.6f}" if published_auc is not None else "Unavailable")
            st.caption(published.get("architecture_summary", ""))
        with custom.container(border=True):
            st.subheader("Our custom model")
            st.metric("Local test ROC-AUC", "Pending")
            st.caption("Training and evaluation on the project’s frozen test split are pending.")
        writeup = published.get("source_urls", {}).get("first_place_writeup")
        if writeup:
            st.markdown(f"[Official first-place write-up]({writeup})")
        return
    custom_spec = model_specs.get(CUSTOM_MODEL_ID, {})
    custom_spec = {
        **custom_spec,
        "name": "New custom neural network",
        "short_name": "Our custom model",
    }
    if not standard:
        custom_spec["version"] = "Training and evaluation pending"
        custom_spec["architecture"] = (
            "The new neural-network architecture and inference package have not been registered yet."
        )

    st.markdown('<div class="dashboard-eyebrow">CS747 · Historical reference and local evaluation</div>', unsafe_allow_html=True)
    st.title("Can Our Model Compete with Kaggle's Champion?")
    subtitle = (
        "Published leaderboard performance versus our independently evaluated AI-text detector."
        if standard
        else "Published Kaggle leaderboard performance; our new custom neural network has not been evaluated yet."
    )
    st.markdown(
        f'<div class="dashboard-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Main Comparison")
    _render_comparison_cards(published, published_auc, custom_spec, standard)

    st.subheader("Performance Comparison")
    if standard:
        st.warning(
            "These scores were obtained on different test datasets. This is a historical reference comparison, not a controlled head-to-head evaluation."
        )
    else:
        st.info(
            "The Kaggle score is historical. Our custom neural network has not been evaluated, so no local score is shown."
        )
    comparison_figure = _comparison_figure(published_auc, standard)
    if comparison_figure is None:
        st.info("ROC-AUC results will appear when a published or local score is available.")
    else:
        st.plotly_chart(comparison_figure, use_container_width=True)
    source_urls = published.get("source_urls", {})
    if isinstance(source_urls, dict):
        link_columns = st.columns(2)
        writeup_url = source_urls.get("first_place_writeup")
        leaderboard_url = source_urls.get("official_leaderboard")
        if writeup_url:
            link_columns[0].markdown(f"[Official first-place write-up]({writeup_url})")
        if leaderboard_url:
            link_columns[1].markdown(f"[Official competition leaderboard]({leaderboard_url})")

    st.subheader("Our Custom Model's Detailed Performance")
    detail_columns = st.columns(5)
    for column, label, key, digits in (
        (detail_columns[0], "ROC-AUC", "roc_auc", 6),
        (detail_columns[1], "Accuracy", "accuracy", 4),
        (detail_columns[2], "Precision", "precision", 4),
        (detail_columns[3], "Recall", "recall", 4),
        (detail_columns[4], "F1", "f1", 4),
    ):
        column.metric(
            label,
            format_metric(finite_metric(standard, key), digits=digits),
        )

    if standard:
        matrix_figure = confusion_matrix_figure(standard, template=plotly_template())
        if matrix_figure is None:
            st.info("No measured confusion matrix is saved for this evaluation.")
        else:
            st.markdown("**Measured confusion matrix**")
            st.plotly_chart(matrix_figure, use_container_width=True)
            tn, fp = standard["confusion_matrix"][0]
            fn, tp = standard["confusion_matrix"][1]
            counts = st.columns(4)
            for column, label, value in zip(
                counts,
                ("True human", "Human predicted AI", "AI predicted human", "True AI"),
                (tn, fp, fn, tp),
            ):
                column.metric(label, value)
    else:
        st.info("No standardized standard-test metrics or confusion matrix are available yet.")

    cpu_columns = st.columns(3)
    cpu_columns[0].metric(
        "CPU latency (ms / essay)",
        format_metric(finite_metric(standard, "latency_ms_per_essay"), digits=3),
    )
    cpu_columns[1].metric(
        "Throughput (essays / second)",
        format_metric(finite_metric(standard, "throughput"), digits=2),
    )
    cpu_columns[2].metric(
        "Model size (MB)",
        format_metric(finite_metric(standard, "model_size_mb"), digits=2),
    )
    if standard:
        benchmark = standard.get("benchmark", {})
        cpu = standard.get("cpu_model") or benchmark.get("cpu_model")
        if cpu:
            st.caption(
                f"Recorded CPU: {cpu}; thread count: {standard.get('thread_count', benchmark.get('thread_count', 'not recorded'))}; "
                f"batch size: {standard.get('batch_size', benchmark.get('batch_size', 'not recorded'))}."
            )

    st.subheader("Generalization and CPU Efficiency")
    unseen_is_valid = bool(
        unseen_manifest
        and unseen_manifest.get("selection_strategy") == "source_family_holdout"
        and unseen_manifest.get("held_out_sources")
    )
    same_dataset = bool(
        standard
        and unseen
        and standard.get("dataset_hash")
        and standard.get("dataset_hash") == unseen.get("dataset_hash")
    )
    if unseen_is_valid and same_dataset:
        standard_auc = finite_metric(standard, "roc_auc")
        unseen_auc = finite_metric(unseen, "roc_auc")
        gap = (
            abs(standard_auc - unseen_auc)
            if standard_auc is not None and unseen_auc is not None
            else None
        )
        generalization_columns = st.columns(3)
        generalization_columns[0].metric(
            "Standard test ROC-AUC",
            format_metric(standard_auc, digits=6),
        )
        generalization_columns[1].metric(
            "Unseen-source test ROC-AUC",
            format_metric(unseen_auc, digits=6),
        )
        generalization_columns[2].metric(
            "Absolute ROC-AUC gap",
            format_metric(gap, digits=6),
            help="Absolute difference between the two measured ROC-AUC values.",
        )
        st.caption(
            "Source-family holdout: "
            + ", ".join(str(source) for source in unseen_manifest["held_out_sources"])
        )
        generalization_chart = _generalization_figure(standard, unseen)
        if generalization_chart is not None:
            st.plotly_chart(generalization_chart, use_container_width=True)
        else:
            st.info("Generalization chart awaits measured ROC-AUC or F1 values.")
    else:
        st.info(
            "Standard and verified unseen-source results are not both available for the same dataset version."
        )

    st.markdown("**CPU efficiency measured for our custom model**")
    cpu_figure = _cpu_efficiency_figure(standard)
    if cpu_figure is not None:
        st.plotly_chart(cpu_figure, use_container_width=True)
    else:
        st.info("Custom-model CPU latency and throughput are not measured yet.")
    st.caption("Kaggle champion CPU latency and throughput: Not available from publication.")
    st.caption("Kaggle champion unseen-source ROC-AUC: Not available from publication.")

    st.subheader("Model Architecture Comparison")
    architecture_columns = st.columns(2)
    with architecture_columns[0]:
        st.markdown("**First-place ensemble**")
        st.write(published.get("architecture_summary") or "Architecture not verified.")
        st.caption(
            "The original fitted detection artifacts were not verified in this project; this model has not been run locally."
        )
    with architecture_columns[1]:
        st.markdown("**Our custom model**")
        st.write(custom_spec.get("architecture") or "Architecture details are not recorded.")
        if standard:
            st.caption("Local results are read from the standardized evaluation records.")
        else:
            st.caption("Training and standardized evaluation are pending.")

    repo_url = source_urls.get("official_solution_repository") if isinstance(source_urls, dict) else None
    if repo_url:
        st.markdown(f"[Official solution repository]({repo_url})")

    st.subheader("Live Detection")
    st.info(
        "Champion inference is unavailable because its original fitted detection artifacts have not been verified. No champion prediction is generated."
    )
    available, availability_detail = model_availability(CUSTOM_MODEL_ID, root)
    if available:
        show_live_predictions(
            (CUSTOM_MODEL_ID,),
            {CUSTOM_MODEL_ID: custom_spec},
            adapter_loader,
            section_title="Live essay prediction · our custom model",
        )
    else:
        st.info(f"Our custom model is not available for live inference: {availability_detail}")

    st.subheader("Scientific Interpretation")
    st.markdown(
        "- **Historical competition result:** the champion's private-leaderboard ROC-AUC is a published Kaggle score, not a local test result."
    )
    if standard:
        st.markdown(
            f"- **Independent local result:** our model's standard-test ROC-AUC is {format_metric(finite_metric(standard, 'roc_auc'), digits=6)} on {standard.get('record_count', 'an unrecorded number of')} held-out examples."
        )
    else:
        st.markdown("- **Independent local result:** no standardized standard-test result is available yet.")
    if standard and finite_metric(standard, "latency_ms_per_essay") is not None:
        st.markdown(
            f"- **CPU inference:** {format_metric(finite_metric(standard, 'latency_ms_per_essay'), digits=3)} ms per essay and {format_metric(finite_metric(standard, 'throughput'), digits=2)} essays per second, as recorded by the local benchmark."
        )
    else:
        st.markdown("- **CPU inference:** no measured custom-model CPU benchmark is available yet.")
    if unseen_is_valid and same_dataset:
        st.markdown(
            f"- **Generalization:** unseen-source ROC-AUC is {format_metric(finite_metric(unseen, 'roc_auc'), digits=6)}; this is a separate source-held-out evaluation."
        )
    else:
        st.markdown("- **Generalization:** verified unseen-source metrics are not available for this dataset version.")
    st.info(
        "The Kaggle private split and our frozen local test sets differ. Identical evaluation data and compatible protocols are required for a definitive model-to-model ranking; no winner is declared here."
    )

    standard_metadata = partition_metadata(split_manifest, "standard_test")
    unseen_metadata = partition_metadata(split_manifest, "unseen_test")
    if standard_metadata or unseen_metadata:
        with st.expander("Evaluation provenance"):
            st.write(
                {
                    "standard_test": standard_metadata,
                    "unseen_test": unseen_metadata,
                    "custom_standard_result_file": standard.get("source_file") if standard else None,
                    "custom_unseen_result_file": unseen.get("source_file") if unseen else None,
                }
            )
