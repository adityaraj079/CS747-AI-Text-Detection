from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app_components.data_loader import finite_metric, latest_result


_METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "latency_ms_per_essay": "CPU latency (ms / essay)",
    "throughput": "Throughput (essays / second)",
}


def metric_comparison_figure(
    results_by_model: dict[str, dict[str, Any] | None],
    model_specs: dict[str, dict[str, Any]],
    metric_keys: list[str],
    *,
    template: str,
) -> go.Figure | None:
    rows = []
    for model_id, record in results_by_model.items():
        if record is None:
            continue
        spec = model_specs[model_id]
        for metric in metric_keys:
            value = finite_metric(record, metric)
            if value is not None:
                rows.append(
                    {
                        "Model": spec["short_name"],
                        "Metric": _METRIC_LABELS.get(metric, metric),
                        "Value": value,
                    }
                )
    if not rows:
        return None
    figure = px.bar(
        pd.DataFrame(rows),
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        template=template,
        color_discrete_sequence=["#0f766e", "#64748b", "#d97706", "#7c3aed"],
    )
    figure.update_layout(
        height=360,
        margin=dict(l=12, r=12, t=24, b=12),
        legend_title_text="",
        yaxis_title="Score",
        xaxis_title="",
    )
    figure.update_yaxes(range=[0, 1] if all(key in {"roc_auc", "accuracy", "precision", "recall", "f1"} for key in metric_keys) else None)
    return figure


def latency_quality_figure(
    records: list[dict[str, Any]],
    model_specs: dict[str, dict[str, Any]],
    *,
    template: str,
    highlight_frontier: bool = False,
) -> go.Figure | None:
    rows = []
    for record in records:
        model_id = record.get("model_id")
        if model_id not in model_specs:
            continue
        latency = finite_metric(record, "latency_ms_per_essay")
        roc_auc = finite_metric(record, "roc_auc")
        if latency is None or roc_auc is None:
            continue
        rows.append(
            {
                "Model": model_specs[model_id]["short_name"],
                "model_id": model_id,
                "Latency (ms / essay)": latency,
                "ROC-AUC": roc_auc,
                "F1": finite_metric(record, "f1"),
                "Throughput (essays / second)": finite_metric(record, "throughput"),
                "CPU": record.get("cpu_model") or "Not recorded",
                "Dataset hash": record.get("dataset_hash") or "Not recorded",
            }
        )
    if not rows:
        return None
    if highlight_frontier and len(rows) >= 3:
        for row in rows:
            row["Pareto status"] = "Pareto-efficient" if not any(
                other["ROC-AUC"] >= row["ROC-AUC"]
                and other["Latency (ms / essay)"] <= row["Latency (ms / essay)"]
                and (
                    other["ROC-AUC"] > row["ROC-AUC"]
                    or other["Latency (ms / essay)"] < row["Latency (ms / essay)"]
                )
                for other in rows
                if other is not row
            ) else "Dominated"
    else:
        for row in rows:
            row["Pareto status"] = "Not assessed"

    figure = px.scatter(
        pd.DataFrame(rows),
        x="Latency (ms / essay)",
        y="ROC-AUC",
        color="Model",
        symbol="Pareto status" if highlight_frontier and len(rows) >= 3 else None,
        hover_name="Model",
        hover_data=[
            "F1",
            "Throughput (essays / second)",
            "CPU",
            "Dataset hash",
            "Pareto status",
        ],
        template=template,
        color_discrete_sequence=["#0f766e", "#64748b", "#d97706", "#7c3aed"],
    )
    figure.update_layout(height=380, margin=dict(l=12, r=12, t=24, b=12), legend_title_text="")
    figure.update_yaxes(range=[0, 1])
    return figure


def generalization_figure(
    records: list[dict[str, Any]],
    model_specs: dict[str, dict[str, Any]],
    *,
    result_category: str,
    unseen_is_valid: bool,
    template: str,
) -> go.Figure | None:
    if not unseen_is_valid:
        return None
    rows = []
    for model_id, spec in model_specs.items():
        standard = latest_result(records, model_id, "standard_test", result_category)
        unseen = latest_result(records, model_id, "unseen_test", result_category)
        if not standard or not unseen or standard.get("dataset_hash") != unseen.get("dataset_hash"):
            continue
        for label, record in (("Standard test", standard), ("Unseen-source test", unseen)):
            for metric in ("roc_auc", "f1"):
                value = finite_metric(record, metric)
                if value is not None:
                    rows.append(
                        {
                            "Model": spec["short_name"],
                            "Evaluation set": label,
                            "Metric": _METRIC_LABELS[metric],
                            "Value": value,
                        }
                    )
    if not rows:
        return None
    figure = px.bar(
        pd.DataFrame(rows),
        x="Model",
        y="Value",
        color="Evaluation set",
        facet_col="Metric",
        barmode="group",
        template=template,
        color_discrete_sequence=["#0f766e", "#d97706"],
    )
    figure.update_layout(height=370, margin=dict(l=12, r=12, t=48, b=12), legend_title_text="")
    figure.update_yaxes(range=[0, 1])
    return figure


def generalization_gap_table(
    records: list[dict[str, Any]],
    model_specs: dict[str, dict[str, Any]],
    model_ids: tuple[str, ...],
    *,
    result_category: str,
    unseen_is_valid: bool,
) -> pd.DataFrame:
    rows = []
    if not unseen_is_valid:
        return pd.DataFrame()
    for model_id in model_ids:
        standard = latest_result(records, model_id, "standard_test", result_category)
        unseen = latest_result(records, model_id, "unseen_test", result_category)
        if not standard or not unseen or standard.get("dataset_hash") != unseen.get("dataset_hash"):
            continue
        standard_auc = finite_metric(standard, "roc_auc")
        unseen_auc = finite_metric(unseen, "roc_auc")
        standard_f1 = finite_metric(standard, "f1")
        unseen_f1 = finite_metric(unseen, "f1")
        rows.append(
            {
                "Model": model_specs[model_id]["short_name"],
                "Standard ROC-AUC": standard_auc,
                "Unseen ROC-AUC": unseen_auc,
                "Absolute ROC-AUC change": abs(unseen_auc - standard_auc) if standard_auc is not None and unseen_auc is not None else None,
                "Standard F1": standard_f1,
                "Unseen F1": unseen_f1,
                "Absolute F1 change": abs(unseen_f1 - standard_f1) if standard_f1 is not None and unseen_f1 is not None else None,
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_figure(record: dict[str, Any] | None, *, template: str) -> go.Figure | None:
    if not record:
        return None
    matrix = record.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        return None
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted human", "Predicted AI"],
            y=["Actual human", "Actual AI"],
            text=[[str(tn), str(fp)], [str(fn), str(tp)]],
            texttemplate="%{text}",
            colorscale=[[0, "#e5f2f1"], [1, "#0f766e"]],
            showscale=False,
            hovertemplate="Actual: %{y}<br>%{x}<br>Count: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        template=template,
        height=340,
        margin=dict(l=12, r=12, t=24, b=12),
        xaxis_title="",
        yaxis_title="",
    )
    return figure


def performance_metric_records(
    records: list[dict[str, Any]],
    model_ids: tuple[str, ...],
    dataset_type: str,
    result_category: str,
) -> dict[str, dict[str, Any] | None]:
    return {
        model_id: latest_result(records, model_id, dataset_type, result_category)
        for model_id in model_ids
    }
