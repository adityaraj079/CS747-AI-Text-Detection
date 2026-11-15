from __future__ import annotations

from typing import Any

import streamlit as st

from app_components.data_loader import format_metric


def render_model_card(
    spec: dict[str, Any],
    result: dict[str, Any] | None,
    availability: tuple[bool, str],
    *,
    compact: bool = False,
) -> None:
    available, detail = availability
    status = "Adapter registered · artifacts detected" if available else spec.get("availability", "Not integrated")
    with st.container(border=True):
        st.markdown(f"### {spec['name']}")
        st.caption(spec.get("member", ""))
        st.markdown(f'<span class="status-pill">{status}</span>', unsafe_allow_html=True)
        st.caption(detail or spec.get("status_detail", ""))
        if spec.get("reference"):
            st.markdown(f"[{spec.get('reference_label', 'Reference')}]({spec['reference']})")
        if not compact and spec.get("architecture"):
            st.write(spec["architecture"])
        if spec.get("competition_track"):
            st.caption(f"Track: {spec['competition_track']}")
        if spec.get("original_leaderboard_roc_auc") is not None:
            st.caption(
                "Historical private leaderboard ROC-AUC: "
                f"{format_metric(spec['original_leaderboard_roc_auc'])} "
                "(not a local benchmark)"
            )
        if not result:
            st.info("Local evaluation: Not evaluated")
            return

        metric_columns = st.columns(4)
        if compact:
            metrics = [
                ("ROC-AUC", "roc_auc"),
                ("Accuracy", "accuracy"),
                ("F1", "f1"),
                ("CPU ms / essay", "latency_ms_per_essay"),
            ]
        else:
            metrics = [
                ("ROC-AUC", "roc_auc"),
                ("F1", "f1"),
                ("CPU ms / essay", "latency_ms_per_essay"),
                ("Model size (MB)", "model_size_mb"),
            ]
        for column, (label, metric) in zip(metric_columns, metrics):
            column.metric(label, format_metric(result.get(metric), digits=3 if metric not in {"latency_ms_per_essay", "model_size_mb"} else 2))
        if compact:
            st.caption(f"Evaluation set: {result.get('dataset_type') or 'Not recorded'}")
