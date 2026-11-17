from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from app_components.data_loader import load_dataset_manifests, load_result_records
from app_components.model_catalog import load_model_catalog
from app_components.model_registry import load_model_adapter
from app_components.styling import apply_styles
from app_pages import champion_comparison, team_comparison


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("CS747_RESULTS_DIR", ROOT / "results" / "metrics")).expanduser()
PAGE_OPTIONS = (
    "Team Model Comparison",
    "Kaggle reference",
)


@st.cache_data(show_spinner=False)
def _cached_results(metrics_dir: str) -> list[dict]:
    return load_result_records(metrics_dir)


@st.cache_data(show_spinner=False)
def _cached_manifests(data_dir: str):
    return load_dataset_manifests(data_dir)


@st.cache_resource(show_spinner="Loading registered model...")
def _cached_adapter(model_id: str):
    if model_id == "daniel_bert":
        # Single-essay inference shares this process with LightGBM's OpenMP runtime.
        import torch
        torch.set_num_threads(1)
    return load_model_adapter(model_id)


def main() -> None:
    st.set_page_config(
        page_title="CS747 · AI Text Detection",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    with st.sidebar:
        st.markdown("## CS747 Research")
        st.caption("AI-generated essay detection")
        page = st.radio(
            "Research views",
            PAGE_OPTIONS,
            label_visibility="collapsed",
        )
    try:
        results = _cached_results(str(RESULTS_DIR))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not load saved metric results: {exc}")
        results = []
    try:
        split_manifest, unseen_manifest = _cached_manifests(str(ROOT / "data"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not load dataset manifests: {exc}")
        split_manifest, unseen_manifest = None, None

    model_specs = load_model_catalog(ROOT)
    if page == "Kaggle reference":
        champion_comparison.render_page(
            ROOT,
            results,
            model_specs,
            split_manifest,
            unseen_manifest,
            _cached_adapter,
        )
    else:
        team_comparison.render_page(
            ROOT,
            results,
            model_specs,
            split_manifest,
            unseen_manifest,
            _cached_adapter,
        )


if __name__ == "__main__":
    main()
