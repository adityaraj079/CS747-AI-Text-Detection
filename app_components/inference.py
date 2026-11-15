from __future__ import annotations

from time import perf_counter
from typing import Any

from evaluation.model_adapters import ModelAdapter, require_probability_vector


DISPLAY_THRESHOLD = 0.5


def warm_predict(model: ModelAdapter, text: str) -> tuple[float, float]:
    """Warm the loaded adapter, then time one validated CPU prediction."""
    model.predict_proba([text])
    start = perf_counter()
    probabilities = require_probability_vector(model.predict_proba([text]), 1)
    elapsed_ms = (perf_counter() - start) * 1000.0
    return float(probabilities[0]), elapsed_ms


def threshold_decision(probability: float, threshold: float = DISPLAY_THRESHOLD) -> str:
    return "AI-generated" if probability >= threshold else "Not AI-generated"


def show_live_predictions(
    model_ids: tuple[str, ...],
    model_specs: dict[str, dict[str, Any]],
    adapter_loader,
    *,
    section_title: str = "Live essay comparison",
) -> None:
    import streamlit as st

    st.subheader(section_title)
    st.caption(
        "Predictions are model probabilities, not ground-truth determinations. "
        "The classification uses a provisional 0.50 threshold; official evaluation should use "
        "a threshold selected from validation data."
    )
    text = st.text_area(
        "Essay text",
        key=f"live-text-{section_title}",
        height=160,
        placeholder="Paste text to compare available registered models.",
    )
    if not st.button("Analyze Essay", key=f"analyze-{section_title}", type="primary"):
        return
    if not text.strip():
        st.warning("Enter non-empty essay text.")
        return

    for model_id in model_ids:
        spec = model_specs[model_id]
        try:
            adapter = adapter_loader(model_id)
            probability, latency_ms = warm_predict(adapter, text)
        except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
            st.info(f"**{spec['short_name']} unavailable:** {exc}")
            continue
        columns = st.columns([2, 1, 1, 1])
        columns[0].markdown(f"**{spec['name']}**")
        columns[1].metric("P(AI-generated)", f"{probability:.3f}")
        columns[2].metric(
            "Classification",
            threshold_decision(probability),
            help=f"AI-generated when P(AI-generated) >= {DISPLAY_THRESHOLD:.2f}; otherwise Not AI-generated.",
        )
        columns[3].metric(
            "Detection time",
            f"{latency_ms:.2f} ms",
            help=(
                "Time this model took to produce this prediction, measured after an untimed "
                "warm-up call. Excludes model loading and warm-up; includes prediction and "
                "output validation."
            ),
        )
