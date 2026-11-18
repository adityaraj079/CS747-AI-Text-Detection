from __future__ import annotations

from time import perf_counter
from typing import Any

from evaluation.model_adapters import ModelAdapter, require_probability_vector


DISPLAY_THRESHOLD = 0.5


def warm_predict(model: ModelAdapter, text: str) -> tuple[float, float]:
    """Time and validate one CPU prediction from an already-loaded adapter."""
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
    st.caption(f"Each detector reports its estimated AI probability. The decision threshold is {DISPLAY_THRESHOLD:.2f}.")
    text = st.text_area(
        "Essay text",
        key=f"live-text-{section_title}",
        height=160,
        placeholder="Paste text to compare available registered models.",
    )
    analyze = st.button("Analyze Essay", key=f"analyze-{section_title}", type="primary")
    result_key = f"predictions-{section_title}"
    if analyze and not text.strip():
        st.warning("Enter non-empty essay text.")
        return

    from app_components.table_graphs import show_bars

    def plot_scores(scores):
        show_bars("Where did each detector score this essay?", list(scores),
                  {"AI probability": list(scores.values())}, percent=True, threshold=DISPLAY_THRESHOLD)

        if scores:
            flagged = sum(score >= DISPLAY_THRESHOLD for score in scores.values())
            st.caption(f"{flagged} of {len(scores)} responding detectors classified this essay as AI-generated.")

    previous = st.session_state.get(result_key)
    if not analyze:
        if previous and previous["text"] == text:
            st.table(previous["rows"])
            plot_scores(previous.get("scores", {}))
            for note in previous.get("notes", []):
                st.caption(note)
        return

    scores = {}
    rows = []
    notes = []
    table = st.empty()
    for model_id in model_ids:
        name = model_specs[model_id]["short_name"]
        with st.spinner(f"Scoring with {name}…"):
            try:
                adapter = adapter_loader(model_id)
                probability, latency_ms = warm_predict(adapter, text)
                if getattr(adapter, "runtime_note", ""):
                    notes.append(f"{name}: {adapter.runtime_note} Saved benchmark scores describe the original ensemble.")
                scores[name] = probability
                row = {"Detector": name, "AI probability": f"{probability:.2%}",
                       "Classification": threshold_decision(probability), "CPU time": f"{latency_ms:.2f} ms"}
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                row = {"Detector": name, "AI probability": "Unavailable",
                       "Classification": str(exc), "CPU time": "—"}
        rows.append(row)
        table.table(rows)
    plot_scores(scores)
    for note in notes:
        st.caption(note)
    st.session_state[result_key] = {"text": text, "rows": rows, "scores": scores, "notes": notes}
