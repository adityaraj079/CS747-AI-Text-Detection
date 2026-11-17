"""Compact explanations of the measurements in dashboard tables."""
import plotly.graph_objects as go
import streamlit as st


def show_bars(title, names, series, *, percent=False, threshold=None):
    """Draw labeled horizontal bars, keeping the table's detector order."""
    if not names:
        return
    fig = go.Figure()
    for (label, values), color in zip(series.items(), ("#0f766e", "#a74c24")):
        fig.add_bar(name=label, y=names, x=values, orientation="h", marker_color=color,
                    texttemplate="%{x:.1%}" if percent else "%{x:,.2f}" if "ms" in label else "%{x:,d}",
                    textposition="outside", cliponaxis=False,
                    hovertemplate="%{y} · " + label + " " + ("%{x:.2%}" if percent else "%{x:,}") + "<extra></extra>")
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dot", annotation_text=f"{threshold:.0%} cutoff")
    fig.update_layout(
        barmode="group", height=105 + len(names) * (50 if len(series) > 1 else 32),
        margin=dict(l=4, r=64, t=25, b=8), showlegend=len(series) > 1,
        legend=dict(orientation="h", yanchor="bottom", y=1),
        yaxis=dict(autorange="reversed", type="category", categoryorder="array", categoryarray=names),
        xaxis=dict(rangemode="tozero", tickformat=".0%" if percent else None,
                   range=[0, 1] if threshold is not None else None),
    )
    st.markdown(f"**{title}**")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def show_test_graph(records, specs, original):
    names = [specs[r["model_id"]]["short_name"] for r in records]
    if original:
        show_bars("How many essays contributed to each original score?", names,
                  {"Test essays": [r["record_count"] for r in records]})
        return
    matrices = [r.get("confusion_matrix") for r in records]
    if all(m and sum(m[0]) and sum(m[1]) for m in matrices):
        show_bars("How often did each detector make each kind of mistake?", names, {
            "Human essays flagged": [m[0][1] / sum(m[0]) for m in matrices],
            "AI essays missed": [m[1][0] / sum(m[1]) for m in matrices],
        }, percent=True)
        st.caption("Each rate uses its own group as the denominator. Shorter bars mean fewer errors.")


def show_architecture(model_id):
    if model_id == "daniel_bert":
        pipeline = '"Essay" -> "WordPiece tokens" -> "BERT-base encoder" -> "Two-class head" -> "AI probability"'
    else:
        classifiers = {
            "aakash_7th_place": '"Multinomial NB" "SGD"',
            "aditya_3rd_place": '"SGD" "CatBoost"',
            "yash_5th_place": '"3 × Complement NB" "3 × SGD" "LightGBM"',
        }[model_id]
        pipeline = ('"Essay" -> "BPE tokens" -> "TF-IDF"; '
                    f'"TF-IDF" -> {{ {classifiers} }} -> "Weighted blend" -> "AI probability"')
    st.graphviz_chart('digraph { rankdir=LR; graph [bgcolor="transparent"]; '
                      'node [shape=box style="rounded,filled" fillcolor="#edf7f6" '
                      'color="#8ebdb7" fontcolor="#183d38" fontname="Helvetica" fontsize=12]; '
                      'edge [color="#718984" arrowsize=0.65]; ' + pipeline + ' }')
