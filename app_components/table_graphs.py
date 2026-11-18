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


def show_test_graph(records, specs, original, unit="essays"):
    names = [specs[r["model_id"]]["short_name"] for r in records]
    if original:
        show_bars("How many essays contributed to each original score?", names,
                  {"Test essays": [r["record_count"] for r in records]})
        counts = [r["record_count"] for r in records]
        st.markdown(f"**Finding.** The original runs tested between {min(counts):,} and {max(counts):,} essays "
                    "on different splits. Differences in accuracy can reflect which essays each detector saw.")
        return
    matrices = [r.get("confusion_matrix") for r in records]
    if all(m and sum(m[0]) and sum(m[1]) for m in matrices):
        show_bars("How often did each detector make each kind of mistake?", names, {
            f"Human {unit} flagged": [m[0][1] / sum(m[0]) for m in matrices],
            f"AI {unit} missed": [m[1][0] / sum(m[1]) for m in matrices],
        }, percent=True)
        human_errors = [m[0][1] for m in matrices]
        ai_errors = [m[1][0] for m in matrices]
        fewest_flags = min(human_errors)
        fewest_misses = min(ai_errors)
        def tied_names(errors, best):
            tied = [n for n, e in zip(names, errors) if e == best]
            return ", ".join(tied[:-1]) + " and " + tied[-1] if len(tied) > 1 else tied[0]

        human_names = tied_names(human_errors, fewest_flags)
        ai_names = tied_names(ai_errors, fewest_misses)
        human_total, ai_total = map(sum, matrices[0])
        if human_names == ai_names:
            finding = (f"{human_names} made the fewest errors in both groups, flagging "
                       f"{fewest_flags:,} of {human_total:,} human {unit} and missing "
                       f"{fewest_misses:,} of {ai_total:,} AI {unit}.")
        else:
            finding = (f"{human_names} flagged the fewest human {unit} ({fewest_flags:,} of {human_total:,}). "
                       f"{ai_names} caught the most AI {unit}, missing {fewest_misses:,} of {ai_total:,}.")
        if min(human_errors) > human_total / 2:
            finding = (f"Every detector flagged more than half the human {unit} "
                       f"({min(human_errors) / human_total:.1%}–{max(human_errors) / human_total:.1%}). "
                       f"{human_names} made the fewest false alarms at this threshold.")
        st.markdown(f"**Finding.** {finding}")


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
