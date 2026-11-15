from __future__ import annotations

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
        [data-testid="stAppViewContainer"] { color: var(--text-color); }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(120, 132, 150, 0.18); }
        .dashboard-eyebrow {
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .dashboard-subtitle {
            color: var(--text-color);
            opacity: 0.76;
            font-size: 1.05rem;
            margin-top: -0.5rem;
            margin-bottom: 1.5rem;
        }
        .status-pill {
            display: inline-block;
            border: 1px solid rgba(15, 118, 110, 0.35);
            border-radius: 999px;
            color: #0f766e;
            font-size: 0.76rem;
            font-weight: 650;
            padding: 0.22rem 0.65rem;
            margin: 0.15rem 0;
        }
        [data-testid="stMetric"] {
            background: var(--secondary-background-color);
            border: 1px solid rgba(120, 132, 150, 0.18);
            border-radius: 0.75rem;
            padding: 0.75rem 0.85rem;
        }
        [data-testid="stMetricLabel"] p { font-size: 0.82rem; }
        div[data-testid="stDataFrame"] { border: 1px solid rgba(120, 132, 150, 0.18); border-radius: 0.65rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_template() -> str:
    try:
        return "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
    except Exception:
        return "plotly_white"
