"""
Reusable layout primitives: headers, cards, metric rows, status emphasis.
"""

from __future__ import annotations

import streamlit as st


def page_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="pc-hero">
          <h1 style="margin:0; font-size: 2rem;">{title}</h1>
          <p style="margin:0.4rem 0 0 0; font-size:1.05rem; color: rgba(49,51,63,0.72);">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, helper: str | None = None) -> None:
    st.markdown(f"### {title}")
    if helper:
        st.caption(helper)


def open_card():
    """Context manager pattern without import contextlib — use with st.container."""
    return st.container()


def render_metric_cell(label: str, value: str, column) -> None:
    with column:
        st.markdown(f'<div class="pc-metric-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pc-metric-value">{value}</div>', unsafe_allow_html=True)


def dosage_status_html(status: str) -> str:
    s = status.strip()
    if s == "Correct dosage":
        cls = "pc-status-ok"
    elif s == "Too few":
        cls = "pc-status-low"
    elif s == "Too many":
        cls = "pc-status-high"
    else:
        cls = "pc-status-pill"
    return f'<span class="pc-status-pill {cls}">{s}</span>'


def render_footer_note() -> None:
    st.markdown(
        """
        <div class="pc-footnote">
          Current version uses UI mock outputs. Model inference will be connected in the next phase.
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str, *, icon: str = "ℹ️") -> None:
    st.info(f"{icon} {message}")
