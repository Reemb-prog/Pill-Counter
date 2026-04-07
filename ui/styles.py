"""
Optional custom CSS for an academically professional, calm dashboard look.
"""

from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    """Inject global styles once per run."""
    css = """
    <style>
      /* Card-like containers */
      div[data-testid="stVerticalBlock"] > div:has(>.stMarkdown) div.stMarkdown p {
        margin-bottom: 0.25rem;
      }
      .pc-hero {
        padding: 1rem 0 0.5rem 0;
        border-bottom: 1px solid rgba(49, 51, 63, 0.12);
        margin-bottom: 1rem;
      }
      .pc-card {
        background: var(--secondary-background-color, #f8f9fb);
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
      }
      .pc-metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: rgba(49, 51, 63, 0.65);
        margin-bottom: 0.15rem;
      }
      .pc-metric-value {
        font-size: 1.65rem;
        font-weight: 600;
        line-height: 1.2;
      }
      .pc-status-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
      }
      .pc-status-ok { background: #e8f5e9; color: #1b5e20; }
      .pc-status-low { background: #fff3e0; color: #e65100; }
      .pc-status-high { background: #ffebee; color: #b71c1c; }
      .pc-footnote {
        font-size: 0.8rem;
        color: rgba(49, 51, 63, 0.6);
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(49, 51, 63, 0.1);
      }
      .pc-sidebar-note {
        font-size: 0.82rem;
        color: rgba(49, 51, 63, 0.75);
      }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
