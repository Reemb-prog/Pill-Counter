"""
Pill Counter for Medication Management — Streamlit entry point.

Run: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from services.api_client import ApiClientConfig, PillCountApiClient
from ui.components import (
    render_evaluation_dashboard,
    render_input_panel,
    render_pipeline_explorer,
    render_results_dashboard,
    render_run_analysis_button,
    render_settings_page,
)
from ui.layout import empty_state, page_hero, render_footer_note, section_header
from ui.styles import inject_custom_css
from utils.config import (
    BACKEND_API,
    BACKEND_LOCAL,
    BACKEND_OPTIONS,
    NAV_PAGES,
    PAGE_ANALYSIS,
    PAGE_EVALUATION,
    PAGE_HOME,
    PAGE_PIPELINE,
    PAGE_SETTINGS,
)
from utils.image_utils import ensure_demo_sample_images
from utils.state import init_session_state


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Section",
            list(NAV_PAGES),
            key="nav_page",
            label_visibility="collapsed",
        )
        st.divider()

        st.markdown("### Backend mode")
        st.selectbox(
            "Inference backend",
            list(BACKEND_OPTIONS),
            key="backend_mode",
            help="Mock uses built-in placeholders. API/Local require wiring in services/.",
        )

        mode = st.session_state.get("backend_mode")
        st.markdown(
            '<p class="pc-sidebar-note">Real remote or local inference is not connected yet — '
            "Mock Mode is recommended for demos.</p>",
            unsafe_allow_html=True,
        )

        if mode == BACKEND_API:
            st.markdown("**API connection**")
            st.text_input("API base URL", key="api_base_url")
            st.text_input("Endpoint path", key="api_endpoint")
            st.number_input("Timeout (seconds)", min_value=1.0, key="api_timeout_s")
            if st.button("Test connection", use_container_width=True):
                cfg = ApiClientConfig(
                    base_url=st.session_state["api_base_url"],
                    endpoint=st.session_state["api_endpoint"],
                    timeout_s=float(st.session_state["api_timeout_s"]),
                )
                ok, msg = PillCountApiClient(cfg).test_connection()
                if ok:
                    st.success(msg)
                else:
                    st.warning(
                        f"{msg}\n\n"
                        "Until you deploy `/health`, this check may fail even when your API is valid — "
                        "adjust `test_connection()` in `services/api_client.py` for your stack."
                    )

        if mode == BACKEND_LOCAL:
            st.markdown("**Local artifacts (placeholders)**")
            st.text_input("Model path (.pkl / .joblib / .onnx / .pt)", key="local_model_path")
            st.text_input("Preprocessor path (optional)", key="local_preprocessor_path")
            st.text_input("Label / config path (optional)", key="local_label_config_path")

        st.divider()
        with st.expander("About"):
            st.markdown(
                """
**Pill Counter for Medication Management** supports image-based pill counting and
dosage verification against a user-entered expected dose.

This release focuses on **UI and architecture**. Connect your Colab-exported model
via REST or local serialization in the next phase.
                """
            )


def _page_home() -> None:
    section_header("Welcome", "Navigate using the sidebar to explore analysis, pipeline stages, and evaluation.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**How it works (planned pipeline)**  
1. Image input  
2. Preprocessing  
3. Segmentation  
4. Connected components / detection  
5. Feature extraction  
6. Counting + dosage decision  
7. Reported output  
            """
        )
    with c2:
        st.markdown(
            """
**Getting started**  
1. Open **Analysis Dashboard**  
2. Upload an image or load a random sample  
3. Enter the expected dosage  
4. Click **Run Analysis**  
            """
        )
    ensure_demo_sample_images()


def _page_analysis() -> None:
    render_input_panel()
    render_run_analysis_button()
    render_results_dashboard()


def main() -> None:
    st.set_page_config(
        page_title="Pill Counter — Medication Management",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    init_session_state()
    _render_sidebar()

    page = st.session_state.get("nav_page", PAGE_HOME)

    page_hero(
        "Pill Counter for Medication Management",
        "Image-based pill counting and dosage verification",
    )

    if page == PAGE_HOME:
        _page_home()
    elif page == PAGE_ANALYSIS:
        _page_analysis()
    elif page == PAGE_PIPELINE:
        render_pipeline_explorer()
    elif page == PAGE_EVALUATION:
        render_evaluation_dashboard()
    elif page == PAGE_SETTINGS:
        render_settings_page()
    else:
        empty_state("Unknown section.")

    render_footer_note()


if __name__ == "__main__":
    main()
