"""
Session state initialization for Pill Counter app.
Centralizes keys so the team can extend state safely.
"""

from __future__ import annotations

import streamlit as st

from utils.config import (
    BACKEND_MOCK,
    DEFAULT_API_BASE_URL,
    DEFAULT_API_ENDPOINT,
    DEFAULT_API_TIMEOUT_S,
    DEFAULT_MAX_REGION_AREA,
    DEFAULT_MIN_REGION_AREA,
    DEFAULT_MORPH_KERNEL,
    DEFAULT_RESIZE_H,
    DEFAULT_RESIZE_W,
    DEFAULT_THRESHOLD,
    PAGE_HOME,
)


def init_session_state() -> None:
    """Initialize default session_state keys if missing."""
    defaults: dict[str, object] = {
        "nav_page": PAGE_HOME,
        "backend_mode": BACKEND_MOCK,
        "api_base_url": DEFAULT_API_BASE_URL,
        "api_endpoint": DEFAULT_API_ENDPOINT,
        "api_timeout_s": DEFAULT_API_TIMEOUT_S,
        "api_last_error": None,
        "local_model_path": "",
        "local_preprocessor_path": "",
        "local_label_config_path": "",
        "input_source": "Upload Image",
        "expected_dosage": 1,
        "resize_w": DEFAULT_RESIZE_W,
        "resize_h": DEFAULT_RESIZE_H,
        "threshold": DEFAULT_THRESHOLD,
        "min_region_area": DEFAULT_MIN_REGION_AREA,
        "max_region_area": DEFAULT_MAX_REGION_AREA,
        "morph_kernel": DEFAULT_MORPH_KERNEL,
        "sample_folder_override": "",
        "working_image": None,  # PIL.Image or None
        "working_image_source_label": "",
        "pipeline_result": None,  # dict from inference layer
        "analysis_error": None,
        "last_analysis_ok": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
