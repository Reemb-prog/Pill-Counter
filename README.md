# Pill Counter for Medication Management

Streamlit dashboard for **image-based pill counting** and **dosage verification** (expected dose vs detected count). This phase provides a **production-style UI and architecture** with **mock pipeline outputs**. Real model inference can be connected in a later phase via a **deployed API** (for example, FastAPI exported from Colab) or **local serialized artifacts** (`joblib`, `pickle`, ONNX, PyTorch, TensorFlow, etc.).

---

## File structure and responsibilities

| File / folder | Responsibility |
|---------------|----------------|
| **`app.py`** | Streamlit entry: `set_page_config` (wide layout), CSS injection, `init_session_state`, **sidebar** (nav, backend mode, API/local fields, test connection, about), routes **Home / Analysis / Pipeline / Evaluation / Settings**, footer note. |
| **`ui/styles.py`** | `inject_custom_css()` — calm dashboard styling (metric labels, status pills, footnote). |
| **`ui/layout.py`** | `page_hero`, section headers, metric cells, **dosage status** HTML (green / amber / red), footer, empty-state helper. |
| **`ui/components.py`** | **Input panel** (upload vs random sample, preview, expected dosage, advanced expander), **Run Analysis** → `run_analysis()`, **results** metrics, **pipeline tabs** + feature `st.dataframe`, **evaluation** dashboards from JSON, **settings** summary. |
| **`services/mock_pipeline.py`** | Mock **preprocessing / segmentation-style** visuals (PIL), deterministic-ish count from image hash, features table, API-shaped dict + `_pil` for fast UI. |
| **`services/api_client.py`** | `PillCountApiClient`: documented **request/response contract**, `test_connection()` (GET `/health` + base URL fallback), `post_predict()` **NotImplemented** with TODO for Colab/FastAPI. |
| **`services/inference.py`** | **Adapter**: `MockRunner`, `ApiRunner`, `LocalArtifactRunner`; `get_runner()`, `run_analysis()`; Base64 → `_pil` helper for future API. |
| **`utils/config.py`** | Paths, backend labels, defaults, nav page ids, allowed extensions. |
| **`utils/state.py`** | Central `st.session_state` defaults (aligned with widget keys). |
| **`utils/image_utils.py`** | Upload load, type check, random sample, **`ensure_demo_sample_images()`** if folder empty. |
| **`data/sample_metrics.json`** | Mock **accuracy, MAE, fold curves, confusion-style matrix, comparison table, failure notes**. |
| **`assets/sample_images/`** | Populated at runtime with synthetic PNGs if empty. |
| **`assets/icons/`** | Placeholder for optional icon assets. |
| **`requirements.txt`** | `streamlit`, `pandas`, `numpy`, `Pillow`, `requests`. |

---

## What to do next

### Run the app locally

From the **project root** (the folder that contains `app.py`):

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

   On Windows PowerShell, if your path contains spaces, quote it:

   ```powershell
   cd "c:\path\to\streamlit pill counter"
   streamlit run app.py
   ```

### Connect a real backend (later phase)

- **API:** Implement `PillCountApiClient.post_predict()` and tune `test_connection()` in `services/api_client.py`. The response should match the JSON contract documented in that module’s docstring. Select **API Mode** in the sidebar and set base URL, endpoint, and timeout.
- **Local artifact:** Implement `LocalArtifactRunner.run()` in `services/inference.py` and return the same structure (or Base64 `images` like the API). Choose **Local Artifact Mode** and set model / preprocessor / label paths in the sidebar.

The footer and sidebar note that **inference is mock/placeholder** until one of these is wired.

### Sample images and evaluation data

- Place PNG or JPEG files under **`assets/sample_images/`**. If the folder is empty on first run, the app creates a few synthetic pill-like images so **Use Random Dataset Image** works immediately.
- Evaluation charts read **`data/sample_metrics.json`**. Restore or edit that file if charts are missing.

---

## Code references

Below are the main wiring points: **app entry + sidebar** and **Run Analysis** calling the unified inference layer.

### `app.py` — entry point, imports, sidebar (navigation and backend)

```python
# app.py (excerpt: lines 1–100)

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
```

*See the full `main()` and page routing in the same file (`app.py`).*

### `ui/components.py` — Run Analysis → `run_analysis()`

```python
# ui/components.py (excerpt: render_run_analysis_button)

def render_run_analysis_button() -> None:
    """Primary action with spinner."""
    err = _validate_before_run()
    if err:
        st.warning(err)
    clicked = st.button("Run Analysis", type="primary", use_container_width=True, disabled=bool(err))
    if not clicked:
        return

    img: Image.Image = st.session_state["working_image"]
    settings = {
        "threshold": st.session_state["threshold"],
        "resize_w": st.session_state["resize_w"] or None,
        "resize_h": st.session_state["resize_h"] or None,
        "min_region_area": st.session_state["min_region_area"],
        "max_region_area": st.session_state["max_region_area"],
        "morph_kernel": st.session_state["morph_kernel"],
    }
    expected = int(st.session_state["expected_dosage"])
    mode = st.session_state["backend_mode"]

    with st.spinner("Running image-processing pipeline (mock or configured backend)…"):
        result, infer_err = run_analysis(
            mode,
            img,
            expected,
            settings,
            api_base_url=st.session_state["api_base_url"],
            api_endpoint=st.session_state["api_endpoint"],
            api_timeout_s=float(st.session_state["api_timeout_s"]),
            local_model_path=st.session_state["local_model_path"],
            local_preprocessor_path=st.session_state["local_preprocessor_path"],
            local_label_config_path=st.session_state["local_label_config_path"],
        )

    if infer_err:
        st.session_state["pipeline_result"] = None
        st.session_state["last_analysis_ok"] = False
        st.session_state["analysis_error"] = infer_err
        st.error(infer_err)
        return

    st.session_state["pipeline_result"] = result
    st.session_state["last_analysis_ok"] = True
    st.session_state["analysis_error"] = None
    st.success("Analysis finished. Review the summary and pipeline tabs below.")
```

*The implementation of `run_analysis()` lives in **`services/inference.py`**.*

---

## Academic / demo notes

- **Mock Mode** is deterministic for a given image: the same upload yields the same demo count.
- The footer states that **inference is mocked** until the backend is wired.
- Naming follows **image-processing** stages (grayscale, denoising, segmentation, components, features).

## License / usage

Use for coursework or research demos; validate clinically relevant use with appropriate oversight and testing before any real medication workflow.
