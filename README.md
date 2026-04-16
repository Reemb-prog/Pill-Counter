# Pill Counter for Medication Management

Streamlit dashboard for **image-based pill counting** and **dosage verification** (expected dose vs detected count). The current app supports:
- `Mock Mode` uses synthetic outputs.
- `Local Artifact Mode` runs real predictions using the trained YOLO weights (`best.pt`).
- `API Mode` remains the future option for a deployed backend.

---

## File structure and responsibilities

| File / folder | Responsibility |
|---------------|----------------|
| **`app.py`** | Streamlit entry: `set_page_config` (wide layout), CSS injection, `init_session_state`, **sidebar** (nav, backend mode, API/local fields, test connection, about), routes **Home / Analysis / Pipeline / Evaluation / Settings**, footer note. |
| **`ui/styles.py`** | `inject_custom_css()` — calm dashboard styling (metric labels, status pills, footnote). |
| **`ui/layout.py`** | `page_hero`, section headers, metric cells, **dosage status** HTML (green / amber / red), footer, empty-state helper. |
| **`ui/components.py`** | **Input panel** (upload vs random sample, preview, expected dosage, advanced expander), **Run Analysis** → `run_analysis()`, **results** metrics, **pipeline tabs** + feature `st.dataframe`, **evaluation** dashboards from JSON, **settings** summary. |
| **`services/mock_pipeline.py`** | Sample-output inference path for Mock Mode, including display-ready stage images and a feature table. |
| **`services/api_client.py`** | `PillCountApiClient`: request/response contract for optional API integration, `test_connection()` (GET `/health` + base URL fallback), `post_predict()` placeholder for an external service. |
| **`services/inference.py`** | **Adapter**: `MockRunner`, `ApiRunner`, `LocalArtifactRunner`; `get_runner()`, `run_analysis()`; Base64 → `_pil` helper for API responses. |
| **`utils/config.py`** | Paths, backend labels, defaults, nav page ids, allowed extensions. |
| **`utils/state.py`** | Central `st.session_state` defaults (aligned with widget keys). |
| **`utils/image_utils.py`** | Upload load, type check, random sample loading, and starter sample-image creation if no sample images exist yet. |
| **`data/sample_metrics.json`** | Bundled evaluation summary data used by the Evaluation page. |
| **`assets/sample_images/`** | Created and populated on first run if missing, so random-sample mode works immediately. |
| **`requirements.txt`** | App + notebook dependencies (`streamlit`, `opencv-python-headless`, `jupyter`, …). |
| **`services/yolo_trained_pipeline.py`** | Trained YOLO inference adapter: loads `best.pt`, runs detection, computes pill count + dosage status, and returns UI-shaped outputs. |
| **`pill couter and dosage decision full pipeline notebook.ipynb`** | Final training + inference workflow source (counting + dosage decision logic). |
| **`data/ground_truth_counts.csv`** | Optional per-filename pill counts for notebook evaluation (bundled for synthetic samples). |

### Run the Jupyter pipeline notebook

From the project root, install dependencies (includes Jupyter and OpenCV), then:

```bash
jupyter notebook "pill couter and dosage decision full pipeline notebook.ipynb"
```

Or use VS Code / Cursor “Run All”. The notebook contains the final YOLO inference + dosage decision logic and exports/uses `best.pt`.

---

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

### Backend modes

- **API:** Optional external-service integration. Implement `PillCountApiClient.post_predict()` and tune `test_connection()` in `services/api_client.py` if you want to call a deployed backend. Select **API Mode** in the sidebar and set base URL, endpoint, and timeout.
- **Local artifact:** Use **Local Artifact Mode** (already wired). Leave the model path blank to default to `best.pt` at the project root, or set a custom path in the sidebar.

Mock Mode provides sample outputs, while Local Artifact Mode uses YOLO with `best.pt`.

Expected outputs in the UI:
- detected pill count (`detected_count`)
- dosage decision/status (`Correct dosage`, `Too few`, `Too many`)
- pipeline explorer stage images and the per-detection feature table

### Sample images and evaluation data

- Place PNG or JPEG files under **`assets/sample_images/`**. If the folder is empty or missing on first run, the app creates a few starter sample images so **Use Random Dataset Image** works immediately.
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
            help="Local Artifact Mode runs trained YOLO, Mock Mode provides sample outputs, and API Mode connects to an external service when configured.",
        )

        mode = st.session_state.get("backend_mode")
        st.markdown(
            '<p class="pc-sidebar-note">Local Artifact Mode runs trained YOLO using `best.pt`. '
            "Mock Mode is available for sample outputs, and API Mode can be used when an external service is configured.</p>",
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
            st.markdown("**Local artifacts (YOLO)**")
            st.text_input("Model path (.pt, optional; blank = best.pt)", key="local_model_path")
            st.text_input("Preprocessor path (optional)", key="local_preprocessor_path")
            st.text_input("Label / config path (optional)", key="local_label_config_path")

        st.divider()
        with st.expander("About"):
            st.markdown(
                """
**Pill Counter for Medication Management** supports image-based pill counting and
dosage verification against a user-entered expected dose.

Use **Local Artifact Mode** for built-in YOLO inference with `best.pt`, or connect an
external service through **API Mode** when needed.
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

    with st.spinner("Running analysis…"):
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

## Notes

- **Mock Mode** is deterministic for a given image: the same upload yields the same sample count.
- Naming in the explorer follows high-level image-analysis stages (grayscale, denoising, segmentation view, detection overlay, features).

## License / usage

Use for coursework or research demos; validate clinically relevant use with appropriate oversight and testing before any real medication workflow.
