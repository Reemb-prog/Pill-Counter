"""
Streamlit UI sections: input panel, results, pipeline tabs, evaluation charts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from services.inference import run_analysis
from ui.layout import (
    dosage_status_html,
    empty_state,
    render_metric_cell,
    section_header,
)
from utils.config import (
    BACKEND_API,
    BACKEND_LOCAL,
    EVALUATION_DATA_JSON,
    SAMPLE_IMAGES_DIR,
)
from utils.image_utils import (
    ensure_demo_sample_images,
    is_allowed_image_file,
    load_image_from_upload,
    load_random_sample,
    session_show_image,
)


def render_input_panel() -> None:
    """Main input card: source selection, preview, dosage, advanced settings."""
    ensure_demo_sample_images()
    section_header(
        "Input",
        "Upload a pill image or load a random sample. Enter the expected dosage from the prescription label.",
    )
    source = st.radio(
        "Image source",
        ["Upload Image", "Use Random Dataset Image"],
        horizontal=True,
        key="input_source",
    )

    if source == "Upload Image":
        up = st.file_uploader("Image file (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        if up is not None:
            if not is_allowed_image_file(up.name):
                st.error("Please upload a PNG or JPEG image.")
            else:
                try:
                    img = load_image_from_upload(up)
                    st.session_state["working_image"] = img
                    st.session_state["working_image_source_label"] = up.name
                    session_show_image(img, caption="Uploaded preview")
                except Exception as e:
                    st.session_state["working_image"] = None
                    st.error(f"We could not read that image. Try another file. ({e})")
        else:
            st.session_state["working_image"] = None
            st.session_state["working_image_source_label"] = ""
            empty_state("No file selected yet — choose a PNG or JPEG above.")
    else:
        st.text_input(
            "Optional sample folder path (leave blank to use bundled samples)",
            key="sample_folder_override",
        )
        override = str(st.session_state.get("sample_folder_override") or "")
        folder = Path(override) if override.strip() else SAMPLE_IMAGES_DIR
        if st.button("Load Random Sample", type="primary", key="btn_random_sample"):
            img, path, err = load_random_sample(folder)
            if err:
                st.warning(err)
                st.session_state["working_image"] = None
            else:
                st.session_state["working_image"] = img
                st.session_state["working_image_source_label"] = path.name if path else "sample"

        if st.session_state.get("working_image") and source == "Use Random Dataset Image":
            session_show_image(
                st.session_state["working_image"],
                caption=f"Sample: {st.session_state.get('working_image_source_label', '')}",
            )
        elif source == "Use Random Dataset Image" and not st.session_state.get("working_image"):
            empty_state('Click "Load Random Sample" or switch to Upload Image.')

    st.number_input(
        "Expected dosage (tablets/capsules)",
        min_value=0,
        step=1,
        key="expected_dosage",
    )

    with st.expander("Advanced preprocessing (placeholders for future pipeline)"):
        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                "Resize width (px, 0 = keep original)",
                min_value=0,
                key="resize_w",
            )
        with c2:
            st.number_input(
                "Resize height (px, 0 = keep original)",
                min_value=0,
                key="resize_h",
            )
        st.slider(
            "Segmentation threshold (0–255)",
            min_value=0.0,
            max_value=255.0,
            key="threshold",
        )
        a1, a2 = st.columns(2)
        with a1:
            st.number_input(
                "Min region area (px²)",
                min_value=0,
                key="min_region_area",
            )
        with a2:
            st.number_input(
                "Max region area (px²)",
                min_value=0,
                key="max_region_area",
            )
        st.number_input(
            "Morphology kernel size (placeholder)",
            min_value=1,
            key="morph_kernel",
        )


def _validate_before_run() -> str | None:
    if st.session_state.get("working_image") is None:
        return "Please upload an image or load a random sample before running analysis."
    mode = st.session_state.get("backend_mode")
    if mode == BACKEND_API:
        url = (st.session_state.get("api_base_url") or "").strip()
        if not url:
            return "API Mode is selected but the base URL is empty. Add it in the sidebar or Settings page."
    if mode == BACKEND_LOCAL:
        mp = (st.session_state.get("local_model_path") or "").strip()
        if not mp:
            return "Local Artifact Mode needs a model file path (placeholder until your artifact is connected)."
    return None


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


def render_results_dashboard() -> None:
    """Summary metrics after a successful run."""
    section_header("Results", "Dosage verification outcome and run metadata.")
    res = st.session_state.get("pipeline_result")
    if not res:
        empty_state(
            "No results yet. Go to **Analysis Dashboard**, prepare an image, and click **Run Analysis**."
        )
        return

    c1, c2, c3 = st.columns(3)
    render_metric_cell("Detected count", str(res.get("detected_count", "—")), c1)
    render_metric_cell("Expected dosage", str(res.get("expected_dosage", "—")), c2)
    status = str(res.get("status", "—"))
    with c3:
        st.markdown('<div class="pc-metric-label">Dosage status</div>', unsafe_allow_html=True)
        st.markdown(dosage_status_html(status), unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    render_metric_cell("Processing time (s)", str(res.get("processing_time", "—")), c4)
    conf = res.get("metrics", {}) or {}
    render_metric_cell("Confidence (placeholder)", str(conf.get("confidence", "—")), c5)
    render_metric_cell("Backend mode", str(res.get("backend_mode", st.session_state.get("backend_mode"))), c6)


def _tab_stage_body(title: str, explanation: str, image: Image.Image | None, details: dict) -> None:
    st.markdown(f"**{title}**")
    st.caption(explanation)
    if image is not None:
        session_show_image(image)
    else:
        st.info("No image for this stage in the current response.")
    if details:
        with st.expander("Technical details (demo)"):
            st.json(details)


def render_pipeline_explorer() -> None:
    """Tabs for each intermediate stage + feature table."""
    section_header(
        "Pipeline explorer",
        "Intermediate outputs from preprocessing through connected-component labeling (mock data in this phase).",
    )
    res = st.session_state.get("pipeline_result")
    if not res:
        empty_state("Run an analysis first to populate pipeline stages.")
        return

    pil_map = res.get("_pil") or {}
    metrics = res.get("metrics") or {}

    tabs = st.tabs(
        [
            "Original",
            "Grayscale",
            "Denoised",
            "Segmented",
            "Detected components",
            "Feature summary",
        ]
    )
    with tabs[0]:
        _tab_stage_body(
            "Original",
            "Raw capture aligned with medication-management verification (cropped flat background recommended).",
            pil_map.get("original"),
            {"size": list(pil_map["original"].size) if pil_map.get("original") else {}},
        )
    with tabs[1]:
        _tab_stage_body(
            "Grayscale",
            "Luminance channel for illumination-normalized segmentation.",
            pil_map.get("grayscale"),
            {"representation": "single-channel displayed as RGB"},
        )
    with tabs[2]:
        _tab_stage_body(
            "Denoised",
            "Noise suppression prior to thresholding (median filter placeholder).",
            pil_map.get("denoised"),
            {"filter": "median 3×3 (demo)"},
        )
    with tabs[3]:
        _tab_stage_body(
            "Segmented",
            "Binary foreground mask / pseudo-color overlay for pill regions.",
            pil_map.get("segmented"),
            {"threshold": metrics.get("threshold_used")},
        )
    with tabs[4]:
        det = {
            "candidate_regions": metrics.get("candidate_regions"),
            "min_region_area": metrics.get("min_region_area"),
            "max_region_area": metrics.get("max_region_area"),
            "morph_kernel": metrics.get("morph_kernel"),
        }
        _tab_stage_body(
            "Detected components",
            "Bounding boxes from connected-component analysis (mock overlays).",
            pil_map.get("components_overlay"),
            det,
        )
    with tabs[5]:
        st.caption("Region-wise shape descriptors used to filter non-pill blobs.")
        feats = res.get("features") or []
        if feats:
            df = pd.DataFrame(feats)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No feature rows in this response.")


def load_evaluation_data() -> dict | None:
    path = Path(EVALUATION_DATA_JSON)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def render_evaluation_dashboard() -> None:
    """Charts and tables from bundled mock evaluation JSON."""
    section_header(
        "Evaluation",
        "Illustrative metrics and error analysis for academic review (synthetic data).",
    )
    data = load_evaluation_data()
    if not data:
        empty_state(
            f"Evaluation file missing at {EVALUATION_DATA_JSON}. Restore sample_metrics.json to see charts."
        )
        return

    summ = data.get("summary") or {}
    m1, m2, m3, m4 = st.columns(4)
    render_metric_cell("Counting accuracy", f'{summ.get("counting_accuracy", 0):.2f}', m1)
    render_metric_cell("MA counting error", f'{summ.get("mean_absolute_counting_error", 0):.2f}', m2)
    render_metric_cell("Dosage decision acc.", f'{summ.get("dosage_decision_accuracy", 0):.2f}', m3)
    render_metric_cell("Eval images (N)", str(summ.get("n_eval_images", "—")), m4)

    st.subheader("Accuracy by fold")
    fold_df = pd.DataFrame(data.get("accuracy_over_fold", []))
    if not fold_df.empty:
        st.line_chart(fold_df.set_index("fold")[["count_acc", "dosage_acc"]])
        st.bar_chart(fold_df.set_index("fold")[["count_acc", "dosage_acc"]])

    st.subheader("Dosage decision confusion (counts)")
    conf = data.get("confusion_dosage") or {}
    labels = conf.get("labels") or []
    matrix = conf.get("matrix") or []
    if labels and matrix:
        conf_df = pd.DataFrame(matrix, index=[f"true: {l}" for l in labels], columns=[f"pred: {l}" for l in labels])
        st.dataframe(conf_df, use_container_width=True)

    st.subheader("Per-image comparison (sample)")
    comp = pd.DataFrame(data.get("sample_comparisons", []))
    if not comp.empty:
        st.dataframe(comp, use_container_width=True, hide_index=True)

    notes = data.get("failure_cases_notes") or []
    if notes:
        st.subheader("Representative failure notes")
        for n in notes:
            st.markdown(f"- {n}")


def render_settings_page() -> None:
    """Dedicated page mirroring backend options with room for documentation."""
    section_header(
        "Settings / Backend mode",
        "Connect a Colab-exported API or a local serialized pipeline in the next phase.",
    )
    st.markdown(
        """
**Mock Mode** runs entirely in this app with synthetic outputs.

**API Mode** will POST the image and dosage metadata to your deployed service.

**Local Artifact Mode** will load `joblib` / `pickle` / ONNX / Torch weights from disk.

Real inference is **not** wired yet — see `services/api_client.py` and `services/inference.py` for TODOs.
        """
    )
    st.subheader("Current configuration (also editable in the sidebar)")
    st.write(
        {
            "backend_mode": st.session_state.get("backend_mode"),
            "api_base_url": st.session_state.get("api_base_url"),
            "api_endpoint": st.session_state.get("api_endpoint"),
            "api_timeout_s": st.session_state.get("api_timeout_s"),
            "local_model_path": st.session_state.get("local_model_path") or "—",
            "local_preprocessor_path": st.session_state.get("local_preprocessor_path") or "—",
            "local_label_config_path": st.session_state.get("local_label_config_path") or "—",
        }
    )
