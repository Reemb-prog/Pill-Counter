"""
Trained YOLO inference adapter for the Streamlit UI.

Source of truth for inference workflow:
- notebook: "pill couter and dosage decision full pipeline notebook"
  - model.predict(..., conf=0.7, iou=0.45, verbose=False)
  - predicted_count = 0 if r.boxes is None else len(r.boxes)
  - dosage decision based on predicted_count vs expected_dosage

This module returns a dict shaped to match what `ui/components.py` expects:
- detected_count, expected_dosage, status ("Correct dosage"/"Too few"/"Too many")
- images + _pil stage images (keys: original, grayscale, denoised, segmented, components_overlay)
- features list derived from detection boxes
- metrics dict with UI-required keys
"""

from __future__ import annotations

import base64
import io
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from utils.config import BACKEND_LOCAL, PROJECT_ROOT


STATUS_CORRECT = "Correct dosage"
STATUS_TOO_FEW = "Too few"
STATUS_TOO_MANY = "Too many"


def _pil_to_base64_png(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _dosage_status(detected: int, expected: int) -> str:
    if detected == expected:
        return STATUS_CORRECT
    if detected < expected:
        return STATUS_TOO_FEW
    return STATUS_TOO_MANY


@lru_cache(maxsize=2)
def _load_yolo_model(model_path: str) -> Any:
    # Import lazily so the app can start even if ultralytics isn't installed yet.
    from ultralytics import YOLO  # type: ignore

    return YOLO(model_path)


def _make_segmented_placeholder(image_rgb: Image.Image, *, threshold: float) -> Image.Image:
    """
    Segmentation-style visualization for the explorer UI.
    Not used for counting; only to populate the "Pipeline explorer" tab safely.
    """

    gray = ImageOps.grayscale(image_rgb)
    arr = np.asarray(gray).astype(np.float32)
    mask = (arr >= float(threshold)).astype(np.uint8) * 255

    # Red-tint overlay where mask is present.
    r = arr.copy()
    r[mask == 255] = 255.0
    g = arr.copy() * 0.45
    b = arr.copy() * 0.45

    out = np.stack([r, g, b], axis=-1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGB")


def _build_features_from_boxes(boxes_xyxy: list[list[float]]) -> list[dict[str, Any]]:
    feats: list[dict[str, Any]] = []
    for i, xyxy in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = xyxy
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        area = w * h
        perimeter = 2.0 * (w + h)
        aspect_ratio = w / max(h, 1e-6)

        feats.append(
            {
                "region_id": i + 1,
                "area_px": round(area, 1),
                "perimeter_px": round(perimeter, 1),
                # Box-derived stand-ins so the existing feature table stays populated.
                "circularity": 1.0,
                "bbox_w_px": round(w, 1),
                "bbox_h_px": round(h, 1),
                "aspect_ratio": round(aspect_ratio, 4),
                "solidity": 1.0,
                "valid_pill": "yes",
            }
        )
    return feats


def _build_detections(boxes_xyxy: list[list[float]], confidences: list[float]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for i, xyxy in enumerate(boxes_xyxy):
        conf = float(confidences[i]) if i < len(confidences) else 0.0
        detections.append(
            {
                "detection_id": i + 1,
                "box_xyxy": xyxy,
                "confidence": round(conf, 4),
            }
        )
    return detections


def predict_count_and_decision_pil(
    image: Image.Image,
    *,
    expected_dosage: int,
    model_path: Path,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Runs trained YOLO inference and returns a dict aligned with the UI contract.
    """

    t0 = time.perf_counter()
    settings = settings or {}

    if image is None or image.size[0] < 2 or image.size[1] < 2:
        raise ValueError("Invalid image provided for inference.")

    expected = int(expected_dosage)

    # Prepare minimal UI-stage images for explorer tabs that are not native YOLO outputs.
    image_rgb = image.convert("RGB")
    grayscale_pil = ImageOps.grayscale(image_rgb).convert("RGB")
    denoised_pil = image_rgb.filter(ImageFilter.MedianFilter(size=3))
    segmented_pil = _make_segmented_placeholder(image_rgb, threshold=float(settings.get("threshold", 127.0)))

    model = _load_yolo_model(str(model_path))

    # Notebook uses conf=0.7, iou=0.45, verbose=False.
    conf_t = float(0.7)
    iou_t = float(0.45)

    # Ultralytics accepts numpy arrays / PIL; use numpy for consistency.
    img_np = np.asarray(image_rgb)

    yolo_results = model.predict(source=img_np, conf=conf_t, iou=iou_t, verbose=False)
    r = yolo_results[0]

    boxes_xyxy: list[list[float]] = []
    confidences: list[float] = []

    if getattr(r, "boxes", None) is not None:
        # Ultralytics boxes fields are torch tensors; be defensive in case a field is missing.
        xyxy_t = getattr(r.boxes, "xyxy", None)
        if xyxy_t is not None:
            boxes_xyxy = xyxy_t.detach().cpu().tolist()
        conf_t = getattr(r.boxes, "conf", None)
        if conf_t is not None:
            confidences = conf_t.detach().cpu().tolist()

    detected_count = int(len(boxes_xyxy))
    status = _dosage_status(detected_count, max(expected, 0))

    # UI expects an annotated image in "Detected components" via components_overlay.
    components_overlay_pil: Image.Image
    try:
        annotated_bgr = r.plot()
        components_overlay_pil = Image.fromarray(annotated_bgr[:, :, ::-1]).convert("RGB")
    except Exception:
        # Fallback: keep something sensible visible.
        components_overlay_pil = image_rgb.copy()

    # Build UI feature rows from YOLO boxes.
    features = _build_features_from_boxes(boxes_xyxy)
    detections = _build_detections(boxes_xyxy, confidences)

    proc = time.perf_counter() - t0
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    # Stage images required by the UI tabs.
    pil_map = {
        "original": image_rgb,
        "grayscale": grayscale_pil,
        "denoised": denoised_pil,
        "segmented": segmented_pil,
        "components_overlay": components_overlay_pil,
    }
    images_b64 = {k: _pil_to_base64_png(v) for k, v in pil_map.items()}

    # UI "metrics" tab expects several keys; fill from UI settings where possible.
    metrics = {
        "confidence": round(min(avg_conf, 0.999), 3),
        "threshold_used": float(settings.get("threshold", 127.0)),
        "min_region_area": int(settings.get("min_region_area", 50)),
        "max_region_area": int(settings.get("max_region_area", 50000)),
        "morph_kernel": int(settings.get("morph_kernel", 3)),
        "candidate_regions": detected_count,
    }

    return {
        "success": True,
        "detected_count": detected_count,
        "expected_dosage": expected,
        "status": status,
        "validation_status": "valid",
        "validation_message": "Image processed successfully.",
        "processing_time": float(round(proc, 4)),
        "backend_mode": f"{BACKEND_LOCAL} (YOLO best.pt)",
        "images": images_b64,
        "features": features,
        "detections": detections,
        "metrics": metrics,
        "_pil": pil_map,
        "warnings": [],
        # Helpful debug info for advanced use (not required by UI).
        "_yolo": {
            "conf_used": conf_t,
            "iou_used": iou_t,
            "boxes_xyxy": boxes_xyxy,
            "confidences": confidences,
        },
    }

