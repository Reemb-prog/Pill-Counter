"""
Sample-output image-processing pipeline for Mock Mode.

Produces outputs shaped like the same UI/API contract used by the rest of the app.
"""

from __future__ import annotations

import base64
import io
import hashlib
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from utils.config import BACKEND_MOCK


def _pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _deterministic_count(pil_image: Image.Image, min_c: int = 1, max_c: int = 12) -> int:
    """Stable pseudo-count from image bytes so the same upload always yields the same sample count."""
    raw = pil_image.tobytes()
    h = hashlib.sha256(raw).hexdigest()
    n = int(h[:8], 16) % (max_c - min_c + 1) + min_c
    return int(n)


def _grayscale(pil: Image.Image) -> Image.Image:
    return ImageOps.grayscale(pil).convert("RGB")


def _denoise(pil: Image.Image) -> Image.Image:
    return pil.filter(ImageFilter.MedianFilter(size=3))


def _segment_placeholder(gray_rgb: Image.Image, threshold: float) -> Image.Image:
    """Simple threshold mask tinted for visualization in Mock Mode."""
    g = ImageOps.grayscale(gray_rgb)
    arr = np.asarray(g)
    mask = (arr >= threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).convert("RGB")
    # overlay red tint on foreground for academic visibility
    r, gch, b = mask_img.split()
    tinted = Image.merge("RGB", (r, ImageOps.invert(gch), ImageOps.invert(b)))
    return tinted


def _components_overlay(original: Image.Image, num_regions: int) -> Image.Image:
    """Draw sample bounding boxes / ellipses for detected components."""
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    rng = np.random.default_rng(seed=abs(hash((w, h, num_regions))) % (2**32))
    for i in range(num_regions):
        x0 = int(rng.integers(5, max(10, w // 4)))
        y0 = int(rng.integers(5, max(10, h // 4)))
        x1 = min(w - 5, x0 + int(rng.integers(w // 10, w // 4)))
        y1 = min(h - 5, y0 + int(rng.integers(h // 10, h // 3)))
        draw.rectangle((x0, y0, x1, y1), outline=(0, 120, 255), width=2)
        draw.text((x0 + 2, y0 + 2), str(i + 1), fill=(0, 90, 200))
    return overlay


def _feature_rows(num_regions: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed=seed)
    rows: list[dict[str, Any]] = []
    for i in range(num_regions):
        area = float(rng.integers(800, 4500))
        perim = float(2 * np.sqrt(np.pi * area) * rng.uniform(0.95, 1.12))
        circ = float(rng.uniform(0.72, 0.98))
        bw = float(rng.uniform(22, 55))
        bh = float(rng.uniform(18, 50))
        ar = float(bw / max(bh, 1e-6))
        solidity = float(rng.uniform(0.86, 0.99))
        valid = bool(rng.random() > 0.12)
        rows.append(
            {
                "region_id": i + 1,
                "area_px": round(area, 1),
                "perimeter_px": round(perim, 1),
                "circularity": round(circ, 3),
                "bbox_w_px": round(bw, 1),
                "bbox_h_px": round(bh, 1),
                "aspect_ratio": round(ar, 3),
                "solidity": round(solidity, 3),
                "valid_pill": "yes" if valid else "no",
            }
        )
    return rows


def _dosage_status(detected: int, expected: int) -> str:
    if detected == expected:
        return "Correct dosage"
    if detected < expected:
        return "Too few"
    return "Too many"


def run_mock_pipeline(
    image: Image.Image,
    expected_dosage: int,
    *,
    threshold: float = 127.0,
    resize_w: int | None = None,
    resize_h: int | None = None,
    min_region_area: int = 50,
    max_region_area: int = 50000,
    morph_kernel: int = 3,
) -> dict[str, Any]:
    """
    Run Mock Mode stages and return a dict aligned with the app's result shape.

    Images in `images` are base64 PNG strings for UI decoding; the app may also
    attach PIL copies in session for faster display — see inference.run_analysis.
    """
    t0 = time.perf_counter()
    work = image.copy()
    if resize_w and resize_h:
        work = work.resize((int(resize_w), int(resize_h)), Image.Resampling.LANCZOS)

    detected = _deterministic_count(work)
    # nudge count occasionally so status vs expected is interesting (still deterministic)
    seed = int(hashlib.md5(work.tobytes()).hexdigest()[:8], 16)
    jitter = (seed % 3) - 1  # -1, 0, 1
    detected = max(0, detected + jitter)

    gray = _grayscale(work)
    denoised = _denoise(gray)
    segmented = _segment_placeholder(denoised, threshold)
    num_feature_rows = max(1, min(detected, 15))
    overlay = _components_overlay(work, num_regions=num_feature_rows)
    features = _feature_rows(num_feature_rows, seed=seed)
    status = _dosage_status(detected, int(expected_dosage))
    confidence = float(0.75 + (seed % 20) / 100.0)
    proc = time.perf_counter() - t0

    result: dict[str, Any] = {
        "success": True,
        "detected_count": detected,
        "expected_dosage": int(expected_dosage),
        "status": status,
        "processing_time": round(proc, 3),
        "backend_mode": BACKEND_MOCK,
        "images": {
            "original": _pil_to_base64_png(work),
            "grayscale": _pil_to_base64_png(gray),
            "denoised": _pil_to_base64_png(denoised),
            "segmented": _pil_to_base64_png(segmented),
            "components_overlay": _pil_to_base64_png(overlay),
        },
        "features": features,
        "metrics": {
            "confidence": round(min(confidence, 0.99), 3),
            "threshold_used": threshold,
            "min_region_area": min_region_area,
            "max_region_area": max_region_area,
            "morph_kernel": morph_kernel,
            "candidate_regions": num_feature_rows,
        },
        "validation_status": "valid",
        "validation_message": "Sample-output analysis completed successfully.",
        "detections": [],
        # PIL objects for immediate UI (stripped before any JSON serialization)
        "_pil": {
            "original": work,
            "grayscale": gray,
            "denoised": denoised,
            "segmented": segmented,
            "components_overlay": overlay,
        },
    }
    return result
