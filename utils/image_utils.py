"""
Image loading, validation, sampling, and optional synthetic sample generation.

TODO: When connecting a real preprocessor from Colab, hook resizing/normalization
here or pass arrays through services/inference.py instead of duplicating logic.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from utils.config import ALLOWED_IMAGE_EXTENSIONS, SAMPLE_IMAGES_DIR


def is_allowed_image_file(name: str | None) -> bool:
    if not name:
        return False
    suffix = Path(name).suffix
    return suffix in ALLOWED_IMAGE_EXTENSIONS


def load_image_from_upload(uploaded_file) -> Image.Image:
    """Load a PIL Image from Streamlit UploadedFile."""
    data = uploaded_file.getvalue()
    return Image.open(io.BytesIO(data)).convert("RGB")


def list_sample_images(folder: Path | None = None) -> list[Path]:
    root = folder if folder is not None else SAMPLE_IMAGES_DIR
    if not root.exists():
        return []
    paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        paths.extend(root.glob(ext))
    return sorted(paths)


def load_random_sample(folder: Path | None = None) -> tuple[Image.Image | None, Path | None, str | None]:
    """
    Pick a random image from sample folder.
    Returns (image, path, error_message).
    """
    root = folder if folder is not None else SAMPLE_IMAGES_DIR
    if not root.exists():
        return None, None, f"Sample folder not found: {root}"
    paths = list_sample_images(root)
    if not paths:
        return None, None, (
            f"No images found in {root}. Add PNG/JPEG files to assets/sample_images/ "
            "or use Upload Image."
        )
    path = random.choice(paths)
    try:
        img = Image.open(path).convert("RGB")
        return img, path, None
    except OSError as e:
        return None, None, f"Could not open image: {e}"


def ensure_demo_sample_images() -> None:
    """
    If the sample folder has no images, create a few simple synthetic 'pill' thumbnails
    so demos work out of the box.
    """
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if list_sample_images():
        return
    for i, (w, h, n) in enumerate([(240, 160, 3), (280, 180, 5), (260, 170, 4)]):
        img = Image.new("RGB", (w, h), (248, 250, 252))
        draw = ImageDraw.Draw(img)
        # light neutral background
        for j in range(n):
            cx = 40 + j * (w // max(n, 1))
            cy = h // 2
            r = min(18, h // 6)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(230, 72, 52), outline=(180, 40, 30))
        out = SAMPLE_IMAGES_DIR / f"synthetic_pills_{i + 1:02d}.png"
        img.save(out)


def session_show_image(image: Image.Image, caption: str | None = None) -> None:
    """Display PIL image in Streamlit with optional caption."""
    st.image(image, caption=caption, use_container_width=True)
