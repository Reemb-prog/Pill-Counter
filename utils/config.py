"""
Application configuration: paths, constants, and defaults.
Students: adjust SAMPLE_IMAGES_DIR or add paths for your deployment environment.
"""

from __future__ import annotations

from pathlib import Path

# Project root (parent of utils/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

ASSETS_DIR: Path = PROJECT_ROOT / "assets"
SAMPLE_IMAGES_DIR: Path = ASSETS_DIR / "sample_images"
ICONS_DIR: Path = ASSETS_DIR / "icons"
DATA_DIR: Path = PROJECT_ROOT / "data"

EVALUATION_DATA_JSON: Path = DATA_DIR / "sample_metrics.json"

# Backend mode labels (must match session_state / sidebar)
BACKEND_MOCK: str = "Mock Mode"
BACKEND_API: str = "API Mode"
BACKEND_LOCAL: str = "Local Artifact Mode"

BACKEND_OPTIONS: tuple[str, ...] = (BACKEND_MOCK, BACKEND_API, BACKEND_LOCAL)

# Default API settings (placeholders for Colab / deployed FastAPI stack)
DEFAULT_API_BASE_URL: str = "https://your-api.example.com"
DEFAULT_API_ENDPOINT: str = "/v1/pill-count"
DEFAULT_API_TIMEOUT_S: float = 30.0

# Default advanced preprocessing placeholders (UI only until pipeline is wired)
DEFAULT_RESIZE_W: int = 512
DEFAULT_RESIZE_H: int = 512
DEFAULT_THRESHOLD: float = 127.0
DEFAULT_MIN_REGION_AREA: int = 50
DEFAULT_MAX_REGION_AREA: int = 50000
DEFAULT_MORPH_KERNEL: int = 3

# Page routing keys (sidebar navigation)
PAGE_HOME: str = "Home"
PAGE_ANALYSIS: str = "Analysis Dashboard"
PAGE_PIPELINE: str = "Pipeline Explorer"
PAGE_EVALUATION: str = "Evaluation"
PAGE_SETTINGS: str = "Settings / Backend Mode"

NAV_PAGES: tuple[str, ...] = (
    PAGE_HOME,
    PAGE_ANALYSIS,
    PAGE_PIPELINE,
    PAGE_EVALUATION,
    PAGE_SETTINGS,
)

# File types for uploader
ALLOWED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"})
