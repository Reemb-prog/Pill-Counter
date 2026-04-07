"""
Unified inference adapter: Mock, API (future), or Local artifact (future).

Use `run_analysis(...)` from the UI layer (see `ui/components.py`).

TODO (Colab → production):
- **API**: Implement PillCountApiClient.post_predict and decode base64 `images` to PIL.
- **Local**: Load joblib/pickle/ONNX/Torch in LocalArtifactRunner and return the same dict shape.
"""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from services.api_client import ApiClientConfig, PillCountApiClient
from services.mock_pipeline import run_mock_pipeline
from utils.config import BACKEND_API, BACKEND_LOCAL, BACKEND_MOCK


def _b64_to_pil(b64: str | None) -> Image.Image | None:
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _attach_pil_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """If API returns only base64, build _pil for the UI."""
    if "_pil" in result and result["_pil"]:
        return result
    imgs = result.get("images") or {}
    pil_map: dict[str, Image.Image] = {}
    for key in ("original", "grayscale", "denoised", "segmented", "components_overlay"):
        p = _b64_to_pil(imgs.get(key))
        if p is not None:
            pil_map[key] = p
    if pil_map:
        result = {**result, "_pil": pil_map}
    return result


class PipelineRunner(ABC):
    """Abstract runner: implement for API and local artifact backends."""

    @abstractmethod
    def run(
        self,
        image: Image.Image,
        expected_dosage: int,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class MockRunner(PipelineRunner):
    def run(
        self,
        image: Image.Image,
        expected_dosage: int,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        return run_mock_pipeline(
            image,
            expected_dosage,
            threshold=float(settings.get("threshold", 127)),
            resize_w=settings.get("resize_w"),
            resize_h=settings.get("resize_h"),
            min_region_area=int(settings.get("min_region_area", 50)),
            max_region_area=int(settings.get("max_region_area", 50000)),
            morph_kernel=int(settings.get("morph_kernel", 3)),
        )


class ApiRunner(PipelineRunner):
    def __init__(self, client: PillCountApiClient) -> None:
        self.client = client

    def run(
        self,
        image: Image.Image,
        expected_dosage: int,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        # TODO: when post_predict is implemented, merge settings into request body
        result = self.client.post_predict(buf.getvalue(), expected_dosage, settings)
        return _attach_pil_from_result(result)


class LocalArtifactRunner(PipelineRunner):
    """
    Placeholder for joblib/pickle/ONNX/PyTorch/TF SavedModel loading.

    TODO:
    - Load preprocessor + model paths from session / config.
    - Run your exported pipeline and build the same response dict as the API.
    """

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str = "",
        label_config_path: str = "",
    ) -> None:
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.label_config_path = label_config_path

    def run(
        self,
        image: Image.Image,
        expected_dosage: int,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Local artifact inference not connected. Export your Colab model to "
            "joblib/pkl/ONNX/.pt and implement LocalArtifactRunner.run(). "
            f"Configured model path: {self.model_path!r}"
        )


def get_runner(
    backend_mode: str,
    *,
    api_base_url: str,
    api_endpoint: str,
    api_timeout_s: float,
    local_model_path: str,
    local_preprocessor_path: str,
    local_label_config_path: str,
) -> PipelineRunner:
    if backend_mode == BACKEND_MOCK:
        return MockRunner()
    if backend_mode == BACKEND_API:
        cfg = ApiClientConfig(base_url=api_base_url, endpoint=api_endpoint, timeout_s=api_timeout_s)
        return ApiRunner(PillCountApiClient(cfg))
    if backend_mode == BACKEND_LOCAL:
        return LocalArtifactRunner(
            model_path=local_model_path,
            preprocessor_path=local_preprocessor_path,
            label_config_path=local_label_config_path,
        )
    return MockRunner()


def run_analysis(
    backend_mode: str,
    image: Image.Image,
    expected_dosage: int,
    settings: dict[str, Any],
    api_base_url: str,
    api_endpoint: str,
    api_timeout_s: float,
    local_model_path: str,
    local_preprocessor_path: str,
    local_label_config_path: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Top-level call used by the UI. Returns (result_dict, error_message).
    """
    runner = get_runner(
        backend_mode,
        api_base_url=api_base_url,
        api_endpoint=api_endpoint,
        api_timeout_s=api_timeout_s,
        local_model_path=local_model_path,
        local_preprocessor_path=local_preprocessor_path,
        local_label_config_path=local_label_config_path,
    )
    try:
        out = runner.run(image, expected_dosage, settings)
        return out, None
    except NotImplementedError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Inference error: {e}"
