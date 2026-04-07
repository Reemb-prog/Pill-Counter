"""
REST API client for future Colab-exported FastAPI/Flask pill counting service.

Expected request (multipart or JSON+base64 — implement to match your notebook):
- image: raw bytes or base64
- expected_dosage: int
- settings: optional dict (resize, threshold, morphology, etc.)

Expected response (document this in your API README / OpenAPI schema):

{
  "success": true,
  "detected_count": 8,
  "expected_dosage": 7,
  "status": "Too many",
  "processing_time": 0.84,
  "images": {
    "original": "<base64 png>",
    "grayscale": "...",
    "denoised": "...",
    "segmented": "...",
    "components_overlay": "..."
  },
  "features": [
    {
      "region_id": 1,
      "area_px": 1200.0,
      "perimeter_px": 120.0,
      "circularity": 0.91,
      "bbox_w_px": 40.0,
      "bbox_h_px": 35.0,
      "aspect_ratio": 1.14,
      "solidity": 0.95,
      "valid_pill": "yes"
    }
  ],
  "metrics": {"confidence": 0.91}
}

TODO:
1. Deploy your model as POST /v1/pill-count (or similar) from Colab.
2. Implement post_predict() below to send files/JSON and parse the response.
3. Decode base64 image fields into PIL in inference.py for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ApiClientConfig:
    base_url: str
    endpoint: str
    timeout_s: float = 30.0


class PillCountApiClient:
    """Thin wrapper around requests for the future deployed API."""

    def __init__(self, config: ApiClientConfig) -> None:
        self.config = config

    def build_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        ep = self.config.endpoint if self.config.endpoint.startswith("/") else f"/{self.config.endpoint}"
        return f"{base}{ep}"

    def test_connection(self) -> tuple[bool, str]:
        """
        Ping the server root or a health endpoint.

        TODO: Point this to your service's GET /health once available.
        """
        try:
            url = self.config.base_url.rstrip("/") + "/health"
            r = requests.get(url, timeout=min(5.0, self.config.timeout_s))
            if r.ok:
                return True, f"OK ({r.status_code}) from {url}"
            # Fallback: try base URL
            r2 = requests.get(self.config.base_url.rstrip("/"), timeout=min(5.0, self.config.timeout_s))
            if r2.ok:
                return True, f"OK ({r2.status_code}) from base URL"
            return False, f"Health check failed: {r.status_code} / {r2.status_code}"
        except requests.RequestException as e:
            return False, str(e)

    def post_predict(
        self,
        image_bytes: bytes,
        expected_dosage: int,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send image + metadata to the inference API.

        TODO: Use multipart/form-data:
          files={"image": ("pill.png", image_bytes, "image/png")}
          data={"expected_dosage": expected_dosage, "settings": json.dumps(settings or {})}

        Or JSON with base64 if your Colab stack prefers that.
        """
        raise NotImplementedError(
            "API inference not connected. Implement post_predict() when your Colab/FastAPI "
            "endpoint is deployed, then wire services/inference.py to call it."
        )
