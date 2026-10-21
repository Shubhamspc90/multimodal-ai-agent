"""API tests for image-understanding requests."""

from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api.routes.image import get_image_service
from backend.app.main import app
from backend.app.services.ollama_image_provider import ImageProviderError

client = TestClient(app)


def create_png() -> bytes:
    """Build a small valid PNG fixture in memory."""
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def mock_image_service():
    service = AsyncMock()
    service.process_image.return_value = "A small red square."
    app.dependency_overrides[get_image_service] = lambda: service

    yield service

    app.dependency_overrides.pop(get_image_service, None)


def test_upload_image_returns_model_description(mock_image_service) -> None:
    """A valid image is validated and its model description is returned."""
    response = client.post(
        "/api/v1/images",
        files={"file": ("red.png", create_png(), "image/png")},
        data={"prompt": "Describe this image."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["response"] == "A small red square."
    assert body["capability"] == "image_understanding"
    assert body["image_format"] == "PNG"
    assert body["width"] == 8
    assert body["height"] == 8

    mock_image_service.process_image.assert_awaited_once()


def test_upload_image_rejects_unsupported_content_type() -> None:
    response = client.post(
        "/api/v1/images",
        files={"file": ("data.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_image_rejects_invalid_image() -> None:
    response = client.post(
        "/api/v1/images",
        files={"file": ("bad.png", b"not a png", "image/png")},
    )

    assert response.status_code == 400


def test_upload_image_returns_gateway_error_on_provider_failure(
    mock_image_service,
) -> None:
    mock_image_service.process_image.side_effect = ImageProviderError(
        "Provider failed."
    )

    response = client.post(
        "/api/v1/images",
        files={"file": ("red.png", create_png(), "image/png")},
    )

    assert response.status_code == 502
