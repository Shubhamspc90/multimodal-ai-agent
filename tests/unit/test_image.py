from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app

client = TestClient(app)


def create_png_bytes() -> bytes:
    """Create a small valid PNG image for testing."""

    buffer = BytesIO()

    with Image.new("RGB", (2, 3), color="red") as image:
        image.save(buffer, format="PNG")

    return buffer.getvalue()


def test_image_endpoint_accepts_valid_png() -> None:
    response = client.post(
        "/api/v1/images",
        files={
            "file": (
                "test.png",
                create_png_bytes(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "filename": "test.png",
        "content_type": "image/png",
        "size_bytes": len(create_png_bytes()),
        "image_format": "PNG",
        "width": 2,
        "height": 3,
    }


def test_image_endpoint_rejects_unsupported_content_type() -> None:
    response = client.post(
        "/api/v1/images",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_image_endpoint_rejects_invalid_image_data() -> None:
    response = client.post(
        "/api/v1/images",
        files={"file": ("test.png", b"invalid image", "image/png")},
    )

    assert response.status_code == 400


def test_image_endpoint_rejects_empty_file() -> None:
    response = client.post(
        "/api/v1/images",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400


def test_image_endpoint_rejects_oversized_file() -> None:
    oversized_content = b"a" * (5 * 1024 * 1024 + 1)

    response = client.post(
        "/api/v1/images",
        files={
            "file": (
                "large.png",
                oversized_content,
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_image_endpoint_rejects_mismatched_content_type() -> None:
    buffer = BytesIO()

    with Image.new("RGB", (2, 2), color="blue") as image:
        image.save(buffer, format="JPEG")

    response = client.post(
        "/api/v1/images",
        files={"file": ("test.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 415
