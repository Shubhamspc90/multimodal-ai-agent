"""API tests for document extraction."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_extracts_uploaded_txt_document() -> None:
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("notes.txt", b"Hello document", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "extracted"
    assert payload["filename"] == "notes.txt"
    assert payload["document_format"] == "TXT"
    assert payload["text"] == "Hello document"
    assert payload["page_count"] is None
    assert payload["size_bytes"] == len(b"Hello document")


def test_rejects_unsupported_extension() -> None:
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("notes.docx", b"document", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert "Only PDF and TXT" in response.json()["detail"]


def test_rejects_empty_txt_document() -> None:
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 422
    assert "must not be empty" in response.json()["detail"]


def test_rejects_document_over_size_limit() -> None:
    from backend.app.api.routes.documents import MAX_DOCUMENT_SIZE_BYTES

    response = client.post(
        "/api/v1/documents/extract",
        files={
            "file": (
                "large.txt",
                b"x" * (MAX_DOCUMENT_SIZE_BYTES + 1),
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
    assert "10 MiB" in response.json()["detail"]
