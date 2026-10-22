"""API tests for document extraction."""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import backend.app.api.routes.documents as documents_routes
from backend.app.main import app
from backend.app.services.ollama_provider import OllamaProviderError
from backend.app.services.summarization_service import SummarizationService

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


def test_summarizes_uploaded_txt_document(monkeypatch) -> None:
    """Return a summary and document metadata without calling Ollama."""

    provider = AsyncMock()
    provider.generate_text.return_value = "Key points from the document."

    monkeypatch.setattr(
        documents_routes,
        "get_summarization_service",
        lambda: SummarizationService(provider),
    )

    response = client.post(
        "/api/v1/documents/summarize",
        files={"file": ("notes.txt", b"Original document text", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "summarized"
    assert payload["filename"] == "notes.txt"
    assert payload["document_format"] == "TXT"
    assert payload["summary"] == "Key points from the document."
    assert payload["size_bytes"] == len(b"Original document text")
    assert payload["page_count"] is None
    provider.generate_text.assert_awaited_once()


def test_summarization_rejects_unsupported_extension() -> None:
    response = client.post(
        "/api/v1/documents/summarize",
        files={"file": ("notes.docx", b"document", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert "Only PDF and TXT" in response.json()["detail"]


def test_summarization_rejects_empty_txt_document() -> None:
    response = client.post(
        "/api/v1/documents/summarize",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 422
    assert "must not be empty" in response.json()["detail"]


def test_summarization_rejects_upload_over_size_limit() -> None:
    response = client.post(
        "/api/v1/documents/summarize",
        files={
            "file": (
                "large.txt",
                b"x" * (documents_routes.MAX_DOCUMENT_SIZE_BYTES + 1),
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
    assert "10 MiB" in response.json()["detail"]


def test_summarization_rejects_extracted_text_over_limit(monkeypatch) -> None:
    """Reject oversized extracted text before initializing the model."""

    monkeypatch.setattr(
        documents_routes.document_service,
        "extract",
        lambda contents, document_format: (
            "x" * (documents_routes.MAX_SUMMARIZATION_CHARACTERS + 1),
            None,
        ),
    )

    def fail_if_initialized():
        raise AssertionError("The model must not be initialized for oversized text.")

    monkeypatch.setattr(
        documents_routes,
        "get_summarization_service",
        fail_if_initialized,
    )

    response = client.post(
        "/api/v1/documents/summarize",
        files={"file": ("notes.txt", b"valid upload", "text/plain")},
    )

    assert response.status_code == 413
    assert "50,000-character" in response.json()["detail"]


def test_summarization_returns_503_when_credentials_are_missing(
    monkeypatch,
) -> None:
    """Return a controlled HTTP error if the provider cannot initialize."""

    def fail_to_initialize():
        raise ValueError("OLLAMA_API_KEY is not configured.")

    monkeypatch.setattr(
        documents_routes,
        "get_summarization_service",
        fail_to_initialize,
    )

    response = client.post(
        "/api/v1/documents/summarize",
        files={"file": ("notes.txt", b"Document text", "text/plain")},
    )

    assert response.status_code == 503
    assert "not configured or available" in response.json()["detail"]


def test_summarization_returns_503_when_provider_fails(monkeypatch) -> None:
    """Do not expose provider details in the HTTP response."""

    provider = AsyncMock()
    provider.generate_text.side_effect = OllamaProviderError(
        "Ollama Cloud returned HTTP 401."
    )

    monkeypatch.setattr(
        documents_routes,
        "get_summarization_service",
        lambda: SummarizationService(provider),
    )

    response = client.post(
        "/api/v1/documents/summarize",
        files={"file": ("notes.txt", b"Document text", "text/plain")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The summarization provider is temporarily unavailable."
    )
    assert "401" not in response.text


def test_answers_question_about_uploaded_txt_document(monkeypatch) -> None:
    """Return an answer and document metadata without calling Ollama."""

    provider = AsyncMock()
    provider.generate_text.return_value = "The project started in 2024."

    from backend.app.services.document_qa_service import DocumentQAService

    monkeypatch.setattr(
        documents_routes,
        "get_document_qa_service",
        lambda: DocumentQAService(provider),
    )

    response = client.post(
        "/api/v1/documents/ask",
        files={"file": ("notes.txt", b"The project started in 2024.", "text/plain")},
        data={"question": "When did the project start?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["filename"] == "notes.txt"
    assert payload["document_format"] == "TXT"
    assert payload["question"] == "When did the project start?"
    assert payload["answer"] == "The project started in 2024."
    assert payload["size_bytes"] == len(b"The project started in 2024.")
    assert payload["page_count"] is None

    provider.generate_text.assert_awaited_once()
    assert "When did the project start?" in provider.generate_text.await_args.args[0]


def test_question_answering_rejects_unsupported_extension() -> None:
    response = client.post(
        "/api/v1/documents/ask",
        files={"file": ("notes.docx", b"document", "application/octet-stream")},
        data={"question": "What is this about?"},
    )

    assert response.status_code == 415


def test_question_answering_rejects_empty_txt_document() -> None:
    response = client.post(
        "/api/v1/documents/ask",
        files={"file": ("empty.txt", b"", "text/plain")},
        data={"question": "What is this about?"},
    )

    assert response.status_code == 422


def test_question_answering_rejects_blank_question() -> None:
    response = client.post(
        "/api/v1/documents/ask",
        files={"file": ("notes.txt", b"Document text", "text/plain")},
        data={"question": "   "},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "The question must not be empty."


def test_question_answering_rejects_extracted_text_over_limit(monkeypatch) -> None:
    """Reject oversized extracted text before initializing the model."""

    monkeypatch.setattr(
        documents_routes.document_service,
        "extract",
        lambda contents, document_format: (
            "x" * (documents_routes.MAX_DOCUMENT_QA_CHARACTERS + 1),
            None,
        ),
    )

    def fail_if_initialized():
        raise AssertionError("The model must not initialize for oversized text.")

    monkeypatch.setattr(
        documents_routes,
        "get_document_qa_service",
        fail_if_initialized,
    )

    response = client.post(
        "/api/v1/documents/ask",
        files={"file": ("notes.txt", b"Valid upload", "text/plain")},
        data={"question": "What does this document say?"},
    )

    assert response.status_code == 413
    assert "50,000-character" in response.json()["detail"]


def test_question_answering_hides_provider_errors(monkeypatch) -> None:
    """Do not expose raw provider errors in API responses."""

    provider = AsyncMock()
    provider.generate_text.side_effect = OllamaProviderError(
        "Ollama Cloud returned HTTP 401."
    )

    from backend.app.services.document_qa_service import DocumentQAService

    monkeypatch.setattr(
        documents_routes,
        "get_document_qa_service",
        lambda: DocumentQAService(provider),
    )

    response = client.post(
        "/api/v1/documents/ask",
        files={"file": ("notes.txt", b"Document text", "text/plain")},
        data={"question": "What is the main point?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The question-answering provider is temporarily unavailable."
    )
    assert "401" not in response.text
