"""Unit tests for document text extraction."""

from io import BytesIO

import pytest
from pypdf import PdfWriter

from backend.app.services.document_extractor import (
    DocumentExtractionError,
    extract_document_text,
)


def test_extracts_utf8_text() -> None:
    text, page_count = extract_document_text(
        b"\xef\xbb\xbfHello document",
        "TXT",
    )

    assert text == "Hello document"
    assert page_count is None


def test_rejects_empty_document() -> None:
    with pytest.raises(DocumentExtractionError, match="must not be empty"):
        extract_document_text(b"", "TXT")


def test_rejects_invalid_utf8() -> None:
    with pytest.raises(DocumentExtractionError, match="UTF-8"):
        extract_document_text(b"\xff\xfe", "TXT")


def test_rejects_unsupported_format() -> None:
    with pytest.raises(DocumentExtractionError, match="Unsupported"):
        extract_document_text(b"data", "DOCX")


def test_rejects_pdf_without_extractable_text() -> None:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(output)

    with pytest.raises(
        DocumentExtractionError,
        match="extractable text",
    ):
        extract_document_text(output.getvalue(), "PDF")
