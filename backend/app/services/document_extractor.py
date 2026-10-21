"""Extract text from supported document formats."""

from io import BytesIO

from pypdf import PdfReader


class DocumentExtractionError(ValueError):
    """Raised when a document cannot be parsed or contains no text."""


def extract_document_text(
    contents: bytes,
    document_format: str,
) -> tuple[str, int | None]:
    """Extract text and optional page count from PDF or TXT bytes."""

    if not contents:
        raise DocumentExtractionError("Document must not be empty.")

    if document_format == "TXT":
        try:
            text = contents.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise DocumentExtractionError(
                "Text documents must use UTF-8 encoding."
            ) from exc

        if not text:
            raise DocumentExtractionError("Document does not contain extractable text.")

        return text, None

    if document_format == "PDF":
        try:
            reader = PdfReader(BytesIO(contents), strict=True)
            if reader.is_encrypted:
                raise DocumentExtractionError(
                    "Encrypted PDF documents are not supported."
                )

            page_text = [page.extract_text() or "" for page in reader.pages]
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(
                "The PDF document could not be parsed."
            ) from exc

        text = "\n".join(part.strip() for part in page_text if part.strip())

        if not text:
            raise DocumentExtractionError(
                "PDF does not contain extractable text. "
                "Scanned PDFs are not supported yet."
            )

        return text, len(reader.pages)

    raise DocumentExtractionError("Unsupported document format.")
