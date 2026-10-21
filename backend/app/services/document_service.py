"""Application service for document extraction."""

from backend.app.services.document_extractor import extract_document_text


class DocumentService:
    """Coordinate extraction from supported document formats."""

    def extract(
        self,
        contents: bytes,
        document_format: str,
    ) -> tuple[str, int | None]:
        """Return extracted text and optional page count."""

        return extract_document_text(contents, document_format)
