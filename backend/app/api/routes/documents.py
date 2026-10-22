"""API routes for document understanding."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.app.core.config import settings
from backend.app.schemas.document import (
    DocumentExtractionResponse,
    DocumentSummarizationResponse,
)
from backend.app.services.document_extractor import DocumentExtractionError
from backend.app.services.document_service import DocumentService
from backend.app.services.ollama_provider import (
    OllamaCloudProvider,
    OllamaProviderError,
)
from backend.app.services.summarization_service import SummarizationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

document_service = DocumentService()

MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024
CHUNK_SIZE_BYTES = 64 * 1024


@router.post(
    "/extract",
    response_model=DocumentExtractionResponse,
    status_code=status.HTTP_200_OK,
)
async def extract_document(
    file: Annotated[UploadFile, File(...)],
) -> DocumentExtractionResponse:
    """Extract text from an uploaded PDF or TXT document."""

    filename = file.filename or "document"
    content_type = file.content_type or "application/octet-stream"

    try:
        suffix = filename.rsplit(".", maxsplit=1)[-1].lower()
        document_format = {"pdf": "PDF", "txt": "TXT"}.get(suffix)

        if document_format is None:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF and TXT documents are supported.",
            )

        contents = bytearray()

        while chunk := await file.read(CHUNK_SIZE_BYTES):
            contents.extend(chunk)

            if len(contents) > MAX_DOCUMENT_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Document exceeds the 10 MiB size limit.",
                )
    finally:
        await file.close()

    try:
        text, page_count = await run_in_threadpool(
            document_service.extract,
            bytes(contents),
            document_format,
        )
    except DocumentExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Document extraction failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document extraction is temporarily unavailable.",
        ) from exc

    return DocumentExtractionResponse(
        filename=filename,
        content_type=content_type,
        size_bytes=len(contents),
        document_format=document_format,
        text=text,
        page_count=page_count,
    )


MAX_SUMMARIZATION_CHARACTERS = 50_000


def get_summarization_service() -> SummarizationService:
    """Build the summarization service using the configured Ollama provider."""

    return SummarizationService(OllamaCloudProvider(settings))


@router.post(
    "/summarize",
    response_model=DocumentSummarizationResponse,
    status_code=status.HTTP_200_OK,
)
async def summarize_document(
    file: Annotated[UploadFile, File(...)],
) -> DocumentSummarizationResponse:
    """Extract and summarize an uploaded PDF or TXT document."""

    filename = file.filename or "document"
    content_type = file.content_type or "application/octet-stream"

    try:
        suffix = filename.rsplit(".", maxsplit=1)[-1].lower()
        document_format = {"pdf": "PDF", "txt": "TXT"}.get(suffix)

        if document_format is None:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF and TXT documents are supported.",
            )

        contents = bytearray()

        while chunk := await file.read(CHUNK_SIZE_BYTES):
            contents.extend(chunk)

            if len(contents) > MAX_DOCUMENT_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Document exceeds the 10 MiB size limit.",
                )
    finally:
        await file.close()

    try:
        text, page_count = await run_in_threadpool(
            document_service.extract,
            bytes(contents),
            document_format,
        )
    except DocumentExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Document extraction failed during summarization")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document extraction is temporarily unavailable.",
        ) from exc

    if len(text) > MAX_SUMMARIZATION_CHARACTERS:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                "Extracted document text exceeds the 50,000-character "
                "summarization limit."
            ),
        )

    try:
        summarization_service = get_summarization_service()
        summary = await summarization_service.summarize(text)
    except ValueError as exc:
        # For example, Ollama credentials may not be configured.
        logger.warning("Document summarization could not be initialized: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document summarization is not configured or available.",
        ) from exc
    except OllamaProviderError as exc:
        logger.warning("Ollama provider failed during document summarization")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The summarization provider is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("Document summarization failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document summarization is temporarily unavailable.",
        ) from exc

    return DocumentSummarizationResponse(
        filename=filename,
        content_type=content_type,
        size_bytes=len(contents),
        document_format=document_format,
        summary=summary,
        page_count=page_count,
    )
