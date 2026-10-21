"""API routes for document understanding."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.document import DocumentExtractionResponse
from backend.app.services.document_extractor import DocumentExtractionError
from backend.app.services.document_service import DocumentService

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
