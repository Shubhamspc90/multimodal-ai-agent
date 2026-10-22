"""Pydantic schemas for document processing."""

from typing import Literal

from pydantic import BaseModel, Field


class DocumentExtractionResponse(BaseModel):
    """Represent extracted document text and metadata."""

    status: Literal["extracted"] = "extracted"
    filename: str
    content_type: str
    size_bytes: int = Field(gt=0)
    document_format: Literal["PDF", "TXT"]
    text: str
    page_count: int | None = Field(default=None, ge=1)


class DocumentSummarizationResponse(BaseModel):
    """Represent a document summarization result."""

    status: Literal["summarized"] = "summarized"
    filename: str
    content_type: str
    size_bytes: int = Field(gt=0)
    document_format: Literal["PDF", "TXT"]
    summary: str
    page_count: int | None = Field(default=None, ge=1)


class DocumentQAResponse(BaseModel):
    """Represent a document question-answering result."""

    status: Literal["answered"] = "answered"
    filename: str
    content_type: str
    size_bytes: int = Field(gt=0)
    document_format: Literal["PDF", "TXT"]
    question: str
    answer: str
    page_count: int | None = Field(default=None, ge=1)
