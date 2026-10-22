"""Unit tests for document chunking."""

import pytest

from backend.app.services.document_chunker import DocumentChunker


def test_split_text_returns_empty_list_for_empty_input() -> None:
    """Empty or whitespace-only input produces no chunks."""

    chunker = DocumentChunker()

    assert chunker.split_text("") == []
    assert chunker.split_text("   \n  ") == []


def test_split_text_keeps_short_text_in_one_chunk() -> None:
    """Short documents do not need to be split."""

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

    assert chunker.split_text("A short document.") == ["A short document."]


def test_split_text_respects_maximum_chunk_size() -> None:
    """Chunks remain within the configured size."""

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "word " * 100

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_split_text_overlaps_adjacent_chunks() -> None:
    """Adjacent chunks share text to preserve boundary context."""

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "A" * 500

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    assert chunks[0][-20:] == chunks[1][:20]


def test_split_text_prefers_natural_boundaries() -> None:
    """The chunker prefers a nearby paragraph boundary."""

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    text = ("A" * 70) + "\n\n" + ("B" * 70)

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    assert chunks[0].endswith("A" * 70)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_constructor_rejects_invalid_configuration(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Invalid chunk settings fail immediately."""

    with pytest.raises(ValueError):
        DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
