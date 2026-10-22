"""Tests for the embedding service."""

from unittest.mock import Mock

import numpy as np
import pytest

from backend.app.services.embedding_service import EmbeddingService


def test_embed_text_returns_384_dimensions() -> None:
    model = Mock()
    model.encode.return_value = np.ones((1, 384))

    service = EmbeddingService(model=model)

    result = service.embed_text("Artificial intelligence")

    assert len(result) == 384
    model.encode.assert_called_once_with(
        ["Artificial intelligence"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def test_embed_texts_returns_one_vector_per_text() -> None:
    model = Mock()
    model.encode.return_value = np.ones((2, 384))

    service = EmbeddingService(model=model)

    result = service.embed_texts(["First document", "Second document"])

    assert len(result) == 2
    assert all(len(vector) == 384 for vector in result)


def test_empty_list_returns_empty_list_without_loading_model() -> None:
    service = EmbeddingService()

    assert service.embed_texts([]) == []
    assert service._model is None


@pytest.mark.parametrize("text", ["", "   "])
def test_embed_text_rejects_blank_text(text: str) -> None:
    service = EmbeddingService(model=Mock())

    with pytest.raises(ValueError, match="non-empty string"):
        service.embed_text(text)


def test_unexpected_embedding_dimensions_raise_error() -> None:
    model = Mock()
    model.encode.return_value = np.ones((1, 256))

    service = EmbeddingService(model=model)

    with pytest.raises(ValueError, match="Unexpected embedding shape"):
        service.embed_text("Test document")
