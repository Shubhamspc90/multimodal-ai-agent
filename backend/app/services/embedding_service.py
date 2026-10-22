"""Generate normalized text embeddings using Sentence Transformers."""

from typing import Any


class EmbeddingService:
    """Convert text into vector embeddings."""

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Any | None = None,
    ) -> None:
        """Initialize the service without loading the model immediately."""
        self._model_name = model_name
        self._model = model

    def _get_model(self) -> Any:
        """Load the embedding model only when it is first needed."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embeddings for a batch of text strings."""
        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text must be a non-empty string.")

        model = self._get_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        if vectors.ndim != 2 or vectors.shape != (
            len(texts),
            self.EMBEDDING_DIMENSIONS,
        ):
            raise ValueError(
                "Unexpected embedding shape: "
                f"{vectors.shape}. Expected "
                f"({len(texts)}, {self.EMBEDDING_DIMENSIONS})."
            )

        return vectors.tolist()

    def embed_text(self, text: str) -> list[float]:
        """Generate one normalized embedding for a text string."""
        return self.embed_texts([text])[0]
