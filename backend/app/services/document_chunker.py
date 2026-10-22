"""Split document text into overlapping chunks for retrieval."""


class DocumentChunker:
    """Create bounded, overlapping text chunks."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks, preferring natural boundaries."""

        text = text.strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        text_length = len(text)
        minimum_boundary = int(self.chunk_size * 0.7)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                search_start = start + minimum_boundary
                boundary = -1

                # Prefer paragraph, line, sentence, then word boundaries.
                for separator in ("\n\n", "\n", ". ", " "):
                    candidate = text.rfind(separator, search_start, end)
                    if candidate > boundary:
                        boundary = candidate + len(separator)

                if boundary > start:
                    end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            next_start = end - self.chunk_overlap

            # Guarantee forward progress even with unusual input.
            start = max(start + 1, next_start)

        return chunks
