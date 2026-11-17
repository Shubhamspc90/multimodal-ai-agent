"""Interfaces for AI model providers."""

from typing import Protocol


class TextModelProvider(Protocol):
    """Define the contract required by a text-generation provider."""

    async def generate_text(self, message: str) -> str:
        """Generate a response for the supplied user message."""
        ...

    async def generate_messages(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Generate a response using structured conversation messages."""
        ...
