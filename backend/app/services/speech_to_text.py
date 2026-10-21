"""Shared speech-to-text result model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionResult:
    """Represent the result of speech recognition."""

    transcript: str
    language: str
    language_probability: float
