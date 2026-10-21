"""Application service for speech-to-text requests."""

from backend.app.services.faster_whisper_provider import FasterWhisperProvider
from backend.app.services.speech_to_text import TranscriptionResult


class SpeechToTextService:
    """Coordinate speech transcription through a provider."""

    def __init__(self, provider: FasterWhisperProvider | None = None) -> None:
        self._provider = provider or FasterWhisperProvider()

    def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        """Validate audio input and request transcription."""

        if not audio_bytes:
            raise ValueError("Audio data must not be empty.")

        return self._provider.transcribe(audio_bytes)
