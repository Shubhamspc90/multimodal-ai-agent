"""Local Faster-Whisper speech-to-text provider."""

from functools import lru_cache
from io import BytesIO

from faster_whisper import WhisperModel

from backend.app.core.config import settings
from backend.app.services.speech_to_text import TranscriptionResult


@lru_cache(maxsize=1)
def _get_whisper_model() -> WhisperModel:
    """Load and cache the configured Whisper model."""

    return WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


class FasterWhisperProvider:
    """Transcribe WAV audio using a local Faster-Whisper model."""

    def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        """Return the recognized transcript and language information."""

        if not audio_bytes:
            raise ValueError("Audio data must not be empty.")

        segments, info = _get_whisper_model().transcribe(
            BytesIO(audio_bytes),
            beam_size=5,
            vad_filter=True,
        )

        transcript = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()

        return TranscriptionResult(
            transcript=transcript,
            language=info.language,
            language_probability=float(info.language_probability),
        )
