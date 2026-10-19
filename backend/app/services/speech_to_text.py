from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

from faster_whisper import WhisperModel


@dataclass(frozen=True)
class TranscriptionResult:
    """Represent the result of speech recognition."""

    transcript: str
    language: str
    language_probability: float


@lru_cache(maxsize=1)
def _get_whisper_model() -> WhisperModel:
    """Load the speech-recognition model once and reuse it."""

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )


def transcribe_wav(audio_bytes: bytes) -> TranscriptionResult:
    """Transcribe speech from validated WAV audio bytes."""

    if not audio_bytes:
        raise ValueError("Audio data must not be empty.")

    segments, info = _get_whisper_model().transcribe(
        BytesIO(audio_bytes),
        beam_size=5,
        vad_filter=True,
    )

    # Consume the segments and combine recognized speech into one transcript.
    transcript = " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    ).strip()

    return TranscriptionResult(
        transcript=transcript,
        language=info.language,
        language_probability=float(info.language_probability),
    )
