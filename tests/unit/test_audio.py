import wave
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

# Import the audio route module so its function can be replaced in the test.
from backend.app.api.routes import audio as audio_route
from backend.app.main import app
from backend.app.services.speech_to_text import TranscriptionResult

# Create a test client for calling the FastAPI application without starting a server.
client = TestClient(app)


def create_wav_bytes() -> bytes:
    """Create a short, valid PCM WAV file for testing."""

    # Use an in-memory buffer to build the WAV file.
    buffer = BytesIO()

    # Create a 1-channel, 16-bit, 16 kHz WAV file with 160 silent samples.
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)

    # Return the complete WAV file as raw bytes.
    return buffer.getvalue()


def test_audio_endpoint_accepts_valid_wav() -> None:
    # Generate a valid WAV file for the upload request.
    content = create_wav_bytes()

    # Upload the WAV file to the audio endpoint.
    response = client.post(
        "/api/v1/audio",
        files={"file": ("sample.wav", content, "audio/wav")},
    )

    # Confirm the API accepts the valid WAV upload.
    assert response.status_code == 200

    # Confirm the API returns the expected audio metadata.
    assert response.json() == {
        "status": "accepted",
        "filename": "sample.wav",
        "content_type": "audio/wav",
        "size_bytes": len(content),
        "audio_format": "WAV",
        "duration_seconds": 0.01,
        "sample_rate_hz": 16_000,
        "channels": 1,
    }


def test_audio_endpoint_rejects_unsupported_content_type() -> None:
    # Upload an MP3 file, which the endpoint does not support.
    response = client.post(
        "/api/v1/audio",
        files={"file": ("sample.mp3", b"audio data", "audio/mpeg")},
    )

    # Confirm unsupported media types return HTTP 415.
    assert response.status_code == 415


def test_audio_endpoint_rejects_invalid_wav_data() -> None:
    # Upload data labeled as WAV but containing invalid WAV content.
    response = client.post(
        "/api/v1/audio",
        files={"file": ("sample.wav", b"not a WAV file", "audio/wav")},
    )

    # Confirm invalid WAV data returns HTTP 400.
    assert response.status_code == 400


def test_audio_endpoint_rejects_empty_file() -> None:
    # Upload a file with no content.
    response = client.post(
        "/api/v1/audio",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )

    # Confirm empty uploads return HTTP 400.
    assert response.status_code == 400


def test_audio_endpoint_rejects_oversized_file() -> None:
    # Create file content that is one byte larger than the 10 MiB limit.
    oversized_content = b"a" * (10 * 1024 * 1024 + 1)

    # Upload the oversized file to the audio endpoint.
    response = client.post(
        "/api/v1/audio",
        files={
            "file": (
                "large.wav",
                oversized_content,
                "audio/wav",
            )
        },
    )

    # Confirm oversized uploads return HTTP 413.
    assert response.status_code == 413


def test_audio_transcription_returns_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    # Create a fake transcription function instead of loading the real Whisper model.
    def fake_transcribe_wav(audio_bytes: bytes) -> TranscriptionResult:
        # Ensure the endpoint passed non-empty validated audio bytes.
        assert audio_bytes

        # Return a predictable transcription result.
        return TranscriptionResult(
            transcript="Explain this error.",
            language="en",
            language_probability=0.98,
        )

    # Replace the real transcription function with the fake one.
    monkeypatch.setattr(
        audio_route.speech_to_text_service,
        "transcribe",
        fake_transcribe_wav,
    )

    # Upload a valid WAV file to the transcription endpoint.
    response = client.post(
        "/api/v1/audio/transcribe",
        files={
            "file": (
                "instruction.wav",
                create_wav_bytes(),
                "audio/wav",
            )
        },
    )

    # Confirm transcription succeeds.
    assert response.status_code == 200

    # Confirm the response contains the expected transcript and language details.
    assert response.json()["transcript"] == "Explain this error."
    assert response.json()["language"] == "en"
    assert response.json()["language_probability"] == 0.98
    assert response.json()["status"] == "transcribed"


def test_audio_transcription_rejects_invalid_wav() -> None:
    # Upload invalid data while claiming it is a WAV file.
    response = client.post(
        "/api/v1/audio/transcribe",
        files={
            "file": (
                "invalid.wav",
                b"not a WAV file",
                "audio/wav",
            )
        },
    )

    # Confirm invalid WAV input is rejected before transcription.
    assert response.status_code == 400


def test_audio_transcription_handles_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Import the audio route module to replace its transcription function.
    from backend.app.api.routes import audio as audio_route

    # Create a fake transcription function that simulates a model failure.
    def failing_transcribe_wav(audio_bytes: bytes) -> TranscriptionResult:
        raise RuntimeError("Simulated model failure")

    # Replace the real transcription function with the failing fake function.
    monkeypatch.setattr(
        audio_route.speech_to_text_service,
        "transcribe",
        failing_transcribe_wav,
    )

    # Upload a valid WAV file to trigger the simulated failure.
    response = client.post(
        "/api/v1/audio/transcribe",
        files={
            "file": (
                "instruction.wav",
                create_wav_bytes(),
                "audio/wav",
            )
        },
    )

    # Confirm model failures return HTTP 503.
    assert response.status_code == 503

    # Confirm the client receives the safe, user-friendly error message.
    assert response.json()["detail"] == (
        "Speech recognition is temporarily unavailable."
    )
