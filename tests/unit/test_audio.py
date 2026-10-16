import wave
from io import BytesIO

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def create_wav_bytes() -> bytes:
    """Create a short, valid PCM WAV file for testing."""

    buffer = BytesIO()

    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)

    return buffer.getvalue()


def test_audio_endpoint_accepts_valid_wav() -> None:
    content = create_wav_bytes()

    response = client.post(
        "/api/v1/audio",
        files={"file": ("sample.wav", content, "audio/wav")},
    )

    assert response.status_code == 200
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
    response = client.post(
        "/api/v1/audio",
        files={"file": ("sample.mp3", b"audio data", "audio/mpeg")},
    )

    assert response.status_code == 415


def test_audio_endpoint_rejects_invalid_wav_data() -> None:
    response = client.post(
        "/api/v1/audio",
        files={"file": ("sample.wav", b"not a WAV file", "audio/wav")},
    )

    assert response.status_code == 400


def test_audio_endpoint_rejects_empty_file() -> None:
    response = client.post(
        "/api/v1/audio",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400


def test_audio_endpoint_rejects_oversized_file() -> None:
    oversized_content = b"a" * (10 * 1024 * 1024 + 1)

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

    assert response.status_code == 413
