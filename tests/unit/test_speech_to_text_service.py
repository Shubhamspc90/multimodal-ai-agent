"""Unit tests for speech-to-text orchestration."""

from unittest.mock import Mock

import pytest

from backend.app.services.speech_to_text import TranscriptionResult
from backend.app.services.speech_to_text_service import SpeechToTextService


def test_service_returns_provider_result() -> None:
    provider = Mock()
    expected = TranscriptionResult(
        transcript="Hello there.",
        language="en",
        language_probability=0.97,
    )
    provider.transcribe.return_value = expected

    service = SpeechToTextService(provider=provider)

    result = service.transcribe(b"audio-bytes")

    assert result == expected
    provider.transcribe.assert_called_once_with(b"audio-bytes")


def test_service_rejects_empty_audio() -> None:
    provider = Mock()
    service = SpeechToTextService(provider=provider)

    with pytest.raises(ValueError, match="must not be empty"):
        service.transcribe(b"")

    provider.transcribe.assert_not_called()
