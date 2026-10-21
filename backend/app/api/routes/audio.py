import logging
import wave
from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.audio import (
    AudioInputResponse,
    AudioTranscriptionResponse,
)
from backend.app.services.speech_to_text_service import SpeechToTextService

# Logger for recording errors and important events in this module.
logger = logging.getLogger(__name__)

# Group all audio-related API endpoints under the "Audio" tag.
router = APIRouter(tags=["Audio"])
speech_to_text_service = SpeechToTextService()
# Maximum allowed uploaded audio size: 10 MiB.
MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024

# WAV content types accepted by the API.
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
}


async def _read_validated_wav(
    file: UploadFile,
) -> tuple[bytes, str, int, int, int]:
    """Read and validate an uploaded WAV file."""

    # Read the uploaded file's MIME type in lowercase for reliable comparison.
    content_type = (file.content_type or "").lower()

    # Reject files that are not declared as WAV audio.
    if content_type not in ALLOWED_CONTENT_TYPES:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio content type. Upload a WAV file.",
        )

    try:
        # Read one extra byte so oversized files can be detected reliably.
        contents = await file.read(MAX_AUDIO_SIZE_BYTES + 1)
    finally:
        # Always close the upload file, even if reading fails.
        await file.close()

    # Reject an uploaded file with no data.
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # Reject files larger than the configured 10 MiB limit.
    if len(contents) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Audio exceeds the 10 MiB upload limit.",
        )

    try:
        # Open the uploaded bytes as a WAV file for validation.
        with wave.open(BytesIO(contents), "rb") as audio:
            # Read basic WAV audio properties.
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            compression_type = audio.getcomptype()

            # Reject invalid or compressed WAV data.
            if (
                channels <= 0
                or sample_width <= 0
                or sample_rate <= 0
                or frame_count <= 0
                or compression_type != "NONE"
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The WAV file has unsupported or invalid audio data.",
                )

            # Read the actual audio frame data from the WAV file.
            frame_data = audio.readframes(frame_count)

            # Calculate the expected number of bytes for all audio frames.
            expected_frame_bytes = frame_count * channels * sample_width

            # Reject files whose audio data is incomplete or corrupted.
            if len(frame_data) < expected_frame_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The WAV file is incomplete or corrupted.",
                )

    # Preserve HTTP errors raised inside the validation block.
    except HTTPException:
        raise

    # Convert WAV parsing or I/O failures into a client-friendly 400 error.
    except (wave.Error, EOFError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid WAV audio file.",
        ) from exc

    # Return the validated audio bytes and useful audio metadata.
    return contents, content_type, sample_rate, channels, frame_count


@router.post(
    "/audio",
    response_model=AudioInputResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_audio(
    file: Annotated[UploadFile, File(...)],
) -> AudioInputResponse:
    """Accept and validate an uploaded WAV audio file."""

    # Validate the uploaded WAV file and extract its metadata.
    (
        contents,
        content_type,
        sample_rate,
        channels,
        frame_count,
    ) = await _read_validated_wav(file)

    # Return details about the accepted audio upload.
    return AudioInputResponse(
        filename=file.filename or "upload.wav",
        content_type=content_type,
        size_bytes=len(contents),
        audio_format="WAV",
        duration_seconds=frame_count / sample_rate,
        sample_rate_hz=sample_rate,
        channels=channels,
    )


@router.post(
    "/audio/transcribe",
    response_model=AudioTranscriptionResponse,
    status_code=status.HTTP_200_OK,
)
async def transcribe_audio(
    file: Annotated[UploadFile, File(...)],
) -> AudioTranscriptionResponse:
    """Validate a WAV upload and transcribe its speech."""

    # Validate the uploaded WAV file before running speech recognition.
    contents, _, sample_rate, _, frame_count = await _read_validated_wav(file)

    try:
        # Run CPU-heavy transcription in a thread pool to avoid blocking the event loop.
        result = await run_in_threadpool(
            speech_to_text_service.transcribe,
            contents,
        )
    except Exception as exc:
        # Log the full transcription failure for debugging.
        logger.exception("Audio transcription failed")

        # Return a safe service-unavailable response without exposing internal errors.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech recognition is temporarily unavailable.",
        ) from exc

    # Return the recognized transcript and related audio metadata.
    return AudioTranscriptionResponse(
        filename=file.filename or "upload.wav",
        transcript=result.transcript,
        language=result.language,
        language_probability=result.language_probability,
        duration_seconds=frame_count / sample_rate,
    )
