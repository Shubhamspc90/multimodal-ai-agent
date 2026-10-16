import wave
from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.schemas.audio import AudioInputResponse

router = APIRouter(tags=["Audio"])

MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
}


@router.post(
    "/audio",
    response_model=AudioInputResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_audio(
    file: Annotated[UploadFile, File(...)],
) -> AudioInputResponse:
    """Accept and validate an uploaded WAV audio file."""

    content_type = (file.content_type or "").lower()

    if content_type not in ALLOWED_CONTENT_TYPES:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio content type. Upload a WAV file.",
        )

    try:
        contents = await file.read(MAX_AUDIO_SIZE_BYTES + 1)
    finally:
        await file.close()

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Audio exceeds the 10 MiB upload limit.",
        )

    try:
        with wave.open(BytesIO(contents), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            compression_type = audio.getcomptype()

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

            frame_data = audio.readframes(frame_count)
            expected_frame_bytes = frame_count * channels * sample_width

            if len(frame_data) < expected_frame_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The WAV file is incomplete or corrupted.",
                )

    except HTTPException:
        raise
    except (wave.Error, EOFError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid WAV audio file.",
        ) from exc

    return AudioInputResponse(
        filename=file.filename or "upload.wav",
        content_type=content_type,
        size_bytes=len(contents),
        audio_format="WAV",
        duration_seconds=frame_count / sample_rate,
        sample_rate_hz=sample_rate,
        channels=channels,
    )
