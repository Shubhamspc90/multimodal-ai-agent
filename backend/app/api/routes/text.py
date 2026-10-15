from fastapi import APIRouter, status

from backend.app.schemas.text import TextInputRequest, TextInputResponse

router = APIRouter(tags=["Text"])


@router.post(
    "/text",
    response_model=TextInputResponse,
    status_code=status.HTTP_200_OK,
)
def submit_text(request: TextInputRequest) -> TextInputResponse:
    """Accept a validated text message."""

    return TextInputResponse(received_message=request.message)
