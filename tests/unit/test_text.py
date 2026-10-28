from fastapi.testclient import TestClient

from backend.app.api.routes import text as text_route
from backend.app.main import app
from backend.app.schemas.agent import AgentInputType, AgentRequest, AgentResponse

client = TestClient(app)


class FakeAgentService:
    """Return a deterministic response without calling an AI provider."""

    async def process_text(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            status="completed",
            input_type=AgentInputType.TEXT,
            response=f"AI-generated answer to: {request.message}",
            capability="text",
        )


def override_agent_service() -> FakeAgentService:
    return FakeAgentService()


def test_text_endpoint_returns_agent_response(monkeypatch) -> None:
    monkeypatch.setattr(text_route, "get_agent_service", override_agent_service)

    response = client.post(
        "/api/v1/text",
        json={"message": "  Explain AI agents.  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "input_type": "text",
        "response": "AI-generated answer to: Explain AI agents.",
        "capability": "text",
    }


def test_text_endpoint_returns_503_when_provider_is_not_configured(
    monkeypatch,
) -> None:
    def unavailable_provider(_settings):
        raise ValueError("Missing Ollama API key")

    monkeypatch.setattr(text_route, "OllamaCloudProvider", unavailable_provider)

    response = client.post(
        "/api/v1/text",
        json={"message": "Explain AI agents."},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Text generation is not configured or available."
    )


def test_text_endpoint_rejects_empty_message() -> None:
    response = client.post("/api/v1/text", json={"message": ""})

    assert response.status_code == 422


def test_text_endpoint_rejects_whitespace_only_message() -> None:
    response = client.post("/api/v1/text", json={"message": "   "})

    assert response.status_code == 422


def test_text_endpoint_rejects_oversized_message() -> None:
    response = client.post(
        "/api/v1/text",
        json={"message": "a" * 10_001},
    )

    assert response.status_code == 422
