"""API tests for persistent chat conversations."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import conversations as conversations_route
from backend.app.main import app
from backend.app.schemas.agent import AgentInputType, AgentRequest, AgentResponse


class FakeChatHistoryStorage:
    """Provide deterministic conversation storage for API tests."""

    def __init__(self) -> None:
        self.conversations: dict[UUID, dict] = {}

    def create_conversation(self, client_id: UUID) -> dict:
        timestamp = datetime.now(UTC)
        conversation_id = uuid4()

        conversation = {
            "id": conversation_id,
            "client_id": client_id,
            "title": "New conversation",
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": [],
        }

        self.conversations[conversation_id] = conversation
        return self._summary(conversation)

    def list_conversations(self, client_id: UUID) -> list[dict]:
        results = [
            self._summary(conversation)
            for conversation in self.conversations.values()
            if conversation["client_id"] == client_id
        ]

        return sorted(
            results,
            key=lambda item: item["updated_at"],
            reverse=True,
        )

    def get_conversation(
        self,
        conversation_id: UUID,
        client_id: UUID,
    ) -> dict | None:
        conversation = self.conversations.get(conversation_id)

        if conversation is None:
            return None

        if conversation["client_id"] != client_id:
            return None

        return {
            **self._summary(conversation),
            "messages": list(conversation["messages"]),
        }

    def append_message(
        self,
        conversation_id: UUID,
        client_id: UUID,
        role: str,
        content: str,
    ) -> dict | None:
        conversation = self.conversations.get(conversation_id)

        if conversation is None:
            return None

        if conversation["client_id"] != client_id:
            return None

        timestamp = datetime.now(UTC)

        message = {
            "id": len(conversation["messages"]) + 1,
            "role": role,
            "content": content.strip(),
            "created_at": timestamp,
        }

        conversation["messages"].append(message)
        conversation["updated_at"] = timestamp

        if role == "user" and conversation["title"] == "New conversation":
            conversation["title"] = (
                content[:77] + "..." if len(content) > 80 else content
            )

        return message

    @staticmethod
    def _summary(conversation: dict) -> dict:
        """Return the public conversation fields."""
        return {
            "id": conversation["id"],
            "client_id": conversation["client_id"],
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
        }


class FakeAgentService:
    """Generate deterministic AI responses without calling Ollama."""

    async def process_text(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            status="completed",
            input_type=AgentInputType.TEXT,
            response=f"AI-generated answer to: {request.message}",
            capability="text",
        )


@pytest.fixture
def api_client():
    """Create an API client with isolated fake conversation storage."""
    storage = FakeChatHistoryStorage()

    app.dependency_overrides[conversations_route.get_chat_history_storage] = lambda: (
        storage
    )

    try:
        with TestClient(app) as client:
            yield client, storage
    finally:
        app.dependency_overrides.pop(
            conversations_route.get_chat_history_storage,
            None,
        )


def test_create_conversation_returns_created_response(api_client):
    client, _ = api_client
    client_id = uuid4()

    response = client.post(
        "/api/v1/conversations",
        json={"client_id": str(client_id)},
    )

    assert response.status_code == 201

    result = response.json()
    assert result["client_id"] == str(client_id)
    assert result["title"] == "New conversation"
    assert result["id"]


def test_list_conversations_returns_only_client_conversations(api_client):
    client, storage = api_client
    client_id = uuid4()
    another_client_id = uuid4()

    storage.create_conversation(client_id)
    storage.create_conversation(client_id)
    storage.create_conversation(another_client_id)

    response = client.get(
        "/api/v1/conversations",
        params={"client_id": str(client_id)},
    )

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert all(item["client_id"] == str(client_id) for item in response.json())


def test_load_conversation_returns_saved_messages(api_client):
    client, storage = api_client
    client_id = uuid4()
    conversation = storage.create_conversation(client_id)
    conversation_id = conversation["id"]

    storage.append_message(
        conversation_id,
        client_id,
        "user",
        "Explain PostgreSQL.",
    )
    storage.append_message(
        conversation_id,
        client_id,
        "assistant",
        "PostgreSQL is a relational database.",
    )

    response = client.get(
        f"/api/v1/conversations/{conversation_id}",
        params={"client_id": str(client_id)},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["id"] == str(conversation_id)
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"


def test_load_conversation_returns_404_for_another_client(api_client):
    client, storage = api_client
    owner_id = uuid4()
    another_client_id = uuid4()

    conversation = storage.create_conversation(owner_id)

    response = client.get(
        f"/api/v1/conversations/{conversation['id']}",
        params={"client_id": str(another_client_id)},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_send_conversation_message_saves_user_and_ai_messages(
    api_client,
    monkeypatch,
):
    client, storage = api_client
    client_id = uuid4()
    conversation = storage.create_conversation(client_id)
    conversation_id = conversation["id"]

    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: FakeAgentService(),
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(client_id),
            "message": "Explain AI agents.",
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["conversation_id"] == str(conversation_id)
    assert result["user_message"]["role"] == "user"
    assert result["user_message"]["content"] == "Explain AI agents."
    assert result["assistant_message"]["role"] == "assistant"
    assert result["assistant_message"]["content"] == (
        "AI-generated answer to: Explain AI agents."
    )

    saved_conversation = storage.get_conversation(
        conversation_id,
        client_id,
    )

    assert saved_conversation is not None
    assert len(saved_conversation["messages"]) == 2


def test_send_conversation_message_rejects_empty_message(api_client):
    client, storage = api_client
    client_id = uuid4()
    conversation = storage.create_conversation(client_id)

    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages",
        json={
            "client_id": str(client_id),
            "message": "   ",
        },
    )

    assert response.status_code == 422
    assert (
        storage.get_conversation(
            conversation["id"],
            client_id,
        )["messages"]
        == []
    )


def test_send_message_returns_404_for_another_client(
    api_client,
    monkeypatch,
):
    client, storage = api_client
    owner_id = uuid4()
    another_client_id = uuid4()

    conversation = storage.create_conversation(owner_id)

    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: FakeAgentService(),
    )

    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages",
        json={
            "client_id": str(another_client_id),
            "message": "Explain AI agents.",
        },
    )

    assert response.status_code == 404
