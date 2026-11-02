"""API tests for authenticated persistent chat conversations."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app.api import dependencies as auth_dependencies
from backend.app.api.routes import conversations as conversations_route
from backend.app.main import app
from backend.app.schemas.agent import AgentInputType, AgentRequest, AgentResponse


class FakeChatHistoryStorage:
    """Provide deterministic user-owned conversation storage for API tests."""

    def __init__(self) -> None:
        self.conversations: dict[UUID, dict] = {}
        self.authenticated_user_id = uuid4()

    def create_conversation(
        self,
        client_id: UUID,
        user_id: UUID | None = None,
    ) -> dict:
        timestamp = datetime.now(UTC)
        conversation_id = uuid4()

        conversation = {
            "id": conversation_id,
            "client_id": client_id,
            "user_id": user_id,
            "title": "New conversation",
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": [],
        }

        self.conversations[conversation_id] = conversation
        return self._summary(conversation)

    def list_conversations(
        self,
        client_id: UUID | None = None,
        user_id: UUID | None = None,
    ) -> list[dict]:
        if user_id is not None:
            results = [
                self._summary(conversation)
                for conversation in self.conversations.values()
                if conversation["user_id"] == user_id
            ]
        elif client_id is not None:
            results = [
                self._summary(conversation)
                for conversation in self.conversations.values()
                if conversation["client_id"] == client_id
                and conversation["user_id"] is None
            ]
        else:
            raise ValueError("An authenticated user or legacy client ID is required.")

        return sorted(
            results,
            key=lambda item: (item["updated_at"], str(item["id"])),
            reverse=True,
        )

    def get_conversation(
        self,
        conversation_id: UUID,
        client_id: UUID | None = None,
        user_id: UUID | None = None,
    ) -> dict | None:
        conversation = self.conversations.get(conversation_id)

        if conversation is None:
            return None

        if user_id is not None:
            if conversation["user_id"] != user_id:
                return None
        elif client_id is not None:
            if (
                conversation["client_id"] != client_id
                or conversation["user_id"] is not None
            ):
                return None
        else:
            raise ValueError("An authenticated user or legacy client ID is required.")

        return {
            **self._summary(conversation),
            "messages": list(conversation["messages"]),
        }

    def append_message(
        self,
        conversation_id: UUID,
        client_id: UUID | None,
        role: str,
        content: str,
        user_id: UUID | None = None,
    ) -> dict | None:
        conversation = self.conversations.get(conversation_id)

        if conversation is None:
            return None

        if user_id is not None:
            if conversation["user_id"] != user_id:
                return None
        elif client_id is not None:
            if (
                conversation["client_id"] != client_id
                or conversation["user_id"] is not None
            ):
                return None
        else:
            raise ValueError("An authenticated user or legacy client ID is required.")

        content = content.strip()

        if role not in {"user", "assistant"} or not content:
            raise ValueError("Invalid message.")

        timestamp = datetime.now(UTC)

        message = {
            "id": len(conversation["messages"]) + 1,
            "role": role,
            "content": content,
            "created_at": timestamp,
        }

        conversation["messages"].append(message)
        conversation["updated_at"] = timestamp

        if role == "user" and conversation["title"] == "New conversation":
            conversation["title"] = (
                content[:77] + "..." if len(content) > 80 else content
            )

        return message

    def link_legacy_conversations(
        self,
        client_id: UUID,
        user_id: UUID,
    ) -> int:
        """Claim only unowned conversations belonging to the browser ID."""
        linked_count = 0

        for conversation in self.conversations.values():
            if (
                conversation["client_id"] == client_id
                and conversation["user_id"] is None
            ):
                conversation["user_id"] = user_id
                linked_count += 1

        return linked_count

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
    """Create an API client with isolated storage and authenticated user."""
    storage = FakeChatHistoryStorage()

    authenticated_user = {
        "id": storage.authenticated_user_id,
        "name": "Test User",
        "email": "test@example.com",
        "created_at": datetime.now(UTC),
    }

    app.dependency_overrides[conversations_route.get_chat_history_storage] = lambda: (
        storage
    )

    csrf_token = "conversation-test-csrf-token"
    current_session = {
        "user": authenticated_user,
        "csrf_token_hash": auth_dependencies.hash_csrf_token(csrf_token),
    }
    app.dependency_overrides[auth_dependencies.get_current_session] = lambda: (
        current_session
    )

    try:
        with TestClient(app) as client:
            client.cookies.set(auth_dependencies.CSRF_COOKIE_NAME, csrf_token)
            client.headers[auth_dependencies.CSRF_HEADER_NAME] = csrf_token
            yield client, storage
    finally:
        app.dependency_overrides.pop(
            conversations_route.get_chat_history_storage,
            None,
        )
        app.dependency_overrides.pop(
            auth_dependencies.get_current_session,
            None,
        )


def test_create_conversation_returns_created_response(api_client):
    client, storage = api_client
    client_id = uuid4()

    response = client.post(
        "/api/v1/conversations",
        json={"client_id": str(client_id)},
    )

    assert response.status_code == 201

    result = response.json()
    conversation_id = UUID(result["id"])

    assert result["client_id"] == str(client_id)
    assert result["title"] == "New conversation"
    assert storage.conversations[conversation_id]["user_id"] == (
        storage.authenticated_user_id
    )


def test_list_conversations_returns_only_authenticated_user_conversations(
    api_client,
):
    client, storage = api_client

    client_id = uuid4()
    another_client_id = uuid4()
    another_user_id = uuid4()

    storage.create_conversation(client_id, storage.authenticated_user_id)
    storage.create_conversation(client_id, storage.authenticated_user_id)
    storage.create_conversation(another_client_id, another_user_id)

    response = client.get("/api/v1/conversations")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_load_conversation_returns_saved_messages(api_client):
    client, storage = api_client

    client_id = uuid4()
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(client_id, user_id)
    conversation_id = conversation["id"]

    storage.append_message(
        conversation_id,
        None,
        "user",
        "Explain PostgreSQL.",
        user_id=user_id,
    )
    storage.append_message(
        conversation_id,
        None,
        "assistant",
        "PostgreSQL is a relational database.",
        user_id=user_id,
    )

    response = client.get(
        f"/api/v1/conversations/{conversation_id}",
    )

    assert response.status_code == 200

    result = response.json()
    assert result["id"] == str(conversation_id)
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"


def test_load_conversation_returns_404_for_another_user(api_client):
    client, storage = api_client

    another_user_id = uuid4()
    conversation = storage.create_conversation(
        uuid4(),
        another_user_id,
    )

    response = client.get(
        f"/api/v1/conversations/{conversation['id']}",
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_send_conversation_message_saves_user_and_ai_messages(
    api_client,
    monkeypatch,
):
    client, storage = api_client

    client_id = uuid4()
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(client_id, user_id)
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
        conversation_id=conversation_id,
        user_id=user_id,
    )

    assert saved_conversation is not None
    assert len(saved_conversation["messages"]) == 2


def test_send_conversation_message_rejects_empty_message(api_client):
    client, storage = api_client

    user_id = storage.authenticated_user_id
    conversation = storage.create_conversation(uuid4(), user_id)

    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages",
        json={
            "client_id": conversation["client_id"].__str__(),
            "message": "   ",
        },
    )

    assert response.status_code == 422

    saved_conversation = storage.get_conversation(
        conversation_id=conversation["id"],
        user_id=user_id,
    )

    assert saved_conversation is not None
    assert saved_conversation["messages"] == []


def test_send_message_returns_404_for_another_user(
    api_client,
    monkeypatch,
):
    client, storage = api_client

    owner_client_id = uuid4()
    another_user_id = uuid4()

    conversation = storage.create_conversation(
        owner_client_id,
        another_user_id,
    )

    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: FakeAgentService(),
    )

    # Even knowing the owner's browser client_id must not grant access.
    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages",
        json={
            "client_id": str(owner_client_id),
            "message": "Explain AI agents.",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."

    assert conversation["id"] in storage.conversations
    assert storage.conversations[conversation["id"]]["messages"] == []


def test_claim_legacy_conversations_links_only_unowned_records(api_client):
    client, storage = api_client

    legacy_client_id = uuid4()
    authenticated_user_id = storage.authenticated_user_id
    another_user_id = uuid4()

    first_legacy = storage.create_conversation(legacy_client_id)
    second_legacy = storage.create_conversation(legacy_client_id)

    already_owned = storage.create_conversation(
        legacy_client_id,
        another_user_id,
    )

    response = client.post(
        "/api/v1/conversations/claim-legacy",
        json={"client_id": str(legacy_client_id)},
    )

    assert response.status_code == 200
    assert response.json() == {"linked_conversations": 2}

    assert storage.conversations[first_legacy["id"]]["user_id"] == authenticated_user_id
    assert (
        storage.conversations[second_legacy["id"]]["user_id"] == authenticated_user_id
    )
    assert storage.conversations[already_owned["id"]]["user_id"] == another_user_id


def test_create_conversation_rejects_missing_csrf_header(api_client):
    client, storage = api_client
    client.headers[auth_dependencies.CSRF_HEADER_NAME] = "incorrect-csrf-token"

    response = client.post(
        "/api/v1/conversations",
        json={"client_id": str(uuid4())},
    )

    assert response.status_code == 403
    assert storage.conversations == {}
