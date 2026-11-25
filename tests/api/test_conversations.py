"""API tests for authenticated persistent chat conversations."""

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app.api import dependencies as auth_dependencies
from backend.app.api.routes import conversations as conversations_route
from backend.app.main import app
from backend.app.schemas.a2ui import A2UI_BASIC_CATALOG_ID
from backend.app.schemas.agent import AgentInputType, AgentRequest, AgentResponse
from backend.app.services.diagram_prompt import (
    DIAGRAM_GENERATION_SYSTEM_PROMPT,
)
from backend.app.services.ollama_provider import OllamaProviderError


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
            "pinned_at": None,
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
            key=lambda item: (
                item["pinned_at"] is None,
                item["pinned_at"] or datetime.max.replace(tzinfo=UTC),
                -item["updated_at"].timestamp(),
                -item["id"].int,
            ),
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

    def get_recent_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        limit: int = 20,
    ) -> list[dict]:
        """Return the latest messages for an authenticated conversation."""
        if limit < 1:
            raise ValueError("Message limit must be at least 1.")

        conversation = self.conversations.get(conversation_id)

        if conversation is None or conversation["user_id"] != user_id:
            return []

        return [
            {
                "id": message["id"],
                "role": message["role"],
                "content": message["content"],
                "created_at": message["created_at"],
            }
            for message in conversation["messages"][-limit:]
        ]

    def pin_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> dict | None:
        """Pin a conversation belonging to the authenticated user."""
        conversation = self.conversations.get(conversation_id)

        if conversation is None or conversation["user_id"] != user_id:
            return None

        if conversation["pinned_at"] is None:
            latest_pin = max(
                (
                    item["pinned_at"]
                    for item in self.conversations.values()
                    if item["user_id"] == user_id and item["pinned_at"] is not None
                ),
                default=None,
            )

            timestamp = datetime.now(UTC)

            if latest_pin is not None and timestamp <= latest_pin:
                from datetime import timedelta

                timestamp = latest_pin + timedelta(microseconds=1)

            conversation["pinned_at"] = timestamp

        return self._summary(conversation)

    def unpin_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> dict | None:
        """Unpin a conversation belonging to the authenticated user."""
        conversation = self.conversations.get(conversation_id)

        if conversation is None or conversation["user_id"] != user_id:
            return None

        conversation["pinned_at"] = None

        return self._summary(conversation)

    def delete_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> bool:
        """Delete a conversation only when it belongs to the authenticated user."""
        conversation = self.conversations.get(conversation_id)

        if conversation is None or conversation["user_id"] != user_id:
            return False

        del self.conversations[conversation_id]
        return True

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
            "pinned_at": conversation.get("pinned_at"),
        }


class FakeAgentService:
    """Generate deterministic AI responses without calling Ollama."""

    def __init__(self) -> None:
        self.received_messages: list[dict[str, str]] | None = None

    async def process_text(
        self,
        request: AgentRequest,
        messages: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        self.received_messages = messages

        return AgentResponse(
            status="completed",
            input_type=AgentInputType.TEXT,
            response=f"AI-generated answer to: {request.message}",
            capability="text",
        )


class FakeLongTermMemoryRetriever:
    """Return no memories by default in conversation API tests."""

    def retrieve_memories(
        self,
        user_id: UUID,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        return []


class FakeLongTermMemoryCaptureService:
    """Skip automatic memory capture in unrelated API tests."""

    async def capture_memories(
        self,
        user_id: UUID,
        messages: list[dict[str, str]],
    ) -> list[dict]:
        return []


@pytest.fixture
def api_client(monkeypatch):
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

    monkeypatch.setattr(
        conversations_route,
        "get_long_term_memory_retriever",
        lambda: FakeLongTermMemoryRetriever(),
    )

    monkeypatch.setattr(
        conversations_route,
        "get_long_term_memory_capture_service",
        lambda agent_service: FakeLongTermMemoryCaptureService(),
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

    agent_service = FakeAgentService()
    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: agent_service,
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

    received_messages = agent_service.received_messages

    assert received_messages is not None
    assert received_messages[0] == {
        "role": "system",
        "content": DIAGRAM_GENERATION_SYSTEM_PROMPT,
    }
    assert received_messages[1:] == [
        {
            "role": "user",
            "content": "Explain AI agents.",
        },
    ]


def test_send_conversation_message_saves_valid_a2ui_response_and_captures_readable_text(
    api_client,
    monkeypatch,
):
    """Persist validated A2UI output without sending protocol JSON to memory."""
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    messages = [
        {
            "version": "1.0",
            "createSurface": {
                "surfaceId": "surface-1",
                "catalogId": A2UI_BASIC_CATALOG_ID,
                "sendDataModel": False,
            },
        },
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": "root",
                        "component": "Column",
                        "children": ["heading", "body"],
                    },
                    {
                        "id": "heading",
                        "component": "Text",
                        "text": "Welcome",
                        "variant": "body",
                    },
                    {
                        "id": "body",
                        "component": "Text",
                        "text": "A2UI is ready.",
                    },
                ],
            },
        },
    ]

    readable_text = "Here is your structured summary."
    provider_response = f"{readable_text}\n\n```a2ui-json\n{json.dumps(messages)}\n```"

    class A2UIAgentService:
        async def process_text(self, request, messages=None):
            return AgentResponse(
                status="completed",
                input_type=AgentInputType.TEXT,
                response=provider_response,
                capability="text",
            )

    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        A2UIAgentService,
    )

    class RecordingCaptureService:
        def __init__(self):
            self.received_messages = None

        async def capture_memories(self, user_id, messages):
            self.received_messages = messages
            return []

    capture_service = RecordingCaptureService()

    monkeypatch.setattr(
        conversations_route,
        "get_long_term_memory_capture_service",
        lambda agent_service: capture_service,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(conversation["client_id"]),
            "message": "Show me a structured summary.",
        },
    )

    assert response.status_code == 200

    result = response.json()
    stored_content = result["assistant_message"]["content"]
    envelope = json.loads(stored_content)

    assert envelope == {
        "type": "a2ui",
        "version": "1.0",
        "text": readable_text,
        "messages": messages,
    }

    saved_conversation = storage.get_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
    )

    assert saved_conversation is not None
    assert saved_conversation["messages"][-1]["content"] == stored_content

    captured_messages = capture_service.received_messages

    assert captured_messages is not None
    assert captured_messages[-1] == {
        "role": "assistant",
        "content": readable_text,
    }
    assert "a2ui-json" not in captured_messages[-1]["content"]
    assert '"createSurface"' not in captured_messages[-1]["content"]


def test_send_conversation_message_falls_back_to_markdown_for_invalid_a2ui(
    api_client,
    monkeypatch,
):
    """Preserve the original response when A2UI validation fails."""
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    provider_response = (
        "Here is the summary.\n\n"
        '```a2ui-json\n[{"version":"0.9","deleteSurface":{"surfaceId":"s1"}}]\n```'
    )

    class InvalidA2UIAgentService:
        async def process_text(self, request, messages=None):
            return AgentResponse(
                status="completed",
                input_type=AgentInputType.TEXT,
                response=provider_response,
                capability="text",
            )

    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        InvalidA2UIAgentService,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(conversation["client_id"]),
            "message": "Show me a structured summary.",
        },
    )

    assert response.status_code == 200
    assert response.json()["assistant_message"]["content"] == provider_response

    saved_conversation = storage.get_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
    )

    assert saved_conversation is not None
    assert saved_conversation["messages"][-1]["content"] == provider_response


def _assert_a2ui_fallback_response_is_persisted(
    api_client,
    monkeypatch,
    provider_response: str,
) -> None:
    """Verify fallback responses are returned and persisted unchanged."""
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    class FallbackA2UIAgentService:
        async def process_text(self, request, messages=None):
            return AgentResponse(
                status="completed",
                input_type=AgentInputType.TEXT,
                response=provider_response,
                capability="text",
            )

    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        FallbackA2UIAgentService,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(conversation["client_id"]),
            "message": "Show me a response.",
        },
    )

    assert response.status_code == 200
    assert response.json()["assistant_message"]["content"] == provider_response

    saved_conversation = storage.get_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
    )

    assert saved_conversation is not None
    assert saved_conversation["messages"][-1]["content"] == provider_response


def test_send_conversation_message_preserves_malformed_a2ui_json(
    api_client,
    monkeypatch,
):
    """Preserve the original response when an A2UI block contains malformed JSON."""
    provider_response = (
        'Here is the summary.\n\n```a2ui-json\n{"version": "1.0", invalid-json}\n```'
    )

    _assert_a2ui_fallback_response_is_persisted(
        api_client,
        monkeypatch,
        provider_response,
    )


def test_send_conversation_message_preserves_multiple_a2ui_blocks(
    api_client,
    monkeypatch,
):
    """Preserve the original response when multiple A2UI blocks are present."""
    provider_response = (
        "Here are two structured sections.\n\n"
        "```a2ui-json\n[]\n```\n\n"
        "```a2ui-json\n[]\n```"
    )

    _assert_a2ui_fallback_response_is_persisted(
        api_client,
        monkeypatch,
        provider_response,
    )


def test_send_conversation_message_preserves_ordinary_markdown_without_a2ui(
    api_client,
    monkeypatch,
):
    """Preserve ordinary Markdown responses without A2UI processing."""
    provider_response = (
        "# Summary\n\n"
        "This is a normal Markdown response.\n\n"
        "- First point\n"
        "- Second point\n\n"
        "```python\n"
        "print('Hello, world!')\n"
        "```"
    )

    _assert_a2ui_fallback_response_is_persisted(
        api_client,
        monkeypatch,
        provider_response,
    )


def test_send_conversation_message_includes_relevant_long_term_memories(
    api_client,
    monkeypatch,
):
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    agent_service = FakeAgentService()

    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: agent_service,
    )

    class RecordingRetriever:
        def __init__(self):
            self.received_user_id = None
            self.received_query = None

        def retrieve_memories(
            self,
            user_id: UUID,
            query: str,
            limit: int = 5,
        ) -> list[dict]:
            self.received_user_id = user_id
            self.received_query = query

            return [
                {
                    "category": "preference",
                    "memory_text": "Prefers concise technical explanations.",
                },
            ]

    retriever = RecordingRetriever()

    monkeypatch.setattr(
        conversations_route,
        "get_long_term_memory_retriever",
        lambda: retriever,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(conversation["client_id"]),
            "message": "Explain AI agents.",
        },
    )

    assert response.status_code == 200

    assert retriever.received_user_id == user_id
    assert retriever.received_query == "Explain AI agents."

    received_messages = agent_service.received_messages

    assert received_messages is not None

    # Diagram instructions and long-term memories remain separate.
    assert received_messages[0] == {
        "role": "system",
        "content": DIAGRAM_GENERATION_SYSTEM_PROMPT,
    }

    assert received_messages[1]["role"] == "system"
    assert "untrusted reference data" in received_messages[1]["content"]
    assert "Prefers concise technical explanations." in received_messages[1]["content"]

    assert received_messages[2:] == [
        {
            "role": "user",
            "content": "Explain AI agents.",
        },
    ]

    saved_conversation = storage.get_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
    )

    assert saved_conversation is not None
    assert len(saved_conversation["messages"]) == 2


def test_send_conversation_message_continues_when_memory_retrieval_fails(
    api_client,
    monkeypatch,
):
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    agent_service = FakeAgentService()

    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: agent_service,
    )

    class FailingRetriever:
        def retrieve_memories(
            self,
            user_id: UUID,
            query: str,
            limit: int = 5,
        ) -> list[dict]:
            raise RuntimeError("Simulated memory retrieval failure.")

    monkeypatch.setattr(
        conversations_route,
        "get_long_term_memory_retriever",
        lambda: FailingRetriever(),
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(conversation["client_id"]),
            "message": "Explain AI agents.",
        },
    )

    assert response.status_code == 200

    received_messages = agent_service.received_messages

    assert received_messages is not None
    assert received_messages[0] == {
        "role": "system",
        "content": DIAGRAM_GENERATION_SYSTEM_PROMPT,
    }
    assert received_messages[1:] == [
        {
            "role": "user",
            "content": "Explain AI agents.",
        },
    ]

    saved_conversation = storage.get_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
    )

    assert saved_conversation is not None
    assert len(saved_conversation["messages"]) == 2
    assert saved_conversation["messages"][0]["role"] == "user"
    assert saved_conversation["messages"][1]["role"] == "assistant"


def test_send_conversation_message_preserves_user_message_on_provider_failure(
    api_client,
    monkeypatch,
):
    client, storage = api_client

    user_id = storage.authenticated_user_id
    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    class FailingAgentService:
        async def process_text(
            self,
            request: AgentRequest,
            messages: list[dict[str, str]] | None = None,
        ) -> AgentResponse:
            raise OllamaProviderError("Simulated provider failure.")

    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: FailingAgentService(),
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(conversation["client_id"]),
            "message": "Explain AI agents.",
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The text generation provider is temporarily unavailable."
    )

    saved_conversation = storage.get_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
    )

    assert saved_conversation is not None
    assert len(saved_conversation["messages"]) == 1
    assert saved_conversation["messages"][0]["role"] == "user"
    assert saved_conversation["messages"][0]["content"] == "Explain AI agents."


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


def test_delete_conversation_returns_no_content_and_removes_record(api_client):
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    storage.append_message(
        conversation_id,
        None,
        "user",
        "Test message",
        user_id=user_id,
    )

    response = client.delete(
        f"/api/v1/conversations/{conversation_id}",
    )

    assert response.status_code == 204
    assert response.content == b""
    assert conversation_id not in storage.conversations


def test_delete_conversation_returns_404_for_another_user(api_client):
    client, storage = api_client

    conversation = storage.create_conversation(
        uuid4(),
        uuid4(),
    )

    response = client.delete(
        f"/api/v1/conversations/{conversation['id']}",
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."
    assert conversation["id"] in storage.conversations


def test_delete_conversation_returns_404_for_unknown_id(api_client):
    client, storage = api_client

    response = client.delete(
        f"/api/v1/conversations/{uuid4()}",
    )

    assert response.status_code == 404
    assert storage.conversations == {}


def test_delete_conversation_rejects_invalid_csrf_token(api_client):
    client, storage = api_client

    conversation = storage.create_conversation(
        uuid4(),
        storage.authenticated_user_id,
    )

    client.headers[auth_dependencies.CSRF_HEADER_NAME] = "incorrect-csrf-token"

    response = client.delete(
        f"/api/v1/conversations/{conversation['id']}",
    )

    assert response.status_code == 403
    assert conversation["id"] in storage.conversations


def test_send_conversation_message_uses_only_latest_twenty_messages(
    api_client,
    monkeypatch,
):
    """Send only the latest 20 persisted messages to the agent."""
    client, storage = api_client

    user_id = storage.authenticated_user_id
    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    previous_messages = []

    for index in range(22):
        role = "user" if index % 2 == 0 else "assistant"
        content = f"Historical message {index + 1}"

        storage.append_message(
            conversation_id,
            None,
            role,
            content,
            user_id=user_id,
        )

        previous_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    agent_service = FakeAgentService()
    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: agent_service,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(conversation["client_id"]),
            "message": "What happened earlier?",
        },
    )

    assert response.status_code == 200

    expected_context = previous_messages[-19:] + [
        {
            "role": "user",
            "content": "What happened earlier?",
        },
    ]

    received_messages = agent_service.received_messages

    assert received_messages is not None
    assert received_messages[0] == {
        "role": "system",
        "content": DIAGRAM_GENERATION_SYSTEM_PROMPT,
    }

    # All twenty original conversation messages must remain unchanged.
    assert received_messages[1:] == expected_context
    assert len(received_messages[1:]) == 20
    assert len(received_messages) == 21


def test_send_conversation_message_captures_only_conversation_messages(
    api_client,
    monkeypatch,
):
    """Capture conversation messages without injected system memory."""
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    agent_service = FakeAgentService()
    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: agent_service,
    )

    class RecordingCaptureService:
        def __init__(self):
            self.received_user_id = None
            self.received_messages = None

        async def capture_memories(self, user_id, messages):
            self.received_user_id = user_id
            self.received_messages = messages
            return []

    capture_service = RecordingCaptureService()

    monkeypatch.setattr(
        conversations_route,
        "get_long_term_memory_capture_service",
        lambda agent_service: capture_service,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(conversation["client_id"]),
            "message": "My favourite programming language is Python.",
        },
    )

    assert response.status_code == 200
    assert capture_service.received_user_id == user_id

    captured_messages = capture_service.received_messages

    assert captured_messages is not None
    assert captured_messages[0] == {
        "role": "user",
        "content": "My favourite programming language is Python.",
    }
    assert all(
        message["role"] in {"user", "assistant"} for message in captured_messages
    )
    assert captured_messages[-1]["role"] == "assistant"


def test_send_conversation_message_excludes_retrieved_memories_from_capture(
    api_client,
    monkeypatch,
):
    """Retrieved system memories must not enter automatic memory capture."""
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    agent_service = FakeAgentService()
    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: agent_service,
    )

    class MemoryReturningRetriever:
        def retrieve_memories(self, user_id, query, limit=5):
            return [
                {
                    "memory_text": "I prefer Python",
                    "category": "preference",
                    "distance": 0.1,
                }
            ]

    monkeypatch.setattr(
        conversations_route,
        "get_long_term_memory_retriever",
        lambda: MemoryReturningRetriever(),
    )

    class RecordingCaptureService:
        def __init__(self):
            self.received_user_id = None
            self.received_messages = None

        async def capture_memories(self, user_id, messages):
            self.received_user_id = user_id
            self.received_messages = messages
            return []

    capture_service = RecordingCaptureService()

    monkeypatch.setattr(
        conversations_route,
        "get_long_term_memory_capture_service",
        lambda agent_service: capture_service,
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(conversation["client_id"]),
            "message": "What programming language do I prefer?",
        },
    )

    assert response.status_code == 200

    # The assistant must receive the retrieved memory as context.
    assert agent_service.received_messages is not None
    assert any(
        "I prefer Python" in message["content"]
        for message in agent_service.received_messages
    )

    # Capture must receive only the actual conversation messages.
    assert capture_service.received_user_id == user_id
    assert capture_service.received_messages is not None
    assert all(
        message["role"] in {"user", "assistant"}
        for message in capture_service.received_messages
    )
    assert not any(
        "I prefer Python" in message["content"]
        for message in capture_service.received_messages
    )


def test_send_conversation_message_continues_when_memory_capture_fails(
    api_client,
    monkeypatch,
):
    """A memory capture failure must not fail a successful chat response."""
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)
    conversation_id = conversation["id"]

    monkeypatch.setattr(
        conversations_route,
        "get_agent_service",
        lambda: FakeAgentService(),
    )

    class FailingCaptureService:
        async def capture_memories(self, user_id, messages):
            raise RuntimeError("Simulated memory capture failure.")

    monkeypatch.setattr(
        conversations_route,
        "get_long_term_memory_capture_service",
        lambda agent_service: FailingCaptureService(),
    )

    response = client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "client_id": str(conversation["client_id"]),
            "message": "I prefer PostgreSQL.",
        },
    )

    assert response.status_code == 200

    saved_conversation = storage.get_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
    )

    assert saved_conversation is not None
    assert len(saved_conversation["messages"]) == 2
    assert saved_conversation["messages"][0]["role"] == "user"
    assert saved_conversation["messages"][1]["role"] == "assistant"


def test_pin_conversation_returns_pinned_conversation(api_client):
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)

    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/pin",
    )

    assert response.status_code == 200

    result = response.json()
    assert result["id"] == str(conversation["id"])
    assert result["pinned_at"] is not None
    assert storage.conversations[conversation["id"]]["pinned_at"] is not None


def test_unpin_conversation_returns_unpinned_conversation(api_client):
    client, storage = api_client
    user_id = storage.authenticated_user_id

    conversation = storage.create_conversation(uuid4(), user_id)
    storage.pin_conversation(conversation["id"], user_id)

    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/unpin",
    )

    assert response.status_code == 200
    assert response.json()["pinned_at"] is None
    assert storage.conversations[conversation["id"]]["pinned_at"] is None


def test_pin_conversation_returns_404_for_another_user(api_client):
    client, storage = api_client

    conversation = storage.create_conversation(uuid4(), uuid4())

    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/pin",
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."
    assert storage.conversations[conversation["id"]]["pinned_at"] is None


def test_unpin_conversation_returns_404_for_another_user(api_client):
    client, storage = api_client

    conversation = storage.create_conversation(uuid4(), uuid4())

    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/unpin",
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_pin_conversation_rejects_invalid_csrf_token(api_client):
    client, storage = api_client

    conversation = storage.create_conversation(
        uuid4(),
        storage.authenticated_user_id,
    )

    client.headers[auth_dependencies.CSRF_HEADER_NAME] = "incorrect-csrf-token"

    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/pin",
    )

    assert response.status_code == 403
    assert storage.conversations[conversation["id"]]["pinned_at"] is None


def test_unpin_conversation_rejects_invalid_csrf_token(api_client):
    client, storage = api_client

    conversation = storage.create_conversation(
        uuid4(),
        storage.authenticated_user_id,
    )
    storage.pin_conversation(
        conversation["id"],
        storage.authenticated_user_id,
    )

    client.headers[auth_dependencies.CSRF_HEADER_NAME] = "incorrect-csrf-token"

    response = client.post(
        f"/api/v1/conversations/{conversation['id']}/unpin",
    )

    assert response.status_code == 403
    assert storage.conversations[conversation["id"]]["pinned_at"] is not None


def test_list_conversations_places_pinned_conversations_first(api_client):
    client, storage = api_client
    user_id = storage.authenticated_user_id

    first = storage.create_conversation(uuid4(), user_id)
    second = storage.create_conversation(uuid4(), user_id)
    recent = storage.create_conversation(uuid4(), user_id)

    storage.pin_conversation(first["id"], user_id)
    storage.pin_conversation(second["id"], user_id)

    response = client.get("/api/v1/conversations")

    assert response.status_code == 200

    results = response.json()
    assert [item["id"] for item in results] == [
        str(first["id"]),
        str(second["id"]),
        str(recent["id"]),
    ]
