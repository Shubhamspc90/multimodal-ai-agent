"""Unit tests for short-term conversation memory."""

from datetime import UTC, datetime
from uuid import uuid4

from backend.app.services.short_term_memory import ShortTermMemoryService


class FakeChatHistoryStorage:
    """Provide deterministic recent messages for testing."""

    def __init__(self, messages: list[dict]) -> None:
        self.messages = messages
        self.call_arguments = None

    def get_recent_messages(
        self,
        conversation_id,
        user_id,
        limit=20,
    ):
        self.call_arguments = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "limit": limit,
        }
        return self.messages[-limit:]


def test_get_context_returns_recent_messages_with_roles():
    conversation_id = uuid4()
    user_id = uuid4()
    timestamp = datetime.now(UTC)

    messages = [
        {
            "id": 1,
            "role": "user",
            "content": "Explain PostgreSQL.",
            "created_at": timestamp,
        },
        {
            "id": 2,
            "role": "assistant",
            "content": "PostgreSQL is a relational database.",
            "created_at": timestamp,
        },
    ]

    storage = FakeChatHistoryStorage(messages)
    memory = ShortTermMemoryService(storage)

    result = memory.get_context(conversation_id, user_id)

    assert result == [
        {
            "role": "user",
            "content": "Explain PostgreSQL.",
        },
        {
            "role": "assistant",
            "content": "PostgreSQL is a relational database.",
        },
    ]

    assert storage.call_arguments == {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "limit": 20,
    }


def test_get_context_returns_empty_list_for_new_conversation():
    storage = FakeChatHistoryStorage([])
    memory = ShortTermMemoryService(storage)

    result = memory.get_context(uuid4(), uuid4())

    assert result == []


def test_get_context_preserves_only_the_latest_twenty_messages():
    timestamp = datetime.now(UTC)

    messages = [
        {
            "id": index,
            "role": "user" if index % 2 else "assistant",
            "content": f"Message {index}",
            "created_at": timestamp,
        }
        for index in range(1, 26)
    ]

    storage = FakeChatHistoryStorage(messages)
    memory = ShortTermMemoryService(storage)

    result = memory.get_context(uuid4(), uuid4())

    assert len(result) == 20
    assert result[0]["content"] == "Message 6"
    assert result[-1]["content"] == "Message 25"
    assert all(set(message) == {"role", "content"} for message in result)
