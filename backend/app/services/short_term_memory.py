"""Short-term memory service for recent conversation context."""

from uuid import UUID

from backend.app.services.chat_history import PostgresChatHistoryStorage


class ShortTermMemoryService:
    """Build recent conversation context for text generation."""

    MESSAGE_LIMIT = 20

    def __init__(
        self,
        chat_history: PostgresChatHistoryStorage,
    ) -> None:
        self._chat_history = chat_history

    def get_context(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> list[dict[str, str]]:
        """Return the latest messages in chronological order."""
        recent_messages = self._chat_history.get_recent_messages(
            conversation_id=conversation_id,
            user_id=user_id,
            limit=self.MESSAGE_LIMIT,
        )

        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in recent_messages
        ]
