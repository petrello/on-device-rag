"""Conversational memory management."""

import logging
from collections import deque
from typing import List, Dict
from config import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversational context with sliding window."""

    def __init__(self, window_size: int = None):
        """
        Initialize conversation memory.

        Args:
            window_size: Number of exchanges to keep in context
        """
        self.window_size = window_size or settings.MEMORY_WINDOW
        # Store pairs of (user, assistant) messages
        self.history = deque(maxlen=self.window_size * 2)
        logger.info(f"Initialized conversation memory with window={self.window_size}")

    def add_exchange(self, user_msg: str, assistant_msg: str):
        """
        Add a conversation exchange.

        Args:
            user_msg: User's message
            assistant_msg: Assistant's response
        """
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})
        logger.debug(f"Added exchange. History size: {len(self.history)}")

    def get_context(self) -> str:
        """
        Get conversation context as formatted string.

        Returns:
            Formatted conversation history
        """
        if not self.history:
            return ""

        context_parts = []
        for msg in self.history:
            role = msg["role"].capitalize()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def get_messages(self) -> List[Dict]:
        """
        Get raw message history.

        Returns:
            List of message dicts
        """
        return list(self.history)

    def clear(self):
        """Clear conversation history."""
        self.history.clear()
        logger.info("Conversation history cleared")

    def is_empty(self) -> bool:
        """Check if history is empty."""
        return len(self.history) == 0