"""
Conversational memory management.

Provides a sliding-window memory for multi-turn conversations,
enabling context-aware responses without unbounded memory growth.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversational context with a fixed-size sliding window.

    Stores recent user/assistant exchanges and provides formatted context
    for query enhancement. Memory is automatically pruned to stay within
    the configured window size.

    Attributes:
        window_size: Number of exchange pairs to retain.
        history: Deque of message dictionaries.
    """

    __slots__ = ('window_size', 'history')

    def __init__(self, window_size: Optional[int] = None) -> None:
        """
        Initialize conversation memory.

        Args:
            window_size: Number of exchanges to keep. Defaults to settings.MEMORY_WINDOW.
        """
        self.window_size: int = window_size or settings.MEMORY_WINDOW
        # Each exchange = 2 messages (user + assistant)
        self.history: Deque[Dict[str, str]] = deque(maxlen=self.window_size * 2)
        logger.info(f"Conversation memory initialized (window={self.window_size})")

    def add_exchange(self, user_msg: str, assistant_msg: str) -> None:
        """
        Add a conversation exchange.

        Args:
            user_msg: The user's message.
            assistant_msg: The assistant's response.
        """
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})
        logger.debug(f"Exchange added. History size: {len(self.history)}")

    def get_context(self) -> str:
        """
        Get conversation context as a formatted string.

        Returns:
            Formatted conversation history, or empty string if no history.
        """
        if not self.history:
            return ""

        context_parts = [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in self.history
        ]
        return "\n".join(context_parts)

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get raw message history.

        Returns:
            List of message dictionaries with 'role' and 'content'.
        """
        return list(self.history)

    def clear(self) -> None:
        """Clear all conversation history."""
        self.history.clear()
        logger.info("Conversation history cleared")

    def is_empty(self) -> bool:
        """Check if history is empty."""
        return len(self.history) == 0

    def __len__(self) -> int:
        """Return number of messages in history."""
        return len(self.history)
