"""
Chat interface components.

Handles chat UI rendering and user interactions in Streamlit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

import streamlit as st

if TYPE_CHECKING:
    from citation.extractor import CitationExtractor

logger = logging.getLogger(__name__)


def display_chat_history(
    messages: List[Dict[str, Any]],
    citation_extractor: Optional[CitationExtractor] = None,
) -> None:
    """
    Display chat message history with optional citations.

    Args:
        messages: List of message dicts with 'role', 'content', and optionally 'citations'.
        citation_extractor: Extractor for formatting citation markdown.
    """
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "citations" in message and message["citations"]:
                with st.expander("📚 View Sources", expanded=False):
                    if citation_extractor:
                        for i, citation in enumerate(message["citations"], 1):
                            st.markdown(
                                citation_extractor.format_citation_markdown(citation, i),
                                unsafe_allow_html=True,
                            )
                    else:
                        for i, citation in enumerate(message["citations"], 1):
                            st.markdown(f"**[{i}]** {citation.get('source', 'Unknown')}")


def render_chat_input() -> str:
    """
    Render the chat input box.

    Returns:
        User input text, or empty string if none.
    """
    prompt = st.chat_input("Ask a question about your documents...")
    return prompt or ""


def display_welcome_message() -> None:
    """Display welcome message for new users."""
    st.markdown("""
    ### 👋 Welcome to On-Device RAG!
    
    I can help you find information in your documents. Try asking:
    - "What is this document about?"
    - "Summarize the main points"
    - "Find information about [specific topic]"
    
    **Tips:**
    - Be specific in your questions
    - Follow-up questions work great with conversation memory
    - Check the sources for detailed information
    """)


def display_thinking_indicator() -> st.delta_generator.DeltaGenerator:
    """
    Display a placeholder for thinking/processing indicator.

    Returns:
        Streamlit empty container for dynamic updates.
    """
    return st.empty()


def format_streaming_response(response_generator: Generator[str, None, None]) -> str:
    """
    Format and display a streaming response.

    Args:
        response_generator: Generator yielding response tokens.

    Returns:
        Complete response text.
    """
    return st.write_stream(response_generator)


def display_error_message(error: Exception) -> None:
    """
    Display a formatted error message.

    Args:
        error: Exception that occurred.
    """
    error_type = type(error).__name__
    error_msg = str(error)

    st.error(f"⚠️ **{error_type}:** {error_msg}")

    with st.expander("Debug Information"):
        st.code(f"{error_type}: {error_msg}")
        logger.error(f"Chat error: {error}", exc_info=True)


def display_query_info(query: str, context_used: bool = False) -> None:
    """
    Display information about the current query.

    Args:
        query: The user query.
        context_used: Whether conversation context was used.
    """
    info_col, _ = st.columns([3, 1])

    with info_col:
        if context_used:
            st.caption("💬 Using conversation context")
        st.caption(f"Query length: {len(query)} characters")


def create_feedback_buttons(message_index: int) -> tuple[bool, bool]:
    """
    Create feedback buttons for a message.

    Args:
        message_index: Index of the message in history.

    Returns:
        Tuple of (helpful_clicked, not_helpful_clicked).
    """
    col1, col2, _ = st.columns([1, 1, 8])

    with col1:
        helpful = st.button("👍", key=f"helpful_{message_index}")
    with col2:
        not_helpful = st.button("👎", key=f"not_helpful_{message_index}")

    return helpful, not_helpful


def display_token_count(response_text: str) -> None:
    """
    Display approximate token count for a response.

    Args:
        response_text: The response text.
    """
    # Rough approximation: 1 token ≈ 4 characters
    approx_tokens = len(response_text) // 4
    st.caption(f"Response: ~{approx_tokens} tokens")


def clear_chat_button() -> bool:
    """
    Render a clear chat button.

    Returns:
        True if button was clicked.
    """
    return st.button("🗑️ Clear Chat", use_container_width=True)