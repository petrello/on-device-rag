"""
Chat interface components.
Handles chat UI rendering and interactions.
"""

import logging
import streamlit as st
from typing import List, Dict

logger = logging.getLogger(__name__)


def display_chat_history(messages: List[Dict], citation_extractor=None):
    """
    Display chat message history.

    Args:
        messages: List of message dictionaries
        citation_extractor: Citation extractor for formatting
    """
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display citations if present
            if "citations" in message and message["citations"]:
                with st.expander("📚 View Sources", expanded=False):
                    if citation_extractor:
                        for i, citation in enumerate(message["citations"], 1):
                            st.markdown(
                                citation_extractor.format_citation_markdown(citation, i),
                                unsafe_allow_html=True
                            )
                    else:
                        # Fallback formatting
                        for i, citation in enumerate(message["citations"], 1):
                            st.markdown(f"**[{i}]** {citation.get('source', 'Unknown')}")


def render_chat_input() -> str:
    """
    Render chat input box.

    Returns:
        User input text or empty string
    """
    prompt = st.chat_input("Ask a question about your documents...")
    return prompt or ""


def display_welcome_message():
    """Display welcome message for new users."""
    st.markdown("""
    ### 👋 Welcome to Edge RAG!
    
    I can help you find information in your documents. Try asking:
    - "What is this document about?"
    - "Summarize the main points"
    - "Find information about [specific topic]"
    
    **Tips:**
    - Be specific in your questions
    - Follow-up questions work great with conversation memory
    - Check the sources for detailed information
    """)


def display_thinking_indicator():
    """Display thinking/processing indicator."""
    return st.empty()


def format_streaming_response(response_generator):
    """
    Format and display streaming response.

    Args:
        response_generator: Response generator from query engine

    Returns:
        Complete response text
    """
    return st.write_stream(response_generator)


def display_error_message(error: Exception):
    """
    Display formatted error message.

    Args:
        error: Exception object
    """
    error_type = type(error).__name__
    error_msg = str(error)

    st.error(f"⚠️ **{error_type}:** {error_msg}")

    with st.expander("Debug Information"):
        st.code(f"{error_type}: {error_msg}")
        logger.error(f"Chat error: {error}", exc_info=True)


def display_query_info(query: str, context_used: bool = False):
    """
    Display information about the query.

    Args:
        query: User query
        context_used: Whether conversation context was used
    """
    info_col, _ = st.columns([3, 1])

    with info_col:
        if context_used:
            st.caption("💬 Using conversation context")
        st.caption(f"Query length: {len(query)} characters")


def create_feedback_buttons(message_index: int):
    """
    Create feedback buttons for a message.

    Args:
        message_index: Index of message in history

    Returns:
        Tuple of (helpful, not_helpful) button states
    """
    col1, col2, _ = st.columns([1, 1, 8])

    with col1:
        helpful = st.button("👍", key=f"helpful_{message_index}")

    with col2:
        not_helpful = st.button("👎", key=f"not_helpful_{message_index}")

    return helpful, not_helpful


def display_token_count(response_text: str):
    """
    Display approximate token count for response.

    Args:
        response_text: Response text
    """
    # Rough approximation: 1 token ≈ 4 characters
    approx_tokens = len(response_text) // 4
    st.caption(f"Response: ~{approx_tokens} tokens")


def clear_chat_button():
    """
    Render clear chat button.

    Returns:
        True if button was clicked
    """
    return st.button("🗑️ Clear Chat", use_container_width=True)