"""
Sidebar components.

Displays system info, settings, and controls in the Streamlit sidebar.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import streamlit as st

from config import settings
from utils import cleanup_memory, get_memory_usage

if TYPE_CHECKING:
    from memory.conversation import ConversationMemory
    from storage.vector_store import VectorStoreInterface

logger = logging.getLogger(__name__)


def render_sidebar(
    vector_store: VectorStoreInterface,
    conversation_memory: Optional[ConversationMemory] = None,
) -> None:
    """
    Render the complete sidebar with system info and controls.

    Args:
        vector_store: Vector store instance for stats.
        conversation_memory: Conversation memory for clear action.
    """
    with st.sidebar:
        st.header("ℹ️ System Info")

        _display_stats(vector_store)
        st.markdown("---")

        _display_controls(conversation_memory)
        st.markdown("---")

        _display_settings()
        st.markdown("---")

        _display_memory_usage()


def _display_stats(vector_store: VectorStoreInterface) -> None:
    """Display vector store statistics."""
    try:
        stats = vector_store.get_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats.get("vector_count", 0))
        with col2:
            st.metric("Backend", stats.get("backend", "unknown").upper())

        with st.expander("📊 Detailed Stats"):
            st.json(stats)

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        st.warning("Unable to load stats")


def _display_controls(conversation_memory: Optional[ConversationMemory]) -> None:
    """Display control buttons for clearing chat and freeing memory."""
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if conversation_memory:
                conversation_memory.clear()
            cleanup_memory()
            st.rerun()

    with col2:
        if st.button("🧹 Free RAM", use_container_width=True):
            cleanup_memory()
            st.success("✓ Cleaned", icon="✅")


def _display_settings() -> None:
    """Display current RAG configuration settings."""
    with st.expander("⚙️ Settings"):
        st.markdown("**RAG Configuration:**")

        # Chunking
        chunking_type = "Hierarchical" if settings.USE_HIERARCHICAL_CHUNKING else "Simple"
        st.text(f"Chunking: {chunking_type}")
        if settings.USE_HIERARCHICAL_CHUNKING:
            st.text(f"  Child: {settings.CHUNK_SIZE} tokens")
            st.text(f"  Parent: {settings.PARENT_CHUNK_SIZE} tokens")
        else:
            st.text(f"  Size: {settings.CHUNK_SIZE} tokens")

        # Retrieval
        st.text(f"Top-K: {settings.SIMILARITY_TOP_K}")
        st.text(f"Hybrid Search: {'✓' if settings.USE_HYBRID_SEARCH else '✗'}")
        if settings.USE_HYBRID_SEARCH:
            st.text(f"  Alpha: {settings.HYBRID_ALPHA}")

        # Features
        st.markdown("**Features:**")
        st.text(f"Conv. Memory: {'✓' if settings.ENABLE_CONVERSATION_MEMORY else '✗'}")
        st.text(f"Citations: {'✓' if settings.ENABLE_CITATIONS else '✗'}")
        st.text(f"Metrics: {'✓' if settings.ENABLE_METRICS else '✗'}")


def _display_memory_usage() -> None:
    """Display memory usage progress bar and details."""
    if not settings.ENABLE_MEMORY_PROFILING:
        return

    mem_info = get_memory_usage()
    if not mem_info:
        return

    st.markdown("**💾 Memory Usage:**")

    memory_percent = min(mem_info["percent"] / 100, 1.0)
    st.progress(memory_percent)
    st.caption(f"{mem_info['rss_mb']:.0f}MB ({mem_info['percent']:.1f}%)")

    if mem_info["rss_mb"] > settings.AUTO_CLEANUP_THRESHOLD_MB:
        st.warning("⚠️ High memory usage", icon="⚠️")


def render_document_stats(
    doc_count: int,
    file_types: Optional[Dict[str, int]] = None,
) -> None:
    """
    Display document statistics.

    Args:
        doc_count: Total number of documents.
        file_types: Mapping of extension to count.
    """
    st.metric("Total Documents", doc_count)

    if file_types:
        with st.expander("📄 File Types"):
            for ext, count in file_types.items():
                st.text(f"{ext.upper()}: {count}")


def render_quick_actions() -> Tuple[bool, bool]:
    """
    Render quick action buttons.

    Returns:
        Tuple of (refresh_clicked, upload_clicked).
    """
    st.markdown("### ⚡ Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        refresh = st.button("🔄 Refresh Index", use_container_width=True)
    with col2:
        upload = st.button("📤 Upload Docs", use_container_width=True)

    return refresh, upload


def render_help_section() -> None:
    """Render help and tips section."""
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        **Query Tips:**
        - Be specific in your questions
        - Use follow-up questions for context
        - Check sources for detailed info
        
        **Features:**
        - 💬 Conversation memory preserves context
        - 📚 Citations link to source documents
        - 🔍 Hybrid search combines semantic + keyword
        
        **Controls:**
        - Clear Chat: Reset conversation
        - Free RAM: Clean up memory
        """)


def render_metrics_link() -> None:
    """Render link to Prometheus metrics endpoint."""
    if settings.ENABLE_METRICS:
        st.markdown(
            f"[📊 View Metrics](http://localhost:{settings.METRICS_PORT}/metrics)"
        )