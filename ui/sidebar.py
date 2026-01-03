"""
Sidebar components.
Displays system info, settings, and controls.
"""

import logging
import streamlit as st
from config import settings
from utils import get_memory_usage, cleanup_memory

logger = logging.getLogger(__name__)


def render_sidebar(vector_store, conversation_memory=None):
    """
    Render complete sidebar.

    Args:
        vector_store: Vector store instance
        conversation_memory: Conversation memory instance
    """
    with st.sidebar:
        st.header("ℹ️ System Info")

        # Display stats
        _display_stats(vector_store)

        st.markdown("---")

        # Control buttons
        _display_controls(conversation_memory)

        st.markdown("---")

        # Settings display
        _display_settings()

        st.markdown("---")

        # Memory usage
        _display_memory_usage()


def _display_stats(vector_store):
    """Display system statistics."""
    try:
        stats = vector_store.get_stats()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Documents",
                stats.get("vector_count", 0)
            )

        with col2:
            st.metric(
                "Backend",
                stats.get("backend", "unknown").upper()
            )

        # Additional stats in expander
        with st.expander("📊 Detailed Stats"):
            st.json(stats)

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        st.warning("Unable to load stats")


def _display_controls(conversation_memory):
    """Display control buttons."""
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


def _display_settings():
    """Display current settings."""
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


def _display_memory_usage():
    """Display memory usage."""
    if not settings.ENABLE_MEMORY_PROFILING:
        return

    mem_info = get_memory_usage()
    if not mem_info:
        return

    st.markdown("**💾 Memory Usage:**")

    # Progress bar
    memory_percent = min(mem_info["percent"] / 100, 1.0)
    st.progress(memory_percent)

    # Details
    st.caption(f"{mem_info['rss_mb']:.0f}MB ({mem_info['percent']:.1f}%)")

    # Warning if high
    if mem_info["rss_mb"] > settings.AUTO_CLEANUP_THRESHOLD_MB:
        st.warning("⚠️ High memory usage", icon="⚠️")


def render_document_stats(doc_count: int, file_types: dict = None):
    """
    Display document statistics.

    Args:
        doc_count: Number of documents
        file_types: Dictionary of file type counts
    """
    st.metric("Total Documents", doc_count)

    if file_types:
        with st.expander("📄 File Types"):
            for ext, count in file_types.items():
                st.text(f"{ext.upper()}: {count}")


def render_quick_actions():
    """Render quick action buttons."""
    st.markdown("### ⚡ Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        refresh = st.button("🔄 Refresh Index", use_container_width=True)

    with col2:
        upload = st.button("📤 Upload Docs", use_container_width=True)

    return refresh, upload


def render_help_section():
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


def render_metrics_link():
    """Render link to metrics."""
    if settings.ENABLE_METRICS:
        st.markdown(
            f"[📊 View Metrics](http://localhost:{settings.METRICS_PORT}/metrics)"
        )