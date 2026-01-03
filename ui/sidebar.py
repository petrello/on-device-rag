"""Sidebar helpers for Streamlit UI."""

import streamlit as st
from monitoring.dashboard import render_metrics
from config import settings
from utils import get_memory_usage, cleanup_memory


def render_sidebar(vector_store_stats: dict | None = None):
    """Render common sidebar elements used by `app.py`.

    Accepts optional vector_store_stats dict for display.
    """
    st.header("ℹ️ System Info")

    stats = vector_store_stats or {}
    st.metric("Documents", stats.get("vector_count", 0))
    st.metric("Backend", stats.get("backend", "unknown").upper())

    if settings.ENABLE_MEMORY_PROFILING:
        mem_info = get_memory_usage()
        if mem_info:
            st.metric("RAM Usage", f"{mem_info['rss_mb']:.0f} MB")

    st.markdown("---")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            if 'conversation_memory' in st.session_state:
                st.session_state.conversation_memory.clear()
            cleanup_memory()
            st.rerun()

    with col2:
        if st.button("🪄 Free RAM"):
            cleanup_memory()
            st.success("✓")

    with st.expander("⚙️ Settings"):
        st.write(f"**Chunking:** {'Hierarchical' if settings.USE_HIERARCHICAL_CHUNKING else 'Simple'}")
        st.write(f"**Hybrid Search:** {'Enabled' if settings.USE_HYBRID_SEARCH else 'Disabled'}")
        st.write(f"**Conv. Memory:** {'Enabled' if settings.ENABLE_CONVERSATION_MEMORY else 'Disabled'}")
        st.write(f"**Top-K:** {settings.SIMILARITY_TOP_K}")
