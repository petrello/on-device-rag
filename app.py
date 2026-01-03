"""
Main Streamlit application.
"""

import streamlit as st
import logging
import time

# Setup logging before other imports
from monitoring.logger import setup_logging
setup_logging()

from llama_index.core import VectorStoreIndex, Settings, PromptTemplate

from config import settings
from core import get_chunker, get_embedding_model, get_llm, get_retriever
from storage import get_vector_store
from memory import ConversationMemory
from citation import CitationExtractor
from document_management import DocumentProcessor, DocumentIndexer
from monitoring import (
    start_metrics_server,
    track_query_metrics,
    update_memory_usage,
    PerformanceDashboard,
    display_system_info,
    display_prometheus_link
)
from ui import (
    display_chat_history,
    render_chat_input,
    display_welcome_message,
    display_error_message,
    format_streaming_response,
    render_sidebar,
    render_document_manager
)
from utils import (
    cleanup_memory,
    check_memory_threshold,
    validate_query,
    sanitize_text
)

logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="On-device RAG",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# INITIALIZE METRICS SERVER
# =============================================================================

if settings.ENABLE_METRICS:
    try:
        start_metrics_server()
        logger.info("Metrics server started")
    except Exception as e:
        logger.warning(f"Metrics server not started: {e}")

# =============================================================================
# CACHED RESOURCES
# =============================================================================

@st.cache_resource
def initialize_models():
    """Initialize LLM and embedding models."""
    logger.info("Initializing models...")

    try:
        # Get embedding model
        embed_model = get_embedding_model()
        logger.info("Embedding model loaded")

        # Get LLM
        llm = get_llm()
        logger.info("LLM loaded")

        # Get chunker
        chunker = get_chunker()
        logger.info(
            f"Chunker initialized: "
            f"{'Hierarchical' if settings.USE_HIERARCHICAL_CHUNKING else 'Simple'}"
        )

        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model

        # Set text splitter
        if hasattr(chunker, 'child_splitter'):
            Settings.text_splitter = chunker.child_splitter
        else:
            Settings.text_splitter = chunker.splitter

        logger.info("Models initialized successfully")
        return llm, embed_model, chunker

    except Exception as e:
        logger.error(f"Failed to initialize models: {e}", exc_info=True)
        st.error(f"Failed to initialize models: {e}")
        st.stop()


@st.cache_resource
def initialize_vector_store():
    """Initialize vector store."""
    logger.info("Initializing vector store...")

    try:
        vector_store = get_vector_store()
        logger.info(f"Vector store initialized: {settings.VECTOR_STORE_TYPE}")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
        st.error(f"Failed to initialize vector store: {e}")
        st.stop()


@st.cache_resource
def initialize_index(_vector_store, _chunker):
    """Initialize document index."""
    logger.info("Initializing document index...")

    try:
        # Check for documents
        doc_processor = DocumentProcessor()
        documents = doc_processor.load_documents()

        if not documents:
            st.warning("⚠️ No documents found in data directory. Please add documents to continue.")
            st.info(f"📁 Add PDF, TXT, DOCX, or MD files to: `{settings.DATA_DIR}`")
            st.stop()

        # Validate and preprocess documents
        documents = doc_processor.preprocess_documents(documents)
        valid_docs, errors = doc_processor.validate_documents(documents)

        if errors:
            logger.warning(f"Document validation errors: {errors}")

        if not valid_docs:
            st.error("❌ No valid documents found")
            st.stop()

        logger.info(f"Loaded {len(valid_docs)} documents")

        # Get retriever
        retriever = get_retriever()

        # Create indexer and index documents
        indexer = DocumentIndexer(_vector_store, retriever)

        with st.spinner("Building index... This may take a few minutes."):
            index, parent_map = indexer.index_documents(
                valid_docs,
                show_progress=True
            )

        # Get stats
        stats = indexer.get_indexing_stats(index)
        logger.info(f"Indexing complete: {stats}")

        return index, parent_map, indexer, retriever

    except Exception as e:
        logger.error(f"Failed to initialize index: {e}", exc_info=True)
        st.error(f"Failed to initialize index: {e}")
        st.stop()

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

QA_PROMPT = PromptTemplate(
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant. Use the provided context to answer the user query. "
    "If the answer is not in the context, state that you do not know. "
    "Be concise, accurate, and cite sources when possible.\n\n"
    "Context:\n{context_str}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application logic."""

    # Header
    st.title("📑 Edge RAG System")
    st.markdown("*Production-ready RAG for edge devices*")

    # Initialize performance dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = PerformanceDashboard()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Documents", "📊 Monitoring"])

    with tab1:
        render_chat_tab()

    with tab2:
        render_documents_tab()

    with tab3:
        render_monitoring_tab()


def render_chat_tab():
    """Render main chat interface."""

    # Initialize models
    with st.spinner("Loading models..."):
        llm, embed_model, chunker = initialize_models()

    # Initialize vector store
    vector_store = initialize_vector_store()

    # Initialize index
    with st.spinner("Building knowledge base..."):
        index, parent_map, indexer, retriever = initialize_index(
            vector_store,
            chunker
        )

    # Create query engine
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=settings.SIMILARITY_TOP_K,
        text_qa_template=QA_PROMPT
    )

    # Initialize citation extractor
    citation_extractor = CitationExtractor()

    # Render sidebar
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory()

    render_sidebar(vector_store, st.session_state.conversation_memory)

    # Initialize conversation memory
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory()

    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Trim chat history if needed
    if len(st.session_state.messages) > settings.MAX_CHAT_HISTORY * 2:
        st.session_state.messages = st.session_state.messages[-(settings.MAX_CHAT_HISTORY * 2):]
        cleanup_memory()
        logger.info("Chat history trimmed")

    # Display welcome message for new users
    if not st.session_state.messages:
        display_welcome_message()

    # Display chat history
    display_chat_history(st.session_state.messages, citation_extractor)

    # Chat input
    prompt = render_chat_input()

    if prompt:
        # Validate query
        is_valid, error_msg = validate_query(prompt)
        if not is_valid:
            st.error(f"⚠️ Invalid query: {error_msg}")
            return

        # Sanitize input
        prompt = sanitize_text(prompt, max_length=5000)

        # Check memory threshold
        if check_memory_threshold():
            logger.warning("Memory threshold exceeded, cleaning up")
            cleanup_memory()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Start timer
                start_time = time.time()

                # Enhance query with conversation context
                enhanced_query = prompt
                if settings.ENABLE_CONVERSATION_MEMORY:
                    context = st.session_state.conversation_memory.get_context()
                    if context:
                        enhanced_query = (
                            f"Previous conversation:\n{context}\n\n"
                            f"Current question: {prompt}"
                        )
                        logger.debug("Using conversation context")

                # Query with metrics tracking
                @track_query_metrics
                def run_query(q):
                    return query_engine.query(q)

                streaming_response = run_query(enhanced_query)

                # Stream response
                response_text = format_streaming_response(streaming_response.response_gen)

                # Calculate latency
                latency = time.time() - start_time

                # Extract citations
                citations = []
                if settings.ENABLE_CITATIONS and streaming_response.source_nodes:
                    try:
                        citations = citation_extractor.extract_citations(
                            response_text,
                            streaming_response.source_nodes
                        )

                        if citations:
                            with st.expander("📚 View Sources", expanded=False):
                                for i, citation in enumerate(citations, 1):
                                    st.markdown(
                                        citation_extractor.format_citation_markdown(citation, i),
                                        unsafe_allow_html=True
                                    )
                    except Exception as e:
                        logger.error(f"Citation extraction failed: {e}")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "citations": citations
                })

                # Update conversation memory
                if settings.ENABLE_CONVERSATION_MEMORY:
                    st.session_state.conversation_memory.add_exchange(
                        prompt,
                        response_text
                    )

                # Record performance
                st.session_state.dashboard.record_query(latency)

                # Update memory metric
                if settings.ENABLE_METRICS:
                    update_memory_usage()

                # Display query info
                st.caption(f"⏱️ Response time: {latency:.2f}s")

            except Exception as e:
                logger.error(f"Query failed: {e}", exc_info=True)
                display_error_message(e)

                # Save error to history
                error_msg = f"⚠️ An error occurred: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


def render_documents_tab():
    """Render document management interface."""
    render_document_manager()

    st.markdown("---")
    st.info(
        "💡 **Note:** After uploading or deleting documents, "
        "please restart the application to rebuild the index."
    )

    # Display document statistics
    doc_processor = DocumentProcessor()
    try:
        documents = doc_processor.load_documents()
        if documents:
            stats = doc_processor.get_document_stats(documents)

            st.subheader("📊 Document Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Documents", stats["total_documents"])
            with col2:
                st.metric("Total Characters", f"{stats['total_characters']:,}")
            with col3:
                st.metric("Avg Length", f"{stats['avg_length']:,}")

            if stats["file_types"]:
                st.markdown("**File Types:**")
                for ext, count in stats["file_types"].items():
                    st.text(f"• {ext.upper()}: {count} file(s)")
    except Exception as e:
        logger.error(f"Failed to load document stats: {e}")


def render_monitoring_tab():
    """Render monitoring and metrics."""
    st.header("📊 System Monitoring")

    # Display performance dashboard
    st.session_state.dashboard.display_metrics()

    st.markdown("---")

    # Display system info
    display_system_info()

    st.markdown("---")

    # Display Prometheus link
    display_prometheus_link()

    # Additional monitoring features
    with st.expander("🔧 Advanced Monitoring"):
        st.markdown("""
        **Available Metrics:**
        - Query throughput (queries/second)
        - Query latency (p50, p95, p99)
        - Memory usage over time
        - Error rates by type
        - Document count
        
        **Grafana Dashboard:**
        Configure Grafana to visualize these metrics.
        Import the dashboard from `monitoring/dashboard.json`
        """)


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"Application error: {e}")
        st.info("Please check logs for details and restart the application.")