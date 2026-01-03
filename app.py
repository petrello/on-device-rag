"""
Main Streamlit application.
Integrates all components into a complete RAG system.
"""

import streamlit as st
import logging
from pathlib import Path

# Setup must happen before other imports
from monitoring.logger import setup_logging

setup_logging()

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    PromptTemplate
)

from config import settings
from core import get_chunker, get_embedding_model, get_llm, get_retriever
from storage import get_vector_store
from memory import ConversationMemory
from citation import CitationExtractor
from monitoring import (
    start_metrics_server,
    track_query_metrics,
    update_memory_usage,
    document_count
)
from utils import cleanup_memory, get_memory_usage, check_memory_threshold
from ui import render_sidebar, stream_response, render_document_viewer

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Edge RAG",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize metrics server
if settings.ENABLE_METRICS:
    try:
        start_metrics_server()
    except Exception as e:
        logger.warning(f"Metrics server not started: {e}")


@st.cache_resource
def initialize_models():
    """Initialize LLM and embedding models."""
    logger.info("Initializing models...")

    # Get embedding model
    embed_model = get_embedding_model()

    # Get LLM
    llm = get_llm()

    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Get chunker
    chunker = get_chunker()
    Settings.text_splitter = chunker.child_splitter if hasattr(chunker, 'child_splitter') else chunker.splitter

    logger.info("Models initialized successfully")
    return llm, embed_model, chunker


@st.cache_resource
def initialize_index(_chunker):
    """Initialize vector store and index."""
    logger.info("Initializing vector store and index...")

    # Check for documents
    if not list(settings.DATA_DIR.glob("*")):
        st.warning("⚠️ No documents found in data directory")
        st.stop()

    # Load documents
    documents = SimpleDirectoryReader(str(settings.DATA_DIR)).load_data()
    logger.info(f"Loaded {len(documents)} documents")

    # Update document count metric
    if settings.ENABLE_METRICS:
        document_count.set(len(documents))

    # Get vector store
    vector_store = get_vector_store()

    # Create hierarchical chunks if enabled
    if settings.USE_HIERARCHICAL_CHUNKING:
        child_nodes, parent_map = _chunker.chunk_documents(documents)
        logger.info(f"Created {len(child_nodes)} child chunks")

        # Build index from child nodes
        index = VectorStoreIndex(
            child_nodes,
            storage_context=vector_store.get_index().storage_context
        )

        return index, parent_map, vector_store
    else:
        # Simple chunking
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=vector_store.get_index().storage_context
        )

        return index, {}, vector_store


# Llama-3.2 prompt template
QA_PROMPT = PromptTemplate(
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant. Use the provided context to answer the user query. "
    "If the answer is not in the context, state that you do not know. "
    "Be concise and accurate.\n\n"
    "Context:\n{context_str}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def main():
    """Main application."""

    # Header
    st.title("📑 Edge RAG System")
    st.markdown("*Production-ready RAG for edge devices*")
    st.markdown("---")

    # Initialize models
    with st.spinner("Loading models..."):
        llm, embed_model, chunker = initialize_models()

    # Initialize index
    with st.spinner("Building knowledge base..."):
        index, parent_map, vector_store = initialize_index(chunker)

    # Initialize retriever
    retriever = get_retriever()
    if settings.USE_HYBRID_SEARCH and hasattr(retriever, 'index_nodes'):
        # Index nodes for BM25
        all_nodes = list(index.docstore.docs.values())
        retriever.index_nodes(all_nodes)

    # Create query engine
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=settings.SIMILARITY_TOP_K,
        text_qa_template=QA_PROMPT
    )

    # Initialize citation extractor
    citation_extractor = CitationExtractor()

    # Sidebar
    with st.sidebar:
        # Use helper
        render_sidebar(vector_store.get_stats())

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
            if st.button("🧹 Free RAM"):
                cleanup_memory()
                st.success("✓")

        # Settings
        with st.expander("⚙️ Settings"):
            st.write(f"**Chunking:** {'Hierarchical' if settings.USE_HIERARCHICAL_CHUNKING else 'Simple'}")
            st.write(f"**Hybrid Search:** {'Enabled' if settings.USE_HYBRID_SEARCH else 'Disabled'}")
            st.write(f"**Conv. Memory:** {'Enabled' if settings.ENABLE_CONVERSATION_MEMORY else 'Disabled'}")
            st.write(f"**Top-K:** {settings.SIMILARITY_TOP_K}")

    # Initialize conversation memory
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Trim history if needed
    if len(st.session_state.messages) > settings.MAX_CHAT_HISTORY * 2:
        st.session_state.messages = st.session_state.messages[-(settings.MAX_CHAT_HISTORY * 2):]
        cleanup_memory()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "citations" in message and message["citations"]:
                with st.expander("📚 View Sources"):
                    for i, citation in enumerate(message["citations"], 1):
                        st.markdown(
                            citation_extractor.format_citation_markdown(citation, i)
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check memory threshold
        if check_memory_threshold():
            cleanup_memory()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Enhanced query with conversation context
                enhanced_query = prompt
                if settings.ENABLE_CONVERSATION_MEMORY:
                    context = st.session_state.conversation_memory.get_context()
                    if context:
                        enhanced_query = f"Conversation history:\n{context}\n\nCurrent question: {prompt}"

                # Query with metrics tracking
                @track_query_metrics
                def run_query(q):
                    return query_engine.query(q)

                streaming_response = run_query(enhanced_query)

                # Stream response
                response_text = stream_response(streaming_response.response_gen)

                # Extract citations
                citations = []
                if settings.ENABLE_CITATIONS and streaming_response.source_nodes:
                    citations = citation_extractor.extract_citations(
                        response_text,
                        streaming_response.source_nodes
                    )

                    with st.expander("📚 View Sources"):
                        for i, citation in enumerate(citations, 1):
                            st.markdown(
                                citation_extractor.format_citation_markdown(citation, i)
                            )

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

                # Update memory metric
                if settings.ENABLE_METRICS:
                    update_memory_usage()

            except Exception as e:
                logger.error(f"Query failed: {e}", exc_info=True)
                error_msg = f"⚠️ An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

    # Optionally render document viewer below main chat
    st.sidebar.markdown("---")
    if st.sidebar.button("Show Documents"):
        render_document_viewer()


if __name__ == "__main__":
    main()