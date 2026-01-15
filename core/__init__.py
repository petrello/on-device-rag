"""
Core RAG components.

This module provides the essential building blocks for the RAG pipeline:
- Chunking strategies (hierarchical and simple)
- Embedding model management
- LLM initialization and management
- Retrieval strategies (hybrid and vector-only)
"""

from core.chunking import HierarchicalChunker, SimpleChunker, get_chunker
from core.embeddings import EmbeddingManager, get_embedding_model
from core.llm import LLMManager, get_llm
from core.retrieval import HybridRetriever, VectorOnlyRetriever, get_retriever

__all__ = [
    # Chunking
    "HierarchicalChunker",
    "SimpleChunker",
    "get_chunker",
    # Embeddings
    "EmbeddingManager",
    "get_embedding_model",
    # LLM
    "LLMManager",
    "get_llm",
    # Retrieval
    "HybridRetriever",
    "VectorOnlyRetriever",
    "get_retriever",
]