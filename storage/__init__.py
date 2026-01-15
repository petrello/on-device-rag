"""
Storage package.

Provides vector store implementations for document embeddings:
- Qdrant: Production-ready vector database (recommended)
- FAISS: Lightweight local storage for offline use
"""

from storage.local_store import LocalStore
from storage.qdrant_store import QdrantStore
from storage.vector_store import VectorStoreInterface

from config import settings


def get_vector_store() -> VectorStoreInterface:
    """
    Factory function to create the configured vector store.

    Returns:
        VectorStoreInterface implementation based on settings.VECTOR_STORE_TYPE.

    Raises:
        ValueError: If vector store type is unknown.
    """
    if settings.VECTOR_STORE_TYPE == "qdrant":
        return QdrantStore()
    elif settings.VECTOR_STORE_TYPE == "faiss":
        return LocalStore()
    else:
        raise ValueError(f"Unknown vector store type: {settings.VECTOR_STORE_TYPE}")


__all__ = [
    "VectorStoreInterface",
    "QdrantStore",
    "LocalStore",
    "get_vector_store",
]