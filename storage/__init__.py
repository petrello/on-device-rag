"""Storage package."""

from storage.vector_store import VectorStoreInterface
from storage.qdrant_store import QdrantStore
from storage.local_store import LocalStore
from config import settings


def get_vector_store() -> VectorStoreInterface:
    """Factory function to get configured vector store."""
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