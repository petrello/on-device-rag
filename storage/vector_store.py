"""
Abstract vector store interface.

Defines the common interface for vector store backends (Qdrant, FAISS).
All storage implementations must inherit from VectorStoreInterface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import NodeWithScore, TextNode


class VectorStoreInterface(ABC):
    """
    Abstract interface for vector store backends.

    Implementations must provide methods for adding, querying,
    and managing vectors. This abstraction allows swapping between
    Qdrant (production) and FAISS (lightweight/offline) backends.
    """

    @abstractmethod
    def add_nodes(self, nodes: List[TextNode]) -> None:
        """
        Add text nodes to the vector store.

        Args:
            nodes: List of text nodes to store.
        """
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[NodeWithScore]:
        """
        Query the vector store with an embedding.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.

        Returns:
            List of nodes with similarity scores.
        """
        pass

    @abstractmethod
    def get_index(self) -> VectorStoreIndex:
        """
        Get the LlamaIndex VectorStoreIndex.

        Returns:
            The vector store index for querying.
        """
        pass

    @abstractmethod
    def get_storage_context(self) -> StorageContext:
        """
        Get storage context for building new indices.

        Returns:
            Storage context configured with this vector store.
        """
        pass

    @abstractmethod
    def delete_all(self) -> None:
        """Delete all nodes from the vector store."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, object]:
        """
        Get vector store statistics.

        Returns:
            Dictionary with backend info, vector count, etc.
        """
        pass