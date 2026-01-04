"""
Abstract vector store interface.
Provides common interface for different vector store backends.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core import VectorStoreIndex


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    def add_nodes(self, nodes: List[TextNode]) -> None:
        """Add nodes to vector store."""
        pass

    @abstractmethod
    def query(
            self,
            query_embedding: List[float],
            top_k: int = 3
    ) -> List[NodeWithScore]:
        """Query vector store with embedding."""
        pass

    @abstractmethod
    def get_index(self) -> VectorStoreIndex:
        """Get LlamaIndex vector store index."""
        pass

    @abstractmethod
    def delete_all(self) -> None:
        """Delete all nodes from vector store."""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        pass