"""
Qdrant vector store implementation.

Provides integration with Qdrant vector database for production deployments.
Qdrant offers fast similarity search with persistence and scalability.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from config import settings
from storage.vector_store import VectorStoreInterface

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class QdrantStore(VectorStoreInterface):
    """
    Qdrant-based vector store implementation.

    Connects to a Qdrant instance (typically running in Docker) and provides
    persistent vector storage with efficient similarity search.

    Attributes:
        client: Qdrant client connection.
        vector_store: LlamaIndex Qdrant vector store wrapper.
    """

    __slots__ = ('client', 'vector_store', '_storage_context')

    def __init__(self) -> None:
        """
        Initialize Qdrant connection and ensure collection exists.

        Raises:
            ConnectionError: If unable to connect to Qdrant.
        """
        logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")

        try:
            self.client = QdrantClient(url=settings.QDRANT_URL, timeout=10)
            self.client.get_collections()  # Test connection
            logger.info("Qdrant connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(
                f"Cannot connect to Qdrant at {settings.QDRANT_URL}. "
                "Ensure Qdrant is running: docker compose up -d qdrant"
            ) from e

        self._ensure_collection()

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.QDRANT_COLLECTION,
        )

        self._storage_context: Optional[StorageContext] = None

    def _ensure_collection(self) -> None:
        """Ensure the collection exists with correct configuration."""
        collections = self.client.get_collections().collections
        exists = any(c.name == settings.QDRANT_COLLECTION for c in collections)

        if not exists:
            logger.info(f"Creating collection: {settings.QDRANT_COLLECTION}")
            self.client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("Collection created")
        else:
            logger.info(f"Using existing collection: {settings.QDRANT_COLLECTION}")

    def add_nodes(self, nodes: List[TextNode]) -> None:
        """
        Add nodes to Qdrant.

        Note: Nodes are typically added through VectorStoreIndex during indexing.

        Args:
            nodes: Text nodes to add.
        """
        logger.info(f"Adding {len(nodes)} nodes to Qdrant")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[NodeWithScore]:
        """
        Query Qdrant with an embedding vector.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of nodes with similarity scores.
        """
        results = self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=top_k,
        )

        nodes_with_scores = []
        for result in results:
            node = TextNode(
                text=result.payload.get("text", ""),
                metadata=result.payload.get("metadata", {}),
            )
            nodes_with_scores.append(NodeWithScore(node=node, score=result.score))

        return nodes_with_scores

    def get_index(self) -> VectorStoreIndex:
        """
        Get a VectorStoreIndex from the existing Qdrant collection.

        Returns:
            VectorStoreIndex backed by Qdrant.
        """
        storage_context = self.get_storage_context()
        return VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context,
        )

    def get_storage_context(self) -> StorageContext:
        """
        Get storage context for building new indices.

        Returns:
            Storage context configured with Qdrant vector store.
        """
        if self._storage_context is None:
            self._storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
        return self._storage_context

    def delete_all(self) -> None:
        """Delete all vectors by recreating the collection."""
        logger.warning(f"Deleting all data from: {settings.QDRANT_COLLECTION}")

        try:
            self.client.delete_collection(settings.QDRANT_COLLECTION)
            self._ensure_collection()
            self._storage_context = None
            logger.info("Collection recreated")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def get_stats(self) -> Dict[str, object]:
        """
        Get collection statistics.

        Returns:
            Dictionary with backend info, vector count, dimension, etc.
        """
        try:
            info = self.client.get_collection(settings.QDRANT_COLLECTION)
            return {
                "backend": "qdrant",
                "collection_name": settings.QDRANT_COLLECTION,
                "vector_count": info.points_count,
                "vector_dimension": settings.EMBEDDING_DIM,
                "distance_metric": "cosine",
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"backend": "qdrant", "error": str(e)}