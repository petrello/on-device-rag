"""
Qdrant vector store implementation.
"""

import logging
from typing import List, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from storage.vector_store import VectorStoreInterface
from config import settings

logger = logging.getLogger(__name__)


class QdrantStore(VectorStoreInterface):
    """Qdrant vector store implementation."""

    def __init__(self):
        """Initialize Qdrant connection."""
        logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")

        try:
            self.client = QdrantClient(url=settings.QDRANT_URL, timeout=10)

            # Test connection
            self.client.get_collections()
            logger.info("Qdrant connection successful")

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(
                f"Cannot connect to Qdrant at {settings.QDRANT_URL}. "
                f"Make sure Qdrant is running (docker compose up -d qdrant)"
            )

        # Ensure collection exists
        self._ensure_collection()

        # Create vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.QDRANT_COLLECTION
        )

        self._storage_context: Optional[StorageContext] = None

    def _ensure_collection(self):
        """Ensure collection exists with correct configuration."""
        collections = self.client.get_collections().collections
        exists = any(c.name == settings.QDRANT_COLLECTION for c in collections)

        if not exists:
            logger.info(f"Creating collection: {settings.QDRANT_COLLECTION}")

            self.client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=models.Distance.COSINE
                ),
            )

            logger.info("Collection created successfully")
        else:
            logger.info(f"Using existing collection: {settings.QDRANT_COLLECTION}")

    def add_nodes(self, nodes: List[TextNode]) -> None:
        """Add nodes to Qdrant."""
        logger.info(f"Adding {len(nodes)} nodes to Qdrant")
        # Nodes are added through LlamaIndex VectorStoreIndex
        pass

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3
    ) -> List[NodeWithScore]:
        """Query Qdrant with embedding."""
        results = self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=top_k
        )

        # Convert to NodeWithScore format
        nodes_with_scores = []
        for result in results:
            node = TextNode(
                text=result.payload.get("text", ""),
                metadata=result.payload.get("metadata", {})
            )
            nodes_with_scores.append(
                NodeWithScore(node=node, score=result.score)
            )

        return nodes_with_scores

    def get_index(self) -> VectorStoreIndex:
        """Get LlamaIndex vector store index."""
        storage_context = self.get_storage_context()
        return VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context
        )

    def get_storage_context(self) -> StorageContext:
        """
        Get storage context for creating new indices.
        This is the method that should be used when building new indices.
        """
        if self._storage_context is None:
            self._storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
        return self._storage_context

    def delete_all(self) -> None:
        """Delete all nodes from collection."""
        logger.warning(f"Deleting all data from collection: {settings.QDRANT_COLLECTION}")

        try:
            self.client.delete_collection(settings.QDRANT_COLLECTION)
            self._ensure_collection()
            self._storage_context = None
            logger.info("Collection recreated")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def get_stats(self) -> dict:
        """Get collection statistics."""
        try:
            collection_info = self.client.get_collection(settings.QDRANT_COLLECTION)

            return {
                "backend": "qdrant",
                "collection_name": settings.QDRANT_COLLECTION,
                "vector_count": collection_info.points_count,
                "vector_dimension": settings.EMBEDDING_DIM,
                "distance_metric": "cosine"
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"backend": "qdrant", "error": str(e)}