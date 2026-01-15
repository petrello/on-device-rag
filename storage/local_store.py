"""
FAISS local vector store implementation.

Provides a lightweight, file-based vector store using FAISS.
Suitable for offline deployments or when Qdrant is not available.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import faiss
import numpy as np
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

from config import settings
from storage.vector_store import VectorStoreInterface

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LocalStore(VectorStoreInterface):
    """
    FAISS-based local vector store.

    Stores vectors in a flat L2 index with metadata persisted to disk.
    Best for small to medium datasets (< 100k vectors) on memory-constrained devices.

    Attributes:
        index_path: Directory for index persistence.
        faiss_index: FAISS index instance.
        nodes: List of stored text nodes (for metadata).
        vector_store: LlamaIndex FAISS vector store wrapper.
    """

    __slots__ = (
        'index_path', 'index_file', 'metadata_file',
        'faiss_index', 'nodes', 'vector_store', '_storage_context'
    )

    def __init__(self) -> None:
        """Initialize FAISS store, loading existing index if available."""
        self.index_path: Path = settings.FAISS_INDEX_PATH
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index_file: Path = self.index_path / "index.faiss"
        self.metadata_file: Path = self.index_path / "metadata.pkl"

        # Initialize FAISS flat L2 index
        self.faiss_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(settings.EMBEDDING_DIM)
        self.nodes: List[TextNode] = []

        # Load existing index if available
        if self.index_file.exists():
            self._load_index()
            logger.info(f"Loaded FAISS index with {len(self.nodes)} nodes")
        else:
            logger.info("Initialized new FAISS index")

        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self._storage_context: Optional[StorageContext] = None

    def _load_index(self) -> None:
        """Load index and metadata from disk."""
        try:
            self.faiss_index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, 'rb') as f:
                self.nodes = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def _save_index(self) -> None:
        """Persist index and metadata to disk."""
        try:
            faiss.write_index(self.faiss_index, str(self.index_file))
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.nodes, f)
            logger.debug("Index saved to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def add_nodes(self, nodes: List[TextNode]) -> None:
        """
        Add nodes to the FAISS index.

        Args:
            nodes: Text nodes to add.
        """
        logger.info(f"Adding {len(nodes)} nodes to FAISS")
        self.nodes.extend(nodes)
        self._save_index()

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[NodeWithScore]:
        """
        Query FAISS with an embedding vector.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of nodes with similarity scores.
        """
        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.faiss_index.search(query_vec, top_k)

        nodes_with_scores = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.nodes):
                # Convert L2 distance to similarity score
                score = 1.0 / (1.0 + dist)
                nodes_with_scores.append(
                    NodeWithScore(node=self.nodes[idx], score=score)
                )

        return nodes_with_scores

    def get_index(self) -> VectorStoreIndex:
        """
        Get a VectorStoreIndex from the FAISS store.

        Returns:
            VectorStoreIndex backed by FAISS.
        """
        if not self.nodes:
            logger.warning("Getting index from empty vector store")

        storage_context = self.get_storage_context()

        if self.nodes:
            return VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=storage_context,
            )
        else:
            return VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
            )

    def get_storage_context(self) -> StorageContext:
        """
        Get storage context for building new indices.

        Returns:
            Storage context configured with FAISS vector store.
        """
        if self._storage_context is None:
            self._storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
        return self._storage_context

    def delete_all(self) -> None:
        """Delete all vectors and reset the index."""
        logger.warning("Deleting all data from FAISS index")
        self.faiss_index = faiss.IndexFlatL2(settings.EMBEDDING_DIM)
        self.nodes = []
        self._storage_context = None
        self._save_index()

    def get_stats(self) -> Dict[str, object]:
        """
        Get index statistics.

        Returns:
            Dictionary with backend info, vector count, disk size, etc.
        """
        disk_size_mb = 0.0
        try:
            disk_size_mb = sum(
                f.stat().st_size for f in self.index_path.glob("*")
            ) / (1024 * 1024)
        except Exception:
            pass

        return {
            "backend": "faiss",
            "vector_count": len(self.nodes),
            "vector_dimension": settings.EMBEDDING_DIM,
            "index_path": str(self.index_path),
            "disk_size_mb": round(disk_size_mb, 2),
        }