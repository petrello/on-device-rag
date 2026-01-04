"""
FAISS local vector store implementation.
"""

import logging
import pickle
from pathlib import Path
from typing import List
import faiss
import numpy as np
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from storage.vector_store import VectorStoreInterface
from config import settings

logger = logging.getLogger(__name__)


class LocalStore(VectorStoreInterface):
    """FAISS-based local vector store."""

    def __init__(self):
        """Initialize FAISS store."""
        self.index_path = settings.FAISS_INDEX_PATH
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_path / "index.faiss"
        self.metadata_file = self.index_path / "metadata.pkl"

        # Initialize FAISS index
        self.faiss_index = faiss.IndexFlatL2(settings.EMBEDDING_DIM)
        self.nodes: List[TextNode] = []

        # Load existing index if available
        if self.index_file.exists():
            self._load_index()
            logger.info(f"Loaded existing FAISS index with {len(self.nodes)} nodes")
        else:
            logger.info("Initialized new FAISS index")

        # Create vector store
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)

    def _load_index(self):
        """Load index from disk."""
        try:
            self.faiss_index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, 'rb') as f:
                self.nodes = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def _save_index(self):
        """Save index to disk."""
        try:
            faiss.write_index(self.faiss_index, str(self.index_file))
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.nodes, f)
            logger.debug("Index saved to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def add_nodes(self, nodes: List[TextNode]) -> None:
        """Add nodes to FAISS."""
        logger.info(f"Adding {len(nodes)} nodes to FAISS")
        self.nodes.extend(nodes)
        self._save_index()

    def query(
            self,
            query_embedding: List[float],
            top_k: int = 3
    ) -> List[NodeWithScore]:
        """Query FAISS with embedding."""
        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.faiss_index.search(query_vec, top_k)

        nodes_with_scores = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.nodes):
                # Convert L2 distance to similarity score
                score = 1.0 / (1.0 + dist)
                nodes_with_scores.append(
                    NodeWithScore(node=self.nodes[idx], score=score)
                )

        return nodes_with_scores

    def get_index(self) -> VectorStoreIndex:
        """Get LlamaIndex vector store index."""
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        return VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context
        )

    def delete_all(self) -> None:
        """Delete all nodes."""
        logger.warning("Deleting all data from FAISS index")
        self.faiss_index = faiss.IndexFlatL2(settings.EMBEDDING_DIM)
        self.nodes = []
        self._save_index()

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "backend": "faiss",
            "vector_count": len(self.nodes),
            "vector_dimension": settings.EMBEDDING_DIM,
            "index_path": str(self.index_path),
            "disk_size_mb": sum(
                f.stat().st_size for f in self.index_path.glob("*")
            ) / 1024 / 1024
        }