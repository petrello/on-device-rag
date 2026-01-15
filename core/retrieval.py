"""
Hybrid retrieval implementation combining vector and BM25 search.

This module provides retrievers that combine dense (vector) and sparse (BM25)
retrieval for improved accuracy. The hybrid approach leverages semantic
similarity from embeddings with keyword matching from BM25.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from llama_index.core.schema import NodeWithScore, TextNode
from rank_bm25 import BM25Okapi

from config import settings

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining dense vector search with sparse BM25.

    The final score is computed as:
        final_score = α × vector_score + (1 - α) × bm25_score

    where α (alpha) controls the balance between semantic and keyword matching.

    Attributes:
        alpha: Weight for vector retrieval (0.0 = pure BM25, 1.0 = pure vector).
        bm25: BM25 index for sparse retrieval.
        corpus_nodes: List of indexed nodes.
        node_id_to_idx: Mapping from node ID to corpus index.
    """

    __slots__ = ('alpha', 'bm25', 'corpus_nodes', 'node_id_to_idx', '_index')

    def __init__(self, alpha: Optional[float] = None) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            alpha: Weight for vector retrieval (0.0-1.0).
                Defaults to settings.HYBRID_ALPHA.
        """
        self.alpha: float = alpha if alpha is not None else settings.HYBRID_ALPHA
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_nodes: List[TextNode] = []
        self.node_id_to_idx: Dict[str, int] = {}
        self._index: Optional[VectorStoreIndex] = None

        logger.info(f"Initialized HybridRetriever with alpha={self.alpha}")

    def set_index(self, index: VectorStoreIndex) -> None:
        """
        Link a LlamaIndex VectorStoreIndex for vector retrieval.

        Args:
            index: The vector store index to use.
        """
        self._index = index

    def index_nodes(self, nodes: List[TextNode]) -> None:
        """
        Build the BM25 index from text nodes.

        Args:
            nodes: Text nodes to index for BM25 retrieval.
        """
        logger.info(f"Indexing {len(nodes)} nodes for BM25")

        self.corpus_nodes = nodes
        self.node_id_to_idx = {node.node_id: idx for idx, node in enumerate(nodes)}

        # Tokenize corpus (simple whitespace tokenization)
        tokenized_corpus = [node.text.lower().split() for node in nodes]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info("BM25 indexing complete")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """
        Perform hybrid retrieval combining vector and BM25 scores.

        Args:
            query: Search query string.
            top_k: Number of results to return. Defaults to settings.SIMILARITY_TOP_K.

        Returns:
            List of nodes ranked by combined score.
        """
        if top_k is None:
            top_k = settings.SIMILARITY_TOP_K

        # 1. Vector search
        vector_results: List[NodeWithScore] = []
        if self._index is not None:
            vector_retriever = self._index.as_retriever(similarity_top_k=top_k)
            vector_results = vector_retriever.retrieve(query)
        else:
            logger.warning("Vector index not set; returning BM25 results only")

        # 2. Fallback if BM25 not ready
        if self.bm25 is None:
            logger.warning("BM25 not initialized; returning vector results only")
            return vector_results

        # 3. Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 4. Normalize scores
        vector_scores = self._normalize_vector_scores(vector_results)
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        # 5. Combine scores from both sources
        combined_scores: Dict[str, Dict] = {}

        # Add vector scores
        for node_with_score in vector_results:
            node_id = node_with_score.node.node_id
            combined_scores[node_id] = {
                "node": node_with_score.node,
                "vector_score": vector_scores.get(node_id, 0.0),
                "bm25_score": 0.0,
            }

        # Add BM25 scores
        for node_id, idx in self.node_id_to_idx.items():
            if node_id not in combined_scores:
                combined_scores[node_id] = {
                    "node": self.corpus_nodes[idx],
                    "vector_score": 0.0,
                    "bm25_score": bm25_scores_norm[idx],
                }
            else:
                combined_scores[node_id]["bm25_score"] = bm25_scores_norm[idx]

        # 6. Compute final weighted scores
        for scores in combined_scores.values():
            scores["final_score"] = (
                self.alpha * scores["vector_score"]
                + (1 - self.alpha) * scores["bm25_score"]
            )

        # 7. Sort and return top_k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["final_score"],
            reverse=True,
        )

        final_results = [
            NodeWithScore(node=scores["node"], score=scores["final_score"])
            for _, scores in sorted_results[:top_k]
        ]

        logger.debug(
            f"Hybrid retrieval: {len(final_results)} results, "
            f"α={self.alpha} (vector/BM25 balance)"
        )

        return final_results

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization to scores.

        Args:
            scores: Raw score array.

        Returns:
            Normalized scores in [0, 1] range.
        """
        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def _normalize_vector_scores(
        self,
        results: List[NodeWithScore],
    ) -> Dict[str, float]:
        """
        Normalize vector similarity scores to [0, 1].

        Args:
            results: Vector search results.

        Returns:
            Dict mapping node_id to normalized score.
        """
        if not results:
            return {}

        scores = np.array([r.score for r in results])
        normalized = self._normalize_scores(scores)

        return {
            results[i].node.node_id: float(normalized[i])
            for i in range(len(results))
        }


class VectorOnlyRetriever:
    """
    Simple retriever that uses only vector similarity search.

    This retriever delegates directly to LlamaIndex's built-in retriever.
    Use this when BM25 is not needed or for lower memory usage.

    Attributes:
        _index: The linked VectorStoreIndex.
    """

    __slots__ = ('_index',)

    def __init__(self) -> None:
        """Initialize the vector-only retriever."""
        self._index: Optional[VectorStoreIndex] = None

    def set_index(self, index: VectorStoreIndex) -> None:
        """
        Link the VectorStoreIndex for retrieval.

        Args:
            index: The vector store index to use.
        """
        self._index = index

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """
        Perform vector-only retrieval.

        Args:
            query: Search query string.
            top_k: Number of results. Defaults to settings.SIMILARITY_TOP_K.

        Returns:
            List of retrieved nodes with scores.
        """
        if self._index is None:
            logger.error("Attempted retrieval before index was set")
            return []

        if top_k is None:
            top_k = settings.SIMILARITY_TOP_K

        base_retriever = self._index.as_retriever(similarity_top_k=top_k)
        return base_retriever.retrieve(query)


def get_retriever(
    use_hybrid: Optional[bool] = None,
) -> HybridRetriever | VectorOnlyRetriever:
    """
    Factory function to create the appropriate retriever.

    Args:
        use_hybrid: If True, return HybridRetriever. If False, VectorOnlyRetriever.
            Defaults to settings.USE_HYBRID_SEARCH.

    Returns:
        Configured retriever instance.
    """
    if use_hybrid is None:
        use_hybrid = settings.USE_HYBRID_SEARCH

    return HybridRetriever() if use_hybrid else VectorOnlyRetriever()
