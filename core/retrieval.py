"""
Hybrid retrieval implementation combining vector and BM25 search.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from llama_index.core.schema import NodeWithScore, TextNode
from config import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval.

    Uses weighted scoring: final_score = α * vector_score + (1-α) * bm25_score
    """

    def __init__(self, alpha: float = None):
        """
        Initialize hybrid retriever.

        Args:
            alpha: Weight for vector retrieval (0.0-1.0)
                   1.0 = pure vector, 0.0 = pure BM25
        """
        self.alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_nodes: List[TextNode] = []
        self.node_id_to_idx: Dict[str, int] = {}

        logger.info(f"Initialized HybridRetriever with alpha={self.alpha}")

    def index_nodes(self, nodes: List[TextNode]):
        """
        Index nodes for BM25 retrieval.

        Args:
            nodes: List of text nodes to index
        """
        logger.info(f"Indexing {len(nodes)} nodes for BM25")

        self.corpus_nodes = nodes
        self.node_id_to_idx = {node.node_id: idx for idx, node in enumerate(nodes)}

        # Tokenize corpus for BM25
        tokenized_corpus = [
            node.text.lower().split()
            for node in nodes
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 indexing complete")

    def retrieve(
            self,
            query: str,
            vector_results: List[NodeWithScore],
            top_k: int = None
    ) -> List[NodeWithScore]:
        """
        Perform hybrid retrieval.

        Args:
            query: Search query
            vector_results: Results from vector search
            top_k: Number of results to return

        Returns:
            List of nodes with combined scores
        """
        if top_k is None:
            top_k = settings.SIMILARITY_TOP_K

        if not self.bm25:
            logger.warning("BM25 not initialized, returning vector results only")
            return vector_results[:top_k]

        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores
        vector_scores = self._normalize_vector_scores(vector_results)
        bm25_scores_norm = self._normalize_scores(bm25_scores)

        # Combine scores
        combined_scores = {}

        # Add vector scores
        for node_with_score in vector_results:
            node_id = node_with_score.node.node_id
            combined_scores[node_id] = {
                "node": node_with_score.node,
                "vector_score": vector_scores.get(node_id, 0.0),
                "bm25_score": 0.0
            }

        # Add BM25 scores
        for node_id, idx in self.node_id_to_idx.items():
            if node_id not in combined_scores:
                combined_scores[node_id] = {
                    "node": self.corpus_nodes[idx],
                    "vector_score": 0.0,
                    "bm25_score": bm25_scores_norm[idx]
                }
            else:
                combined_scores[node_id]["bm25_score"] = bm25_scores_norm[idx]

        # Calculate final scores
        for node_id in combined_scores:
            vec_score = combined_scores[node_id]["vector_score"]
            bm25_score = combined_scores[node_id]["bm25_score"]
            combined_scores[node_id]["final_score"] = (
                    self.alpha * vec_score + (1 - self.alpha) * bm25_score
            )

        # Sort by final score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["final_score"],
            reverse=True
        )

        # Convert to NodeWithScore
        final_results = []
        for node_id, scores in sorted_results[:top_k]:
            final_results.append(
                NodeWithScore(
                    node=scores["node"],
                    score=scores["final_score"]
                )
            )

        logger.debug(
            f"Hybrid retrieval: {len(final_results)} results, "
            f"α={self.alpha} (vector/BM25 weights)"
        )

        return final_results

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalization."""
        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def _normalize_vector_scores(
            self,
            results: List[NodeWithScore]
    ) -> Dict[str, float]:
        """
        Normalize vector similarity scores.

        Returns:
            Dict mapping node_id to normalized score
        """
        if not results:
            return {}

        scores = np.array([r.score for r in results])
        normalized = self._normalize_scores(scores)

        return {
            results[i].node.node_id: normalized[i]
            for i in range(len(results))
        }


class VectorOnlyRetriever:
    """Simple vector-only retrieval (no BM25)."""

    def retrieve(
            self,
            query: str,
            vector_results: List[NodeWithScore],
            top_k: int = None
    ) -> List[NodeWithScore]:
        """Return vector results as-is."""
        if top_k is None:
            top_k = settings.SIMILARITY_TOP_K
        return vector_results[:top_k]


def get_retriever(use_hybrid: bool = None) -> HybridRetriever | VectorOnlyRetriever:
    """
    Factory function to get appropriate retriever.

    Args:
        use_hybrid: Override settings if provided

    Returns:
        Retriever instance
    """
    if use_hybrid is None:
        use_hybrid = settings.USE_HYBRID_SEARCH

    if use_hybrid:
        return HybridRetriever()
    else:
        return VectorOnlyRetriever()