"""
Tests for retrieval module.
"""

import pytest
import numpy as np
from llama_index.core.schema import TextNode, NodeWithScore
from core.retrieval import HybridRetriever, VectorOnlyRetriever, get_retriever


@pytest.fixture
def sample_nodes():
    """Create sample text nodes."""
    nodes = [
        TextNode(
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "doc1.pdf"}
        ),
        TextNode(
            text="Python is a popular programming language for data science.",
            metadata={"source": "doc2.pdf"}
        ),
        TextNode(
            text="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "doc3.pdf"}
        ),
    ]

    # Set node IDs
    for i, node in enumerate(nodes):
        node.node_id = f"node_{i}"

    return nodes


@pytest.fixture
def sample_vector_results(sample_nodes):
    """Create sample vector search results."""
    results = [
        NodeWithScore(node=sample_nodes[0], score=0.9),
        NodeWithScore(node=sample_nodes[1], score=0.7),
        NodeWithScore(node=sample_nodes[2], score=0.5),
    ]
    return results


class TestHybridRetriever:
    """Test hybrid retrieval functionality."""

    def test_initialization(self):
        """Test retriever initialization."""
        retriever = HybridRetriever(alpha=0.7)

        assert retriever.alpha == 0.7
        assert retriever.bm25 is None  # Not indexed yet
        assert len(retriever.corpus_nodes) == 0

    def test_index_nodes(self, sample_nodes):
        """Test node indexing for BM25."""
        retriever = HybridRetriever()
        retriever.index_nodes(sample_nodes)

        assert retriever.bm25 is not None
        assert len(retriever.corpus_nodes) == len(sample_nodes)
        assert len(retriever.node_id_to_idx) == len(sample_nodes)

    def test_retrieve_without_index(self, sample_vector_results):
        """Test retrieval without BM25 index (fallback to vector only)."""
        retriever = HybridRetriever()

        results = retriever.retrieve(
            "test query",
            sample_vector_results,
            top_k=2
        )

        # Should fallback to vector results
        assert len(results) == 2
        assert all(isinstance(r, NodeWithScore) for r in results)

    def test_retrieve_with_index(self, sample_nodes, sample_vector_results):
        """Test hybrid retrieval with BM25 index."""
        retriever = HybridRetriever(alpha=0.5)
        retriever.index_nodes(sample_nodes)

        results = retriever.retrieve(
            "machine learning artificial intelligence",
            sample_vector_results,
            top_k=2
        )

        # Should return combined results
        assert len(results) == 2
        assert all(isinstance(r, NodeWithScore) for r in results)

        # Scores should be normalized
        for result in results:
            assert 0 <= result.score <= 1

    def test_different_alpha_values(self, sample_nodes, sample_vector_results):
        """Test retrieval with different alpha values."""
        for alpha in [0.0, 0.5, 1.0]:
            retriever = HybridRetriever(alpha=alpha)
            retriever.index_nodes(sample_nodes)

            results = retriever.retrieve(
                "python programming",
                sample_vector_results,
                top_k=3
            )

            assert len(results) <= 3
            assert all(isinstance(r, NodeWithScore) for r in results)

    def test_normalize_scores(self):
        """Test score normalization."""
        retriever = HybridRetriever()
        scores = np.array([0.5, 1.0, 2.0, 3.0])

        normalized = retriever._normalize_scores(scores)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == len(scores)

    def test_normalize_equal_scores(self):
        """Test normalization with equal scores."""
        retriever = HybridRetriever()
        scores = np.array([1.0, 1.0, 1.0])

        normalized = retriever._normalize_scores(scores)

        # All should be 1.0
        assert all(s == 1.0 for s in normalized)


class TestVectorOnlyRetriever:
    """Test vector-only retrieval."""

    def test_retrieve(self, sample_vector_results):
        """Test vector-only retrieval."""
        retriever = VectorOnlyRetriever()

        results = retriever.retrieve(
            "test query",
            sample_vector_results,
            top_k=2
        )

        # Should return first top_k results
        assert len(results) == 2
        assert results[0].score == 0.9
        assert results[1].score == 0.7

    def test_retrieve_all(self, sample_vector_results):
        """Test retrieving all results."""
        retriever = VectorOnlyRetriever()

        results = retriever.retrieve(
            "test query",
            sample_vector_results,
            top_k=10  # More than available
        )

        # Should return all available results
        assert len(results) == len(sample_vector_results)


class TestRetrieverFactory:
    """Test retriever factory function."""

    def test_get_hybrid_retriever(self):
        """Test getting hybrid retriever."""
        retriever = get_retriever(use_hybrid=True)

        assert isinstance(retriever, HybridRetriever)

    def test_get_vector_only_retriever(self):
        """Test getting vector-only retriever."""
        retriever = get_retriever(use_hybrid=False)

        assert isinstance(retriever, VectorOnlyRetriever)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self, sample_nodes, sample_vector_results):
        """Test with empty query."""
        retriever = HybridRetriever()
        retriever.index_nodes(sample_nodes)

        results = retriever.retrieve(
            "",
            sample_vector_results,
            top_k=2
        )

        # Should handle gracefully
        assert isinstance(results, list)

    def test_empty_nodes_list(self, sample_vector_results):
        """Test with empty nodes list."""
        retriever = HybridRetriever()
        retriever.index_nodes([])

        results = retriever.retrieve(
            "test query",
            sample_vector_results,
            top_k=2
        )

        # Should fallback to vector results
        assert len(results) == 2

    def test_top_k_zero(self, sample_nodes, sample_vector_results):
        """Test with top_k=0."""
        retriever = HybridRetriever()
        retriever.index_nodes(sample_nodes)

        results = retriever.retrieve(
            "test",
            sample_vector_results,
            top_k=0
        )

        assert len(results) == 0

    def test_special_characters_query(self, sample_nodes, sample_vector_results):
        """Test query with special characters."""
        retriever = HybridRetriever()
        retriever.index_nodes(sample_nodes)

        results = retriever.retrieve(
            "test@#$%^&*()",
            sample_vector_results,
            top_k=2
        )

        # Should handle gracefully
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])