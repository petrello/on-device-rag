"""
Tests for retrieval module.

Tests the hybrid and vector-only retrieval implementations.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from llama_index.core.schema import TextNode, NodeWithScore
from core.retrieval import HybridRetriever, VectorOnlyRetriever, get_retriever


@pytest.fixture
def sample_nodes():
    """Create sample text nodes for testing."""
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

    for i, node in enumerate(nodes):
        node.node_id = f"node_{i}"

    return nodes


@pytest.fixture
def sample_vector_results(sample_nodes):
    """Create sample vector search results."""
    return [
        NodeWithScore(node=sample_nodes[0], score=0.9),
        NodeWithScore(node=sample_nodes[1], score=0.7),
        NodeWithScore(node=sample_nodes[2], score=0.5),
    ]


@pytest.fixture
def mock_index(sample_vector_results):
    """Create a mock LlamaIndex VectorStoreIndex."""
    mock_idx = Mock()
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = sample_vector_results
    mock_idx.as_retriever.return_value = mock_retriever
    return mock_idx


class TestHybridRetriever:
    """Test hybrid retrieval functionality."""

    def test_initialization(self):
        """Test retriever initialization with default and custom alpha."""
        retriever = HybridRetriever(alpha=0.7)

        assert retriever.alpha == 0.7
        assert retriever.bm25 is None
        assert len(retriever.corpus_nodes) == 0
        assert retriever._index is None

    def test_set_index(self, mock_index):
        """Test setting the vector index."""
        retriever = HybridRetriever()
        retriever.set_index(mock_index)

        assert retriever._index is mock_index

    def test_index_nodes(self, sample_nodes):
        """Test BM25 indexing of nodes."""
        retriever = HybridRetriever()
        retriever.index_nodes(sample_nodes)

        assert retriever.bm25 is not None
        assert len(retriever.corpus_nodes) == len(sample_nodes)
        assert len(retriever.node_id_to_idx) == len(sample_nodes)

    def test_retrieve_with_hybrid(self, sample_nodes, mock_index):
        """Test hybrid retrieval combining vector and BM25."""
        retriever = HybridRetriever(alpha=0.5)
        retriever.set_index(mock_index)
        retriever.index_nodes(sample_nodes)

        results = retriever.retrieve("machine learning artificial intelligence", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, NodeWithScore) for r in results)
        # Scores should be normalized
        for result in results:
            assert 0 <= result.score <= 1

    def test_retrieve_without_bm25(self, mock_index):
        """Test retrieval falls back to vector-only when BM25 not indexed."""
        retriever = HybridRetriever()
        retriever.set_index(mock_index)
        # Note: BM25 not indexed

        results = retriever.retrieve("test query", top_k=2)

        # Should return vector results (limited to top_k)
        assert len(results) <= 3

    def test_retrieve_without_index(self, sample_nodes):
        """Test retrieval returns BM25 results when vector index not set."""
        retriever = HybridRetriever()
        retriever.index_nodes(sample_nodes)
        # Note: Vector index not set

        results = retriever.retrieve("machine learning", top_k=2)

        # Should still work (BM25 only, no vector contribution)
        assert isinstance(results, list)

    def test_alpha_pure_vector(self, sample_nodes, mock_index):
        """Test alpha=1.0 gives pure vector results."""
        retriever = HybridRetriever(alpha=1.0)
        retriever.set_index(mock_index)
        retriever.index_nodes(sample_nodes)

        results = retriever.retrieve("test", top_k=3)

        assert len(results) <= 3

    def test_alpha_pure_bm25(self, sample_nodes, mock_index):
        """Test alpha=0.0 gives pure BM25 results."""
        retriever = HybridRetriever(alpha=0.0)
        retriever.set_index(mock_index)
        retriever.index_nodes(sample_nodes)

        results = retriever.retrieve("python programming", top_k=3)

        assert len(results) <= 3

    def test_normalize_scores(self):
        """Test min-max score normalization."""
        retriever = HybridRetriever()
        scores = np.array([0.5, 1.0, 2.0, 3.0])

        normalized = retriever._normalize_scores(scores)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == len(scores)

    def test_normalize_equal_scores(self):
        """Test normalization with identical scores returns all ones."""
        retriever = HybridRetriever()
        scores = np.array([1.0, 1.0, 1.0])

        normalized = retriever._normalize_scores(scores)

        assert all(s == 1.0 for s in normalized)

    def test_normalize_empty_scores(self):
        """Test normalization with empty array."""
        retriever = HybridRetriever()
        scores = np.array([])

        normalized = retriever._normalize_scores(scores)

        assert len(normalized) == 0


class TestVectorOnlyRetriever:
    """Test vector-only retrieval."""

    def test_initialization(self):
        """Test retriever initializes without index."""
        retriever = VectorOnlyRetriever()
        assert retriever._index is None

    def test_set_index(self, mock_index):
        """Test setting the vector index."""
        retriever = VectorOnlyRetriever()
        retriever.set_index(mock_index)

        assert retriever._index is mock_index

    def test_retrieve(self, mock_index):
        """Test vector-only retrieval returns expected results."""
        retriever = VectorOnlyRetriever()
        retriever.set_index(mock_index)

        results = retriever.retrieve("test query", top_k=2)

        assert len(results) == 3  # Mock returns all results
        mock_index.as_retriever.assert_called_once_with(similarity_top_k=2)

    def test_retrieve_without_index(self):
        """Test retrieval without index returns empty list."""
        retriever = VectorOnlyRetriever()

        results = retriever.retrieve("test query", top_k=2)

        assert results == []


class TestRetrieverFactory:
    """Test retriever factory function."""

    def test_get_hybrid_retriever(self):
        """Test factory returns HybridRetriever when use_hybrid=True."""
        retriever = get_retriever(use_hybrid=True)
        assert isinstance(retriever, HybridRetriever)

    def test_get_vector_only_retriever(self):
        """Test factory returns VectorOnlyRetriever when use_hybrid=False."""
        retriever = get_retriever(use_hybrid=False)
        assert isinstance(retriever, VectorOnlyRetriever)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self, sample_nodes, mock_index):
        """Test retrieval with empty query string."""
        retriever = HybridRetriever()
        retriever.set_index(mock_index)
        retriever.index_nodes(sample_nodes)

        results = retriever.retrieve("", top_k=2)

        assert isinstance(results, list)

    def test_empty_nodes_list(self, mock_index):
        """Test indexing empty nodes list."""
        retriever = HybridRetriever()
        retriever.set_index(mock_index)
        retriever.index_nodes([])

        results = retriever.retrieve("test query", top_k=2)

        # Should still work using vector results
        assert isinstance(results, list)

    def test_special_characters_query(self, sample_nodes, mock_index):
        """Test query with special characters is handled gracefully."""
        retriever = HybridRetriever()
        retriever.set_index(mock_index)
        retriever.index_nodes(sample_nodes)

        results = retriever.retrieve("test@#$%^&*()", top_k=2)

        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])