"""
Tests for vector store module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llama_index.core.schema import TextNode
from storage.vector_store import VectorStoreInterface


class MockVectorStore(VectorStoreInterface):
    """Mock vector store for testing."""

    def __init__(self):
        self.nodes = []
        self.stats = {
            "backend": "mock",
            "vector_count": 0
        }

    def add_nodes(self, nodes):
        self.nodes.extend(nodes)
        self.stats["vector_count"] = len(self.nodes)

    def query(self, query_embedding, top_k=3):
        # Return mock results
        return self.nodes[:top_k]

    def get_index(self):
        return Mock()

    def delete_all(self):
        self.nodes = []
        self.stats["vector_count"] = 0

    def get_stats(self):
        return self.stats


@pytest.fixture
def mock_store():
    """Create mock vector store."""
    return MockVectorStore()


@pytest.fixture
def sample_nodes():
    """Create sample text nodes."""
    nodes = []
    for i in range(5):
        node = TextNode(
            text=f"Sample text {i}",
            metadata={"source": f"doc{i}.pdf"}
        )
        node.node_id = f"node_{i}"
        nodes.append(node)
    return nodes


class TestVectorStoreInterface:
    """Test vector store interface."""

    def test_add_nodes(self, mock_store, sample_nodes):
        """Test adding nodes to store."""
        mock_store.add_nodes(sample_nodes)

        assert len(mock_store.nodes) == len(sample_nodes)
        assert mock_store.stats["vector_count"] == len(sample_nodes)

    def test_query(self, mock_store, sample_nodes):
        """Test querying vector store."""
        mock_store.add_nodes(sample_nodes)

        results = mock_store.query([0.1] * 384, top_k=3)

        assert len(results) == 3

    def test_delete_all(self, mock_store, sample_nodes):
        """Test deleting all nodes."""
        mock_store.add_nodes(sample_nodes)
        assert len(mock_store.nodes) > 0

        mock_store.delete_all()

        assert len(mock_store.nodes) == 0
        assert mock_store.stats["vector_count"] == 0

    def test_get_stats(self, mock_store):
        """Test getting statistics."""
        stats = mock_store.get_stats()

        assert "backend" in stats
        assert "vector_count" in stats
        assert stats["backend"] == "mock"


class TestQdrantStore:
    """Test Qdrant vector store."""

    @patch('storage.qdrant_store.QdrantClient')
    def test_initialization(self, mock_client_class):
        """Test Qdrant store initialization."""
        from storage.qdrant_store import QdrantStore

        # Mock client
        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client

        store = QdrantStore()

        assert store.client is not None
        mock_client.get_collections.assert_called()

    @patch('storage.qdrant_store.QdrantClient')
    def test_collection_creation(self, mock_client_class):
        """Test collection creation."""
        from storage.qdrant_store import QdrantStore

        # Mock client
        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client

        store = QdrantStore()

        # Should create collection
        mock_client.recreate_collection.assert_called_once()

    @patch('storage.qdrant_store.QdrantClient')
    def test_get_stats(self, mock_client_class):
        """Test getting Qdrant statistics."""
        from storage.qdrant_store import QdrantStore

        # Mock client
        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_collection_info = Mock(points_count=100)
        mock_client.get_collection.return_value = mock_collection_info
        mock_client_class.return_value = mock_client

        store = QdrantStore()
        stats = store.get_stats()

        assert "backend" in stats
        assert stats["backend"] == "qdrant"
        assert stats["vector_count"] == 100


class TestLocalStore:
    """Test local FAISS store."""

    @patch('storage.local_store.faiss')
    def test_initialization(self, mock_faiss):
        """Test local store initialization."""
        from storage.local_store import LocalStore

        # Mock FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        store = LocalStore()

        assert store.faiss_index is not None
        assert isinstance(store.nodes, list)

    @patch('storage.local_store.faiss')
    def test_add_nodes(self, mock_faiss, sample_nodes):
        """Test adding nodes to local store."""
        from storage.local_store import LocalStore

        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        store = LocalStore()
        store.add_nodes(sample_nodes)

        assert len(store.nodes) == len(sample_nodes)

    @patch('storage.local_store.faiss')
    def test_get_stats(self, mock_faiss):
        """Test getting local store statistics."""
        from storage.local_store import LocalStore

        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        store = LocalStore()
        stats = store.get_stats()

        assert "backend" in stats
        assert stats["backend"] == "faiss"
        assert "vector_count" in stats


class TestStorageFactory:
    """Test storage factory function."""

    @patch('storage.qdrant_store.QdrantClient')
    @patch('config.settings.VECTOR_STORE_TYPE', 'qdrant')
    def test_get_qdrant_store(self, mock_client):
        """Test getting Qdrant store from factory."""
        from storage import get_vector_store

        mock_client.return_value.get_collections.return_value = Mock(collections=[])

        store = get_vector_store()

        from storage.qdrant_store import QdrantStore
        assert isinstance(store, QdrantStore)

    @patch('storage.local_store.faiss')
    @patch('config.settings.VECTOR_STORE_TYPE', 'faiss')
    def test_get_local_store(self, mock_faiss):
        """Test getting local store from factory."""
        from storage import get_vector_store

        mock_faiss.IndexFlatL2.return_value = Mock()

        store = get_vector_store()

        from storage.local_store import LocalStore
        assert isinstance(store, LocalStore)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_nodes_list(self, mock_store):
        """Test with empty nodes list."""
        mock_store.add_nodes([])

        assert len(mock_store.nodes) == 0

    def test_query_with_zero_top_k(self, mock_store, sample_nodes):
        """Test query with top_k=0."""
        mock_store.add_nodes(sample_nodes)

        results = mock_store.query([0.1] * 384, top_k=0)

        assert len(results) == 0

    def test_query_empty_store(self, mock_store):
        """Test querying empty store."""
        results = mock_store.query([0.1] * 384, top_k=3)

        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])