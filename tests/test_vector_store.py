"""
Tests for vector store module.

Tests the abstract interface and concrete implementations (Qdrant, FAISS).
"""

import pytest
from unittest.mock import Mock, patch
from llama_index.core.schema import TextNode
from storage.vector_store import VectorStoreInterface


class MockVectorStore(VectorStoreInterface):
    """Mock vector store implementation for testing."""

    def __init__(self):
        self.nodes = []
        self._storage_context = None
        self.stats = {
            "backend": "mock",
            "vector_count": 0,
        }

    def add_nodes(self, nodes):
        self.nodes.extend(nodes)
        self.stats["vector_count"] = len(self.nodes)

    def query(self, query_embedding, top_k=3):
        return self.nodes[:top_k]

    def get_index(self):
        return Mock()

    def get_storage_context(self):
        if self._storage_context is None:
            self._storage_context = Mock()
        return self._storage_context

    def delete_all(self):
        self.nodes = []
        self._storage_context = None
        self.stats["vector_count"] = 0

    def get_stats(self):
        return self.stats


@pytest.fixture
def mock_store():
    """Create mock vector store instance."""
    return MockVectorStore()


@pytest.fixture
def sample_nodes():
    """Create sample text nodes for testing."""
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
    """Test the abstract vector store interface via mock implementation."""

    def test_add_nodes(self, mock_store, sample_nodes):
        """Test adding nodes updates the store."""
        mock_store.add_nodes(sample_nodes)

        assert len(mock_store.nodes) == len(sample_nodes)
        assert mock_store.stats["vector_count"] == len(sample_nodes)

    def test_query(self, mock_store, sample_nodes):
        """Test querying returns top_k results."""
        mock_store.add_nodes(sample_nodes)

        results = mock_store.query([0.1] * 768, top_k=3)

        assert len(results) == 3

    def test_get_storage_context(self, mock_store):
        """Test get_storage_context returns a context object."""
        context = mock_store.get_storage_context()

        assert context is not None
        # Should return same instance on subsequent calls
        assert mock_store.get_storage_context() is context

    def test_delete_all(self, mock_store, sample_nodes):
        """Test delete_all clears all nodes."""
        mock_store.add_nodes(sample_nodes)
        assert len(mock_store.nodes) > 0

        mock_store.delete_all()

        assert len(mock_store.nodes) == 0
        assert mock_store.stats["vector_count"] == 0
        assert mock_store._storage_context is None

    def test_get_stats(self, mock_store):
        """Test get_stats returns expected keys."""
        stats = mock_store.get_stats()

        assert "backend" in stats
        assert "vector_count" in stats
        assert stats["backend"] == "mock"


class TestQdrantStore:
    """Test Qdrant vector store implementation."""

    @patch('storage.qdrant_store.QdrantClient')
    def test_initialization(self, mock_client_class):
        """Test Qdrant store connects and creates collection."""
        from storage.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client

        store = QdrantStore()

        assert store.client is not None
        mock_client.get_collections.assert_called()

    @patch('storage.qdrant_store.QdrantClient')
    def test_collection_creation(self, mock_client_class):
        """Test collection is created when it doesn't exist."""
        from storage.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client

        QdrantStore()

        mock_client.recreate_collection.assert_called_once()

    @patch('storage.qdrant_store.QdrantClient')
    def test_existing_collection_reused(self, mock_client_class):
        """Test existing collection is reused, not recreated."""
        from storage.qdrant_store import QdrantStore
        from config import settings

        mock_collection = Mock()
        mock_collection.name = settings.QDRANT_COLLECTION
        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[mock_collection])
        mock_client_class.return_value = mock_client

        QdrantStore()

        mock_client.recreate_collection.assert_not_called()

    @patch('storage.qdrant_store.QdrantClient')
    def test_get_stats(self, mock_client_class):
        """Test statistics retrieval."""
        from storage.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.get_collection.return_value = Mock(points_count=100)
        mock_client_class.return_value = mock_client

        store = QdrantStore()
        stats = store.get_stats()

        assert stats["backend"] == "qdrant"
        assert stats["vector_count"] == 100

    @patch('storage.qdrant_store.QdrantClient')
    def test_get_storage_context(self, mock_client_class):
        """Test storage context is created and cached."""
        from storage.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client

        store = QdrantStore()
        context1 = store.get_storage_context()
        context2 = store.get_storage_context()

        assert context1 is context2  # Same instance

    @patch('storage.qdrant_store.QdrantClient')
    def test_delete_all_recreates_collection(self, mock_client_class):
        """Test delete_all recreates the collection."""
        from storage.qdrant_store import QdrantStore

        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client_class.return_value = mock_client

        store = QdrantStore()
        mock_client.recreate_collection.reset_mock()

        store.delete_all()

        mock_client.delete_collection.assert_called_once()
        mock_client.recreate_collection.assert_called_once()


class TestLocalStore:
    """Test local FAISS vector store implementation."""

    @patch('storage.local_store.faiss')
    def test_initialization(self, mock_faiss):
        """Test FAISS store initializes correctly."""
        from storage.local_store import LocalStore

        mock_faiss.IndexFlatL2.return_value = Mock()

        store = LocalStore()

        assert store.faiss_index is not None
        assert isinstance(store.nodes, list)

    @patch('storage.local_store.faiss')
    def test_add_nodes(self, mock_faiss, sample_nodes):
        """Test adding nodes to FAISS store."""
        from storage.local_store import LocalStore

        mock_faiss.IndexFlatL2.return_value = Mock()

        store = LocalStore()
        store.add_nodes(sample_nodes)

        assert len(store.nodes) == len(sample_nodes)

    @patch('storage.local_store.faiss')
    def test_get_stats(self, mock_faiss):
        """Test statistics retrieval."""
        from storage.local_store import LocalStore

        mock_faiss.IndexFlatL2.return_value = Mock()

        store = LocalStore()
        stats = store.get_stats()

        assert stats["backend"] == "faiss"
        assert "vector_count" in stats
        assert "disk_size_mb" in stats

    @patch('storage.local_store.faiss')
    def test_get_storage_context(self, mock_faiss):
        """Test storage context is created and cached."""
        from storage.local_store import LocalStore

        mock_faiss.IndexFlatL2.return_value = Mock()

        store = LocalStore()
        context1 = store.get_storage_context()
        context2 = store.get_storage_context()

        assert context1 is context2

    @patch('storage.local_store.faiss')
    def test_delete_all_resets_index(self, mock_faiss):
        """Test delete_all resets the FAISS index."""
        from storage.local_store import LocalStore

        mock_faiss.IndexFlatL2.return_value = Mock()

        store = LocalStore()
        store.nodes = [Mock(), Mock()]
        store._storage_context = Mock()

        store.delete_all()

        assert len(store.nodes) == 0
        assert store._storage_context is None
        # Should create new index
        assert mock_faiss.IndexFlatL2.call_count == 2


class TestStorageFactory:
    """Test storage factory function."""

    @patch('storage.qdrant_store.QdrantClient')
    def test_get_qdrant_store(self, mock_client_class):
        """Test factory returns QdrantStore for qdrant type."""
        from storage import get_vector_store
        from storage.qdrant_store import QdrantStore
        from config import settings

        # Temporarily override setting
        original = settings.VECTOR_STORE_TYPE
        try:
            object.__setattr__(settings, 'VECTOR_STORE_TYPE', 'qdrant')

            mock_client = Mock()
            mock_client.get_collections.return_value = Mock(collections=[])
            mock_client_class.return_value = mock_client

            store = get_vector_store()

            assert isinstance(store, QdrantStore)
        finally:
            object.__setattr__(settings, 'VECTOR_STORE_TYPE', original)

    @patch('storage.local_store.faiss')
    def test_get_local_store(self, mock_faiss):
        """Test factory returns LocalStore for faiss type."""
        from storage import get_vector_store
        from storage.local_store import LocalStore
        from config import settings

        original = settings.VECTOR_STORE_TYPE
        try:
            object.__setattr__(settings, 'VECTOR_STORE_TYPE', 'faiss')

            mock_faiss.IndexFlatL2.return_value = Mock()

            store = get_vector_store()

            assert isinstance(store, LocalStore)
        finally:
            object.__setattr__(settings, 'VECTOR_STORE_TYPE', original)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_nodes_list(self, mock_store):
        """Test adding empty nodes list."""
        mock_store.add_nodes([])

        assert len(mock_store.nodes) == 0

    def test_query_with_zero_top_k(self, mock_store, sample_nodes):
        """Test query with top_k=0 returns empty list."""
        mock_store.add_nodes(sample_nodes)

        results = mock_store.query([0.1] * 768, top_k=0)

        assert len(results) == 0

    def test_query_empty_store(self, mock_store):
        """Test querying empty store returns empty list."""
        results = mock_store.query([0.1] * 768, top_k=3)

        assert len(results) == 0

    def test_query_top_k_greater_than_nodes(self, mock_store, sample_nodes):
        """Test query with top_k greater than available nodes."""
        mock_store.add_nodes(sample_nodes[:2])

        results = mock_store.query([0.1] * 768, top_k=10)

        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])