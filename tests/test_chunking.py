"""
Tests for chunking module.

Tests hierarchical and simple chunking implementations.
"""

import pytest
from llama_index.core.schema import Document
from core.chunking import HierarchicalChunker, SimpleChunker, get_chunker


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    text = "This is a test document. " * 100  # ~2500 chars
    return Document(text=text, metadata={"file_name": "test.pdf"})


@pytest.fixture
def sample_documents():
    """Create multiple sample documents."""
    docs = []
    for i in range(3):
        text = f"Document {i}. " * 100
        docs.append(Document(
            text=text,
            metadata={"file_name": f"test_{i}.pdf"}
        ))
    return docs


class TestHierarchicalChunker:
    """Test hierarchical chunking functionality."""

    def test_initialization(self):
        """Test chunker initialization with custom sizes."""
        chunker = HierarchicalChunker(
            child_chunk_size=400,
            parent_chunk_size=1200
        )

        assert chunker.child_chunk_size == 400
        assert chunker.parent_chunk_size == 1200

    def test_chunk_documents(self, sample_document):
        """Test that documents are split into hierarchical chunks."""
        chunker = HierarchicalChunker(
            child_chunk_size=100,
            parent_chunk_size=300
        )

        child_nodes, parent_map = chunker.chunk_documents([sample_document])

        assert len(child_nodes) > 0
        assert len(parent_map) > 0
        # All children should have parent mappings
        for child in child_nodes:
            assert child.node_id in parent_map

    def test_parent_child_relationship(self, sample_document):
        """Test parent-child relationship via metadata."""
        chunker = HierarchicalChunker(
            child_chunk_size=100,
            parent_chunk_size=300
        )

        child_nodes, parent_map = chunker.chunk_documents([sample_document])

        for child in child_nodes:
            assert "parent_id" in child.metadata
            assert "child_index" in child.metadata

    def test_multiple_documents(self, sample_documents):
        """Test chunking preserves document source information."""
        chunker = HierarchicalChunker()

        child_nodes, parent_map = chunker.chunk_documents(sample_documents)

        assert len(child_nodes) > len(sample_documents)

        # Check file names are preserved
        file_names = {
            child.metadata.get("file_name")
            for child in child_nodes
            if child.metadata.get("file_name")
        }
        assert len(file_names) == len(sample_documents)

    def test_get_stats(self, sample_documents):
        """Test statistics generation."""
        chunker = HierarchicalChunker()

        stats = chunker.get_stats(sample_documents)

        assert stats["total_documents"] == len(sample_documents)
        assert "total_parent_chunks" in stats
        assert "total_child_chunks" in stats
        assert stats["total_child_chunks"] > 0


class TestSimpleChunker:
    """Test simple chunking functionality."""

    def test_initialization(self):
        """Test chunker initialization."""
        chunker = SimpleChunker(chunk_size=500, overlap=50)

        assert chunker.splitter is not None

    def test_chunk_documents(self, sample_document):
        """Test document chunking."""
        chunker = SimpleChunker(chunk_size=200)

        nodes, parent_map = chunker.chunk_documents([sample_document])

        # Should create chunks
        assert len(nodes) > 0
        # No parent map for simple chunking
        assert len(parent_map) == 0

    def test_metadata_preservation(self, sample_document):
        """Test that metadata is preserved."""
        chunker = SimpleChunker()

        nodes, _ = chunker.chunk_documents([sample_document])

        for node in nodes:
            assert "file_name" in node.metadata
            assert node.metadata["file_name"] == "test.pdf"


class TestChunkerFactory:
    """Test chunker factory function."""

    def test_get_hierarchical_chunker(self):
        """Test getting hierarchical chunker."""
        chunker = get_chunker(use_hierarchical=True)

        assert isinstance(chunker, HierarchicalChunker)

    def test_get_simple_chunker(self):
        """Test getting simple chunker."""
        chunker = get_chunker(use_hierarchical=False)

        assert isinstance(chunker, SimpleChunker)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_document(self):
        """Test with empty document."""
        chunker = HierarchicalChunker()
        doc = Document(text="", metadata={})

        child_nodes, parent_map = chunker.chunk_documents([doc])

        # Should handle gracefully
        assert isinstance(child_nodes, list)
        assert isinstance(parent_map, dict)

    def test_very_small_document(self):
        """Test with very small document."""
        chunker = HierarchicalChunker(child_chunk_size=100)
        doc = Document(text="Short text.", metadata={})

        child_nodes, parent_map = chunker.chunk_documents([doc])

        # Should create at least one chunk
        assert len(child_nodes) >= 1

    def test_large_document(self):
        """Test with large document."""
        chunker = HierarchicalChunker()
        text = "This is a test. " * 10000  # Large document
        doc = Document(text=text, metadata={})

        child_nodes, parent_map = chunker.chunk_documents([doc])

        # Should create many chunks
        assert len(child_nodes) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])