"""
Hierarchical chunking implementation.

Creates parent-child chunk relationships for better retrieval precision.
Small child chunks are indexed for precise matching, while parent chunks
provide expanded context to the LLM.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

from config import settings

if TYPE_CHECKING:
    from llama_index.core.node_parser import NodeParser


class HierarchicalChunker:
    """
    Hierarchical chunking strategy for improved RAG retrieval.

    This chunker creates a two-level hierarchy:
    - **Parent chunks**: Provide broader context to the LLM.
    - **Child chunks**: Indexed for precise vector search.

    Only child chunks are stored in the vector index. When retrieved, the
    corresponding parent text is expanded for LLM context, improving answer
    quality without sacrificing retrieval precision.

    Attributes:
        child_splitter: Splitter for creating small retrieval chunks.
        parent_splitter: Splitter for creating larger context chunks.
        child_chunk_size: Token size for child chunks.
        parent_chunk_size: Token size for parent chunks.
    """

    __slots__ = ('child_splitter', 'parent_splitter', 'child_chunk_size', 'parent_chunk_size')

    def __init__(
        self,
        child_chunk_size: Optional[int] = None,
        parent_chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> None:
        """
        Initialize the hierarchical chunker.

        Args:
            child_chunk_size: Token size for child chunks (default from settings).
            parent_chunk_size: Token size for parent chunks (default from settings).
            overlap: Token overlap between consecutive chunks.
        """
        child_chunk_size = child_chunk_size or settings.CHUNK_SIZE
        parent_chunk_size = parent_chunk_size or settings.PARENT_CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        self.child_splitter: NodeParser = SentenceSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=overlap,
        )
        self.parent_splitter: NodeParser = SentenceSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=overlap * 2,  # Larger overlap for parent continuity
        )
        self.child_chunk_size: int = child_chunk_size
        self.parent_chunk_size: int = parent_chunk_size

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> Tuple[List[TextNode], Dict[str, str]]:
        """
        Create hierarchical chunks from documents.

        This method:
        1. Splits each document into parent chunks.
        2. Splits each parent into child chunks.
        3. Links children to parents via metadata and a mapping dict.

        Args:
            documents: Source documents to chunk.

        Returns:
            A tuple containing:
                - child_nodes: Child chunks for vector store indexing.
                - parent_map: Mapping of child_id -> parent_text for context expansion.
        """
        child_nodes: List[TextNode] = []
        parent_map: Dict[str, str] = {}

        for doc in documents:
            # Create parent chunks first
            parent_chunks = self.parent_splitter.get_nodes_from_documents([doc])

            for parent_idx, parent in enumerate(parent_chunks):
                parent_id = self._generate_id(parent.text)

                # Wrap parent text as Document for child splitting
                child_doc = Document(
                    text=parent.text,
                    metadata={
                        **parent.metadata,
                        "parent_id": parent_id,
                        "parent_chunk_index": parent_idx,
                    },
                )

                # Split parent into children
                children = self.child_splitter.get_nodes_from_documents([child_doc])

                for child_idx, child in enumerate(children):
                    child.metadata["parent_id"] = parent_id
                    child.metadata["child_index"] = child_idx
                    child_nodes.append(child)
                    parent_map[child.node_id] = parent.text

        return child_nodes, parent_map

    @staticmethod
    def _generate_id(text: str) -> str:
        """Generate a deterministic ID from text content using MD5."""
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

    def get_stats(self, documents: List[Document]) -> Dict[str, int | float]:
        """
        Compute chunking statistics for the given documents.

        Args:
            documents: Documents to analyze.

        Returns:
            Statistics dictionary with chunk counts and sizes.
        """
        child_nodes, parent_map = self.chunk_documents(documents)
        unique_parents = len(set(parent_map.values()))

        return {
            "total_documents": len(documents),
            "total_parent_chunks": unique_parents,
            "total_child_chunks": len(child_nodes),
            "avg_children_per_parent": len(child_nodes) / max(unique_parents, 1),
            "child_chunk_size": self.child_chunk_size,
            "parent_chunk_size": self.parent_chunk_size,
        }


class SimpleChunker:
    """
    Simple flat chunking without hierarchy.

    Use this when memory is extremely constrained or hierarchical
    context expansion is not needed. Returns an empty parent_map
    for API compatibility with HierarchicalChunker.

    Attributes:
        splitter: The sentence splitter used for chunking.
    """

    __slots__ = ('splitter',)

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> None:
        """
        Initialize the simple chunker.

        Args:
            chunk_size: Token size per chunk (default from settings).
            overlap: Token overlap between chunks (default from settings).
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        self.splitter: NodeParser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> Tuple[List[TextNode], Dict[str, str]]:
        """
        Create flat chunks from documents.

        Returns the same signature as HierarchicalChunker for compatibility,
        but parent_map is always empty.

        Args:
            documents: Documents to chunk.

        Returns:
            Tuple of (nodes, empty_parent_map).
        """
        nodes = self.splitter.get_nodes_from_documents(documents)
        return nodes, {}


def get_chunker(
    use_hierarchical: Optional[bool] = None,
) -> HierarchicalChunker | SimpleChunker:
    """
    Factory function to create the appropriate chunker.

    Args:
        use_hierarchical: If True, return HierarchicalChunker.
            If None, uses settings.USE_HIERARCHICAL_CHUNKING.

    Returns:
        Configured chunker instance.
    """
    if use_hierarchical is None:
        use_hierarchical = settings.USE_HIERARCHICAL_CHUNKING

    return HierarchicalChunker() if use_hierarchical else SimpleChunker()
