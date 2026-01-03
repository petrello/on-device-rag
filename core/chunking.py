"""
Hierarchical chunking implementation.
Creates parent-child chunk relationships for better retrieval.
"""

import hashlib
from typing import Dict, List, Tuple
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter
from config import settings


class HierarchicalChunker:
    """
    Implements hierarchical chunking strategy.

    Strategy:
    - Create large parent chunks (1200 tokens) for context
    - Split parents into small child chunks (400 tokens) for retrieval
    - Index only children in vector store
    - Map children back to parents for LLM context expansion
    """

    def __init__(
            self,
            child_chunk_size: int = None,
            parent_chunk_size: int = None,
            overlap: int = None
    ):
        """
        Initialize chunker with configurable sizes.

        Args:
            child_chunk_size: Size of child chunks for retrieval
            parent_chunk_size: Size of parent chunks for context
            overlap: Overlap between chunks
        """
        child_chunk_size = child_chunk_size or settings.CHUNK_SIZE
        parent_chunk_size = parent_chunk_size or settings.PARENT_CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        self.child_splitter = SentenceSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=overlap
        )

        self.parent_splitter = SentenceSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=overlap * 2  # More overlap for parents
        )

        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size

    def chunk_documents(
            self,
            documents: List[Document]
    ) -> Tuple[List[TextNode], Dict[str, str]]:
        """
        Create hierarchical chunks from documents.

        Args:
            documents: List of documents to chunk

        Returns:
            Tuple of (child_nodes, parent_map) where:
            - child_nodes: List of child chunks for vector store
            - parent_map: Dict mapping child_id -> parent_text for expansion
        """
        child_nodes = []
        parent_map = {}

        for doc in documents:
            # Create parent chunks first
            parent_chunks = self.parent_splitter.get_nodes_from_documents([doc])

            for parent_idx, parent in enumerate(parent_chunks):
                parent_id = self._generate_id(parent.text)

                # Create document from parent for child splitting
                child_doc = Document(
                    text=parent.text,
                    metadata={
                        **parent.metadata,
                        "parent_id": parent_id,
                        "parent_chunk_index": parent_idx
                    }
                )

                # Create child chunks from parent
                children = self.child_splitter.get_nodes_from_documents([child_doc])

                # Link children to parent
                for child_idx, child in enumerate(children):
                    child.metadata["parent_id"] = parent_id
                    child.metadata["child_index"] = child_idx
                    child_nodes.append(child)

                    # Store parent text for later retrieval
                    parent_map[child.node_id] = parent.text

        return child_nodes, parent_map

    @staticmethod
    def _generate_id(text: str) -> str:
        """Generate deterministic ID from text content."""
        return hashlib.md5(text.encode()).hexdigest()

    def get_stats(self, documents: List[Document]) -> Dict:
        """
        Get chunking statistics.

        Args:
            documents: Documents to analyze

        Returns:
            Dict with statistics
        """
        child_nodes, parent_map = self.chunk_documents(documents)

        return {
            "total_documents": len(documents),
            "total_parent_chunks": len(set(parent_map.values())),
            "total_child_chunks": len(child_nodes),
            "avg_children_per_parent": len(child_nodes) / max(len(set(parent_map.values())), 1),
            "child_chunk_size": self.child_chunk_size,
            "parent_chunk_size": self.parent_chunk_size
        }


class SimpleChunker:
    """
    Simple chunking without hierarchy (fallback option).
    """

    def __init__(
            self,
            chunk_size: int = None,
            overlap: int = None
    ):
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def chunk_documents(
            self,
            documents: List[Document]
    ) -> Tuple[List[TextNode], Dict[str, str]]:
        """
        Create simple chunks without hierarchy.

        Returns same format as HierarchicalChunker for compatibility.
        """
        nodes = self.splitter.get_nodes_from_documents(documents)
        # Empty parent map (no hierarchy)
        parent_map = {}
        return nodes, parent_map


def get_chunker(use_hierarchical: bool = None) -> HierarchicalChunker | SimpleChunker:
    """
    Factory function to get appropriate chunker.

    Args:
        use_hierarchical: Override settings if provided

    Returns:
        Chunker instance
    """
    if use_hierarchical is None:
        use_hierarchical = settings.USE_HIERARCHICAL_CHUNKING

    if use_hierarchical:
        return HierarchicalChunker()
    else:
        return SimpleChunker()