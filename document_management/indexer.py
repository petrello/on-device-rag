"""Indexing helpers to push nodes/documents into configured vector store."""

import logging
from typing import List
from config import settings
from storage import get_vector_store
from llama_index.core.schema import TextNode, Document

logger = logging.getLogger(__name__)


class Indexer:
    """High level indexer that writes documents or nodes into the vector store."""

    def __init__(self):
        self.store = get_vector_store()

    def index_documents(self, documents: List[Document]) -> None:
        """Convert documents to nodes (TextNode) and add to store.

        For simplicity we create TextNode objects with document text and metadata.
        """
        nodes: List[TextNode] = []
        for doc in documents:
            node = TextNode(text=doc.text or "", metadata=doc.metadata or {})
            nodes.append(node)

        logger.info(f"Indexing {len(nodes)} nodes into vector store")
        self.store.add_nodes(nodes)

    def clear_index(self) -> None:
        """Clear the underlying vector store entirely."""
        logger.warning("Clearing vector store via Indexer.clear_index")
        self.store.delete_all()
