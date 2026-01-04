"""
Indexing and refresh logic.
Manages document indexing into vector store.
"""

import logging
from typing import List, Dict, Optional
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import TextNode
from config import settings
from core import get_chunker
from monitoring import document_count

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Manages document indexing operations."""

    def __init__(self, vector_store, retriever=None):
        """
        Initialize indexer.

        Args:
            vector_store: Vector store instance
            retriever: Retriever instance (for hybrid search)
        """
        self.vector_store = vector_store
        self.retriever = retriever
        self.chunker = get_chunker()
        self.parent_map: Dict[str, str] = {}

    def index_documents(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> tuple[VectorStoreIndex, Dict[str, str]]:
        """
        Index documents into vector store.

        Args:
            documents: Documents to index
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (index, parent_map)
        """
        try:
            logger.info(f"Indexing {len(documents)} documents...")

            # Create chunks
            if settings.USE_HIERARCHICAL_CHUNKING:
                child_nodes, self.parent_map = self.chunker.chunk_documents(documents)
                logger.info(
                    f"Created {len(child_nodes)} child chunks "
                    f"from {len(set(self.parent_map.values()))} parents"
                )
                nodes_to_index = child_nodes
            else:
                nodes_to_index, _ = self.chunker.chunk_documents(documents)
                logger.info(f"Created {len(nodes_to_index)} chunks")

            # Get storage context from vector store
            storage_context = self.vector_store.get_storage_context()

            # Build index
            index = VectorStoreIndex(
                nodes_to_index,
                storage_context=storage_context,
                show_progress=show_progress
            )

            # Index for hybrid retrieval if enabled
            if self.retriever and hasattr(self.retriever, 'index_nodes'):
                logger.info("Indexing nodes for hybrid retrieval...")
                self.retriever.index_nodes(nodes_to_index)

            # Update metrics
            if settings.ENABLE_METRICS:
                document_count.set(len(documents))

            logger.info("Indexing complete")
            return index, self.parent_map

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise

    def add_documents(
        self,
        index: VectorStoreIndex,
        documents: List[Document]
    ) -> VectorStoreIndex:
        """
        Add new documents to existing index.

        Args:
            index: Existing index
            documents: New documents to add

        Returns:
            Updated index
        """
        try:
            logger.info(f"Adding {len(documents)} documents to index...")

            # Create chunks
            if settings.USE_HIERARCHICAL_CHUNKING:
                child_nodes, parent_map = self.chunker.chunk_documents(documents)
                self.parent_map.update(parent_map)
                nodes_to_add = child_nodes
            else:
                nodes_to_add, _ = self.chunker.chunk_documents(documents)

            # Add to index
            for node in nodes_to_add:
                index.insert(node)

            # Update retriever if needed
            if self.retriever and hasattr(self.retriever, 'index_nodes'):
                all_nodes = list(index.docstore.docs.values())
                self.retriever.index_nodes(all_nodes)

            logger.info(f"Added {len(nodes_to_add)} chunks to index")
            return index

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def remove_documents(
        self,
        index: VectorStoreIndex,
        document_ids: List[str]
    ) -> VectorStoreIndex:
        """
        Remove documents from index.

        Args:
            index: Existing index
            document_ids: IDs of documents to remove

        Returns:
            Updated index
        """
        try:
            logger.info(f"Removing {len(document_ids)} documents from index...")

            # Remove from index
            for doc_id in document_ids:
                try:
                    index.delete_ref_doc(doc_id, delete_from_docstore=True)
                except Exception as e:
                    logger.warning(f"Failed to remove document {doc_id}: {e}")

            # Update retriever if needed
            if self.retriever and hasattr(self.retriever, 'index_nodes'):
                all_nodes = list(index.docstore.docs.values())
                self.retriever.index_nodes(all_nodes)

            logger.info("Documents removed from index")
            return index

        except Exception as e:
            logger.error(f"Failed to remove documents: {e}")
            raise

    def refresh_index(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> tuple[VectorStoreIndex, Dict[str, str]]:
        """
        Refresh entire index (clear and rebuild).

        Args:
            documents: All documents to index
            show_progress: Whether to show progress

        Returns:
            Tuple of (new_index, parent_map)
        """
        try:
            logger.info("Refreshing index (full rebuild)...")

            # Clear existing index
            self.vector_store.delete_all()
            self.parent_map = {}

            # Rebuild index
            return self.index_documents(documents, show_progress)

        except Exception as e:
            logger.error(f"Failed to refresh index: {e}")
            raise

    def get_indexing_stats(self, index: VectorStoreIndex) -> dict:
        """
        Get indexing statistics.

        Args:
            index: Vector store index

        Returns:
            Statistics dictionary
        """
        try:
            all_nodes = list(index.docstore.docs.values())

            # Count unique documents
            unique_docs = set()
            for node in all_nodes:
                if isinstance(node, TextNode):
                    file_name = node.metadata.get('file_name', 'unknown')
                    unique_docs.add(file_name)

            vector_stats = self.vector_store.get_stats()

            return {
                "total_chunks": len(all_nodes),
                "unique_documents": len(unique_docs),
                "parent_chunks": len(set(self.parent_map.values())),
                "hierarchical_enabled": settings.USE_HIERARCHICAL_CHUNKING,
                "vector_store": vector_stats
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}