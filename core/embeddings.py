"""
Embedding model management.
Handles loading and caching of embedding models.
"""

import logging
from typing import Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding model lifecycle."""

    _instance: Optional[HuggingFaceEmbedding] = None

    @classmethod
    def get_embedding_model(cls, force_reload: bool = False) -> HuggingFaceEmbedding:
        """
        Get or create embedding model (singleton pattern).

        Args:
            force_reload: Force reload of model

        Returns:
            HuggingFaceEmbedding instance
        """
        if cls._instance is None or force_reload:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")

            try:
                cls._instance = HuggingFaceEmbedding(
                    model_name=settings.EMBEDDING_MODEL,
                    device="cpu",
                    cache_folder=str(settings.EMBEDDINGS_DIR),
                    embed_batch_size=32,  # Process in batches for efficiency
                )

                logger.info(
                    f"Embedding model loaded successfully. "
                    f"Dimension: {settings.EMBEDDING_DIM}"
                )

            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

        return cls._instance

    @classmethod
    def unload_model(cls):
        """Unload model to free memory."""
        if cls._instance is not None:
            logger.info("Unloading embedding model")
            cls._instance = None

            # Force garbage collection
            import gc
            gc.collect()


def get_embedding_model() -> HuggingFaceEmbedding:
    """
    Convenience function to get embedding model.

    Returns:
        HuggingFaceEmbedding instance
    """
    return EmbeddingManager.get_embedding_model()