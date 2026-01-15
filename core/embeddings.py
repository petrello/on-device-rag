"""
Embedding model management.

Provides singleton-based loading and caching of embedding models
to minimize memory usage on resource-constrained devices.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding model lifecycle with singleton pattern.

    The embedding model is loaded once and reused across requests
    to avoid repeated initialization overhead and memory pressure.

    Attributes:
        _instance: Singleton embedding model instance.
    """

    _instance: Optional[HuggingFaceEmbedding] = None

    @classmethod
    def get_embedding_model(cls, force_reload: bool = False) -> HuggingFaceEmbedding:
        """
        Get or create the embedding model (singleton).

        Args:
            force_reload: Force reload even if model is already loaded.

        Returns:
            The HuggingFace embedding model instance.

        Raises:
            RuntimeError: If model fails to load.
        """
        if cls._instance is None or force_reload:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")

            try:
                cls._instance = HuggingFaceEmbedding(
                    model_name=settings.EMBEDDING_MODEL,
                    device="cpu",
                    cache_folder=str(settings.EMBEDDINGS_DIR),
                    embed_batch_size=32,  # Batch for efficiency
                )
                logger.info(
                    f"Embedding model loaded (dim={settings.EMBEDDING_DIM})"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Embedding model initialization failed: {e}") from e

        return cls._instance

    @classmethod
    def unload_model(cls) -> None:
        """Unload the embedding model to free memory."""
        if cls._instance is not None:
            logger.info("Unloading embedding model")
            cls._instance = None
            gc.collect()


def get_embedding_model() -> HuggingFaceEmbedding:
    """
    Convenience function to get the embedding model.

    Returns:
        The singleton HuggingFace embedding model.
    """
    return EmbeddingManager.get_embedding_model()