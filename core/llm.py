"""
LLM initialization and management.

Provides singleton-based loading of local LLM models in GGUF format,
optimized for CPU-only inference on resource-constrained devices.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

from llama_index.llms.llama_cpp import LlamaCPP

from config import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manages LLM model lifecycle with singleton pattern.

    The model is loaded once and reused to avoid repeated initialization
    overhead (GGUF loading can take 10-30 seconds).

    Attributes:
        _instance: Singleton LLM instance.
    """

    _instance: Optional[LlamaCPP] = None

    @classmethod
    def get_llm(cls, force_reload: bool = False) -> LlamaCPP:
        """
        Get or create the LLM instance (singleton).

        Args:
            force_reload: Force reload even if model is already loaded.

        Returns:
            The LlamaCPP model instance.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If model fails to load.
        """
        if cls._instance is None or force_reload:
            model_path = Path(settings.LLM_MODEL_PATH)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"LLM model not found at: {model_path}\n"
                    "Download a GGUF model and place it in the models/ directory."
                )

            logger.info(f"Loading LLM: {model_path.name}")
            logger.info(
                f"Config: temp={settings.LLM_TEMPERATURE}, "
                f"max_tokens={settings.LLM_MAX_TOKENS}, "
                f"ctx={settings.LLM_CONTEXT_WINDOW}"
            )

            try:
                cls._instance = LlamaCPP(
                    model_path=str(model_path),
                    temperature=settings.LLM_TEMPERATURE,
                    max_new_tokens=settings.LLM_MAX_TOKENS,
                    context_window=settings.LLM_CONTEXT_WINDOW,
                    model_kwargs={
                        "n_gpu_layers": 0,  # Force CPU-only
                        "n_threads": settings.LLM_THREADS,
                        "n_batch": settings.LLM_BATCH_SIZE,
                        "use_mlock": True,  # Lock model in RAM
                        "use_mmap": False,  # Disable mmap for stability
                        "low_vram": True,   # Enable low memory mode
                    },
                    verbose=False,
                )
                logger.info("LLM loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                raise RuntimeError(f"LLM initialization failed: {e}") from e

        return cls._instance

    @classmethod
    def unload_model(cls) -> None:
        """Unload the LLM to free memory."""
        if cls._instance is not None:
            logger.info("Unloading LLM model")
            cls._instance = None
            gc.collect()


def get_llm() -> LlamaCPP:
    """
    Convenience function to get the LLM.

    Returns:
        The singleton LlamaCPP instance.
    """
    return LLMManager.get_llm()