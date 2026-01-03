"""
LLM initialization and management.
Handles loading and configuration of local LLM models.
"""

import logging
from pathlib import Path
from typing import Optional
from llama_index.llms.llama_cpp import LlamaCPP
from config import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM model lifecycle."""

    _instance: Optional[LlamaCPP] = None

    @classmethod
    def get_llm(cls, force_reload: bool = False) -> LlamaCPP:
        """
        Get or create LLM instance (singleton pattern).

        Args:
            force_reload: Force reload of model

        Returns:
            LlamaCPP instance
        """
        if cls._instance is None or force_reload:
            model_path = Path(settings.LLM_MODEL_PATH)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"LLM model not found at: {model_path}\n"
                    f"Please download a GGUF model and place it in the models/ directory."
                )

            logger.info(f"Loading LLM model: {model_path}")
            logger.info(
                f"Configuration: "
                f"temperature={settings.LLM_TEMPERATURE}, "
                f"max_tokens={settings.LLM_MAX_TOKENS}, "
                f"context_window={settings.LLM_CONTEXT_WINDOW}"
            )

            try:
                cls._instance = LlamaCPP(
                    model_path=str(model_path),
                    temperature=settings.LLM_TEMPERATURE,
                    max_new_tokens=settings.LLM_MAX_TOKENS,
                    context_window=settings.LLM_CONTEXT_WINDOW,
                    model_kwargs={
                        "n_gpu_layers": 0,  # Force CPU
                        "n_threads": settings.LLM_THREADS,
                        "n_batch": settings.LLM_BATCH_SIZE,
                        "use_mlock": False,  # Don't lock memory
                        "use_mmap": True,  # Use memory mapping
                        "low_vram": True,  # Enable low VRAM mode
                    },
                    verbose=False,
                )

                logger.info("LLM model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load LLM model: {e}")
                raise

        return cls._instance

    @classmethod
    def unload_model(cls):
        """Unload model to free memory."""
        if cls._instance is not None:
            logger.info("Unloading LLM model")
            cls._instance = None

            # Force garbage collection
            import gc
            gc.collect()


def get_llm() -> LlamaCPP:
    """
    Convenience function to get LLM.

    Returns:
        LlamaCPP instance
    """
    return LLMManager.get_llm()