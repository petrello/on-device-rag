"""
LLM initialization and management.

Provides singleton-based loading of local LLM models in GGUF format,
optimized for CPU-only inference on resource-constrained devices.
"""

from __future__ import annotations

import gc
import hashlib
import logging
from pathlib import Path
from typing import Optional

from llama_index.llms.llama_cpp import LlamaCPP

from config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# PROMPT CACHE CONSTANTS
# =============================================================================

# System prompt prefix that remains constant across queries.
# This is cached in the KV cache for faster subsequent inferences.
SYSTEM_PROMPT_PREFIX = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant. Use the provided context to answer the user query. "
    "If the answer is not in the context, state that you do not know. "
    "Be concise, accurate, and cite sources when possible.\n\n"
    "Context:\n"
)


def get_cache_path() -> Path:
    """Get the path for the prompt cache file."""
    cache_dir = Path(settings.MODELS_DIR) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique cache filename based on model and settings
    model_name = Path(settings.LLM_MODEL_PATH).stem
    cache_key = f"{model_name}_{settings.LLM_CONTEXT_WINDOW}_{settings.LLM_BATCH_SIZE}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]

    return cache_dir / f"prompt_cache_{cache_hash}.bin"


class LLMManager:
    """
    Manages LLM model lifecycle with singleton pattern and prompt caching.

    The model is loaded once and reused to avoid repeated initialization
    overhead (GGUF loading can take 10-30 seconds).

    Prompt Caching Strategy:
        - llama-cpp-python automatically caches KV states for common prefixes
        - By keeping the system prompt consistent, subsequent queries benefit
          from cached prompt processing
        - The cache is maintained in-memory during the session
        - Optional: Save/load cache state to disk for persistence

    Attributes:
        _instance: Singleton LLM instance.
        _cache_initialized: Whether the prompt cache has been warmed up.
    """

    _instance: Optional[LlamaCPP] = None
    _cache_initialized: bool = False
    _last_system_prompt_tokens: int = 0

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
                f"ctx={settings.LLM_CONTEXT_WINDOW}, "
                f"batch={settings.LLM_BATCH_SIZE}, "
                f"threads={settings.LLM_THREADS}"
            )

            # Check for prompt caching setting
            enable_cache = getattr(settings, 'LLM_ENABLE_PROMPT_CACHE', True)

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
                        "use_mmap": True,   # Enable mmap for faster loading
                    },
                    verbose=False,
                )

                logger.info("LLM loaded successfully")

                # Warm up the prompt cache if enabled
                if enable_cache:
                    cls._warm_up_cache()

            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                raise RuntimeError(f"LLM initialization failed: {e}") from e

        return cls._instance

    @classmethod
    def _warm_up_cache(cls) -> None:
        """
        Warm up the KV cache by processing the system prompt prefix.

        This pre-computes the KV cache for the static system prompt,
        reducing TTFT for subsequent queries that share this prefix.
        """
        if cls._cache_initialized or cls._instance is None:
            return

        try:
            logger.info("Warming up prompt cache...")

            # Access the underlying llama-cpp model to warm up cache
            # LlamaIndex's LlamaCPP wraps the llama-cpp-python Llama class
            if hasattr(cls._instance, '_model'):
                model = cls._instance._model

                # Tokenize the system prompt prefix
                tokens = model.tokenize(SYSTEM_PROMPT_PREFIX.encode('utf-8'))
                cls._last_system_prompt_tokens = len(tokens)

                # Evaluate tokens to populate KV cache (without generating)
                # This primes the cache for the system prompt
                model.eval(tokens)

                logger.info(
                    f"Prompt cache warmed up: {cls._last_system_prompt_tokens} tokens cached"
                )
                cls._cache_initialized = True
            else:
                logger.warning("Could not access underlying model for cache warmup")

        except Exception as e:
            logger.warning(f"Prompt cache warmup failed (non-critical): {e}")
            # Non-fatal - continue without cache warmup

    @classmethod
    def get_cache_stats(cls) -> dict:
        """
        Get prompt cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        return {
            "cache_initialized": cls._cache_initialized,
            "cached_tokens": cls._last_system_prompt_tokens,
            "cache_enabled": getattr(settings, 'LLM_ENABLE_PROMPT_CACHE', True),
        }

    @classmethod
    def reset_cache(cls) -> None:
        """Reset the prompt cache (useful when changing system prompts)."""
        cls._cache_initialized = False
        cls._last_system_prompt_tokens = 0

        if cls._instance is not None and hasattr(cls._instance, '_model'):
            try:
                cls._instance._model.reset()
                logger.info("Prompt cache reset")
            except Exception as e:
                logger.warning(f"Cache reset failed: {e}")

    @classmethod
    def unload_model(cls) -> None:
        """Unload the LLM to free memory and reset cache state."""
        if cls._instance is not None:
            logger.info("Unloading LLM model")
            cls._instance = None
            cls._cache_initialized = False
            cls._last_system_prompt_tokens = 0
            gc.collect()


def get_llm() -> LlamaCPP:
    """
    Convenience function to get the LLM.

    Returns:
        The singleton LlamaCPP instance.
    """
    return LLMManager.get_llm()


def get_cache_stats() -> dict:
    """
    Get prompt cache statistics.

    Returns:
        Dictionary with cache information.
    """
    return LLMManager.get_cache_stats()


def reset_prompt_cache() -> None:
    """Reset the prompt cache."""
    LLMManager.reset_cache()
