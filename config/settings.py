"""
Configuration management with validation.

Centralized application settings loaded from environment variables.
Uses Pydantic for validation and type safety.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with validation.

    All settings can be overridden via environment variables or a `.env` file.
    Path fields are automatically created if they don't exist.
    """

    # =========================================================================
    # PATHS
    # =========================================================================
    DATA_DIR: Path = Field(
        default=Path("data"),
        description="Directory containing source documents",
    )
    MODELS_DIR: Path = Field(
        default=Path("models"),
        description="Directory for GGUF model files",
    )
    EMBEDDINGS_DIR: Path = Field(
        default=Path("embeddings"),
        description="Cache directory for embedding models",
    )

    # =========================================================================
    # VECTOR STORE
    # =========================================================================
    VECTOR_STORE_TYPE: Literal["qdrant", "faiss"] = Field(
        default="qdrant",
        description="Vector store backend: 'qdrant' or 'faiss'",
    )
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    QDRANT_COLLECTION: str = Field(
        default="edge_rag",
        description="Qdrant collection name",
    )
    FAISS_INDEX_PATH: Path = Field(
        default=Path("faiss_storage/faiss_index"),
        description="Path for FAISS index persistence",
    )

    # =========================================================================
    # MODELS
    # =========================================================================
    EMBEDDING_MODEL: str = Field(
        default="ibm-granite/granite-embedding-278m-multilingual",
        description="HuggingFace embedding model identifier",
    )
    EMBEDDING_DIM: int = Field(
        default=768,
        description="Embedding vector dimension",
    )
    LLM_MODEL_PATH: Path = Field(
        default="models/llama-3.2-1b-instruct-q8_0.gguf",
        description="Path to GGUF model file",
    )
    HF_HOME: Path = Field(
        default=Path("embeddings"),
        description="HuggingFace cache directory",
    )

    # =========================================================================
    # LLM PARAMETERS
    # =========================================================================
    LLM_TEMPERATURE: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Generation temperature (0=deterministic)",
    )
    LLM_MAX_TOKENS: int = Field(
        default=2048, ge=128, le=8192,
        description="Maximum tokens to generate",
    )
    LLM_CONTEXT_WINDOW: int = Field(
        default=8192, ge=512, le=16384,
        description="Context window size in tokens",
    )
    LLM_THREADS: int = Field(
        default=4, ge=1, le=32,
        description="CPU threads for LLM inference",
    )
    LLM_BATCH_SIZE: int = Field(
        default=512, ge=32, le=2048,
        description="Batch size for token processing",
    )
    LLM_ENABLE_PROMPT_CACHE: bool = Field(
        default=True,
        description="Enable prompt caching for faster TTFT",
    )

    # =========================================================================
    # CHUNKING STRATEGY
    # =========================================================================
    CHUNK_SIZE: int = Field(
        default=400, ge=100, le=5000,
        description="Child chunk size in tokens",
    )
    CHUNK_OVERLAP: int = Field(
        default=40, ge=0, le=500,
        description="Overlap between consecutive chunks",
    )
    USE_HIERARCHICAL_CHUNKING: bool = Field(
        default=False,
        description="Enable parent-child chunk hierarchy",
    )
    PARENT_CHUNK_SIZE: int = Field(
        default=1200, ge=300, le=5000,
        description="Parent chunk size in tokens",
    )

    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    SIMILARITY_TOP_K: int = Field(
        default=3, ge=1, le=20,
        description="Number of chunks to retrieve",
    )
    USE_HYBRID_SEARCH: bool = Field(
        default=False,
        description="Enable vector + BM25 hybrid search",
    )
    HYBRID_ALPHA: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Vector weight (0=BM25 only, 1=vector only)",
    )

    # =========================================================================
    # MEMORY MANAGEMENT
    # =========================================================================
    MAX_CHAT_HISTORY: int = Field(
        default=10, ge=1, le=100,
        description="Maximum chat messages to display",
    )
    ENABLE_MEMORY_PROFILING: bool = Field(
        default=False,
        description="Enable detailed memory profiling",
    )
    AUTO_CLEANUP_THRESHOLD_MB: int = Field(
        default=5000, ge=1000, le=32000,
        description="Memory threshold for automatic cleanup (MB)",
    )

    # =========================================================================
    # CONVERSATIONAL MEMORY
    # =========================================================================
    ENABLE_CONVERSATION_MEMORY: bool = Field(
        default=True,
        description="Enable conversation context for follow-ups",
    )
    MEMORY_WINDOW: int = Field(
        default=5, ge=1, le=20,
        description="Number of exchange pairs to remember",
    )

    # =========================================================================
    # CITATION
    # =========================================================================
    ENABLE_CITATIONS: bool = Field(
        default=True,
        description="Enable source citations in responses",
    )
    CITATION_HIGHLIGHT_LENGTH: int = Field(
        default=200, ge=50, le=1000,
        description="Characters of context around citations",
    )

    # =========================================================================
    # MONITORING
    # =========================================================================
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable Prometheus metrics server",
    )
    METRICS_PORT: int = Field(
        default=8001, ge=1024, le=65535,
        description="Port for metrics endpoint",
    )
    LOG_FORMAT: Literal["json", "text"] = Field(
        default="json",
        description="Log output format",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )

    # =========================================================================
    # DOCUMENT MANAGEMENT
    # =========================================================================
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=50, ge=1, le=500,
        description="Maximum upload file size in MB",
    )
    ALLOWED_EXTENSIONS: str = Field(
        default="pdf,txt,docx,md",
        description="Comma-separated list of allowed file extensions",
    )

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("DATA_DIR", "MODELS_DIR", "EMBEDDINGS_DIR", "FAISS_INDEX_PATH")
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist, creating them if necessary."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

    @validator("HF_HOME", pre=True, always=True)
    def set_hf_home(cls, v, values) -> str:
        """Set HuggingFace home from embeddings dir if not provided."""
        if v is None:
            return str(values.get("EMBEDDINGS_DIR", "embeddings"))
        return str(v)

    @validator("LLM_MODEL_PATH")
    def validate_model_path(cls, v: Path) -> str:
        """Validate that the model file exists."""
        path = Path(v)
        if not path.exists():
            logging.warning(f"Model file not found: {v}")
            # Don't raise during import; let runtime handle missing model
        return str(path)

    def get_allowed_extensions(self) -> List[str]:
        """Parse allowed extensions into a list."""
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",")]


# Global settings instance (singleton)
settings = Settings()
