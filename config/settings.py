"""
Configuration management with validation.
Centralized settings loaded from environment variables.
"""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import validator, Field


class Settings(BaseSettings):
    """Application settings with validation and type hints."""

    # ========================================================================
    # PATHS
    # ========================================================================
    DATA_DIR: Path = Field(default=Path("data"), description="Documents directory")
    MODELS_DIR: Path = Field(default=Path("models"), description="LLM models directory")
    EMBEDDINGS_DIR: Path = Field(default=Path("embeddings"), description="Embeddings cache")
    STORAGE_DIR: Path = Field(default=Path("storage"), description="Local storage")

    # ========================================================================
    # VECTOR STORE
    # ========================================================================
    VECTOR_STORE_TYPE: Literal["qdrant", "faiss"] = Field(
        default="qdrant",
        description="Vector store backend"
    )
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Qdrant connection URL"
    )
    QDRANT_COLLECTION: str = Field(
        default="edge_rag",
        description="Qdrant collection name"
    )
    FAISS_INDEX_PATH: Path = Field(
        default=Path("storage/faiss_index"),
        description="FAISS index storage path"
    )

    # ========================================================================
    # MODELS
    # ========================================================================
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model"
    )
    EMBEDDING_DIM: int = Field(
        default=384,
        description="Embedding dimension"
    )
    LLM_MODEL_PATH: str = Field(
        default="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        description="Path to GGUF model file"
    )
    HF_HOME: Optional[str] = Field(
        default=None,
        description="Hugging Face cache directory"
    )

    # ========================================================================
    # LLM PARAMETERS
    # ========================================================================
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2048, ge=128, le=8192)
    LLM_CONTEXT_WINDOW: int = Field(default=4096, ge=512, le=16384)
    LLM_THREADS: int = Field(default=4, ge=1, le=32)
    LLM_BATCH_SIZE: int = Field(default=256, ge=32, le=2048)

    # ========================================================================
    # CHUNKING STRATEGY
    # ========================================================================
    CHUNK_SIZE: int = Field(default=400, ge=100, le=2000)
    CHUNK_OVERLAP: int = Field(default=40, ge=0, le=500)
    USE_HIERARCHICAL_CHUNKING: bool = Field(default=True)
    PARENT_CHUNK_SIZE: int = Field(default=1200, ge=300, le=5000)

    # ========================================================================
    # RETRIEVAL
    # ========================================================================
    SIMILARITY_TOP_K: int = Field(default=3, ge=1, le=20)
    USE_HYBRID_SEARCH: bool = Field(default=True)
    HYBRID_ALPHA: float = Field(default=0.7, ge=0.0, le=1.0)
    ENABLE_RERANKING: bool = Field(default=False)
    MIN_CONFIDENCE_SCORE: float = Field(default=0.0, ge=0.0, le=1.0)

    # ========================================================================
    # MEMORY MANAGEMENT
    # ========================================================================
    MAX_CHAT_HISTORY: int = Field(default=10, ge=1, le=100)
    ENABLE_MEMORY_PROFILING: bool = Field(default=False)
    AUTO_CLEANUP_THRESHOLD_MB: int = Field(default=6000, ge=1000, le=32000)

    # ========================================================================
    # CONVERSATIONAL MEMORY
    # ========================================================================
    ENABLE_CONVERSATION_MEMORY: bool = Field(default=True)
    MEMORY_WINDOW: int = Field(default=5, ge=1, le=20)

    # ========================================================================
    # CITATION
    # ========================================================================
    ENABLE_CITATIONS: bool = Field(default=True)
    CITATION_HIGHLIGHT_LENGTH: int = Field(default=200, ge=50, le=1000)

    # ========================================================================
    # MONITORING
    # ========================================================================
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=8001, ge=1024, le=65535)
    LOG_FORMAT: Literal["json", "text"] = Field(default="json")
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # ========================================================================
    # DOCUMENT MANAGEMENT
    # ========================================================================
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, ge=1, le=500)
    ALLOWED_EXTENSIONS: str = Field(default="pdf,txt,docx,md")
    AUTO_REFRESH_DOCUMENTS: bool = Field(default=False)

    # ========================================================================
    # ADVANCED
    # ========================================================================
    ENABLE_QUERY_CACHE: bool = Field(default=True)
    QUERY_CACHE_SIZE: int = Field(default=100, ge=10, le=1000)
    ENABLE_QUERY_REWRITING: bool = Field(default=False)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("DATA_DIR", "MODELS_DIR", "EMBEDDINGS_DIR", "STORAGE_DIR", "FAISS_INDEX_PATH")
    def create_directories(cls, v):
        """Ensure directories exist."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

    @validator("HF_HOME", pre=True, always=True)
    def set_hf_home(cls, v, values):
        """Set HuggingFace home if not provided."""
        if v is None:
            return str(values.get("EMBEDDINGS_DIR", "embeddings"))
        return v

    @validator("LLM_MODEL_PATH")
    def validate_model_path(cls, v):
        """Check if model file exists."""
        path = Path(v)
        if not path.exists():
            # Don't fail validation, just warn
            import warnings
            warnings.warn(f"Model file not found: {v}")
        return str(path)

    def get_allowed_extensions(self) -> list[str]:
        """Parse allowed extensions list."""
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",")]


# Global settings instance
settings = Settings()

# Set environment variables for HuggingFace
if settings.HF_HOME:
    os.environ["HF_HOME"] = settings.HF_HOME