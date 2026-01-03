# Multi-stage build for optimized image size
FROM python:3.12-bookworm as builder

# Build arguments
ARG PYTHON_VERSION=3.12

# Prevent Python from writing .pyc files and ensure immediate log output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install llama-cpp-python with CPU optimizations
# This is the most time-consuming step, so we do it separately for better caching
RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir llama-cpp-python --force-reinstall --upgrade

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Final stage - smaller image
# ============================================================================
FROM python:3.12-slim-bookworm

# Metadata
LABEL description="On-Device RAG - Production-ready CPU-optimized RAG system"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Streamlit configuration
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash rag && \
    mkdir -p /app && \
    chown -R rag:rag /app

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=rag:rag . .

# Create necessary directories with correct permissions
RUN mkdir -p data models embeddings storage logs && \
    chown -R rag:rag data models embeddings storage logs

# Switch to non-root user
USER rag

# Expose ports
EXPOSE 8501
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]