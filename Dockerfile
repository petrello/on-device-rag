# Base image
FROM python:3.12-bookworm

# Metadata
LABEL description="On-Device RAG - CPU-optimized RAG system"
LABEL version="1.0.0"

# Environment variables
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Prevent Python from writing .pyc files and ensure immediate log output
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Streamlit configuration
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 1. System dependencies (Cached)
# Added libgomp1 for OpenMP support and libopenblas-dev for math acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install llama-cpp-python FIRST (The "Heavy" part)
# This layer is now locked. Docker skips this unless you change the CMAKE_ARGS.
RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" FORCE_CMAKE=1 \
    pip install llama-cpp-python --no-cache-dir

# 3. Install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the code
COPY . .

# Create directories
RUN mkdir -p data models embeddings storage

# Expose Streamlit (8501) and Prometheus Metrics (8001)
EXPOSE 8501
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]