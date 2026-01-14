# On-Device RAG

> **Experimental Retrieval-Augmented Generation for Resource-Constrained Devices**

A lightweight, CPU-only RAG system designed for local deployment on edge devices, laptops, and air-gapped environments. Built with LlamaIndex, it runs entirely offline without requiring GPUs or cloud APIs.

---

## ⚠️ Project Status

**This is an experimental, research-oriented project.** It is not production-ready software.

- Created for learning, experimentation, and academic use
- Optimized for constrained hardware (≤4 CPU cores, ≤6GB RAM)
- Actively developed; APIs may change

---

## 🎯 What This Project Is

On-Device RAG is a personal project that demonstrates how to build a complete RAG pipeline that runs locally on modest hardware. It's designed for:

- **Students & Researchers** exploring RAG architectures
- **ML Engineers** prototyping edge deployment scenarios
- **Privacy-conscious users** who need fully offline document Q&A

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Offline Operation** | No internet required after initial model download |
| **CPU-Only Inference** | Runs on laptops without dedicated GPUs |
| **Hierarchical Chunking** | Small chunks for retrieval, large chunks for context |
| **Hybrid Search** | Combines vector similarity with BM25 keyword matching |
| **Conversational Memory** | Maintains context across multi-turn conversations |
| **Docker Deployment** | Consistent environment across platforms |

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph UI["🖥️ User Interface"]
        ST[Streamlit App]
    end

    subgraph RAG["🔄 RAG Pipeline"]
        direction TB
        QE[Query Enhancement]
        RET[Retriever]
        QEN[Query Engine]
        
        subgraph Retrieval["Retrieval Layer"]
            VEC[Vector Search]
            BM25[BM25 Search]
            HYB[Hybrid Scorer]
        end
    end

    subgraph Storage["💾 Storage Layer"]
        direction LR
        QD[(Qdrant)]
        FS[(FAISS)]
    end

    subgraph Models["🧠 Models"]
        EMB[Embedding Model<br/>granite-278m]
        LLM[LLM<br/>Llama 3.2 1B Q8]
    end

    subgraph Documents["📁 Documents"]
        PDF[PDF]
        TXT[TXT]
        DOCX[DOCX]
    end

    Documents --> |Process & Chunk| RAG
    ST --> |Query| QE
    QE --> RET
    RET --> VEC & BM25
    VEC --> HYB
    BM25 --> HYB
    HYB --> QEN
    VEC <--> Storage
    EMB --> VEC
    QEN --> LLM
    LLM --> ST
```

### Component Overview

| Component | Purpose | Location |
|-----------|---------|----------|
| **Core** | Chunking, embedding, LLM, retrieval | `core/` |
| **Storage** | Vector store abstraction (Qdrant/FAISS) | `storage/` |
| **Document Management** | Upload, process, index documents | `document_management/` |
| **Memory** | Conversational context management | `memory/` |
| **Monitoring** | Prometheus metrics, logging | `monitoring/` |
| **UI** | Streamlit chat interface | `ui/` |

---

## 📁 Project Structure

```
on-device-rag/
├── app.py                      # Main Streamlit application
├── config/
│   └── settings.py             # Pydantic-based configuration
├── core/
│   ├── chunking.py             # Hierarchical/simple chunking
│   ├── embeddings.py           # Embedding model management
│   ├── llm.py                  # LLM initialization (LlamaCPP)
│   └── retrieval.py            # Hybrid retrieval (vector + BM25)
├── storage/
│   ├── vector_store.py         # Abstract interface
│   ├── qdrant_store.py         # Qdrant implementation
│   └── local_store.py          # FAISS implementation
├── document_management/
│   ├── processor.py            # Document loading & preprocessing
│   ├── indexer.py              # Document indexing logic
│   └── uploader.py             # File upload handling
├── memory/
│   └── conversation.py         # Sliding-window conversation memory
├── citation/
│   └── extractor.py            # Citation extraction & formatting
├── monitoring/
│   ├── metrics.py              # Prometheus metrics
│   ├── logger.py               # Structured JSON logging
│   └── dashboard.py            # Performance dashboard
├── ui/
│   ├── chat.py                 # Chat interface components
│   ├── sidebar.py              # Sidebar with stats
│   └── document_viewer.py      # Document management UI
├── tests/
│   ├── test_chunking.py        # Chunking tests
│   ├── test_retrieval.py       # Retrieval tests
│   └── test_vector_store.py    # Vector store tests
├── data/                       # Your documents go here
├── models/                     # GGUF model files
├── embeddings/                 # Cached embedding models
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # Application container
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- **Docker Desktop** (v20.10+)
- **8GB+ RAM** (6GB for containers, 2GB for system)
- **10GB disk space** (models + indices)

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/on-device-rag.git
cd on-device-rag
```

### 2. Download Models

Download a GGUF model and place it in `models/`:

```bash
# Example: Llama 3.2 1B Instruct (Q8 quantization)
# Download from HuggingFace and save as:
# models/llama-3.2-1b-instruct-q8_0.gguf
```

The embedding model (`ibm-granite/granite-embedding-278m-multilingual`) downloads automatically on first run.

### 3. Add Documents

Place your documents in the `data/` directory:

```bash
cp /path/to/your/documents/*.pdf data/
```

Supported formats: PDF, TXT, DOCX, MD

### 4. Start the Application

```bash
docker compose up --build
```

Wait for initialization (~2-5 minutes on first run), then open:

- **Chat UI**: http://localhost:8501
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Metrics**: http://localhost:8001/metrics (if enabled)

### 5. Stop the Application

```bash
docker compose down
```

---

## 🐳 Docker Usage

This section provides comprehensive Docker commands for managing the application.

### Building & Starting

```bash
# Build and start all containers (first time or after code changes)
docker compose up --build

# Start without rebuilding (faster, uses cached images)
docker compose up

# Start in detached mode (run in background)
docker compose up -d

# Start only the RAG application (without Qdrant)
docker compose up rag-app

# Force rebuild without cache
docker compose build --no-cache
docker compose up
```

### Stopping & Cleanup

```bash
# Stop all containers (preserves data volumes)
docker compose down

# Stop and remove volumes (WARNING: deletes indexed data)
docker compose down -v

# Stop a specific service
docker compose stop rag-app

# Remove stopped containers
docker compose rm
```

### Logs & Debugging

```bash
# View logs from all services
docker compose logs

# Follow logs in real-time
docker compose logs -f

# View logs for a specific service
docker compose logs rag-app
docker compose logs qdrant

# View last 100 lines of logs
docker compose logs --tail=100
```

### Running Commands Inside Containers

```bash
# Open a shell inside the running RAG container
docker compose exec rag-app bash

# Run tests inside the container
docker compose exec rag-app pytest tests/ -v

# Check Python environment
docker compose exec rag-app pip list

# Run a one-off command
docker compose run --rm rag-app python -c "from config import settings; print(settings)"
```

### Container Management

```bash
# List running containers
docker compose ps

# Restart a specific service
docker compose restart rag-app

# View resource usage
docker stats

# Inspect container details
docker compose inspect rag-app
```

---

## 📊 Memory Profiling

The application includes built-in memory profiling to help optimize performance on resource-constrained devices.

### What Profiling Provides

Memory profiling tracks:

- **RSS (Resident Set Size)**: Physical memory currently used by the process
- **VMS (Virtual Memory Size)**: Total virtual memory allocated
- **Memory percentage**: Proportion of system memory used
- **Automatic cleanup triggers**: When memory exceeds configured thresholds

This is useful for:

- Identifying memory leaks during development
- Optimizing chunk sizes and batch parameters
- Monitoring production deployments
- Understanding memory patterns during inference

### Enabling Memory Profiling

Set the environment variable in your `.env` file or docker-compose:

```bash
# In .env file
ENABLE_MEMORY_PROFILING=true

# Or via command line
docker compose run -e ENABLE_MEMORY_PROFILING=true rag-app
```

### Running with Profiling via Docker

```bash
# Start with memory profiling enabled
docker compose run -e ENABLE_MEMORY_PROFILING=true -p 8501:8501 rag-app

# Or modify docker-compose.yml to include:
# environment:
#   - ENABLE_MEMORY_PROFILING=true
```

When profiling is enabled, you'll see:

1. **Memory usage in the sidebar**: Real-time RSS and percentage display
2. **Memory warnings**: Alerts when usage exceeds `AUTO_CLEANUP_THRESHOLD_MB`
3. **Automatic garbage collection**: Triggered when thresholds are exceeded

### Profiling Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MEMORY_PROFILING` | `false` | Show memory stats in UI |
| `AUTO_CLEANUP_THRESHOLD_MB` | `5000` | Trigger GC when RSS exceeds this |

### Advanced Profiling

For detailed memory analysis, you can use Python's built-in profilers:

```bash
# Run with memory profiler (requires installing memory-profiler)
docker compose exec rag-app python -m memory_profiler app.py

# Profile specific functions by adding @profile decorator
# Then run with:
docker compose exec rag-app python -m memory_profiler -o mem_report.txt app.py
```

### Monitoring with Prometheus

Memory metrics are also exposed via Prometheus:

```bash
# View raw metrics
curl http://localhost:8001/metrics | grep rag_memory

# Example output:
# rag_memory_usage_mb 1234.5
```

---

## ⚙️ Configuration

Configuration is managed via environment variables or a `.env` file.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_TYPE` | `qdrant` | Vector backend: `qdrant` or `faiss` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `LLM_MODEL_PATH` | `models/llama-3.2-1b-instruct-q8_0.gguf` | Path to GGUF model |
| `EMBEDDING_MODEL` | `ibm-granite/granite-embedding-278m-multilingual` | HuggingFace model ID |

### Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `400` | Child chunk size (tokens) |
| `PARENT_CHUNK_SIZE` | `1200` | Parent chunk size (tokens) |
| `CHUNK_OVERLAP` | `40` | Overlap between chunks |
| `USE_HIERARCHICAL_CHUNKING` | `false` | Enable parent-child chunking |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_TOP_K` | `3` | Number of chunks to retrieve |
| `USE_HYBRID_SEARCH` | `false` | Enable vector + BM25 hybrid |
| `HYBRID_ALPHA` | `0.7` | Vector weight (0=BM25, 1=vector) |

### LLM Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `LLM_MAX_TOKENS` | `4096` | Max tokens to generate |
| `LLM_CONTEXT_WINDOW` | `8192` | Context window size |
| `LLM_THREADS` | `4` | CPU threads for inference |

### Example `.env` File

```bash
# Vector Store
VECTOR_STORE_TYPE=qdrant
QDRANT_URL=http://qdrant:6333

# Models
LLM_MODEL_PATH=models/llama-3.2-1b-instruct-q8_0.gguf
EMBEDDING_MODEL=ibm-granite/granite-embedding-278m-multilingual

# Retrieval
USE_HIERARCHICAL_CHUNKING=true
USE_HYBRID_SEARCH=true
HYBRID_ALPHA=0.7
SIMILARITY_TOP_K=5

# Performance
LLM_THREADS=4
LLM_MAX_TOKENS=2048
```

---

## 🔧 RAG Pipeline Details

### Document Processing Flow

```
Document → Load → Preprocess → Chunk → Embed → Store
```

1. **Load**: SimpleDirectoryReader handles PDF, TXT, DOCX, MD
2. **Preprocess**: Text cleaning, whitespace normalization
3. **Chunk**: Split into retrievable units (hierarchical or flat)
4. **Embed**: Convert to vectors using `granite-278m`
5. **Store**: Persist in Qdrant or FAISS

### Hierarchical Chunking

When enabled, documents are split into two levels:

- **Parent chunks** (~1200 tokens): Provide context to the LLM
- **Child chunks** (~400 tokens): Indexed for precise retrieval

On retrieval, child chunks are matched, but the corresponding parent chunk is passed to the LLM for richer context.

### Hybrid Retrieval

Combines two retrieval strategies:

1. **Vector Search**: Semantic similarity using embeddings
2. **BM25**: Keyword-based matching

Final score: `α × vector_score + (1-α) × bm25_score`

Use `HYBRID_ALPHA` to balance semantic vs. keyword matching.

---

## 📊 Performance Considerations

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU Cores | 2 | 4 |
| RAM | 4GB | 8GB |
| Disk | 5GB | 20GB |

### Memory Optimization

The system includes automatic memory management:

- Singleton pattern for models (loaded once)
- Automatic garbage collection when threshold exceeded
- Memory-mapped model loading (configurable)

### Typical Latencies

On a 4-core CPU with 8GB RAM:

| Operation | Typical Time |
|-----------|--------------|
| Model loading | 10-30s |
| Document indexing | 5-30s (varies by size) |
| Query retrieval | 0.1-0.5s |
| LLM inference | 5-30s (varies by response length) |
| Time to first token | 0.5-2s |

---

## 🧪 Testing

Run the test suite:

```bash
# Inside the container
docker compose exec rag-app pytest tests/ -v

# Or locally (requires dependencies installed)
pytest tests/ -v
```

Tests cover:

- Chunking logic (hierarchical and simple)
- Retrieval (hybrid and vector-only)
- Vector store interface

---

## 🛠️ Development

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp-python with CPU optimizations
CMAKE_ARGS="-DGGML_BLAS=ON" pip install llama-cpp-python

# Run the app
streamlit run app.py
```

### Code Style

The project uses:

- Type hints throughout
- Google-style docstrings
- `ruff` for linting (see `requirements.txt`)

---

## 🗺️ Roadmap

> **Note**: This is an experimental project. These items are aspirational.

### Planned

- [ ] Query caching for repeated questions
- [ ] Reranking with cross-encoder
- [ ] Streaming improvements
- [ ] Better citation extraction

### Under Consideration

- [ ] Multi-modal support (images)
- [ ] Knowledge graph integration
- [ ] Fine-tuning support
- [ ] WebUI improvements

---

## 🤝 Contributing

Contributions are welcome! This is a learning project, so:

1. **Issues**: Bug reports, questions, and suggestions are all helpful
2. **Pull Requests**: Keep them focused and well-documented
3. **Discussions**: Share your experiments and use cases

Please note:

- This is a personal project with limited maintenance bandwidth
- Large architectural changes should be discussed first
- Performance improvements are especially welcome

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with:

- [LlamaIndex](https://www.llamaindex.ai/) - RAG orchestration
- [Qdrant](https://qdrant.tech/) - Vector database
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU inference
- [Streamlit](https://streamlit.io/) - UI framework
- [IBM Granite](https://huggingface.co/ibm-granite) - Embedding model

---

## 📧 Contact

This is an experimental project by a student researcher. For questions:

- Open an issue on GitHub
- Check existing discussions

---

*Built with ❤️ for the open-source AI community*

