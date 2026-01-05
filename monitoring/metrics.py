"""Prometheus metrics collection."""

import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps
import time
from config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# Query-Level Metrics (Overall RAG Pipeline)
# ============================================================================

query_counter = Counter(
    'rag_queries_total',
    'Total number of queries processed'
)

query_latency = Histogram(
    'rag_query_latency_seconds',
    'Total query processing latency (RAG pipeline)',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
)

# ============================================================================
# Retrieval Metrics
# ============================================================================

retrieval_counter = Counter(
    'rag_retrieval_total',
    'Total number of retrieval operations'
)

retrieval_latency = Histogram(
    'rag_retrieval_latency_seconds',
    'Time taken to retrieve documents (vector search + BM25)',
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
)

retrieval_docs_returned = Histogram(
    'rag_retrieval_docs_returned',
    'Number of documents retrieved per query',
    buckets=(1, 5, 10, 20, 50, 100)
)

# ============================================================================
# LLM Inference Metrics
# ============================================================================

inference_counter = Counter(
    'rag_inference_total',
    'Total number of inference calls'
)

time_to_first_token = Histogram(
    'rag_time_to_first_token_seconds',
    'Time to first token from LLM (inference latency start)',
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
)

inference_latency = Histogram(
    'rag_inference_latency_seconds',
    'Total inference latency (full response generation)',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)
)

tokens_generated = Histogram(
    'rag_tokens_generated',
    'Number of tokens generated per response',
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2000)
)

tokens_per_second = Histogram(
    'rag_tokens_per_second',
    'Token generation speed (tokens/second)',
    buckets=(1, 5, 10, 20, 50, 100)
)

# ============================================================================
# System-Level Metrics
# ============================================================================

error_counter = Counter(
    'rag_errors_total',
    'Total number of errors',
    ['type']
)

active_sessions = Gauge(
    'rag_active_sessions',
    'Number of active user sessions'
)

memory_usage_mb = Gauge(
    'rag_memory_usage_mb',
    'Memory usage in MB'
)

document_count = Gauge(
    'rag_document_count',
    'Number of indexed documents'
)


def track_query_metrics(func):
    """Decorator to track query metrics."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        query_counter.inc()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            query_latency.observe(time.time() - start_time)
            return result
        except Exception as e:
            error_counter.labels(type=type(e).__name__).inc()
            raise

    return wrapper


def track_retrieval_metrics(func):
    """Decorator to track retrieval metrics."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        retrieval_counter.inc()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            retrieval_latency.observe(elapsed)

            # Track number of documents returned if available
            if isinstance(result, list):
                retrieval_docs_returned.observe(len(result))

            return result
        except Exception as e:
            error_counter.labels(type=type(e).__name__).inc()
            raise

    return wrapper


def track_inference_metrics(func):
    """Decorator to track inference metrics."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        inference_counter.inc()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            inference_latency.observe(elapsed)
            return result
        except Exception as e:
            error_counter.labels(type=type(e).__name__).inc()
            raise

    return wrapper


class PerformanceTracker:
    """Context manager for detailed performance tracking."""

    def __init__(self, operation_name: str = "operation"):
        """Initialize tracker.

        Args:
            operation_name: Name of operation being tracked
        """
        self.operation_name = operation_name
        self.start_time = None
        self.first_token_time = None
        self.end_time = None
        self.token_count = 0

    def __enter__(self):
        """Start tracking."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End tracking and record metrics."""
        self.end_time = time.time()

        if exc_type is not None:
            error_counter.labels(type=exc_type.__name__).inc()
            return False

        return True

    def record_first_token(self):
        """Record the time when first token was generated."""
        if self.start_time and self.first_token_time is None:
            self.first_token_time = time.time()
            ttft = self.first_token_time - self.start_time
            time_to_first_token.observe(ttft)

    def record_completion(self, token_count: int = 0):
        """Record inference completion with token count.

        Args:
            token_count: Number of tokens generated
        """
        if self.start_time and self.end_time:
            elapsed = self.end_time - self.start_time
            inference_latency.observe(elapsed)

            if token_count > 0:
                tokens_generated.observe(token_count)
                tps = token_count / elapsed if elapsed > 0 else 0
                tokens_per_second.observe(tps)





def start_metrics_server(port: int = None):
    """Start Prometheus metrics server."""
    port = port or settings.METRICS_PORT

    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.warning(f"Failed to start metrics server: {e}")


def update_memory_usage():
    """Update memory usage metric."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        memory_usage_mb.set(mem_info.rss / 1024 / 1024)
    except ImportError:
        pass  # psutil not installed
    except Exception as e:
        logger.debug(f"Failed to update memory metric: {e}")