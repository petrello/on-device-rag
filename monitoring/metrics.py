"""Prometheus metrics collection."""

import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps
import time
from config import settings

logger = logging.getLogger(__name__)

# Define metrics
query_counter = Counter(
    'rag_queries_total',
    'Total number of queries processed'
)

query_latency = Histogram(
    'rag_query_latency_seconds',
    'Query processing latency',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)

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