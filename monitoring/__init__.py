"""
Monitoring package.

Provides observability tools:
- Prometheus metrics for RAG pipeline performance
- Structured JSON logging
- Performance dashboard for Streamlit
"""

from monitoring.dashboard import (
    PerformanceDashboard,
    display_prometheus_link,
    display_system_info,
)
from monitoring.logger import setup_logging
from monitoring.metrics import (
    PerformanceTracker,
    document_count,
    error_counter,
    inference_latency,
    query_counter,
    query_latency,
    retrieval_latency,
    start_metrics_server,
    time_to_first_token,
    track_inference_metrics,
    track_query_metrics,
    track_retrieval_metrics,
    update_memory_usage,
)

__all__ = [
    # Metrics
    "track_query_metrics",
    "track_retrieval_metrics",
    "track_inference_metrics",
    "PerformanceTracker",
    "start_metrics_server",
    "update_memory_usage",
    "query_counter",
    "query_latency",
    "retrieval_latency",
    "time_to_first_token",
    "inference_latency",
    "error_counter",
    "document_count",
    # Logging
    "setup_logging",
    # Dashboard
    "PerformanceDashboard",
    "display_system_info",
    "display_prometheus_link",
]