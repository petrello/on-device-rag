"""Simple dashboard helpers for Streamlit UI to show metrics."""

import logging
from monitoring.metrics import (
    query_counter,
    query_latency,
    error_counter,
    memory_usage_mb,
    document_count,
)

logger = logging.getLogger(__name__)


def render_metrics():
    """Return a dict of metric values suitable for display in a UI."""
    try:
        metrics = {
            "queries_total": query_counter._value.get(),
            "document_count": document_count._value.get(),
            # histogram provides sum/count attributes
            "query_latency_count": getattr(query_latency, 'count', 0),
        }
        return metrics
    except Exception as e:
        logger.debug(f"Failed to read metrics: {e}")
        return {}
