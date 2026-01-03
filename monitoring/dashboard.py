"""
Performance tracking dashboard.
Provides metrics visualization and monitoring.
"""

import logging
from typing import Dict
from collections import deque
from datetime import datetime

import streamlit as st

from monitoring.metrics import (
    query_counter,
    query_latency,
    error_counter,
    active_sessions,
    memory_usage_mb,
    document_count,
    update_memory_usage,
)
from utils import get_memory_usage

logger = logging.getLogger(__name__)


def _safe_get_counter_value(counter) -> float:
    """Safely extract a numeric total from a Counter or labeled Counter.

    Works with simple Counters/Gauges (._value.get()) and with labeled counters
    (._value is a dict of values).
    """
    try:
        # simple counter/gauge
        return float(counter._value.get())
    except Exception:
        pass

    try:
        vals = getattr(counter, '_value')
        if hasattr(vals, 'values'):
            total = 0.0
            for v in vals.values():
                try:
                    # v may be a ValueClass with .get()
                    total += float(v.get())
                except Exception:
                    try:
                        total += float(v)
                    except Exception:
                        continue
            return total
    except Exception:
        pass

    # As a last resort, try collect() and sum sample values
    try:
        total = 0.0
        for metric in counter.collect():
            for sample in metric.samples:
                # sample is a tuple-like object with .value attribute in some versions
                val = getattr(sample, 'value', None)
                if val is None:
                    try:
                        # older interface: sample[2]
                        val = float(sample[2])
                    except Exception:
                        val = 0.0
                total += float(val or 0.0)
        return total
    except Exception:
        return 0.0


def _safe_get_histogram_avg(hist) -> float:
    """Compute average from a prometheus Histogram (sum / count) if available."""
    try:
        count = float(hist._count.get())
        if count == 0:
            return 0.0
        total = float(hist._sum.get())
        return total / count
    except Exception:
        # fallback: return 0
        try:
            # try collect
            for metric in hist.collect():
                sum_v = 0.0
                count_v = 0.0
                for sample in metric.samples:
                    name = sample.name if hasattr(sample, 'name') else (sample[0] if len(sample) > 0 else '')
                    val = getattr(sample, 'value', None) or (float(sample[2]) if len(sample) > 2 else 0.0)
                    if 'sum' in name:
                        sum_v = float(val)
                    if 'count' in name:
                        count_v = float(val)
                if count_v > 0:
                    return sum_v / count_v
        except Exception:
            pass
    return 0.0


def _ensure_streamlit() -> bool:
    """Return True if streamlit is available, otherwise log and return False."""
    if st is None:
        logger.debug("Streamlit not available; dashboard display functions are disabled.")
        return False
    return True


class PerformanceDashboard:
    """Tracks and displays performance metrics.

    This dashboard uses the Prometheus metrics objects defined in
    `monitoring.metrics` as the primary source of truth, while keeping a small
    in-memory history for plotting min/max and recent trends.
    """

    def __init__(self, history_size: int = 100):
        """
        Initialize dashboard.

        Args:
            history_size: Number of historical data points to keep
        """
        self.history_size = history_size
        self.query_history = deque(maxlen=history_size)  # stores timestamps of recorded samples
        self.latency_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)

    def record_query(self, latency: float):
        """
        Record a query execution both in local history and in Prometheus metrics.

        Args:
            latency: Query latency in seconds
        """
        timestamp = datetime.now()
        # Update in-memory history
        self.query_history.append(timestamp)
        self.latency_history.append(latency)
        self.timestamp_history.append(timestamp)

        # Update prometheus metrics
        try:
            query_counter.inc()
        except Exception as e:
            logger.debug(f"Failed to increment query_counter: {e}")

        try:
            query_latency.observe(latency)
        except Exception as e:
            logger.debug(f"Failed to observe latency histogram: {e}")

        # Update memory metric using the helper in metrics.py (if available)
        try:
            update_memory_usage()
            # try to read psutil-based value via memory_usage_mb
            mem_val = _safe_get_counter_value(memory_usage_mb)
            if mem_val:
                self.memory_history.append(mem_val)
            else:
                # fallback to utils.get_memory_usage
                mem_info = get_memory_usage()
                self.memory_history.append(mem_info["rss_mb"] if mem_info else None)
        except Exception:
            mem_info = get_memory_usage()
            self.memory_history.append(mem_info["rss_mb"] if mem_info else None)

    def sync_from_metrics(self):
        """Pull current values from Prometheus metrics and append to history.

        This is useful if the application increments metrics elsewhere and we want
        the dashboard to reflect those external updates.
        """
        ts = datetime.now()
        avg_latency = _safe_get_histogram_avg(query_latency)
        # append a sample point
        self.timestamp_history.append(ts)
        self.latency_history.append(avg_latency)

        # memory
        try:
            update_memory_usage()
            mem = _safe_get_counter_value(memory_usage_mb)
            self.memory_history.append(mem if mem else None)
        except Exception:
            mem_info = get_memory_usage()
            self.memory_history.append(mem_info["rss_mb"] if mem_info else None)

        # record a logical query sample if the counter has increased since last stored
        try:
            total_queries = int(_safe_get_counter_value(query_counter))
            # store as one entry per sample, but to avoid huge jumps we only store a 1 when total_queries increased
            prev_total = getattr(self, '_last_total_queries', 0)
            if total_queries > prev_total:
                self.query_history.append(ts)
            self._last_total_queries = total_queries
        except Exception:
            pass

    def get_stats(self) -> Dict:
        """
        Get current statistics using Prometheus metrics where possible.

        Returns:
            Statistics dictionary
        """
        total_queries = int(_safe_get_counter_value(query_counter))
        avg_latency = _safe_get_histogram_avg(query_latency)
        min_latency = min(self.latency_history) if self.latency_history else 0.0
        max_latency = max(self.latency_history) if self.latency_history else 0.0

        # memory: prefer prometheus gauge
        try:
            update_memory_usage()
            current_memory = _safe_get_counter_value(memory_usage_mb)
        except Exception:
            mem_info = get_memory_usage()
            current_memory = mem_info["rss_mb"] if mem_info else 0.0

        # active sessions & documents
        active = int(_safe_get_counter_value(active_sessions))
        docs = int(_safe_get_counter_value(document_count))

        # errors: sum over labeled error_counter
        errors = int(_safe_get_counter_value(error_counter))

        return {
            "total_queries": total_queries,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "current_memory_mb": current_memory,
            "active_sessions": active,
            "document_count": docs,
            "total_errors": errors,
        }

    def display_metrics(self):
        """Display metrics in Streamlit."""
        if not _ensure_streamlit():
            return

        st.subheader("📊 Performance Metrics")

        # ensure we pull latest metric values
        try:
            self.sync_from_metrics()
        except Exception as e:
            logger.debug(f"Failed to sync metrics: {e}")

        stats = self.get_stats()

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Queries", f"{stats['total_queries']}")
            st.text(f"Errors: {stats['total_errors']}")

        with col2:
            st.metric("Avg Latency", f"{stats['avg_latency']:.2f}s")
            st.text(f"Active Sessions: {stats['active_sessions']}")

        with col3:
            st.metric("Min / Max", f"{stats['min_latency']:.2f}s", delta=f"{stats['max_latency']:.2f}s")
            st.text(f"Indexed Docs: {stats['document_count']}")

        with col4:
            if stats['current_memory_mb'] > 0:
                st.metric("Memory", f"{stats['current_memory_mb']:.0f}MB")

        # Display charts if there's data
        if len(self.latency_history) > 0:
            self._display_latency_chart()

        if len(self.memory_history) > 0 and any(m is not None for m in self.memory_history):
            self._display_memory_chart()

    def _display_latency_chart(self):
        """Display latency over time chart."""
        if not _ensure_streamlit():
            return
        try:
            import pandas as pd

            if not self.timestamp_history:
                return

            df = pd.DataFrame({
                'Time': list(self.timestamp_history),
                'Latency (s)': list(self.latency_history)
            })

            st.line_chart(df.set_index('Time'))

        except ImportError:
            st.info("Install pandas for charts: pip install pandas")

    def _display_memory_chart(self):
        """Display memory usage over time chart."""
        if not _ensure_streamlit():
            return
        try:
            import pandas as pd

            # Filter out None values
            valid_data = [
                (t, m) for t, m in zip(self.timestamp_history, self.memory_history)
                if m is not None
            ]

            if not valid_data:
                return

            times, memory = zip(*valid_data)

            df = pd.DataFrame({
                'Time': times,
                'Memory (MB)': memory
            })

            st.line_chart(df.set_index('Time'))

        except ImportError:
            pass


def display_system_info():
    """Display system information."""
    if not _ensure_streamlit():
        return
    st.subheader("🖥️ System Information")

    from config import settings

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Configuration:**")
        st.text(f"Vector Store: {settings.VECTOR_STORE_TYPE.upper()}")
        st.text(f"Embedding Model: {settings.EMBEDDING_MODEL.split('/')[-1]}")
        st.text(f"Chunk Size: {settings.CHUNK_SIZE}")
        st.text(f"Top-K: {settings.SIMILARITY_TOP_K}")

    with col2:
        st.markdown("**Features:**")
        st.text(f"Hierarchical: {'✓' if settings.USE_HIERARCHICAL_CHUNKING else '✗'}")
        st.text(f"Hybrid Search: {'✓' if settings.USE_HYBRID_SEARCH else '✗'}")
        st.text(f"Conv. Memory: {'✓' if settings.ENABLE_CONVERSATION_MEMORY else '✗'}")
        st.text(f"Citations: {'✓' if settings.ENABLE_CITATIONS else '✗'}")

    # Memory usage (try Prometheus gauge first)
    try:
        update_memory_usage()
        mem = _safe_get_counter_value(memory_usage_mb)
        if mem:
            st.markdown("**Memory Usage:**")
            st.progress(min(mem / 100.0, 1.0))
            st.text(f"{mem:.0f}MB")
        else:
            mem_info = get_memory_usage()
            if mem_info:
                st.markdown("**Memory Usage:**")
                st.progress(min(mem_info["percent"] / 100, 1.0))
                st.text(f"{mem_info['rss_mb']:.0f}MB / {mem_info['percent']:.1f}%")
    except Exception:
        mem_info = get_memory_usage()
        if mem_info:
            st.markdown("**Memory Usage:**")
            st.progress(min(mem_info["percent"] / 100, 1.0))
            st.text(f"{mem_info['rss_mb']:.0f}MB / {mem_info['percent']:.1f}%")


def display_prometheus_link():
    """Display link to Prometheus metrics."""
    if not _ensure_streamlit():
        return
    from config import settings

    if settings.ENABLE_METRICS:
        st.markdown("---")
        st.markdown(
            f"**Prometheus Metrics:** "
            f"[View Raw Metrics](http://localhost:{settings.METRICS_PORT}/metrics)"
        )