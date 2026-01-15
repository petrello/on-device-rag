"""
Performance tracking dashboard.

Provides real-time metrics visualization and monitoring for the RAG pipeline.
Uses Prometheus metrics as the source of truth with in-memory history for trends.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Optional

import streamlit as st

from monitoring.metrics import (
    active_sessions,
    document_count,
    error_counter,
    inference_counter,
    inference_latency,
    memory_usage_mb,
    query_counter,
    query_latency,
    retrieval_counter,
    retrieval_docs_returned,
    retrieval_latency,
    time_to_first_token,
    tokens_generated,
    tokens_per_second,
    update_memory_usage,
)
from utils import get_memory_usage

logger = logging.getLogger(__name__)


def _safe_get_counter_value(counter: Any) -> float:
    """
    Safely extract a numeric total from a Prometheus Counter or Gauge.

    Handles both simple counters and labeled counters with multiple values.

    Args:
        counter: Prometheus metric object.

    Returns:
        Numeric total value, or 0.0 on failure.
    """
    try:
        return float(counter._value.get())
    except Exception:
        pass

    try:
        vals = getattr(counter, '_value')
        if hasattr(vals, 'values'):
            total = 0.0
            for v in vals.values():
                try:
                    total += float(v.get())
                except Exception:
                    try:
                        total += float(v)
                    except Exception:
                        continue
            return total
    except Exception:
        pass

    try:
        total = 0.0
        for metric in counter.collect():
            for sample in metric.samples:
                val = getattr(sample, 'value', None)
                if val is None:
                    try:
                        val = float(sample[2])
                    except Exception:
                        val = 0.0
                total += float(val or 0.0)
        return total
    except Exception:
        return 0.0


def _safe_get_histogram_avg(hist: Any) -> float:
    """
    Compute average from a Prometheus Histogram (sum / count).

    Args:
        hist: Prometheus Histogram object.

    Returns:
        Average value, or 0.0 on failure.
    """
    try:
        count = float(hist._count.get())
        if count == 0:
            return 0.0
        total = float(hist._sum.get())
        return total / count
    except Exception:
        try:
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
    """Check if Streamlit is available for rendering."""
    if st is None:
        logger.debug("Streamlit not available; dashboard disabled.")
        return False
    return True


class PerformanceDashboard:
    """
    Tracks and displays performance metrics.

    Uses Prometheus metrics as the primary source of truth while maintaining
    in-memory history for min/max calculations and trend visualization.

    Attributes:
        history_size: Maximum number of data points to retain.
    """

    def __init__(self, history_size: int = 100) -> None:
        """
        Initialize the dashboard.

        Args:
            history_size: Number of historical data points to keep.
        """
        self.history_size = history_size
        self.query_history: Deque[datetime] = deque(maxlen=history_size)
        self.latency_history: Deque[float] = deque(maxlen=history_size)
        self.retrieval_latency_history: Deque[float] = deque(maxlen=history_size)
        self.inference_latency_history: Deque[float] = deque(maxlen=history_size)
        self.ttft_history: Deque[float] = deque(maxlen=history_size)
        self.memory_history: Deque[Optional[float]] = deque(maxlen=history_size)
        self.timestamp_history: Deque[datetime] = deque(maxlen=history_size)

    def record_query(self, latency: float) -> None:
        """
        Record a query execution.

        Args:
            latency: Query latency in seconds.
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
            mem_val = _safe_get_counter_value(memory_usage_mb)
            if mem_val:
                self.memory_history.append(mem_val)
            else:
                mem_info = get_memory_usage()
                self.memory_history.append(mem_info["rss_mb"] if mem_info else None)
        except Exception:
            mem_info = get_memory_usage()
            self.memory_history.append(mem_info["rss_mb"] if mem_info else None)

    def record_retrieval(self, latency: float, docs_count: int = 0):
        """Record retrieval operation metrics.

        Args:
            latency: Retrieval latency in seconds
            docs_count: Number of documents retrieved
        """
        timestamp = datetime.now()
        self.retrieval_latency_history.append(latency)

        try:
            retrieval_counter.inc()
            retrieval_latency.observe(latency)
            if docs_count > 0:
                retrieval_docs_returned.observe(docs_count)
        except Exception as e:
            logger.debug(f"Failed to record retrieval metrics: {e}")

    def record_inference(self, ttft: float = None, total_latency: float = None,
                        token_count: int = 0):
        """Record inference operation metrics.

        Args:
            ttft: Time to first token in seconds
            total_latency: Total inference latency in seconds
            token_count: Number of tokens generated
        """
        timestamp = datetime.now()

        if ttft is not None:
            self.ttft_history.append(ttft)
            try:
                time_to_first_token.observe(ttft)
            except Exception as e:
                logger.debug(f"Failed to record TTFT: {e}")

        if total_latency is not None:
            self.inference_latency_history.append(total_latency)
            try:
                inference_counter.inc()
                inference_latency.observe(total_latency)

                if token_count > 0:
                    tokens_generated.observe(token_count)
                    tps = token_count / total_latency if total_latency > 0 else 0
                    tokens_per_second.observe(tps)
            except Exception as e:
                logger.debug(f"Failed to record inference metrics: {e}")

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

        # Retrieval metrics
        total_retrievals = int(_safe_get_counter_value(retrieval_counter))
        avg_retrieval_latency = _safe_get_histogram_avg(retrieval_latency)
        min_retrieval_latency = min(self.retrieval_latency_history) if self.retrieval_latency_history else 0.0
        max_retrieval_latency = max(self.retrieval_latency_history) if self.retrieval_latency_history else 0.0

        # Inference metrics
        total_inferences = int(_safe_get_counter_value(inference_counter))
        avg_ttft = _safe_get_histogram_avg(time_to_first_token)
        avg_inference_latency = _safe_get_histogram_avg(inference_latency)
        min_ttft = min(self.ttft_history) if self.ttft_history else 0.0
        max_ttft = max(self.ttft_history) if self.ttft_history else 0.0
        min_inference_latency = min(self.inference_latency_history) if self.inference_latency_history else 0.0
        max_inference_latency = max(self.inference_latency_history) if self.inference_latency_history else 0.0
        avg_tokens_generated = _safe_get_histogram_avg(tokens_generated)
        avg_tps = _safe_get_histogram_avg(tokens_per_second)

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
            # Query-level
            "total_queries": total_queries,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,

            # Retrieval-level
            "total_retrievals": total_retrievals,
            "avg_retrieval_latency": avg_retrieval_latency,
            "min_retrieval_latency": min_retrieval_latency,
            "max_retrieval_latency": max_retrieval_latency,

            # Inference-level
            "total_inferences": total_inferences,
            "avg_ttft": avg_ttft,
            "min_ttft": min_ttft,
            "max_ttft": max_ttft,
            "avg_inference_latency": avg_inference_latency,
            "min_inference_latency": min_inference_latency,
            "max_inference_latency": max_inference_latency,
            "avg_tokens_generated": avg_tokens_generated,
            "avg_tps": avg_tps,

            # System
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

        # Create tabs for different metric categories
        tab1, tab2, tab3, tab4 = st.tabs(["🔄 RAG Pipeline", "🔍 Retrieval", "🤖 Inference", "💾 System"])

        with tab1:
            self._display_query_metrics(stats)

        with tab2:
            self._display_retrieval_metrics(stats)

        with tab3:
            self._display_inference_metrics(stats)

        with tab4:
            self._display_system_metrics(stats)

    def _display_query_metrics(self, stats):
        """Display overall RAG pipeline metrics."""
        st.markdown("### Overall RAG Pipeline")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Queries", f"{stats['total_queries']}")
        with col2:
            st.metric("Avg Total Latency", f"{stats['avg_latency']:.2f}s")
        with col3:
            st.metric("Min / Max", f"{stats['min_latency']:.2f}s", delta=f"{stats['max_latency']:.2f}s")

        if len(self.latency_history) > 0:
            st.markdown("**Query Latency Trend**")
            self._display_latency_chart()

    def _display_retrieval_metrics(self, stats):
        """Display retrieval performance metrics."""
        st.markdown("### Document Retrieval Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Retrievals", f"{stats['total_retrievals']}")
        with col2:
            st.metric("Avg Retrieval Time", f"{stats['avg_retrieval_latency']:.3f}s")
        with col3:
            st.metric("Min / Max", f"{stats['min_retrieval_latency']:.3f}s", delta=f"{stats['max_retrieval_latency']:.3f}s")

        if len(self.retrieval_latency_history) > 0:
            st.markdown("**Retrieval Latency Trend**")
            self._display_retrieval_latency_chart()

    def _display_inference_metrics(self, stats):
        """Display LLM inference performance metrics."""
        st.markdown("### LLM Inference Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Inferences", f"{stats['total_inferences']}")
        with col2:
            st.metric("Avg TTFT", f"{stats['avg_ttft']:.3f}s")
            st.caption(f"Min/Max: {stats['min_ttft']:.3f}s / {stats['max_ttft']:.3f}s")
        with col3:
            st.metric("Avg Inference Time", f"{stats['avg_inference_latency']:.2f}s")
            st.caption(f"Min/Max: {stats['min_inference_latency']:.2f}s / {stats['max_inference_latency']:.2f}s")
        with col4:
            st.metric("Avg Tokens", f"{stats['avg_tokens_generated']:.0f}")
            st.metric("Avg Throughput", f"{stats['avg_tps']:.1f} tok/s")

        col1, col2 = st.columns(2)
        with col1:
            if len(self.ttft_history) > 0:
                st.markdown("**Time-to-First-Token Trend**")
                self._display_ttft_chart()

        with col2:
            if len(self.inference_latency_history) > 0:
                st.markdown("**Inference Latency Trend**")
                self._display_inference_latency_chart()

    def _display_system_metrics(self, stats):
        """Display system-level metrics."""
        st.markdown("### System Resources")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Active Sessions", f"{stats['active_sessions']}")
        with col2:
            st.metric("Indexed Documents", f"{stats['document_count']}")
        with col3:
            st.metric("Total Errors", f"{stats['total_errors']}")

        if stats['current_memory_mb'] > 0:
            st.markdown("**Memory Usage**")
            st.metric("Current Memory", f"{stats['current_memory_mb']:.0f}MB")

        if len(self.memory_history) > 0 and any(m is not None for m in self.memory_history):
            st.markdown("**Memory Trend**")
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

    def _display_retrieval_latency_chart(self):
        """Display retrieval latency over time chart."""
        if not _ensure_streamlit():
            return
        try:
            import pandas as pd

            if not self.retrieval_latency_history:
                return

            # Use a subset of timestamp history to match retrieval latency history length
            times = list(self.timestamp_history)[-len(self.retrieval_latency_history):]

            df = pd.DataFrame({
                'Time': times,
                'Retrieval Latency (s)': list(self.retrieval_latency_history)
            })

            st.line_chart(df.set_index('Time'))

        except ImportError:
            st.info("Install pandas for charts: pip install pandas")

    def _display_inference_latency_chart(self):
        """Display inference latency over time chart."""
        if not _ensure_streamlit():
            return
        try:
            import pandas as pd

            if not self.inference_latency_history:
                return

            times = list(self.timestamp_history)[-len(self.inference_latency_history):]

            df = pd.DataFrame({
                'Time': times,
                'Inference Latency (s)': list(self.inference_latency_history)
            })

            st.line_chart(df.set_index('Time'))

        except ImportError:
            st.info("Install pandas for charts: pip install pandas")

    def _display_ttft_chart(self):
        """Display time-to-first-token over time chart."""
        if not _ensure_streamlit():
            return
        try:
            import pandas as pd

            if not self.ttft_history:
                return

            times = list(self.timestamp_history)[-len(self.ttft_history):]

            df = pd.DataFrame({
                'Time': times,
                'TTFT (s)': list(self.ttft_history)
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