"""
Memory management utilities.

Provides tools for monitoring and managing memory usage on
resource-constrained devices.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)


class MemoryInfo(TypedDict):
    """Memory usage information."""
    rss_mb: float
    vms_mb: float
    percent: float


def cleanup_memory() -> int:
    """
    Force garbage collection to free unused memory.

    Returns:
        Number of objects collected.
    """
    collected = gc.collect()
    logger.debug(f"Garbage collection freed {collected} objects")
    return collected


def get_memory_usage() -> Optional[MemoryInfo]:
    """
    Get current process memory usage.

    Returns:
        MemoryInfo dict with RSS, VMS, and percentage, or None if unavailable.
    """
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()

        return MemoryInfo(
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            percent=process.memory_percent(),
        )
    except ImportError:
        logger.debug("psutil not available for memory monitoring")
        return None
    except Exception as e:
        logger.debug(f"Failed to get memory usage: {e}")
        return None


def check_memory_threshold() -> bool:
    """
    Check if memory usage exceeds the configured threshold.

    Returns:
        True if cleanup is needed, False otherwise.
    """
    from config import settings

    mem_info = get_memory_usage()
    if mem_info and mem_info["rss_mb"] > settings.AUTO_CLEANUP_THRESHOLD_MB:
        logger.warning(
            f"Memory usage {mem_info['rss_mb']:.0f}MB exceeds "
            f"threshold {settings.AUTO_CLEANUP_THRESHOLD_MB}MB"
        )
        return True

    return False