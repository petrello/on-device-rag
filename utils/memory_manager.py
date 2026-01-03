"""Memory management utilities."""

import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def cleanup_memory():
    """Force garbage collection."""
    collected = gc.collect()
    logger.debug(f"Garbage collection: {collected} objects collected")


def get_memory_usage() -> Optional[dict]:
    """Get current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()

        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"Failed to get memory usage: {e}")
        return None


def check_memory_threshold() -> bool:
    """
    Check if memory usage exceeds threshold.

    Returns:
        True if cleanup needed
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