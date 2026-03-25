# tools/gc_helper.py

"""
Garbage collection helper for memory optimization.

FLAW-9.2 FIX: Explicit GC After Large Operations
- Explicit GC after memory-intensive operations
- Memory monitoring before/after GC
- Automatic GC triggers at thresholds
"""

import gc
import logging
import psutil
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# GC thresholds (configurable via env)
GC_MEMORY_THRESHOLD_GB = float(os.environ.get("PROFESSOR_GC_MEMORY_THRESHOLD", "4.0"))
GC_FORCE_THRESHOLD_GB = float(os.environ.get("PROFESSOR_GC_FORCE_THRESHOLD", "5.0"))


def get_memory_usage_gb() -> float:
    """
    Get current process memory usage in GB.
    
    Returns:
        Memory usage in GB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def run_gc(verbose: bool = False) -> Tuple[float, float]:
    """
    Run garbage collection and return memory stats.
    
    Args:
        verbose: Log detailed GC info
    
    Returns:
        (memory_before_gb, memory_after_gb)
    """
    memory_before = get_memory_usage_gb()
    
    # Run GC
    gc.collect()
    
    memory_after = get_memory_usage_gb()
    freed = memory_before - memory_after
    
    if verbose or freed > 0.1:  # Log if freed > 100MB
        logger.info(
            f"[GC] Memory: {memory_before:.2f} GB → {memory_after:.2f} GB "
            f"(freed: {freed:.2f} GB)"
        )
    
    return memory_before, memory_after


def gc_if_needed(
    threshold_gb: float = GC_MEMORY_THRESHOLD_GB,
    force_threshold_gb: float = GC_FORCE_THRESHOLD_GB,
    verbose: bool = False,
) -> bool:
    """
    Run GC if memory usage exceeds threshold.
    
    Args:
        threshold_gb: Run GC if memory exceeds this
        force_threshold_gb: Force GC if memory exceeds this
        verbose: Log detailed info
    
    Returns:
        True if GC was run
    """
    memory_usage = get_memory_usage_gb()
    
    # Force GC if above force threshold
    if memory_usage > force_threshold_gb:
        logger.warning(
            f"[GC] Memory critical: {memory_usage:.2f} GB > {force_threshold_gb} GB. "
            f"Forcing GC."
        )
        run_gc(verbose=verbose)
        return True
    
    # Run GC if above normal threshold
    if memory_usage > threshold_gb:
        logger.info(
            f"[GC] Memory high: {memory_usage:.2f} GB > {threshold_gb} GB. "
            f"Running GC."
        )
        run_gc(verbose=verbose)
        return True
    
    return False


def gc_after_operation(
    operation_name: str,
    threshold_gb: float = GC_MEMORY_THRESHOLD_GB,
) -> dict:
    """
    Run GC after a memory-intensive operation.
    
    Usage:
        result = expensive_operation()
        gc_after_operation("expensive_operation")
    
    Args:
        operation_name: Name of operation for logging
        threshold_gb: Memory threshold to trigger GC
    
    Returns:
        GC stats dict
    """
    memory_before = get_memory_usage_gb()
    
    # Run GC
    gc.collect()
    
    memory_after = get_memory_usage_gb()
    freed = memory_before - memory_after
    
    stats = {
        "operation": operation_name,
        "memory_before_gb": round(memory_before, 3),
        "memory_after_gb": round(memory_after, 3),
        "freed_gb": round(freed, 3),
        "gc_triggered": freed > 0.05,  # Freed > 50MB
    }
    
    if stats["gc_triggered"]:
        logger.info(
            f"[GC] After {operation_name}: "
            f"{memory_before:.2f} GB → {memory_after:.2f} GB "
            f"(freed: {freed:.2f} GB)"
        )
    
    return stats


def clear_large_objects(objects: list) -> int:
    """
    Clear list of large objects and run GC.
    
    Args:
        objects: List of objects to clear
    
    Returns:
        Number of objects cleared
    """
    count = len(objects)
    
    # Delete objects
    for i, obj in enumerate(objects):
        objects[i] = None
    
    # Clear list
    objects.clear()
    
    # Run GC
    gc.collect()
    
    logger.debug(f"[GC] Cleared {count} large objects")
    return count


def get_gc_stats() -> dict:
    """
    Get garbage collector statistics.
    
    Returns:
        GC stats dict
    """
    stats = gc.get_stats()
    
    result = {
        "enabled": gc.isenabled(),
        "threshold": gc.get_threshold(),
        "count": gc.get_count(),
        "collections": [],
    }
    
    for i, gen_stats in enumerate(stats):
        result["collections"].append({
            "generation": i,
            "collections": gen_stats.get("collections", 0),
            "collected": gen_stats.get("collected", 0),
            "uncollectable": gen_stats.get("uncollectable", 0),
        })
    
    return result


def log_gc_summary() -> None:
    """Log GC summary."""
    stats = get_gc_stats()
    memory = get_memory_usage_gb()
    
    logger.info("=" * 70)
    logger.info("GARBAGE COLLECTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"GC enabled: {stats['enabled']}")
    logger.info(f"GC threshold: {stats['threshold']}")
    logger.info(f"GC count: {stats['count']}")
    logger.info(f"Current memory: {memory:.2f} GB")
    
    for gen in stats["collections"]:
        logger.info(
            f"Gen {gen['generation']}: "
            f"collections={gen['collections']}, "
            f"collected={gen['collected']}, "
            f"uncollectable={gen['uncollectable']}"
        )
    
    logger.info("=" * 70)
