# tools/performance_monitor.py

"""
Performance and memory monitoring for Professor pipeline nodes.

FLAW-6.1 FIX: Performance Monitoring
- Tracks execution time per node
- Logs performance metrics to state
- Alerts on slow-running nodes

FLAW-9.1 FIX: Memory Profiling
- Tracks memory usage per node
- Records peak RSS memory
- Alerts on high memory consumption
"""

import os
import gc
import time
import logging
import psutil
from functools import wraps
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Performance thresholds (configurable via env)
SLOW_NODE_THRESHOLD_SEC = float(os.environ.get("PROFESSOR_SLOW_NODE_THRESHOLD", "30"))
VERY_SLOW_THRESHOLD_SEC = float(os.environ.get("PROFESSOR_VERY_SLOW_THRESHOLD", "120"))

# Memory thresholds (configurable via env, in GB)
HIGH_MEMORY_THRESHOLD_GB = float(os.environ.get("PROFESSOR_HIGH_MEMORY_THRESHOLD", "4.0"))
CRITICAL_MEMORY_THRESHOLD_GB = float(os.environ.get("PROFESSOR_CRITICAL_MEMORY_THRESHOLD", "5.5"))
MAX_MEMORY_GB = float(os.environ.get("PROFESSOR_MAX_MEMORY_GB", "6.0"))


class NodeTiming:
    """Records timing and memory information for a pipeline node."""
    
    def __init__(
        self,
        node_name: str,
        start_time: float,
        end_time: float,
        started_at: datetime,
        ended_at: datetime,
        start_memory_gb: float,
        end_memory_gb: float,
        peak_memory_gb: float,
    ):
        self.node_name = node_name
        self.start_time = start_time  # monotonic
        self.end_time = end_time  # monotonic
        self.started_at = started_at  # wall clock
        self.ended_at = ended_at  # wall clock
        self.duration_sec = end_time - start_time
        self.start_memory_gb = start_memory_gb
        self.end_memory_gb = end_memory_gb
        self.peak_memory_gb = peak_memory_gb
        self.memory_delta_gb = end_memory_gb - start_memory_gb
    
    def to_dict(self) -> dict:
        """Convert to serializable dict for state/logging."""
        return {
            "node_name": self.node_name,
            "duration_sec": round(self.duration_sec, 3),
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "is_slow": self.duration_sec > SLOW_NODE_THRESHOLD_SEC,
            "is_very_slow": self.duration_sec > VERY_SLOW_THRESHOLD_SEC,
            # Memory metrics (FLAW-9.1)
            "start_memory_gb": round(self.start_memory_gb, 3),
            "end_memory_gb": round(self.end_memory_gb, 3),
            "peak_memory_gb": round(self.peak_memory_gb, 3),
            "memory_delta_gb": round(self.memory_delta_gb, 3),
            "is_high_memory": self.peak_memory_gb > HIGH_MEMORY_THRESHOLD_GB,
            "is_critical_memory": self.peak_memory_gb > CRITICAL_MEMORY_THRESHOLD_GB,
        }


def _get_memory_gb() -> float:
    """Get current RSS memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def timed_node(func: Callable) -> Callable:
    """
    Decorator to add timing and memory monitoring to pipeline nodes.
    
    Usage:
        @timed_node
        def run_ml_optimizer(state):
            ...
    """
    @wraps(func)
    def wrapper(state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        node_name = func.__name__
        
        # Record start (time + memory)
        start_mono = time.monotonic()
        start_wall = datetime.now(timezone.utc)
        start_memory = _get_memory_gb()
        peak_memory = start_memory
        
        logger.info(
            f"[{node_name}] Starting at {start_wall.isoformat()} "
            f"(memory: {start_memory:.2f} GB)"
        )
        
        try:
            # Execute node
            result = func(state, *args, **kwargs)
            
            # Record end (time + memory)
            end_mono = time.monotonic()
            end_wall = datetime.now(timezone.utc)
            end_memory = _get_memory_gb()
            duration = end_mono - start_mono
            
            # Log timing and memory
            timing = NodeTiming(
                node_name=node_name,
                start_time=start_mono,
                end_time=end_mono,
                started_at=start_wall,
                ended_at=end_wall,
                start_memory_gb=start_memory,
                end_memory_gb=end_memory,
                peak_memory_gb=peak_memory,
            )
            
            # Log warnings for slow or memory-heavy nodes
            if duration > VERY_SLOW_THRESHOLD_SEC:
                logger.warning(
                    f"[{node_name}] VERY SLOW: {duration:.2f}s "
                    f"(threshold: {VERY_SLOW_THRESHOLD_SEC}s)"
                )
            elif duration > SLOW_NODE_THRESHOLD_SEC:
                logger.warning(
                    f"[{node_name}] Slow: {duration:.2f}s "
                    f"(threshold: {SLOW_NODE_THRESHOLD_SEC}s)"
                )
            else:
                logger.info(f"[{node_name}] Completed in {duration:.2f}s")
            
            # Memory logging
            if end_memory > CRITICAL_MEMORY_THRESHOLD_GB:
                logger.warning(
                    f"[{node_name}] CRITICAL MEMORY: {end_memory:.2f} GB "
                    f"(threshold: {CRITICAL_MEMORY_THRESHOLD_GB} GB)"
                )
            elif end_memory > HIGH_MEMORY_THRESHOLD_GB:
                logger.warning(
                    f"[{node_name}] High memory: {end_memory:.2f} GB "
                    f"(threshold: {HIGH_MEMORY_THRESHOLD_GB} GB)"
                )
            else:
                logger.info(
                    f"[{node_name}] Memory: {end_memory:.2f} GB "
                    f"(delta: {timing.memory_delta_gb:+.3f} GB)"
                )
            
            # Add timing to state for downstream tracking
            if result is None:
                result = {}
            
            # Initialize performance_log if not present
            if "performance_log" not in state:
                state["performance_log"] = []
            
            # Append timing (don't mutate input state dict)
            result = dict(result)
            result["performance_log"] = state.get("performance_log", []) + [timing.to_dict()]
            
            return result
            
        except Exception as e:
            # Record failure timing and memory
            end_mono = time.monotonic()
            end_wall = datetime.now(timezone.utc)
            end_memory = _get_memory_gb()
            duration = end_mono - start_mono
            
            logger.error(
                f"[{node_name}] Failed after {duration:.2f}s: {e} "
                f"(memory: {end_memory:.2f} GB)"
            )
            
            # Still record the partial timing
            if "performance_log" not in state:
                state["performance_log"] = []
            
            timing = NodeTiming(
                node_name=node_name,
                start_time=start_mono,
                end_time=end_mono,
                started_at=start_wall,
                ended_at=end_wall,
                start_memory_gb=start_memory,
                end_memory_gb=end_memory,
                peak_memory_gb=peak_memory,
            )
            
            # Add failure info
            timing_info = timing.to_dict()
            timing_info["error"] = str(e)
            timing_info["failed"] = True
            
            state["performance_log"] = state.get("performance_log", []) + [timing_info]
            
            # Re-raise
            raise
    
    return wrapper


def get_performance_summary(performance_log: list) -> dict:
    """
    Generate performance and memory summary from performance_log.
    
    Returns:
        Summary dict with total time, avg time, slow nodes, memory stats, etc.
    """
    if not performance_log:
        return {
            "total_duration_sec": 0,
            "node_count": 0,
            "avg_duration_sec": 0,
            "slow_nodes": [],
            "very_slow_nodes": [],
            "failed_nodes": [],
            # Memory metrics (FLAW-9.1)
            "peak_memory_gb": 0,
            "total_memory_delta_gb": 0,
            "high_memory_nodes": [],
            "critical_memory_nodes": [],
        }
    
    total_duration = sum(entry.get("duration_sec", 0) for entry in performance_log)
    slow_nodes = [e for e in performance_log if e.get("is_slow")]
    very_slow_nodes = [e for e in performance_log if e.get("is_very_slow")]
    failed_nodes = [e for e in performance_log if e.get("failed")]
    
    # Memory metrics
    peak_memory = max(entry.get("peak_memory_gb", 0) for entry in performance_log)
    total_memory_delta = sum(entry.get("memory_delta_gb", 0) for entry in performance_log)
    high_memory_nodes = [e for e in performance_log if e.get("is_high_memory")]
    critical_memory_nodes = [e for e in performance_log if e.get("is_critical_memory")]
    
    return {
        "total_duration_sec": round(total_duration, 3),
        "node_count": len(performance_log),
        "avg_duration_sec": round(total_duration / len(performance_log), 3) if performance_log else 0,
        "slow_nodes": [e["node_name"] for e in slow_nodes],
        "very_slow_nodes": [e["node_name"] for e in very_slow_nodes],
        "failed_nodes": [e["node_name"] for e in failed_nodes],
        # Memory metrics (FLAW-9.1)
        "peak_memory_gb": round(peak_memory, 3),
        "total_memory_delta_gb": round(total_memory_delta, 3),
        "high_memory_nodes": [e["node_name"] for e in high_memory_nodes],
        "critical_memory_nodes": [e["node_name"] for e in critical_memory_nodes],
    }


def log_performance_report(performance_log: list) -> None:
    """Log a formatted performance and memory report."""
    summary = get_performance_summary(performance_log)
    
    logger.info("=" * 70)
    logger.info("PERFORMANCE & MEMORY REPORT")
    logger.info("=" * 70)
    logger.info(f"Total nodes: {summary['node_count']}")
    logger.info(f"Total duration: {summary['total_duration_sec']:.2f}s")
    logger.info(f"Average node time: {summary['avg_duration_sec']:.2f}s")
    logger.info(f"Peak memory: {summary['peak_memory_gb']:.2f} GB")
    logger.info(f"Total memory delta: {summary['total_memory_delta_gb']:+.3f} GB")
    
    if summary['slow_nodes']:
        logger.warning(f"Slow nodes (> {SLOW_NODE_THRESHOLD_SEC}s): {summary['slow_nodes']}")
    
    if summary['very_slow_nodes']:
        logger.warning(f"Very slow nodes (> {VERY_SLOW_THRESHOLD_SEC}s): {summary['very_slow_nodes']}")
    
    if summary['high_memory_nodes']:
        logger.warning(f"High memory nodes (> {HIGH_MEMORY_THRESHOLD_GB} GB): {summary['high_memory_nodes']}")
    
    if summary['critical_memory_nodes']:
        logger.error(f"Critical memory nodes (> {CRITICAL_MEMORY_THRESHOLD_GB} GB): {summary['critical_memory_nodes']}")
    
    if summary['failed_nodes']:
        logger.error(f"Failed nodes: {summary['failed_nodes']}")
    
    logger.info("=" * 70)
