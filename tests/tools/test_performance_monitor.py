"""
Tests for performance and memory monitoring.

FLAW-6.1: Performance Monitoring
FLAW-9.1: Memory Profiling
"""
import pytest
import time
from tools.performance_monitor import (
    timed_node,
    NodeTiming,
    get_performance_summary,
    log_performance_report,
    SLOW_NODE_THRESHOLD_SEC,
    VERY_SLOW_THRESHOLD_SEC,
    HIGH_MEMORY_THRESHOLD_GB,
    CRITICAL_MEMORY_THRESHOLD_GB,
    _get_memory_gb,
)


class TestNodeTiming:
    """Test NodeTiming dataclass."""

    def test_node_timing_creation(self):
        """Test NodeTiming can be created."""
        import time
        from datetime import datetime, timezone
        
        start = time.monotonic()
        time.sleep(0.1)
        end = time.monotonic()
        
        timing = NodeTiming(
            node_name="test_node",
            start_time=start,
            end_time=end,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            start_memory_gb=2.0,
            end_memory_gb=2.1,
            peak_memory_gb=2.1,
        )
        
        assert timing.node_name == "test_node"
        assert timing.duration_sec >= 0.1
        assert timing.duration_sec < 1.0
        assert timing.start_memory_gb == 2.0
        assert timing.end_memory_gb == 2.1
        assert abs(timing.memory_delta_gb - 0.1) < 1e-6  # Floating point tolerance

    def test_node_timing_to_dict(self):
        """Test NodeTiming converts to dict."""
        import time
        from datetime import datetime, timezone
        
        start = time.monotonic()
        time.sleep(0.05)
        end = time.monotonic()
        
        timing = NodeTiming(
            node_name="test_node",
            start_time=start,
            end_time=end,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            start_memory_gb=2.0,
            end_memory_gb=2.1,
            peak_memory_gb=2.1,
        )
        
        result = timing.to_dict()
        
        assert result["node_name"] == "test_node"
        assert "duration_sec" in result
        assert "started_at" in result
        assert "ended_at" in result
        assert "is_slow" in result
        assert "is_very_slow" in result
        # Memory metrics
        assert "start_memory_gb" in result
        assert "end_memory_gb" in result
        assert "peak_memory_gb" in result
        assert "memory_delta_gb" in result
        assert "is_high_memory" in result
        assert "is_critical_memory" in result


class TestTimedNodeDecorator:
    """Test timed_node decorator."""

    def test_timed_node_adds_timing_to_state(self):
        """Test decorator adds timing to state."""
        
        @timed_node
        def dummy_node(state):
            return {"result": "success"}
        
        state = {}
        result = dummy_node(state)
        
        assert "performance_log" in result
        assert len(result["performance_log"]) == 1
        assert result["performance_log"][0]["node_name"] == "dummy_node"
        assert "duration_sec" in result["performance_log"][0]
        # Memory metrics
        assert "start_memory_gb" in result["performance_log"][0]
        assert "end_memory_gb" in result["performance_log"][0]
        assert "peak_memory_gb" in result["performance_log"][0]

    def test_timed_node_preserves_existing_result(self):
        """Test decorator preserves node's return values."""
        
        @timed_node
        def dummy_node(state):
            return {"custom_key": "custom_value", "number": 42}
        
        state = {}
        result = dummy_node(state)
        
        assert result["custom_key"] == "custom_value"
        assert result["number"] == 42
        assert "performance_log" in result

    def test_timed_node_appends_to_existing_log(self):
        """Test decorator appends to existing performance_log."""
        
        @timed_node
        def dummy_node(state):
            return {"result": "success"}
        
        state = {
            "performance_log": [
                {"node_name": "previous_node", "duration_sec": 1.0}
            ]
        }
        result = dummy_node(state)
        
        assert len(result["performance_log"]) == 2
        assert result["performance_log"][0]["node_name"] == "previous_node"
        assert result["performance_log"][1]["node_name"] == "dummy_node"

    def test_timed_node_records_failure(self, caplog):
        """Test decorator records timing even on failure."""
        import logging
        
        @timed_node
        def failing_node(state):
            raise ValueError("Test failure")
        
        state = {}
        
        with pytest.raises(ValueError, match="Test failure"):
            failing_node(state)
        
        # Should have logged the failure
        assert "Failed after" in caplog.text
        assert "failing_node" in caplog.text


class TestPerformanceSummary:
    """Test performance summary generation."""

    def test_empty_log(self):
        """Test summary with empty log."""
        summary = get_performance_summary([])
        
        assert summary["total_duration_sec"] == 0
        assert summary["node_count"] == 0
        assert summary["avg_duration_sec"] == 0
        assert summary["slow_nodes"] == []
        assert summary["very_slow_nodes"] == []
        assert summary["failed_nodes"] == []

    def test_single_node(self):
        """Test summary with single node."""
        log = [
            {"node_name": "node1", "duration_sec": 5.0, "is_slow": False, "is_very_slow": False}
        ]
        summary = get_performance_summary(log)
        
        assert summary["total_duration_sec"] == 5.0
        assert summary["node_count"] == 1
        assert summary["avg_duration_sec"] == 5.0

    def test_multiple_nodes(self):
        """Test summary with multiple nodes."""
        log = [
            {"node_name": "node1", "duration_sec": 5.0, "is_slow": False, "is_very_slow": False},
            {"node_name": "node2", "duration_sec": 10.0, "is_slow": False, "is_very_slow": False},
            {"node_name": "node3", "duration_sec": 15.0, "is_slow": False, "is_very_slow": False},
        ]
        summary = get_performance_summary(log)
        
        assert summary["total_duration_sec"] == 30.0
        assert summary["node_count"] == 3
        assert summary["avg_duration_sec"] == 10.0

    def test_detects_slow_nodes(self):
        """Test slow nodes are detected."""
        log = [
            {"node_name": "fast_node", "duration_sec": 5.0, "is_slow": False, "is_very_slow": False},
            {"node_name": "slow_node", "duration_sec": SLOW_NODE_THRESHOLD_SEC + 10, "is_slow": True, "is_very_slow": False},
        ]
        summary = get_performance_summary(log)
        
        assert "slow_node" in summary["slow_nodes"]
        assert "fast_node" not in summary["slow_nodes"]

    def test_detects_very_slow_nodes(self):
        """Test very slow nodes are detected."""
        log = [
            {"node_name": "normal_node", "duration_sec": 10.0, "is_slow": False, "is_very_slow": False},
            {"node_name": "very_slow_node", "duration_sec": VERY_SLOW_THRESHOLD_SEC + 30, "is_slow": True, "is_very_slow": True},
        ]
        summary = get_performance_summary(log)
        
        assert "very_slow_node" in summary["very_slow_nodes"]
        assert "very_slow_node" in summary["slow_nodes"]

    def test_detects_failed_nodes(self):
        """Test failed nodes are detected."""
        log = [
            {"node_name": "success_node", "duration_sec": 5.0, "failed": False},
            {"node_name": "failed_node", "duration_sec": 2.0, "failed": True, "error": "Test error"},
        ]
        summary = get_performance_summary(log)
        
        assert "failed_node" in summary["failed_nodes"]
        assert "success_node" not in summary["failed_nodes"]


class TestPerformanceReport:
    """Test performance report logging."""

    def test_log_performance_report(self, caplog):
        """Test performance report is logged."""
        import logging
        
        log = [
            {"node_name": "node1", "duration_sec": 5.0, "is_slow": False, "is_very_slow": False},
            {"node_name": "node2", "duration_sec": 10.0, "is_slow": False, "is_very_slow": False},
        ]
        
        with caplog.at_level(logging.INFO):
            log_performance_report(log)

        # Updated for FLAW-9.1: Now includes memory in report title
        assert "PERFORMANCE & MEMORY REPORT" in caplog.text
        assert "Total nodes: 2" in caplog.text
        assert "Total duration:" in caplog.text


class TestThresholds:
    """Test threshold configuration."""

    def test_slow_threshold_default(self):
        """Test default slow threshold."""
        assert SLOW_NODE_THRESHOLD_SEC >= 10  # At least 10 seconds

    def test_very_slow_threshold_default(self):
        """Test default very slow threshold."""
        assert VERY_SLOW_THRESHOLD_SEC >= 60  # At least 60 seconds
        assert VERY_SLOW_THRESHOLD_SEC > SLOW_NODE_THRESHOLD_SEC


class TestMemoryMonitoring:
    """Test memory monitoring (FLAW-9.1)."""

    def test_get_memory_gb(self):
        """Test memory reading function."""
        memory = _get_memory_gb()
        
        assert isinstance(memory, float)
        assert memory > 0.0
        assert memory < 32.0  # Reasonable upper bound

    def test_memory_delta_tracked(self):
        """Test memory delta is tracked."""
        
        @timed_node
        def memory_node(state):
            # Allocate some memory
            data = [0] * 1000000
            time.sleep(0.01)
            return {"result": "done"}
        
        state = {}
        result = memory_node(state)
        
        timing = result["performance_log"][0]
        assert "memory_delta_gb" in timing
        assert isinstance(timing["memory_delta_gb"], float)

    def test_high_memory_detection(self):
        """Test high memory nodes are detected."""
        # Create a log entry that exceeds threshold
        log = [
            {
                "node_name": "memory_hog",
                "duration_sec": 5.0,
                "peak_memory_gb": HIGH_MEMORY_THRESHOLD_GB + 1.0,
                "is_high_memory": True,
                "is_critical_memory": False,
            }
        ]
        summary = get_performance_summary(log)
        
        assert "memory_hog" in summary["high_memory_nodes"]
        assert summary["peak_memory_gb"] > HIGH_MEMORY_THRESHOLD_GB

    def test_critical_memory_detection(self):
        """Test critical memory nodes are detected."""
        log = [
            {
                "node_name": "critical_memory_node",
                "duration_sec": 5.0,
                "peak_memory_gb": CRITICAL_MEMORY_THRESHOLD_GB + 0.5,
                "is_high_memory": True,
                "is_critical_memory": True,
            }
        ]
        summary = get_performance_summary(log)
        
        assert "critical_memory_node" in summary["critical_memory_nodes"]
        assert "critical_memory_node" in summary["high_memory_nodes"]

    def test_memory_summary_includes_metrics(self):
        """Test memory metrics in summary."""
        log = [
            {
                "node_name": "node1",
                "duration_sec": 5.0,
                "start_memory_gb": 2.0,
                "end_memory_gb": 2.1,
                "peak_memory_gb": 2.1,
                "memory_delta_gb": 0.1,
                "is_high_memory": False,
                "is_critical_memory": False,
            },
            {
                "node_name": "node2",
                "duration_sec": 10.0,
                "start_memory_gb": 2.1,
                "end_memory_gb": 2.3,
                "peak_memory_gb": 2.3,
                "memory_delta_gb": 0.2,
                "is_high_memory": False,
                "is_critical_memory": False,
            },
        ]
        summary = get_performance_summary(log)
        
        assert "peak_memory_gb" in summary
        assert "total_memory_delta_gb" in summary
        assert "high_memory_nodes" in summary
        assert "critical_memory_nodes" in summary
        assert summary["peak_memory_gb"] >= 2.3
        assert abs(summary["total_memory_delta_gb"] - 0.3) < 0.01


class TestPerformanceReportWithMemory:
    """Test performance report includes memory (FLAW-9.1)."""

    def test_log_performance_report_includes_memory(self, caplog):
        """Test memory metrics are logged."""
        import logging
        
        log = [
            {
                "node_name": "node1",
                "duration_sec": 5.0,
                "peak_memory_gb": 2.5,
                "memory_delta_gb": 0.1,
                "is_high_memory": False,
                "is_critical_memory": False,
            }
        ]
        
        with caplog.at_level(logging.INFO):
            log_performance_report(log)
        
        assert "PERFORMANCE & MEMORY REPORT" in caplog.text
        assert "Peak memory:" in caplog.text
        assert "Total memory delta:" in caplog.text
