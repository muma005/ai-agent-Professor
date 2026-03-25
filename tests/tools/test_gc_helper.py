"""
Tests for GC helper.

FLAW-9.2: Explicit GC After Large Operations
"""
import pytest
from tools.gc_helper import (
    get_memory_usage_gb,
    run_gc,
    gc_if_needed,
    gc_after_operation,
    clear_large_objects,
    get_gc_stats,
)


class TestMemoryMonitoring:
    """Test memory monitoring functions."""

    def test_get_memory_usage_gb(self):
        """Test memory usage reading."""
        memory = get_memory_usage_gb()
        
        assert isinstance(memory, float)
        assert memory > 0.0
        assert memory < 32.0  # Reasonable upper bound


class TestGarbageCollection:
    """Test GC functions."""

    def test_run_gc(self):
        """Test running GC."""
        memory_before, memory_after = run_gc()
        
        assert isinstance(memory_before, float)
        assert isinstance(memory_after, float)
        assert memory_before >= 0.0
        assert memory_after >= 0.0

    def test_gc_if_needed_below_threshold(self):
        """Test GC not triggered below threshold."""
        result = gc_if_needed(threshold_gb=100.0)  # Very high threshold
        
        assert result is False

    def test_gc_after_operation(self):
        """Test GC after operation."""
        stats = gc_after_operation("test_operation")
        
        assert "operation" in stats
        assert "memory_before_gb" in stats
        assert "memory_after_gb" in stats
        assert "freed_gb" in stats
        assert stats["operation"] == "test_operation"

    def test_clear_large_objects(self):
        """Test clearing large objects."""
        # Create large objects
        large_objects = [[0] * 10000 for _ in range(10)]
        
        count = clear_large_objects(large_objects)
        
        assert count == 10
        assert len(large_objects) == 0

    def test_gc_stats(self):
        """Test GC statistics."""
        stats = get_gc_stats()
        
        assert "enabled" in stats
        assert "threshold" in stats
        assert "count" in stats
        assert "collections" in stats
        assert isinstance(stats["collections"], list)
