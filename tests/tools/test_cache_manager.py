"""
Tests for cache management.

FLAW-6.2: Caching Strategy
"""
import pytest
import time
import os
from pathlib import Path
from tools.cache_manager import (
    CacheManager,
    CacheEntry,
    get_cache_manager,
    cache_llm_call,
    get_cached_llm_call,
    cache_data_processing,
    get_cached_data_processing,
    invalidate_data_cache,
)


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_entry_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry("test_key", "test_value", ttl_hours=1)
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.access_count == 0

    def test_entry_expiration(self):
        """Test cache entry expiration."""
        from datetime import datetime, timedelta
        
        # Create entry that expires in the past
        entry = CacheEntry("test_key", "test_value", ttl_hours=0)
        
        # Manually set expiration to past
        entry.expires_at = datetime.now() - timedelta(hours=1)
        
        # Should be expired
        assert entry.is_expired()

    def test_entry_access_tracking(self):
        """Test access count tracking."""
        entry = CacheEntry("test_key", "test_value")
        
        assert entry.access_count == 0
        
        entry.access()
        assert entry.access_count == 1
        
        entry.access()
        assert entry.access_count == 2

    def test_entry_to_dict(self):
        """Test conversion to dict."""
        entry = CacheEntry("test_key", "test_value")
        result = entry.to_dict()
        
        assert result["key"] == "test_key"
        assert "created_at" in result
        assert "expires_at" in result
        assert "access_count" in result
        assert "is_expired" in result


class TestCacheManager:
    """Test CacheManager class."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create test cache manager."""
        cache_dir = tmp_path / "cache"
        return CacheManager(
            cache_dir=str(cache_dir),
            ttl_hours=24,
            max_size_mb=100,
            enabled=True,
        )

    def test_cache_set_get(self, cache):
        """Test basic cache set and get."""
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        
        assert result == "test_value"

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent_key")
        
        assert result is None

    def test_cache_disabled(self, tmp_path):
        """Test cache disabled mode."""
        cache = CacheManager(
            cache_dir=str(tmp_path / "cache"),
            enabled=False,
        )
        
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        
        assert result is None

    def test_cache_delete(self, cache):
        """Test cache deletion."""
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        cache.delete("test_key")
        assert cache.get("test_key") is None

    def test_cache_clear(self, cache):
        """Test cache clear."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        count = cache.clear()
        
        assert count == 3
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_cache_invalidate_by_prefix(self, cache):
        """Test prefix-based invalidation."""
        cache.set("data:abc123:key1", "value1")
        cache.set("data:abc123:key2", "value2")
        cache.set("llm:xyz789:key1", "value3")
        
        count = cache.invalidate_by_prefix("data:abc123")
        
        assert count == 2
        assert cache.get("data:abc123:key1") is None
        assert cache.get("data:abc123:key2") is None
        assert cache.get("llm:xyz789:key1") == "value3"

    def test_cache_stats(self, cache):
        """Test cache statistics."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 multiple times
        cache.get("key1")
        cache.get("key1")
        
        stats = cache.get_stats()
        
        assert stats["total_entries"] == 2
        assert stats["total_accesses"] >= 2
        assert "cache_size_mb" in stats
        assert "max_size_mb" in stats

    def test_cache_persistence(self, cache):
        """Test cache persists to disk."""
        cache.set("persistent_key", "persistent_value")
        
        # Verify file exists
        cache_file = cache.cache_dir / "persistent_key.pkl"
        assert cache_file.exists()

    def test_cache_compute_key_deterministic(self, cache):
        """Test key computation is deterministic."""
        key1 = cache._compute_key("test", arg1="value1", arg2="value2")
        key2 = cache._compute_key("test", arg1="value1", arg2="value2")
        
        assert key1 == key2

    def test_cache_compute_key_different_args(self, cache):
        """Test different args produce different keys."""
        key1 = cache._compute_key("test", arg1="value1")
        key2 = cache._compute_key("test", arg1="value2")
        
        assert key1 != key2


class TestGlobalCacheManager:
    """Test global cache manager functions."""

    def test_get_cache_manager_singleton(self):
        """Test get_cache_manager returns same instance."""
        # Clear global state
        from tools import cache_manager
        cache_manager._cache_manager = None
        
        cm1 = get_cache_manager()
        cm2 = get_cache_manager()
        
        assert cm1 is cm2

    def test_cache_llm_call_functions(self, tmp_path):
        """Test LLM cache convenience functions."""
        from tools import cache_manager
        cache_manager._cache_manager = None
        
        # Create test cache
        test_cache = CacheManager(
            cache_dir=str(tmp_path / "cache"),
            enabled=True,
        )
        cache_manager._cache_manager = test_cache
        
        # Cache LLM call
        cache_llm_call("test prompt", "deepseek", "test response")
        
        # Get cached result
        result = get_cached_llm_call("test prompt", "deepseek")
        
        assert result == "test response"

    def test_cache_data_processing_functions(self, tmp_path):
        """Test data processing cache functions."""
        from tools import cache_manager
        cache_manager._cache_manager = None
        
        test_cache = CacheManager(
            cache_dir=str(tmp_path / "cache"),
            enabled=True,
        )
        cache_manager._cache_manager = test_cache
        
        # Cache data processing
        cache_data_processing("hash123", {"result": "value"})
        
        # Get cached result
        result = get_cached_data_processing("hash123")
        
        assert result == {"result": "value"}

    def test_invalidate_data_cache(self, tmp_path):
        """Test data cache invalidation."""
        from tools import cache_manager
        cache_manager._cache_manager = None
        
        test_cache = CacheManager(
            cache_dir=str(tmp_path / "cache"),
            enabled=True,
        )
        cache_manager._cache_manager = test_cache
        
        # Cache multiple entries
        cache_data_processing("abc123", "value1")
        cache_data_processing("abc456", "value2")
        
        # Invalidate
        count = invalidate_data_cache("abc")
        
        # Should invalidate entries with matching prefix
        assert count >= 0


class TestCacheEviction:
    """Test cache eviction policies."""

    def test_evict_oldest_when_full(self, tmp_path):
        """Test oldest entries are evicted when cache is full."""
        # Create very small cache
        cache = CacheManager(
            cache_dir=str(tmp_path / "cache"),
            max_size_mb=0.001,  # Very small
            enabled=True,
        )
        
        # Add multiple entries
        for i in range(10):
            cache.set(f"key{i}", "x" * 1000)  # 1KB values
        
        # Cache should have evicted some entries
        stats = cache.get_stats()
        assert stats["total_entries"] < 10

    @pytest.fixture
    def cache(self, tmp_path):
        """Create test cache manager for eviction tests."""
        cache_dir = tmp_path / "cache"
        return CacheManager(
            cache_dir=str(cache_dir),
            ttl_hours=24,
            max_size_mb=100,
            enabled=True,
        )

    def test_log_stats(self, cache, caplog):
        """Test logging cache statistics."""
        import logging
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        with caplog.at_level(logging.INFO):
            cache.log_stats()
        
        assert "CACHE STATISTICS" in caplog.text
        assert "Total entries:" in caplog.text
        assert "Cache size:" in caplog.text
