# tools/cache_manager.py

"""
Caching strategy for expensive operations.

FLAW-6.2 FIX: Caching Strategy
- Caches expensive LLM calls
- Caches data processing results
- Cache invalidation by data hash
- Memory-efficient cache with TTL
"""

import os
import json
import hashlib
import logging
import time
import pickle
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache configuration (configurable via env)
CACHE_ENABLED = os.environ.get("PROFESSOR_CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_HOURS = int(os.environ.get("PROFESSOR_CACHE_TTL_HOURS", "24"))
CACHE_DIR = os.environ.get("PROFESSOR_CACHE_DIR", "cache")
MAX_CACHE_SIZE_MB = float(os.environ.get("PROFESSOR_MAX_CACHE_SIZE_MB", "500"))


class CacheEntry:
    """Represents a cached item with metadata."""
    
    def __init__(self, key: str, value: Any, ttl_hours: int = CACHE_TTL_HOURS):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(hours=ttl_hours)
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.expires_at
    
    def access(self) -> Any:
        """Record access and return value."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        return self.value
    
    def to_dict(self) -> dict:
        """Convert to serializable dict (metadata only)."""
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "is_expired": self.is_expired(),
        }


class CacheManager:
    """
    Manages caching of expensive operations.
    
    Features:
    - File-based persistence
    - TTL-based expiration
    - LRU eviction
    - Memory size limits
    """
    
    def __init__(
        self,
        cache_dir: str = CACHE_DIR,
        ttl_hours: int = CACHE_TTL_HOURS,
        max_size_mb: float = MAX_CACHE_SIZE_MB,
        enabled: bool = CACHE_ENABLED,
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.max_size_mb = max_size_mb
        self.enabled = enabled
        
        # In-memory cache
        self._cache: Dict[str, CacheEntry] = {}
        
        # Create cache directory
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
        
        logger.info(
            f"[CacheManager] Initialized -- dir: {cache_dir}, "
            f"ttl: {ttl_hours}h, max_size: {max_size_mb}MB, enabled: {enabled}"
        )
    
    def _compute_key(self, prefix: str, *args, **kwargs) -> str:
        """Compute cache key from arguments."""
        # Serialize args and kwargs
        data = {
            "prefix": prefix,
            "args": args,
            "kwargs": kwargs,
        }
        serialized = json.dumps(data, sort_keys=True, default=str)
        hash_digest = hashlib.sha256(serialized.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_digest}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        if not self.enabled:
            return None
        
        entry = self._cache.get(key)
        
        if entry is None:
            logger.debug(f"[CacheManager] Cache miss: {key}")
            return None
        
        if entry.is_expired():
            logger.debug(f"[CacheManager] Cache expired: {key}")
            self.delete(key)
            return None
        
        logger.debug(f"[CacheManager] Cache hit: {key} (access #{entry.access_count + 1})")
        return entry.access()
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_hours: Optional[int] = None,
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_hours: Override TTL in hours
        """
        if not self.enabled:
            return
        
        # Check size limit
        if self._get_cache_size_mb() >= self.max_size_mb:
            self._evict_oldest()
        
        # Create entry
        ttl = ttl_hours if ttl_hours is not None else self.ttl_hours
        entry = CacheEntry(key, value, ttl_hours=ttl)
        self._cache[key] = entry
        
        # Persist to disk
        self._save_to_disk(key)
        
        logger.debug(f"[CacheManager] Cache set: {key} (TTL: {ttl}h)")
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted
        """
        if key in self._cache:
            del self._cache[key]
            
            # Remove from disk
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            logger.debug(f"[CacheManager] Cache deleted: {key}")
            return True
        
        return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        
        # Clear disk cache
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        
        logger.info(f"[CacheManager] Cache cleared: {count} entries")
        return count
    
    def invalidate_by_prefix(self, prefix: str) -> int:
        """
        Invalidate all cache entries with given prefix.
        
        Args:
            prefix: Key prefix to invalidate
        
        Returns:
            Number of entries invalidated
        """
        keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
        
        for key in keys_to_delete:
            self.delete(key)
        
        logger.info(f"[CacheManager] Invalidated {len(keys_to_delete)} entries with prefix '{prefix}'")
        return len(keys_to_delete)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_entries = len(self._cache)
        expired_entries = sum(1 for e in self._cache.values() if e.is_expired())
        total_accesses = sum(e.access_count for e in self._cache.values())
        
        return {
            "enabled": self.enabled,
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "total_accesses": total_accesses,
            "cache_size_mb": round(self._get_cache_size_mb(), 2),
            "max_size_mb": self.max_size_mb,
            "ttl_hours": self.ttl_hours,
            "cache_dir": str(self.cache_dir),
        }
    
    def _get_cache_size_mb(self) -> float:
        """Get current cache size in MB."""
        if not self.cache_dir.exists():
            return 0.0
        
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        return total_size / (1024 * 1024)
    
    def _evict_oldest(self) -> int:
        """Evict oldest cache entries to make room."""
        if not self._cache:
            return 0
        
        # Sort by last accessed
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed,
        )
        
        # Evict oldest 10%
        evict_count = max(1, len(sorted_entries) // 10)
        evicted = 0
        
        for key, _ in sorted_entries[:evict_count]:
            self.delete(key)
            evicted += 1
        
        logger.info(f"[CacheManager] Evicted {evicted} oldest cache entries")
        return evicted
    
    def _save_to_disk(self, key: str) -> None:
        """Save cache entry to disk."""
        entry = self._cache.get(key)
        if entry is None:
            return
        
        cache_file = self.cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"[CacheManager] Failed to save cache entry {key}: {e}")
    
    def _load_from_disk(self) -> int:
        """Load cache entries from disk."""
        loaded = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    entry = pickle.load(f)
                
                # Only load if not expired
                if not entry.is_expired():
                    self._cache[entry.key] = entry
                    loaded += 1
                else:
                    # Remove expired file
                    cache_file.unlink()
            
            except Exception as e:
                logger.warning(f"[CacheManager] Failed to load cache entry {cache_file}: {e}")
                try:
                    cache_file.unlink()
                except:
                    pass
        
        logger.info(f"[CacheManager] Loaded {loaded} cache entries from disk")
        return loaded
    
    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.get_stats()
        
        logger.info("=" * 70)
        logger.info("CACHE STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Enabled: {stats['enabled']}")
        logger.info(f"Total entries: {stats['total_entries']}")
        logger.info(f"Active entries: {stats['active_entries']}")
        logger.info(f"Expired entries: {stats['expired_entries']}")
        logger.info(f"Total accesses: {stats['total_accesses']}")
        logger.info(f"Cache size: {stats['cache_size_mb']} MB / {stats['max_size_mb']} MB")
        logger.info(f"TTL: {stats['ttl_hours']} hours")
        logger.info("=" * 70)


# Global cache instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
    
    return _cache_manager


def cache_llm_call(prompt: str, model: str, response: str) -> None:
    """
    Cache LLM call result.
    
    Args:
        prompt: Input prompt
        model: Model name
        response: LLM response to cache
    """
    cache = get_cache_manager()
    key = cache._compute_key("llm", prompt=prompt, model=model)
    cache.set(key, {"response": response, "model": model})


def get_cached_llm_call(prompt: str, model: str) -> Optional[str]:
    """
    Get cached LLM call result.
    
    Args:
        prompt: Input prompt
        model: Model name
    
    Returns:
        Cached response or None
    """
    cache = get_cache_manager()
    key = cache._compute_key("llm", prompt=prompt, model=model)
    entry = cache.get(key)
    
    if entry and isinstance(entry, dict):
        return entry.get("response")
    
    return None


def cache_data_processing(data_hash: str, result: Any) -> None:
    """
    Cache data processing result.
    
    Args:
        data_hash: Hash of input data
        result: Processing result to cache
    """
    cache = get_cache_manager()
    key = cache._compute_key("data", data_hash=data_hash)
    cache.set(key, result)


def get_cached_data_processing(data_hash: str) -> Optional[Any]:
    """
    Get cached data processing result.
    
    Args:
        data_hash: Hash of input data
    
    Returns:
        Cached result or None
    """
    cache = get_cache_manager()
    key = cache._compute_key("data", data_hash=data_hash)
    return cache.get(key)


def invalidate_data_cache(data_hash: str) -> int:
    """
    Invalidate cache for specific data.
    
    Args:
        data_hash: Hash of data to invalidate
    
    Returns:
        Number of entries invalidated
    """
    cache = get_cache_manager()
    return cache.invalidate_by_prefix(f"data:{data_hash[:8]}")
