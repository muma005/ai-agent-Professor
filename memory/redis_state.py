# memory/redis_state.py

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_redis_client = None   # module-level singleton — built once, reused


def get_redis_client():
    """
    Returns a Redis client. Tries real Redis first, falls back to fakeredis.
    Fall back is logged as a WARNING — it is never silent.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db   = int(os.getenv("REDIS_DB", "0"))

    # Try real Redis first
    try:
        import redis
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            socket_connect_timeout=3,   # fail fast if Docker is not running
            socket_timeout=5,
            decode_responses=True,
        )
        client.ping()   # validates the connection immediately
        logger.info(f"[Redis] Connected to real Redis at {redis_host}:{redis_port}")
        _redis_client = client
        return _redis_client

    except Exception as real_redis_err:
        logger.warning(
            f"[Redis] Real Redis unavailable at {redis_host}:{redis_port}: {real_redis_err}. "
            f"Falling back to fakeredis. "
            f"WARNING: State will not persist across process restarts. "
            f"HITL checkpointing is disabled for this session. "
            f"Fix: docker run -d -p 6379:6379 redis:7-alpine"
        )
        try:
            import fakeredis
            _redis_client = fakeredis.FakeRedis(decode_responses=True)
        except ImportError:
            logger.warning(
                "[Redis] fakeredis not installed. Using in-memory dict fallback. "
                "pip install fakeredis to enable local Redis emulation."
            )
            _redis_client = _DictRedis()
        return _redis_client


class _DictRedis:
    """Minimal Redis-like interface backed by a plain dict.
    Used as last-resort fallback when neither redis nor fakeredis is available."""

    def __init__(self):
        self._store = {}
        self._ttls = {}

    def ping(self):
        return True

    def set(self, key, value, ex=None, **kwargs):
        import time
        self._store[key] = value
        if ex is not None:
            self._ttls[key] = time.time() + ex
        else:
            self._ttls.pop(key, None)
        return True

    def get(self, key):
        import time
        if key in self._ttls and time.time() > self._ttls[key]:
            self._store.pop(key, None)
            self._ttls.pop(key, None)
            return None
        return self._store.get(key)

    def delete(self, *keys):
        for k in keys:
            self._store.pop(k, None)
            self._ttls.pop(k, None)

    def exists(self, key):
        return key in self._store

    def ttl(self, key):
        if key not in self._store:
            return -2  # key does not exist
        return self._ttls.get(key, -1)  # -1 = no expiry set


def save_state(session_id: str, state: dict, ttl_seconds: int = 86400 * 7) -> bool:
    """
    Serialises and saves ProfessorState to Redis.
    Returns True on success, False on failure (never raises).
    """
    client = get_redis_client()
    key    = f"professor:state:{session_id}"
    try:
        payload = json.dumps({k: v for k, v in state.items() if _is_serialisable(v)})
        client.set(key, payload, ex=ttl_seconds)
        return True
    except Exception as e:
        logger.error(f"[Redis] Failed to save state for session {session_id}: {e}")
        return False


def load_state(session_id: str) -> Optional[dict]:
    """
    Loads ProfessorState from Redis. Returns None if not found.
    """
    client = get_redis_client()
    key    = f"professor:state:{session_id}"
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.error(f"[Redis] Failed to load state for session {session_id}: {e}")
        return None


def _is_serialisable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
