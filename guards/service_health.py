# guards/service_health.py

import time
import logging
import functools
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class ServiceUnavailable(Exception):
    """Raised when a service fails all retries and has no fallback."""
    pass


def with_retry(
    max_attempts: int,
    base_delay_s: float,
    service_name: str,
    fallback: Optional[Callable] = None,
):
    """
    Decorator factory. Wraps any function with exponential backoff retry.
    If all attempts fail and fallback is provided, calls fallback(*args, **kwargs).
    If all attempts fail and no fallback, raises ServiceUnavailable.

    Usage:
        @with_retry(max_attempts=3, base_delay_s=2.0, service_name="Groq API")
        def call_groq(prompt: str) -> str:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    delay = base_delay_s * (2 ** (attempt - 1))
                    if attempt < max_attempts:
                        logger.warning(
                            f"[ServiceHealth] {service_name} attempt {attempt}/{max_attempts} "
                            f"failed: {e}. Retrying in {delay:.1f}s."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[ServiceHealth] {service_name} failed all {max_attempts} attempts. "
                            f"Last error: {e}."
                        )

            if fallback is not None:
                logger.warning(
                    f"[ServiceHealth] {service_name} unavailable. "
                    f"Activating fallback: {fallback.__name__}."
                )
                return fallback(*args, **kwargs)

            raise ServiceUnavailable(
                f"{service_name} failed {max_attempts} attempts. Last error: {last_error}. "
                f"No fallback configured."
            ) from last_error

        return wrapper
    return decorator


# ── Pre-configured fallbacks ──────────────────────────────────────────────────

def _groq_fallback(*args, **kwargs):
    """Fall back to Gemini Flash if Groq is down."""
    logger.warning("[ServiceHealth] Falling back to Gemini Flash (Groq unavailable).")
    from tools.llm_client import call_llm
    return call_llm(*args, model="gemini", **kwargs)


def _chromadb_fallback(*args, **kwargs):
    """Fall back to empty memory — log warning so engineer knows memory is cold."""
    logger.warning(
        "[ServiceHealth] ChromaDB unavailable. Returning empty memory. "
        "Optuna warm-start disabled for this session."
    )
    return []


def _redis_fallback(key: str, value=None, operation: str = "get"):
    """Fall back to in-memory dict for active session if Redis is unavailable."""
    logger.warning(
        "[ServiceHealth] Redis unavailable. Using in-memory state for this session. "
        "State will not survive process restart."
    )
    _memory_store = getattr(_redis_fallback, "_store", {})
    _redis_fallback._store = _memory_store
    if operation == "set":
        _memory_store[key] = value
        return True
    return _memory_store.get(key)


# ── Public retry-wrapped callables ────────────────────────────────────────────

@with_retry(max_attempts=3, base_delay_s=2.0, service_name="Groq API", fallback=_groq_fallback)
def call_groq_safe(prompt: str, model: str = "deepseek", **kwargs) -> str:
    from tools.llm_client import call_llm
    return call_llm(prompt=prompt, model=model, **kwargs)


@with_retry(max_attempts=2, base_delay_s=5.0, service_name="Docker Sandbox")
def run_in_sandbox_safe(code: str, timeout: int = 600, **kwargs):
    from tools.e2b_sandbox import run_in_sandbox
    return run_in_sandbox(code, timeout=timeout, **kwargs)


@with_retry(max_attempts=3, base_delay_s=60.0, service_name="Kaggle API")
def call_kaggle_api_safe(fn: Callable, *args, **kwargs):
    """Wrap any kaggle.api call with 60s exponential backoff."""
    return fn(*args, **kwargs)


@with_retry(max_attempts=3, base_delay_s=1.0, service_name="ChromaDB", fallback=_chromadb_fallback)
def query_chromadb_safe(collection, query_texts: list, n_results: int = 5):
    return collection.query(query_texts=query_texts, n_results=n_results)


@with_retry(max_attempts=2, base_delay_s=0.5, service_name="Redis")
def redis_set_safe(client, key: str, value: str, **kwargs):
    return client.set(key, value, **kwargs)


@with_retry(max_attempts=2, base_delay_s=0.5, service_name="Redis")
def redis_get_safe(client, key: str):
    return client.get(key)
