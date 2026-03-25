# tools/api_retry.py

"""
API retry with exponential backoff.

FLAW-8.3 FIX: API Retry with Exponential Backoff
- Automatic retry on transient failures
- Exponential backoff with jitter
- Rate limit handling
- Circuit breaker integration
"""

import os
import time
import random
import logging
from typing import Callable, Any, Optional, Tuple, Type
from functools import wraps

logger = logging.getLogger(__name__)

# Configuration (from env)
DEFAULT_MAX_RETRIES = int(os.environ.get("PROFESSOR_API_MAX_RETRIES", "5"))
DEFAULT_BASE_DELAY = float(os.environ.get("PROFESSOR_API_BASE_DELAY", "1.0"))
DEFAULT_MAX_DELAY = float(os.environ.get("PROFESSOR_API_MAX_DELAY", "60.0"))
DEFAULT_JITTER = float(os.environ.get("PROFESSOR_API_JITTER", "0.1"))


class APIRetryError(Exception):
    """Raised when API retries are exhausted."""
    def __init__(self, message: str, last_error: Optional[Exception] = None, attempts: int = 0):
        super().__init__(message)
        self.last_error = last_error
        self.attempts = attempts


def exponential_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for API retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Jitter factor (0.0-1.0)
        exceptions: Tuple of exceptions to catch
    
    Usage:
        @exponential_backoff(max_retries=5, base_delay=1.0)
        def call_api():
            return requests.get(url)
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_error = e
                    
                    # Check if should retry
                    if attempt >= max_retries:
                        logger.error(
                            f"[APIRetry] {func.__name__} failed after "
                            f"{max_retries + 1} attempts: {e}"
                        )
                        raise APIRetryError(
                            f"{func.__name__} failed after {max_retries + 1} attempts",
                            last_error=e,
                            attempts=attempt + 1,
                        ) from e
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    if jitter > 0:
                        jitter_amount = delay * jitter
                        delay += random.uniform(-jitter_amount, jitter_amount)
                    
                    # Log retry
                    logger.warning(
                        f"[APIRetry] {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): "
                        f"{e}. Retrying in {delay:.2f}s..."
                    )
                    
                    # Wait before retry
                    time.sleep(delay)
            
            # Should never reach here
            raise APIRetryError(
                f"{func.__name__} failed unexpectedly",
                last_error=last_error,
                attempts=max_retries + 1,
            )
        
        return wrapper
    
    return decorator


def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    **kwargs,
) -> Any:
    """
    Call function with retry and exponential backoff.
    
    Args:
        func: Function to call
        *args: Positional arguments
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Jitter factor
        exceptions: Exceptions to catch
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    
    Raises:
        APIRetryError: If all retries exhausted
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
            
        except exceptions as e:
            last_error = e
            
            if attempt >= max_retries:
                logger.error(
                    f"[APIRetry] {func.__name__} failed after "
                    f"{max_retries + 1} attempts: {e}"
                )
                raise APIRetryError(
                    f"{func.__name__} failed after {max_retries + 1} attempts",
                    last_error=e,
                    attempts=attempt + 1,
                ) from e
            
            # Calculate delay
            delay = min(base_delay * (2 ** attempt), max_delay)
            
            if jitter > 0:
                jitter_amount = delay * jitter
                delay += random.uniform(-jitter_amount, jitter_amount)
            
            logger.warning(
                f"[APIRetry] {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): "
                f"{e}. Retrying in {delay:.2f}s..."
            )
            
            time.sleep(delay)
    
    raise APIRetryError(
        f"{func.__name__} failed unexpectedly",
        last_error=last_error,
        attempts=max_retries + 1,
    )


class RateLimitHandler:
    """
    Handle API rate limits.
    
    Features:
    - Track request timestamps
    - Enforce rate limits
    - Automatic backoff on rate limit
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 60,
        max_requests_per_hour: int = 1000,
    ):
        """
        Initialize rate limit handler.
        
        Args:
            max_requests_per_minute: Max requests per minute
            max_requests_per_hour: Max requests per hour
        """
        self.max_per_minute = max_requests_per_minute
        self.max_per_hour = max_requests_per_hour
        self.requests_minute = []
        self.requests_hour = []
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Clean old timestamps
        self.requests_minute = [t for t in self.requests_minute if now - t < 60]
        self.requests_hour = [t for t in self.requests_hour if now - t < 3600]
        
        # Check minute limit
        if len(self.requests_minute) >= self.max_per_minute:
            oldest = min(self.requests_minute)
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logger.info(f"[RateLimit] Waiting {wait_time:.2f}s (minute limit)")
                time.sleep(wait_time)
                self.wait_if_needed()  # Recursive check
                return
        
        # Check hour limit
        if len(self.requests_hour) >= self.max_per_hour:
            oldest = min(self.requests_hour)
            wait_time = 3600 - (now - oldest)
            if wait_time > 0:
                logger.info(f"[RateLimit] Waiting {wait_time:.2f}s (hour limit)")
                time.sleep(wait_time)
                self.wait_if_needed()
                return
        
        # Record request
        self.requests_minute.append(now)
        self.requests_hour.append(now)
    
    def get_status(self) -> dict:
        """Get rate limit status."""
        now = time.time()
        
        return {
            "requests_last_minute": len([t for t in self.requests_minute if now - t < 60]),
            "requests_last_hour": len([t for t in self.requests_hour if now - t < 3600]),
            "limit_per_minute": self.max_per_minute,
            "limit_per_hour": self.max_per_hour,
            "minute_utilization": len(self.requests_minute) / self.max_per_minute * 100,
            "hour_utilization": len(self.requests_hour) / self.max_per_hour * 100,
        }


# Global rate limit handler
_rate_limiter: Optional[RateLimitHandler] = None


def get_rate_limiter() -> RateLimitHandler:
    """Get or create global rate limiter."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimitHandler(
            max_requests_per_minute=int(os.environ.get("PROFESSOR_API_RATE_PER_MINUTE", "60")),
            max_requests_per_hour=int(os.environ.get("PROFESSOR_API_RATE_PER_HOUR", "1000")),
        )
    
    return _rate_limiter


def rate_limited_call(func: Callable, *args, **kwargs) -> Any:
    """
    Call function with rate limiting.
    
    Args:
        func: Function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    """
    limiter = get_rate_limiter()
    limiter.wait_if_needed()
    
    return func(*args, **kwargs)
