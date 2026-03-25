# core/timeout.py
"""
Timeout context managers for operations.

Provides timeout functionality for both Unix and Windows.
"""

import sys
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


@contextmanager
def timeout_unix(seconds: int, operation_name: str = "Operation"):
    """
    Timeout context manager for Unix systems.
    
    Uses signal.SIGALRM which is not available on Windows.
    
    Args:
        seconds: Timeout in seconds
        operation_name: Name of operation for error message
    
    Usage:
        with timeout_unix(30, "API call"):
            make_api_call()
    """
    import signal
    
    def handler(signum, frame):
        raise TimeoutError(f"{operation_name} timed out after {seconds}s")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@contextmanager
def timeout_windows(seconds: int, operation_name: str = "Operation"):
    """
    Timeout context manager for Windows.
    
    Uses threading.Timer instead of signal.
    
    Args:
        seconds: Timeout in seconds
        operation_name: Name of operation for error message
    
    Usage:
        with timeout_windows(30, "API call"):
            make_api_call()
    """
    import threading
    
    timeout_event = threading.Event()
    
    def timeout_handler():
        if not timeout_event.is_set():
            logger.error(f"{operation_name} timed out after {seconds}s")
            # Note: Can't actually interrupt the operation on Windows
            # This is a limitation - we just log the timeout
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    
    try:
        yield
    finally:
        timeout_event.set()
        timer.cancel()


# Cross-platform timeout
@contextmanager
def timeout(seconds: int, operation_name: str = "Operation"):
    """
    Cross-platform timeout context manager.
    
    Automatically selects the appropriate implementation for the platform.
    
    Args:
        seconds: Timeout in seconds
        operation_name: Name of operation for error message
    
    Usage:
        with timeout(30, "API call"):
            make_api_call()
    """
    if sys.platform == "win32":
        # Windows: use threading-based timeout
        # Note: This only logs timeout, doesn't interrupt
        with timeout_windows(seconds, operation_name):
            yield
    else:
        # Unix: use signal-based timeout
        with timeout_unix(seconds, operation_name):
            yield


def with_timeout(seconds: int, operation_name: str = "Operation"):
    """
    Decorator to apply timeout to function.
    
    Args:
        seconds: Timeout in seconds
        operation_name: Name of operation for error message
    
    Usage:
        @with_timeout(30, "API call")
        def make_api_call():
            ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with timeout(seconds, operation_name):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


# Import wraps for decorator
from functools import wraps
