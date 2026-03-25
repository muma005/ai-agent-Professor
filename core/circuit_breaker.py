# core/circuit_breaker.py
"""
Circuit breaker pattern for API calls.

Prevents cascading failures by:
1. Tracking failure count
2. Opening circuit after threshold
3. Allowing test requests after timeout
4. Closing circuit on success
"""

import time
from typing import Optional, Callable, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class APICircuitBreaker:
    """
    Circuit breaker for API calls.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "API",
        max_calls_per_minute: int = 10,
        budget_limit: float = 2.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        self.max_calls_per_minute = max_calls_per_minute
        self.budget_limit = budget_limit
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self.calls = []  # timestamps of recent calls
        self.total_cost = 0.0
    
    def can_make_call(self) -> bool:
        """Check if call is allowed."""
        now = time.time()
        
        # Check circuit state
        if self.state == "open":
            if now - self.last_failure_time < self.recovery_timeout:
                logger.warning(f"{self.name} circuit breaker is OPEN")
                return False
            else:
                # Try half-open
                self.state = "half-open"
                logger.info(f"{self.name} circuit breaker is HALF-OPEN (testing)")
        
        # Check rate limit
        recent_calls = [t for t in self.calls if now - t < 60]
        if len(recent_calls) >= self.max_calls_per_minute:
            logger.warning(f"{self.name} rate limit exceeded")
            return False
        
        # Check budget
        if self.total_cost >= self.budget_limit * 0.9:
            logger.warning(f"{self.name} approaching budget limit (90%)")
            return False
        
        return True
    
    def record_call(self, cost: float = 0.0):
        """Record successful API call."""
        now = time.time()
        self.calls.append(now)
        self.calls = [t for t in self.calls if now - t < 60]  # Keep last minute
        self.total_cost += cost
        
        # Reset on success
        if self.state == "half-open":
            logger.info(f"{self.name} circuit breaker CLOSED (recovered)")
            self.state = "closed"
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed API call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(
                f"{self.name} circuit breaker OPEN "
                f"({self.failure_count} failures)"
            )
    
    def record_cost(self, cost: float):
        """Record API cost."""
        self.total_cost += cost
    
    def reset(self):
        """Reset circuit breaker state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"
        self.calls = []
        self.total_cost = 0.0


# Decorator for easy usage
def with_circuit_breaker(
    breaker: APICircuitBreaker,
    cost_fn: Callable = None,
):
    """
    Decorator to apply circuit breaker to function.
    
    Args:
        breaker: APICircuitBreaker instance
        cost_fn: Function to extract cost from result (optional)
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            if not breaker.can_make_call():
                raise CircuitBreakerError(
                    f"{breaker.name} circuit breaker is open"
                )
            
            try:
                result = fn(*args, **kwargs)
                
                # Record success
                cost = cost_fn(result) if cost_fn else 0.0
                breaker.record_call(cost)
                
                return result
                
            except breaker.expected_exception as e:
                breaker.record_failure()
                raise
        
        return wrapper
    return decorator


# Global circuit breakers for common APIs
llm_circuit_breaker = APICircuitBreaker(
    name="LLM API",
    failure_threshold=5,
    recovery_timeout=60,
    max_calls_per_minute=10,
    budget_limit=2.0,
)

kaggle_circuit_breaker = APICircuitBreaker(
    name="Kaggle API",
    failure_threshold=5,
    recovery_timeout=60,
    max_calls_per_minute=5,
    budget_limit=0.0,  # Kaggle API is free
)
