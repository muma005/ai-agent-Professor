"""
Tests for API retry with backoff.

FLAW-8.3: API Retry with Exponential Backoff
"""
import pytest
import time
from tools.api_retry import (
    exponential_backoff,
    retry_with_backoff,
    APIRetryError,
    RateLimitHandler,
    get_rate_limiter,
    rate_limited_call,
)


class TestExponentialBackoff:
    """Test exponential_backoff decorator."""

    def test_backoff_success_on_first_try(self):
        """Test function succeeds on first try."""
        call_count = 0
        
        @exponential_backoff(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        
        assert result == "success"
        assert call_count == 1

    def test_backoff_success_after_retries(self):
        """Test function succeeds after retries."""
        call_count = 0
        
        @exponential_backoff(max_retries=3, base_delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = flaky_func()
        
        assert result == "success"
        assert call_count == 3

    def test_backoff_exhausts_retries(self):
        """Test function fails after max retries."""
        call_count = 0
        
        @exponential_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(APIRetryError) as exc_info:
            always_fails()
        
        assert exc_info.value.attempts == 3  # 1 initial + 2 retries
        assert call_count == 3

    def test_backoff_delay_increases(self):
        """Test delay increases with each retry."""
        call_times = []
        
        @exponential_backoff(max_retries=3, base_delay=0.1, max_delay=1.0, jitter=0)
        def slow_fail():
            call_times.append(time.time())
            raise ValueError("Fail")
        
        with pytest.raises(APIRetryError):
            slow_fail()
        
        # Check delays increase exponentially
        assert len(call_times) == 4  # 1 initial + 3 retries
        
        delays = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
        
        # Delays should roughly double (with some tolerance)
        assert delays[1] > delays[0] * 1.5  # Second delay > first * 2
        assert delays[2] > delays[1] * 1.5  # Third delay > second * 2


class TestRetryWithBackoff:
    """Test retry_with_backoff function."""

    def test_retry_success(self):
        """Test retry with success."""
        def success_func():
            return 42
        
        result = retry_with_backoff(success_func)
        
        assert result == 42

    def test_retry_with_args(self):
        """Test retry with arguments."""
        def add(a, b):
            return a + b
        
        result = retry_with_backoff(add, 5, 3)
        
        assert result == 8

    def test_retry_with_kwargs(self):
        """Test retry with keyword arguments."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        result = retry_with_backoff(greet, "World", greeting="Hi")
        
        assert result == "Hi, World!"

    def test_retry_exhausts(self):
        """Test retry exhausts attempts."""
        def always_fail():
            raise RuntimeError("Fail")
        
        with pytest.raises(APIRetryError) as exc_info:
            retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)
        
        assert exc_info.value.attempts == 3


class TestRateLimitHandler:
    """Test RateLimitHandler class."""

    def test_rate_limiter_allows_under_limit(self):
        """Test rate limiter allows requests under limit."""
        limiter = RateLimitHandler(max_requests_per_minute=10)
        
        # Should not wait
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should be immediate

    def test_rate_limiter_waits_at_limit(self):
        """Test rate limiter waits when at limit."""
        limiter = RateLimitHandler(max_requests_per_minute=3)
        
        # Use up the limit
        for _ in range(3):
            limiter.wait_if_needed()
        
        # Next call should wait
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        
        # Should have waited some time
        assert elapsed > 0.01  # At least some wait

    def test_rate_limiter_status(self):
        """Test rate limiter status."""
        limiter = RateLimitHandler(max_requests_per_minute=10, max_requests_per_hour=100)
        
        # Make some requests
        for _ in range(5):
            limiter.wait_if_needed()
        
        status = limiter.get_status()
        
        assert status["requests_last_minute"] == 5
        assert status["limit_per_minute"] == 10
        assert status["minute_utilization"] == 50.0

    def test_rate_limiter_cleanup_old_timestamps(self):
        """Test rate limiter cleans up old timestamps."""
        limiter = RateLimitHandler(max_requests_per_minute=10)
        
        # Add old timestamps manually
        old_time = time.time() - 120  # 2 minutes ago
        limiter.requests_minute.append(old_time)
        
        # Wait should clean up
        limiter.wait_if_needed()
        
        # Old timestamp should be removed
        assert old_time not in limiter.requests_minute


class TestGlobalRateLimiter:
    """Test global rate limiter functions."""

    def test_get_rate_limiter_singleton(self):
        """Test get_rate_limiter returns same instance."""
        # Clear global state
        from tools import api_retry
        api_retry._rate_limiter = None
        
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        assert limiter1 is limiter2

    def test_rate_limited_call(self):
        """Test rate_limited_call function."""
        # Clear global state
        from tools import api_retry
        api_retry._rate_limiter = None
        
        call_count = 0
        
        def test_func():
            nonlocal call_count
            call_count += 1
            return "called"
        
        result = rate_limited_call(test_func)
        
        assert result == "called"
        assert call_count == 1


class TestAPIRetryError:
    """Test APIRetryError exception."""

    def test_api_retry_error_attributes(self):
        """Test APIRetryError has correct attributes."""
        error = ValueError("Test error")
        retry_error = APIRetryError("Failed", last_error=error, attempts=5)
        
        assert str(retry_error) == "Failed"
        assert retry_error.last_error is error
        assert retry_error.attempts == 5

    def test_api_retry_error_from_exception(self):
        """Test APIRetryError raised from another exception."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                retry_error = APIRetryError("Retry failed", last_error=e, attempts=3)
                raise retry_error from e
        except APIRetryError as e:
            # Should catch the APIRetryError
            assert str(e) == "Retry failed"
            assert e.attempts == 3
            assert isinstance(e.last_error, ValueError)
