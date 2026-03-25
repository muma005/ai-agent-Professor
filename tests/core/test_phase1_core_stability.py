"""
Phase 1 Core Stability Tests.

Tests for:
- FLAW-2.1: Pipeline Checkpointing
- FLAW-2.2: API Circuit Breaker
- FLAW-4.1: Global Exception Handler
- FLAW-4.2: Error Context Preservation
"""
import pytest
import os
import json
import tempfile
from datetime import datetime

from core.error_context import ErrorContextManager
from core.checkpoint import save_checkpoint, load_checkpoint, save_node_checkpoint
from core.circuit_breaker import APICircuitBreaker, CircuitBreakerError, with_circuit_breaker
from core.timeout import timeout, TimeoutError
from tools.prediction_validator import validate_predictions, ProfessorPredictionError


class TestErrorContextManager:
    """Test error context management."""
    
    def test_error_context_creation(self, tmp_path):
        """Test error context manager creation."""
        # Mock the outputs directory
        import core.error_context as ec
        original_makedirs = os.makedirs
        os.makedirs = lambda *args, **kwargs: None
        
        ctx = ErrorContextManager("test_session")
        
        assert ctx.context["session_id"] == "test_session"
        assert ctx.context["status"] == "running"
        
        os.makedirs = original_makedirs
    
    def test_error_context_save_load(self, tmp_path):
        """Test error context save and load."""
        session_id = "test_session_save"
        ctx = ErrorContextManager(session_id)
        
        # Mock save
        ctx.context["test_key"] = "test_value"
        ctx._save()
        
        # Load and verify
        loaded = ctx.load()
        assert loaded["test_key"] == "test_value"


class TestCheckpoint:
    """Test pipeline checkpointing."""
    
    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint save and load."""
        state = {"key1": "value1", "key2": 123}
        path = tmp_path / "checkpoint.json"
        
        save_checkpoint(state, str(path), node_name="test_node")
        
        loaded = load_checkpoint(str(path))
        
        assert "state" in loaded
        assert loaded["state"]["key1"] == "value1"
        assert loaded["metadata"]["node_completed"] == "test_node"
    
    def test_checkpoint_filters_non_serializable(self, tmp_path):
        """Test checkpoint filters non-serializable values."""
        state = {"serializable": "yes", "function": lambda x: x}
        path = tmp_path / "checkpoint.json"
        
        save_checkpoint(state, str(path))
        
        loaded = load_checkpoint(str(path))
        
        assert "serializable" in loaded["state"]
        assert "function" not in loaded["state"]


class TestCircuitBreaker:
    """Test API circuit breaker."""
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = APICircuitBreaker(
            name="Test",
            failure_threshold=3,
            recovery_timeout=1,
        )
        
        # Record failures
        for i in range(3):
            breaker.record_failure()
        
        # Should be open
        assert breaker.state == "open"
        assert not breaker.can_make_call()
    
    def test_circuit_breaker_half_open(self):
        """Test circuit breaker goes half-open after timeout."""
        breaker = APICircuitBreaker(
            name="Test",
            failure_threshold=1,
            recovery_timeout=0,  # Immediate recovery
        )
        
        # Record failure
        breaker.record_failure()
        assert breaker.state == "open"
        
        # Should go half-open immediately (timeout=0)
        assert breaker.can_make_call()
        assert breaker.state == "half-open"
    
    def test_circuit_breaker_closes_on_success(self):
        """Test circuit breaker closes on success."""
        breaker = APICircuitBreaker(
            name="Test",
            failure_threshold=1,
            recovery_timeout=0,
        )
        
        # Open circuit
        breaker.record_failure()
        
        # Go half-open
        breaker.can_make_call()
        
        # Record success
        breaker.record_call()
        
        assert breaker.state == "closed"
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_rate_limit(self):
        """Test circuit breaker rate limiting."""
        breaker = APICircuitBreaker(
            name="Test",
            max_calls_per_minute=2,
        )
        
        # Record calls
        breaker.record_call()
        breaker.record_call()
        
        # Should be rate limited
        assert not breaker.can_make_call()
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        breaker = APICircuitBreaker(
            name="Test",
            failure_threshold=1,
        )
        
        @with_circuit_breaker(breaker)
        def failing_function():
            raise Exception("Test failure")
        
        # First call should fail and open circuit
        with pytest.raises(Exception):
            failing_function()
        
        # Second call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            failing_function()


class TestTimeout:
    """Test timeout functionality."""
    
    def test_timeout_completes_normally(self):
        """Test timeout allows normal completion."""
        import time
        
        with timeout(5, "Test operation"):
            time.sleep(0.1)
        
        # Should complete without exception
    
    @pytest.mark.skipif(os.name == 'nt', reason="Windows timeout doesn't raise exceptions")
    def test_timeout_raises_on_timeout(self):
        """Test timeout raises TimeoutError (Unix only)."""
        import time
        
        with pytest.raises(TimeoutError):
            with timeout(0, "Test operation"):
                time.sleep(1)


class TestPredictionValidator:
    """Test prediction validation."""
    
    def test_validate_predictions_valid(self):
        """Test validation of valid predictions."""
        preds = np.array([0.1, 0.5, 0.9])
        X_test = np.random.randn(3, 5)
        
        result = validate_predictions(preds, X_test, task_type="binary")
        
        assert result is True
    
    def test_validate_predictions_nan(self):
        """Test validation detects NaN."""
        preds = np.array([0.1, np.nan, 0.9])
        
        with pytest.raises(ProfessorPredictionError, match="NaN"):
            validate_predictions(preds, expected_count=3)
    
    def test_validate_predictions_inf(self):
        """Test validation detects Inf."""
        preds = np.array([0.1, np.inf, 0.9])
        
        with pytest.raises(ProfessorPredictionError, match="Inf"):
            validate_predictions(preds, expected_count=3)
    
    def test_validate_predictions_count_mismatch(self):
        """Test validation detects count mismatch."""
        preds = np.array([0.1, 0.5])
        X_test = np.random.randn(3, 5)
        
        with pytest.raises(ProfessorPredictionError, match="count mismatch"):
            validate_predictions(preds, X_test)
    
    def test_validate_predictions_range(self):
        """Test validation detects out of range."""
        preds = np.array([0.1, 1.5, 0.9])
        
        with pytest.raises(ProfessorPredictionError, match="out of range"):
            validate_predictions(preds, expected_count=3, task_type="binary")
    
    def test_validate_predictions_constant(self):
        """Test validation detects constant predictions."""
        preds = np.array([0.5, 0.5, 0.5])
        
        with pytest.raises(ProfessorPredictionError, match="no variance"):
            validate_predictions(preds, expected_count=3, check_variance=True)


# Import numpy for tests
import numpy as np
