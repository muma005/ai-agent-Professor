"""
Phase 1 Core Stability Tests.

Tests for:
- FLAW-2.1: Pipeline Checkpointing
- FLAW-2.2: API Circuit Breaker
- FLAW-2.3: LLM Output Validation
- FLAW-4.1: Global Exception Handler
- FLAW-4.2: Error Context Preservation
- FLAW-4.3: Model Training Fallback
- FLAW-4.4: Prediction Validation
"""
import pytest
import os
import json
import tempfile
import numpy as np
from datetime import datetime

from core.error_context import ErrorContextManager
from core.checkpoint import save_checkpoint, load_checkpoint, save_node_checkpoint
from core.circuit_breaker import APICircuitBreaker, CircuitBreakerError, with_circuit_breaker
from core.timeout import timeout, TimeoutError
from tools.prediction_validator import validate_predictions, ProfessorPredictionError
from tools.llm_client import validate_llm_output, LLMOutputValidationError
from agents.ml_optimizer import train_with_fallback, ProfessorModelTrainingError


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


class TestLLMOutputValidation:
    """Test LLM output validation."""
    
    def test_validate_text_output(self):
        """Test text validation passes."""
        output = "This is valid text output"
        
        result = validate_llm_output(output, expected_type="text")
        
        assert result is True
    
    def test_validate_json_output_valid(self):
        """Test valid JSON validation."""
        output = '{"key": "value", "number": 123}'
        
        result = validate_llm_output(output, expected_type="json")
        
        assert result is True
    
    def test_validate_json_output_invalid(self):
        """Test invalid JSON detection."""
        output = 'This is not JSON'
        
        with pytest.raises(LLMOutputValidationError, match="No JSON object found"):
            validate_llm_output(output, expected_type="json")
    
    def test_validate_code_output_valid(self):
        """Test valid code validation."""
        output = 'def my_function():\n    import os\n    return "hello"'
        
        result = validate_llm_output(output, expected_type="code")
        
        assert result is True
    
    def test_validate_code_output_suspicious(self):
        """Test suspicious code detection."""
        output = '__import__("os").system("ls")'
        
        with pytest.raises(LLMOutputValidationError, match="Suspicious code pattern"):
            validate_llm_output(output, expected_type="code")
    
    def test_validate_list_output_valid(self):
        """Test valid list validation."""
        output = '["item1", "item2", "item3"]'
        
        result = validate_llm_output(output, expected_type="list")
        
        assert result is True
    
    def test_validate_list_output_invalid(self):
        """Test invalid list detection."""
        output = 'This is not a list'
        
        with pytest.raises(LLMOutputValidationError, match="No list found"):
            validate_llm_output(output, expected_type="list")
    
    def test_validate_empty_output(self):
        """Test empty output detection."""
        output = ''
        
        with pytest.raises(LLMOutputValidationError, match="Empty output"):
            validate_llm_output(output)


class TestModelTrainingFallback:
    """Test model training fallback."""
    
    def test_fallback_with_valid_data(self):
        """Test fallback chain works with valid data."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)
        params = {}
        
        model, model_type = train_with_fallback(X, y, params, "lgbm")
        
        assert model is not None
        assert model_type in ["lgbm", "logistic", "dummy"]
    
    def test_fallback_to_logistic(self):
        """Test fallback to logistic regression."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)
        params = {}
        
        # Force fallback by using invalid primary model type
        model, model_type = train_with_fallback(
            X, y, params, 
            primary_model_type="invalid_model",
            fallback_chain=["logistic"]
        )
        
        assert model is not None
        assert model_type == "logistic"
    
    def test_fallback_to_dummy(self):
        """Test fallback to dummy classifier."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)
        params = {}
        
        model, model_type = train_with_fallback(
            X, y, params,
            primary_model_type="invalid",
            fallback_chain=["also_invalid", "dummy"]
        )
        
        assert model is not None
        assert model_type == "dummy"
    
    def test_all_models_fail_raises_error(self):
        """Test all models fail raises error."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)
        params = {}
        
        with pytest.raises(ProfessorModelTrainingError):
            train_with_fallback(
                X, y, params,
                primary_model_type="invalid1",
                fallback_chain=["invalid2"]
            )


# Import numpy for tests
import numpy as np
