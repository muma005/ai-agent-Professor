# Phase 1: Core Stability - Implementation Plan

**Priority:** CRITICAL  
**Estimated Effort:** 15-20 hours  
**Status:** 📋 READY TO START  

---

## Flaws to Fix (8 Critical)

| # | Flaw ID | Severity | Component | Estimated Time |
|---|---------|----------|-----------|----------------|
| 1 | FLAW-2.1 | 🔴 CRITICAL | Pipeline Checkpointing | 3-4 hours |
| 2 | FLAW-2.2 | 🔴 CRITICAL | API Circuit Breaker | 2-3 hours |
| 3 | FLAW-2.3 | 🔴 CRITICAL | LLM Output Validation | 2-3 hours |
| 4 | FLAW-2.4 | 🔴 CRITICAL | Timeout for Operations | 1-2 hours |
| 5 | FLAW-4.1 | 🔴 CRITICAL | Global Exception Handler | 2-3 hours |
| 6 | FLAW-4.2 | 🟠 HIGH | Error Context Preservation | 2-3 hours |
| 7 | FLAW-4.3 | 🟠 HIGH | Model Training Fallback | 2-3 hours |
| 8 | FLAW-4.4 | 🟠 HIGH | Prediction Validation | 1-2 hours |

---

## Implementation Order

### Step 1: Global Exception Handler (FLAW-4.1)
**File:** `core/professor.py`

**Why First:** All other improvements depend on having a central error handling mechanism.

**Implementation:**
```python
# core/professor.py

class ProfessorPipelineError(Exception):
    """Custom exception for pipeline failures."""
    def __init__(self, message, node=None, state_snapshot=None):
        super().__init__(message)
        self.node = node
        self.state_snapshot = state_snapshot
        self.timestamp = datetime.utcnow().isoformat()

def run_professor(state: ProfessorState) -> ProfessorState:
    """Run the full Professor graph with comprehensive error handling."""
    session_id = state.get("session_id", "unknown")
    
    try:
        # Initialize error context
        error_context = {
            "session_id": session_id,
            "start_time": datetime.utcnow().isoformat(),
            "nodes_completed": [],
            "errors": [],
        }
        
        # Run graph
        graph = get_graph()
        result = graph.invoke(state)
        
        # Record success
        error_context["end_time"] = datetime.utcnow().isoformat()
        error_context["status"] = "success"
        _save_error_context(session_id, error_context)
        
        return result
        
    except Exception as e:
        # Record failure
        error_context["end_time"] = datetime.utcnow().isoformat()
        error_context["status"] = "failed"
        error_context["errors"].append({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat(),
        })
        _save_error_context(session_id, error_context)
        
        # Save checkpoint for recovery
        _save_checkpoint(state, f"outputs/{session_id}/failure_checkpoint.json")
        
        # Re-raise with context
        raise ProfessorPipelineError(
            f"Pipeline failed: {e}",
            node=error_context.get("current_node"),
            state_snapshot=state
        ) from e
```

**Tests:**
- Test pipeline completes successfully
- Test pipeline saves error context on failure
- Test pipeline saves checkpoint on failure
- Test custom exception includes context

---

### Step 2: Pipeline Checkpointing (FLAW-2.1)
**Files:** `core/professor.py`, `core/checkpoint.py` (new)

**Implementation:**
```python
# core/checkpoint.py

import json
import os
from datetime import datetime
from typing import Any, Dict
from core.state import ProfessorState

def _is_serializable(value: Any) -> bool:
    """Check if value is JSON-serializable."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False

def save_checkpoint(state: ProfessorState, path: str, node_name: str = None):
    """
    Save pipeline checkpoint for recovery.
    
    Args:
        state: Current ProfessorState
        path: Path to save checkpoint
        node_name: Name of node that just completed
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # Filter to serializable fields only
    serializable_state = {
        k: v for k, v in state.items()
        if _is_serializable(v)
    }
    
    # Add metadata
    checkpoint = {
        "state": serializable_state,
        "metadata": {
            "node_completed": node_name,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0",
        }
    }
    
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    
    logger.info(f"Checkpoint saved to {path}")

def load_checkpoint(path: str) -> Dict:
    """
    Load pipeline checkpoint for recovery.
    
    Returns:
        Dict with "state" and "metadata" keys
    """
    with open(path) as f:
        return json.load(f)

def get_latest_checkpoint(session_id: str) -> Optional[str]:
    """
    Get path to latest checkpoint for session.
    
    Returns:
        Path to latest checkpoint, or None if no checkpoints exist
    """
    checkpoint_dir = f"outputs/{session_id}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".json")
    ]
    
    if not checkpoints:
        return None
    
    # Return most recent
    checkpoints.sort()
    return os.path.join(checkpoint_dir, checkpoints[-1])
```

**Integration in professor.py:**
```python
# core/professor.py

def run_professor(state: ProfessorState, resume_from: str = None) -> ProfessorState:
    """
    Run the full Professor graph with checkpointing and resume capability.
    
    Args:
        state: Initial ProfessorState
        resume_from: Path to checkpoint to resume from (optional)
    
    Returns:
        Final ProfessorState
    """
    session_id = state.get("session_id", "unknown")
    
    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = load_checkpoint(resume_from)
        state.update(checkpoint["state"])
        state["resumed_from_checkpoint"] = resume_from
    
    # ... rest of function ...
    
    # Save checkpoint after each node (in graph compilation)
    # This is done by adding a wrapper node in the graph
```

**Tests:**
- Test checkpoint saves correctly
- Test checkpoint loads correctly
- Test resume from checkpoint works
- Test latest checkpoint detection works

---

### Step 3: API Circuit Breaker (FLAW-2.2)
**Files:** `tools/llm_client.py`, `tools/circuit_breaker.py` (new)

**Implementation:**
```python
# tools/circuit_breaker.py

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
    
    Prevents cascading failures by:
    1. Tracking failure count
    2. Opening circuit after threshold
    3. Allowing test requests after timeout
    4. Closing circuit on success
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
```

**Integration:**
```python
# tools/llm_client.py

from tools.circuit_breaker import APICircuitBreaker, with_circuit_breaker

# Create global circuit breaker
llm_circuit_breaker = APICircuitBreaker(
    name="LLM API",
    failure_threshold=5,
    recovery_timeout=60,
    max_calls_per_minute=10,
    budget_limit=2.0,
)

@with_circuit_breaker(llm_circuit_breaker)
def call_llm(prompt: str, model: str = "deepseek", ...) -> str:
    # ... existing implementation ...
```

**Tests:**
- Test circuit opens after threshold failures
- Test circuit allows test request after timeout
- Test circuit closes on success
- Test rate limiting works
- Test budget limiting works

---

### Step 4: LLM Output Validation (FLAW-2.3)
**Files:** `tools/llm_client.py`

**Implementation:**
```python
# tools/llm_client.py

import json
import re

class LLMOutputValidationError(Exception):
    """Raised when LLM output validation fails."""
    pass

def validate_llm_output(output: str, expected_type: str = "text") -> bool:
    """
    Validate LLM output before using it.
    
    Args:
        output: Raw LLM output
        expected_type: "text", "json", "code", "list"
    
    Returns:
        True if valid
    
    Raises:
        LLMOutputValidationError if invalid
    """
    if not output or not output.strip():
        raise LLMOutputValidationError("Empty output")
    
    if expected_type == "json":
        try:
            # Try to extract JSON
            start = output.find("{")
            end = output.rfind("}")
            if start == -1 or end == -1:
                raise LLMOutputValidationError("No JSON object found")
            
            json.loads(output[start:end+1])
            return True
            
        except json.JSONDecodeError as e:
            raise LLMOutputValidationError(f"Invalid JSON: {e}")
    
    elif expected_type == "code":
        # Check for common code patterns
        if not any(pattern in output for pattern in ["def ", "import ", "class ", "return "]):
            raise LLMOutputValidationError("No code detected")
        
        # Check for suspicious patterns
        suspicious = ["__import__", "eval(", "exec(", "subprocess", "os.system"]
        for pattern in suspicious:
            if pattern in output:
                raise LLMOutputValidationError(f"Suspicious code pattern: {pattern}")
        
        return True
    
    elif expected_type == "list":
        # Try to extract list
        start = output.find("[")
        end = output.rfind("]")
        if start == -1 or end == -1:
            raise LLMOutputValidationError("No list found")
        
        return True
    
    return True  # text is always valid

def call_llm_validated(
    prompt: str,
    expected_type: str = "text",
    max_retries: int = 3,
    **kwargs
) -> str:
    """
    Call LLM with output validation and retry logic.
    
    Args:
        prompt: Prompt to send
        expected_type: Expected output type
        max_retries: Maximum retry attempts
        **kwargs: Passed to call_llm
    
    Returns:
        Validated LLM output
    
    Raises:
        LLMOutputValidationError if validation fails after retries
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            output = call_llm(prompt, **kwargs)
            validate_llm_output(output, expected_type)
            return output
            
        except LLMOutputValidationError as e:
            last_error = e
            logger.warning(
                f"LLM output validation failed (attempt {attempt+1}/{max_retries}): {e}"
            )
            
            # Retry with validation hint
            prompt = f"{prompt}\n\nNOTE: Output must be valid {expected_type}. Please retry."
    
    raise LLMOutputValidationError(
        f"LLM output validation failed after {max_retries} retries: {last_error}"
    )
```

**Tests:**
- Test JSON validation works
- Test code validation works
- Test list validation works
- Test retry logic works
- Test suspicious code detection works

---

### Step 5: Timeout for Operations (FLAW-2.4)
**Files:** `core/timeout.py` (new)

**Implementation:**
```python
# core/timeout.py

import signal
from contextlib import contextmanager
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Raised when operation times out."""
    pass

@contextmanager
def timeout(seconds: int, operation_name: str = "Operation"):
    """
    Context manager for operation timeout.
    
    Args:
        seconds: Timeout in seconds
        operation_name: Name of operation for error message
    
    Usage:
        with timeout(30, "API call"):
            make_api_call()
    """
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

# For Windows (which doesn't support SIGALRM)
import threading

@contextmanager
def timeout_windows(seconds: int, operation_name: str = "Operation"):
    """
    Timeout context manager for Windows.
    
    Uses threading.Timer instead of signal.
    """
    timer = threading.Timer(seconds, lambda: None)
    
    def timeout_handler():
        raise TimeoutError(f"{operation_name} timed out after {seconds}s")
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    
    try:
        yield
    finally:
        timer.cancel()
```

**Integration:**
```python
# core/professor.py

from core.timeout import timeout, timeout_windows

def run_professor(state: ProfessorState, timeout_seconds: int = 600) -> ProfessorState:
    """
    Run the full Professor graph with timeout.
    
    Args:
        state: Initial ProfessorState
        timeout_seconds: Maximum execution time (default: 10 minutes)
    """
    import sys
    
    TimeoutContext = timeout_windows if sys.platform == "win32" else timeout
    
    with TimeoutContext(timeout_seconds, "Pipeline execution"):
        # ... existing implementation ...
```

**Tests:**
- Test timeout raises TimeoutError
- Test timeout works on Windows
- Test timeout doesn't interfere with normal execution

---

### Step 6: Error Context Preservation (FLAW-4.2)
**Files:** `core/error_context.py` (new)

**Implementation:**
```python
# core/error_context.py

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ErrorContextManager:
    """
    Manages error context for debugging and recovery.
    
    Saves error context to disk for:
    1. Debugging failures
    2. Resuming from failures
    3. Analyzing failure patterns
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.context_path = f"outputs/{session_id}/error_context.json"
        self.context = {
            "session_id": session_id,
            "start_time": None,
            "end_time": None,
            "status": "running",  # running, success, failed
            "nodes_completed": [],
            "errors": [],
            "state_snapshots": [],
        }
    
    def start(self):
        """Mark pipeline start."""
        self.context["start_time"] = datetime.utcnow().isoformat()
        self._save()
    
    def complete_node(self, node_name: str, state_snapshot: Optional[Dict] = None):
        """Mark node completion."""
        self.context["nodes_completed"].append({
            "node": node_name,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        if state_snapshot:
            # Save serializable subset
            snapshot = {k: v for k, v in state_snapshot.items() if self._is_serializable(v)}
            self.context["state_snapshots"].append({
                "node": node_name,
                "state": snapshot,
                "timestamp": datetime.utcnow().isoformat(),
            })
        
        self._save()
    
    def record_error(
        self,
        error: Exception,
        node_name: Optional[str] = None,
        traceback_str: Optional[str] = None,
    ):
        """Record error."""
        self.context["errors"].append({
            "node": node_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback_str,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._save()
    
    def fail(self):
        """Mark pipeline failure."""
        self.context["status"] = "failed"
        self.context["end_time"] = datetime.utcnow().isoformat()
        self._save()
    
    def success(self):
        """Mark pipeline success."""
        self.context["status"] = "success"
        self.context["end_time"] = datetime.utcnow().isoformat()
        self._save()
    
    def _is_serializable(self, value: Any) -> bool:
        """Check if value is JSON-serializable."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def _save(self):
        """Save context to disk."""
        os.makedirs(os.path.dirname(self.context_path) or ".", exist_ok=True)
        with open(self.context_path, "w") as f:
            json.dump(self.context, f, indent=2)
    
    def load(self) -> Dict:
        """Load context from disk."""
        if os.path.exists(self.context_path):
            with open(self.context_path) as f:
                return json.load(f)
        return self.context
```

**Integration:**
```python
# core/professor.py

from core.error_context import ErrorContextManager

def run_professor(state: ProfessorState) -> ProfessorState:
    """Run the full Professor graph with error context preservation."""
    session_id = state.get("session_id", "unknown")
    error_context = ErrorContextManager(session_id)
    
    try:
        error_context.start()
        
        # ... run pipeline ...
        
        error_context.success()
        return result
        
    except Exception as e:
        error_context.record_error(e, traceback_str=traceback.format_exc())
        error_context.fail()
        raise
```

**Tests:**
- Test error context saves correctly
- Test error context loads correctly
- Test node completion tracking works
- Test error recording works

---

### Step 7: Model Training Fallback (FLAW-4.3)
**Files:** `agents/ml_optimizer.py`

**Implementation:**
```python
# agents/ml_optimizer.py

def train_with_fallback(X, y, params, model_type, fallback_chain=None):
    """
    Train model with fallback chain.
    
    Args:
        X: Features
        y: Target
        params: Model parameters
        model_type: Primary model type
        fallback_chain: List of fallback model types (default: ["lgbm", "logistic", "dummy"])
    
    Returns:
        (model, model_type_used)
    
    Raises:
        ProfessorModelTrainingError if all models fail
    """
    if fallback_chain is None:
        fallback_chain = ["lgbm", "logistic", "dummy"]
    
    if model_type not in fallback_chain:
        fallback_chain = [model_type] + fallback_chain
    
    last_error = None
    
    for fallback_model in fallback_chain:
        try:
            logger.info(f"Attempting to train {fallback_model}...")
            model = _train_single_model(X, y, params, fallback_model)
            logger.info(f"Successfully trained {fallback_model}")
            return model, fallback_model
            
        except Exception as e:
            last_error = e
            logger.warning(f"{fallback_model} training failed: {e}")
            continue
    
    # All models failed
    raise ProfessorModelTrainingError(
        f"All models failed: {fallback_chain}. Last error: {last_error}"
    )

def _train_single_model(X, y, params, model_type):
    """Train a single model type."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from lightgbm import LGBMClassifier
    
    if model_type == "lgbm":
        model = LGBMClassifier(**params)
        model.fit(X, y)
        return model
    
    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        return model
    
    elif model_type == "dummy":
        model = DummyClassifier(strategy="stratified")
        model.fit(X, y)
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

**Tests:**
- Test fallback chain works
- Test primary model succeeds
- Test fallback to logistic works
- Test fallback to dummy works
- Test all fail raises error

---

### Step 8: Prediction Validation (FLAW-4.4)
**Files:** `agents/ml_optimizer.py`, `tools/prediction_validator.py` (new)

**Implementation:**
```python
# tools/prediction_validator.py

import numpy as np
from typing import Union

class ProfessorPredictionError(Exception):
    """Raised when prediction validation fails."""
    pass

def validate_predictions(
    preds: np.ndarray,
    X_test: np.ndarray,
    task_type: str = "binary",
) -> bool:
    """
    Validate predictions before submission.
    
    Args:
        preds: Predictions
        X_test: Test features (for count validation)
        task_type: "binary", "multiclass", "regression"
    
    Returns:
        True if valid
    
    Raises:
        ProfessorPredictionError if invalid
    """
    # Check count
    if len(preds) != len(X_test):
        raise ProfessorPredictionError(
            f"Prediction count mismatch: {len(preds)} vs {len(X_test)}"
        )
    
    # Check for NaN
    if np.any(np.isnan(preds)):
        nan_count = np.sum(np.isnan(preds))
        raise ProfessorPredictionError(
            f"Predictions contain {nan_count} NaN values"
        )
    
    # Check for Inf
    if np.any(np.isinf(preds)):
        inf_count = np.sum(np.isinf(preds))
        raise ProfessorPredictionError(
            f"Predictions contain {inf_count} Inf values"
        )
    
    # Check range for classification
    if task_type in ["binary", "multiclass"]:
        if np.any(preds < 0) or np.any(preds > 1):
            raise ProfessorPredictionError(
                f"Predictions out of range [0, 1]: "
                f"min={preds.min():.4f}, max={preds.max():.4f}"
            )
    
    # Check variance (detect constant predictions)
    if np.std(preds) < 1e-6:
        raise ProfessorPredictionError(
            "Predictions have no variance (constant predictions)"
        )
    
    return True
```

**Integration:**
```python
# agents/ml_optimizer.py

from tools.prediction_validator import validate_predictions

def run_ml_optimizer(state: ProfessorState) -> ProfessorState:
    # ... existing implementation ...
    
    # Generate predictions
    preds = model.predict_proba(X_test)[:, 1]
    
    # Validate before saving
    validate_predictions(preds, X_test, task_type=task_type)
    
    # ... rest of function ...
```

**Tests:**
- Test count validation works
- Test NaN detection works
- Test Inf detection works
- Test range validation works
- Test variance detection works

---

## Testing Strategy

### Unit Tests
Each component gets unit tests:
- Checkpoint save/load
- Circuit breaker state transitions
- LLM output validation
- Timeout behavior
- Error context preservation
- Fallback chain
- Prediction validation

### Integration Tests
Test components work together:
- Pipeline completes with checkpointing
- Circuit breaker protects against API failures
- Errors are preserved and recoverable
- Fallback models work when primary fails

### Regression Tests
Ensure fixes don't break existing functionality:
- Pipeline still completes successfully
- CV scores unchanged
- Submission format unchanged

---

## Success Criteria

- [ ] All 8 flaws fixed
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] No regression in existing tests
- [ ] Documentation complete
- [ ] Code reviewed

---

**Document Version:** 1.0  
**Created:** 2026-03-25  
**Status:** 📋 READY TO IMPLEMENT
