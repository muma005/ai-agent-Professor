# Professor Project - Code Style Guide

**Version:** 1.0
**Effective Date:** 2026-03-25
**Enforcement:** Pre-commit hooks + CI/CD

---

## Overview

This guide establishes coding standards for the Professor project. All code must adhere to these standards to ensure consistency, maintainability, and quality.

---

## 1. Code Formatting

### 1.1 Black Formatting

All Python code must be formatted with **Black**:

```bash
black --line-length=100 --target-version=py311 .
```

**Key rules:**
- Line length: 100 characters
- Target version: Python 3.11
- Single quotes preferred (Black default)
- Trailing commas in multi-line structures

**Example:**
```python
# ✅ Good
def process_data(
    df: pl.DataFrame,
    target_col: str,
    features: list[str],
) -> pl.DataFrame:
    return df.select(features + [target_col])

# ❌ Bad
def process_data(df: pl.DataFrame, target_col: str, features: list[str]) -> pl.DataFrame:
    return df.select(features + [target_col])
```

---

### 1.2 Import Sorting (isort)

Imports must be sorted with **isort**:

```bash
isort --profile=black --line-length=100 .
```

**Order:**
1. Standard library
2. Third-party
3. Local imports

**Example:**
```python
# ✅ Good
import os
import sys
from typing import Any

import numpy as np
import polars as pl

from core.state import ProfessorState
from tools.data_tools import read_parquet

# ❌ Bad (mixed order)
import polars as pl
from core.state import ProfessorState
import os
import numpy as np
```

---

## 2. Type Hints

### 2.1 Function Signatures

All public functions must have type hints:

```python
# ✅ Good
def calculate_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> float:
    ...

# ❌ Bad (no types)
def calculate_cv_score(X, y, n_folds=5):
    ...
```

### 2.2 Complex Types

Use descriptive types for complex structures:

```python
# ✅ Good
from typing import Dict, List, Optional, Union

def process_features(
    features: Dict[str, List[float]],
    target: Optional[np.ndarray] = None,
) -> Union[np.ndarray, pl.DataFrame]:
    ...

# ❌ Bad (vague types)
def process_features(features: dict, target=None):
    ...
```

### 2.3 Type Aliases

Use type aliases for complex types:

```python
# ✅ Good
FeatureMatrix = np.ndarray
TargetVector = np.ndarray

def train_model(X: FeatureMatrix, y: TargetVector) -> Any:
    ...

# ❌ Bad (repeated complex types)
def train_model(X: np.ndarray, y: np.ndarray) -> Any:
    ...
```

---

## 3. Documentation

### 3.1 Docstrings

All public classes and functions must have docstrings:

```python
# ✅ Good
def run_ml_optimizer(state: ProfessorState) -> ProfessorState:
    """
    Run ML optimizer with Optuna HPO and calibration.

    Args:
        state: Professor state with feature_data_path

    Returns:
        Updated state with model_registry, cv_scores

    Raises:
        ValueError: If feature_data_path missing
    """
    ...

# ❌ Bad (no docstring)
def run_ml_optimizer(state):
    ...
```

### 3.2 Module Docstrings

All modules should have a module-level docstring:

```python
# ✅ Good
"""
ML Optimizer agent with Optuna HPO.

Handles model training, hyperparameter optimization, and calibration.
"""

import optuna
...

# ❌ Bad (no module docstring)
import optuna
...
```

---

## 4. Naming Conventions

### 4.1 Variables and Functions

- Use `snake_case` for variables and functions
- Use `UPPER_CASE` for constants
- Use `PascalCase` for classes

```python
# ✅ Good
max_memory_gb = 6.0
DEFAULT_SEED = 42

def calculate_score():
    ...

class ModelOptimizer:
    ...

# ❌ Bad
maxMemoryGB = 6.0
default_seed = 42

def CalculateScore():
    ...
```

### 4.2 Private Members

Use single underscore for private members:

```python
# ✅ Good
class DataProcessor:
    def _validate_data(self):
        """Internal validation."""
        ...

# ❌ Bad (double underscore without reason)
class DataProcessor:
    def __validate_data(self):
        ...
```

---

## 5. Error Handling

### 5.1 Specific Exceptions

Catch specific exceptions:

```python
# ✅ Good
try:
    df = pl.read_parquet(path)
except FileNotFoundError:
    logger.error(f"File not found: {path}")
    raise

# ❌ Bad (bare except)
try:
    df = pl.read_parquet(path)
except:
    logger.error("Error reading file")
```

### 5.2 Custom Exceptions

Use custom exceptions for domain-specific errors:

```python
# ✅ Good
class ProfessorPipelineError(Exception):
    """Custom exception for pipeline failures."""

def run_pipeline():
    if not state.get("feature_data_path"):
        raise ProfessorPipelineError("feature_data_path missing")

# ❌ Bad (generic exception)
def run_pipeline():
    if not state.get("feature_data_path"):
        raise Exception("Error")
```

---

## 6. Testing

### 6.1 Test Structure

Tests must follow pytest conventions:

```python
# ✅ Good
class TestModelComparison:
    """Test model comparison framework."""

    def test_compare_different_models(self):
        """Test comparing two different models."""
        ...

    def test_compare_identical_models(self):
        """Test comparing identical models."""
        ...

# ❌ Bad (not following conventions)
def test_model_comparison():
    ...
```

### 6.2 Test Coverage

Minimum 80% coverage required for new code:

```bash
coverage run -m pytest tests/
coverage report --fail-under=80
```

---

## 7. Security

### 7.1 API Keys

Never hardcode API keys:

```python
# ✅ Good
import os
api_key = os.environ.get("API_KEY")

# ❌ Bad (hardcoded key)
api_key = "sk-1234567890abcdef"
```

### 7.2 Dangerous Functions

Avoid dangerous functions:

```python
# ✅ Good
result = safe_eval(expression, allowed_modules)

# ❌ Bad (security risk)
result = eval(expression)
```

---

## 8. Performance

### 8.1 Memory Efficiency

Use generators for large datasets:

```python
# ✅ Good
def process_batches(data: Iterator) -> Iterator:
    for batch in data:
        yield process_batch(batch)

# ❌ Bad (loads all into memory)
def process_all(data: list) -> list:
    return [process_batch(batch) for batch in data]
```

### 8.2 Caching

Cache expensive operations:

```python
# ✅ Good
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_feature_hash(feature_name: str) -> str:
    ...

# ❌ Bad (recomputes every time)
def compute_feature_hash(feature_name: str) -> str:
    ...
```

---

## 9. Pre-commit Hooks

All developers must install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

**Hooks enforced:**
- Black (formatting)
- isort (imports)
- flake8 (style)
- pylint (quality)
- mypy (types)
- bandit (security)
- pytest (tests)

---

## 10. CI/CD Integration

All PRs must pass:
- ✅ All pre-commit hooks
- ✅ All tests (pytest)
- ✅ Coverage > 80%
- ✅ No security issues (bandit)
- ✅ No type errors (mypy)

---

## Enforcement

**Automatic:**
- Pre-commit hooks block commits
- CI/CD blocks merges

**Manual:**
- Code review checks style
- Tech debt tracked in TODO.md

---

**Questions?** See tech lead or open issue.
