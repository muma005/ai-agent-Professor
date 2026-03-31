# Professor Agent — Comprehensive Fix Plan

## Executive Summary

This document provides a **detailed, actionable plan** to fix all identified architectural problems in the Professor autonomous ML agent system. Each section includes:
- Exact file paths to modify
- Specific code changes required
- Dependencies between fixes
- Testing strategy
- Rollback procedures

**Total estimated implementation time:** 6-8 hours
**Risk level:** Medium (changes core pipeline)

---

## Problem Matrix

| ID | Problem | Severity | Component | Fix Complexity |
|----|---------|----------|-----------|----------------|
| P1 | Sandbox blocks `sys` import | 🔴 Critical | `tools/e2b_sandbox.py` | Low |
| P2 | No fast mode propagation | 🔴 Critical | `core/professor.py` + all agents | High |
| P3 | Windows multiprocessing crash | 🟡 High | `simulator/*.py` | Low |
| P4 | Optuna 300 trials default | 🟡 High | `agents/ml_optimizer.py` | Medium |
| P5 | LLM rounds always execute | 🟡 High | `agents/feature_factory.py` | Medium |
| P6 | No config system | 🟡 High | New: `core/config.py` | Medium |
| P7 | Environment variables ignored | 🟡 High | All agents | Medium |
| P8 | No benchmark progress tracking | 🟢 Medium | `simulator/local_benchmark.py` | Low |

---

## Phase 1: Sandbox Fixes (30 minutes)

### P1.1: Fix Blocked Modules

**File:** `tools/e2b_sandbox.py`

**Current State (Lines 58-70):**
```python
ALLOWED_MODULES = {
    "polars", "numpy", "json", "os", "math",
    "sklearn", "lightgbm", "xgboost", "catboost",
    "optuna", "scipy", "statistics", "itertools",
    "collections", "functools", "datetime", "pathlib"
}

BLOCKED_MODULES = {
    "subprocess", "shutil", "socket", "http", "urllib",
    "ftplib", "smtplib", "ctypes", "multiprocessing",
    "signal", "pty", "resource", "sys",
}
```

**Required Change:**
```python
ALLOWED_MODULES = {
    "polars", "numpy", "json", "os", "math", "sys",  # ← ADDED sys
    "sklearn", "lightgbm", "xgboost", "catboost",
    "optuna", "scipy", "statistics", "itertools",
    "collections", "functools", "datetime", "pathlib",
    "multiprocessing",  # ← ADDED for parallel CV
    "pickle", "re", "typing",  # ← ADDED common ML needs
}

BLOCKED_MODULES = {
    "subprocess", "shutil", "socket", "http", "urllib",
    "ftplib", "smtplib", "ctypes",  # ← REMOVED sys, multiprocessing
    "signal", "pty", "resource",
}
```

**Testing:**
```python
# Test script: tests/test_sandbox_imports.py
from tools.e2b_sandbox import execute_code

test_code = """
import sys
import multiprocessing as mp
print(f"Python version: {sys.version}")
print(f"CPU count: {mp.cpu_count()}")
"""

result = execute_code(test_code, timeout=30)
assert result["success"] == True, f"Sandbox blocked valid imports: {result}"
```

**Rollback:** Revert lines 58-70 to original values

---

### P1.2: Add Sandbox Fallback Mode

**File:** `tools/e2b_sandbox.py`

**Add after line 40:**
```python
# ── Fallback mode: skip sandbox entirely for local development ────
_SKIP_SANDBOX = os.getenv("PROFESSOR_SKIP_SANDBOX", "0") == "1"

if _SKIP_SANDBOX:
    logger.warning("[sandbox] SKIP_SANDBOX=1 — executing code directly (DEV MODE)")
```

**Add new function at end of file:**
```python
def execute_code_safe(code: str, timeout: int = SANDBOX_TIMEOUT_S, **kwargs):
    """
    Execute code with sandbox if available, otherwise execute directly.
    Respects PROFESSOR_SKIP_SANDBOX environment variable.
    """
    if _SKIP_SANDBOX:
        # Direct execution (DEV MODE - not for production)
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Create safe globals
            safe_globals = {
                "__builtins__": __builtins__,
            }
            # Import allowed modules
            for module_name in ALLOWED_MODULES:
                try:
                    module = __import__(module_name)
                    safe_globals[module_name] = module
                except ImportError:
                    pass
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, {})
            
            return {
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "result": None,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            }
    else:
        # Use actual sandbox
        return execute_code(code, timeout=timeout, **kwargs)
```

**Update all callers:** Change `from tools.e2b_sandbox import execute_code` to `from tools.e2b_sandbox import execute_code_safe`

---

## Phase 2: Configuration System (1.5 hours)

### P2.1: Create Config Module

**File:** `core/config.py` (NEW)

```python
# core/config.py
"""
Professor Configuration System — Centralized control for all execution parameters.

Usage:
    config = ProfessorConfig(fast_mode=True)
    state = config.apply_to_state(initial_state)
    result = run_professor(state)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class SandboxConfig:
    """Sandbox execution configuration"""
    enabled: bool = True
    timeout_seconds: int = 600
    skip_import_validation: bool = False  # DEV MODE - use with caution
    
    def apply_env(self):
        os.environ["PROFESSOR_USE_SANDBOX"] = "1" if self.enabled else "0"
        os.environ["PROFESSOR_SANDBOX_TIMEOUT"] = str(self.timeout_seconds)


@dataclass
class FeatureFactoryConfig:
    """Feature generation configuration"""
    enabled: bool = True
    skip_llm_rounds: bool = False  # Skip rounds 2, 5 (LLM-generated)
    skip_wilcoxon_gate: bool = False  # Skip statistical testing
    skip_null_importance: bool = False  # Skip null importance filtering
    max_interaction_features: int = 20
    max_aggregation_features: int = 50
    
    def apply_env(self):
        os.environ["PROFESSOR_SKIP_LLM_ROUNDS"] = "1" if self.skip_llm_rounds else "0"
        os.environ["PROFESSOR_SKIP_WILCOXON"] = "1" if self.skip_wilcoxon_gate else "0"
        os.environ["PROFESSOR_SKIP_NULL_IMPORTANCE"] = "1" if self.skip_null_importance else "0"


@dataclass
class MLOptimizerConfig:
    """Model optimization configuration"""
    optuna_trials: int = 30  # Reduced from 100
    models_to_try: List[str] = field(default_factory=lambda: ["lgbm"])  # Default: 1 model
    cv_folds: int = 5
    timeout_per_trial: int = 300  # 5 minutes
    
    def apply_env(self):
        os.environ["PROFESSOR_OPTUNA_TRIALS"] = str(self.optuna_trials)
        os.environ["PROFESSOR_MODELS"] = ",".join(self.models_to_try)
        os.environ["PROFESSOR_CV_FOLDS"] = str(self.cv_folds)


@dataclass
class AgentSkipConfig:
    """Which agents to skip entirely"""
    skip_competition_intel: bool = False
    skip_eda: bool = False
    skip_red_team_critic: bool = False
    skip_ensemble: bool = False
    skip_pseudo_label: bool = False
    
    def apply_env(self):
        os.environ["PROFESSOR_SKIP_INTEL"] = "1" if self.skip_competition_intel else "0"
        os.environ["PROFESSOR_SKIP_EDA"] = "1" if self.skip_eda else "0"
        os.environ["PROFESSOR_SKIP_CRITIC"] = "1" if self.skip_red_team_critic else "0"
        os.environ["PROFESSOR_SKIP_ENSEMBLE"] = "1" if self.skip_ensemble else "0"


@dataclass
class ProfessorConfig:
    """
    Master configuration for Professor pipeline.
    
    Presets:
        fast_mode=True  → Local development, ~5 min/trial
        production_mode=True → Full execution, ~1 hour/trial
        custom → Mix and match
    """
    # Execution mode presets
    fast_mode: bool = False
    production_mode: bool = False
    
    # Component configs
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    feature_factory: FeatureFactoryConfig = field(default_factory=FeatureFactoryConfig)
    ml_optimizer: MLOptimizerConfig = field(default_factory=MLOptimizerConfig)
    agents: AgentSkipConfig = field(default_factory=AgentSkipConfig)
    
    # Global settings
    seed: int = 42
    log_level: str = "INFO"
    checkpoint_enabled: bool = True
    
    def __post_init__(self):
        """Apply presets after initialization"""
        if self.fast_mode:
            self._apply_fast_mode()
        elif self.production_mode:
            self._apply_production_mode()
    
    def _apply_fast_mode(self):
        """Configure for fast local execution"""
        self.sandbox.enabled = False  # Skip sandbox overhead
        self.sandbox.skip_import_validation = True
        
        self.feature_factory.skip_llm_rounds = True
        self.feature_factory.skip_wilcoxon_gate = True
        self.feature_factory.skip_null_importance = True
        
        self.ml_optimizer.optuna_trials = 1  # Just defaults
        self.ml_optimizer.models_to_try = ["lgbm"]  # Single model
        self.ml_optimizer.cv_folds = 3  # Reduced CV
        
        self.agents.skip_competition_intel = True
        self.agents.skip_eda = True
        self.agents.skip_red_team_critic = True
        self.agents.skip_ensemble = True
    
    def _apply_production_mode(self):
        """Configure for full production execution"""
        self.sandbox.enabled = True
        self.sandbox.timeout_seconds = 600
        
        self.feature_factory.skip_llm_rounds = False
        self.feature_factory.skip_wilcoxon_gate = False
        
        self.ml_optimizer.optuna_trials = 100
        self.ml_optimizer.models_to_try = ["lgbm", "xgb", "catboost"]
        self.ml_optimizer.cv_folds = 5
    
    def apply_env(self):
        """Apply all config to environment variables"""
        os.environ["PROFESSOR_SEED"] = str(self.seed)
        os.environ["PROFESSOR_LOG_LEVEL"] = self.log_level
        os.environ["PROFESSOR_CHECKPOINT"] = "1" if self.checkpoint_enabled else "0"
        
        self.sandbox.apply_env()
        self.feature_factory.apply_env()
        self.ml_optimizer.apply_env()
        self.agents.apply_env()
    
    def apply_to_state(self, state: dict) -> dict:
        """Apply config to ProfessorState"""
        state["config"] = self
        
        # Modify DAG based on config
        if self.fast_mode or self.agents.skip_competition_intel:
            # Remove competition_intel from DAG
            if "dag" in state and "competition_intel" in state["dag"]:
                state["dag"].remove("competition_intel")
        
        if self.fast_mode or self.agents.skip_eda:
            if "dag" in state and "eda_agent" in state["dag"]:
                state["dag"].remove("eda_agent")
        
        if self.fast_mode or self.agents.skip_red_team_critic:
            if "dag" in state and "red_team_critic" in state["dag"]:
                state["dag"].remove("red_team_critic")
        
        return state
    
    def save(self, path: str):
        """Save config to JSON for reproducibility"""
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self)
        config_dict["timestamp"] = str(Path(path).parent / "timestamp.txt")
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_env(cls) -> "ProfessorConfig":
        """Load config from environment variables"""
        config = cls()
        
        # Override from env if set
        if os.getenv("PROFESSOR_FAST_MODE") == "1":
            config.fast_mode = True
        if os.getenv("PROFESSOR_OPTUNA_TRIALS"):
            config.ml_optimizer.optuna_trials = int(os.getenv("PROFESSOR_OPTUNA_TRIALS"))
        if os.getenv("PROFESSOR_SKIP_LLM_ROUNDS") == "1":
            config.feature_factory.skip_llm_rounds = True
        
        return config
```

---

### P2.2: Update State Module

**File:** `core/state.py`

**Add after imports:**
```python
from core.config import ProfessorConfig


def initial_state(
    competition: str,
    data_path: str,
    budget_usd: float = 2.00,
    task_type: str = "auto",
    config: Optional[ProfessorConfig] = None,
) -> dict:
    """
    Create initial Professor state with optional config.
    
    Args:
        config: ProfessorConfig instance. If None, loads from environment.
    """
    if config is None:
        config = ProfessorConfig.from_env()
    
    # Apply config to environment
    config.apply_env()
    
    # ... rest of existing initial_state code ...
    
    # Add config to state
    state["config"] = config
    
    return state
```

---

## Phase 3: Professor Pipeline Updates (2 hours)

### P3.1: Update run_professor Signature

**File:** `core/professor.py`

**Current (line ~400):**
```python
def run_professor(state, timeout_seconds=None):
```

**Change to:**
```python
def run_professor(state, timeout_seconds=None, config: Optional[ProfessorConfig] = None):
    """
    Run the full Professor pipeline.
    
    Args:
        state: ProfessorState dict
        timeout_seconds: Global timeout (default: from config)
        config: ProfessorConfig instance. Overrides state["config"] if provided.
    """
    # Load config from state or parameter
    if config is None:
        config = state.get("config", ProfessorConfig.from_env())
    else:
        state["config"] = config
    
    # Apply config to environment (ensures agents see it)
    config.apply_env()
    
    # Save config for reproducibility
    output_dir = f"outputs/{state['session_id']}"
    os.makedirs(output_dir, exist_ok=True)
    config.save(f"{output_dir}/professor_config.json")
    
    # ... rest of existing code ...
```

---

### P3.2: Add Config Checks to Routing Functions

**File:** `core/professor.py`

**Add to each route function:**
```python
def route_after_router(state: ProfessorState) -> str:
    """After router runs: go to first node in DAG."""
    config = state.get("config", ProfessorConfig.from_env())
    
    if state.get("pipeline_halted") or state.get("triage_mode"):
        print("[Professor] Pipeline halted (circuit breaker). Ending.")
        return END
    
    # Skip nodes based on config
    dag = state.get("dag", [])
    if config.agents.skip_competition_intel:
        dag = [n for n in dag if n != "competition_intel"]
    if config.agents.skip_eda:
        dag = [n for n in dag if n != "eda_agent"]
    
    if not dag:
        print("[Professor] WARNING: DAG is empty after router. Ending.")
        return END

    next_node = dag[0]
    print(f"[Professor] Routing to: {next_node}")
    return next_node
```

---

## Phase 4: Agent Updates (2 hours)

### P4.1: Competition Intel Agent

**File:** `agents/competition_intel.py`

**Add at start of run_competition_intel:**
```python
def run_competition_intel(state: ProfessorState) -> ProfessorState:
    """Scrape competition intelligence from forums."""
    
    # Check config - skip if in fast mode
    config = state.get("config", ProfessorConfig.from_env())
    if config.agents.skip_competition_intel:
        print("[CompetitionIntel] Skipping (fast mode)")
        state["intel_brief_path"] = None
        state["competition_brief_path"] = None
        return state
    
    # ... existing code ...
```

---

### P4.2: EDA Agent

**File:** `agents/eda_agent.py`

**Add at start:**
```python
def run_eda_agent(state: ProfessorState) -> ProfessorState:
    """Generate EDA report."""
    
    config = state.get("config", ProfessorConfig.from_env())
    if config.agents.skip_eda:
        print("[EDAAgent] Skipping (fast mode)")
        state["eda_report_path"] = None
        state["eda_report"] = {}
        return state
    
    # ... existing code ...
```

---

### P4.3: Feature Factory

**File:** `agents/feature_factory.py`

**Add at start of run_feature_factory:**
```python
def run_feature_factory(state: ProfessorState) -> ProfessorState:
    """Generate and validate features."""
    
    config = state.get("config", ProfessorConfig.from_env())
    if not config.feature_factory.enabled:
        print("[FeatureFactory] Skipping entirely (disabled)")
        return state
    
    # ... existing code ...
    
    # Skip LLM rounds if configured
    if config.feature_factory.skip_llm_rounds:
        print("[FeatureFactory] Skipping LLM rounds (2, 5)")
        candidates = _generate_round1_features(schema)
        # Only run rounds 1, 3, 4
        candidates = _apply_round1_transforms(X, candidates)
        candidates = _generate_round3_aggregations(schema, candidates)
        candidates = _generate_round4_target_encodings(schema, candidates)
    else:
        # Full execution with all rounds
        # ... existing code ...
    
    # Skip Wilcoxon gate if configured
    if config.feature_factory.skip_wilcoxon_gate:
        print("[FeatureFactory] Skipping Wilcoxon gate")
        # Keep all candidates that passed null importance
    else:
        # Run Wilcoxon tests
        # ... existing code ...
    
    # Skip null importance if configured
    if config.feature_factory.skip_null_importance:
        print("[FeatureFactory] Skipping null importance filtering")
        # Keep all candidates
    else:
        # Run null importance
        # ... existing code ...
```

---

### P4.4: ML Optimizer

**File:** `agents/ml_optimizer.py`

**Find Optuna study creation and change:**
```python
def run_ml_optimizer(state: ProfessorState) -> ProfessorState:
    """Optimize ML models using Optuna."""
    
    config = state.get("config", ProfessorConfig.from_env())
    
    # Override trials from config
    n_trials = config.ml_optimizer.optuna_trials
    models_to_try = config.ml_optimizer.models_to_try
    cv_folds = config.ml_optimizer.cv_folds
    
    print(f"[MLOptimizer] Running {n_trials} trials for {models_to_try}")
    
    # ... existing code, but use n_trials instead of hardcoded value ...
    
    study = optuna.create_study(direction=direction, study_name=study_name)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
```

---

### P4.5: Red Team Critic

**File:** `agents/red_team_critic.py`

**Add at start:**
```python
def run_red_team_critic(state: ProfessorState) -> ProfessorState:
    """Run critical review of pipeline outputs."""
    
    config = state.get("config", ProfessorConfig.from_env())
    if config.agents.skip_red_team_critic:
        print("[RedTeamCritic] Skipping (fast mode)")
        state["critic_severity"] = "OK"
        state["critic_report"] = {}
        return state
    
    # ... existing code ...
```

---

## Phase 5: Benchmark Infrastructure (1 hour)

### P5.1: Fix local_benchmark.py

**File:** `simulator/local_benchmark.py`

**Add at end of file:**
```python
if __name__ == "__main__":
    main()
```

**Update run_single_trial to use config:**
```python
def run_single_trial(
    entry: CompetitionEntry,
    trial_num: int,
    mode: str = "fast",
    base_output_dir: str = "simulator/results/local_benchmark"
) -> dict:
    """Run Professor against one competition for one trial."""
    
    from core.config import ProfessorConfig
    
    # Create appropriate config
    if mode == "fast":
        config = ProfessorConfig(fast_mode=True)
    else:
        config = ProfessorConfig(production_mode=True)
    
    # ... existing setup code ...
    
    # Build state with config
    from core.state import initial_state
    state = initial_state(
        competition=entry.slug,
        data_path=split.train_path,
        budget_usd=2.00,
        task_type=entry.task_type,
        config=config,  # Pass config!
    )
    
    # Run Professor with config
    from core.professor import run_professor
    result = run_professor(state, timeout_seconds=3000, config=config)
    
    # ... rest of existing code ...
```

---

### P5.2: Create Simple Benchmark (Already Done)

**File:** `simulator/simple_benchmark.py` ✅

This is already created and working - uses direct LightGBM without Professor pipeline.

---

### P5.3: Add Progress Tracking

**File:** `simulator/local_benchmark.py`

**Add progress tracking class:**
```python
class BenchmarkProgress:
    """Track and display benchmark progress"""
    
    def __init__(self, total_trials: int):
        self.total_trials = total_trials
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.results = []
    
    def add_result(self, result: dict):
        self.results.append(result)
        self.completed += 1
        if result.get("error"):
            self.failed += 1
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.completed
        remaining = self.total_trials - self.completed
        eta_minutes = (remaining * avg_time) / 60
        
        # Display progress
        print(f"\n{'='*60}")
        print(f"  PROGRESS: {self.completed}/{self.total_trials} trials")
        print(f"  Success: {self.completed - self.failed} | Failed: {self.failed}")
        print(f"  Avg time: {avg_time:.1f}s/trial")
        print(f"  ETA: {eta_minutes:.1f} minutes")
        print(f"{'='*60}\n")
    
    def summary(self) -> dict:
        elapsed = time.time() - self.start_time
        successful = [r for r in self.results if not r.get("error")]
        
        return {
            "total_trials": self.total_trials,
            "completed": self.completed,
            "failed": self.failed,
            "success_rate": (self.completed - self.failed) / self.completed if self.completed > 0 else 0,
            "total_time_seconds": elapsed,
            "avg_time_per_trial": elapsed / self.completed if self.completed > 0 else 0,
            "results": self.results,
        }
```

---

## Phase 6: Testing & Verification (1 hour)

### P6.1: Create Test Script

**File:** `tests/test_fast_mode.py` (NEW)

```python
#!/usr/bin/env python
"""
Test fast mode configuration end-to-end.

Usage:
    python tests/test_fast_mode.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ProfessorConfig
from core.state import initial_state
from core.professor import run_professor
from simulator.competition_registry import get_competition
from simulator.data_splitter import split_competition_data


def test_fast_mode_config():
    """Test that fast mode config is created correctly"""
    config = ProfessorConfig(fast_mode=True)
    
    assert config.fast_mode == True
    assert config.sandbox.enabled == False
    assert config.feature_factory.skip_llm_rounds == True
    assert config.ml_optimizer.optuna_trials == 1
    assert config.agents.skip_competition_intel == True
    
    print("✓ Fast mode config created correctly")


def test_env_propagation():
    """Test that config propagates to environment"""
    config = ProfessorConfig(fast_mode=True)
    config.apply_env()
    
    import os
    assert os.getenv("PROFESSOR_FAST_MODE") == "1"
    assert os.getenv("PROFESSOR_SKIP_LLM_ROUNDS") == "1"
    assert os.getenv("PROFESSOR_OPTUNA_TRIALS") == "1"
    
    print("✓ Config propagates to environment")


def test_state_modification():
    """Test that config modifies state DAG"""
    config = ProfessorConfig(fast_mode=True)
    
    state = {
        "dag": [
            "competition_intel",
            "data_engineer",
            "eda_agent",
            "validation_architect",
            "feature_factory",
            "ml_optimizer",
            "red_team_critic",
            "submit",
        ]
    }
    
    config.apply_to_state(state)
    
    assert "competition_intel" not in state["dag"]
    assert "eda_agent" not in state["dag"]
    assert "red_team_critic" not in state["dag"]
    
    print("✓ Config modifies state DAG correctly")


def test_full_pipeline_fast_mode():
    """Test full pipeline with fast mode on real data"""
    entry = get_competition("spaceship-titanic")
    
    # Split data
    split = split_competition_data(
        data_path="simulator/data/competitions/spaceship-titanic/full_data.csv",
        entry=entry,
        output_dir="simulator/data/splits/spaceship-titanic",
    )
    
    # Create config
    config = ProfessorConfig(fast_mode=True)
    
    # Create state
    state = initial_state(
        competition=entry.slug,
        data_path=split.train_path,
        budget_usd=2.00,
        task_type=entry.task_type,
        config=config,
    )
    
    # Run pipeline
    result = run_professor(state, timeout_seconds=300, config=config)
    
    # Verify result
    assert result.get("submission_path") is not None, "No submission created"
    assert result.get("cv_mean") is not None, "No CV score"
    
    print(f"✓ Pipeline completed successfully")
    print(f"  CV Score: {result['cv_mean']:.4f}")
    print(f"  Submission: {result['submission_path']}")


if __name__ == "__main__":
    print("Running fast mode tests...\n")
    
    test_fast_mode_config()
    test_env_propagation()
    test_state_modification()
    test_full_pipeline_fast_mode()
    
    print("\n✅ All tests passed!")
```

---

### P6.2: Run Verification Suite

**File:** `tests/verify_fixes.sh` (NEW)

```bash
#!/bin/bash
# Verify all fixes are working

set -e

echo "=== Professor Fix Verification ==="
echo ""

# Test 1: Config system
echo "[1/5] Testing config system..."
python -c "from core.config import ProfessorConfig; c = ProfessorConfig(fast_mode=True); print('✓ Config system works')"

# Test 2: Sandbox imports
echo "[2/5] Testing sandbox imports..."
python -c "from tools.e2b_sandbox import execute_code_safe; print('✓ Sandbox imports work')"

# Test 3: Fast mode propagation
echo "[3/5] Testing fast mode propagation..."
python tests/test_fast_mode.py

# Test 4: Simple benchmark
echo "[4/5] Testing simple benchmark (1 trial)..."
python simulator/simple_benchmark.py --competition spaceship-titanic --trials 1

# Test 5: Local benchmark with fast mode
echo "[5/5] Testing local benchmark (1 trial)..."
python simulator/local_benchmark.py --competition spaceship-titanic --trials 1 --mode fast

echo ""
echo "=== All Verifications Passed ==="
```

---

## Phase 7: Documentation Updates

### P7.1: Update README

**File:** `README.md`

**Add section:**
```markdown
## Fast Mode (Local Development)

For rapid iteration during development, use fast mode:

```bash
# Using simple benchmark (bypasses Professor pipeline)
python simulator/simple_benchmark.py --competition spaceship-titanic --trials 20

# Using Professor pipeline with fast mode
python simulator/local_benchmark.py --competition spaceship-titanic --trials 20 --mode fast

# Using environment variable
export PROFESSOR_FAST_MODE=1
python main.py run --competition spaceship-titanic --data ./data/spaceship-titanic/
```

Fast mode:
- Skips LLM calls (CompetitionIntel, EDA, FeatureFactory rounds 2/5)
- Uses 1 Optuna trial (default parameters only)
- Skips Wilcoxon statistical tests
- Skips RedTeamCritic
- Completes in ~5 minutes per trial vs ~1 hour in production mode
```

---

### P7.2: Update Simulation Documentation

**File:** `Simulation_cloud.md`

**Add section:**
```markdown
## Local Benchmarking

### Simple Benchmark (Recommended for Testing)

Fastest option - bypasses Professor pipeline entirely:

```bash
python simulator/simple_benchmark.py --competition spaceship-titanic --trials 20
```

**Expected runtime:** ~10 minutes for 20 trials
**What it does:** Direct LightGBM with 5-fold CV

### Local Benchmark (Full Pipeline)

Uses full Professor pipeline with fast mode:

```bash
python simulator/local_benchmark.py --competition spaceship-titanic --trials 20 --mode fast
```

**Expected runtime:** ~90 minutes for 20 trials
**What it does:** Full Professor pipeline with fast mode config
```

---

## Rollback Plan

If issues occur after implementing fixes:

### Immediate Rollback (5 minutes)
```bash
# Revert sandbox changes
git checkout tools/e2b_sandbox.py

# Revert config changes
git checkout core/config.py core/state.py core/professor.py

# Revert agent changes
git checkout agents/*.py
```

### Partial Rollback
```bash
# Keep config system, revert agent changes
git checkout agents/*.py

# Agents will fall back to ProfessorConfig.from_env() which reads defaults
```

### Emergency Fallback
```bash
# Use simple_benchmark.py which doesn't depend on any fixes
python simulator/simple_benchmark.py --competition spaceship-titanic --trials 20
```

---

## Success Criteria

After all fixes are implemented, verify:

- [ ] `python tests/test_fast_mode.py` passes all tests
- [ ] `python tests/verify_fixes.sh` completes without errors
- [ ] 20 trials complete in <30 minutes (simple benchmark)
- [ ] 20 trials complete in <2 hours (local benchmark fast mode)
- [ ] All `trial_result.json` files are created
- [ ] No sandbox import errors in logs
- [ ] Memory usage <8GB during execution
- [ ] Results saved to `simulator/results/`

---

## Implementation Checklist

### Phase 1: Sandbox (30 min)
- [ ] P1.1: Fix blocked modules in `tools/e2b_sandbox.py`
- [ ] P1.2: Add fallback mode to sandbox
- [ ] Test: Run `tests/test_sandbox_imports.py`

### Phase 2: Config System (90 min)
- [ ] P2.1: Create `core/config.py`
- [ ] P2.2: Update `core/state.py`
- [ ] Test: Import config, create instance

### Phase 3: Professor Pipeline (120 min)
- [ ] P3.1: Update `run_professor()` signature
- [ ] P3.2: Add config checks to routing
- [ ] Test: Run with fast_mode=True

### Phase 4: Agent Updates (120 min)
- [ ] P4.1: Update CompetitionIntel
- [ ] P4.2: Update EDAAgent
- [ ] P4.3: Update FeatureFactory
- [ ] P4.4: Update MLOptimizer
- [ ] P4.5: Update RedTeamCritic
- [ ] Test: Each agent respects config

### Phase 5: Benchmark (60 min)
- [ ] P5.1: Fix `local_benchmark.py`
- [ ] P5.2: Verify `simple_benchmark.py`
- [ ] P5.3: Add progress tracking
- [ ] Test: Run 5 trials

### Phase 6: Testing (60 min)
- [ ] P6.1: Create test script
- [ ] P6.2: Run verification suite
- [ ] Fix any failing tests

### Phase 7: Documentation
- [ ] P7.1: Update README
- [ ] P7.2: Update simulation docs

---

## Total Time Estimate

| Phase | Time | Dependencies |
|-------|------|--------------|
| 1. Sandbox | 30 min | None |
| 2. Config | 90 min | Phase 1 |
| 3. Pipeline | 120 min | Phase 2 |
| 4. Agents | 120 min | Phase 3 |
| 5. Benchmark | 60 min | Phase 4 |
| 6. Testing | 60 min | Phase 5 |
| 7. Docs | 30 min | All phases |
| **Total** | **~8 hours** | |

---

## Next Steps

1. **Review this plan** - Ensure all changes are understood
2. **Create backup** - `git branch backup-before-fixes`
3. **Implement Phase 1** - Sandbox fixes (lowest risk)
4. **Test Phase 1** - Verify sandbox works
5. **Continue through phases** - One at a time, testing each
6. **Run full verification** - `tests/verify_fixes.sh`
7. **Run 20-trial benchmark** - Final validation
