# Professor Agent — Root Cause Analysis

## Executive Summary

The Professor autonomous ML agent is **failing to complete benchmark runs** due to fundamental architectural issues, not cosmetic configuration problems.

---

## Problem Symptoms

1. **Benchmark runs crash silently** - No `trial_result.json` files created
2. **Python processes terminate unexpectedly** - Tasks show 0 completed trials
3. **Intermediate files created but pipeline stops** - `features.parquet`, `preprocessor.pkl` exist but no submission

---

## Root Causes (In Order of Severity)

### 1. **Sandbox Import Restrictions Block Core Functionality** 🔴 CRITICAL

**Location:** `tools/e2b_sandbox.py` lines 58-70

```python
BLOCKED_MODULES = {
    "subprocess", "shutil", "socket", "http", "urllib",
    "ftplib", "smtplib", "ctypes", "multiprocessing",
    "signal", "pty", "resource", "sys",  # ← sys is blocked!
}
```

**Impact:**
- FeatureFactory generates code that imports `sys` for path manipulation
- Null importance filtering fails when trying to import blocked modules
- Error message seen: `Import of 'sys' is not allowed in sandbox`

**Why This Breaks Everything:**
- The sandbox is designed for **security** (prevent LLM-generated code from escaping)
- But legitimate ML code needs `sys`, `multiprocessing`, etc.
- **No fallback mechanism** when imports fail

---

### 2. **No Fast Mode / Local Execution Path** 🟡 HIGH

**Location:** `core/professor.py` - entire pipeline

**Current Behavior:**
```python
# Every run executes:
1. CompetitionIntel (LLM calls, forum scraping)
2. DataEngineer (sandbox code execution)
3. EDAAgent (LLM calls for insights)
4. ValidationArchitect (OK - fast)
5. FeatureFactory (LLM rounds 1-5 + Wilcoxon tests)
6. MLOptimizer (Optuna: 100-200 trials × 3 models)
7. RedTeamCritic (LLM calls)
8. Submit
```

**Problem:**
- **No environment variable checks** for skipping stages
- **No "fast mode" flag** that bypasses expensive operations
- Config dict passed to `run_professor()` is **ignored** by individual agents

**Evidence:**
```python
# local_benchmark.py sets:
os.environ["PROFESSOR_SKIP_FEATURE_FACTORY"] = "True"
os.environ["PROFESSOR_OPTUNA_TRIALS"] = "1"

# But core/professor.py never checks these!
```

---

### 3. **Windows Multiprocessing Without Guard** 🟡 HIGH

**Location:** `simulator/local_benchmark.py`

**Problem:**
```python
from multiprocessing import Pool

# Missing:
if __name__ == "__main__":
    # Pool creation must be inside this guard on Windows
```

**Impact:**
- On Windows, child processes re-import the module
- Without `if __name__ == "__main__"`, this creates infinite process spawning
- Causes silent crashes

---

### 4. **Optuna Default Configuration Too Expensive** 🟢 MEDIUM

**Location:** `agents/ml_optimizer.py`

**Default Config:**
```python
optuna_trials = 100  # Per model type
n_models = 3         # LGBM, XGB, CatBoost
total_trials = 300   # Each trial = 5-fold CV = 1500 model fits!
```

**Time Cost:**
- Each trial: ~30-60 seconds (5-fold CV)
- Total time: 300 × 45s = **3.75 hours per trial**

---

### 5. **FeatureFactory LLM Rounds** 🟢 MEDIUM

**Location:** `agents/feature_factory.py`

**Current Flow:**
```
Round 1: Generic transforms (log, sqrt, missing flags) - FAST
Round 2: LLM-generated domain features - SLOW (API calls)
Round 3: Aggregation features - MEDIUM
Round 4: Target encoding - MEDIUM
Round 5: Hypothesis + interactions - SLOW (LLM)
Wilcoxon Gate: Statistical tests for each feature - SLOW
Null Importance: 50 shuffle iterations - SLOW
```

**Time Cost:** 5-10 minutes per trial (even without Optuna)

---

## Why "Cosmetic Fixes" Don't Work

### Attempted Fix #1: Environment Variables
```python
os.environ["PROFESSOR_SKIP_FEATURE_FACTORY"] = "1"
```
**Why It Fails:** `core/professor.py` and `agents/*.py` never check these variables

### Attempted Fix #2: Config Dict
```python
config = {"optuna_trials": 1, "skip_feature_factory": True}
run_professor(state, config)
```
**Why It Fails:** `run_professor()` doesn't propagate config to agents

### Attempted Fix #3: Multiprocessing Pool
```python
with Pool(processes=4) as pool:
    pool.map(run_trial, trials)
```
**Why It Fails:** Windows requires `if __name__ == "__main__"` guard

---

## Comprehensive Solution

### Phase 1: Immediate Fixes (1-2 hours)

#### 1.1 Add `if __name__ == "__main__"` Guard
**File:** `simulator/local_benchmark.py`, `simulator/simple_benchmark.py`

```python
def main():
    # ... existing code ...

if __name__ == "__main__":
    main()
```

#### 1.2 Fix Sandbox Blocked Modules
**File:** `tools/e2b_sandbox.py`

```python
# Move sys from BLOCKED to ALLOWED (it's needed for path manipulation)
ALLOWED_MODULES = {
    "polars", "numpy", "json", "os", "math", "sys",  # ← Add sys
    "sklearn", "lightgbm", "xgboost", "catboost",
    "optuna", "scipy", "statistics", "itertools",
    "collections", "functools", "datetime", "pathlib",
    "multiprocessing",  # ← Needed for parallel CV
}

BLOCKED_MODULES = {
    "socket", "http", "urllib", "ftplib", "smtplib",
    "ctypes", "signal", "pty", "resource",  # ← Remove sys, multiprocessing
}
```

#### 1.3 Add Fast Mode to Professor Pipeline
**File:** `core/professor.py`

```python
def run_professor(state, timeout_seconds=None, fast_mode=False):
    """
    Args:
        fast_mode: If True, skip expensive operations:
            - Skip CompetitionIntel (no forum scraping)
            - Skip EDAAgent (no LLM insights)
            - Skip FeatureFactory Rounds 2,5 (no LLM)
            - Skip Wilcoxon gate
            - Use 1 Optuna trial (defaults only)
            - Skip RedTeamCritic
    """
    if fast_mode:
        state["dag"] = [
            "data_engineer",
            "validation_architect",
            "feature_factory",  # Only rounds 1,3,4
            "ml_optimizer",     # 1 trial
            "submit",
        ]
        os.environ["PROFESSOR_FAST_MODE"] = "1"
```

#### 1.4 Make Agents Check Fast Mode Flag
**File:** `agents/competition_intel.py`, `agents/eda_agent.py`, etc.

```python
def run_competition_intel(state):
    if os.getenv("PROFESSOR_FAST_MODE") == "1":
        print("[CompetitionIntel] Skipping (fast mode)")
        return state  # Skip entirely

    # ... existing code ...
```

---

### Phase 2: Architectural Improvements (1 day)

#### 2.1 Add Configuration System
**File:** `core/config.py` (new)

```python
@dataclass
class ProfessorConfig:
    # Execution mode
    fast_mode: bool = False
    triage_mode: bool = False

    # Feature generation
    skip_feature_factory: bool = False
    skip_llm_rounds: bool = False
    skip_wilcoxon_gate: bool = False
    skip_null_importance: bool = False

    # Model optimization
    optuna_trials: int = 100
    models_to_try: list = field(default_factory=lambda: ["lgbm", "xgb", "catboost"])

    # Sandbox
    use_sandbox: bool = True
    sandbox_timeout: int = 600

    def apply_to_state(self, state):
        """Apply config to ProfessorState"""
        state["config"] = self
        if self.fast_mode:
            state["dag"] = self._get_fast_dag()
        return state
```

#### 2.2 Refactor Agents to Accept Config
**File:** `agents/*.py`

```python
def run_feature_factory(state: ProfessorState, config: ProfessorConfig):
    if config.skip_feature_factory:
        return state

    if config.skip_llm_rounds:
        # Only run rounds 1, 3, 4 (no LLM)
        return _run_rounds_1_3_4(state)

    if config.skip_wilcoxon_gate:
        # Skip statistical testing
        return _skip_wilcoxon(state)

    # ... existing code ...
```

---

### Phase 3: Benchmark-Specific Optimizations

#### 3.1 Use `simple_benchmark.py` for Local Testing
**File:** `simulator/simple_benchmark.py` (already created)

This bypasses the Professor pipeline entirely:
- Direct LightGBM with defaults
- No LLM calls
- No sandbox execution
- No Optuna
- **Completes 20 trials in ~10 minutes**

#### 3.2 Hybrid Approach for Production
**File:** `simulator/local_benchmark.py`

```python
def run_single_trial(entry, trial_num, mode="fast"):
    if mode == "fast":
        # Use simple_benchmark logic
        return run_simple_lightgbm_trial(entry, trial_num)
    else:
        # Use full Professor pipeline
        return run_full_professor_trial(entry, trial_num)
```

---

## Recommended Action Plan

### Today (Immediate):
1. ✅ Use `simple_benchmark.py` for 20 trials (already running)
2. Fix `BLOCKED_MODULES` in `tools/e2b_sandbox.py`
3. Add `if __name__ == "__main__"` to all multiprocessing scripts

### This Week:
1. Add `fast_mode` parameter to `run_professor()`
2. Make agents check `PROFESSOR_FAST_MODE` env var
3. Reduce default Optuna trials to 30 for local runs

### Next Week:
1. Implement `ProfessorConfig` dataclass
2. Refactor all agents to accept config
3. Add proper checkpointing for long runs

---

## Verification Checklist

After fixes, verify:
- [ ] 20 trials complete in <30 minutes
- [ ] All `trial_result.json` files created
- [ ] No sandbox import errors
- [ ] CPU/memory usage reasonable (<8GB RAM)
- [ ] Results saved to `simulator/results/`

---

## Conclusion

The core issue is **architectural rigidity** - the pipeline was designed for production (full LLM calls, Optuna, sandbox security) without a fast local execution path.

**Immediate fix:** Use `simple_benchmark.py` which bypasses the Professor pipeline.

**Long-term fix:** Add `fast_mode` configuration that propagates through all agents.
