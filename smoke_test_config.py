# Minimal Smoke Test Configuration for Professor Pipeline
# Run this to identify ALL bugs across all agents in < 2 minutes

import os
import sys
import time
from datetime import datetime

# ── Configuration: Minimal settings for fast execution ───────────────────────

# Use a tiny synthetic dataset (created on-the-fly)
CREATE_SYNTHETIC_DATA = True
SYNTHETIC_ROWS = 100
SYNTHETIC_FEATURES = 5
SYNTHETIC_TARGET = "target"

# Optuna: absolute minimum trials
N_OPTUNA_TRIALS = 1

# CV: minimum folds
N_CV_FOLDS = 2

# Feature Factory: skip expensive rounds
FEATURE_FACTORY_ROUND_1 = True   # Generic transforms only
FEATURE_FACTORY_ROUND_2 = False  # Skip LLM domain features
FEATURE_FACTORY_ROUND_3 = False  # Skip aggregation
FEATURE_FACTORY_ROUND_4 = False  # Skip target encoding
FEATURE_FACTORY_ROUND_5 = False  # Skip interactions

# Critic: run minimal vectors
CRITIC_VECTOR_1 = True   # Shuffled target
CRITIC_VECTOR_2 = False  # ID-only model
CRITIC_VECTOR_3 = False  # Adversarial
CRITIC_VECTOR_4 = False  # Preprocessing leakage

# Disable expensive features
PSEUDO_LABELING = False  # Known broken
ENSEMBLE = False         # Not in pipeline
EXTERNAL_DATA = False    # Skip API calls

# Budget: tiny
BUDGET_USD = 0.10

# Timeout: hard fail after 2 minutes
PIPELINE_TIMEOUT_SECONDS = 120

# ── Expected Bugs to Surface ──────────────────────────────────────────────────

EXPECTED_BUGS = [
    # pseudo_label_agent
    "pseudo_label_agent: NameError (X_train, X_test, y_train, metric undefined)",
    "pseudo_label_agent: Missing import (is_significantly_better)",
    "pseudo_label_agent: State contract (feature_data_path not set)",
    
    # ensemble_architect
    "ensemble_architect: Not added to LangGraph pipeline",
    
    # submission_strategist
    "submission_strategist: File is empty",
    
    # Potential ml_optimizer bugs
    "ml_optimizer: May not set feature_data_path",
    "ml_optimizer: May not set feature_order",
    
    # Potential competition_intel bugs
    "competition_intel: May fail without Kaggle API credentials",
    
    # Potential data_engineer bugs
    "data_engineer: May fail if sample_submission.csv not found",
]

# ── Test Script ───────────────────────────────────────────────────────────────

TEST_SCRIPT = """
import os
import sys
import time
import polars as pl
import numpy as np
from datetime import datetime

# Set minimal config BEFORE importing professor
os.environ["N_OPTUNA_TRIALS"] = "1"
os.environ["N_CV_FOLDS"] = "2"
os.environ["PROFESSOR_BUDGET_USD"] = "0.10"

# Disable LangSmith tracing (saves cost, speeds up)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Create synthetic data
print("[SmokeTest] Creating synthetic dataset...")
os.makedirs("data", exist_ok=True)

np.random.seed(42)
n_rows = 100
n_features = 5

X = np.random.randn(n_rows, n_features)
y = (X[:, 0] + X[:, 1] + np.random.randn(n_rows) * 0.5 > 0).astype(int)

# Create train.csv
train_df = pl.DataFrame({
    f"feature_{i}": X[:, i] for i in range(n_features)
})
train_df = train_df.with_columns(pl.Series("target", y))
train_df.write_csv("data/train.csv")

# Create test.csv (same schema, no target)
test_df = pl.DataFrame({
    f"feature_{i}": X[:50, i] for i in range(n_features)
})
test_df.write_csv("data/test.csv")

# Create sample_submission.csv
sample_df = pl.DataFrame({
    "id": list(range(50)),
    "target": [0] * 50
})
sample_df.write_csv("data/sample_submission.csv")

print(f"[SmokeTest] Created synthetic data: {n_rows} rows, {n_features} features")

# Run the pipeline
print("[SmokeTest] Starting Professor pipeline...")
start_time = time.time()

try:
    from core.state import initial_state
    from core.professor import run_professor
    
    # Create initial state
    state = initial_state(
        competition="smoke_test",
        data_path="data/train.csv",
        budget_usd=0.10,
        task_type="tabular"
    )
    
    print(f"[SmokeTest] Session ID: {state['session_id']}")
    print(f"[SmokeTest] Starting pipeline at {datetime.now().isoformat()}")
    
    # Run pipeline with timeout
    result = run_professor(state)
    
    elapsed = time.time() - start_time
    
    print(f"[SmokeTest] Pipeline completed in {elapsed:.1f} seconds")
    print(f"[SmokeTest] Final state keys: {list(result.keys())}")
    
    # Check for expected outputs
    print("\\n[SmokeTest] Checking outputs...")
    print(f"  submission_path: {result.get('submission_path', 'NOT SET')}")
    print(f"  model_registry: {len(result.get('model_registry', []))} models")
    print(f"  cv_mean: {result.get('cv_mean')}")
    print(f"  critic_severity: {result.get('critic_severity')}")
    
    if result.get("pipeline_halted"):
        print(f"\\n[SmokeTest] WARNING: Pipeline halted - {result.get('pipeline_halt_reason')}")
    
    if result.get("hitl_required"):
        print(f"\\n[SmokeTest] WARNING: HITL required - {result.get('hitl_reason')}")
    
    # Check for errors in lineage
    from core.lineage import read_lineage
    lineage = read_lineage(state["session_id"])
    errors = [e for e in lineage if "error" in str(e).lower()]
    if errors:
        print(f"\\n[SmokeTest] Found {len(errors)} error events in lineage")
        for e in errors[:5]:
            print(f"  - {e}")
    
    print(f"\\n[SmokeTest] {'='*60}")
    print(f"[SmokeTest] SMOKE TEST {'PASSED' if elapsed < 120 else 'FAILED (timeout)'}")
    print(f"[SmokeTest] {'='*60}")
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\\n[SmokeTest] {'='*60}")
    print(f"[SmokeTest] SMOKE TEST FAILED after {elapsed:.1f} seconds")
    print(f"[SmokeTest] {'='*60}")
    print(f"\\nError type: {type(e).__name__}")
    print(f"Error message: {e}")
    
    import traceback
    print(f"\\nFull traceback:")
    traceback.print_exc()
    
    sys.exit(1)
"""

# ── Instructions ──────────────────────────────────────────────────────────────

INSTRUCTIONS = """
# How to Run the Smoke Test

## Step 1: Save the test script

The test script is embedded above in the TEST_SCRIPT variable.
Save it as `run_smoke_test.py` in the project root.

## Step 2: Run the test

```bash
cd c:\\Users\\ADMIN\\Desktop\\Professor\\ai-agent-Professor
python run_smoke_test.py
```

## Step 3: Expected Duration

- **Should complete in:** 60-90 seconds
- **Should timeout at:** 120 seconds
- **Expected outcome:** Multiple errors will surface

## Step 4: What This Will Find

### Bugs that WILL surface:
1. **pseudo_label_agent** - NameError on undefined variables (if agent is in pipeline)
2. **ml_optimizer** - Any Optuna configuration issues, missing state writes
3. **data_engineer** - File I/O issues, schema detection bugs
4. **eda_agent** - Any crashes in analysis vectors
5. **validation_architect** - CV strategy detection issues
6. **feature_factory** - Feature generation crashes
7. **red_team_critic** - Any vector that crashes
8. **submit** - Submission validation issues

### Bugs that MAY NOT surface:
- Bugs in disabled features (Round 2-5 features, pseudo-labeling)
- Bugs that only trigger on specific data patterns
- Memory leaks (dataset too small)
- API rate limiting (external calls disabled)

## Step 5: After the Test

Collect all error messages and tracebacks. Each error represents a bug
that needs to be fixed before the pipeline can run successfully.

Common error patterns to look for:
- `NameError` - undefined variables
- `KeyError` - missing state keys
- `FileNotFoundError` - missing files
- `ValueError` - invalid data or configuration
- `AttributeError` - wrong object types
- `ImportError` - missing imports
"""

# ── Generate Test File ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Write the test script
    with open("run_smoke_test.py", "w") as f:
        f.write(TEST_SCRIPT)
    
    print("Smoke test script created: run_smoke_test.py")
    print("\n" + "="*60)
    print(INSTRUCTIONS)
    print("="*60)
    print(f"\nExpected bugs to surface: {len(EXPECTED_BUGS)}")
    for bug in EXPECTED_BUGS:
        print(f"  - {bug}")
