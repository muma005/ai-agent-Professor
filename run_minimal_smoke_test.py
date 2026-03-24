# Minimal End-to-End Smoke Test
# Configuration: 1 Optuna trial, minimal data, fast execution

import os
import sys
import time
import polars as pl
import numpy as np
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

# Optuna: MINIMUM trials
os.environ["N_OPTUNA_TRIALS"] = "1"

# CV: minimum folds  
os.environ["N_CV_FOLDS"] = "2"

# Budget: tiny
os.environ["PROFESSOR_BUDGET_USD"] = "0.10"

# Disable LangSmith (saves cost, speeds up)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Disable LangFuse (no keys needed for test)
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

# ── Create Synthetic Data ─────────────────────────────────────────────────────

print("[SmokeTest] Creating MINIMAL synthetic dataset...")
os.makedirs("data", exist_ok=True)

np.random.seed(42)
n_rows = 50  # Even smaller
n_features = 3  # Even smaller

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
    f"feature_{i}": X[:20, i] for i in range(n_features)
})
test_df.write_csv("data/test.csv")

# Create sample_submission.csv
sample_df = pl.DataFrame({
    "id": list(range(20)),
    "target": [0] * 20
})
sample_df.write_csv("data/sample_submission.csv")

print(f"[SmokeTest] Created: train.csv ({n_rows} rows), test.csv (20 rows), sample_submission.csv")

# ── Run Pipeline ──────────────────────────────────────────────────────────────

print("[SmokeTest] Starting Professor pipeline (MINIMAL CONFIG)...")
start_time = time.time()

try:
    from core.state import initial_state
    from core.professor import run_professor
    
    # Create initial state
    state = initial_state(
        competition="smoke_test_minimal",
        data_path="data/train.csv",
        budget_usd=0.10,
        task_type="tabular"
    )
    
    print(f"[SmokeTest] Session ID: {state['session_id']}")
    print(f"[SmokeTest] Starting at {datetime.now().isoformat()}")
    print(f"[SmokeTest] Config: 1 Optuna trial, {n_rows} rows, {n_features} features")
    
    # Run pipeline
    result = run_professor(state)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"[SmokeTest] Pipeline completed in {elapsed:.1f} seconds")
    print(f"{'='*60}")
    print(f"[SmokeTest] Final state keys: {list(result.keys())}")
    
    # Check outputs
    print("\n[SmokeTest] Output checks:")
    print(f"  submission_path: {result.get('submission_path', 'NOT SET')}")
    print(f"  model_registry: {len(result.get('model_registry', []))} models")
    print(f"  cv_mean: {result.get('cv_mean')}")
    print(f"  critic_severity: {result.get('critic_severity')}")
    print(f"  pipeline_halted: {result.get('pipeline_halted')}")
    print(f"  hitl_required: {result.get('hitl_required')}")
    
    # Check lineage for errors
    from core.lineage import read_lineage
    lineage = read_lineage(state["session_id"])
    print(f"\n[SmokeTest] Lineage events: {len(lineage)}")
    
    # Check for errors in lineage
    errors = [e for e in lineage if "error" in str(e).get("action", "").lower() or e.get("agent") == "circuit_breaker"]
    if errors:
        print(f"\n[SmokeTest] WARNING: Found {len(errors)} error events in lineage:")
        for e in errors[:5]:
            print(f"  - {e.get('agent')}: {e.get('action')}")
    
    # Summary
    print(f"\n{'='*60}")
    if elapsed < 120:
        print(f"[SmokeTest] RESULT: ✅ COMPLETED in {elapsed:.1f}s")
    else:
        print(f"[SmokeTest] RESULT: ⚠️ SLOW ({elapsed:.1f}s)")
    
    if result.get("submission_path"):
        print(f"[SmokeTest] RESULT: ✅ SUBMISSION GENERATED")
    else:
        print(f"[SmokeTest] RESULT: ⚠️ NO SUBMISSION")
        
    if result.get("pipeline_halted"):
        print(f"[SmokeTest] RESULT: ⚠️ PIPELINE HALTED - {result.get('pipeline_halt_reason')}")
    else:
        print(f"[SmokeTest] RESULT: ✅ PIPELINE COMPLETED")
        
    print(f"{'='*60}")
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[SmokeTest] RESULT: ❌ FAILED after {elapsed:.1f} seconds")
    print(f"{'='*60}")
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {e}")
    
    import traceback
    print(f"\nFull traceback:")
    traceback.print_exc()
    
    sys.exit(1)
