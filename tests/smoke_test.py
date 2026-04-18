import os
import shutil
import traceback
from pathlib import Path
import polars as pl
import numpy as np

# Set environment encoding to UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"

# Mock directory
TEMP_DIR = Path("tools/smoke_data")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(parents=True)

print("🚀 Generating microscopic mock dataset...")

# Create tiny 10-row dataset
np.random.seed(42)
train_df = pl.DataFrame({
    "id": range(10),
    "feat_1": np.random.randn(10),
    "feat_2": ["A", "B"] * 5,
    "target": [0, 1] * 5
})
test_df = pl.DataFrame({
    "id": range(10, 20),
    "feat_1": np.random.randn(10),
    "feat_2": ["B", "A"] * 5
})

train_df.write_csv(TEMP_DIR / "train.csv")
test_df.write_csv(TEMP_DIR / "test.csv")

# Create dummy sample submission properly
sample_sub_df = pl.DataFrame({
    "id": range(10, 20),
    "target": [0] * 10
})
sample_sub_df.write_csv(TEMP_DIR / "sample_submission.csv")

try:
    print("🚀 Initializing Professor state...")
    from core.state import initial_state
    from core.config import ProfessorConfig
    from core.professor import run_professor

    # Initialize fast config
    config = ProfessorConfig(fast_mode=True)
    # Further neuter it for instant run
    config.ml_optimizer.optuna_trials = 1
    config.ml_optimizer.cv_folds = 2

    # Build state
    state = initial_state(
        competition="smoke-test",
        data_path=str(TEMP_DIR),
        task_type="binary",
        config=config
    )

    print("🚀 Running Professor...")
    final_state = run_professor(state)

    if final_state.get("status") == "failed" or final_state.get("error"):
        print("\n❌ PROFESSOR FAILED BUT DID NOT THROW EXCEPTION:")
        print(f"Error Context: {final_state.get('error')}")
        exit(1)

    print("\n✅ PROFESSOR COMPLETED SUCCESSFULLY!")
    print(f"Submission saved to: {final_state.get('submission_path')}")

except Exception as e:
    print("\n🚨 PROFESSOR CRASHED!")
    print("-" * 50)
    traceback.print_exc()
    print("-" * 50)
finally:
    # Cleanup
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
