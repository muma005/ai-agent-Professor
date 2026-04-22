# scripts/smoke_test_v2.py

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.professor import run_professor
from core.state import ProfessorState

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("smoke_test")

def run_v2_smoke_test():
    """
    Minimal E2E run of Professor v2.0 on dummy data.
    """
    logger.info("🚀 Starting Professor v2.0 Smoke Test...")
    
    # 1. Define input paths
    base_data = Path("data/dummy_run")
    train_path = base_data / "train.csv"
    test_path = base_data / "test.csv"
    sample_sub = base_data / "sample_submission.csv"

    # 2. Define minimal DAG
    # (Semantic Router would normally do this, but we force it for the smoke test)
    minimal_dag = [
        "data_engineer",
        "eda_agent",
        "validation_architect",
        "ml_optimizer",
        "submission_strategist"
    ]

    # 3. Initialize State
    initial_state = {
        "session_id": "professor_smoke_test_v2",
        "competition_name": "dummy_competition",
        "raw_data_path": str(train_path),
        "test_data_path": str(test_path),
        "sample_submission_path": str(sample_sub),
        "dag": minimal_dag,
        "current_node": "preflight" # Start at entry
    }

    try:
        # 4. Run Pipeline
        final_state = run_professor(initial_state, timeout_seconds=300)
        
        # 5. Verify Outputs
        logger.info("✅ Pipeline Run Completed.")
        
        # Check for key artifacts
        sub_path = final_state.get("submission_path")
        if sub_path and os.path.exists(sub_path):
            logger.info(f"✨ Success! Submission generated at: {sub_path}")
            print("\n" + "="*50)
            print("SMOKE TEST PASSED")
            print("="*50)
        else:
            logger.error("❌ Failure: Submission file missing.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"💥 Smoke Test Failed with Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_v2_smoke_test()
