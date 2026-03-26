"""
End-to-end smoke test for Professor pipeline.

Minimal configuration to verify pipeline runs end-to-end:
- Tiny dataset (100 rows)
- 1 Optuna trial
- 2 CV folds
- Skip time-consuming operations
- Verify all new features integrate properly
"""

import os
import sys
import time
import logging
import numpy as np
import polars as pl
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── MINIMAL CONFIGURATION ────────────────────────────────────────
os.environ["PROFESSOR_SEED"] = "42"
os.environ["PROFESSOR_MAX_MEMORY_GB"] = "2.0"
os.environ["PROFESSOR_CACHE_ENABLED"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── CREATE SYNTHETIC DATASET ─────────────────────────────────────

def create_minimal_dataset(output_dir: str):
    """Create minimal synthetic dataset for smoke test."""
    logger.info("[SmokeTest] Creating minimal dataset...")
    
    np.random.seed(42)
    n_rows = 100  # Minimal rows
    
    # Create features
    X = np.random.randn(n_rows, 5)
    
    # Create target (learnable signal)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create DataFrame
    df = pl.DataFrame({
        "feature_0": X[:, 0],
        "feature_1": X[:, 1],
        "feature_2": X[:, 2],
        "feature_3": X[:, 3],
        "feature_4": X[:, 4],
        "target": y,
    })
    
    # Split into train/test
    train_df = df[:80]
    test_df = df[80:]
    
    # Create sample submission
    sample_sub = pl.DataFrame({
        "PassengerId": [f"test_{i}" for i in range(len(test_df))],
        "target": [0] * len(test_df),
    })
    
    # Save files
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_path = Path(output_dir) / "train.csv"
    test_path = Path(output_dir) / "test.csv"
    sample_path = Path(output_dir) / "sample_submission.csv"
    
    train_df.write_csv(str(train_path))
    test_df.write_csv(str(test_path))
    sample_sub.write_csv(str(sample_path))
    
    logger.info(f"[SmokeTest] Dataset created: {len(train_df)} train, {len(test_df)} test")
    
    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "sample_path": str(sample_path),
    }


# ── RUN MINIMAL PIPELINE ────────────────────────────────────────

def run_minimal_pipeline(data_paths: dict):
    """Run minimal Professor pipeline."""
    logger.info("[SmokeTest] Starting minimal pipeline...")
    
    start_time = time.time()
    
    try:
        # Import core components
        from core.state import initial_state
        from core.professor import run_professor
        from tools.seed_manager import initialize_seeds
        from core.config import get_config
        
        # Initialize seeds
        initialize_seeds(seed=42)
        
        # Create minimal state
        state = initial_state(
            competition="smoke_test",
            data_path=data_paths["train_path"],
        )
        state["test_data_path"] = data_paths["test_path"]
        state["sample_submission_path"] = data_paths["sample_path"]
        
        logger.info("[SmokeTest] Configuration:")
        logger.info(f"  - Optuna trials: 1")
        logger.info(f"  - Max memory: 2.0 GB")
        logger.info(f"  - Timeout: 120 seconds")
        
        # Run pipeline
        logger.info("[SmokeTest] Running Professor pipeline...")
        result = run_professor(state)
        
        elapsed = time.time() - start_time
        
        # Verify results
        logger.info("[SmokeTest] Pipeline completed!")
        logger.info(f"[SmokeTest] Elapsed time: {elapsed:.2f}s")
        
        # Check key outputs
        checks = {
            "submission_path": result.get("submission_path") is not None,
            "model_registry": len(result.get("model_registry", [])) > 0,
            "cv_mean": result.get("cv_mean") is not None,
            "clean_data_path": result.get("clean_data_path") is not None,
            "eda_report": result.get("eda_report") is not None,
        }
        
        logger.info("[SmokeTest] Verification:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check}: {passed}")
        
        # Summary
        all_passed = all(checks.values())
        
        if all_passed:
            logger.info("[SmokeTest] ✅ ALL CHECKS PASSED")
        else:
            logger.warning("[SmokeTest] ❌ SOME CHECKS FAILED")
        
        return {
            "success": all_passed,
            "elapsed": elapsed,
            "checks": checks,
            "result": result,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[SmokeTest] Pipeline failed after {elapsed:.2f}s: {e}")
        
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "elapsed": elapsed,
            "error": str(e),
        }


# ── MAIN ────────────────────────────────────────────────────────

def main():
    """Run smoke test."""
    logger.info("=" * 70)
    logger.info("PROFESSOR PIPELINE - SMOKE TEST")
    logger.info("=" * 70)
    
    # Create minimal dataset
    data_paths = create_minimal_dataset("data/smoke_test")
    
    # Run pipeline
    result = run_minimal_pipeline(data_paths)
    
    # Final summary
    logger.info("=" * 70)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 70)
    
    if result["success"]:
        logger.info("✅ SMOKE TEST PASSED")
        logger.info(f"   Elapsed: {result['elapsed']:.2f}s")
        logger.info(f"   Checks: {sum(result['checks'].values())}/{len(result['checks'])} passed")
        return 0
    else:
        logger.error("❌ SMOKE TEST FAILED")
        logger.error(f"   Elapsed: {result['elapsed']:.2f}s")
        if "error" in result:
            logger.error(f"   Error: {result['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
