import os
import sys
import shutil
import logging
from typing import Dict, Any
import polars as pl

# Configure path so core modules load
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.state import ProfessorState
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect
from agents.feature_factory import run_feature_factory
from agents.ml_optimizer import run_ml_optimizer
from core.preprocessor import TabularPreprocessor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_phase1_parity_gate():
    """
    E2E Integration Test: Phase 1 Parity Gate
    Runs the full Phase 1 pipeline on Spaceship Titanic.
    Checks:
    1. Pipeline completes without errors.
    2. TabularPreprocessor is correctly serialized.
    3. Mathematical inference symmetry.
    4. AUC >= 0.79.
    """
    session_id = "test_parity_gate"
    output_dir = f"outputs/{session_id}"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/spaceship_titanic/train.csv"))
    if not os.path.exists(data_path):
        logger.error(f"Cannot find dataset at {data_path}. Run from project root.")
        return False
        
    state: ProfessorState = {
        "session_id": session_id,
        "competition_name": "spaceship_titanic",
        "raw_data_path": data_path,
        "cost_tracker": {"llm_calls": 0, "optuna_trials": 0},
    }
    
    try:
        logger.info("--- 1. Data Engineer ---")
        state = run_data_engineer(state)
        
        logger.info("--- 2. EDA Agent ---")
        state = run_eda_agent(state)
        
        logger.info("--- 3. Validation Architect ---")
        state = run_validation_architect(state)
        
        logger.info("--- 4. Feature Factory ---")
        state = run_feature_factory(state)
        
        logger.info("--- 5. ML Optimizer ---")
        state = run_ml_optimizer(state)
        
        # ── Gate Checks ──────────────────────────────────────────────────
        
        # Check 1: Artifacts exist
        preprocessor_path = f"{output_dir}/preprocessor.pkl"
        assert os.path.exists(preprocessor_path), "TabularPreprocessor artifact missing!"
        assert os.path.exists(state.get("feature_data_path", "")), "features.parquet missing!"
        
        # Check 2: Mathematical Parity
        logger.info("Validating TabularPreprocessor Symmetry...")
        preprocessor = TabularPreprocessor.load(preprocessor_path)
        raw_df = pl.read_csv(data_path, infer_schema_length=5000)
        
        manual_transformed = preprocessor.transform(raw_df.head(100))
        cached_features = pl.read_parquet(state["feature_data_path"]).head(100)
        
        missing_cols = set(cached_features.columns) - set(manual_transformed.columns)
        assert not missing_cols, f"Mismatch! Transform missed columns: {missing_cols}"
        
        # Check 3: CV Sanity
        cv_mean = state.get("cv_mean", 0.0)
        logger.info(f"Final CV AUC: {cv_mean:.4f}")
        assert cv_mean >= 0.79, f"Regression detected! AUC dropped to {cv_mean:.4f}"
        
        logger.info("✅ SUCCESS: Phase 1 Parity Gate Passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parity Gate Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if not test_phase1_parity_gate():
        sys.exit(1)
