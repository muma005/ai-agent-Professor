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
from simulator.competition_registry import get_competition


def test_fast_mode_config():
    """Test that fast mode config is created correctly"""
    config = ProfessorConfig(fast_mode=True)
    
    assert config.fast_mode == True
    assert config.sandbox.enabled == False
    assert config.feature_factory.skip_llm_rounds == True
    assert config.feature_factory.skip_wilcoxon_gate == True
    assert config.ml_optimizer.optuna_trials == 1
    assert config.agents.skip_competition_intel == True
    assert config.agents.skip_eda == True
    assert config.agents.skip_red_team_critic == True
    
    print("✓ Fast mode config created correctly")


def test_production_mode_config():
    """Test that production mode config is created correctly"""
    config = ProfessorConfig(production_mode=True)
    
    assert config.production_mode == True
    assert config.sandbox.enabled == True
    assert config.ml_optimizer.optuna_trials == 100
    assert config.ml_optimizer.models_to_try == ["lgbm", "xgb", "catboost"]
    
    print("✓ Production mode config created correctly")


def test_env_propagation():
    """Test that config propagates to environment"""
    config = ProfessorConfig(fast_mode=True)
    config.apply_env()
    
    import os
    assert os.getenv("PROFESSOR_FAST_MODE") == "1"
    assert os.getenv("PROFESSOR_SKIP_LLM_ROUNDS") == "1"
    assert os.getenv("PROFESSOR_OPTUNA_TRIALS") == "1"
    assert os.getenv("PROFESSOR_SKIP_SANDBOX") == "1"
    
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


def test_initial_state_with_config():
    """Test that initial_state accepts and uses config"""
    config = ProfessorConfig(fast_mode=True)
    
    state = initial_state(
        competition="test-competition",
        data_path="/tmp/test.csv",
        budget_usd=2.00,
        config=config,
    )
    
    assert state["config"] is not None
    assert state["config"].fast_mode == True
    
    print("✓ initial_state accepts and uses config")


def test_sandbox_allowed_imports():
    """Test that sandbox allows sys and multiprocessing"""
    from tools.e2b_sandbox import ALLOWED_MODULES
    
    assert "sys" in ALLOWED_MODULES
    assert "multiprocessing" in ALLOWED_MODULES
    assert "pickle" in ALLOWED_MODULES
    
    print("✓ Sandbox allows required imports")


def test_sandbox_execute_code_safe():
    """Test that execute_code_safe function exists"""
    from tools.e2b_sandbox import execute_code_safe
    
    # Test with simple code
    test_code = """
import sys
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
"""
    result = execute_code_safe(test_code, timeout=30)
    
    # Should succeed (either via sandbox or direct exec)
    assert "success" in result
    
    print("✓ execute_code_safe function works")


if __name__ == "__main__":
    print("="*60)
    print("Running fast mode configuration tests...")
    print("="*60)
    print()
    
    try:
        test_fast_mode_config()
        test_production_mode_config()
        test_env_propagation()
        test_state_modification()
        test_initial_state_with_config()
        test_sandbox_allowed_imports()
        test_sandbox_execute_code_safe()
        
        print()
        print("="*60)
        print("✅ All tests passed!")
        print("="*60)
        
    except AssertionError as e:
        print()
        print("="*60)
        print(f"❌ Test failed: {e}")
        print("="*60)
        sys.exit(1)
    except Exception as e:
        print()
        print("="*60)
        print(f"❌ Unexpected error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
