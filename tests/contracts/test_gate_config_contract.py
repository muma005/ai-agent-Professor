import pytest
from tools.gate_config import get_gate_config
from agents.feature_factory import run_feature_factory
from core.state import ProfessorState, initial_state
from unittest.mock import patch
import polars as pl
import numpy as np

def test_very_small_data():
    config = get_gate_config(1000)
    assert config["wilcoxon_p"] == 0.10
    assert config["cv_folds"] == 5
    assert config["regime"] == "very_small"

def test_small_data():
    config = get_gate_config(3000)
    assert config["wilcoxon_p"] == 0.10
    assert config["cv_folds"] == 7
    assert config["regime"] == "small"

def test_medium_data():
    config = get_gate_config(20000)
    assert config["wilcoxon_p"] == 0.05
    assert config["cv_folds"] == 5
    assert config["regime"] == "medium"

def test_large_data():
    config = get_gate_config(100000)
    assert config["wilcoxon_p"] == 0.02
    assert config["null_importance_percentile"] == 97
    assert config["cv_folds"] == 5
    assert config["regime"] == "large"

def test_boundary_5000():
    config = get_gate_config(5000)
    assert config["regime"] == "medium"

def test_boundary_50000():
    config = get_gate_config(50000)
    assert config["regime"] == "medium"

def test_boundary_50001():
    config = get_gate_config(50001)
    assert config["regime"] == "large"

def test_all_keys_present():
    keys = ["wilcoxon_p", "null_importance_percentile", "null_importance_shuffles", "cv_folds", "regime"]
    for n in [1000, 3000, 20000, 100000]:
        config = get_gate_config(n)
        for k in keys:
            assert k in config

@patch("agents.feature_factory.run_adaptive_gate", return_value=(["f1"], [{"feature": "f1", "is_beneficial": True}]))
@patch("agents.feature_factory.run_in_sandbox")
@patch("agents.feature_factory.llm_call")
def test_feature_factory_reads_gate_config(mock_llm, mock_sandbox, mock_adaptive_gate, tmp_path):
    mock_llm.return_value = "df=df"
    mock_sandbox.return_value = {"success": True, "entry": {"entry_id": "1", "code": "df=df", "success": True}}
    
    custom_gate_config = {"wilcoxon_p": 0.99, "cv_folds": 2}
    state_dict = initial_state(
        feature_candidates=[{"name": "f1"}],
        session_id="test-gate-config",
        pipeline_depth="sprint"
    )
    state = ProfessorState(**state_dict)
    state.gate_config = custom_gate_config
    
    run_feature_factory(state)
    
    # Verify the state passed to adaptive_gate has the custom gate_config
    passed_state = mock_adaptive_gate.call_args[0][0]
    assert passed_state.gate_config == custom_gate_config
