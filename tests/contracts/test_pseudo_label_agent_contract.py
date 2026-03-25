"""
Contract tests for Pseudo-Label Agent.

FLAW-5.3: Contract Tests for All Agents

Contract: run_pseudo_label_agent
- Input: state with feature_data_path, feature_data_path_test, model_registry
- Output: state with pseudo_label_data_path, pseudo_labels_applied, pseudo_label_cv_improvement
- Invariants:
  - Gracefully skips if required data not found
  - Always writes pseudo_labels_applied (bool)
  - Always writes pseudo_label_cv_improvement (float)
  - Never crashes on missing data
"""
import pytest
import polars as pl
from pathlib import Path
from tools.data_tools import write_parquet, write_json


@pytest.fixture
def pseudo_label_test_data(tmp_path):
    """Create test data for pseudo-label tests."""
    session_dir = tmp_path / "outputs" / "test_pl_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training data
    X_train = pl.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,  # 100 rows
        "feature2": [2.0, 4.0, 6.0, 8.0, 10.0] * 20,
        "target": [0, 1, 0, 1, 0] * 20,
    })
    
    # Create test data
    X_test = pl.DataFrame({
        "feature1": [1.5, 2.5, 3.5, 4.5, 5.5] * 10,  # 50 rows
        "feature2": [3.0, 5.0, 7.0, 9.0, 11.0] * 10,
    })
    
    train_path = session_dir / "X_train.parquet"
    test_path = session_dir / "X_test.parquet"
    
    write_parquet(X_train, str(train_path))
    write_parquet(X_test, str(test_path))
    
    # Create model registry
    model_registry = [
        {
            "model_id": "test_model",
            "model_path": str(session_dir / "model.pkl"),
            "cv_mean": 0.85,
            "params": {"n_estimators": 10},
        }
    ]
    
    return {
        "session_dir": session_dir,
        "feature_data_path": str(train_path),
        "feature_data_path_test": str(test_path),
        "model_registry": model_registry,
    }


@pytest.fixture
def pseudo_label_state(tmp_path, pseudo_label_test_data):
    """Create test state for pseudo-label tests."""
    return {
        "session_id": "test_pl_session",
        "competition_name": "test_competition",
        "feature_data_path": pseudo_label_test_data["feature_data_path"],
        "feature_data_path_test": pseudo_label_test_data["feature_data_path_test"],
        "model_registry": pseudo_label_test_data["model_registry"],
        "target_col": "target",
        "cv_mean": 0.85,
        "cost_tracker": {
            "total_usd": 0.0,
            "groq_tokens_in": 0,
            "groq_tokens_out": 0,
            "gemini_tokens": 0,
            "llm_calls": 0,
            "budget_usd": 10.0,
            "warning_threshold": 0.7,
            "throttle_threshold": 0.85,
            "triage_threshold": 0.95,
        },
    }


class TestPseudoLabelContract:
    """Test Pseudo-Label Agent contract."""

    def test_contract_graceful_skip_missing_train_data(self, pseudo_label_state):
        """Test pseudo-label gracefully skips if train data not found."""
        pseudo_label_state["feature_data_path"] = "/nonexistent/path.parquet"
        
        result = run_pseudo_label_agent(pseudo_label_state)
        
        # Should not crash, should return False
        assert result["pseudo_labels_applied"] is False
        assert result["pseudo_label_cv_improvement"] == 0.0

    def test_contract_graceful_skip_missing_test_data(self, pseudo_label_state):
        """Test pseudo-label gracefully skips if test data not found."""
        pseudo_label_state["feature_data_path_test"] = "/nonexistent/path.parquet"
        
        result = run_pseudo_label_agent(pseudo_label_state)
        
        # Should not crash, should return False
        assert result["pseudo_labels_applied"] is False
        assert result["pseudo_label_cv_improvement"] == 0.0

    def test_contract_always_writes_pseudo_labels_applied(self, pseudo_label_state):
        """Test pseudo-label always writes pseudo_labels_applied."""
        result = run_pseudo_label_agent(pseudo_label_state)
        
        assert "pseudo_labels_applied" in result
        assert isinstance(result["pseudo_labels_applied"], bool)

    def test_contract_always_writes_cv_improvement(self, pseudo_label_state):
        """Test pseudo-label always writes pseudo_label_cv_improvement."""
        result = run_pseudo_label_agent(pseudo_label_state)
        
        assert "pseudo_label_cv_improvement" in result
        assert isinstance(result["pseudo_label_cv_improvement"], (int, float))

    def test_contract_preserves_state(self, pseudo_label_state):
        """Test pseudo-label preserves existing state keys."""
        original_keys = set(pseudo_label_state.keys())
        
        result = run_pseudo_label_agent(pseudo_label_state)
        
        # All original keys should still be present
        for key in original_keys:
            assert key in result

    def test_contract_handles_empty_model_registry(self, pseudo_label_state):
        """Test pseudo-label handles empty model registry."""
        pseudo_label_state["model_registry"] = []
        
        # Should not crash
        result = run_pseudo_label_agent(pseudo_label_state)
        
        # Should have appropriate output fields
        assert "pseudo_labels_applied" in result

    def test_contract_requires_target_col(self, pseudo_label_state):
        """Test pseudo-label handles missing target_col gracefully."""
        if "target_col" in pseudo_label_state:
            del pseudo_label_state["target_col"]

        # Should handle missing target_col gracefully
        result = run_pseudo_label_agent(pseudo_label_state)

        # Should have output fields even if failed
        assert "pseudo_labels_applied" in result
        assert result["pseudo_labels_applied"] is False


class TestPseudoLabelDataHandling:
    """Test pseudo-label data handling."""

    def test_contract_reads_feature_data_from_state(self, pseudo_label_state):
        """Test pseudo-label reads feature_data_path from state."""
        # Contract: uses state paths, not hardcoded
        result = run_pseudo_label_agent(pseudo_label_state)
        
        # Should complete (may or may not apply labels depending on data)
        assert isinstance(result["pseudo_labels_applied"], bool)

    def test_contract_writes_pseudo_label_data_path_if_applied(self, pseudo_label_state):
        """Test pseudo-label writes pseudo_label_data_path if labels applied."""
        result = run_pseudo_label_agent(pseudo_label_state)
        
        # If labels were applied, should write path
        if result["pseudo_labels_applied"]:
            assert "pseudo_label_data_path" in result
        # If not applied, may or may not write path (implementation detail)

    def test_contract_never_crashes_on_data_issues(self, pseudo_label_state):
        """Test pseudo-label never crashes on data issues."""
        # Corrupt data paths
        pseudo_label_state["feature_data_path"] = ""
        pseudo_label_state["feature_data_path_test"] = ""
        
        # Should not raise, should return gracefully
        result = run_pseudo_label_agent(pseudo_label_state)
        
        assert result["pseudo_labels_applied"] is False


# Import at bottom to avoid circular imports
from agents.pseudo_label_agent import run_pseudo_label_agent
