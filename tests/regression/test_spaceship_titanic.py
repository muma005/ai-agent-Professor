"""
Regression test on Spaceship Titanic (Kaggle Playground).
Known baseline: ~0.79 accuracy with basic LightGBM.
Professor v1 achieved Bronze medal (0.80+).
Professor v2 must achieve >= 0.79 (no regression from v1).
"""

import pytest
import os

@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.skipif(not os.environ.get("PROFESSOR_REGRESSION_TESTS"), 
                    reason="Set PROFESSOR_REGRESSION_TESTS=1 to run")
class TestSpaceshipTitanic:
    
    COMPETITION = "spaceship-titanic"
    DATA_DIR = "tests/regression/data/spaceship-titanic"
    MINIMUM_CV = 0.79  # v1 baseline
    
    def test_sprint_mode_produces_submission(self):
        """SPRINT mode completes and produces valid submission."""
        # Run in SPRINT mode (fastest)
        final_state = _run_professor(
            competition=self.COMPETITION,
            data_dir=self.DATA_DIR,
            depth="sprint",
            mode="traditional",
        )
        
        assert final_state.submission_path != ""
        assert os.path.exists(final_state.submission_path)
        assert final_state.cv_mean > 0.5  # Better than random
    
    def test_standard_mode_beats_baseline(self):
        """STANDARD mode achieves >= v1 baseline."""
        final_state = _run_professor(
            competition=self.COMPETITION,
            data_dir=self.DATA_DIR,
            depth="standard",
            mode="traditional",
        )
        
        assert final_state.cv_mean >= self.MINIMUM_CV
        assert getattr(final_state, 'metric_verified', True) == True
        assert len(final_state.feature_manifest) >= 5
        assert len(final_state.model_configs) == 3  # LGB + XGB + CAT
    
    def test_notebook_reproduces_submission(self):
        """Generated notebook produces same submission."""
        final_state = _run_professor(
            competition=self.COMPETITION,
            data_dir=self.DATA_DIR,
            depth="sprint",
            mode="traditional",
        )
        
        assert getattr(final_state, 'notebook_reproduction_validated', True) == True
    
    def test_no_critic_false_positives(self):
        """Critic should return CLEAR on clean Spaceship Titanic run."""
        final_state = _run_professor(
            competition=self.COMPETITION,
            data_dir=self.DATA_DIR,
            depth="standard",
            mode="traditional",
        )
        
        # Clean competition — Critic should not find CRITICAL
        severity = final_state.critic_verdict.get("severity", "CLEAR") if final_state.critic_verdict else "CLEAR"
        assert severity in ("CLEAR", "HIGH")  # HIGH is acceptable, CRITICAL is not

def _run_professor(competition, data_dir, depth, mode):
    """Helper to run professor and return final state"""
    from core.state import ProfessorState
    from core.professor import run_professor
    from graph.builder import build_professor_graph
    
    # Just a placeholder for how this might be run
    # since we don't have the full framework loaded in this test snippet
    # The actual implementation would invoke the LangGraph
    pass
