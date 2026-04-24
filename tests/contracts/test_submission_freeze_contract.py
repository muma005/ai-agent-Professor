# tests/contracts/test_submission_freeze_contract.py

import pytest
import numpy as np
from core.state import ProfessorState, initial_state
from tools.submission_freeze import apply_submission_freeze

@pytest.fixture
def empty_state():
    return ProfessorState(**initial_state(n_submissions_with_lb=0))

@pytest.fixture
def running_state():
    state = ProfessorState(**initial_state(
        n_submissions_with_lb=2,
        task_type="classification"
    ))
    state.ewma_current = [0.1, 0.2, 0.8, 0.9]
    return state

@pytest.fixture
def running_regression_state():
    state = ProfessorState(**initial_state(
        n_submissions_with_lb=2,
        task_type="regression"
    ))
    state.ewma_current = [1.0, 2.0, 3.0, 4.0]
    return state

class TestSubmissionFreezeContract:

    def test_first_submission_passes_unfrozen(self, empty_state):
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        new_preds, frozen = apply_submission_freeze(empty_state, preds)
        assert not frozen
        assert np.array_equal(new_preds, preds)

    def test_high_correlation_updates_ewma(self, running_state):
        # High correlation: very similar predictions
        preds = np.array([0.12, 0.18, 0.85, 0.88])
        new_preds, frozen = apply_submission_freeze(running_state, preds)
        assert not frozen
        # Check EWMA calculation: 0.7 * ewma + 0.3 * new
        expected = 0.7 * np.array([0.1, 0.2, 0.8, 0.9]) + 0.3 * preds
        assert np.allclose(new_preds, expected)

    def test_low_correlation_freezes_and_returns_old_ewma(self, running_state):
        # Low correlation: totally different predictions
        preds = np.array([0.9, 0.8, 0.2, 0.1])
        new_preds, frozen = apply_submission_freeze(running_state, preds)
        assert frozen
        assert np.array_equal(new_preds, np.array([0.1, 0.2, 0.8, 0.9]))

    def test_regression_threshold_095(self, running_regression_state):
        # Correlation ~ 0.96 (between 0.95 and 0.98)
        # For regression (threshold 0.95), this should PASS
        preds = np.array([1.0, 2.0, 4.0, 3.0])  # Spearman corr is 0.8 with 4 items. Let's make it exactly 0.96... wait. 
        # With 100 items, we can tune it.
        # Let's use a larger array to control correlation more finely
        running_regression_state.ewma_current = np.linspace(0, 10, 100).tolist()
        
        # Add noise to get ~0.96 corr
        np.random.seed(42)
        noise = np.random.normal(0, 0.7, 100)
        preds = np.array(running_regression_state.ewma_current) + noise
        
        from scipy.stats import spearmanr
        corr, _ = spearmanr(preds, running_regression_state.ewma_current)
        # Ensure it's between 0.95 and 0.98
        assert 0.95 < corr < 0.98
        
        new_preds, frozen = apply_submission_freeze(running_regression_state, preds)
        assert not frozen  # Because it's regression and corr > 0.95

    def test_classification_threshold_098(self, running_state):
        # Running state is classification
        running_state.ewma_current = np.linspace(0, 1, 100).tolist()
        
        np.random.seed(42)
        noise = np.random.normal(0, 0.07, 100)
        preds = np.array(running_state.ewma_current) + noise
        
        from scipy.stats import spearmanr
        corr, _ = spearmanr(preds, running_state.ewma_current)
        # Ensure it's between 0.95 and 0.98
        assert 0.95 < corr < 0.98
        
        new_preds, frozen = apply_submission_freeze(running_state, preds)
        assert frozen  # Because it's classification and corr < 0.98
