# tests/contracts/test_ml_optimizer_contract.py
# ─────────────────────────────────────────────────────────────────
# Written: Day 4
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: run_ml_optimizer()
#   INPUT:   state["clean_data_path"] — must exist
#            state["schema_path"]     — must exist
#   OUTPUT:  outputs/{session_id}/best_model.pkl — must exist
#            outputs/{session_id}/metrics.json   — must have
#              cv_mean (float), cv_std (float), fold_scores (list)
#   STATE:   model_registry — list, at least 1 entry after run
#            cv_mean        — float, > 0
#            cv_scores      — list of length n_folds
#            cost_tracker   — not None
#   NEVER:   optimise toward forbidden metrics
#            put raw model object in state (only file pointer)
# ─────────────────────────────────────────────────────────────────
import pytest
import os
import json
import pickle
import numpy as np
from core.state import initial_state
from core.metric_contract import FORBIDDEN_METRICS
from agents.data_engineer import run_data_engineer
from agents.ml_optimizer import run_ml_optimizer

FIXTURE_CSV = "tests/fixtures/tiny_train.csv"


@pytest.fixture(scope="module")
def optimized_state():
    """Run Data Engineer → ML Optimizer pipeline once for all tests."""
    state = initial_state(
        competition="test-titanic",
        data_path=FIXTURE_CSV,
        budget_usd=2.0
    )
    state["target_col"] = "Transported"  # Required for data_engineer schema authority
    state = run_data_engineer(state)
    state = run_ml_optimizer(state)
    return state


class TestMLOptimizerContract:

    def test_runs_without_error(self, optimized_state):
        assert optimized_state is not None

    def test_best_model_pkl_exists(self, optimized_state):
        assert os.path.exists(optimized_state["model_registry"][0]["model_path"]), \
            "best_model.pkl must exist on disk"

    def test_model_is_loadable(self, optimized_state):
        model_path = optimized_state["model_registry"][0]["model_path"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict"), "Loaded object must have predict()"

    def test_metrics_json_exists(self, optimized_state):
        session_id   = optimized_state["session_id"]
        metrics_path = f"outputs/{session_id}/metrics.json"
        assert os.path.exists(metrics_path), "metrics.json must exist"

    def test_metrics_has_cv_mean(self, optimized_state):
        session_id   = optimized_state["session_id"]
        metrics      = json.load(open(f"outputs/{session_id}/metrics.json"))
        assert "cv_mean" in metrics
        assert isinstance(metrics["cv_mean"], float)

    def test_metrics_has_cv_std(self, optimized_state):
        session_id = optimized_state["session_id"]
        metrics    = json.load(open(f"outputs/{session_id}/metrics.json"))
        assert "cv_std" in metrics
        assert isinstance(metrics["cv_std"], float)

    def test_metrics_has_fold_scores(self, optimized_state):
        session_id = optimized_state["session_id"]
        metrics    = json.load(open(f"outputs/{session_id}/metrics.json"))
        assert "fold_scores" in metrics
        assert isinstance(metrics["fold_scores"], list)
        assert len(metrics["fold_scores"]) == 5

    def test_cv_mean_is_positive(self, optimized_state):
        assert optimized_state["cv_mean"] > 0, \
            "CV mean must be positive"

    def test_cv_mean_is_reasonable(self, optimized_state):
        # AUC should be above 0.5 (random baseline) on any real data
        assert optimized_state["cv_mean"] > 0.5, \
            f"CV mean {optimized_state['cv_mean']} is below random baseline (0.5)"

    def test_cv_scores_length_matches_folds(self, optimized_state):
        assert len(optimized_state["cv_scores"]) == 5

    def test_model_registry_updated(self, optimized_state):
        assert optimized_state["model_registry"] is not None
        assert len(optimized_state["model_registry"]) >= 1

    def test_model_registry_entry_has_required_fields(self, optimized_state):
        entry = optimized_state["model_registry"][0]
        for field in ["model_path", "model_type", "cv_mean", "scorer_name"]:
            assert field in entry, f"model_registry entry missing '{field}'"

    def test_model_path_is_string_not_object(self, optimized_state):
        entry = optimized_state["model_registry"][0]
        assert isinstance(entry["model_path"], str), \
            "model_path must be a str pointer — never a model object"

    def test_no_model_object_in_state(self, optimized_state):
        import lightgbm as lgb
        for key, value in optimized_state.items():
            assert not isinstance(value, (lgb.LGBMClassifier, lgb.LGBMRegressor)), \
                f"Model object found in state['{key}'] — only file pointers allowed"

    def test_oof_predictions_path_exists(self, optimized_state):
        assert optimized_state.get("oof_predictions_path") is not None
        assert os.path.exists(optimized_state["oof_predictions_path"])

    def test_oof_predictions_loadable(self, optimized_state):
        oof = np.load(optimized_state["oof_predictions_path"])
        assert len(oof) > 0

    def test_never_optimises_forbidden_metrics(self, optimized_state):
        session_id = optimized_state["session_id"]
        metrics    = json.load(open(f"outputs/{session_id}/metrics.json"))
        scorer     = metrics["scorer_name"]
        assert scorer not in FORBIDDEN_METRICS, \
            f"Scorer '{scorer}' is in FORBIDDEN_METRICS — never optimise toward this"

    def test_requires_clean_data_path(self):
        state = initial_state("test", "tests/fixtures/tiny_train.csv")
        state = {**state, "clean_data_path": None}
        with pytest.raises((ValueError, TypeError)):
            run_ml_optimizer(state)
