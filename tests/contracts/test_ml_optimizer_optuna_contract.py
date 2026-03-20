# tests/contracts/test_ml_optimizer_optuna_contract.py
# ─────────────────────────────────────────────────────────────────
# Written: Day 19
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: agents/ml_optimizer.py (Optuna + stability)
#
# INVARIANTS:
#   - After Optuna: top-10 configs re-run with exactly 5 seeds
#   - Winner ranked by stability_score = mean - 1.5*std, NOT peak CV
#   - model_registry records: cv_mean, cv_std, stability_score, seed_results, fold_scores
#   - Calibration runs when metric in PROBABILITY_METRICS
#   - is_calibrated, calibration_method, calibration_score present in registry entry
#   - fold_scores stored in trial user_attrs during Optuna
#   - OOF predictions present in every registry entry
#   - stability_score = mean - 1.5 * std (not mean - 1.0*std, not mean alone)
# ─────────────────────────────────────────────────────────────────
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from agents.ml_optimizer import (
    PROBABILITY_METRICS,
    _run_calibration,
    _select_calibration_method,
    _split_calibration_fold,
    _update_model_registry_with_calibration,
    _suggest_lgbm_params,
    _suggest_xgb_params,
    _suggest_catboost_params,
    _suggest_params,
    _get_study_direction,
    _train_and_optionally_calibrate,
)
from tools.stability_validator import (
    run_with_seeds,
    rank_by_stability,
    StabilityResult,
    DEFAULT_SEEDS,
    STABILITY_PENALTY,
)


class TestMLOptimizerOptunaContract:

    def test_model_registry_has_all_required_fields(self):
        """Registry entries must have the 12+ required fields."""
        REQUIRED = {
            "model_id", "cv_mean", "cv_std", "fold_scores",
            "stability_score", "seed_results", "params",
            "oof_predictions", "data_hash",
            "is_calibrated", "calibration_method", "calibration_score",
        }
        # Build a minimal entry that matches what run_ml_optimizer produces
        entry = {
            "model_id": "lgbm_test",
            "cv_mean": 0.85,
            "cv_std": 0.01,
            "fold_scores": [0.84, 0.85, 0.86, 0.85, 0.84],
            "stability_score": 0.83,
            "seed_results": [0.85, 0.84, 0.85, 0.84, 0.85],
            "params": {"n_estimators": 100},
            "oof_predictions": [0.5] * 100,
            "data_hash": "abc123",
        }
        calib_info = {
            "is_calibrated": True,
            "calibration_method": "sigmoid",
            "calibration_score": 0.12,
            "calibration_n_samples": 150,
        }
        entry = _update_model_registry_with_calibration(entry, calib_info)
        missing = REQUIRED - set(entry.keys())
        assert not missing, f"Missing required registry fields: {missing}"

    def test_winner_ranked_by_stability_not_peak_cv(self):
        """stability_score must match mean - 1.5 * std formula."""
        seeds = [0.85, 0.83, 0.86, 0.84, 0.82]
        result = run_with_seeds(
            config={"n_estimators": 100},
            train_fn=lambda cfg, s: seeds.pop(0),
            seeds=[1, 2, 3, 4, 5],
        )
        computed = float(np.mean([0.85, 0.83, 0.86, 0.84, 0.82])) - 1.5 * float(np.std([0.85, 0.83, 0.86, 0.84, 0.82]))
        assert abs(result.stability_score - computed) < 1e-4, (
            f"stability_score={result.stability_score} does not match "
            f"mean - 1.5*std = {computed:.6f}"
        )

    def test_seed_results_has_five_entries(self):
        """Exactly 5 seeds by default."""
        call_count = [0]
        def train_fn(cfg, seed):
            call_count[0] += 1
            return 0.85
        result = run_with_seeds(config={}, train_fn=train_fn)
        assert len(result.seed_results) == 5

    def test_fold_scores_stored_in_trial_user_attrs(self):
        """Optuna trials must store fold_scores for Wilcoxon gate."""
        import optuna
        from agents.ml_optimizer import _objective
        from sklearn.model_selection import StratifiedKFold
        from unittest.mock import MagicMock

        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.randint(0, 2, 200)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_folds = list(cv.split(X, y))

        contract = MagicMock()
        contract.requires_proba = False
        contract.scorer_fn = lambda y_t, y_p: float(np.mean(y_t == y_p))

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: _objective(trial, X, y, cv_folds, "classification", contract, 8.0),
            n_trials=2,
        )
        for trial in study.trials:
            if trial.state.name == "COMPLETE":
                assert "fold_scores" in trial.user_attrs
                assert len(trial.user_attrs["fold_scores"]) >= 3

    def test_calibration_runs_for_probability_metric(self):
        """Calibration fires for log_loss."""
        assert "log_loss" in PROBABILITY_METRICS
        assert "brier_score" in PROBABILITY_METRICS
        method = _select_calibration_method(500)
        assert method == "sigmoid"
        method = _select_calibration_method(1000)
        assert method == "isotonic"

    def test_calibration_skipped_for_auc_metric(self):
        """auc must NOT be in PROBABILITY_METRICS."""
        assert "auc" not in PROBABILITY_METRICS
        assert "rmse" not in PROBABILITY_METRICS
        assert "accuracy" not in PROBABILITY_METRICS

    def test_oof_predictions_present_and_correct_length(self):
        """OOF predictions must have same length as training data."""
        from agents.ml_optimizer import _get_oof_predictions
        from unittest.mock import MagicMock

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        contract = MagicMock()
        contract.requires_proba = False
        params = {"n_estimators": 10, "verbosity": -1, "n_jobs": 1, "model_type": "lgbm"}
        oof = _get_oof_predictions(X, y, params, "classification", contract)
        assert len(oof) == 100

    def test_study_direction_correct_for_metrics(self):
        """Study direction must match metric type."""
        assert _get_study_direction("auc") == "maximize"
        assert _get_study_direction("accuracy") == "maximize"
        assert _get_study_direction("log_loss") == "minimize"
        assert _get_study_direction("cross_entropy") == "minimize"
        assert _get_study_direction("brier_score") == "minimize"
        assert _get_study_direction("rmse") == "minimize"
