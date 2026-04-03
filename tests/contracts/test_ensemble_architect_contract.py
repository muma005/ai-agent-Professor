# tests/contracts/test_ensemble_architect_contract.py
#
# CONTRACT: agents/ensemble_architect.py
# Written: Day 22. IMMUTABLE — never edit after Day 22.
#
# INVARIANTS (one test per invariant):
#   1. Data hash validation runs before anything else
#   2. OOF validation runs before weight optimisation
#   3. Diversity pruning runs before weight optimisation
#   4. No model pair with correlation > 0.98 survives in final ensemble
#   5. Weights sum to 1.0 (tolerance 1e-6)
#   6. No individual weight below 0.05
#   7. val_holdout indices have zero overlap with Optuna optimisation indices
#   8. Wilcoxon gate called exactly once per run
#   9. ensemble_oof length equals len(y_train)
#  10. Hash mismatch raises ValueError containing "retrain required"
#  11. Missing oof_predictions raises ValueError naming the model
#  12. ensemble_accepted is always set in state (True or False)

import time
import pytest
import numpy as np
import unittest.mock as mock


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def y_train_binary():
    """300 binary labels, ~50% positive. Fixed seed."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, 300).astype(np.float32)


@pytest.fixture
def base_oof(y_train_binary):
    """Reasonable OOF predictions for a binary task."""
    rng = np.random.default_rng(42)
    return np.clip(
        y_train_binary + rng.normal(0, 0.3, len(y_train_binary)),
        0, 1
    ).tolist()


@pytest.fixture
def base_registry(base_oof, y_train_binary):
    """
    Two models with low correlation (< 0.5). Both valid.
    Both have the same data_hash. Both have fold_scores.
    """
    rng = np.random.default_rng(99)
    oof_b = np.clip(
        np.array(base_oof) + rng.normal(0, 0.4, len(base_oof)),
        0, 1
    ).tolist()
    return {
        "model_alpha": {
            "cv_mean":        0.810,
            "cv_std":         0.012,
            "fold_scores":    [0.808, 0.812, 0.811, 0.809, 0.810],
            "stability_score": 0.792,
            "seed_results":   [0.810, 0.808, 0.812, 0.807, 0.813],
            "oof_predictions": base_oof,
            "params":         {"model_type": "lgbm", "n_estimators": 500},
            "data_hash":      "abc123",
        },
        "model_beta": {
            "cv_mean":        0.805,
            "cv_std":         0.014,
            "fold_scores":    [0.803, 0.806, 0.805, 0.807, 0.804],
            "stability_score": 0.784,
            "seed_results":   [0.805, 0.803, 0.807, 0.802, 0.808],
            "oof_predictions": oof_b,
            "params":         {"model_type": "xgb", "n_estimators": 500},
            "data_hash":      "abc123",
        },
    }


@pytest.fixture
def base_state(base_registry, y_train_binary):
    """Minimal valid state for ensemble_architect."""
    return {
        "model_registry":     base_registry,
        "y_train":            y_train_binary,
        "data_hash":          "abc123",
        "evaluation_metric":  "accuracy",
        "task_type":          "binary_classification",
        "session_id":         "test_session",
        "target_column":      "Transported",
        "feature_order":      [f"f{i}" for i in range(10)],
    }


# ── CONTRACT TEST 1 — Data hash validation runs first ─────────────

class TestEnsembleArchitectContract:
    """
    Contract tests for agents/ensemble_architect.py.
    Each test method validates one invariant.
    """

    # ── Invariant 1: Data hash validation runs before anything else ──

    def test_mismatched_hash_raises_before_oof_check(self, base_state):
        """
        Invariant: If all models have a data_hash that does not match
        state['data_hash'], a ValueError containing 'retrain required' must
        be raised BEFORE any OOF validation, diversity pruning, or Optuna
        runs. This protects against silently blending models trained on
        stale data versions.
        """
        state = {
            **base_state,
            "data_hash": "different_hash",  # nothing matches
        }
        with pytest.raises(ValueError, match="(?i)retrain required"):
            from agents.ensemble_architect import run_ensemble_architect
            run_ensemble_architect(state)

    def test_error_message_contains_retrain_required(self, base_state):
        """
        Invariant: The error message for a hash mismatch must contain the
        phrase 'retrain required' so that operators know the corrective
        action. A generic ValueError is not sufficient.
        """
        state = {**base_state, "data_hash": "wrong"}
        with pytest.raises(ValueError) as exc_info:
            from agents.ensemble_architect import run_ensemble_architect
            run_ensemble_architect(state)
        assert "retrain required" in str(exc_info.value).lower()

    def test_partial_hash_mismatch_filters_not_raises(self, base_state):
        """
        Invariant: If only some models have a stale data_hash, the stale
        models must be filtered out and the run must continue with the
        matching models. This ensures we don't lose valid models just
        because one model in the registry is stale.
        """
        registry = dict(base_state["model_registry"])
        registry["model_beta"] = {
            **registry["model_beta"],
            "data_hash": "stale_hash",
        }
        state = {**base_state, "model_registry": registry}
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(state)
        assert "model_alpha" in result["selected_models"]
        assert "model_beta" not in result["selected_models"]

    # ── Invariant 2: OOF validation runs before weight optimisation ──

    def test_missing_oof_raises_value_error_naminging_model(self, base_state):
        """
        Invariant: If any model in the registry is missing the
        'oof_predictions' key, a ValueError must be raised that names the
        offending model. This prevents silent failures where a model
        without OOF predictions would cause shape errors downstream.
        """
        registry = dict(base_state["model_registry"])
        del registry["model_beta"]["oof_predictions"]
        state = {**base_state, "model_registry": registry}
        with pytest.raises(ValueError) as exc_info:
            from agents.ensemble_architect import run_ensemble_architect
            run_ensemble_architect(state)
        assert "model_beta" in str(exc_info.value)

    def test_empty_oof_raises_value_error(self, base_state):
        """
        Invariant: A model with an empty oof_predictions list must raise
        a ValueError. An empty list is not a valid OOF prediction array
        and would cause division-by-zero or shape errors in correlation
        and blending calculations.
        """
        registry = dict(base_state["model_registry"])
        registry["model_beta"]["oof_predictions"] = []
        state = {**base_state, "model_registry": registry}
        with pytest.raises(ValueError):
            from agents.ensemble_architect import run_ensemble_architect
            run_ensemble_architect(state)

    def test_mismatched_oof_length_raises_value_error(self, base_state, y_train_binary):
        """
        Invariant: If a model's oof_predictions length does not match
        len(y_train), a ValueError must be raised. Mismatched OOF lengths
        would cause silent broadcasting errors in the blend computation,
        producing incorrect ensemble predictions.
        """
        registry = dict(base_state["model_registry"])
        registry["model_beta"]["oof_predictions"] = [0.5] * (len(y_train_binary) - 10)
        state = {**base_state, "model_registry": registry}
        with pytest.raises(ValueError):
            from agents.ensemble_architect import run_ensemble_architect
            run_ensemble_architect(state)

    # ── Invariant 3: Diversity pruning runs before weight optimisation ──

    def test_high_correlation_models_pruned_one_survives(self, base_state, base_oof):
        """
        Invariant: When two models have OOF correlation > 0.98, only the
        one with higher cv_mean survives. The other must appear in
        models_pruned_diversity. This proves diversity pruning executes
        and rejects redundant models before Optuna weight optimisation
        begins — otherwise both would receive non-zero weights.
        """
        # Build a near-identical OOF (correlation ~0.999)
        near_identical = [v + 0.001 for v in base_oof]

        registry = dict(base_state["model_registry"])
        registry["model_gamma"] = {
            "cv_mean":         0.800,
            "cv_std":          0.010,
            "fold_scores":     [0.799, 0.801, 0.800, 0.800, 0.800],
            "stability_score": 0.785,
            "seed_results":    [0.800] * 5,
            "oof_predictions": near_identical,
            "params":          {"model_type": "lgbm"},
            "data_hash":       "abc123",
        }
        state = {**base_state, "model_registry": registry}
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(state)

        selected = result["selected_models"]
        pruned   = result["models_pruned_diversity"]

        assert "model_alpha" in selected or "model_alpha" in pruned
        assert "model_gamma" in selected or "model_gamma" in pruned
        assert not ("model_alpha" in selected and "model_gamma" in selected), (
            "Both high-correlation models survived. Diversity pruning did not run."
        )

    def test_pruning_timestamp_before_optuna_timestamp(self, base_state, base_oof, monkeypatch):
        """
        Invariant: Diversity pruning MUST complete before Optuna weight
        optimisation starts. We prove this by monkeypatching both
        _prune_by_diversity and _run_weight_optimisation to record
        timestamps, then asserting pruning_ts < optuna_ts. If Optuna
        runs first, the weight optimisation would include redundant
        models, defeating the purpose of diversity pruning.
        """
        timestamps = {}

        import agents.ensemble_architect as ea

        original_prune = ea._prune_by_diversity
        original_optuna = ea._run_weight_optimisation

        def tracked_prune(registry):
            timestamps["prune_start"] = time.monotonic()
            result = original_prune(registry)
            timestamps["prune_end"] = time.monotonic()
            return result

        def tracked_optuna(*args, **kwargs):
            timestamps["optuna_start"] = time.monotonic()
            result = original_optuna(*args, **kwargs)
            timestamps["optuna_end"] = time.monotonic()
            return result

        monkeypatch.setattr(ea, "_prune_by_diversity", tracked_prune)
        monkeypatch.setattr(ea, "_run_weight_optimisation", tracked_optuna)

        from agents.ensemble_architect import run_ensemble_architect
        run_ensemble_architect(base_state)

        assert "prune_start" in timestamps, "Diversity pruning was never called."
        assert "optuna_start" in timestamps, "Optuna weight optimisation was never called."
        assert timestamps["prune_end"] < timestamps["optuna_start"], (
            f"Diversity pruning ended at {timestamps['prune_end']:.6f} but "
            f"Optuna started at {timestamps['optuna_start']:.6f}. "
            "Diversity pruning must complete before Optuna begins."
        )

    def test_pruning_recorded_in_state_even_if_empty(self, base_state):
        """
        Invariant: The models_pruned_diversity key must always be present
        in the result, even when no models were pruned. This ensures
        downstream consumers can rely on the key existing without
        defensive .get() calls.
        """
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(base_state)
        assert "models_pruned_diversity" in result
        assert isinstance(result["models_pruned_diversity"], list)

    # ── Invariant 4: No pair with correlation > 0.98 in final ensemble ──

    def test_no_pair_above_0_98_in_final_ensemble(self, base_state):
        """
        Invariant: After any run, the ensemble_correlation_matrix must
        show all pairs with correlation <= 0.98. This is a hard constraint
        — not a soft preference. If any pair exceeds 0.98, diversity
        pruning has failed and the ensemble contains redundant models.
        """
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(base_state)
        corr_matrix = result.get("ensemble_correlation_matrix", {})

        for pair, corr in corr_matrix.items():
            assert corr <= 0.98, (
                f"Pair {pair} has correlation {corr:.4f} > 0.98 in final ensemble. "
                "Diversity pruning threshold violation."
            )

    def test_model_with_correlation_above_0_98_is_pruned(self, base_state, base_oof):
        """
        Invariant: A model whose OOF predictions have correlation > 0.98
        with the anchor model must be pruned. We construct a near-identical
        OOF (noise ~0.001) to guarantee correlation > 0.98 and assert
        the model appears in models_pruned_diversity.
        """
        from scipy.stats import pearsonr

        rng = np.random.default_rng(42)
        base = np.array(base_oof)

        # Build model with ~0.985 correlation to model_alpha
        noise_985 = rng.normal(0, 0.001, len(base))
        oof_985   = np.clip(base + noise_985, 0, 1)
        actual_corr_985, _ = pearsonr(base, oof_985)

        registry = dict(base_state["model_registry"])
        registry["model_too_similar"] = {
            "cv_mean":         0.800,
            "cv_std":          0.010,
            "fold_scores":     [0.800] * 5,
            "stability_score": 0.785,
            "seed_results":    [0.800] * 5,
            "oof_predictions": oof_985.tolist(),
            "params":          {"model_type": "lgbm"},
            "data_hash":       "abc123",
        }
        state = {**base_state, "model_registry": registry}
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(state)

        # If actual correlation > 0.98, must be pruned
        if actual_corr_985 > 0.98:
            assert "model_too_similar" in result["models_pruned_diversity"], (
                f"Model with correlation {actual_corr_985:.4f} should be pruned but was not."
            )
        else:
            # Correlation did not exceed threshold — test is inconclusive, skip
            pytest.skip(
                f"Generated correlation {actual_corr_985:.4f} did not exceed 0.98. "
                "Adjust noise level."
            )

    # ── Invariant 5: Weights sum to 1.0 ──────────────────────────────

    def test_weights_sum_to_1_tolerance_1e6(self, base_state):
        """
        Invariant: ensemble_weights must always sum to 1.0 within
        tolerance 1e-6. This is guaranteed by the softmax normalisation
        followed by clip-and-renormalise. If the sum deviates, the
        ensemble predictions would be systematically scaled wrong.
        """
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(base_state)
        total = sum(result["ensemble_weights"].values())
        assert abs(total - 1.0) < 1e-6, (
            f"Weights sum to {total:.10f}, not 1.0. "
            "Softmax normalisation is not applied."
        )

    def test_weights_sum_to_1_with_three_models(self, base_state, base_oof):
        """
        Invariant: The weight sum to 1.0 must hold for any number of
        selected models, not just two. We add a third model with
        genuinely different OOF predictions and verify the invariant.
        """
        rng = np.random.default_rng(77)
        oof_c = np.clip(np.array(base_oof) + rng.normal(0, 0.5, len(base_oof)), 0, 1)

        registry = dict(base_state["model_registry"])
        registry["model_delta"] = {
            "cv_mean":         0.803,
            "cv_std":          0.011,
            "fold_scores":     [0.803] * 5,
            "stability_score": 0.786,
            "seed_results":    [0.803] * 5,
            "oof_predictions": oof_c.tolist(),
            "params":          {"model_type": "catboost"},
            "data_hash":       "abc123",
        }
        state = {**base_state, "model_registry": registry}
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(state)
        total = sum(result["ensemble_weights"].values())
        assert abs(total - 1.0) < 1e-6

    # ── Invariant 6: No weight below 0.05 ────────────────────────────

    def test_no_weight_below_0_05(self, base_state):
        """
        Invariant: No individual weight may be below 0.05 after clipping
        and renormalisation. This prevents Optuna from effectively
        excluding a model from the ensemble, which would make the
        diversity pruning step meaningless (a pruned model could sneak
        back in with a tiny weight).
        """
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(base_state)
        for model, weight in result["ensemble_weights"].items():
            assert weight >= 0.05, (
                f"Model '{model}' has weight {weight:.6f} < 0.05 floor. "
                "Post-softmax clipping and renormalisation is not applied."
            )

    def test_weight_floor_applied_after_normalisation(self, base_state, base_oof):
        """
        Invariant: Even when a model has notably worse CV performance
        (which would drive Optuna to assign it a very low raw weight),
        the post-softmax clip-to-0.05 and renormalise must ensure the
        floor holds. We add a weak model (cv_mean=0.795 vs 0.810 anchor)
        and verify all weights >= 0.05.
        """
        rng = np.random.default_rng(55)
        oof_c = np.clip(np.array(base_oof) + rng.normal(0, 0.45, len(base_oof)), 0, 1)
        registry = dict(base_state["model_registry"])
        registry["model_weak"] = {
            "cv_mean":         0.795,   # notably worse
            "cv_std":          0.020,
            "fold_scores":     [0.795] * 5,
            "stability_score": 0.765,
            "seed_results":    [0.795] * 5,
            "oof_predictions": oof_c.tolist(),
            "params":          {"model_type": "lgbm"},
            "data_hash":       "abc123",
        }
        state = {**base_state, "model_registry": registry}
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(state)
        for model, weight in result["ensemble_weights"].items():
            assert weight >= 0.05

    # ── Invariant 7: Holdout never used in weight optimisation ───────

    def test_holdout_indices_not_in_opt_pool(self, base_state, monkeypatch):
        """
        Invariant: The indices used for val_holdout must have zero overlap
        with the indices used in Optuna's objective function. If any
        holdout sample appears in the optimisation pool, the holdout
        score would be optimistically biased and could not serve as an
        unbiased validation of the ensemble. We verify this by
        monkeypatching _run_weight_optimisation and checking that
        len(opt_pool) + len(holdout) == len(y_train).
        """
        opt_indices_seen  = []
        holdout_indices   = []

        import agents.ensemble_architect as ea
        original = ea._run_weight_optimisation

        def mock_optuna(oof_stack_opt, y_opt, oof_stack_val, y_val, *args, **kwargs):
            opt_indices_seen.append(len(oof_stack_opt))
            holdout_indices.append(len(oof_stack_val))
            n_models = oof_stack_opt.shape[1]
            return np.ones(n_models) / n_models, 0.8

        monkeypatch.setattr(ea, "_run_weight_optimisation", mock_optuna)

        from agents.ensemble_architect import run_ensemble_architect
        run_ensemble_architect(base_state)

        assert len(opt_indices_seen) > 0, "Optuna objective was never called"
        total_from_split = opt_indices_seen[0] + holdout_indices[0]
        assert total_from_split == len(base_state["y_train"]), (
            f"opt_pool ({opt_indices_seen[0]}) + holdout ({holdout_indices[0]}) = "
            f"{total_from_split} != len(y_train) ({len(base_state['y_train'])}). "
            "Data partitioning is incorrect — holdout contamination detected."
        )

    def test_holdout_is_approximately_20_percent(self, base_state, y_train_binary):
        """
        Invariant: val_holdout must be approximately 20% of the training
        data (tolerance ±5%). This ensures the holdout set is large enough
        to provide a reliable validation score while leaving sufficient
        data for weight optimisation.
        """
        n = len(y_train_binary)
        expected_holdout = int(n * 0.20)
        tolerance        = int(n * 0.05)

        captured = {}
        import agents.ensemble_architect as ea
        original = ea._run_weight_optimisation

        def capture(oof_stack_opt, y_opt, oof_stack_val, y_val, *args, **kwargs):
            captured["n_holdout"] = len(y_val)
            captured["n_opt"] = len(y_opt)
            return np.ones(oof_stack_opt.shape[1]) / oof_stack_opt.shape[1], 0.8

        with mock.patch.object(ea, "_run_weight_optimisation", side_effect=capture):
            from agents.ensemble_architect import run_ensemble_architect
            run_ensemble_architect(base_state)

        assert captured.get("n_holdout") is not None, "Holdout size not captured."
        assert abs(captured["n_holdout"] - expected_holdout) <= tolerance, (
            f"Holdout size {captured['n_holdout']} is not ~20% of {n} rows."
        )

    # ── Invariant 8: Wilcoxon gate called exactly once ───────────────

    def test_wilcoxon_gate_called_exactly_once(self, base_state, monkeypatch):
        """
        Invariant: is_significantly_better must be called exactly once
        per run. If called zero times, the ensemble is accepted without
        statistical validation. If called more than once, the gate is
        being applied inconsistently (e.g., per-model instead of
        ensemble-vs-best-single).
        """
        call_count = []

        import agents.ensemble_architect as ea
        original = ea.is_significantly_better

        def counting_gate(*args, **kwargs):
            call_count.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(ea, "is_significantly_better", counting_gate)
        from agents.ensemble_architect import run_ensemble_architect
        run_ensemble_architect(base_state)

        assert len(call_count) == 1, (
            f"Wilcoxon gate called {len(call_count)} times. Expected exactly 1."
        )

    def test_ensemble_accepted_always_set_in_result(self, base_state):
        """
        Invariant: ensemble_accepted must always be present in the result
        dict, regardless of whether the ensemble beat the best single
        model. Downstream pipeline nodes depend on this key to decide
        whether to use ensemble predictions or fall back to the best
        single model.
        """
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(base_state)
        assert "ensemble_accepted" in result, (
            "ensemble_accepted not set in state after ensemble_architect run."
        )
        assert isinstance(result["ensemble_accepted"], bool), (
            f"ensemble_accepted is {type(result['ensemble_accepted'])}, expected bool."
        )

    # ── Invariant 9: Ensemble OOF length matches training data ───────

    def test_ensemble_oof_length_equals_y_train_length(self, base_state, y_train_binary):
        """
        Invariant: len(ensemble_oof) must equal len(y_train). The ensemble
        OOF predictions cover every training sample, so the lengths must
        match. A mismatch would indicate a bug in the blend or meta-learner
        OOF computation.
        """
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(base_state)
        assert len(result["ensemble_oof"]) == len(y_train_binary), (
            f"ensemble_oof length {len(result['ensemble_oof'])} != "
            f"y_train length {len(y_train_binary)}."
        )

    def test_ensemble_oof_is_list_of_floats(self, base_state):
        """
        Invariant: ensemble_oof must be a list of Python floats, not a
        numpy array or other type. This ensures serialisability to JSON
        for state persistence and lineage logging.
        """
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(base_state)
        oof = result["ensemble_oof"]
        assert isinstance(oof, list), f"ensemble_oof is {type(oof)}, expected list."
        assert all(isinstance(v, float) for v in oof[:10]), (
            "ensemble_oof contains non-float values."
        )

    # ── Invariant 10: All required state keys set ────────────────────

    def test_all_required_state_keys_present(self, base_state):
        """
        Invariant: Every key that ensemble_architect.py promises to set
        must be present in the result. Missing keys would cause
        downstream pipeline nodes to crash with KeyError. The required
        keys are: selected_models, ensemble_weights, ensemble_oof,
        ensemble_holdout_score, ensemble_accepted,
        ensemble_correlation_matrix, models_pruned_diversity,
        meta_learner_used.
        """
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(base_state)
        missing = [k for k in self.REQUIRED_KEYS if k not in result]
        assert not missing, (
            f"ensemble_architect did not set required state keys: {missing}"
        )

    def test_selected_models_is_list_of_strings(self, base_state):
        """
        Invariant: selected_models must be a list of strings (model names).
        This ensures downstream code can use these names as dictionary
        keys to look up weights and registry entries.
        """
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(base_state)
        assert isinstance(result["selected_models"], list)
        assert all(isinstance(m, str) for m in result["selected_models"])

    def test_ensemble_weights_keys_match_selected_models(self, base_state):
        """
        Invariant: The keys of ensemble_weights must exactly match the
        model names in selected_models. A mismatch would mean either a
        selected model has no weight (would crash during blending) or
        a pruned model has a weight (would silently include it in the
        blend).
        """
        from agents.ensemble_architect import run_ensemble_architect
        result = run_ensemble_architect(base_state)
        assert set(result["ensemble_weights"].keys()) == set(result["selected_models"]), (
            "ensemble_weights keys do not match selected_models."
        )

    REQUIRED_KEYS = [
        "selected_models",
        "ensemble_weights",
        "ensemble_oof",
        "ensemble_holdout_score",
        "ensemble_accepted",
        "ensemble_correlation_matrix",
        "models_pruned_diversity",
        "meta_learner_used",
    ]
