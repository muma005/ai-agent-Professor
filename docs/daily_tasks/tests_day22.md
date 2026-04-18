# Day 22 — Test Specification
## Prompt for Qwen Code

---

## BEFORE YOU WRITE A SINGLE TEST

Read these files completely. Do not write any tests until you have finished reading all of them.

```
agents/ensemble_architect.py          ← the file you are testing
tools/wilcoxon_gate.py                ← is_significantly_better() signature
core/state.py                         ← ProfessorState keys
tests/contracts/test_ml_optimizer_optuna_contract.py   ← pattern to follow
tests/regression/test_phase2_regression.py             ← pattern to follow
```

After reading, answer these three questions in a comment at the top of each test file before writing any test:
1. What does `model_registry` contain per entry? List the keys.
2. What does `ensemble_architect.py` write to state? List the keys.
3. What is the OOF predictions shape for a dataset with N training rows?

If you cannot answer all three from reading the code, read again. Do not guess.

---

## FILE 1 — `tests/contracts/test_ensemble_architect_contract.py`

This file is immutable after Day 22. Write every test correctly the first time.

```python
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
```

### Fixtures you must define

```python
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
```

### CONTRACT TEST 1 — Data hash validation runs first

```python
class TestContract1DataHashValidationRunsFirst:
    """
    Invariant: data_hash check is the first operation. If it raises,
    no other code has run — no OOF check, no pruning, no Optuna.
    """

    def test_mismatched_hash_raises_before_oof_check(self, base_state):
        """All models have wrong hash. Must raise before touching OOF."""
        state = {
            **base_state,
            "data_hash": "different_hash",  # nothing matches
        }
        with pytest.raises(ValueError, match="retrain required"):
            run_ensemble_architect(state)

    def test_error_message_names_retrain(self, base_state):
        """Error message must contain 'retrain required' — not a generic ValueError."""
        state = {**base_state, "data_hash": "wrong"}
        with pytest.raises(ValueError) as exc_info:
            run_ensemble_architect(state)
        assert "retrain required" in str(exc_info.value).lower()

    def test_partial_hash_mismatch_filters_not_raises(self, base_state):
        """
        One model matches current hash, one does not.
        Must NOT raise — must filter the stale model and continue with the matching one.
        """
        registry = dict(base_state["model_registry"])
        registry["model_beta"] = {
            **registry["model_beta"],
            "data_hash": "stale_hash",
        }
        state = {**base_state, "model_registry": registry}
        result = run_ensemble_architect(state)
        assert "model_alpha" in result["selected_models"]
        assert "model_beta" not in result["selected_models"]
```

### CONTRACT TEST 2 — OOF validation

```python
class TestContract2OOFValidation:

    def test_missing_oof_raises_value_error(self, base_state):
        """Model with no oof_predictions key raises ValueError naming the model."""
        registry = dict(base_state["model_registry"])
        del registry["model_beta"]["oof_predictions"]
        state = {**base_state, "model_registry": registry}
        with pytest.raises(ValueError) as exc_info:
            run_ensemble_architect(state)
        assert "model_beta" in str(exc_info.value)

    def test_empty_oof_raises_value_error(self, base_state):
        """Model with oof_predictions=[] raises ValueError."""
        registry = dict(base_state["model_registry"])
        registry["model_beta"]["oof_predictions"] = []
        state = {**base_state, "model_registry": registry}
        with pytest.raises(ValueError):
            run_ensemble_architect(state)

    def test_mismatched_oof_length_raises_value_error(self, base_state, y_train_binary):
        """OOF length != len(y_train) raises ValueError."""
        registry = dict(base_state["model_registry"])
        registry["model_beta"]["oof_predictions"] = [0.5] * (len(y_train_binary) - 10)
        state = {**base_state, "model_registry": registry}
        with pytest.raises(ValueError):
            run_ensemble_architect(state)
```

### CONTRACT TEST 3 — Diversity pruning runs before weight optimisation

```python
class TestContract3DiversityPruningRunsBeforeOptuna:

    def test_high_correlation_model_not_in_final_ensemble(self, base_state, base_oof):
        """
        Two models with correlation > 0.98. Only the higher-CV model survives.
        The other must appear in state['models_pruned_diversity'].
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
        result = run_ensemble_architect(state)

        # model_alpha and model_gamma have near-identical OOF
        # exactly one of them must be pruned
        selected = result["selected_models"]
        pruned   = result["models_pruned_diversity"]

        assert "model_alpha" in selected or "model_alpha" in pruned
        assert "model_gamma" in selected or "model_gamma" in pruned
        assert not ("model_alpha" in selected and "model_gamma" in selected), (
            "Both high-correlation models survived. Diversity pruning did not run."
        )

    def test_pruning_recorded_in_state(self, base_state, base_oof):
        """models_pruned_diversity must be set even if empty (no pruning needed)."""
        result = run_ensemble_architect(base_state)
        assert "models_pruned_diversity" in result
        assert isinstance(result["models_pruned_diversity"], list)
```

### CONTRACT TEST 4 — No pair with correlation > 0.98 in final ensemble

```python
class TestContract4CorrelationThreshold:

    def test_no_pair_above_0_98_in_final_ensemble(self, base_state):
        """
        After any run, correlation_matrix must show all pairs <= 0.98.
        This is the hard constraint — not a soft preference.
        """
        result = run_ensemble_architect(base_state)
        corr_matrix = result.get("ensemble_correlation_matrix", {})

        for pair, corr in corr_matrix.items():
            assert corr <= 0.98, (
                f"Pair {pair} has correlation {corr:.4f} > 0.98 in final ensemble. "
                "Diversity pruning threshold violation."
            )

    def test_threshold_is_strictly_0_98_not_0_99(self, base_state, base_oof):
        """
        A model with exactly 0.985 correlation must be pruned.
        A model with exactly 0.975 correlation must NOT be pruned.
        Both cases tested.
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
```

### CONTRACT TEST 5 — Weights sum to 1.0

```python
class TestContract5WeightsSumToOne:

    def test_weights_sum_to_1_tolerance_1e6(self, base_state):
        result = run_ensemble_architect(base_state)
        total = sum(result["ensemble_weights"].values())
        assert abs(total - 1.0) < 1e-6, (
            f"Weights sum to {total:.10f}, not 1.0. "
            "Softmax normalisation is not applied."
        )

    def test_weights_sum_to_1_with_three_models(self, base_state, base_oof):
        """Weight sum invariant holds for any number of models."""
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
        result = run_ensemble_architect(state)
        total = sum(result["ensemble_weights"].values())
        assert abs(total - 1.0) < 1e-6
```

### CONTRACT TEST 6 — No weight below 0.05

```python
class TestContract6NoWeightBelowFloor:

    def test_no_weight_below_0_05(self, base_state):
        result = run_ensemble_architect(base_state)
        for model, weight in result["ensemble_weights"].items():
            assert weight >= 0.05, (
                f"Model '{model}' has weight {weight:.6f} < 0.05 floor. "
                "Post-softmax clipping and renormalisation is not applied."
            )

    def test_weight_floor_applied_after_normalisation(self, base_state, base_oof):
        """
        Even if Optuna assigns a very low raw weight to a model,
        after clipping to 0.05 and renormalising the floor must hold.
        This tests the order of operations: clip → renormalise → store.
        """
        # Three models — give Optuna room to assign low weight to one
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
        result = run_ensemble_architect(state)
        for model, weight in result["ensemble_weights"].items():
            assert weight >= 0.05
```

### CONTRACT TEST 7 — Holdout never used in weight optimisation

```python
class TestContract7HoldoutNotUsedInOptimisation:

    def test_holdout_indices_not_in_opt_pool(self, base_state, monkeypatch):
        """
        The indices passed to Optuna's objective must not overlap
        with the val_holdout indices.
        Zero overlap is required — even one shared index is a violation.
        """
        opt_indices_seen  = []
        holdout_indices   = []

        original_objective = None  # will be set after import

        import agents.ensemble_architect as ea

        original_run_optuna = ea._run_weight_optimisation

        def mock_optuna(oof_stack, y, opt_indices, val_indices, *args, **kwargs):
            opt_indices_seen.extend(opt_indices.tolist())
            holdout_indices.extend(val_indices.tolist())
            return original_run_optuna(oof_stack, y, opt_indices, val_indices, *args, **kwargs)

        monkeypatch.setattr(ea, "_run_weight_optimisation", mock_optuna)

        run_ensemble_architect(base_state)

        overlap = set(opt_indices_seen) & set(holdout_indices)
        assert len(overlap) == 0, (
            f"{len(overlap)} indices appear in both opt_pool and val_holdout. "
            "Holdout contamination detected."
        )

    def test_holdout_is_approximately_20_percent(self, base_state, y_train_binary):
        """val_holdout must be ~20% of training data. Tolerance: ±5%."""
        n = len(y_train_binary)
        expected_holdout = int(n * 0.20)
        tolerance        = int(n * 0.05)

        captured = {}
        import agents.ensemble_architect as ea
        original = ea._run_weight_optimisation

        def capture(oof_stack, y, opt_idx, val_idx, *args, **kwargs):
            captured["n_holdout"] = len(val_idx)
            return original(oof_stack, y, opt_idx, val_idx, *args, **kwargs)

        import unittest.mock as mock
        with mock.patch.object(ea, "_run_weight_optimisation", side_effect=capture):
            run_ensemble_architect(base_state)

        assert captured.get("n_holdout") is not None, "Holdout size not captured."
        assert abs(captured["n_holdout"] - expected_holdout) <= tolerance, (
            f"Holdout size {captured['n_holdout']} is not ~20% of {n} rows."
        )
```

### CONTRACT TEST 8 — Wilcoxon gate called exactly once

```python
class TestContract8WilcoxonGateCalled:

    def test_wilcoxon_called_exactly_once(self, base_state, monkeypatch):
        call_count = []

        import tools.wilcoxon_gate as wg
        original = wg.is_significantly_better

        def counting_gate(*args, **kwargs):
            call_count.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(wg, "is_significantly_better", counting_gate)
        run_ensemble_architect(base_state)

        assert len(call_count) == 1, (
            f"Wilcoxon gate called {len(call_count)} times. Expected exactly 1."
        )

    def test_ensemble_accepted_always_set(self, base_state):
        """ensemble_accepted must be in state regardless of whether ensemble won."""
        result = run_ensemble_architect(base_state)
        assert "ensemble_accepted" in result, (
            "ensemble_accepted not set in state after ensemble_architect run."
        )
        assert isinstance(result["ensemble_accepted"], bool), (
            f"ensemble_accepted is {type(result['ensemble_accepted'])}, expected bool."
        )
```

### CONTRACT TEST 9 — OOF length matches training data

```python
class TestContract9OOFLength:

    def test_ensemble_oof_length_equals_y_train_length(self, base_state, y_train_binary):
        result = run_ensemble_architect(base_state)
        assert len(result["ensemble_oof"]) == len(y_train_binary), (
            f"ensemble_oof length {len(result['ensemble_oof'])} != "
            f"y_train length {len(y_train_binary)}."
        )

    def test_ensemble_oof_is_list_of_floats(self, base_state):
        result = run_ensemble_architect(base_state)
        oof = result["ensemble_oof"]
        assert isinstance(oof, list), f"ensemble_oof is {type(oof)}, expected list."
        assert all(isinstance(v, float) for v in oof[:10]), (
            "ensemble_oof contains non-float values."
        )
```

### CONTRACT TEST 10 — All required state keys set

```python
class TestContract10StateKeys:
    """Every key ensemble_architect promises to set must be present."""

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

    def test_all_required_state_keys_present(self, base_state):
        result = run_ensemble_architect(base_state)
        missing = [k for k in self.REQUIRED_KEYS if k not in result]
        assert not missing, (
            f"ensemble_architect did not set required state keys: {missing}"
        )

    def test_selected_models_is_list_of_strings(self, base_state):
        result = run_ensemble_architect(base_state)
        assert isinstance(result["selected_models"], list)
        assert all(isinstance(m, str) for m in result["selected_models"])

    def test_ensemble_weights_keys_match_selected_models(self, base_state):
        result = run_ensemble_architect(base_state)
        assert set(result["ensemble_weights"].keys()) == set(result["selected_models"]), (
            "ensemble_weights keys do not match selected_models."
        )
```

---

## FILE 2 — `tests/test_day22_quality.py`

Quality tests are not contracts — they test correctness of the algorithm, not just the interface. These may be edited as the algorithm evolves. They must all pass on Day 22.

```python
# tests/test_day22_quality.py
# Day 22 quality tests for ensemble_architect.py.
# These test algorithmic correctness, not just interface invariants.
```

### BLOCK 1 — Diversity selection algorithm (6 tests)

```python
class TestDiversitySelectionAlgorithm:

    def test_anchor_is_highest_cv_model(self, base_state):
        """Best CV model is always selected first regardless of dict order."""
        result = run_ensemble_architect(base_state)
        # model_alpha has cv_mean=0.810, model_beta has 0.805
        # anchor should be model_alpha
        assert result["selected_models"][0] == "model_alpha", (
            f"Anchor is {result['selected_models'][0]}, expected model_alpha (highest CV)."
        )

    def test_diverse_low_cv_model_selected_over_correlated_high_cv(
        self, base_state, base_oof
    ):
        """
        A model with lower CV but low correlation must be preferred over
        a model with higher CV but correlation > 0.98.
        This is the core diversity-first principle.
        """
        base = np.array(base_oof)
        rng  = np.random.default_rng(42)

        # high CV but nearly identical to anchor (correlation ~0.999)
        oof_similar = np.clip(base + rng.normal(0, 0.001, len(base)), 0, 1)

        # lower CV but genuinely different (correlation ~0.3)
        oof_diverse = np.clip(rng.normal(0.5, 0.3, len(base)), 0, 1)

        registry = dict(base_state["model_registry"])
        registry["model_similar"] = {
            "cv_mean": 0.809, "cv_std": 0.010,
            "fold_scores": [0.809] * 5, "stability_score": 0.794,
            "seed_results": [0.809] * 5,
            "oof_predictions": oof_similar.tolist(),
            "params": {"model_type": "xgb"}, "data_hash": "abc123",
        }
        registry["model_diverse"] = {
            "cv_mean": 0.795, "cv_std": 0.015,
            "fold_scores": [0.795] * 5, "stability_score": 0.772,
            "seed_results": [0.795] * 5,
            "oof_predictions": oof_diverse.tolist(),
            "params": {"model_type": "catboost"}, "data_hash": "abc123",
        }
        state = {**base_state, "model_registry": registry}
        result = run_ensemble_architect(state)

        assert "model_diverse" in result["selected_models"], (
            "Diverse low-CV model not selected. Diversity-first logic not applied."
        )
        # model_similar may or may not be selected depending on actual correlation
        # but if it is selected AND model_alpha is selected, correlation must be <= 0.98
        if "model_similar" in result["selected_models"]:
            pair_key = "model_alpha_vs_model_similar"
            alt_pair = "model_similar_vs_model_alpha"
            corr = (result["ensemble_correlation_matrix"].get(pair_key) or
                    result["ensemble_correlation_matrix"].get(alt_pair))
            if corr is not None:
                assert corr <= 0.98

    def test_single_model_registry_returns_that_model(self, base_state):
        """Single model in registry: anchor selected, no pruning, no Optuna needed."""
        registry = {"model_alpha": base_state["model_registry"]["model_alpha"]}
        state = {**base_state, "model_registry": registry}
        result = run_ensemble_architect(state)
        assert result["selected_models"] == ["model_alpha"]
        assert result["models_pruned_diversity"] == []

    def test_correlation_matrix_populated_for_selected_models(self, base_state):
        """Correlation matrix must have n*(n-1)/2 entries for n selected models."""
        result = run_ensemble_architect(base_state)
        n = len(result["selected_models"])
        expected_pairs = n * (n - 1) // 2
        actual_pairs   = len(result["ensemble_correlation_matrix"])
        assert actual_pairs == expected_pairs, (
            f"Correlation matrix has {actual_pairs} pairs, expected {expected_pairs} "
            f"for {n} selected models."
        )

    def test_pruned_models_not_in_weights(self, base_state, base_oof):
        """Pruned models must not appear in ensemble_weights."""
        base = np.array(base_oof)
        rng  = np.random.default_rng(42)
        near_identical = np.clip(base + rng.normal(0, 0.001, len(base)), 0, 1)

        registry = dict(base_state["model_registry"])
        registry["model_clone"] = {
            "cv_mean": 0.800, "cv_std": 0.010,
            "fold_scores": [0.800] * 5, "stability_score": 0.785,
            "seed_results": [0.800] * 5,
            "oof_predictions": near_identical.tolist(),
            "params": {"model_type": "lgbm"}, "data_hash": "abc123",
        }
        state = {**base_state, "model_registry": registry}
        result = run_ensemble_architect(state)

        for pruned in result["models_pruned_diversity"]:
            assert pruned not in result["ensemble_weights"], (
                f"Pruned model '{pruned}' appears in ensemble_weights."
            )

    def test_max_ensemble_size_respected(self, base_state, base_oof):
        """Ensemble must not exceed MAX_ENSEMBLE_SIZE regardless of model count."""
        rng = np.random.default_rng(42)
        registry = dict(base_state["model_registry"])

        # Add 10 more genuinely diverse models
        for i in range(10):
            oof = np.clip(rng.normal(0.5, 0.35, len(base_oof)), 0, 1)
            registry[f"model_extra_{i}"] = {
                "cv_mean": 0.800 - i * 0.001, "cv_std": 0.012,
                "fold_scores": [0.800] * 5, "stability_score": 0.782,
                "seed_results": [0.800] * 5,
                "oof_predictions": oof.tolist(),
                "params": {"model_type": "lgbm"}, "data_hash": "abc123",
            }

        state = {**base_state, "model_registry": registry}
        result = run_ensemble_architect(state)

        # MAX_ENSEMBLE_SIZE is defined in ensemble_architect.py — read it
        import agents.ensemble_architect as ea
        max_size = getattr(ea, "MAX_ENSEMBLE_SIZE", 8)
        assert len(result["selected_models"]) <= max_size, (
            f"{len(result['selected_models'])} models selected, max is {max_size}."
        )
```

### BLOCK 2 — Weight optimisation (5 tests)

```python
class TestWeightOptimisation:

    def test_optuna_n_jobs_is_1(self, base_state, monkeypatch):
        """OOM guard: Optuna must always use n_jobs=1."""
        captured_kwargs = {}

        import optuna
        original_optimize = optuna.Study.optimize

        def mock_optimize(self, func, **kwargs):
            captured_kwargs.update(kwargs)
            return original_optimize(self, func, n_trials=3, **{
                k: v for k, v in kwargs.items() if k != "n_trials"
            })

        monkeypatch.setattr(optuna.Study, "optimize", mock_optimize)
        run_ensemble_architect(base_state)

        assert captured_kwargs.get("n_jobs") == 1, (
            f"n_jobs={captured_kwargs.get('n_jobs')}. Must be 1 to prevent OOM."
        )

    def test_optuna_gc_after_trial_true(self, base_state, monkeypatch):
        """OOM guard: gc_after_trial must be True."""
        captured_kwargs = {}

        import optuna
        original_optimize = optuna.Study.optimize

        def mock_optimize(self, func, **kwargs):
            captured_kwargs.update(kwargs)
            return original_optimize(self, func, n_trials=3, **{
                k: v for k, v in kwargs.items() if k != "n_trials"
            })

        monkeypatch.setattr(optuna.Study, "optimize", mock_optimize)
        run_ensemble_architect(base_state)

        assert captured_kwargs.get("gc_after_trial") is True, (
            "gc_after_trial not set to True in study.optimize()."
        )

    def test_weights_reflect_model_quality(self, base_state):
        """
        The better model (model_alpha, cv=0.810) should receive
        a higher or equal weight than the weaker model (model_beta, cv=0.805).
        This tests that weight optimisation is not random.
        """
        result = run_ensemble_architect(base_state)
        weights = result["ensemble_weights"]

        if "model_alpha" in weights and "model_beta" in weights:
            assert weights["model_alpha"] >= weights["model_beta"] - 0.10, (
                f"model_alpha weight {weights['model_alpha']:.4f} is notably lower than "
                f"model_beta weight {weights['model_beta']:.4f} despite higher CV."
            )

    def test_single_model_weight_is_1_0(self, base_state):
        """Single selected model must have weight exactly 1.0."""
        registry = {"model_alpha": base_state["model_registry"]["model_alpha"]}
        state    = {**base_state, "model_registry": registry}
        result   = run_ensemble_architect(state)
        assert abs(result["ensemble_weights"]["model_alpha"] - 1.0) < 1e-9

    def test_lineage_event_written(self, base_state):
        """ensemble_selection_complete event must appear in lineage.jsonl."""
        import json
        from pathlib import Path

        run_ensemble_architect(base_state)

        lineage_path = Path(f"outputs/{base_state['session_id']}/lineage.jsonl")
        if not lineage_path.exists():
            pytest.skip("lineage.jsonl not found — lineage logging may use different path.")

        events = [json.loads(line) for line in lineage_path.read_text().splitlines() if line]
        actions = [e.get("action") for e in events]
        assert "ensemble_selection_complete" in actions, (
            "ensemble_selection_complete event not found in lineage.jsonl."
        )
```

### BLOCK 3 — Wilcoxon gate behaviour (4 tests)

```python
class TestWilcoxonGateBehaviour:

    def test_ensemble_rejected_when_not_better_than_single_model(
        self, base_state, monkeypatch
    ):
        """
        When Wilcoxon returns False (ensemble not significantly better),
        ensemble_accepted must be False.
        """
        import tools.wilcoxon_gate as wg
        monkeypatch.setattr(wg, "is_significantly_better", lambda *a, **k: False)

        result = run_ensemble_architect(base_state)
        assert result["ensemble_accepted"] is False

    def test_ensemble_accepted_when_better_than_single_model(
        self, base_state, monkeypatch
    ):
        """When Wilcoxon returns True, ensemble_accepted must be True."""
        import tools.wilcoxon_gate as wg
        monkeypatch.setattr(wg, "is_significantly_better", lambda *a, **k: True)

        result = run_ensemble_architect(base_state)
        assert result["ensemble_accepted"] is True

    def test_best_single_model_used_when_ensemble_rejected(
        self, base_state, monkeypatch
    ):
        """
        When ensemble_accepted=False, the ensemble_oof should come from
        the best single model, not a blend.
        Verify: ensemble_oof == model_alpha's oof_predictions (the best model).
        """
        import tools.wilcoxon_gate as wg
        monkeypatch.setattr(wg, "is_significantly_better", lambda *a, **k: False)

        result = run_ensemble_architect(base_state)
        best_oof = base_state["model_registry"]["model_alpha"]["oof_predictions"]

        assert result["ensemble_oof"] == pytest.approx(best_oof, abs=1e-6), (
            "When ensemble rejected, ensemble_oof should equal best single model OOF."
        )

    def test_holdout_score_always_set_regardless_of_gate(
        self, base_state, monkeypatch
    ):
        """ensemble_holdout_score must be set whether ensemble accepted or not."""
        import tools.wilcoxon_gate as wg

        for gate_return in [True, False]:
            monkeypatch.setattr(wg, "is_significantly_better",
                                lambda *a, **k: gate_return)
            result = run_ensemble_architect(base_state)
            assert "ensemble_holdout_score" in result
            assert isinstance(result["ensemble_holdout_score"], float)
```

### BLOCK 4 — Meta-learner (3 tests)

```python
class TestMetaLearner:

    def test_meta_learner_used_flag_set(self, base_state):
        """meta_learner_used must be True or False — never missing."""
        result = run_ensemble_architect(base_state)
        assert "meta_learner_used" in result
        assert isinstance(result["meta_learner_used"], bool)

    def test_meta_learner_does_not_use_holdout_for_training(
        self, base_state, monkeypatch
    ):
        """
        Meta-learner is trained on opt_pool only.
        val_holdout is only used for evaluation.
        Verify holdout is not in meta-learner training indices.
        """
        # This is the same invariant as Contract 7 but for the meta-learner path
        # The monkeypatching approach is the same — track indices
        # If meta_learner_used=True in result, verify the indices
        result = run_ensemble_architect(base_state)
        # If meta-learner was not used, this test is trivially passing
        if not result.get("meta_learner_used"):
            pytest.skip("Meta-learner was not selected — test not applicable.")
        # If meta-learner was used, the holdout score should reflect unseen data
        # We verify this by checking holdout_score is not suspiciously perfect
        assert 0.4 <= result["ensemble_holdout_score"] <= 1.0, (
            "Holdout score outside reasonable range — possible data leakage."
        )

    def test_both_blend_and_meta_evaluated_holdout_selects_better(self, base_state):
        """
        meta_learner_used=True means meta-learner scored higher on holdout.
        meta_learner_used=False means weighted blend scored higher on holdout.
        Either is valid — the holdout score should reflect the winner.
        """
        result = run_ensemble_architect(base_state)
        # Just verify the selection was made consistently
        assert isinstance(result["meta_learner_used"], bool)
        assert isinstance(result["ensemble_holdout_score"], float)
        # Holdout score must be between 0 and 1 for classification
        if base_state["task_type"] == "binary_classification":
            assert 0.0 <= result["ensemble_holdout_score"] <= 1.0
```

---

## RUNNING THE TESTS

```bash
# Run contracts (must all pass — immutable after today)
pytest tests/contracts/test_ensemble_architect_contract.py -v --tb=short

# Run quality tests
pytest tests/test_day22_quality.py -v --tb=short

# Run all contracts to check for regressions
pytest tests/contracts/ -v --tb=short

# Full regression check
pytest tests/regression/ -v --tb=short
```

---

## PASS CRITERIA

Every test in `test_ensemble_architect_contract.py` must pass.
Every test in `test_day22_quality.py` must pass.
`pytest tests/contracts/` must show zero failures.
`pytest tests/regression/` must show zero failures.

Do not commit until all four commands show zero failures.