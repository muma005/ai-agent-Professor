# Day 22 — Ensemble Architect + Phase 3 Regression Freeze
## Prompt for Qwen Code

---

## BEFORE YOU WRITE A SINGLE LINE

Read these files completely first. Do not start coding until you have read all of them.

```
CLAUDE.md
AGENTS.md
core/state.py
core/professor.py
agents/ml_optimizer.py          ← understand what model_registry contains
tools/wilcoxon_gate.py          ← you will use is_significantly_better()
tools/stability_validator.py    ← understand StabilityResult structure
tests/regression/test_phase1_regression.py   ← if it exists
tests/regression/test_phase2_regression.py   ← if it exists
```

After reading, write one paragraph summarising what `model_registry` contains per entry before writing any code. If you cannot describe it from reading the code, read `ml_optimizer.py` again.

---

## TASK 1 — Build `agents/ensemble_architect.py`

Build the file from scratch. The pipeline calls this agent after `ml_optimizer` completes and all model variants are in `model_registry`.

### What this agent does

1. Takes all models in `model_registry`
2. Prunes models that add no diversity (correlation > 0.98 with any already-selected model)
3. Optimises blend weights using Optuna with constraints
4. Trains a stacking meta-learner on top of the blended OOF predictions
5. Validates the ensemble on a holdout fold that was never used in weight optimisation
6. Accepts the ensemble only if Wilcoxon confirms it beats the best single model
7. Writes results to state

### Exact requirements — implement every one

**Requirement 1: Data hash validation first**

Before anything else, validate that all models in `model_registry` share the same `data_hash` as `state["data_hash"]`. If any model has a different hash, log a WARNING, remove it from the working registry, and continue with the remaining models. If the remaining registry is empty after filtering, raise `ValueError("No models match current data_hash. Retrain required.")`.

This check must be the first thing that runs. Nothing else runs before it.

**Requirement 2: OOF validation**

After hash validation, verify every remaining model has `oof_predictions` in its registry entry and that `len(oof_predictions) == len(y_train)`. If any model fails this check, raise `ValueError` naming the model. Do not silently continue with mismatched shapes.

**Requirement 3: Diversity pruning — correlation > 0.98 threshold**

Implement greedy diversity selection:
- Sort models by `cv_mean` descending. The best model is the anchor.
- For each remaining model in order: compute Pearson correlation between its `oof_predictions` and the `oof_predictions` of every already-selected model.
- If correlation with any selected model exceeds 0.98, reject this model. Log which model was rejected and what its maximum correlation was.
- If correlation is <= 0.98 with all selected models, add it to the ensemble.
- Minimum ensemble size is 1 (the anchor always survives).

After pruning, the final ensemble must not contain any pair of models with correlation > 0.98. This is a hard constraint, not a soft preference.

**Requirement 4: Holdout split for weight validation**

Before weight optimisation, split the training data:
- 80% used for weight optimisation (call this `opt_pool`)
- 20% held out for ensemble validation (call this `val_holdout`)
- Split must be stratified for classification, random for regression
- Fixed seed: 42
- `val_holdout` is never used during weight optimisation. Never.

The OOF predictions from each selected model must be split the same way so the shapes align.

**Requirement 5: Constrained Optuna weight optimisation**

Use Optuna to find optimal blend weights. The objective is to maximise ensemble CV score on `opt_pool`.

Constraints that must be enforced on every trial:
- Weights sum to 1.0. Implement this by normalising: `weights = softmax(raw_params)`.
- No individual weight below 0.05. After softmax, clip any weight below 0.05 to 0.05 and renormalise.
- Maximum number of free parameters = `max(1, len(selected_models) // 100)` — for small ensembles this is always 1, meaning Optuna searches a single temperature parameter and weights are derived from model CV scores scaled by that temperature.

Use `n_trials=50`. Use `n_jobs=1`. Use `gc_after_trial=True`. These are not negotiable — same OOM rules as `ml_optimizer`.

**Requirement 6: Stacking meta-learner**

After finding optimal weights:
- Stack the OOF predictions of selected models as columns (shape: n_samples × n_selected_models)
- Train a `LogisticRegression(C=0.1, max_iter=1000)` for classification or `Ridge(alpha=10.0)` for regression as the meta-learner
- Use 5-fold CV for the meta-learner to avoid leakage
- The meta-learner is an alternative to the weighted blend — not a replacement. Keep both and select whichever scores higher on `val_holdout`.

**Requirement 7: Wilcoxon validation gate**

After computing ensemble OOF predictions on `opt_pool`:
- Get the best single model's `fold_scores` from `model_registry`
- Compute ensemble fold scores using the same CV folds
- Call `is_significantly_better(ensemble_fold_scores, best_single_fold_scores)` from `tools/wilcoxon_gate.py`
- If the gate returns `False`: log a WARNING that ensemble does not significantly beat best single model, set `state["ensemble_accepted"] = False`, and use the best single model's predictions instead
- If the gate returns `True`: set `state["ensemble_accepted"] = True`

**Requirement 8: Validate on holdout**

After selecting ensemble vs single model:
- Score whichever was selected on `val_holdout`
- Log: holdout score, whether ensemble or single model was used, number of models in ensemble, weights
- This holdout score goes into state as `state["ensemble_holdout_score"]`

**Requirement 9: State outputs**

Set all of these in state before returning:

```python
state["selected_models"]           # list[str] — model names in final ensemble
state["ensemble_weights"]          # dict[str, float] — weights summing to 1.0
state["ensemble_oof"]              # list[float] — final blended OOF predictions
state["ensemble_holdout_score"]    # float — score on val_holdout
state["ensemble_accepted"]         # bool — True if ensemble beat single model
state["ensemble_correlation_matrix"] # dict — pairwise Pearson correlations of selected models
state["models_pruned_diversity"]   # list[str] — models removed for correlation > 0.98
state["meta_learner_used"]         # bool — True if meta-learner beat weighted blend
```

**Requirement 10: Lineage**

Call `log_event()` once at the end with:
```python
{
    "action": "ensemble_selection_complete",
    "agent": "ensemble_architect",
    "n_candidates": len(model_registry),
    "n_selected": len(selected_models),
    "n_pruned_diversity": len(models_pruned_diversity),
    "ensemble_accepted": ensemble_accepted,
    "ensemble_holdout_score": ensemble_holdout_score,
    "weights": ensemble_weights,
}
```

### What not to do

- Do not read raw CSV data directly. Use what `ml_optimizer` put into `model_registry` and `state`.
- Do not hardcode the metric. Read it from `state["evaluation_metric"]`.
- Do not silently swallow exceptions. If a model is missing a required field, raise with a clear message naming the model and the field.
- Do not run Optuna with `n_jobs != 1`.

---

## TASK 2 — Write `tests/contracts/test_ensemble_architect_contract.py`

This file is immutable after Day 22. Write it correctly the first time.

The contract tests verify the interface and invariants of `ensemble_architect.py`. They do not test performance — they test correctness of the process.

### Contract 1: Diversity pruning runs before weight optimisation

Inject two models with OOF Pearson correlation = 0.99. Assert that only one of them appears in `state["selected_models"]` after the agent runs. The correlation matrix must not contain any pair with correlation > 0.98.

Prove the order by monkeypatching: record when diversity pruning was called and when Optuna was called. Assert diversity pruning timestamp < Optuna start timestamp.

### Contract 2: Weights sum to 1.0

After any run with any number of selected models, assert:
```python
assert abs(sum(state["ensemble_weights"].values()) - 1.0) < 1e-6
```

### Contract 3: No weight below 0.05

After any run, assert every value in `state["ensemble_weights"]` is >= 0.05.

### Contract 4: Holdout never used in weight optimisation

Verify that the indices used for `val_holdout` do not appear in the data passed to Optuna's objective function. Do this by tracking which sample indices each uses. There must be zero overlap.

### Contract 5: Wilcoxon gate applied

Monkeypatch `is_significantly_better` to track calls. Run the agent. Assert it was called exactly once. Assert `state["ensemble_accepted"]` is set (True or False — either is valid, but it must be set).

### Contract 6: Ensemble OOF length matches training data length

Assert `len(state["ensemble_oof"]) == len(y_train)`.

### Contract 7: Hash mismatch raises ValueError

Put two models in `model_registry` with different `data_hash` values than `state["data_hash"]`. Assert `ValueError` is raised with a message containing "retrain required".

### Contract 8: Missing OOF raises ValueError

Put a model in `model_registry` with no `oof_predictions` key. Assert `ValueError` is raised naming that model.

### Test structure

```python
# tests/contracts/test_ensemble_architect_contract.py
#
# CONTRACT: agents/ensemble_architect.py
# Written: Day 22. IMMUTABLE — never edit after Day 22.
#
# INVARIANTS:
#   1. Diversity pruning runs before weight optimisation
#   2. Weights sum to 1.0 (tolerance 1e-6)
#   3. No weight below 0.05
#   4. val_holdout indices never appear in Optuna objective
#   5. Wilcoxon gate called exactly once
#   6. ensemble_oof length == len(y_train)
#   7. Hash mismatch raises ValueError("retrain required")
#   8. Missing oof_predictions raises ValueError naming the model

import pytest
import numpy as np

# ... (implement 8 contract tests)
```

Each test class name must be `TestEnsembleArchitectContract`. Each test method must have a docstring explaining what invariant it protects and why.

---

## TASK 3 — Write `tests/regression/test_phase3_regression.py`

**Only write this file after Phase 3 gate has passed and you have a gate result file.**

Check whether `tests/phase3_gate_results/` contains any `gate_result.json`. If it does not exist, do not create this file. Print: "Phase 3 gate has not passed yet. Do not freeze regression until gate passes."

If the gate result exists, read it and use the actual values to set the floor thresholds.

```python
# tests/regression/test_phase3_regression.py
#
# PHASE 3 REGRESSION FREEZE
# Written: [DATE]
# Commit hash at freeze: [git rev-parse HEAD]
# Gate session: [session_id from gate_result.json]
# Gate public score: [public_score from gate_result.json]
# Gate CV score: [cv_score from gate_result.json]
#
# IMMUTABLE. Never edit. Never relax thresholds.
# If a test fails: fix the underlying capability. Never fix the test.
```

### Frozen test 1: Phase 1 and 2 CV floors still hold

```python
def test_phase1_and_phase2_cv_floors_still_hold(self, benchmark_state):
    phase2_baseline = json.loads(Path("tests/regression/phase2_baseline.json").read_text())
    phase2_floor = float(phase2_baseline["cv_mean"])
    current_cv = float(benchmark_state["cv_mean"])
    assert current_cv >= phase2_floor - 0.002, (
        f"REGRESSION: CV {current_cv:.5f} dropped below Phase 2 floor "
        f"{phase2_floor:.5f}. Phase 3 additions regressed baseline performance."
    )
```

### Frozen test 2: CV score floor from gate result

Set the floor to `gate_cv_score - 0.020`. Read `gate_cv_score` from the actual gate result file. Do not hardcode a number — read it at test definition time and embed the actual number in the assertion message.

### Frozen test 3: Null importance filter is running and removing features

```python
def test_null_importance_filter_removes_features(self, benchmark_state):
    n_final = benchmark_state.get("n_features_final", 0)
    assert 5 <= n_final <= 200, (
        f"REGRESSION: {n_final} features survived null importance. "
        "Either filter is not running (>200) or is too aggressive (<5)."
    )
    dropped = (benchmark_state.get("stage1_drop_count", 0) +
               benchmark_state.get("stage2_drop_count", 0))
    assert dropped > 0, (
        "REGRESSION: Null importance filter dropped 0 features. Filter is not running."
    )
```

### Frozen test 4: Optuna stability ranking beats peak ranking

```python
def test_stability_ranking_beats_peak_ranking(self, benchmark_state):
    registry = benchmark_state.get("model_registry", {})
    if len(registry) < 2:
        pytest.skip("Need at least 2 models to compare stability vs peak.")
    winner = max(registry.values(), key=lambda e: e.get("stability_score", 0.0))
    assert "seed_results" in winner, "REGRESSION: Winner missing seed_results."
    assert len(winner["seed_results"]) == 5, (
        f"REGRESSION: Winner has {len(winner['seed_results'])} seed results, expected 5."
    )
    computed_stability = (
        float(np.mean(winner["seed_results"])) -
        1.5 * float(np.std(winner["seed_results"]))
    )
    assert abs(winner["stability_score"] - computed_stability) < 1e-5, (
        "REGRESSION: stability_score formula changed. "
        f"Expected mean - 1.5*std = {computed_stability:.6f}, "
        f"got {winner['stability_score']:.6f}."
    )
```

### Frozen test 5: All 4 core Critic vectors fire on injected failures

Test each of these four vectors individually by injecting a known failure:
- `shuffled_target` vector: inject a target-derived feature → must return CRITICAL
- `preprocessing_audit` vector: inject `fit_transform(X)` before split → must flag it
- `pr_curve_imbalance` vector: inject a model predicting all majority class on imbalanced data → must return HIGH or CRITICAL
- `temporal_leakage` vector: inject a feature that is the row index with `has_dates=True` → must flag it

Each test must prove the vector is live. A vector that does not fire on a known injection is broken.

### Frozen test 6: Wilcoxon gate rejects noise

```python
def test_wilcoxon_gate_rejects_noise_level_difference(self):
    from tools.wilcoxon_gate import is_significantly_better
    a = [0.8012, 0.8009, 0.8015, 0.8011, 0.8013]
    b = [0.8010, 0.8012, 0.8013, 0.8009, 0.8014]
    assert is_significantly_better(a, b) is False, (
        "REGRESSION: Wilcoxon gate approved noise-level difference. "
        "Complex models being selected on lucky seeds."
    )
```

### Frozen test 7: Ensemble architect diversity pruning enforced

```python
def test_ensemble_architect_prunes_correlated_models(self, benchmark_state):
    corr_matrix = benchmark_state.get("ensemble_correlation_matrix", {})
    for pair, corr in corr_matrix.items():
        assert corr <= 0.98, (
            f"REGRESSION: Model pair {pair} has correlation {corr:.4f} > 0.98 "
            "in the final ensemble. Diversity pruning is not working."
        )
```

---

## COMMIT SEQUENCE

Make one commit per completed task. Do not bundle them.

```
git commit -m "Day 22: agents/ensemble_architect.py — diversity pruning, constrained weights, Wilcoxon gate"
git commit -m "Day 22: tests/contracts/test_ensemble_architect_contract.py — 8 immutable contracts"
git commit -m "Day 22: tests/regression/test_phase3_regression.py — frozen after Phase 3 gate"
```

---

## VERIFICATION BEFORE EACH COMMIT

Run these before committing each task:

```bash
# After Task 1:
python -c "from agents.ensemble_architect import run_ensemble_architect; print('Import OK')"
pytest tests/contracts/test_ensemble_architect_contract.py -v --tb=short

# After Task 2:
pytest tests/contracts/ -v --tb=short   # all contracts must pass

# After Task 3 (only if gate has passed):
pytest tests/regression/ -v --tb=short  # all regression tests must pass
```

If any test fails, fix it before committing. Do not commit a failing test.