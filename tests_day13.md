# Professor Agent — Day 13 Test Specification
**Status: IMMUTABLE after Day 13** | **48 tests**

```bash
pytest tests/test_day13_quality.py -v --tb=short 2>&1 | tee tests/logs/day13_quality.log
```

---

## BLOCK 1 — COLUMN ORDER ENFORCEMENT (16 tests)
**Class:** `TestColumnOrderEnforcement`

The silent bug: `X_test` misaligned with model. Shape matches. Predictions wrong. LB collapses. No error.

---

### TEST 1.1 — `test_feature_order_written_to_metrics_json`
**Bug:** `feature_order` key missing from `metrics.json` — submit node cannot enforce order.

Run `ml_optimizer` on a toy dataset. Load `metrics.json`. Assert `"feature_order"` key exists and is a non-empty list of strings.

---

### TEST 1.2 — `test_feature_order_matches_training_columns_exactly`
**Bug:** `feature_order` written as sorted list, not insertion order (common mistake with `sorted(df.columns)`).

Train on DataFrame with columns in order `["b_col", "a_col", "c_col"]`. Assert `metrics["feature_order"] == ["b_col", "a_col", "c_col"]`. Not alphabetically sorted.

---

### TEST 1.3 — `test_feature_order_stored_in_state`
After `ml_optimizer` completes: assert `state["feature_order"]` is non-empty and matches `metrics.json` feature_order exactly.

---

### TEST 1.4 — `test_submit_loads_feature_order_from_metrics_not_state`
**Bug:** Submit node reads `feature_order` from `state["feature_order"]` (stale between sessions) instead of `metrics.json` (authoritative).

Delete `state["feature_order"]`. Assert `build_submission()` still works by loading from `metrics.json`. If it raises `KeyError`, it's reading state instead of disk.

---

### TEST 1.5 — `test_submit_selects_columns_in_training_order`
Test DataFrame has columns in different order than training: `["c_col", "a_col", "b_col"]`. Training order was `["b_col", "a_col", "c_col"]`. Assert `test_subset.columns == ["b_col", "a_col", "c_col"]` after `select(feature_order)`.

---

### TEST 1.6 — `test_assert_fires_when_polars_select_returns_wrong_order`
**Bug:** `.select()` call correct but the hard `assert` was removed to "clean up the code".

Monkeypatch `test_df.select` to return columns in wrong order despite correct call. Assert `AssertionError` is raised before `.to_numpy()`. Without the assert, prediction proceeds silently on wrong-order data.

---

### TEST 1.7 — `test_submit_raises_value_error_on_missing_test_column`
Training feature `"important_feature"` absent from test DataFrame. Assert `ValueError` raised (wrapping `ColumnNotFoundError`). Message must name the missing column. Must not produce a prediction.

---

### TEST 1.8 — `test_submit_raises_file_not_found_when_metrics_json_missing`
Delete `metrics.json` before calling `build_submission()`. Assert `FileNotFoundError` with message naming `metrics.json`. Never produces predictions from stale state.

---

### TEST 1.9 — `test_submit_raises_value_error_when_feature_order_missing_from_metrics`
Write `metrics.json` without the `"feature_order"` key (old format). Assert `ValueError` with message mentioning `feature_order` and instructing re-run.

---

### TEST 1.10 — `test_submit_with_correct_column_order_produces_prediction`
Happy path: test DataFrame has same columns in same order as training. Assert `build_submission()` returns a DataFrame with `id` and prediction column. No errors.

---

### TEST 1.11 — `test_feature_order_preserved_across_polars_read`
**Bug:** `pl.read_csv()` may reorder columns internally on some platforms.

Write train DataFrame to CSV, read back with `pl.read_csv()`. Assert column order matches original. If it doesn't, the `select()` call saves us — but this test documents the assumption.

---

### TEST 1.12 — `test_feature_order_excludes_target_column`
**Bug:** Target column included in `feature_order` — prediction time raises `ColumnNotFoundError` because test.csv has no target column.

Assert `state["feature_order"]` does not contain the target column name (e.g. `"Survived"`, `"target"`). `feature_order` must be feature columns only.

---

### TEST 1.13 — `test_feature_order_excludes_id_columns`
Assert `state["feature_order"]` does not contain columns identified as IDs (e.g. `"PassengerId"`, `"id"`, `"row_id"`). IDs are excluded from training features — test selection must match.

---

### TEST 1.14 — `test_contract_feature_order_saved`
Contract test (permanent): run full pipeline through ml_optimizer. Assert `metrics.json` exists AND contains `feature_order`. This test belongs in `test_submit_column_order_contract.py`.

---

### TEST 1.15 — `test_contract_submit_raises_on_missing_column`
Contract test: submit with a test DataFrame missing one training column. Assert `ValueError`. Never silent wrong prediction.

---

### TEST 1.16 — `test_contract_submit_raises_on_wrong_column_order_after_polars_select_bypass`
Contract test: monkeypatch `.select()` to no-op (returns original wrong-order df). Assert `AssertionError` from the hard assert catches it. The assert is the last line of defence.

---

## BLOCK 2 — DATA HASH VALIDATION (12 tests)
**Class:** `TestDataHashValidation`

The silent bug: Kaggle releases corrected data. Registry has models from both versions. Ensemble blends them. Submission wrong. No error.

---

### TEST 2.1 — `test_validation_passes_when_all_hashes_match`
All registry entries have `data_hash="abc123"`. Current `state["data_hash"]="abc123"`. Assert validation completes without raising. All models retained.

---

### TEST 2.2 — `test_validation_logs_warning_on_hash_mismatch`
Registry has 2 models with `data_hash="abc123"` and 1 model with `data_hash="xyz789"`. Assert `logger.warning` was called with a message containing both hash values. Warning must name the mismatch explicitly.

---

### TEST 2.3 — `test_validation_filters_to_current_hash_only`
Mixed registry: 2 models on `"abc123"`, 1 on `"xyz789"`. `state["data_hash"]="abc123"`. After validation: assert `state["model_registry"]` contains exactly 2 models (the current-hash ones). The stale model is excluded.

---

### TEST 2.4 — `test_validation_raises_when_filtered_registry_empty`
All registry entries have `data_hash="abc123"`. `state["data_hash"]="xyz789"` (all models are stale). Assert `ValueError` raised with message containing `"retrain required"`. Must not attempt blending with zero models.

---

### TEST 2.5 — `test_validation_raises_when_registry_empty`
`model_registry={}`. Assert `ValueError` raised before any hash checking. Empty registry is caught first.

---

### TEST 2.6 — `test_validation_degrades_gracefully_when_state_hash_none`
`state["data_hash"]=None`. Assert validation logs a `WARNING` and continues (does not filter, does not raise). Some competitions don't track data_hash — must not crash.

---

### TEST 2.7 — `test_validation_degrades_gracefully_when_registry_entry_missing_hash`
One registry entry has no `data_hash` key (old model from before Day 4). Assert validation logs a warning about incomplete entries and excludes them from uniqueness check. Must not raise `KeyError`.

---

### TEST 2.8 — `test_validation_event_logged_to_lineage`
After clean validation (all hashes match): assert `lineage.jsonl` contains entry with `action="data_hash_validated"` and `models_checked > 0`.

---

### TEST 2.9 — `test_blend_not_called_when_registry_empty_after_filter`
Mixed hashes, all stale after filter. Assert `_blend_models_weighted()` is never called (monkeypatch to verify call count = 0). The ValueError must halt before blending.

---

### TEST 2.10 — `test_validation_called_before_weight_computation`
**Bug:** Weight computation called before hash validation — blends stale models, then validation fires too late.

Monkeypatch `_compute_ensemble_weights()` and `_validate_data_hash_consistency()`. Assert validation is called first (call order check with `Mock` side_effect sequence).

---

### TEST 2.11 — `test_state_returned_with_filtered_registry`
After filtering: assert the state dict returned from `blend_models()` has `model_registry` containing only the current-hash models. Filtered state must propagate downstream.

---

### TEST 2.12 — `test_validation_handles_single_model_registry`
Registry with exactly 1 model. `data_hash` matches. Assert validation passes. Ensemble with 1 model is valid (just passes through that model's predictions).

---

## BLOCK 3 — WILCOXON GATE: CORRECTNESS (12 tests)
**Class:** `TestWilcoxonGate`

---

### TEST 3.1 — `test_returns_true_when_a_significantly_better`
`fold_scores_a = [0.85, 0.87, 0.86, 0.88, 0.87]`
`fold_scores_b = [0.80, 0.81, 0.80, 0.82, 0.81]`
Clear systematic improvement. Assert `is_significantly_better(a, b) == True`.

---

### TEST 3.2 — `test_returns_false_when_difference_is_noise`
`fold_scores_a = [0.800, 0.810, 0.805, 0.808, 0.803]`
`fold_scores_b = [0.802, 0.808, 0.807, 0.806, 0.804]`
Random noise. Assert `is_significantly_better(a, b) == False`. The simpler model should be kept.

---

### TEST 3.3 — `test_returns_false_when_a_is_worse`
`fold_scores_a` consistently below `fold_scores_b`. Assert returns `False`. Never returns True when A < B.

---

### TEST 3.4 — `test_returns_false_when_all_differences_zero`
`fold_scores_a == fold_scores_b` (identical scores). Assert `False`. Models are equivalent.

---

### TEST 3.5 — `test_never_raises_on_mismatched_fold_counts`
**Bug:** `len(a) != len(b)` — `wilcoxon()` raises `ValueError`. Must be caught.

`fold_scores_a = [0.85, 0.86, 0.87]`, `fold_scores_b = [0.84, 0.85]`. Assert returns `False`. Never raises.

---

### TEST 3.6 — `test_falls_back_to_mean_comparison_below_5_folds`
`fold_scores_a = [0.85, 0.86, 0.84]` (3 folds — below minimum). Assert the function falls back to mean comparison (`np.mean(a) > np.mean(b)`). Assert no exception. Assert log message mentions fallback.

---

### TEST 3.7 — `test_never_raises_when_scipy_wilcoxon_throws`
Monkeypatch `scipy.stats.wilcoxon` to raise `ValueError`. Assert `is_significantly_better()` returns a bool (True or False) — never propagates the exception. Falls back to mean comparison.

---

### TEST 3.8 — `test_p_threshold_respected`
Use real fold data where p-value is known to be ~0.04. At `p_threshold=0.05`: returns `True`. At `p_threshold=0.03`: returns `False`. The threshold parameter is honoured.

---

### TEST 3.9 — `test_gate_result_has_all_required_keys`
Call `gate_result(a, b, "challenger", "champion")`. Assert result dict has all 9 keys:
`gate_passed, selected_model, mean_a, mean_b, mean_delta, p_threshold, n_folds, model_name_a, model_name_b, reason`.

---

### TEST 3.10 — `test_gate_result_selected_model_is_b_when_gate_fails`
Gate fails (not significant). Assert `result["selected_model"] == "champion"` (the B/baseline model). Gate failure = keep existing.

---

### TEST 3.11 — `test_gate_result_selected_model_is_a_when_gate_passes`
Gate passes (significant). Assert `result["selected_model"] == "challenger"` (the A/new model).

---

### TEST 3.12 — `test_mean_delta_is_correct_sign`
`mean(a) > mean(b)`. Assert `result["mean_delta"] > 0`. `mean(a) < mean(b)`. Assert `result["mean_delta"] < 0`. Sign must be A minus B.

---

## BLOCK 4 — WILCOXON GATE: OPTIMIZER INTEGRATION (8 tests)
**Class:** `TestWilcoxonGateOptimizerIntegration`

---

### TEST 4.1 — `test_fold_scores_stored_in_trial_user_attrs`
After `study.optimize()`: for every completed trial, assert `trial.user_attrs["fold_scores"]` is a list of floats with `len == n_cv_folds`. Required for the gate to operate on actual fold-level data.

---

### TEST 4.2 — `test_gate_applied_to_every_trial_comparison`
Run optimizer with 3 trials. Assert `lineage.jsonl` contains exactly 2 `action="wilcoxon_gate_decision"` entries (one per comparison: trial 1 vs trial 2, trial 2 winner vs trial 3). Gate applied at every step.

---

### TEST 4.3 — `test_non_significant_trial_not_selected_as_best`
Set up optimizer where trial 2 is marginally better than trial 1 (delta < noise level). Assert selected trial is trial 1 (the original). Gate blocked the marginal improvement.

---

### TEST 4.4 — `test_significantly_better_trial_is_selected`
Trial 2 is clearly and significantly better than trial 1 (large delta, p << 0.05). Assert selected trial is trial 2. Gate correctly lets the improvement through.

---

### TEST 4.5 — `test_cross_model_gate_keeps_simpler_when_not_significant`
LightGBM and XGBoost have nearly identical fold scores. Assert `_select_best_model_type()` returns `"lgbm"` (simpler model). XGBoost must beat LGBM significantly to be selected.

---

### TEST 4.6 — `test_cross_model_gate_selects_complex_when_significantly_better`
XGBoost fold scores clearly better than LightGBM (large systematic delta). Assert `_select_best_model_type()` returns `"xgb"`. Gate correctly upgrades to more complex model.

---

### TEST 4.7 — `test_gate_decision_logged_with_comparison_type`
After cross-model comparison: `lineage.jsonl` entry has `"comparison_type": "cross_model"`. After Optuna trial comparison: entry has `"comparison_type"` absent or `"optuna_trial"`. Engineer must be able to distinguish gate types in post-mortem.

---

### TEST 4.8 — `test_gate_falls_back_gracefully_when_fold_scores_unavailable`
**Bug:** Old model in registry has no `fold_scores` in `user_attrs` (pre-Day 13 model). Gate function receives empty lists.

Assert `_select_best_model_type()` handles empty fold lists gracefully — falls back to mean CV score comparison without raising. Logs a warning. Must not crash the optimizer.

---

## THE 5 BUGS THAT WILL DEFINITELY BE PRESENT

**Bug 1 — Test 1.2:** `feature_order` saved as `sorted(df.columns)`. Natural impulse for determinism. Wrong — must preserve insertion order. A DataFrame with columns `["b", "a", "c"]` after sorted becomes `["a", "b", "c"]`, swapping a and b.

**Bug 2 — Test 1.4:** Submit node reads from `state["feature_order"]` not `metrics.json`. State can be stale if the pipeline resumed from checkpoint, or if the session ID changed. `metrics.json` is the authoritative record written at training time.

**Bug 3 — Test 1.12:** Target column in `feature_order`. The most common Day 13 implementation error: `feature_order = list(X_train.columns)` but `X_train` still has the target column in it (feature engineering step didn't drop it before saving). Test prediction time: `ColumnNotFoundError` on the target.

**Bug 4 — Test 2.10:** `_compute_ensemble_weights()` called before `_validate_data_hash_consistency()`. LLM puts validation at the end as a "check". Must be first — you cannot allow any blending to proceed before the hash check.

**Bug 5 — Test 3.5:** `wilcoxon()` raises `ValueError` on mismatched fold counts. Natural call: `wilcoxon(fold_scores_a, fold_scores_b)`. If lengths differ, scipy raises. The gate must catch this and return `False` conservatively.

---

## CONTRACT FILES (new, immutable after Day 13)

**`tests/contracts/test_submit_column_order_contract.py`** (3 tests):
- `feature_order` in `metrics.json` after training
- `build_submission()` raises on missing column
- `build_submission()` raises on wrong column order (assert catches it)

**`tests/contracts/test_wilcoxon_gate_contract.py`** (4 tests):
- `is_significantly_better()` returns bool, never raises
- Returns False when fold counts differ
- Returns False when all differences zero
- Falls back to mean comparison below `MIN_FOLDS_REQUIRED` folds

---

## TOTAL: 48 + 7 CONTRACT TESTS

| Block | Tests |
|---|---|
| 1 — Column order enforcement | 16 |
| 2 — Data hash validation | 12 |
| 3 — Wilcoxon gate correctness | 12 |
| 4 — Optimizer integration | 8 |
| New column order contract | 3 |
| New Wilcoxon gate contract | 4 |
| **Total** | **55** |

---

## DEFINITION OF DONE

- [ ] 48 tests in `tests/test_day13_quality.py` — all pass
- [ ] `tests/contracts/test_submit_column_order_contract.py` — all pass, immutable
- [ ] `tests/contracts/test_wilcoxon_gate_contract.py` — all pass, immutable
- [ ] `pytest tests/regression/` — green
- [ ] `pytest tests/contracts/` — green (all previous contracts still pass)
- [ ] Manual check: confirm `metrics.json` contains `feature_order` after a real training run
- [ ] `git commit -m "Day 13: column order, hash guard, Wilcoxon gate — 48 tests green"`