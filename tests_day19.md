# Professor Agent — Day 19 Test Specification
**Status: IMMUTABLE after Day 19** | **50 tests + 8 contract tests**
```bash
pytest tests/test_day19_quality.py -v --tb=short 2>&1 | tee tests/logs/day19_quality.log
pytest tests/contracts/test_ml_optimizer_optuna_contract.py -v
```

---

## BLOCK 1 — PREDICTION CALIBRATION (16 tests)
**Class:** `TestPredictionCalibration`

The bugs: `cv='prefit'` not set — CalibratedClassifierCV refits the base model on the calibration fold, discarding the trained model. Calibration fold overlaps with CV folds (split happens after CV, not before). `brier_score_loss` computed on training data instead of calibration fold. `sigmoid` and `isotonic` boundary is `<= 1000` instead of `< 1000`.

---

### TEST 1.1 — `test_calibration_triggered_for_log_loss`
`metric="log_loss"`. Run `_train_and_optionally_calibrate()`. Assert `calibration_info["is_calibrated"] == True`. Assert `calibration_info["calibration_method"] in ("sigmoid", "isotonic")`.

---

### TEST 1.2 — `test_calibration_triggered_for_brier_score`
`metric="brier_score"`. Assert `is_calibrated == True`.

---

### TEST 1.3 — `test_calibration_triggered_for_cross_entropy`
`metric="cross_entropy"`. Assert `is_calibrated == True`.

---

### TEST 1.4 — `test_calibration_not_triggered_for_auc`
`metric="auc"`. Assert `is_calibrated == False`. Assert `calibration_method == "none"`.

---

### TEST 1.5 — `test_calibration_not_triggered_for_rmse`
`metric="rmse"`. Assert `is_calibrated == False`.

---

### TEST 1.6 — `test_sigmoid_method_for_small_calibration_set`
`n_calib_samples = 800` (< 1000). Assert `_select_calibration_method(800) == "sigmoid"`.

---

### TEST 1.7 — `test_isotonic_method_for_large_calibration_set`
`n_calib_samples = 1000` (>= 1000). Assert `_select_calibration_method(1000) == "isotonic"`.

---

### TEST 1.8 — `test_boundary_at_1000_is_isotonic_not_sigmoid`
**Bug:** `< 1000` vs `<= 1000` — at exactly 1000 samples the boundary matters.
`_select_calibration_method(1000)` must return `"isotonic"`. `_select_calibration_method(999)` must return `"sigmoid"`. Both tested.

---

### TEST 1.9 — `test_cv_prefit_used_not_cross_val`
**Bug:** `cv='prefit'` not passed — CalibratedClassifierCV refits the model from scratch on the calibration fold. The trained LightGBM model is thrown away.

Inspect the `CalibratedClassifierCV` constructor call. Assert `cv="prefit"` is passed. Without this, calibration trains a new model — not the one you just optimised.

---

### TEST 1.10 — `test_calibration_fold_carved_out_before_cv`
**Bug:** Calibration fold split happens after CV training — calibration data may have appeared in CV training folds.

Monkeypatch `_split_calibration_fold` to record the indices it returns. Assert these indices do NOT appear in any CV fold's `train_idx`. The calibration split must happen before the CV loop begins.

---

### TEST 1.11 — `test_brier_score_computed_on_calibration_fold_not_training`
**Bug:** Brier score computed on `X_train_cv` (training data) instead of `X_calib` (held-out).

Feed calibration fold where all true labels are 1 (100% positive). The model would produce low Brier on training data but high on the calibration fold. Assert `calibration_score` reflects the calibration fold's score, not the training fold's.

---

### TEST 1.12 — `test_calibration_failure_falls_back_gracefully`
Monkeypatch `CalibratedClassifierCV.fit` to raise `ValueError`. Assert `_run_calibration()` returns the uncalibrated base model. Assert returns `(base_model, None, "none")`. Must not raise.

---

### TEST 1.13 — `test_calibration_info_stored_in_model_registry`
After `run_ml_optimizer()` with log_loss metric: assert `model_registry` entry has `is_calibrated=True`, `calibration_method` ∈ {"sigmoid", "isotonic"}, `calibration_score` is a float > 0, `calibration_n_samples > 0`.

---

### TEST 1.14 — `test_calibrated_model_produces_better_brier_than_uncalibrated`
Train on a deliberately miscalibrated dataset (predictions cluster near 0 and 1). Compare Brier score of uncalibrated vs calibrated model on held-out data. Assert calibrated model has lower Brier score. Calibration must actually improve calibration.

---

### TEST 1.15 — `test_probability_metrics_frozenset_contains_expected_values`
Assert `PROBABILITY_METRICS` contains: `"log_loss"`, `"cross_entropy"`, `"brier_score"`, `"logloss"`. Assert `"auc"` is NOT in `PROBABILITY_METRICS`. Assert `"rmse"` is NOT in `PROBABILITY_METRICS`.

---

### TEST 1.16 — `test_critic_checks_calibration_for_probability_metrics`
Run full pipeline with `metric="log_loss"` and an uncalibrated model in registry. Run red_team_critic. Assert `critic_verdict` contains a finding about calibration. Assert `finding["severity"] == "HIGH"` or `"WARNING"`.

---

## BLOCK 2 — STABILITY VALIDATOR (14 tests)
**Class:** `TestStabilityValidator`

The bugs: `stability_score = mean - 1.0 * std` (penalty is 1.0 not 1.5). `rank_by_stability` sorts ascending instead of descending (worst config selected). `run_with_seeds` raises when one seed fails (breaks the entire stability check). `seed_results` contains strings instead of floats.

---

### TEST 2.1 — `test_stability_score_formula_is_mean_minus_1_5_std`
`seed_results = [0.85, 0.83, 0.86, 0.84, 0.82]`. Mean=0.84, std=0.015. Assert `result.stability_score ≈ 0.84 - 1.5 * 0.015 = 0.8175`. Not `0.84 - 1.0 * 0.015`.

---

### TEST 2.2 — `test_stable_config_beats_variable_config_despite_lower_mean`
Config A: seed_results all 0.83 (std=0). stability_score=0.83.
Config B: seed_results [0.88, 0.78, 0.89, 0.77, 0.88]. mean≈0.84, std≈0.055. stability_score≈0.84-0.083=0.757.
Assert `rank_by_stability([A_cfg, B_cfg], [A_res, B_res])[0][0] == A_cfg`. The stable config with lower mean wins.

---

### TEST 2.3 — `test_rank_by_stability_sorts_descending`
3 configs with stability scores 0.81, 0.84, 0.78. Assert first in ranked list has stability_score=0.84. Descending order — best first.

---

### TEST 2.4 — `test_run_with_seeds_uses_default_5_seeds`
Call `run_with_seeds(config, train_fn)` with no seeds argument. Assert `len(result.seed_results) == 5`. Default seeds = [42, 7, 123, 999, 2024].

---

### TEST 2.5 — `test_run_with_seeds_handles_single_seed_failure`
Monkeypatch `train_fn` to raise on seed=123. Assert `run_with_seeds()` completes. Assert `len(result.seed_results) == 4` (one failed, four succeeded). Assert warning logged.

---

### TEST 2.6 — `test_run_with_seeds_handles_all_seeds_failing`
All seeds raise. Assert returns `StabilityResult` with `stability_score=0.0`, `seed_results=[]`. Never raises.

---

### TEST 2.7 — `test_seed_results_are_floats_not_strings`
`train_fn` returns an integer (e.g. `return 1`). Assert `result.seed_results` contains `float(1.0)` not `int(1)` or `str("1")`.

---

### TEST 2.8 — `test_spread_is_max_minus_min`
`seed_results = [0.80, 0.87, 0.83, 0.85, 0.82]`. Assert `result.spread ≈ 0.87 - 0.80 = 0.07`. Not std × 2 or any other formula.

---

### TEST 2.9 — `test_rank_by_stability_raises_on_mismatched_lengths`
`len(configs)=3, len(stability_results)=2`. Assert `ValueError` raised. Not `IndexError`.

---

### TEST 2.10 — `test_custom_penalty_respected`
`penalty=2.0`. `mean=0.85, std=0.02`. Assert `stability_score ≈ 0.85 - 2.0 * 0.02 = 0.81`. Not `0.85 - 1.5 * 0.02`.

---

### TEST 2.11 — `test_format_stability_report_produces_human_readable_string`
Call `format_stability_report(ranked, top_n=3)`. Assert result is a non-empty string. Assert it contains "stability=" for each entry. Smoke test — must not raise.

---

### TEST 2.12 — `test_stability_result_is_dataclass_not_dict`
Assert `isinstance(result, StabilityResult)`. Assert `result.stability_score` accessible as attribute, not `result["stability_score"]`.

---

### TEST 2.13 — `test_stability_result_excluded_from_redis_checkpoint`
**Bug:** `StabilityResult` is a dataclass — `json.dumps(state)` raises `TypeError`.

Trigger HITL (checkpoint state to Redis). Assert no `TypeError`. `StabilityResult` must be excluded from checkpoint serialisation.

---

### TEST 2.14 — `test_top_10_configs_selected_by_mean_cv_not_stability`
**Bug:** Top-K configs selected by `stability_score` from the Optuna study — but stability hasn't been run yet at that point. Initial selection must be by `mean_cv` (from Optuna `user_attrs["mean_cv"]`). Stability ranking comes AFTER re-running the top-K.

Assert that the 10 configs passed to `run_with_seeds()` are the 10 with the highest Optuna `mean_cv`, not the 10 with the highest (undefined-at-that-point) stability score.

---

## BLOCK 3 — OPTUNA HPO INTEGRATION (12 tests)
**Class:** `TestOptunaHPOIntegration`

---

### TEST 3.1 — `test_study_direction_maximize_for_auc`
`metric="auc"`. Assert `study.direction.name == "MAXIMIZE"`.

---

### TEST 3.2 — `test_study_direction_minimize_for_log_loss`
**Bug:** Study direction hardcoded to "maximize" regardless of metric.

`metric="log_loss"`. Assert `study.direction.name == "MINIMIZE"`.

---

### TEST 3.3 — `test_all_three_model_types_searchable`
After 100+ trials: assert at least one trial with `user_attrs["params"]["model_type"] == "lgbm"`, one `"xgb"`, one `"catboost"`. All three model types must appear in search space.

---

### TEST 3.4 — `test_top_k_rerun_takes_exactly_10_configs`
Mock Optuna study with 200 completed trials. Assert exactly 10 configs passed to stability validator. Not 9, not 11.

---

### TEST 3.5 — `test_winner_has_highest_stability_score_among_top_k`
After optimizer runs: assert the model in `model_registry` has `stability_score` equal to the maximum stability score among all top-10 re-runs. Not just highest mean CV.

---

### TEST 3.6 — `test_n_jobs_is_1_in_study_optimize`
**Bug:** `n_jobs=-1` causes each worker to hold its own LightGBM model — OOM on 8GB.

Inspect the `study.optimize()` call. Assert `n_jobs=1`. Day 12 OOM guard must be preserved.

---

### TEST 3.7 — `test_gc_after_trial_is_true`
Assert `gc_after_trial=True` in `study.optimize()` call. Day 12 belt-and-braces GC preserved.

---

### TEST 3.8 — `test_langsmith_tracing_disabled_during_study`
`LANGCHAIN_TRACING_V2` must be `"false"` during `study.optimize()`. Day 15 cost guard preserved. Assert using monkeypatched `_disable_langsmith_tracing()` call count check.

---

### TEST 3.9 — `test_fold_scores_in_trial_user_attrs`
After study: every COMPLETE trial has `user_attrs["fold_scores"]` — a list of floats. Required for Day 13 Wilcoxon gate.

---

### TEST 3.10 — `test_model_registry_updated_not_replaced`
State has existing `model_registry` with 1 entry from a previous run. After `run_ml_optimizer()`: assert the original entry is still present AND a new entry was added. Registry is augmented, not replaced.

---

### TEST 3.11 — `test_xgb_params_exclude_label_encoder`
XGBoost suggested params must include `use_label_encoder=False`. Without this, XGBoost >= 1.6 logs deprecation warnings that pollute output and can cause failures in some environments.

---

### TEST 3.12 — `test_catboost_params_have_thread_count_1`
CatBoost suggested params must include `thread_count=1`. OOM guard — same principle as `n_jobs=1` for LightGBM/XGBoost.

---

## BLOCK 4 — END-TO-END + CONTRACT (8 tests)
**Class:** `TestMLOptimizerEndToEnd`

---

### TEST 4.1 — `test_full_optimizer_run_produces_valid_registry_entry`
Run `run_ml_optimizer()` end-to-end on a toy dataset (1000 rows, 10 features). Assert `model_registry` has exactly 1 entry. Assert all 12 required fields present. Assert `cv_mean > 0.5`.

---

### TEST 4.2 — `test_calibration_and_stability_both_run_for_log_loss`
`metric="log_loss"`. Assert both `is_calibrated=True` AND `len(seed_results)==5` in the registry entry. Both features active simultaneously.

---

### TEST 4.3 — `test_optimizer_complete_event_in_lineage`
After `run_ml_optimizer()`: assert `lineage.jsonl` has `action="ml_optimizer_complete"` with `stability_score`, `cv_mean`, `is_calibrated` fields.

---

### TEST 4.4 — `test_wilcoxon_gate_applied_vs_existing_champion`
State has existing champion with `fold_scores=[0.83]*5`. New optimizer produces `fold_scores=[0.84]*5` (clear improvement). Assert `lineage.jsonl` has `action="wilcoxon_gate_decision"`. Gate must be applied even when new model is obviously better.

---

### TEST 4.5 — `test_day12_oom_guards_not_regressed`
Run optimizer with monkeypatched `psutil.Process().memory_info().rss` returning 7GB on fold 3. Assert `optuna.TrialPruned` raised at fold 3. Day 12 memory guardrail not regressed by Day 19 changes.

---

### TEST 4.6 — `test_day13_column_order_preserved`
After `run_ml_optimizer()`: assert `state["feature_order"]` is set and matches the columns used for training. Day 13 column order guard not regressed.

---

### TEST 4.7 — `test_contract_winner_ranked_by_stability_not_peak`
CONTRACT: Run optimizer twice — once with a high-peak/high-variance config winning Optuna, once with a medium-peak/low-variance config. Assert the stable config is selected. This is the core contract.

---

### TEST 4.8 — `test_all_previous_optimizer_contracts_still_pass`
Run `pytest tests/contracts/test_wilcoxon_gate_contract.py` (Day 13). Run `pytest tests/contracts/test_submit_column_order_contract.py` (Day 13). Assert all pass. Day 19 additions must not regress any prior contracts.

---

## THE 5 BUGS THAT WILL DEFINITELY BE PRESENT

**Bug 1 — Test 1.9:** `cv='prefit'` missing from `CalibratedClassifierCV`. The natural call is `CalibratedClassifierCV(base_model, method='isotonic')`. Without `cv='prefit'`, sklearn will re-run cross-validation internally to fit the calibrator — it refits the underlying model from scratch, discarding all of Optuna's work. The fitted LightGBM model is ignored.

**Bug 2 — Test 2.1:** `stability_score = mean - 1.0 * std`. The spec says 1.5. The natural implementation writes `mean - std` (implicit multiplier of 1.0). This creates a less conservative stability penalty that allows higher-variance models to win.

**Bug 3 — Test 3.2:** Study direction hardcoded to `"maximize"`. For log-loss and cross-entropy, lower is better — the study must be created with `direction="minimize"`. The natural implementation copies the AUC direction without checking the metric. A log-loss study with `maximize` direction will select the worst trial as the winner.

**Bug 4 — Test 2.14:** Top-K selected by stability score from Optuna. But stability hasn't been run at Optuna time — `user_attrs` only has `mean_cv`. The implementation might attempt `sorted(trials, key=lambda t: t.user_attrs.get("stability_score", 0))`, which would sort by 0 for all trials (key absent) and return an effectively random selection. Must sort by `user_attrs["mean_cv"]`.

**Bug 5 — Test 1.10:** Calibration fold carved out after CV. The natural order: run CV, pick best model, then split a calibration fold. But if the calibration data was part of the CV training folds, the calibration is fitting on data the model already partially learned from. The split must happen at the very beginning of the training function, before any fold indices are created.

---

## TOTAL: 50 + 8 CONTRACT TESTS

| Block | Tests |
|---|---|
| 1 — Prediction calibration | 16 |
| 2 — Stability validator | 14 |
| 3 — Optuna HPO integration | 12 |
| 4 — End-to-end + contract | 8 |
| ML Optimizer Optuna contract | 8 |
| **Total** | **58** |

---

## DEFINITION OF DONE

- [ ] 50 tests in `tests/test_day19_quality.py` — all pass
- [ ] `tests/contracts/test_ml_optimizer_optuna_contract.py` — 8 tests, all pass, immutable
- [ ] `pytest tests/contracts/` — all previous contracts still pass
- [ ] `pytest tests/regression/` — Phase 1 + Phase 2 frozen tests still pass
- [ ] Manual check: `model_registry` entry has `is_calibrated`, `stability_score`, `seed_results` after a real run
- [ ] Manual check: study direction is "minimize" when metric="log_loss"
- [ ] `git commit -m "Day 19: calibration, stability validator, Optuna HPO, optimizer contract — 50 tests green"`