# Professor Agent — Day 17 Test Specification
**Status: IMMUTABLE after Day 17** | **44 tests**
```bash
pytest tests/test_day17_quality.py -v --tb=short 2>&1 | tee tests/logs/day17_quality.log
```

---

## BLOCK 1 — WILCOXON FEATURE GATE: CORRECTNESS (10 tests)
**Class:** `TestWilcoxonFeatureGate`

The bugs: `is_feature_worth_adding` uses `alternative="less"` instead of `"greater"` (inverted — keeps bad features, drops good ones). `feature_gate_result` has `gate_type` hardcoded as `"model_comparison"` (copy-paste from Day 13). 3-fold quick CV scores passed to `is_feature_worth_adding` — the MIN_FOLDS guard falls back to mean comparison (correct behaviour, but the log must say "fallback", not "Wilcoxon p=...").

---

### TEST 1.1 — `test_returns_true_when_feature_clearly_improves_cv`
**Setup:** Baseline scores `[0.80, 0.81, 0.80]`, augmented scores `[0.85, 0.86, 0.85]`. Assert `is_feature_worth_adding(baseline, augmented, "good_feature") == True`. Clear improvement, should pass.

---

### TEST 1.2 — `test_returns_false_when_feature_adds_noise`
Scores differ by < 0.001 across all folds. Assert `False`. Lucky feature does not survive.

---

### TEST 1.3 — `test_returns_false_when_feature_hurts_performance`
Augmented scores consistently below baseline. Assert `False`. Bad feature rejected.

---

### TEST 1.4 — `test_alternative_is_greater_not_less`
**Bug:** `alternative="less"` — test checks if augmented is WORSE than baseline, which always returns True for bad features.

Inspect the call to `is_significantly_better()` inside `is_feature_worth_adding()`. Assert `alternative="greater"` is passed. If `"less"` is present, every feature passes the gate — the filter is inverted.

---

### TEST 1.5 — `test_three_fold_quick_cv_falls_back_to_mean_comparison`
Pass 3-fold scores (below `MIN_FOLDS_REQUIRED=5`). Assert function returns a bool without raising. Assert log message contains "fallback" or "mean comparison". The 3-fold case must be handled gracefully.

---

### TEST 1.6 — `test_feature_gate_result_has_gate_type_field`
Call `feature_gate_result(...)`. Assert `result["gate_type"] == "feature_selection"`. Not `"model_comparison"` (Day 13 value). Copy-paste bug.

---

### TEST 1.7 — `test_feature_gate_result_has_decision_keep_or_drop`
Gate passed: `result["decision"] == "KEEP"`. Gate failed: `result["decision"] == "DROP"`. Both cases tested.

---

### TEST 1.8 — `test_feature_gate_result_has_feature_name`
Call `feature_gate_result(..., feature_name="target_enc_cabin")`. Assert `result["feature_name"] == "target_enc_cabin"`. Used in lineage — must be present.

---

### TEST 1.9 — `test_is_feature_worth_adding_never_raises`
Call with: mismatched lengths, empty lists, all-zero scores, NaN scores. Assert all return `False` without raising. Same guarantee as `is_significantly_better`.

---

### TEST 1.10 — `test_feature_factory_logs_gate_decision_to_lineage`
Run `_evaluate_candidate_feature()` from feature_factory. Assert `lineage.jsonl` contains entry with `action="wilcoxon_feature_gate"` and `feature_name` present. Every gate decision must be auditable.

---

## BLOCK 2 — STAGE 1 PERMUTATION FILTER (10 tests)
**Class:** `TestStage1PermutationFilter`

The bugs: `actual_importances` computed from shuffled model instead of real model (swapped). `drop_percentile=0.65` applied to absolute importances instead of importance ratios (doesn't account for features with naturally high null importance). Safety fallback not triggered when Stage 1 drops all features. GC not called after each shuffle model (memory accumulates across 5 shuffles).

---

### TEST 2.1 — `test_stage1_drops_approximately_65_percent`
Run Stage 1 on a dataset with 20 features, 5 of which are genuinely predictive, 15 are pure noise. Assert `len(dropped) / 20` is between 0.55 and 0.75. The filter should remove the majority of noise features.

---

### TEST 2.2 — `test_stage1_actual_importances_from_real_y_not_shuffled`
**Bug:** Actual importances computed from a shuffled-target model (first null shuffle) instead of from the real-y model.

Add a feature with a known strong signal (e.g. `feature = y * 2`). Assert this feature's entry in `actual_importances` is substantially higher than its null importance. If both are similar (real model trained on shuffled y), the bug is present.

---

### TEST 2.3 — `test_stage1_uses_importance_ratio_not_absolute`
**Bug:** Threshold applied to absolute importance values. A feature with naturally high null importance (e.g. a high-cardinality column) would survive even if its actual/null ratio is poor.

Create two features: Feature A has actual=100, null=95 (ratio=1.05). Feature B has actual=5, null=1 (ratio=5.0). Assert Feature B is NOT dropped and Feature A IS dropped. Ratio-based filtering must be used, not absolute value filtering.

---

### TEST 2.4 — `test_stage1_safety_fallback_when_all_features_dropped`
Manipulate `drop_percentile=1.0` (drop everything). Assert `run_null_importance_filter()` returns all features in `survivors`. Safety fallback must prevent the filter from eliminating every feature.

---

### TEST 2.5 — `test_stage1_gc_called_after_each_null_model`
**Bug:** `del model_null; gc.collect()` not inside the shuffle loop — only after the loop ends. Memory for all 5 null models held simultaneously.

Monkeypatch `gc.collect` to count calls. After Stage 1 with 5 shuffles: assert `gc.collect()` called at least 5 times (once per shuffle). If called only once (after the loop), memory accumulates.

---

### TEST 2.6 — `test_stage1_uses_fixed_random_seed`
Run Stage 1 twice on the same data. Assert `dropped` list is identical both times. Fixed seed (`np.random.default_rng(seed=42)`) ensures deterministic results.

---

### TEST 2.7 — `test_stage1_returns_correct_actual_importances_dict`
Assert `actual_importances` dict has exactly `len(feature_names)` keys. Assert all values are non-negative floats. Assert keys match `feature_names` exactly.

---

### TEST 2.8 — `test_stage1_skipped_on_fewer_than_10_features`
`len(feature_names) = 7`. Assert `run_null_importance_filter()` returns all 7 features as survivors without running any shuffles (no LightGBM calls). Assert result has `stage1_drop_count == 0`.

---

### TEST 2.9 — `test_stage1_lgbm_params_n_jobs_is_1`
**Bug:** `n_jobs=-1` in stage 1 parameters — 8 LightGBM processes × 5 shuffles × memory → OOM on 8GB.

Inspect the `lgb_params` dict inside `_run_stage1_permutation_filter`. Assert `n_jobs == 1`.

---

### TEST 2.10 — `test_stage1_result_includes_correct_drop_count`
Run Stage 1 on 20 features. Assert `result.stage1_drop_count == len(result.dropped_stage1)`. Assert `result.stage1_drop_count + len(result.survivors_after_stage1) == 20`. Count consistency.

---

## BLOCK 3 — STAGE 2 NULL IMPORTANCE (10 tests)
**Class:** `TestStage2NullImportance`

The bugs: sandbox returns non-JSON stdout (progress messages mixed with JSON output) — `json.loads()` fails. `threshold_percentile=95` applied but the null distribution has fewer than 20 entries — 95th percentile of 5 values is the max, not a meaningful threshold. `execute_code()` called 50 times (one per shuffle) instead of once (all shuffles in the script).

---

### TEST 3.1 — `test_stage2_runs_in_single_execute_code_call`
**Bug:** `execute_code()` called in a loop — 50 container spin-ups.

Monkeypatch `execute_code` to count calls. Run Stage 2 with `n_shuffles=50`. Assert `execute_code` called exactly once. This is the persistent sandbox requirement.

---

### TEST 3.2 — `test_stage2_script_outputs_json_to_stdout`
The generated script's final line is `print(json.dumps(result))`. Assert that the script template, when rendered and executed with test data, produces stdout that is valid JSON. Assert `json.loads(stdout)` succeeds.

---

### TEST 3.3 — `test_stage2_script_progress_messages_go_to_stderr`
Progress messages (`"Progress: 10/50 shuffles complete"`) use `print(..., file=sys.stderr)`. Assert they do NOT appear in stdout. stdout must be pure JSON for `json.loads()` to succeed.

---

### TEST 3.4 — `test_stage2_keeps_features_above_95th_percentile_threshold`
Build a feature with actual importance = 150, null distribution = [1, 2, 3, ... 50] (95th pct = ~47.5). Assert feature in `stage2_survivors`. Actual >> null threshold → keep.

---

### TEST 3.5 — `test_stage2_drops_features_below_95th_percentile_threshold`
Feature with actual importance = 10, null distribution = [5, 8, 9, 10, 11, 12, ... 50] (95th pct = 48). Assert feature in `stage2_dropped`. Actual < null threshold → drop.

---

### TEST 3.6 — `test_stage2_graceful_fallback_on_sandbox_failure`
**Bug:** `execute_code()` returns `returncode=-1` (sandbox crashed) → Stage 2 raises or silently drops all features.

Mock `execute_code` to return `{"returncode": -1, "stdout": "", "stderr": "OOM", "timed_out": False, "backend": "docker"}`. Assert Stage 2 returns all Stage 1 survivors unchanged. Log must contain "Stage 2 sandbox failed".

---

### TEST 3.7 — `test_stage2_graceful_fallback_on_invalid_json_output`
Mock `execute_code` to return stdout `"not valid json"`. Assert graceful fallback — all survivors returned, no `json.JSONDecodeError` propagated.

---

### TEST 3.8 — `test_stage2_null_distributions_correct_length`
Run Stage 2 with `n_shuffles=10`. Assert each feature's `null_distributions[feature]` has exactly 10 entries. If the script ran only 8 shuffles (e.g. early exit), the distribution is undersized.

---

### TEST 3.9 — `test_stage2_threshold_percentiles_stored_per_feature`
After Stage 2: assert `result.threshold_percentiles` has one entry per Stage 1 survivor. Assert each value is a non-negative float. These are needed for the post-run audit trail.

---

### TEST 3.10 — `test_stage2_actual_vs_threshold_dict_has_all_survivors`
`result.actual_vs_threshold` must have one entry per Stage 1 survivor. Each entry has keys: `actual`, `threshold`, `ratio`, `passed`. Assert for all survivor features.

---

## BLOCK 4 — PERSISTENT SANDBOX PATTERN (6 tests)
**Class:** `TestPersistentSandboxPattern`

---

### TEST 4.1 — `test_stage2_script_is_self_contained_python`
Render the Stage 2 script template with test data. Assert it is valid Python (parse with `ast.parse()`). Self-contained means no imports from outside the standard library + numpy + lightgbm + json.

---

### TEST 4.2 — `test_stage2_script_handles_large_input_without_truncation`
Render script with `X` having 100 features × 10000 rows. Assert the rendered script length is within limits (< 5MB — the sandbox's max output cap). Large inputs must not produce truncated scripts.

---

### TEST 4.3 — `test_execute_code_timeout_set_to_600_seconds`
Inspect the `execute_code(script, timeout=...)` call inside `_run_stage2_null_importance_persistent_sandbox`. Assert `timeout=600`. Not the default (which might be shorter), not None (which would hang forever).

---

### TEST 4.4 — `test_stage2_uses_execute_code_not_subprocess_directly`
Assert that Stage 2 uses `from tools.e2b_sandbox import execute_code` and calls `execute_code()`. Must not call `subprocess.run()` directly — that bypasses the Docker sandbox and its resource limits.

---

### TEST 4.5 — `test_null_importance_result_excluded_from_redis_checkpoint`
**Bug:** `NullImportanceResult` dataclass is stored in `state["null_importance_result"]`. When HITL triggers and `json.dumps(state)` is called for Redis checkpoint, `TypeError` on the dataclass.

Run pipeline that triggers HITL. Assert no `TypeError`. The checkpoint serialisation must exclude `null_importance_result` (or serialise only its JSON-safe fields).

---

### TEST 4.6 — `test_stage2_script_gc_collect_inside_shuffle_loop`
**Bug:** `gc.collect()` at end of script (after all 50 shuffles) instead of inside the loop. Inside a Docker container with a memory limit, not calling GC per shuffle risks OOM in the container.

Check the script template. Assert `gc.collect()` appears inside the shuffle loop body (before `del model_null` closes the iteration). The count of `gc.collect()` calls inside the rendered script's loop body should be ≥ 1.

---

## BLOCK 5 — FULL PIPELINE INTEGRATION (8 tests)
**Class:** `TestNullImportancePipelineIntegration`

---

### TEST 5.1 — `test_null_importance_filter_reduces_feature_count`
Run `run_null_importance_filter()` on a dataset with 30 features (20 noise, 10 signal). Assert `result.total_features_output < result.total_features_input`. The filter must actually remove features — not a no-op.

---

### TEST 5.2 — `test_null_importance_filter_preserves_known_signal_features`
Dataset: 5 features perfectly correlated with target, 20 pure noise features. Assert all 5 signal features appear in `result.survivors`. True signal must not be dropped.

---

### TEST 5.3 — `test_null_importance_result_logged_to_lineage`
After `_apply_null_importance_filter()` in feature_factory: assert `lineage.jsonl` contains entry with `action="null_importance_filter_complete"` and `total_input`, `total_output`, `elapsed_s` fields.

---

### TEST 5.4 — `test_wilcoxon_gate_and_null_importance_both_run_in_feature_factory`
Run `feature_factory` end-to-end. Assert `lineage.jsonl` contains both `action="wilcoxon_feature_gate"` AND `action="null_importance_filter_complete"`. Both filters must run — neither replaces the other.

---

### TEST 5.5 — `test_features_dropped_stage1_stored_in_state`
After `_apply_null_importance_filter()`: assert `state["features_dropped_stage1"]` is a list (may be empty). Used for post-mortem analysis.

---

### TEST 5.6 — `test_features_dropped_stage2_stored_in_state`
Same as 5.5 for `state["features_dropped_stage2"]`.

---

### TEST 5.7 — `test_null_importance_elapsed_seconds_is_reasonable`
Run on a small dataset (1000 rows, 15 features). Assert `result.elapsed_seconds < 120`. Stage 1 + Stage 2 combined should not take > 2 minutes on small data.

---

### TEST 5.8 — `test_day13_wilcoxon_model_contracts_still_pass`
Run `pytest tests/contracts/test_wilcoxon_gate_contract.py`. Assert all 4 original contracts pass. Adding `is_feature_worth_adding` and `feature_gate_result` must not break `is_significantly_better` or `gate_result`.

---

## THE 5 BUGS THAT WILL DEFINITELY BE PRESENT

**Bug 1 — Test 1.4:** `alternative="less"` in `is_feature_worth_adding`. When copy-pasting from `is_significantly_better`, the `alternative` parameter keeps the default `"greater"` — but a developer might think "I want to test that augmented is greater than baseline" and invert it. The test explicitly verifies the direction.

**Bug 2 — Test 2.2:** Actual importances computed from the first shuffled-y model. The natural implementation order: create the LGBMClassifier, fit it... then create the null models. If the developer accidentally fits the null model first and reads importances from it, actual importances come from a noise model.

**Bug 3 — Test 3.1:** `execute_code()` called in a loop. The most natural implementation: "for each shuffle, run the model, collect result". Translates directly to a loop calling `execute_code()` 50 times. The persistent sandbox pattern requires inverting this — write a script that loops internally. This is a non-obvious architecture change that almost every first implementation gets wrong.

**Bug 4 — Test 3.3:** Progress messages mixed into stdout. The script template uses `print(f"Progress: {i+1}/{n_shuffles}...")` to stdout during the shuffle loop. The final `print(json.dumps(result))` outputs valid JSON. But `json.loads(stdout)` fails because stdout is `"Progress: 10/50...\nProgress: 20/50...\n{...json...}"`. Fix: all progress to `sys.stderr`, final JSON to stdout only.

**Bug 5 — Test 4.5:** `NullImportanceResult` dataclass in Redis checkpoint. The pattern from Day 12: all state goes into the checkpoint. `null_importance_result` is a dataclass — `json.dumps` raises `TypeError`. The fix (exclude from checkpoint) requires explicit exclusion in `_checkpoint_state_to_redis`, which already has one exclusion (`_langfuse_trace` from Day 15). Adding the second is straightforward once the bug is identified.

---

## TOTAL: 44 TESTS

| Block | Tests |
|---|---|
| 1 — Wilcoxon feature gate | 10 |
| 2 — Stage 1 permutation filter | 10 |
| 3 — Stage 2 null importance | 10 |
| 4 — Persistent sandbox pattern | 6 |
| 5 — Pipeline integration | 8 |
| **Total** | **44** |

---

## DEFINITION OF DONE

- [ ] 44 tests in `tests/test_day17_quality.py` — all pass
- [ ] `pytest tests/contracts/test_wilcoxon_gate_contract.py` — still green (Day 13 contracts unchanged)
- [ ] `pytest tests/regression/` — still green (Phase 1 + Phase 2 frozen)
- [ ] Manual check: run `docker ps -a` during Stage 2 — confirm only 1 container running (not 50)
- [ ] `git commit -m "Day 17: Wilcoxon feature gate + null importance two-stage filter — 44 tests green"`