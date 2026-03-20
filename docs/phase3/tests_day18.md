# Professor Agent — Day 18 Test Specification
**Status: IMMUTABLE after Day 18** | **52 tests**
```bash
pytest tests/test_day18_quality.py -v --tb=short 2>&1 | tee tests/logs/day18_quality.log
```

---

## BLOCK 1 — FEATURE FACTORY ROUNDS 3 + 4 (14 tests)
**Class:** `TestFeatureFactoryRounds3And4`

---

### TEST 1.1 — `test_round3_generates_all_five_agg_functions`
Schema with 1 categorical (n_unique=10) and 1 numeric. Assert exactly 5 candidates generated (mean, std, min, max, count). All have `round=3` and `transform_type="groupby_agg"`.

---

### TEST 1.2 — `test_round3_caps_at_max_candidates`
Schema with 10 categoricals × 10 numerics = 500 pairs × 5 = 2500 candidates. Assert `len(round3_candidates) <= MAX_ROUND3_CANDIDATES`. Cap must fire.

---

### TEST 1.3 — `test_round3_cap_prioritises_low_cardinality_categorical`
Two categoricals: A (n_unique=3), B (n_unique=100). Assert candidates with A as the groupby column are ranked higher (lower cardinality = more stable group stats). When cap fires, B-based features are dropped first.

---

### TEST 1.4 — `test_round3_skips_id_and_target_columns`
Schema with an id column and a target column. Assert neither appears as a `source_column` in any Round 3 candidate.

---

### TEST 1.5 — `test_round3_apply_produces_correct_group_means`
Apply groupby_agg mean on a toy DataFrame. Verify: all rows with `cat="A"` get the mean of the numeric column for group A. Assert the new column is correct using a manual calculation.

---

### TEST 1.6 — `test_round3_apply_uses_join_not_apply`
**Bug:** `_apply_round3_transforms()` uses `pl.col(cat).apply(lambda x: ...)` — O(n) Python loops instead of Polars native group_by+join.

Monkeypatch `pl.DataFrame.apply` to raise `NotImplementedError`. Assert `_apply_round3_transforms()` completes successfully. It must use `group_by().agg()` + `join()`, not `apply()`.

---

### TEST 1.7 — `test_round4_generates_only_suitable_categoricals`
Schema with: col A (dtype=str, n_unique=15), col B (dtype=str, n_unique=1, binary), col C (dtype=float64, n_unique=300 — too high cardinality), col D (dtype=str, n_unique=250 — too high). Assert only col A generates a Round 4 candidate. Binary and very-high-cardinality columns excluded.

---

### TEST 1.8 — `test_round4_caps_at_max_candidates`
30+ eligible categoricals. Assert `len(round4_candidates) <= MAX_ROUND4_CANDIDATES`.

---

### TEST 1.9 — `test_round4_encoding_never_uses_validation_fold_target`
**The most important Round 4 test.**

Inject a column where `mean(target | category="A") = 1.0` exactly. On a 5-fold CV where fold 3 is the validation fold: assert the encoded value for fold 3's "A" samples was computed from folds 1,2,4,5 only — not from the full training set (which would include fold 3's target values).

Verify by checking that the encoded value for "A" in fold 3 differs from the global mean of "A" computed on all rows.

---

### TEST 1.10 — `test_round4_smoothing_formula_correct`
Category "X" appears 5 times in the training portion of a fold. Their targets: [1, 1, 1, 0, 1] → group_mean=0.8. Global mean=0.5. Smoothing=30. Assert encoded value ≈ `(5 * 0.8 + 30 * 0.5) / (5 + 30) = 19.0 / 35.0 ≈ 0.543`. Not 0.8 (unsmoothed), not 0.5 (global mean).

---

### TEST 1.11 — `test_round4_unseen_category_gets_global_mean`
Category "Z" appears in validation but never in training. Assert encoded value = global_mean. Must not raise `KeyError`.

---

### TEST 1.12 — `test_round4_candidates_have_round_4_field`
All candidates from `_generate_round4_target_encoding_candidates()` have `round == 4` and `transform_type == "target_encoding"`.

---

### TEST 1.13 — `test_round4_high_cardinality_sorted_first`
Categoricals: A (n_unique=50), B (n_unique=20), C (n_unique=35). Assert returned candidates in order B, C, A — wait, sorted descending by n_unique: A(50), C(35), B(20). Verify ordering.

---

### TEST 1.14 — `test_round3_round4_both_appear_in_manifest`
Run full `run_feature_factory()`. Assert `feature_manifest.json` contains candidates with both `"round": 3` and `"round": 4`. Both rounds active simultaneously.

---

## BLOCK 2 — FEATURE FACTORY ROUND 5 + INTERACTION BUDGET CAP (14 tests)
**Class:** `TestFeatureFactoryRound5AndBudgetCap`

---

### TEST 2.1 — `test_round5a_generates_from_unvalidated_insights_only`
`competition_brief["insights"]`: 3 with `validated=False`, 2 with `validated=True`. Assert Round 5a generates exactly 3 (or fewer if LLM can't form a feature for all) candidates. Validated insights must never trigger feature generation.

---

### TEST 2.2 — `test_round5a_validates_source_columns_against_schema`
Mock LLM to return candidate with `source_columns=["ghost_column"]`. Assert candidate excluded. Warning logged.

---

### TEST 2.3 — `test_round5a_capped_at_10_candidates`
Mock LLM to return 15 candidates (all valid). Assert `len(round5a_candidates) <= 10`.

---

### TEST 2.4 — `test_round5a_graceful_on_no_unvalidated_hypotheses`
`competition_brief["insights"] = []` (no insights). Assert returns `[]`. No LLM call made (monkeypatch to verify zero calls).

---

### TEST 2.5 — `test_round5b_limited_to_top_k_features`
`top_features_by_importance` has 50 features. `max_k=20`. Assert Round 5b only generates interactions for the first 20 features. No pair involves a feature ranked > 20 by importance.

---

### TEST 2.6 — `test_round5b_domain_pairs_included_first`
`competition_brief["meaningful_interactions"] = [["fare", "pclass"]]`. `top_k` includes both "fare" and "pclass". Assert `"fare_x_pclass"` (or equivalent) appears in Round 5b candidates, and appears before non-domain pairs (first in the list).

---

### TEST 2.7 — `test_budget_cap_fires_above_500_interactions`
Generate 600 interaction candidates. Assert `_apply_interaction_budget_cap()` returns exactly 500 interaction candidates. Non-interaction candidates (Rounds 1-4, 5a) preserved.

---

### TEST 2.8 — `test_budget_cap_does_not_fire_below_500_interactions`
Generate 300 interaction candidates. Assert all 300 returned unchanged. Cap only fires when exceeded.

---

### TEST 2.9 — `test_budget_cap_scores_domain_pairs_higher`
Domain pairs (`meaningful_interactions`) must survive the cap ahead of non-domain pairs with equal importance scores. Simulate: 600 candidates, 10 domain-guided + 590 non-domain. Assert all 10 domain candidates in the top 500. `domain_relevance=2.0` ensures this.

---

### TEST 2.10 — `test_budget_cap_preserves_non_interaction_candidates`
500 interaction candidates + 150 non-interaction (Rounds 1–4, 5a). Apply cap with `max_cap=500`. Assert all 150 non-interaction candidates preserved. Cap applies to interactions only.

---

### TEST 2.11 — `test_round5b_generates_correct_interaction_names`
Pair (A, B) with multiply operation: assert candidate name is `"A_x_B"` or `"B_x_A"` (consistent ordering). No whitespace or special characters in the name.

---

### TEST 2.12 — `test_max_interaction_features_constant_is_20`
Assert `MAX_INTERACTION_FEATURES == 20` in the module. Hard-coded in constants — not derived from config. Changing it requires code review.

---

### TEST 2.13 — `test_max_interaction_candidates_constant_is_500`
Assert `MAX_INTERACTION_CANDIDATES == 500` in the module. Same guard.

---

### TEST 2.14 — `test_total_candidates_after_all_rounds_within_budget`
Run `run_feature_factory()` end-to-end with a medium schema (20 features, 5 categoricals). Assert `len(all_candidates)` does not exceed `50 (R1) + 30 (R2) + 200 (R3) + 30 (R4) + 10 (R5a) + 500 (R5b cap) = 820`. This is the absolute ceiling on candidate count.

---

## BLOCK 3 — PSEUDO-LABELING: CORRECTNESS (14 tests)
**Class:** `TestPseudoLabelCorrectness`

The bugs: validation fold contaminated with pseudo-labels (confidence check passes but val fold has seen pseudo-label targets). Confidence computed as `pred` instead of `abs(pred - 0.5)` (always selects top 10% by prediction score, not by certainty). Max iterations loop runs 4 times (off-by-one on `range(1, MAX_PL_ITERATIONS+1)`). CV improvement checked against iteration-1 baseline for all iterations (should compare each iteration to the previous).

---

### TEST 3.1 — `test_confidence_for_binary_is_distance_from_0_5`
`y_pred = [0.9, 0.1, 0.6, 0.4, 0.5]`. Assert confidence = `[0.4, 0.4, 0.1, 0.1, 0.0]`. Not `[0.9, 0.1, 0.6, 0.4, 0.5]` (raw prediction). The 0.9 and 0.1 predictions tie for highest confidence — both are 0.4 away from 0.5.

---

### TEST 3.2 — `test_top_10_percent_selection_correct`
100 test samples. Confidence computed. Assert exactly 10 selected (top 10%). If confidence values have ties at the threshold, assert `n_selected >= 10` (acceptable) but `n_selected <= 15` (not runaway).

---

### TEST 3.3 — `test_validation_fold_never_sees_pseudo_labels`
**The most critical pseudo-labeling test.**

Run `_run_cv_with_pseudo_labels()` with a detectable pseudo-label pattern. Create pseudo-labels where all samples have target=1.0 (100% positive). Run CV. Assert validation fold scores don't benefit from this signal — the pseudo-labels must be invisible to the validation fold.

Verify by: monkeypatching the model fit to record what `y_fold_train` it received. For each fold, assert no pseudo-label samples appear in the validation portion.

---

### TEST 3.4 — `test_cv_gate_stops_iteration_when_no_improvement`
Mock `_run_cv_with_pseudo_labels` to return scores slightly worse than baseline. Assert `result.iterations_completed == 0`. Assert `result.halt_reason == "cv_did_not_improve"`. Assert `pseudo_labels_applied == False`.

---

### TEST 3.5 — `test_cv_gate_proceeds_when_improvement_significant`
Mock CV to return scores 0.005 better than baseline, with Wilcoxon p < 0.05 (clear improvement). Assert `result.iterations_completed >= 1`. Assert `pseudo_labels_applied == True`.

---

### TEST 3.6 — `test_wilcoxon_gate_applied_to_cv_improvement`
**Bug:** CV mean compared numerically (`cv_with > cv_without + threshold`) without Wilcoxon. This allows a lucky seed to push pseudo-labeling forward on noise.

Monkeypatch `is_significantly_better` to track calls. Run pseudo-label agent. Assert `is_significantly_better` was called at least once per iteration.

---

### TEST 3.7 — `test_max_iterations_is_3_not_4`
All 3 iterations improve CV. Assert loop stops at exactly 3. Assert `result.iterations_completed == 3`. Assert `result.halt_reason == "max_iterations"`. `range(1, 4)` produces 1, 2, 3 — verify off-by-one not present.

---

### TEST 3.8 — `test_confidence_comparison_for_regression_uses_interval_width`
Metric = "rmse". No quantile model available (fallback). Assert `_compute_confidence()` returns uniform confidence array (all 1.0). Must not raise. Must not use binary classification formula.

---

### TEST 3.9 — `test_multiclass_confidence_is_margin_between_top_2_classes`
`y_pred = [[0.7, 0.2, 0.1], [0.4, 0.35, 0.25], [0.9, 0.05, 0.05]]`. Assert confidence = `[0.5, 0.05, 0.85]`. Margin = top_class_prob - second_class_prob.

---

### TEST 3.10 — `test_no_pseudo_labels_when_no_confident_samples`
Mock confidence to all be below threshold (all samples exactly at 0.5 — zero confidence). Assert `result.halt_reason == "no_confident_samples"`. Assert `result.pseudo_labels_added == []`.

---

### TEST 3.11 — `test_pseudo_labels_accumulated_across_iterations`
Iteration 1 selects 10 samples. Iteration 2 selects 8 MORE samples (from remaining pool). Assert `sum(result.pseudo_labels_added) == 18`. Assert accumulated pseudo-label array has 18 entries. Iterations don't re-select already-labeled samples.

---

### TEST 3.12 — `test_already_pseudolabeled_samples_excluded_from_subsequent_iterations`
After iteration 1 labels test samples {0, 5, 10}: assert iterations 2 and 3 never re-select those indices. `current_test_mask` must correctly track which test samples have been labeled.

---

### TEST 3.13 — `test_pseudo_label_result_excluded_from_redis_checkpoint`
Run pipeline that triggers HITL checkpoint. `state["pseudo_label_result"]` is a dataclass. Assert no `TypeError` in `json.dumps`. Must be excluded from checkpoint serialization.

---

### TEST 3.14 — `test_pseudo_label_agent_skipped_when_no_selected_models`
`state["selected_models"] = []`. Assert `run_pseudo_label_agent()` returns state unchanged with `pseudo_labels_applied=False`. No exception.

---

## BLOCK 4 — PSEUDO-LABELING: INTEGRATION (10 tests)
**Class:** `TestPseudoLabelIntegration`

---

### TEST 4.1 — `test_pseudo_label_agent_runs_after_ensemble`
Run full pipeline. Assert `lineage.jsonl` has `action="ensemble_selection_complete"` BEFORE `action="pseudo_label_complete"`. Pipeline order enforced.

---

### TEST 4.2 — `test_x_train_with_pseudo_larger_than_original`
After successful pseudo-labeling: assert `len(state["X_train_with_pseudo"]) > len(X_train_original)`. Pseudo-labels are actually added to the training set.

---

### TEST 4.3 — `test_x_train_with_pseudo_same_schema_as_x_train`
Assert `state["X_train_with_pseudo"].schema == state["X_train"].schema`. No new columns, same dtypes.

---

### TEST 4.4 — `test_y_train_with_pseudo_length_matches_x_train_with_pseudo`
`len(y_train_with_pseudo) == len(X_train_with_pseudo)`. Must be aligned.

---

### TEST 4.5 — `test_pseudo_label_cv_improvement_logged_to_lineage`
After pseudo-label agent: assert `lineage.jsonl` contains `action="pseudo_label_complete"` with `cv_improvement`, `iterations`, `total_pl_added` fields.

---

### TEST 4.6 — `test_pseudo_labels_applied_false_when_cv_does_not_improve`
Mock CV to never improve. Assert `state["pseudo_labels_applied"] == False`. Assert `state["X_train_with_pseudo"]` is identical to original `X_train`. No pseudo-labels added.

---

### TEST 4.7 — `test_submission_uses_model_trained_on_augmented_data`
When `pseudo_labels_applied=True`: assert the final submission predictions come from a model trained on `X_train_with_pseudo`, not `X_train`. The submission_strategist must check `pseudo_labels_applied` and use the right training set.

---

### TEST 4.8 — `test_confidence_threshold_stored_per_iteration`
After 2 successful iterations: assert `len(result.confidence_thresholds) == 2`. Each iteration's threshold is stored for post-mortem analysis.

---

### TEST 4.9 — `test_zero_iterations_when_first_cv_fails_gate`
First iteration CV doesn't improve. Assert `result.iterations_completed == 0`. Assert state unchanged. The iteration counter must correctly reflect 0 when the first iteration is reverted.

---

### TEST 4.10 — `test_day16_ensemble_oof_still_available_after_pseudo_labeling`
After pseudo-label agent runs: assert `state["ensemble_oof"]` (set by ensemble_architect) is unchanged. Pseudo-labeling must not overwrite or corrupt ensemble outputs.

---

## THE 5 BUGS THAT WILL DEFINITELY BE PRESENT

**Bug 1 — Test 3.3:** Pseudo-labels in validation fold. The implementation uses `np.vstack([X_np[train_idx], X_pseudo_np])` but a developer might instead concatenate before the CV loop and re-split, which would include pseudo-labels in validation. The critical invariant — pseudo-labels concatenated to `X_fold_train` INSIDE the loop, AFTER `train_idx, val_idx` are determined — is easy to violate by restructuring the loop.

**Bug 2 — Test 3.1:** Confidence = `pred` instead of `abs(pred - 0.5)`. The natural implementation for "high confidence" is "high prediction score" — selecting the 10% of samples the model predicts are most likely positive. But this selects only from one class. A sample predicted 0.95 positive and a sample predicted 0.05 positive are both equally confident. `abs(pred - 0.5)` correctly measures distance from the decision boundary for both classes.

**Bug 3 — Test 1.9:** Round 4 target encoding leaks through the validation fold. The implementation creates a mapping of `{category: mean_target}` from the full training set, then applies it to all rows including the validation fold. This leaks because validation rows' own targets contributed to the group mean. Fix: compute group stats from `X_train[train_idx]` only, inside the fold loop.

**Bug 4 — Test 2.7:** Budget cap fires but `non_interaction` candidates are also filtered. The natural implementation: `sorted(all_candidates, key=score)[:500]`. If non-interaction candidates have scores (e.g. 1.0 × 1.0 = 1.0 by default), they can be ranked below some interaction candidates and get excluded. The cap must apply to interaction candidates only — non-interaction candidates always survive.

**Bug 5 — Test 3.6:** Wilcoxon gate not applied to CV improvement. The CV check reads `if cv_mean_with > cv_mean_without + MIN_CV_IMPROVEMENT` — pure numerical comparison. A 0.001 improvement on a noisy CV is within sampling variance. Without Wilcoxon, pseudo-labeling can proceed on noise and hurt LB. The gate requires importing `is_significantly_better` from `tools.wilcoxon_gate` — easy to forget when the numerical check "seems sufficient."

---

## TOTAL: 52 TESTS

| Block | Tests |
|---|---|
| 1 — Feature factory Rounds 3 + 4 | 14 |
| 2 — Feature factory Round 5 + budget cap | 14 |
| 3 — Pseudo-labeling correctness | 14 |
| 4 — Pseudo-labeling integration | 10 |
| **Total** | **52** |

---

## DEFINITION OF DONE

- [ ] 52 tests in `tests/test_day18_quality.py` — all pass
- [ ] `pytest tests/contracts/test_feature_factory_contract.py` — still green (Day 16 contracts)
- [ ] `pytest tests/contracts/test_wilcoxon_gate_contract.py` — still green (Day 13)
- [ ] `pytest tests/regression/` — Phase 1 + Phase 2 frozen tests still pass
- [ ] Manual check: `feature_manifest.json` contains `round: 3`, `round: 4`, `round: 5` entries
- [ ] Manual check: interaction count in manifest never exceeds 500
- [ ] `git commit -m "Day 18: feature factory rounds 3-5, interaction cap, pseudo-labeling — 52 tests green"`