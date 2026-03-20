# Professor Agent — Day 16 Test Specification
**Status: IMMUTABLE after Day 16** | **46 tests + 9 contract tests**
```bash
pytest tests/test_day16_quality.py -v --tb=short 2>&1 | tee tests/logs/day16_quality.log
pytest tests/contracts/test_feature_factory_contract.py -v
```

---

## BLOCK 1 — DIVERSITY-FIRST ENSEMBLE SELECTION (18 tests)
**Class:** `TestDiversityEnsembleSelection`

The bugs: correlation computed against best model only (not against current ensemble mean — which evolves as models are added). Prize candidate check uses OR instead of AND (low correlation OR competitive CV, not both). OOF validation passes silently when all OOF arrays are empty lists. Ensemble mean updated with equal weights during selection (correct) but the ensemble_oof in state uses the final unequal weights (inconsistent).

---

### TEST 1.1 — `test_anchor_is_highest_cv_model`
Build registry with 3 models: CVs 0.85, 0.87, 0.82. Assert `result["anchor"]` is the model with CV=0.87. Assert it appears first in `selected_models`.

---

### TEST 1.2 — `test_high_correlation_model_rejected`
Two models: anchor CV=0.87, candidate CV=0.86 but OOF correlation=0.98. Assert candidate in `selection_log` with `decision="REJECTED_TOO_CORRELATED"`. Assert candidate NOT in `selected_models`.

---

### TEST 1.3 — `test_correlation_threshold_at_exactly_0_97`
Candidate OOF correlation = 0.97 exactly. Assert `decision="REJECTED_TOO_CORRELATED"` (correlation > 0.97 is reject, 0.97 is the boundary — must be `>` not `>=`). Test both sides: 0.970 → rejected, 0.969 → evaluated.

---

### TEST 1.4 — `test_diverse_model_selected_over_higher_cv_correlated_model`
Model A: CV=0.863, OOF correlation with anchor=0.95. Model B: CV=0.855, OOF correlation=0.70. Assert Model B selected before Model A (diversity score: B = 0.855 × 0.30 = 0.257, A = 0.863 × 0.05 = 0.043). Diversity wins over marginal CV gain.

---

### TEST 1.5 — `test_correlation_computed_against_ensemble_mean_not_anchor_only`
**Bug:** Correlation computed against anchor OOF rather than current ensemble OOF mean. Once 3 models are in the ensemble, a new candidate's correlation should be with the ensemble mean, not just the anchor.

Add anchor, then model B. When evaluating model C: assert correlation is computed against `mean(anchor_oof, model_B_oof)`, not just `anchor_oof`.

---

### TEST 1.6 — `test_ensemble_oof_mean_updated_after_each_selection`
After selecting anchor: assert `ensemble_oof_mean == anchor_oof`. After selecting model B: assert `ensemble_oof_mean == (anchor_oof + model_B_oof) / 2`. The running mean must update correctly.

---

### TEST 1.7 — `test_prize_candidate_requires_both_conditions`
**Bug:** Prize candidate check uses OR — any low-correlation model becomes a prize candidate regardless of CV.

Model with correlation=0.80 but CV=0.75 (much worse than best CV=0.87, delta=0.12 > 0.01): assert NOT a prize candidate. Both conditions required: `corr < 0.85` AND `abs(cv - best_cv) <= 0.01`.

---

### TEST 1.8 — `test_prize_candidate_identified_correctly`
Model with correlation=0.72 and CV=0.868 (delta from best=0.002 <= 0.01). Assert it appears in `prize_candidates` with both `correlation` and `cv_delta_from_best` fields.

---

### TEST 1.9 — `test_max_ensemble_size_respected`
Build registry with 10 models, all low-correlation. Assert `len(selected_models) <= MAX_ENSEMBLE_SIZE`. Models beyond the cap have `decision="SKIPPED_MAX_SIZE"` in `selection_log`.

---

### TEST 1.10 — `test_validate_oof_present_raises_on_missing`
Registry entry with `oof_predictions=[]` (empty list). Assert `ValueError` raised before any selection begins. Error message must name the model with missing OOF.

---

### TEST 1.11 — `test_validate_oof_present_raises_on_absent_key`
Registry entry missing `oof_predictions` key entirely. Assert `ValueError`. Not `KeyError`.

---

### TEST 1.12 — `test_selection_log_has_all_models`
Build registry with 5 models. Assert `len(result["selection_log"]) == 5`. Every model must appear in the log — selected, rejected, or skipped. No model silently disappears.

---

### TEST 1.13 — `test_correlation_matrix_contains_selected_pairs`
3 models selected. Assert `correlation_matrix` contains `n*(n-1)/2 = 3` pairwise entries. Keys formatted as `"{model_a}_vs_{model_b}"`.

---

### TEST 1.14 — `test_equal_weights_sum_to_one`
`sum(ensemble_weights.values()) == 1.0` (or very close: abs < 1e-9). Equal weights: `1/n` per model.

---

### TEST 1.15 — `test_single_model_registry_returns_that_model`
Registry with exactly 1 model. Assert `selected_models == [model_name]`. Assert `len(prize_candidates) == 0`. Single-model "ensemble" is valid.

---

### TEST 1.16 — `test_empty_registry_raises_value_error`
`model_registry={}`. Assert `ValueError` with message "empty". Must not return empty selection.

---

### TEST 1.17 — `test_diversity_selection_called_before_blend`
**Bug:** Naive top-N blending still present, diversity selection called after blending (result ignored).

Monkeypatch `select_diverse_ensemble` to track calls. Monkeypatch `_blend_models_weighted` to track calls. Assert `select_diverse_ensemble` called before `_blend_models_weighted`.

---

### TEST 1.18 — `test_selection_result_written_to_state`
After `blend_models()`: assert all 5 new state fields set:
`ensemble_selection`, `selected_models`, `ensemble_weights`, `ensemble_oof`, `prize_candidates`.
All must be non-None and have correct types.

---

## BLOCK 2 — FEATURE FACTORY ROUND 1 (10 tests)
**Class:** `TestFeatureFactoryRound1`

---

### TEST 2.1 — `test_log1p_candidate_generated_for_positive_numeric`
Schema column: `{"name": "Fare", "dtype": "float64", "min": 0.0, "n_unique": 281, "is_id": false, "is_target": false}`. Assert `"log1p_Fare"` appears in Round 1 candidates.

---

### TEST 2.2 — `test_log1p_not_generated_for_negative_min`
Column with `"min": -5.0`. Assert no `"log1p_*"` candidate generated. Log-transform undefined for negative values.

---

### TEST 2.3 — `test_missingness_flag_generated_for_high_null_fraction`
Column with `"null_fraction": 0.15`. Assert `"missing_{name}"` candidate generated.

---

### TEST 2.4 — `test_missingness_flag_not_generated_for_complete_columns`
Column with `"null_fraction": 0.0`. Assert no `"missing_{name}"` candidate.

---

### TEST 2.5 — `test_no_candidates_for_id_columns`
Column with `"is_id": true`. Assert ZERO candidates generated from this column (no log, sqrt, or missingness flag).

---

### TEST 2.6 — `test_no_candidates_for_target_column`
Column with `"is_target": true`. Assert ZERO candidates from this column.

---

### TEST 2.7 — `test_round_field_is_1_for_all_round1_candidates`
All candidates from `_generate_round1_features()` have `round == 1`.

---

### TEST 2.8 — `test_apply_round1_transforms_produces_new_columns`
Apply Round 1 transforms to a toy DataFrame. Assert new columns exist in result. Assert original columns preserved. Row count unchanged.

---

### TEST 2.9 — `test_log1p_transform_is_log_base_e_not_base_2`
Apply log1p transform. For input value 1.0: assert result ≈ `math.log1p(1.0) = 0.693`. Not `math.log2(2.0) = 1.0`. Base e (natural log) required for standard interpretation.

---

### TEST 2.10 — `test_missingness_flag_is_binary_int_not_boolean`
Apply missingness flag. Assert output column dtype is `Int8` (or `Int32`). Not `Boolean`. LightGBM handles 0/1 int better than boolean in some edge cases.

---

## BLOCK 3 — FEATURE FACTORY ROUND 2 (8 tests)
**Class:** `TestFeatureFactoryRound2`

---

### TEST 3.1 — `test_round2_candidates_have_round_2_field`
All candidates from `_generate_round2_features()` have `round == 2`.

---

### TEST 3.2 — `test_round2_rejects_candidates_with_unknown_source_columns`
Mock LLM to return a candidate with `source_columns=["NonExistentColumn"]`. Assert this candidate is NOT added to the list. Warning logged.

---

### TEST 3.3 — `test_round2_capped_at_15_candidates`
Mock LLM to return 25 candidates. Assert `len(round2_candidates) <= 15`.

---

### TEST 3.4 — `test_round2_graceful_on_llm_failure`
Monkeypatch `llm_call` to raise `TimeoutError`. Assert `_generate_round2_features()` returns empty list. No exception propagated.

---

### TEST 3.5 — `test_round2_graceful_on_invalid_json_response`
Mock LLM to return `"not valid json at all"`. Assert returns empty list. No `json.JSONDecodeError` propagated.

---

### TEST 3.6 — `test_round2_uses_domain_from_competition_brief`
Assert the LLM prompt contains the `domain` value from `competition_brief.json`. Verify by monkeypatching `llm_call` to capture the prompt string.

---

### TEST 3.7 — `test_round2_uses_known_winning_features`
`competition_brief["known_winning_features"] = ["Title from Name", "Family size"]`. Assert prompt contains at least one of these strings.

---

### TEST 3.8 — `test_round2_runs_without_competition_brief`
`competition_brief.json` does not exist. Assert `_generate_round2_features()` returns empty list gracefully. No `FileNotFoundError`.

---

## BLOCK 4 — FEATURE MANIFEST AND CONTRACT (10 tests)
**Class:** `TestFeatureManifest`

---

### TEST 4.1 — `test_manifest_written_to_correct_path`
After `run_feature_factory()`: assert `outputs/{session_id}/feature_manifest.json` exists.

---

### TEST 4.2 — `test_manifest_counts_consistent`
`total_candidates == len(features)`. `total_kept + total_dropped == sum of KEEP+DROP verdicts`. `total_candidates >= total_kept + total_dropped` (PENDING verdicts allowed in Day 16).

---

### TEST 4.3 — `test_manifest_has_generated_at_timestamp`
`manifest["generated_at"]` is a valid ISO 8601 timestamp. Assert `datetime.fromisoformat()` does not raise.

---

### TEST 4.4 — `test_feature_state_fields_set_correctly`
After `run_feature_factory()`: `state["feature_candidates"]` is a list of strings. `state["round1_features"]` is a subset of `state["feature_candidates"]`. `state["round2_features"]` is a subset of `state["feature_candidates"]`.

---

### TEST 4.5 — `test_manifest_empty_when_all_columns_are_id_or_target`
Schema with only id and target columns. Assert `len(manifest["features"]) == 0`. Assert `total_candidates == 0`. No crash.

---

### TEST 4.6 — `test_run_feature_factory_raises_on_missing_schema`
`schema.json` absent. Assert `FileNotFoundError` with message mentioning `schema.json` and `data_engineer`.

---

### TEST 4.7 — `test_lineage_event_written_after_feature_factory`
After `run_feature_factory()`: `lineage.jsonl` contains entry with `action="feature_factory_complete"` and `round1_candidates`, `round2_candidates` fields.

---

### TEST 4.8 — `test_feature_candidates_excludes_target_and_id`
`state["feature_candidates"]` must not contain the target column name or id column name.

---

### TEST 4.9 — `test_all_pending_verdicts_in_day16_stub`
In Day 16 (before Day 17 filtering): assert all features have `verdict == "KEEP"`. The stub sets all to KEEP — Day 17 will set real verdicts. This test verifies Day 16 does not accidentally pre-filter.

---

### TEST 4.10 — `test_day13_ensemble_contracts_still_pass`
Run `pytest tests/contracts/test_wilcoxon_gate_contract.py`. Assert all 4 original contracts pass. Day 16 diversity selection additions must not break the model-level Wilcoxon contract.

---

## THE 5 BUGS THAT WILL DEFINITELY BE PRESENT

**Bug 1 — Test 1.5:** Correlation computed against anchor OOF not ensemble mean. The greedy selection loop updates `ensemble_oof_mean` but the correlation check inside the loop still reads `oof_arrays[anchor_name]` (the initial value). As the ensemble grows, all new candidates are correlated against only the first model — not the evolving blend. After selecting 3 models, the 4th candidate's true diversity contribution relative to the blend is not measured.

**Bug 2 — Test 1.7:** Prize candidate check uses `or` instead of `and`. Code reads `if corr < 0.85 or abs(cv - best_cv) <= 0.01`. Any model with low correlation (including terrible models) becomes a prize candidate. The semantic is "diverse AND competitive" — both conditions must hold.

**Bug 3 — Test 1.3:** Boundary condition `>` vs `>=` on the correlation threshold. If the code uses `>= 0.97` for rejection, a candidate with exactly 0.97 correlation is rejected. The spec says reject if `> 0.97` — models at exactly 0.97 are borderline useful and should be evaluated, not rejected.

**Bug 4 — Test 2.9:** `log1p` implemented as `log(base=2)` instead of natural log. Polars' `.log(base=...)` API requires specifying the base explicitly. The natural implementation might use `pl.col(src).log(base=10)` (log10) or `pl.col(src).log(base=2)` (log2) — neither is the standard feature engineering log1p. Correct: `(pl.col(src) + 1.0).log(base=math.e)` or equivalently the dedicated `.log1p()` method if available in the Polars version.

**Bug 5 — Test 3.2:** No validation of source columns against schema for Round 2 candidates. LLM returns a candidate with `source_columns=["family_size"]` — a column that doesn't exist in the competition's CSV. The feature is added to the manifest. When `_apply_round2_transforms()` runs against the actual DataFrame, it raises `ColumnNotFoundError`. The validation must happen at candidate generation time (in `_generate_round2_features()`), not at transform application time.

---

## TOTAL: 46 + 9 CONTRACT TESTS

| Block | Tests |
|---|---|
| 1 — Diversity ensemble selection | 18 |
| 2 — Feature factory Round 1 | 10 |
| 3 — Feature factory Round 2 | 8 |
| 4 — Feature manifest and contract | 10 |
| Feature factory contract file | 9 |
| **Total** | **55** |

---

## DEFINITION OF DONE

- [ ] 46 tests in `tests/test_day16_quality.py` — all pass
- [ ] `tests/contracts/test_feature_factory_contract.py` — 9 tests, all pass, immutable
- [ ] `pytest tests/contracts/` — all previous contracts still pass
- [ ] `pytest tests/regression/` — Phase 1 + Phase 2 frozen tests still pass
- [ ] Day 17 prerequisite verified: `feature_manifest.json` has `verdict="PENDING"` for all features (stub correctly in place for Day 17 to replace)
- [ ] `git commit -m "Day 16: diversity ensemble, feature factory rounds 1+2, contract test — 46 tests green"`