# Professor Agent — Day 10 Test Specification
**For: Claude Code**
**Status: IMMUTABLE after Day 10**
**Philosophy: Day 10 is about quality conscience. Tests must prove Professor catches its own mistakes — not just that it runs.**

All tests in `tests/test_day10_quality.py`.

```bash
pytest tests/test_day10_quality.py -v --tb=short 2>&1 | tee tests/logs/day10_quality.log
```

---

## BLOCK 1 — MEMORY SCHEMA V2: FINGERPRINT QUALITY
**Class:** `TestMemorySchemaFingerprintQuality` — 10 tests
**The bug this block catches:** A fingerprint that is identical for all competitions (no discrimination), a text embedding that contains no semantic content, a round-trip that loses float precision, a query that returns `None` instead of `[]`.

---

### TEST 1.1 — `test_fingerprint_has_all_required_keys`
Build a fingerprint from a fully-populated state. Assert all 7 required keys present:
`task_type, imbalance_ratio, n_categorical_high_cardinality, n_rows_bucket, has_temporal_feature, n_features_bucket, target_type`.

---

### TEST 1.2 — `test_fingerprint_n_rows_bucket_correct_for_each_tier`
Bug: bucketing logic uses wrong boundary values.

| n_rows | expected bucket |
|---|---|
| 500 | "tiny" |
| 5000 | "small" |
| 50000 | "medium" |
| 500000 | "large" |
| 2000000 | "huge" |

Test each one. One off-by-one fails the whole tier.

---

### TEST 1.3 — `test_fingerprint_imbalance_ratio_is_fraction_not_count`
Bug: imbalance stored as minority count (e.g. 150) instead of ratio (e.g. 0.15).
Build a dataset where minority = 150, total = 1000. Assert `fingerprint["imbalance_ratio"] < 1.0` and is close to 0.15.

---

### TEST 1.4 — `test_fingerprint_high_cardinality_count_excludes_target`
Bug: target column counted as high-cardinality feature.
Build schema where target has 10 unique values and one feature has 200 unique values. Assert `n_categorical_high_cardinality == 1`, not 2.

---

### TEST 1.5 — `test_fingerprint_text_contains_imbalance_language_for_severe_imbalance`
Bug: text for severely imbalanced dataset (2% minority) uses generic language — embedding doesn't distinguish from balanced.

Fingerprint with `imbalance_ratio=0.02`. Text must contain at least one of: `"severely"`, `"fraud"`, `"anomaly"`, `"imbalanced"`.

---

### TEST 1.6 — `test_fingerprint_text_contains_temporal_language_when_dates_present`
Bug: temporal fingerprint produces identical text to non-temporal.

Fingerprint with `has_temporal_feature=True`. Text must contain at least one of: `"temporal"`, `"time-series"`, `"date"`.

---

### TEST 1.7 — `test_two_different_fingerprints_produce_different_text`
Bug: fingerprint_to_text produces identical output for different inputs (hash collision in text space).

Fingerprint A: `binary, imbalance=0.5, no temporal`. Fingerprint B: `binary, imbalance=0.03, temporal=True`. Assert `text_A != text_B`.

---

### TEST 1.8 — `test_store_and_retrieve_preserves_validated_approaches`
Bug: `store_pattern` saves the fingerprint but the `validated_approaches` list is truncated or lost.

Store pattern with 3 validated approaches. Query back. Assert retrieved pattern has exactly 3 approaches. Assert the approach strings match exactly.

---

### TEST 1.9 — `test_query_returns_empty_list_not_none_on_empty_collection`
Bug: `query_similar_competitions` returns `None` when collection is empty — downstream does `len(results)` and crashes.

Query against an empty collection path. Assert result `== []` (not `None`, not exception).

---

### TEST 1.10 — `test_high_distance_pattern_filtered_from_warm_start`
Bug: `get_warm_start_priors` returns priors from a highly dissimilar competition (distance > 0.8). These priors add noise, not signal.

Store a pattern with fingerprint `{task_type: "nlp", target_type: "multiclass"}`. Query with fingerprint `{task_type: "tabular", target_type: "binary", imbalance_ratio: 0.03}`. Assert the NLP pattern is NOT returned by `get_warm_start_priors` (distance too high).

---

## BLOCK 2 — RED TEAM CRITIC: ALL 6 VECTORS MUST RUN
**Class:** `TestCriticVectorCoverage` — 6 tests
**The bug this block catches:** A critic that silently skips vectors. A vector that throws an exception and is swallowed (so it appears to pass). A vector that always returns "OK" regardless of input.

---

### TEST 2.1 — `test_all_six_vectors_appear_in_verdict`
Run critic on any dataset. Assert `verdict["vectors_checked"]` contains all 6:
`shuffled_target, id_only_model, adversarial_classifier, preprocessing_audit, pr_curve_imbalance, temporal_leakage`.

**Failure:** If even one is missing, the critic is incomplete and must not pass this test.

---

### TEST 2.2 — `test_shuffled_target_vector_not_trivially_passing`
Bug: shuffled_target vector always returns "OK" because it doesn't actually train the model.

Inject a copy of the target column as a feature. Run the critic. Assert `shuffled_target` verdict is `CRITICAL`, not `OK`.

---

### TEST 2.3 — `test_preprocessing_audit_vector_not_trivially_passing`
Bug: preprocessing_audit always returns "OK" because the regex never matches anything.

Feed code with explicit `scaler.fit_transform(X)` before split. Assert vector returns `CRITICAL`.

---

### TEST 2.4 — `test_pr_curve_vector_not_trivially_passing`
Bug: pr_curve_imbalance always skips (trigger condition wrong — fires at 25% instead of 15%).

Build a state with `imbalance_ratio=0.08` and OOF predictions that always predict 0. Assert vector returns `CRITICAL`, not `OK` with "skipped" note.

---

### TEST 2.5 — `test_temporal_vector_checks_correct_thing`
Bug: temporal vector checks whether date columns exist, but not whether features are correlated with row order.

Build a dataset with a monotonically increasing numeric column (row_order_correlation = 0.99) and no date columns. Set `eda_report["temporal_profile"]["has_dates"] = True`. Assert temporal vector flags the monotonic column.

---

### TEST 2.6 — `test_adversarial_classifier_vector_triggers_on_real_shift`
Build two datasets: train has features sampled from N(0,1), test has features sampled from N(3,1) — extreme distribution shift. Run adversarial classifier vector. Assert verdict is at least `HIGH` (not `OK`).

---

## BLOCK 3 — CRITIC SEVERITY ESCALATION
**Class:** `TestCriticSeverityEscalation` — 8 tests
**The bug this block catches:** Wrong severity aggregation (max not taken), CRITICAL verdict that doesn't halt the pipeline, OK on clean data that shouldn't trigger, HIGH severity that incorrectly triggers HITL.

---

### TEST 3.1 — `test_overall_severity_is_max_of_all_findings`
Bug: overall_severity is set to the last finding's severity, not the maximum.

Inject findings: `[OK, MEDIUM, HIGH, CRITICAL, OK]`. Assert `overall_severity == "CRITICAL"`.

---

### TEST 3.2 — `test_clean_data_produces_ok_verdict`
Run critic on clean Spaceship Titanic data (no injected leakage). Assert `overall_severity == "OK"` and `clean == True` and `hitl_required` not set.

---

### TEST 3.3 — `test_critical_verdict_sets_hitl_required`
CRITICAL finding → `hitl_required=True` in returned state.

---

### TEST 3.4 — `test_critical_verdict_sets_replan_requested`
CRITICAL finding → `replan_requested=True` in returned state.

---

### TEST 3.5 — `test_high_severity_does_not_halt_pipeline`
**Bug:** HIGH severity (adversarial shift) incorrectly sets `hitl_required=True` — pipeline halted when it should continue with a warning.

Inject moderate train/test shift (adversarial AUC ~0.65). Assert `critic_severity == "HIGH"` AND `hitl_required` is NOT True.

---

### TEST 3.6 — `test_medium_severity_does_not_halt_pipeline`
Same as 3.5 but for MEDIUM.

---

### TEST 3.7 — `test_critic_verdict_json_written_to_disk`
Assert `outputs/{session_id}/critic_verdict.json` exists and is valid JSON after critic runs.

---

### TEST 3.8 — `test_verdict_clean_flag_is_false_when_findings_exist`
If `findings` list is non-empty, `clean` must be `False`. Even if all findings are MEDIUM severity.

---

## BLOCK 4 — CRITIC CONTRACT: REPLAN INSTRUCTIONS
**Class:** `TestCriticReplanInstructions` — 7 tests
**The bug this block catches:** CRITICAL finding with empty `replan_instructions`, `rerun_nodes` that lists nodes that don't exist, `remove_features` that is `None` instead of `[]`.

---

### TEST 4.1 — `test_critical_finding_has_replan_instructions`
Every finding with `severity == "CRITICAL"` must have `replan_instructions` key.

---

### TEST 4.2 — `test_replan_instructions_has_rerun_nodes`
`replan_instructions["rerun_nodes"]` must be a non-empty list for CRITICAL findings.

---

### TEST 4.3 — `test_replan_instructions_has_remove_features`
`replan_instructions["remove_features"]` must be a list (may be empty, but not absent and not None).

---

### TEST 4.4 — `test_rerun_nodes_names_are_valid_agent_names`
Bug: `rerun_nodes` contains `"feature engineering"` (invalid) instead of `"feature_factory"` (valid).

Assert every node name in `rerun_nodes` is a known agent: one of `{data_engineer, eda_agent, validation_architect, feature_factory, ml_optimizer, red_team_critic}`.

---

### TEST 4.5 — `test_preprocessing_leakage_rerun_includes_data_engineer`
Bug: preprocessing leakage triggers rerun of `ml_optimizer` only — but the fix must happen in `data_engineer`.

When preprocessing_audit finds CRITICAL: assert `"data_engineer"` is in `rerun_nodes`.

---

### TEST 4.6 — `test_shuffled_target_leakage_rerun_includes_feature_factory`
Injected target leakage → rerun must include `"feature_factory"` (where the leaking feature was created).

---

### TEST 4.7 — `test_state_replan_fields_aggregate_all_critical_nodes`
**Bug:** `state["replan_rerun_nodes"]` only contains the last CRITICAL finding's nodes, not the union of all.

Inject two CRITICAL findings: one for `feature_factory + ml_optimizer`, one for `data_engineer + ml_optimizer`. Assert `state["replan_rerun_nodes"]` contains all three unique nodes.

---

## BLOCK 5 — PREPROCESSING LEAKAGE AUDIT: STATIC ANALYSIS PRECISION
**Class:** `TestPreprocessingLeakageAuditPrecision` — 9 tests
**The bug this block catches:** The regex that either misses leakage (false negative) or flags clean code (false positive). These are the most dangerous test failures — false negatives allow bad models through, false positives block good code.**

---

### TEST 5.1 — `test_standard_scaler_before_split_is_flagged`
```python
code = "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nkf = KFold(5)\n..."
```
Assert `CRITICAL`.

---

### TEST 5.2 — `test_standard_scaler_inside_fold_is_clean`
```python
code = "for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):\n    scaler = StandardScaler()\n    X_train_s = scaler.fit_transform(X[train_idx])\n    X_val_s = scaler.transform(X[val_idx])"
```
Assert `OK`. False positive here breaks legitimate code.

---

### TEST 5.3 — `test_minmaxscaler_before_split_is_flagged`
Same pattern as 5.1 but with `MinMaxScaler`. Assert `CRITICAL`.

---

### TEST 5.4 — `test_simple_imputer_before_split_is_flagged`
```python
code = "imp = SimpleImputer()\nX_imp = imp.fit_transform(X)\ntrain_X, val_X = train_test_split(X_imp, ...)"
```
Assert `CRITICAL`.

---

### TEST 5.5 — `test_target_encoder_before_split_is_flagged`
```python
code = "te = TargetEncoder()\nX['cat'] = te.fit_transform(X['cat'], y)\nfor train_idx, val_idx in kf.split(X)..."
```
Assert `CRITICAL`.

---

### TEST 5.6 — `test_pca_before_split_is_flagged`
```python
code = "pca = PCA(n_components=10)\nX_pca = pca.fit_transform(X)\nX_train, X_val = train_test_split(X_pca)"
```
Assert `CRITICAL`.

---

### TEST 5.7 — `test_empty_code_string_returns_ok_not_crash`
Call `_check_preprocessing_leakage("")`. Assert result is `{"verdict": "OK", ...}`. Must not raise.

---

### TEST 5.8 — `test_none_code_string_returns_ok_not_crash`
Call `_check_preprocessing_leakage(None)`. Assert `OK`. Must not raise.

---

### TEST 5.9 — `test_fit_transform_inside_pipeline_object_is_not_flagged`
**Tricky false positive:** `Pipeline([('scaler', StandardScaler()), ('clf', LGBMClassifier())])`. The pipeline object handles fold-correct fitting internally — this is NOT leakage.

```python
code = "pipe = Pipeline([('scaler', StandardScaler()), ('clf', LGBMClassifier())])\ncv_scores = cross_val_score(pipe, X, y, cv=5)"
```
Assert `OK`. This is the most common false positive in naive implementations.

---

## BLOCK 6 — PR CURVE AUDIT: IMBALANCE DETECTION PRECISION
**Class:** `TestPRCurveAuditPrecision` — 8 tests
**The bug this block catches:** Wrong imbalance threshold (15% boundary), wrong recall threshold (50% boundary), PR audit running on multiclass, PR audit crashing when y_prob is None.**

---

### TEST 6.1 — `test_trigger_at_14pct_minority_not_16pct`
At `imbalance_ratio=0.14` → audit runs (below threshold).
At `imbalance_ratio=0.16` → audit skipped with "skipped" note.

Test both boundaries.

---

### TEST 6.2 — `test_recall_49pct_triggers_critical`
Build dataset: 5% minority. OOF probs give best-F1 recall of exactly 0.49. Assert `CRITICAL`.

---

### TEST 6.3 — `test_recall_51pct_does_not_trigger_critical`
Same dataset but OOF probs give best-F1 recall of 0.51. Assert NOT `CRITICAL`.

---

### TEST 6.4 — `test_pr_auc_barely_above_random_triggers_high_not_critical`
Build: `imbalance_ratio=0.08`, `pr_auc=0.10`, `random_baseline=0.08` (barely 1.25x above random, below 1.5x threshold). Assert `HIGH`, not `CRITICAL`.

---

### TEST 6.5 — `test_pr_audit_skipped_for_multiclass`
Call with `target_type="multiclass"`. Assert `OK` with note. Must not attempt PR curve computation.

---

### TEST 6.6 — `test_pr_audit_handles_none_y_prob_gracefully`
Call with `y_prob=None`. Assert `OK` with note. Must not raise `TypeError`.

---

### TEST 6.7 — `test_pr_auc_value_present_in_ok_result`
Even when verdict is `OK`, the result must contain `pr_auc` value. This is diagnostic information the engineer needs.

---

### TEST 6.8 — `test_random_baseline_is_imbalance_ratio`
Assert `result["random_baseline"] == imbalance_ratio`. The random baseline for PR-AUC on an imbalanced dataset equals the prevalence of the positive class, not 0.5.

---

## BLOCK 7 — INTEGRATION: FULL PIPELINE WITH CRITIC
**Class:** `TestFullPipelineWithCritic` — 5 tests
**The bug this block catches:** Critic that passes unit tests but breaks when inserted into the full LangGraph pipeline. A missing conditional edge. A state key that is set by the critic but never read by the routing logic.**

---

### TEST 7.1 — `test_critic_in_full_pipeline_does_not_break_routing`
Run full pipeline (competition_intel → data_engineer → eda_agent → validation_architect → red_team_critic). Assert pipeline reaches critic. Assert critic verdict written to state. Assert routing logic reads `critic_severity` correctly.

---

### TEST 7.2 — `test_critical_verdict_routes_to_hitl_not_ensemble`
**Bug:** LangGraph conditional edge routes to ensemble_architect even when `hitl_required=True`.

Inject leakage to force CRITICAL. Assert the graph's conditional edge routes to `hitl_handler`, not `ensemble_architect`. Assert `ensemble_architect` was never called.

---

### TEST 7.3 — `test_ok_verdict_routes_to_ensemble`
Clean data → CRITICAL must not trigger → routing proceeds to `ensemble_architect`.

---

### TEST 7.4 — `test_warm_start_priors_injected_into_ml_optimizer_state`
After a second run on a similar competition (pattern from first run is now in memory): assert `state["warm_start_priors"]` is non-empty when `ml_optimizer` runs.

---

### TEST 7.5 — `test_competition_fingerprint_written_to_state_after_eda`
After `run_eda_agent()` and `run_validation_architect()`, call `build_competition_fingerprint(state)`. Assert all keys populated. Assert `n_rows_bucket` is not `"tiny"` for Spaceship Titanic (it has 8693 rows → `"small"`).

---

## TOTAL: 53 TESTS

| Block | Class | Tests |
|---|---|---|
| 1 — Memory Schema Fingerprint | `TestMemorySchemaFingerprintQuality` | 10 |
| 2 — Critic Vector Coverage | `TestCriticVectorCoverage` | 6 |
| 3 — Critic Severity Escalation | `TestCriticSeverityEscalation` | 8 |
| 4 — Critic Replan Instructions | `TestCriticReplanInstructions` | 7 |
| 5 — Preprocessing Audit Precision | `TestPreprocessingLeakageAuditPrecision` | 9 |
| 6 — PR Curve Audit Precision | `TestPRCurveAuditPrecision` | 8 |
| 7 — Full Pipeline Integration | `TestFullPipelineWithCritic` | 5 |
| **Total** | | **53** |

---

## THE HARDEST TESTS (READ THESE FIRST)

**Test 5.9** — Pipeline object false positive. The naive regex for `fit_transform` will flag `cross_val_score(Pipeline([...]), X, y, cv=5)` as leakage because it contains `StandardScaler`. But this is correct code — sklearn Pipeline handles fold-correct fitting internally. The implementation must not flag this.

**Test 2.6** — Adversarial classifier on real shift. This test requires actually running a RandomForest to classify train vs test. If the vector skips this check "for speed" and always returns OK, this test catches it.

**Test 1.10** — High-distance filtering in warm-start. An NLP pattern should not influence tabular binary classification. If `get_warm_start_priors` returns all stored patterns regardless of similarity distance, this test catches it.

**Test 7.2** — LangGraph routing with CRITICAL verdict. This test is the end-to-end integration test that proves the circuit actually breaks when leakage is found. Without this test, the critic can return CRITICAL but the pipeline still trains and submits a leaking model.

---

## DEFINITION OF DONE

- [ ] All 53 tests written in `tests/test_day10_quality.py`
- [ ] All 53 pass
- [ ] `pytest tests/regression/` — green (Phase 1 + Day 8 + Day 9 unchanged)
- [ ] `pytest tests/contracts/test_critic_contract.py` — green, immutable
- [ ] `git commit -m "Day 10: 53 adversarial quality tests — memory schema v2, critic all 6 vectors, all green"`