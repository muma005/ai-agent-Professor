# Phase 2 Test Specifications — Consolidated Summary
**Days 8–15 | March 10–15, 2026**
**Branch:** `phase-2`
**Status:** All tests IMMUTABLE after their creation day

---

## Overview

| Day | Test File | Tests | Theme |
|-----|-----------|-------|-------|
| 8 | `tests/test_day8_quality.py` | 57 | ChromaDB semantics, state fields, validation architect, EDA agent, competition intel |
| 9 | `tests/test_day9_quality.py` | 54 | Circuit breaker escalation, subprocess sandbox, service health fallbacks, Redis persistence, parallel DAG |
| 10 | `tests/test_day10_quality.py` | 53 | Memory schema fingerprints, critic 6-vector coverage, severity escalation, replan instructions, preprocessing leakage |
| 12 | `tests/test_day12_quality.py` | 52 + 2 contract files | ML optimizer memory management, LangSmith cost control, HITL prompt generation, full HITL integration |
| 13 | `tests/test_day13_quality.py` | 48 + 7 contract tests (55) | Column order enforcement, data hash validation, Wilcoxon gate correctness, optimizer integration |
| 14 | `tests/test_day14_quality.py` | 30 + 7 gate + 5 regression (42) | Historical failures vector, query_critic_failure_patterns, compounding advantage, Phase 2 gate, regression freeze |
| 15 | `tests/test_day15_quality.py` | 48 + 9 contract tests (57) | Graph singleton, Docker sandbox, LangFuse observability, external data scout |
| **Total** | | **~370** | |

---

## Day 8 — ChromaDB, State, Validation, EDA, Intel

### Block 1: ChromaDB Embedding Semantics (6 tests)
**Class:** `TestChromaDBEmbeddingSemanticsNotJustStartup`

Catches: ChromaDB silent fallback to random embeddings — invisible failure, memory queries return garbage.

| Test | Bug It Catches |
|------|---------------|
| `test_embedding_dimension_is_exactly_384` | Wrong model loaded (dim ≠ 384) |
| `test_semantically_similar_query_returns_correct_top_result` | Random embeddings — gradient boosting query should return tabular doc, not NLP |
| `test_dissimilar_query_does_not_contaminate_top_result` | Embedding space too flat — all docs cluster together |
| `test_cosine_similarity_between_similar_docs_exceeds_threshold` | sim(LGBM,XGB) must be > 0.80 and > sim(LGBM,BERT)+0.10 |
| `test_client_bypassing_build_chroma_client_raises_runtime_error` | Direct `chromadb.Client()` bypasses validated embeddings |
| `test_embedding_is_deterministic_across_two_calls` | Non-deterministic embeddings break memory retrieval consistency |

### Block 2: State Fields Boundary Logic (8 tests)
**Class:** `TestStateFieldsBoundaryLogic`

| Test | Bug It Catches |
|------|---------------|
| `test_task_type_initial_value_is_unknown` | Field defaults to "tabular" before router runs |
| `test_data_hash_changes_when_file_contents_change` | Hash reads filename not contents |
| `test_data_hash_is_stable_for_identical_content` | Non-deterministic hashing |
| `test_data_hash_is_16_hex_characters` | Wrong truncation length |
| `test_strategy_at_conservative_boundary` (parametrized) | Off-by-one in strategy boundary conditions |
| `test_competition_context_has_all_required_keys` | Missing context keys = contract violation |
| `test_data_hash_written_to_state_after_data_engineer` | Data engineer never calls hash_dataset() |
| `test_model_registry_entry_contains_data_hash` | Ensemble mixes models from different data versions |

### Block 3: Validation Architect Strategy Correctness (14 tests)
**Class:** `TestValidationArchitectStrategyCorrectness`

| Test | Bug It Catches |
|------|---------------|
| `test_stratified_kfold_for_binary_target` | Wrong CV type for binary |
| `test_group_kfold_when_group_column_present` | Group column ignored → inflated CV |
| `test_timeseries_split_when_datetime_column_present` | Time structure ignored → future leaks into past |
| `test_group_kfold_takes_priority_over_datetime_column` | Priority ordering wrong when both exist |
| `test_kfold_for_continuous_target` | Stratification on continuous target |
| `test_n_splits_is_always_5` | n_splits not 5 across configurations |
| `test_mismatch_detected_stratified_plus_datetime` | Mismatch detector doesn't fire |
| `test_mismatch_reason_names_the_offending_column` | Vague mismatch reason |
| `test_no_false_positive_mismatch_on_clean_tabular` | False positive halts valid pipeline |
| `test_metric_contract_direction_correct_for_auc` | Optuna minimises AUC |
| `test_metric_contract_direction_correct_for_rmse` | Optuna maximises RMSE |
| `test_metric_contract_forbidden_metrics_is_non_empty_list` | Empty forbidden list allows accuracy on imbalanced data |
| `test_validation_strategy_json_written_even_when_mismatch_halts` | Agent raises instead of writing strategy file |
| `test_metric_contract_not_written_when_mismatch_halts` | Metric contract written before mismatch detected |

### Block 4: EDA Agent Threshold Accuracy (15+ tests)
**Class:** `TestEDAAgentThresholdAccuracy`

Covers outlier strategy thresholds (<1% keep, 1-5% winsorize, 5-10% cap, >10% remove), leakage detection, ID conflict detection, correlation vs feature importance, and EDA report completeness.

### Block 5: Competition Intel (14+ tests)
**Class:** `TestCompetitionIntelBriefQuality`

Covers brief.json structure, LLM fallback on Kaggle API failure, brief truncation for context limits, empty notebook handling, and forum scraper rate limiting.

---

## Day 9 — Resilience Layer

### Block 1: Circuit Breaker Escalation Precision (14 tests)
**Class:** `TestCircuitBreakerEscalationPrecision`

| Test | Bug It Catches |
|------|---------------|
| `test_failure_count_1_is_micro_not_macro` | Off-by-one — MACRO fires on first failure |
| `test_failure_count_2_is_macro_not_hitl` | Wrong threshold |
| `test_failure_count_3_is_hitl_not_macro` | Wrong threshold |
| `test_triage_fires_at_5pct_budget_not_10pct` | Triage threshold too high |
| `test_triage_overrides_failure_count_1` | Triage not checked before failure count |
| `test_triage_fires_at_2_hours_remaining_not_3` | Wrong time threshold |
| `test_micro_error_context_contains_full_traceback` | Only exception message stored, not traceback |
| `test_micro_failure_count_increments_by_exactly_1` | Increment off-by-one |
| `test_macro_dag_version_increments_from_current_value` | MACRO resets dag_version instead of incrementing |
| `test_macro_does_not_halt_pipeline` | MACRO sets pipeline_halted=True |
| `test_hitl_reason_contains_session_id_and_resume_command` | Vague HITL reason |
| `test_hitl_does_not_raise_when_redis_is_down` | Redis failure crashes circuit breaker |
| `test_reset_failure_count_clears_error_context` | Error context not cleared on reset |
| `test_handle_escalation_never_raises_for_any_level` | Any escalation level raises |

### Block 2: Subprocess Sandbox Real Execution (10 tests)
**Class:** `TestSubprocessSandboxRealExecution`

Covers numpy/polars/lightgbm/sklearn execution, timeout enforcement, returncode on crash, stdout capture, extra files in sandbox, memory limit, and existing contract compatibility.

### Block 3: Service Health Fallback Quality (8 tests)
**Class:** `TestServiceHealthFallbackQuality`

Covers Groq→Gemini fallback activation, warning logging (not silence), exponential backoff verification, service unavailable on all retries exhausted, Kaggle API 60s base delay, ChromaDB returns [] not None, Redis in-memory store persistence, and no false retries on success.

### Block 4: Docker Redis Persistence (8 tests)
**Class:** `TestDockerRedisPersistenceGuarantees`

Covers save/load state round-trip, nested dict preservation, non-serialisable value handling, type preservation after load.

### Block 5: Parallel DAG Execution (14 tests)
**Class:** `TestParallelDAGExecution`

Covers intelligence fan-out (competition_intel + data_engineer parallel), model trial fan-out, critic vector fan-out, fan-join state merging, error in one branch not blocking others, and timing verification (parallel < serial).

---

## Day 10 — Quality Conscience

### Block 1: Memory Schema Fingerprint Quality (10 tests)
**Class:** `TestMemorySchemaFingerprintQuality`

Covers fingerprint key completeness, n_rows_bucket boundaries, imbalance ratio as fraction not count, high-cardinality count excludes target, text contains imbalance/temporal language, different fingerprints produce different text, store/retrieve preserves validated approaches, empty collection returns [] not None, high-distance patterns filtered from warm start.

### Block 2: Critic Vector Coverage (6 tests)
**Class:** `TestCriticVectorCoverage`

| Test | Bug It Catches |
|------|---------------|
| `test_all_six_vectors_appear_in_verdict` | Critic silently skips vectors |
| `test_shuffled_target_vector_not_trivially_passing` | Vector always returns OK |
| `test_preprocessing_audit_vector_not_trivially_passing` | Regex never matches |
| `test_pr_curve_vector_not_trivially_passing` | Trigger condition wrong |
| `test_temporal_vector_checks_correct_thing` | Checks date columns, not row order correlation |
| `test_adversarial_classifier_vector_triggers_on_real_shift` | Misses N(0,1)→N(3,1) shift |

### Block 3: Critic Severity Escalation (8 tests)
**Class:** `TestCriticSeverityEscalation`

Covers max severity aggregation, clean data → OK, CRITICAL → hitl_required, CRITICAL → replan_requested, HIGH does NOT halt, MEDIUM does NOT halt, verdict JSON written to disk, clean flag false when findings exist.

### Block 4: Critic Replan Instructions (7 tests)
**Class:** `TestCriticReplanInstructions`

Covers CRITICAL finding has replan_instructions, rerun_nodes is non-empty list, remove_features is list not None, rerun node names are valid agents, preprocessing leakage reruns data_engineer, shuffled target reruns feature_factory, state replan fields aggregate all critical nodes.

### Block 5: Preprocessing Leakage Audit Precision (9 tests)
**Class:** `TestPreprocessingLeakageAuditPrecision`

| Test | Bug It Catches |
|------|---------------|
| `test_standard_scaler_before_split_is_flagged` | False negative — leakage not caught |
| `test_standard_scaler_inside_fold_is_clean` | False positive — clean code blocked |
| `test_minmaxscaler_before_split_is_flagged` | Only catches StandardScaler |
| `test_simple_imputer_before_split_is_flagged` | Imputer leakage missed |
| `test_target_encoder_before_split_is_flagged` | Target encoding leakage missed |
| `test_pca_before_split_is_flagged` | PCA leakage missed |
| `test_empty_code_string_returns_ok_not_crash` | Crash on empty input |
| `test_none_code_string_returns_ok_not_crash` | Crash on None input |
| `test_fit_transform_inside_pipeline_object_is_not_flagged` | False positive on sklearn Pipeline |

---

## Day 12 — Podium-Level Hardening

### Block 1: ML Optimizer Memory Management (14 tests)
**Class:** `TestMLOptimizerMemoryManagement`

Covers: models deleted in finally block, GC after del, per-fold memory check (not per-trial), TrialPruned on OOM (not MemoryError), oom_risk user attr, gc_after_trial flag, n_jobs=1 default, max_memory_gb from env var, invalid env var fallback, peak memory in state, oom_risk flag, pruned trial count.

### Block 2: LangSmith Cost Control (10 tests)
**Class:** `TestLangSmithTracingControl`

Covers: tracing disabled inside Optuna, tracing restored after Optuna, tracing restored on exception (try/finally), restored when original was false, restored when env var absent, sampling rate from env, sampling rate default 0.10, .env.example contains sampling rate, .env.example contains max_memory_gb, cost estimation logged.

### Block 3: HITL Prompt Generation (14 tests)
**Class:** `TestHITLPromptGeneration`

Covers: prompt has all 9 keys, exactly 3 interventions per error class, intervention has required keys, IDs are 1/2/3, error classification for data_quality/memory/api_timeout/unknown, error message truncated to 500 chars, prompt written to disk, write failure doesn't propagate, resume resets failure count, resume clears hitl_required, corrupt checkpoint returns error state.

### Block 4: Full HITL Integration (14 tests)
**Class:** `TestHITLFullIntegration`

Covers: 3x failure → HITL (not MACRO), state saved to Redis, checkpoint contains full state, pipeline halted after HITL, prompt generated with 3 interventions, error class matches injected error, AUTO intervention applies state change, MANUAL intervention doesn't change state, invalid intervention_id returns error.

---

## Day 13 — Submission Integrity

### Block 1: Column Order Enforcement (16 tests)
**Class:** `TestColumnOrderEnforcement`

Covers: feature_order in metrics.json, order matches training columns (not sorted), stored in state, submit loads from metrics (not state), select columns in training order, assert fires on wrong order, ValueError on missing column, FileNotFoundError on missing metrics.json, ValueError on missing feature_order key, correct order produces prediction, order preserved across Polars read, excludes target column, excludes ID columns, plus 3 contract tests.

### Block 2: Data Hash Validation (12 tests)
**Class:** `TestDataHashValidation`

Covers: passes when all hashes match, logs warning on mismatch, filters to current hash, raises when filtered registry empty, raises when registry empty, degrades when state hash None, degrades when entry missing hash, event logged to lineage, blend not called after filter empties registry, validation called before weight computation, state returned with filtered registry, single model registry works.

### Block 3: Wilcoxon Gate Correctness (12 tests)
**Class:** `TestWilcoxonGate`

Covers: returns true when significantly better, false when noise, false when worse, false when identical, never raises on mismatched folds, falls back to mean below 5 folds, never raises when scipy throws, p_threshold respected, gate_result has all keys, selected_model correct on pass/fail, mean_delta correct sign.

### Block 4: Optimizer Integration (8 tests)
**Class:** `TestWilcoxonGateOptimizerIntegration`

Covers: fold_scores in trial user_attrs, gate applied every comparison, non-significant trial not selected, significant trial selected, cross-model gate keeps simpler, cross-model selects complex when significant, gate decision logged with comparison type, graceful fallback when fold_scores unavailable.

---

## Day 14 — Compounding Advantage + Phase 2 Gate

### Block 1: Historical Failures Vector (14 tests)
**Class:** `TestHistoricalFailuresVector`

Covers: vector appears in vectors_checked (8 total), OK when collection empty, OK when no patterns within distance, OK when patterns found but feature not present, CRITICAL when high confidence feature present, HIGH for medium confidence, MEDIUM for low confidence, below 0.50 not flagged, verdict is max across patterns, substring feature matching, short string matching guard (min 4 chars), finding contains evidence string, CRITICAL includes replan instructions, ChromaDB failure returns OK.

### Block 2: query_critic_failure_patterns (8 tests)
**Class:** `TestQueryCriticFailurePatterns`

Covers: empty list when collection missing, empty when collection empty, returns within distance threshold, respects n_results limit, required metadata fields present, uses fingerprint_to_text for query, never raises on any input, sorted by distance ascending.

### Block 3: Compounding Advantage End-to-End (8 tests)
**Class:** `TestCompoundingAdvantage`

Covers: post_mortem write → critic read loop, dissimilar competition not retrieved, evidence includes competition name, multiple competitions accumulate, high confidence triggers replan, OK doesn't block pipeline, metadata format from post_mortem correct, 10th competition has more patterns than 1st.

### Block 4: Phase 2 Gate Conditions (7 tests)
**File:** `tests/phase2_gate.py`

- Critic catches target-derived feature (CRITICAL)
- Critic catches row_id as feature (CRITICAL)
- Critic clean on legitimate features (not CRITICAL)
- Validation architect blocks AUC on regression
- Validation architect blocks RMSE on binary
- Validation architect passes correct metric
- Phase 2 CV beats Phase 1 baseline by ≥ 0.005

### Block 5: Phase 2 Regression Freeze (5 tests)
**File:** `tests/regression/test_phase2_regression.py` — IMMUTABLE

- CV above Phase 1 baseline
- Critic CRITICAL on target-derived feature
- AUC blocked on regression target
- HITL triggered after 3 consecutive failures
- Wilcoxon gate rejects noise-level difference

---

## Day 15 — Phase 2 Finale

### Block 1: Graph Singleton (10 tests)
**Class:** `TestGraphSingleton`

| Test | Bug It Catches |
|------|---------------|
| `test_graph_compiled_only_once_across_multiple_invocations` | Missing `global _GRAPH` — function creates local variable |
| `test_get_graph_returns_same_object_on_repeated_calls` | Object identity check — same Python object, not just equal |
| `test_cache_clear_forces_recompilation_on_next_call` | cache_clear doesn't actually clear module-level variable |
| `test_run_professor_uses_get_graph_not_build_graph_directly` | run_professor still calls build_graph() directly |
| `test_singleton_thread_safe_under_concurrent_calls` | Two threads both compile (missing double-checked locking) |
| `test_singleton_resets_on_cache_clear_not_on_access` | Repeated access triggers recompilation |
| `test_cache_clear_sets_module_level_graph_to_none` | cache_clear clears local copy not module-level |
| `test_get_graph_builds_valid_compilable_graph` | Returned object has .invoke method |
| `test_compilation_time_improvement_over_repeated_calls` | Performance regression — something is recompiling |
| `test_conftest_cache_clear_fixture_resets_between_tests` | Test isolation — test A's graph leaks to test B |

### Block 2: Docker Sandbox (14 tests)
**Class:** `TestDockerSandbox`

Covers: stdout capture, stderr separation, backend field indicates docker/subprocess, container destroyed after success, container destroyed after timeout, unique container names, stdout capped at MAX_OUTPUT_BYTES, timed_out flag set/unset correctly, fallback to subprocess, --network none flag, --read-only flag, --memory flag, never raises on any input.

### Block 3: LangFuse Observability (10 tests)
**Class:** `TestLangFuseObservability`

Covers: disabled when keys absent, init called once at module load, one trace per invocation, flush called after run, trace not serialised into Redis checkpoint, span per outer node (not Optuna), error trace on pipeline exception, JSONL + LangFuse coexist, init failure falls back to JSONL, sampling rate env var respected.

### Block 4: External Data Scout (14 tests)
**Class:** `TestExternalDataScout`

Covers: skipped when not allowed, manifest written even when skipped, runs when allowed, schema validation catches missing keys, catches out-of-range relevance, scores are floats not strings, failure returns empty manifest not exception, state field set, only high relevance in recommended, data engineer logs high relevance sources, no log when manifest empty, total_sources_found matches length, prompt capped to 20 features, manifest JSON is valid.

**Known issue:** `test_data_engineer_logs_high_relevance_sources` fails — mock target mismatch (`agents.data_engineer.log_event` not imported at module level).

---

## Contract Tests (IMMUTABLE)

| Day | Contract Test File | Tests |
|-----|-------------------|-------|
| 8 | `test_validation_architect_contract.py` | ~8 |
| 9 | `test_circuit_breaker_contract.py` | ~5 |
| 10 | `test_critic_contract.py` | ~7 |
| 12 | `test_hitl_prompt_contract.py`, `test_resume_checkpoint_contract.py` | ~12 |
| 13 | `test_submit_column_order_contract.py` | 7 |
| 15 | `test_competition_intel_contract.py` | 9 |
| **Total** | | **~48** |

---

## Regression Tests (FROZEN at Phase Gates)

| Gate | File | Tests | Frozen At |
|------|------|-------|-----------|
| Phase 1 | `test_phase1_regression.py` | 5 | CV 0.8798, commit `b60b615` |
| Phase 2 | `test_phase2_regression.py` | 5 | Gate commit `57294eb` |

---

## Test Philosophy

1. **Tests do not ask "did it run?" — they ask "did it work?"**
2. Every test exists because a specific, real failure mode was anticipated
3. The test name IS the bug it catches
4. False negatives (missed bugs) are worse than false positives (over-strict)
5. Contract tests are IMMUTABLE after creation day
6. Regression tests are FROZEN at phase gates
7. Silent failures are the enemy — every test makes a silent failure loud
