# Professor Agent — Day 14 Test Specification
**Status: IMMUTABLE after Day 14** | **44 tests + 5 frozen regression tests + 7 gate conditions**
```bash
pytest tests/test_day14_quality.py -v --tb=short 2>&1 | tee tests/logs/day14_quality.log
pytest tests/phase2_gate.py -v                    # run once, after quality suite passes
pytest tests/regression/ -v                       # after gate passes and regression frozen
```

---

## BLOCK 1 — HISTORICAL FAILURES VECTOR: CORRECTNESS (14 tests)
**Class:** `TestHistoricalFailuresVector`

The bugs here: distance filter not applied (irrelevant competitions fire). Confidence threshold off-by-one (0.70 returns MEDIUM instead of HIGH). Feature matching too loose (every short string matches everything). Collection-empty case raises instead of returning OK.

---

### TEST 1.1 — `test_historical_vector_appears_in_vectors_checked`
**Bug:** Vector 8 added to `VECTOR_FUNCTIONS` but not to the orchestrator's `vectors_checked` list build.

Run critic. Assert `"historical_failures" in verdict["vectors_checked"]`. Assert `len(verdict["vectors_checked"]) == 8`. Count is definitive — 7 means the wire was missed.

---

### TEST 1.2 — `test_returns_ok_when_collection_empty`
**Bug:** `client.get_collection()` raises when collection doesn't exist (before first competition) — propagates as unhandled exception.

Drop/skip `critic_failure_patterns` collection. Call `_check_historical_failures()`. Assert returns `{"verdict": "OK", "note": "...", "patterns_retrieved": 0, "findings": []}`. Must not raise.

---

### TEST 1.3 — `test_returns_ok_when_no_patterns_within_distance`
Store 3 patterns with dissimilar fingerprints (distance > 0.75 from current fingerprint). Call `_check_historical_failures()`. Assert `verdict == "OK"`, `patterns_retrieved == 0`. Distance filter must exclude irrelevant competitions.

---

### TEST 1.4 — `test_returns_ok_when_patterns_found_but_feature_not_present`
Store a pattern flagging `"target_enc_cabin"`. Current `feature_names` does not contain `"target_enc_cabin"` or any similar name. Assert `verdict == "OK"`. Pattern retrieved but no match — should not fire.

---

### TEST 1.5 — `test_critical_when_high_confidence_feature_present`
Store pattern: `confidence=0.90`, `feature_flagged="target_enc_cabin"`, matching fingerprint (distance < 0.75). Current `feature_names = ["target_enc_cabin", "age", "fare"]`. Assert `verdict == "CRITICAL"`.

---

### TEST 1.6 — `test_high_when_medium_confidence_feature_present`
Pattern: `confidence=0.75`. Feature present. Assert `verdict == "HIGH"`.

---

### TEST 1.7 — `test_medium_when_low_confidence_feature_present`
Pattern: `confidence=0.55`. Feature present. Assert `verdict == "MEDIUM"`.

---

### TEST 1.8 — `test_below_0_50_confidence_not_flagged`
Pattern: `confidence=0.45`. Feature present. Assert `verdict == "OK"`. Sub-threshold confidence must be silently ignored — not even MEDIUM.

---

### TEST 1.9 — `test_verdict_is_max_across_multiple_patterns`
Patterns: one HIGH-confidence match + one MEDIUM-confidence match. Assert overall `verdict == "HIGH"`. Max severity across all matches, not last match.

---

### TEST 1.10 — `test_feature_matching_uses_substring_check`
Pattern flags `"target_enc"`. Current features include `"target_enc_cabin"`. Assert the feature is considered matched (substring `"target_enc"` is in `"target_enc_cabin"`). Assert `verdict != "OK"`.

---

### TEST 1.11 — `test_feature_matching_not_too_loose`
**Bug:** Short pattern name like `"id"` matches every feature containing the letter sequence "id" (e.g. `"period"`, `"invalid"`, `"grid_ref"`).

Pattern flags `"id"`. Current features: `["period", "latitude", "validity_score"]`. None of these should be flagged — exact substring match `"id" in "period"` is True but semantically wrong. Assert `verdict == "OK"`.

The fix: require minimum match length — if `feature_flagged` has fewer than 4 characters, use exact-match-only (no substring). Assert the 4-character minimum guard is in place.

---

### TEST 1.12 — `test_finding_contains_evidence_string`
When a finding is generated: assert `finding["evidence"]` contains the competition name, the failure mode, and the cv_lb_gap value. The evidence must be human-readable — not just metadata.

---

### TEST 1.13 — `test_critical_finding_includes_replan_instructions`
CRITICAL finding: assert `finding["replan_instructions"]["remove_features"]` contains the flagged feature name. Assert `finding["replan_instructions"]["rerun_nodes"]` contains `"feature_factory"`. CRITICAL historical failure must trigger replan.

---

### TEST 1.14 — `test_chromadb_failure_returns_ok_not_exception`
**Bug:** ChromaDB connection fails (Redis down, wrong path) → unhandled exception crashes the entire critic run.

Monkeypatch `build_chroma_client()` to raise `ConnectionError`. Assert `_check_historical_failures()` returns `{"verdict": "OK", "note": "ChromaDB query failed..."}`. Never raises.

---

## BLOCK 2 — `query_critic_failure_patterns()` (8 tests)
**Class:** `TestQueryCriticFailurePatterns`

---

### TEST 2.1 — `test_returns_empty_list_when_collection_missing`
`critic_failure_patterns` collection does not exist. Assert returns `[]`. No exception. This is the first-competition state.

---

### TEST 2.2 — `test_returns_empty_list_when_collection_empty`
Collection exists but `count() == 0`. Assert returns `[]`.

---

### TEST 2.3 — `test_returns_patterns_within_distance_threshold`
Store 5 patterns: 3 with distance < 0.75, 2 with distance > 0.75. Query with `max_distance=0.75`. Assert exactly 3 returned.

---

### TEST 2.4 — `test_respects_n_results_limit`
Store 10 patterns all within distance. Query with `n_results=5`. Assert at most 5 returned.

---

### TEST 2.5 — `test_returned_patterns_have_required_metadata_fields`
Each returned pattern must have: `competition_name, feature_flagged, failure_mode, cv_lb_gap, confidence, distance`. Assert all fields present. Missing field = `get()` fallback was used correctly.

---

### TEST 2.6 — `test_uses_fingerprint_to_text_for_query`
**Bug:** Query uses `str(fingerprint)` instead of `fingerprint_to_text(fingerprint)` — produces non-semantic text that doesn't embed well.

Monkeypatch `fingerprint_to_text` to track calls. Assert it was called exactly once per `query_critic_failure_patterns()` invocation.

---

### TEST 2.7 — `test_never_raises_on_any_input`
Call with `fingerprint={}`, `fingerprint=None`, `n_results=0`, `max_distance=0.0`. Assert none of these raise. All return `[]`.

---

### TEST 2.8 — `test_patterns_sorted_by_distance_ascending`
Return 3 patterns with distances 0.6, 0.3, 0.5. Assert returned list is sorted by distance ascending (0.3, 0.5, 0.6). Most similar pattern first.

---

## BLOCK 3 — COMPOUNDING ADVANTAGE: END-TO-END (8 tests)
**Class:** `TestCompoundingAdvantage`

These tests prove the full feedback loop works: post_mortem writes → critic reads → patterns improve over competitions.

---

### TEST 3.1 — `test_pattern_written_by_post_mortem_is_retrieved_by_critic`
Step 1: Run `store_critic_failure_pattern()` with a specific `feature_flagged="target_enc_cabin"` and matching fingerprint.
Step 2: Call `_check_historical_failures()` with matching fingerprint and `feature_names=["target_enc_cabin"]`.
Assert the stored pattern is retrieved and flagged. Full write → read loop.

---

### TEST 3.2 — `test_pattern_not_retrieved_for_dissimilar_competition`
Store pattern for NLP/multiclass fingerprint. Query with tabular/binary fingerprint (large distance). Assert pattern NOT retrieved. Cross-competition contamination must be blocked by distance filter.

---

### TEST 3.3 — `test_critic_verdict_includes_historical_context_in_evidence`
When a historical pattern fires: assert `verdict["findings"][0]["evidence"]` contains the source competition name. The engineer must be able to trace which past competition triggered this warning.

---

### TEST 3.4 — `test_multiple_competitions_accumulate_in_collection`
Run `store_critic_failure_pattern()` three times (three different competitions). Assert `collection.count() == 3`. Patterns accumulate — they do not overwrite each other.

---

### TEST 3.5 — `test_high_confidence_historical_pattern_triggers_replan`
CRITICAL historical finding with `confidence=0.90`. Run full critic. Assert `state["replan_requested"] == True` (CRITICAL routing via supervisor). Historical findings route through the same CRITICAL path as static vectors.

---

### TEST 3.6 — `test_historical_vector_ok_does_not_block_pipeline`
No historical patterns stored. Assert critic returns `overall_severity != "CRITICAL"` on clean data. Absence of historical patterns must not block the pipeline.

---

### TEST 3.7 — `test_critic_failure_pattern_metadata_format_from_post_mortem`
Run `run_post_mortem_agent()` with `gap_root_cause="critic_missed"`. Load the stored pattern from ChromaDB. Assert all 7 required metadata fields are present and non-empty. Confirms post_mortem writes the format that critic reads.

---

### TEST 3.8 — `test_10th_competition_has_more_patterns_than_1st`
Simulate: store 0 patterns before competition 1 critic run, store 5 patterns before competition 10 critic run. Assert `len(retrieved_patterns_comp10) > len(retrieved_patterns_comp1)`. Professor compounding advantage is structurally verified.

---

## BLOCK 4 — PHASE 2 GATE CONDITIONS (7 tests)
**File:** `tests/phase2_gate.py`
**Run:** `pytest tests/phase2_gate.py -v` — explicitly, after quality suite passes.

### Condition 1: Critic catches leakage (3 tests)
- `test_critic_catches_target_derived_feature` — CRITICAL on target-shifted feature
- `test_critic_catches_id_as_feature` — CRITICAL on row_id-as-predictor
- `test_critic_clean_on_legitimate_features` — NOT CRITICAL on clean features (false-positive guard)

### Condition 2: Validation Architect blocks wrong metric (3 tests)
- `test_validation_architect_blocks_auc_on_regression` — blocks AUC on continuous target
- `test_validation_architect_blocks_rmse_on_binary` — blocks RMSE on binary target
- `test_validation_architect_passes_correct_metric` — does NOT block correct combination

### Condition 3: Phase 2 CV beats Phase 1 baseline (1 test)
- `test_phase2_cv_beats_phase1_baseline` — improvement >= 0.005 over Phase 1 CV

---

## BLOCK 5 — PHASE 2 REGRESSION FREEZE (5 tests)
**File:** `tests/regression/test_phase2_regression.py`
**Written:** After Phase 2 gate passes. IMMUTABLE.

- `test_cv_above_phase1_baseline` — Phase 2 CV >= Phase 1 floor - 0.002
- `test_critic_critical_on_target_derived_feature` — always CRITICAL (no regressions)
- `test_auc_blocked_on_regression_target` — always blocked (no regressions)
- `test_hitl_triggered_after_3_consecutive_failures` — always fires on 3rd failure
- `test_gate_returns_false_for_noise_level_difference` — Wilcoxon always rejects noise

---

## THE 5 BUGS THAT WILL DEFINITELY BE PRESENT

**Bug 1 — Test 1.2:** `client.get_collection()` raises `ValueError` when collection doesn't exist (first competition — no patterns stored yet). Must be caught and return `[]`, not propagate. Every ChromaDB `get_collection` call on a potentially-absent collection needs a `try/except`.

**Bug 2 — Test 1.11:** Feature matching too loose. `"id" in "period"` is `True`. Short pattern names (2-3 chars) match everything. Minimum length guard of 4 characters required before substring matching is allowed.

**Bug 3 — Test 2.6:** Query text is `str(fingerprint)` — produces something like `"{'task_type': 'binary', 'n_rows_bucket': 'medium', ...}"`. Not semantic. ChromaDB embeds it but the embedding is meaningless. Must use `fingerprint_to_text()` which produces human-readable natural language that embeds correctly.

**Bug 4 — Test 1.1:** `vectors_checked` hardcoded as a 7-element list in the critic orchestrator — not built dynamically from `VECTOR_FUNCTIONS.keys()`. Adding vector 8 to the function dict doesn't automatically add it to `vectors_checked`. Must be `list(VECTOR_FUNCTIONS.keys())` not a hardcoded list.

**Bug 5 — Test 3.5:** CRITICAL historical finding uses its own escalation path instead of routing through the existing CRITICAL handling in `handle_escalation()`. The historical vector must produce the same `replan_instructions` format as static vectors so the supervisor can union them correctly. Custom escalation = two divergent code paths that get out of sync.

---

## TOTAL TEST COUNT

| Block | Tests |
|---|---|
| 1 — Historical failures vector | 14 |
| 2 — `query_critic_failure_patterns()` | 8 |
| 3 — Compounding advantage end-to-end | 8 |
| 4 — Phase 2 gate conditions | 7 |
| 5 — Phase 2 regression freeze (immutable) | 5 |
| **Total** | **42** |

---

## DEFINITION OF DONE

- [ ] 30 tests in `tests/test_day14_quality.py` (Blocks 1–3) — all pass
- [ ] `pytest tests/phase2_gate.py -v` — all 7 conditions pass
- [ ] `tests/regression/phase2_baseline.json` written by gate (auto-written on pass)
- [ ] `tests/regression/test_phase2_regression.py` written with commit hash in header
- [ ] `pytest tests/regression/` — all 5 frozen tests pass (including Phase 1)
- [ ] `git commit -m "Day 14: GM-CAP 4, Phase 2 gate passed, regression frozen"`

**After this commit: Phase 2 is locked. Phase 3 begins.**