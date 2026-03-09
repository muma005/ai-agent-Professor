# Professor Agent — Day 8 Test Specification
**For: Claude Code**
**Status: IMMUTABLE after Day 8 — never edit these tests once written**
**Philosophy: These tests do not ask "did it run?" They ask "did it work?"**

Every test in this document exists because a specific, real failure mode was anticipated. The test name is the bug it catches. If a test feels too strict, that is intentional — production ML systems fail silently, and these tests exist to make silent failures loud.

---

## ⚠️ PRE-TEST SETUP REQUIREMENTS

Before writing any test code, read these files in full:
- `memory/chroma_client.py` — understand exactly how the embedding function is built
- `agents/validation_architect.py` — understand the mismatch detection logic
- `agents/eda_agent.py` — understand the outlier thresholds and leakage formula
- `agents/competition_intel.py` — understand the brief structure and fallback logic
- `core/state.py` — understand all new Phase 2 fields and their defaults

All tests live in `tests/test_day8_quality.py`. Run the full file with:
```bash
pytest tests/test_day8_quality.py -v --tb=short 2>&1 | tee tests/logs/day8_quality.log
```

---

## BLOCK 1 — CHROMADB: EMBEDDING SEMANTICS
**File:** `TestChromaDBEmbeddingSemanticsNotJustStartup`
**The bug this block catches:** ChromaDB silently falls back to random embeddings. The system starts without error. Memory queries return garbage. Optuna warm-start is pure noise. The failure is invisible unless you test semantic correctness.

---

### TEST 1.1 — `test_embedding_dimension_is_exactly_384`
**Bug it catches:** Wrong model loaded (e.g. `all-MiniLM-L12-v2` has dim 384 too but different semantics; a completely wrong model may have dim 768 or 1536).

**What to test:**
Call `_build_embedding_function()` directly. Embed the string `"test"`. Assert `len(embedding[0]) == 384`. Not "approximately 384". Exactly 384.

**Failure message must say:**
`"Expected embedding dim 384, got {N}. Wrong model is loaded or ChromaDB fell back to a different embedding."`

---

### TEST 1.2 — `test_semantically_similar_query_returns_correct_top_result`
**Bug it catches:** Random embeddings. With random embeddings, a query about gradient boosting has equal probability of returning an NLP document as a tabular one.

**What to test:**
Pre-populate a test collection with exactly 3 documents:
- `"doc_lgbm"`: `"LightGBM gradient boosting tabular classification AUC"`
- `"doc_lstm"`: `"LSTM recurrent neural network time series forecasting"`
- `"doc_bert"`: `"BERT transformer NLP text classification cross-entropy"`

Query with: `"gradient boosting trees tabular data"`. Assert `results["ids"][0][0] == "doc_lgbm"`.

**Failure message must say:**
`"Semantic query for gradient boosting returned '{top_doc}' instead of 'doc_lgbm'. Embeddings are not semantic — ChromaDB may be using random or wrong embeddings."`

---

### TEST 1.3 — `test_dissimilar_query_does_not_contaminate_top_result`
**Bug it catches:** Embedding space that is too flat — all documents cluster together regardless of content.

**What to test:**
Using the same 3-document collection from 1.2, query with `"BERT tokenizer text classification"`. Assert `results["ids"][0][0] != "doc_lgbm"`. The NLP query must not surface a tabular ML document as its top hit.

---

### TEST 1.4 — `test_cosine_similarity_between_similar_docs_exceeds_threshold`
**Bug it catches:** Degraded embedding space where semantically related text is not closer than unrelated text.

**What to test:**
Compute embeddings for three strings directly via the embedding function:
- A: `"LightGBM gradient boosting tabular AUC"`
- B: `"XGBoost gradient boosted trees tabular classification"`
- C: `"BERT transformer NLP language model"`

Compute cosine similarity: `sim(A,B)` and `sim(A,C)`.

Assert `sim(A,B) > 0.80` — two gradient boosting descriptions must be highly similar.
Assert `sim(A,B) > sim(A,C) + 0.10` — the similar pair must be meaningfully more similar than the dissimilar pair.

**Failure message must say:**
`"sim(A,B)={:.3f} < 0.80 — embeddings are not producing a coherent semantic space"` or
`"sim(A,B)={:.3f} not >10pts above sim(A,C)={:.3f} — embedding space is too flat"`

---

### TEST 1.5 — `test_client_bypassing_build_chroma_client_raises_runtime_error`
**Bug it catches:** Any code path that calls `chromadb.Client()` directly instead of `build_chroma_client()` will produce a client with no `_professor_ef` attribute, silently using ChromaDB's default (potentially random) embedding.

**What to test:**
Create a raw `chromadb.Client()` (not via `build_chroma_client()`). Call `get_or_create_collection(raw_client, "test")`. Assert `RuntimeError` is raised. Assert the error message contains the string `"build_chroma_client"`.

---

### TEST 1.6 — `test_embedding_is_deterministic_across_two_calls`
**Bug it catches:** Non-deterministic embeddings. If embeddings are random or re-initialised differently across calls, memory retrieval will be inconsistent between pipeline runs.

**What to test:**
Embed the string `"tabular binary classification"` twice using the same embedding function instance. Assert the two vectors are identical element-by-element. Tolerance: zero — these must be bit-for-bit identical.

---

## BLOCK 2 — STATE FIELDS: BOUNDARY LOGIC
**File:** `TestStateFieldsBoundaryLogic`
**The bug this block catches:** Fields that are present but wrong. `task_type` that never changes from `"unknown"`. A `data_hash` that doesn't change when data changes. Strategy logic that flips at the wrong boundary.

---

### TEST 2.1 — `test_task_type_initial_value_is_unknown`
Call `initial_state()`. Assert `state["task_type"] == "unknown"`. It must not default to `"tabular"` or anything assumed — the router hasn't run yet.

---

### TEST 2.2 — `test_data_hash_changes_when_file_contents_change`
**Bug it catches:** A hash function that reads the filename rather than the contents, or caches the result incorrectly.

**What to test:**
Write two CSV files to `/tmp/` that differ by exactly one value in one cell. Call `hash_dataset()` on each. Assert the hashes are different.

**Failure message:** `"data_hash is the same for two files with different contents. SHA-256 is reading wrong content or caching incorrectly."`

---

### TEST 2.3 — `test_data_hash_is_stable_for_identical_content`
Call `hash_dataset()` on the same file twice. Assert the results are equal. Hashing must be deterministic.

---

### TEST 2.4 — `test_data_hash_is_16_hex_characters`
Assert `len(hash_dataset(any_file)) == 16` and that every character is in `"0123456789abcdef"`. Not 32. Not 64. Exactly 16.

---

### TEST 2.5 — `test_strategy_at_conservative_boundary` *(parametrize all boundary cases)*
**Bug it catches:** Off-by-one in the strategy boundary conditions. A bug at `percentile=0.10` vs `percentile=0.11` silently flips conservative/aggressive — which in a real competition means protecting a top-10% rank vs betting everything on a Hail Mary.

Parametrize `_determine_strategy()` across all boundary conditions. Every case is a separate assertion:

| `percentile` | `days_remaining` | expected strategy | reason |
|---|---|---|---|
| `0.05` | `2` | `conservative` | top 5%, almost done |
| `0.10` | `2` | `conservative` | exactly at boundary — still conservative |
| `0.11` | `2` | `aggressive` | just outside top 10%, time running out |
| `0.50` | `2` | `aggressive` | mid-table, time running out |
| `0.41` | `8` | `aggressive` | far from goal, time available |
| `0.99` | `30` | `aggressive` | bottom, lots of time |
| `0.30` | `10` | `balanced` | normal case |
| `0.05` | `10` | `balanced` | top 5% but plenty of time |
| `0.40` | `8` | `balanced` | exactly at 40% boundary — balanced (not aggressive) |
| — | — (missing) | `balanced` | incomplete context defaults to balanced |

**Failure message for each case:**
`"_determine_strategy(percentile={p}, days={d}) returned '{got}', expected '{want}'"`

---

### TEST 2.6 — `test_competition_context_has_all_required_keys`
Call `initial_state()`. Assert `state["competition_context"]` contains exactly these keys: `days_remaining, hours_remaining, submissions_used, submissions_remaining, current_public_rank, total_competitors, current_percentile, shakeup_risk, strategy, last_updated`. Missing any key is a contract violation.

---

### TEST 2.7 — `test_data_hash_written_to_state_after_data_engineer`
Run `run_data_engineer(state)`. Assert `state["data_hash"] != ""`. An empty string means the Data Engineer never called `hash_dataset()`.

---

### TEST 2.8 — `test_model_registry_entry_contains_data_hash`
**Bug it catches:** Ensemble Architect silently mixes models trained on different data versions.

Run the optimizer for at least 1 trial. Inspect the model_registry. Assert every entry has a `"data_hash"` key with a non-empty string value. If any registry entry lacks `data_hash`, the Ensemble Architect has no way to detect version conflicts.

---

## BLOCK 3 — VALIDATION ARCHITECT: STRATEGY CORRECTNESS
**File:** `TestValidationArchitectStrategyCorrectness`
**The bug this block catches:** CV strategy that ignores group columns (inflated CV), mismatch detection that triggers on false positives (halts good pipelines), metric contract with wrong direction (Optuna minimises when it should maximise).

---

### TEST 3.1 — `test_stratified_kfold_for_binary_target`
Run on clean tabular fixture with binary target. Assert `validation_strategy["cv_type"] == "StratifiedKFold"`.

---

### TEST 3.2 — `test_group_kfold_when_group_column_present`
**Bug it catches:** Agent ignores group columns. This causes rows from the same patient/user/store to appear in both train and validation — inflated CV that collapses on LB.

Inject `"customer_id": "Utf8"` into the schema. Run `run_validation_architect()`. Assert:
- `cv_type == "GroupKFold"`
- `group_col == "customer_id"`

**Failure message:** `"Group column 'customer_id' present but CV strategy is '{got}', not GroupKFold. This inflates CV by leaking rows from the same group."`

---

### TEST 3.3 — `test_timeseries_split_when_datetime_column_present`
**Bug it catches:** Agent ignores time structure. StratifiedKFold on time-ordered data leaks future data into past folds.

Inject `"transaction_date": "Date"` into schema. Assert `cv_type == "TimeSeriesSplit"`.

**Failure message:** `"Datetime column present but CV strategy is '{got}', not TimeSeriesSplit. This leaks future data into past folds."`

---

### TEST 3.4 — `test_group_kfold_takes_priority_over_datetime_column`
**Bug it catches:** Priority ordering bug — when both group and datetime exist, which wins?

Inject both `"patient_id": "Utf8"` and `"visit_date": "Date"`. Assert `cv_type == "GroupKFold"`. Group splits are more critical than time splits because group leakage is the more catastrophic failure.

---

### TEST 3.5 — `test_kfold_for_continuous_target`
Patch schema to make target column have `n_unique > 100` and dtype `Float64`. Assert `cv_type in ("KFold", "StratifiedKFold")` — stratification makes no sense on continuous targets.

---

### TEST 3.6 — `test_n_splits_is_always_5`
Run on 3 different schema configurations (binary, group, timeseries). Assert `n_splits == 5` in all cases. Not 3. Not 10. Exactly 5 unless overridden by config.

---

### TEST 3.7 — `test_mismatch_detected_stratified_plus_datetime`
**Bug it catches:** The mismatch detector failing to fire when it should.

Inject `"order_date": "Datetime"` into schema. Run. Assert `result["hitl_required"] is True`.

**Failure message:** `"Datetime column with StratifiedKFold should trigger mismatch detection. hitl_required was not set to True."`

---

### TEST 3.8 — `test_mismatch_reason_names_the_offending_column`
**Bug it catches:** A mismatch reason like `"CV/LB mismatch detected"` is useless for debugging.

When mismatch is triggered by `"signup_date"` column, assert `"signup_date" in result["hitl_reason"]`. Engineers cannot debug without knowing which column caused the halt.

---

### TEST 3.9 — `test_no_false_positive_mismatch_on_clean_tabular`
Run on clean tabular fixture with no time or group columns. Assert `result["hitl_required"] is False`. A false positive halts a valid pipeline.

---

### TEST 3.10 — `test_metric_contract_direction_correct_for_auc`
If `scorer_name == "auc"`, assert `metric_contract["direction"] == "maximize"`. AUC must be maximised. If Optuna minimises AUC, it searches for the worst model.

---

### TEST 3.11 — `test_metric_contract_direction_correct_for_rmse`
If `scorer_name == "rmse"`, assert `metric_contract["direction"] == "minimize"`. RMSE must be minimised.

---

### TEST 3.12 — `test_metric_contract_forbidden_metrics_is_non_empty_list`
Assert `isinstance(metric_contract["forbidden_metrics"], list)` and `len(...) > 0`. An empty forbidden list means there is nothing preventing the optimizer from using accuracy on an imbalanced dataset.

---

### TEST 3.13 — `test_validation_strategy_json_written_even_when_mismatch_halts`
**Bug it catches:** Agent that raises an exception on mismatch instead of writing the strategy file and returning cleanly.

Trigger a mismatch. Assert `validation_strategy.json` exists on disk. Engineers need this file to understand why the pipeline halted.

---

### TEST 3.14 — `test_metric_contract_not_written_when_mismatch_halts`
**Bug it catches:** Agent that writes the metric contract before detecting the mismatch.

Trigger a mismatch. Assert `result.get("metric_contract_path")` is either `None` or points to a file that does not exist. The optimizer must not receive a metric contract when the CV strategy is unsafe.

---

## BLOCK 4 — EDA AGENT: THRESHOLD ACCURACY AND DETECTION QUALITY
**File:** `TestEDAAgentThresholdAccuracy`
**The bug this block catches:** Outlier strategies at the wrong thresholds, leakage that isn't flagged, ID conflicts that pass silently, a `summary` field that is empty or generic.

---

### TEST 4.1 — `test_eda_report_has_all_required_keys`
Run `run_eda_agent()` on any fixture. Assert the report contains exactly these keys: `target_distribution, feature_correlations, outlier_profile, duplicate_analysis, temporal_profile, leakage_fingerprint, drop_candidates, summary`. Missing any key is a contract violation.

---

### TEST 4.2 — `test_outlier_strategy_keep_below_1pct`
**Bug it catches:** Wrong threshold — agent applies winsorize to a column with 0.5% outliers.

Construct a DataFrame where one column has exactly 0.5% outliers. Run EDA. Find that column in `outlier_profile`. Assert `strategy == "keep"`.

---

### TEST 4.3 — `test_outlier_strategy_winsorize_between_1_and_5pct`
Construct a column with 3% outliers. Assert `strategy == "winsorize"`. Not `"cap"`. Not `"keep"`. Exactly `"winsorize"`.

---

### TEST 4.4 — `test_outlier_strategy_cap_between_5_and_10pct`
Construct a column with 7% outliers. Assert `strategy == "cap"`.

---

### TEST 4.5 — `test_outlier_strategy_remove_above_10pct`
Construct a column with 15% outliers. Assert `strategy == "remove"`.

---

### TEST 4.6 — `test_leakage_flag_triggers_above_095_correlation`
**Bug it catches:** Leakage fingerprint that uses 0.90 instead of 0.95 as the FLAG threshold, causing legitimate features to be flagged.

Construct a DataFrame where one feature has an exact Pearson correlation of 0.96 with the target. Run EDA. Find that feature in `leakage_fingerprint`. Assert `verdict == "FLAG"`.

---

### TEST 4.7 — `test_leakage_watch_between_080_and_095`
Construct a feature with correlation 0.85. Assert `verdict == "WATCH"`.

---

### TEST 4.8 — `test_leakage_ok_below_080`
Construct a feature with correlation 0.70. Assert `verdict == "OK"`.

---

### TEST 4.9 — `test_flagged_leakage_feature_in_drop_candidates`
**Bug it catches:** Agent flags leakage but doesn't add it to `drop_candidates`. Feature Factory then uses it anyway.

Any feature with `verdict == "FLAG"` must appear in `drop_candidates`.

**Failure message:** `"Feature '{name}' is flagged as leakage but not in drop_candidates. Feature Factory will use it."`

---

### TEST 4.10 — `test_id_conflict_detection`
**Bug it catches:** Duplicate IDs with different target values — the most dangerous form of label noise. If undetected, the model trains on contradictory examples.

Construct a DataFrame where `id` column has value `"A"` appearing twice with different targets (`0` and `1`). Run EDA. Assert:
- `duplicate_analysis["id_conflict_count"] >= 1`
- `"id"` in `duplicate_analysis["id_conflict_columns"]`

**Failure message:** `"ID conflict not detected: same ID with different target values is the most dangerous label noise pattern."`

---

### TEST 4.11 — `test_exact_duplicate_count_is_correct`
Construct a DataFrame with exactly 3 exact duplicate rows. Run EDA. Assert `duplicate_analysis["exact_count"] == 3`.

---

### TEST 4.12 — `test_temporal_profile_detects_date_column`
Inject a `"signup_date"` column with dtype `Date`. Run EDA. Assert `temporal_profile["has_dates"] is True` and `"signup_date" in temporal_profile["date_columns"]`.

---

### TEST 4.13 — `test_summary_is_non_empty_and_specific`
**Bug it catches:** Agent writes `summary = ""` or `summary = "EDA complete."` — useless for downstream agents.

Assert `len(report["summary"]) > 100`. Assert the summary mentions at least one of: the target distribution, outlier findings, or leakage warnings. The summary is injected into the Feature Factory and Critic system prompts — a generic summary produces generic agents.

---

### TEST 4.14 — `test_zero_variance_feature_in_drop_candidates`
Construct a column where every value is identical. Run EDA. Assert that column is in `drop_candidates`. Zero-variance features carry no signal and can cause NaN SHAP values in tree models.

---

### TEST 4.15 — `test_target_skew_computed_for_continuous_target`
Construct a DataFrame with a highly right-skewed continuous target (e.g. exponential distribution). Run EDA. Assert `target_distribution["skew"] > 1.0` and `target_distribution["recommended_transform"] in ("log", "sqrt", "boxcox")`. The agent must detect the need to transform, not just report the skew.

---

### TEST 4.16 — `test_eda_report_written_to_disk`
After running EDA, assert `os.path.exists(state["eda_report_path"])`. Assert the file is valid JSON. Assert it can be parsed and contains all required keys. The file on disk is what downstream agents read — the state dict alone is not enough.

---

## BLOCK 5 — COMPETITION INTEL: BRIEF QUALITY
**File:** `TestCompetitionIntelBriefQuality`
**The bug this block catches:** A brief that runs but produces empty lists, a `dominant_approach` of `""`, or a `shakeup_risk` of `None`. These pass existence checks but provide no signal.

---

### TEST 5.1 — `test_intel_brief_has_all_required_keys`
Run `run_competition_intel()` on Spaceship Titanic. Assert `intel_brief` contains all of: `critical_findings, proven_features, known_leaks, external_datasets, dominant_approach, cv_strategy_hint, forbidden_techniques, shakeup_risk, source_post_count, scraped_at`.

---

### TEST 5.2 — `test_shakeup_risk_is_valid_value`
Assert `intel_brief["shakeup_risk"] in ("low", "medium", "high")`. Not `None`, not `""`, not `"unknown"` (unless the API returned nothing and `source_post_count == 0`).

---

### TEST 5.3 — `test_dominant_approach_is_non_empty_when_posts_scraped`
If `source_post_count > 0`, assert `len(intel_brief["dominant_approach"]) > 10`. A dominant approach of `""` or `"unknown"` when posts were successfully scraped means the LLM synthesis failed silently.

---

### TEST 5.4 — `test_graceful_degradation_on_private_competition`
**Bug it catches:** Agent crashes when the competition has no public forum.

Pass a competition name that doesn't exist on Kaggle (e.g. `"nonexistent-competition-xyz-99"`). Assert:
- No exception raised
- `intel_brief["source_post_count"] == 0`
- All list fields are empty lists (not `None`)
- `intel_brief["dominant_approach"]` is a string (possibly `"unknown"`)

---

### TEST 5.5 — `test_competition_brief_written_to_disk`
After running `run_competition_intel()`, assert `os.path.exists(state["competition_brief_path"])`. The file must be valid JSON readable by `json.load()`.

---

### TEST 5.6 — `test_critical_findings_are_strings_not_dicts`
Assert every item in `intel_brief["critical_findings"]` is a `str`. LLM synthesis sometimes returns structured objects instead of strings — this breaks downstream string injection into system prompts.

---

### TEST 5.7 — `test_scraped_at_is_valid_iso_timestamp`
Assert `intel_brief["scraped_at"]` can be parsed as an ISO 8601 datetime string. `None` or an empty string means the scraper never completed.

---

## BLOCK 6 — INTEGRATION: CROSS-AGENT CONTRACT ENFORCEMENT
**File:** `TestCrossAgentContractEnforcement`
**The bug this block catches:** Agents that pass unit tests in isolation but break when chained. These are the bugs you never find until the full pipeline runs — or until a test catches them.

---

### TEST 6.1 — `test_eda_report_read_by_validation_architect`
**Bug it catches:** Validation Architect that ignores the EDA report even when `eda_report_path` is in state.

Run EDA agent, then run Validation Architect on the resulting state. Assert `result["validation_strategy"]` reflects EDA findings — specifically, if EDA flagged a datetime column via `temporal_profile["date_columns"]`, then `cv_type` must be `"TimeSeriesSplit"`.

---

### TEST 6.2 — `test_validation_halt_prevents_ml_optimizer_from_running`
**Bug it catches:** LangGraph conditional edge that ignores `hitl_required=True`.

Inject a mismatch. Run `run_validation_architect()`. Assert `hitl_required is True`. Then assert that calling `run_ml_optimizer(state)` raises an error or returns without producing a model — the optimizer must never run on an unsafe CV strategy. If no guard exists, add one to `run_ml_optimizer()`: check `state.get("hitl_required")` at entry.

---

### TEST 6.3 — `test_intel_brief_injected_into_validation_architect`
Run the full chain: `competition_intel → data_engineer → eda_agent → validation_architect`. Assert `result["validation_strategy"]` contains a `"cv_strategy_hint"` key (or that the competition brief's hint influenced the strategy). The brief must reach the Validation Architect.

---

### TEST 6.4 — `test_drop_candidates_from_eda_respected_by_feature_factory`
**Bug it catches:** Feature Factory that reads the EDA report but ignores `drop_candidates`.

Run EDA on a dataset with a flagged leakage column. Then run Feature Factory. Assert the flagged column does not appear in the generated feature list or in the feature code.

---

### TEST 6.5 — `test_data_hash_in_state_after_full_chain`
Run `competition_intel → data_engineer → eda_agent`. Assert `state["data_hash"] != ""` at the end. The hash must survive through all three agents unchanged.

---

### TEST 6.6 — `test_phase1_regression_still_green`
This is non-negotiable. After all Day 8 code is in place:
```bash
pytest tests/regression/test_phase1_regression.py -v
```
All tests must pass. Any failure here means Day 8 broke Phase 1. Do not commit until this is green.

---

## END-TO-END QUALITY GATE

After all 6 blocks pass, run this smoke test. It must complete without error and produce output that demonstrates each agent did real work:

```bash
python -c "
from core.state import initial_state
from agents.competition_intel import run_competition_intel
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect

state = initial_state('spaceship-titanic', 'data/spaceship_titanic/train.csv')
state = run_competition_intel(state)
state = run_data_engineer(state)
state = run_eda_agent(state)
state = run_validation_architect(state)

# Quality assertions — not just existence
assert state['data_hash'] != '', 'data_hash is empty'
assert state['task_type'] != 'unknown', 'task_type was never set by router'
assert state['validation_strategy']['cv_type'] in ('StratifiedKFold','GroupKFold','TimeSeriesSplit','KFold')
assert state['hitl_required'] is False, f'Clean data triggered HITL: {state.get(\"hitl_reason\")}'
assert len(state['eda_report']['drop_candidates']) >= 0  # may be empty — fine
assert len(state['eda_report']['summary']) > 100, 'EDA summary is too short'
assert state['eda_report']['duplicate_analysis']['exact_count'] >= 0
assert state['competition_brief']['source_post_count'] >= 0

print('CV strategy:', state['validation_strategy']['cv_type'])
print('Metric:', state['validation_strategy']['scorer_name'])
print('Direction:', state['metric_contract']['direction'] if 'metric_contract' in state else '(from contract file)')
print('EDA summary:', state['eda_report']['summary'][:120], '...')
print('Drop candidates:', state['eda_report']['drop_candidates'])
print('Leakage flags:', [f['feature'] for f in state['eda_report']['leakage_fingerprint'] if f['verdict'] == 'FLAG'])
print('Intel source posts:', state['competition_brief']['source_post_count'])
print('Dominant approach:', state['competition_brief']['dominant_approach'])
print('Competition strategy:', state['competition_context']['strategy'])
print()
print('[PASS] Day 8 quality gate — all assertions passed')
"
```

---

## TOTAL TEST COUNT

| Block | Class | Tests |
|---|---|---|
| 1 — ChromaDB Semantics | `TestChromaDBEmbeddingSemanticsNotJustStartup` | 6 |
| 2 — State Boundary Logic | `TestStateFieldsBoundaryLogic` | 8 |
| 3 — Validation Architect | `TestValidationArchitectStrategyCorrectness` | 14 |
| 4 — EDA Agent Quality | `TestEDAAgentThresholdAccuracy` | 16 |
| 5 — Intel Brief Quality | `TestCompetitionIntelBriefQuality` | 7 |
| 6 — Cross-Agent Integration | `TestCrossAgentContractEnforcement` | 6 |
| **Total** | | **57** |

57 tests. Every one finds a specific named bug. If all 57 are green, Professor is podium-ready for Day 8.

---

## DEFINITION OF DONE FOR TESTS

- [ ] All 57 tests written in `tests/test_day8_quality.py`
- [ ] All 57 tests pass
- [ ] `pytest tests/regression/test_phase1_regression.py` — still green
- [ ] `pytest tests/contracts/` — still green
- [ ] Test run logged to `tests/logs/day8_quality.log`
- [ ] `git commit -m "Day 8: 57 adversarial quality tests — all green"`