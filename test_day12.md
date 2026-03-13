# Professor Agent — Day 12 Test Specification
**For: Claude Code**
**Status: IMMUTABLE after Day 12**
**Philosophy: These tests prove Professor survives overnight unsupervised — and fails gracefully when it doesn't.**

All tests in `tests/test_day12_quality.py`.

```bash
pytest tests/test_day12_quality.py -v --tb=short 2>&1 | tee tests/logs/day12_quality.log
```

---

## BLOCK 1 — MEMORY MANAGEMENT: ml_optimizer.py (14 tests)
**Class:** `TestMLOptimizerMemoryManagement`

The bugs in this block: GC called but models not deleted first (still referenced). Memory check at end of trial, not mid-fold (OOM happens in fold 4 with no handling). n_jobs defaulting to -1 (multiplies memory per CPU core). Peak memory not propagated to state.

---

### TEST 1.1 — `test_models_deleted_in_finally_block_not_just_on_success`
**Bug:** `del models; gc.collect()` inside the `try` block — never runs if trial raises.

Monkeypatch the last fold to raise `RuntimeError`. Assert `models` list was deleted despite the exception. Verify by checking that the objects are no longer in `gc.get_objects()` after the call. If `del` was inside `try`, the reference persists in the exception frame.

---

### TEST 1.2 — `test_gc_called_after_del_not_before`
**Bug:** `gc.collect()` called before `del models` — models still referenced, GC can't free them.

Verify call order using a `MagicMock` sequence. Assert `del` precedes `gc.collect()` in the `finally` block.

---

### TEST 1.3 — `test_memory_check_per_fold_not_per_trial`
**Bug:** Memory check only at end of trial. On a 5-fold CV with 1GB-per-fold models, fold 5 OOMs with no handling.

Monkeypatch `psutil.Process().memory_info().rss` to return 7GB on fold 3. Assert `optuna.TrialPruned` is raised at fold 3 — not after fold 5 and not after the trial completes.

---

### TEST 1.4 — `test_trial_pruned_cleanly_on_oom_not_killed`
**Bug:** Memory exceeded → Python MemoryError kills the process (OOM kill) instead of `optuna.TrialPruned`.

Monkeypatch RSS to exceed `max_memory_gb`. Assert that `_objective()` raises `optuna.TrialPruned`, not `MemoryError`. The trial is marked PRUNED in the study, not as an exception.

---

### TEST 1.5 — `test_trial_user_attr_oom_risk_set_when_pruned`
After a pruned trial: `study.trials[-1].user_attrs["oom_risk"] == True`. Also verify `oom_at_fold` and `oom_rss_gb` attributes are set. These attributes are how the post-run memory report is built.

---

### TEST 1.6 — `test_no_oom_risk_attr_on_successful_trial`
Successful trial (RSS stays below threshold throughout). Assert `study.trials[-1].user_attrs.get("oom_risk")` is either `False` or not present. Successful trials should not pollute the OOM report.

---

### TEST 1.7 — `test_gc_after_trial_flag_set_in_study_optimize`
**Bug:** `gc_after_trial=True` not passed to `study.optimize()`. Optuna's own inter-trial GC is not called.

Inspect the `study.optimize()` call args (use monkeypatch on `optuna.Study.optimize`). Assert `gc_after_trial=True` was passed. Not optional — this is belt-and-braces GC on top of the manual `del`.

---

### TEST 1.8 — `test_n_jobs_defaults_to_1_not_minus_1`
**Bug:** `n_jobs=-1` causes each parallel worker to hold its own model copy. On 8GB with 8 cores: 8 × 200MB = instant OOM.

Assert that `run_optimization()` called with no `n_jobs` argument uses `n_jobs=1`. Assert the default is not read from a system value (not `os.cpu_count()`).

---

### TEST 1.9 — `test_max_memory_gb_read_from_env_var`
Set `PROFESSOR_MAX_MEMORY_GB=5.0` in environment. Create optimizer with no explicit `max_memory_gb`. Assert the threshold used is 5.0, not 6.0.

---

### TEST 1.10 — `test_max_memory_gb_env_var_invalid_falls_back_to_default`
Set `PROFESSOR_MAX_MEMORY_GB=not_a_number`. Assert optimizer uses default 6.0. Must not raise on startup.

---

### TEST 1.11 — `test_peak_memory_written_to_state`
After `run_optimization()` completes: assert `state["memory_peak_gb"]` is a float > 0. If all trials had no OOM attrs (normal run), peak is taken from current `psutil.Process().memory_info().rss`. Must not be 0.0.

---

### TEST 1.12 — `test_oom_risk_flag_true_when_any_trial_pruned`
If at least one trial was pruned for memory: assert `state["memory_oom_risk"] == True`.

---

### TEST 1.13 — `test_oom_risk_flag_false_when_no_trials_pruned`
No trials pruned. Assert `state["memory_oom_risk"] == False`.

---

### TEST 1.14 — `test_pruned_trial_count_correct_in_state`
3 trials pruned out of 10. Assert `state["optuna_pruned_trials"] == 3`.

---

## BLOCK 2 — LANGSMITH COST CONTROL: professor.py (10 tests)
**Class:** `TestLangSmithTracingControl`

The bugs: tracing disabled but not restored after Optuna finishes (try/finally missing). Sampling rate set in .env but not applied at startup (not read as setdefault). Tracing "disabled" but env var not propagated to subprocesses (child processes inherit parent env).

---

### TEST 2.1 — `test_tracing_disabled_inside_optuna_loop`
During `study.optimize()`: assert `os.environ["LANGCHAIN_TRACING_V2"] == "false"`. Mock the optimize call to capture the env state mid-execution.

---

### TEST 2.2 — `test_tracing_restored_after_optuna_loop_completes`
Set `LANGCHAIN_TRACING_V2=true` before `run_optimization()`. After it returns: assert `os.environ["LANGCHAIN_TRACING_V2"] == "true"`. The original value, not "false", not missing.

---

### TEST 2.3 — `test_tracing_restored_even_when_optimize_raises`
Monkeypatch `study.optimize()` to raise `RuntimeError`. Assert `LANGCHAIN_TRACING_V2` is restored to its original value in the exception handler. Requires `try/finally`, not `try/except`.

---

### TEST 2.4 — `test_tracing_restored_when_original_was_false`
`LANGCHAIN_TRACING_V2=false` before the call. After `run_optimization()`: assert still `"false"`. The restore must use the original value, not hardcode `"true"`.

---

### TEST 2.5 — `test_tracing_restored_when_env_var_was_absent`
Delete `LANGCHAIN_TRACING_V2` from env before the call. After `run_optimization()`: assert the key is absent from `os.environ`. The restore must handle the case where the key didn't exist (don't set it to `None` or `"None"`).

---

### TEST 2.6 — `test_sampling_rate_set_from_env_var`
Set `LANGCHAIN_TRACING_SAMPLING_RATE=0.05` in environment. Start professor. Assert `os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] == "0.05"`. The startup code must read and apply the env var, not overwrite it with the hardcoded default.

---

### TEST 2.7 — `test_sampling_rate_defaults_to_0_10_when_not_set`
Unset `LANGCHAIN_TRACING_SAMPLING_RATE`. Start professor. Assert `os.environ.get("LANGCHAIN_TRACING_SAMPLING_RATE") == "0.10"`.

---

### TEST 2.8 — `test_env_example_contains_tracing_sampling_rate`
Read `.env.example` from the file system. Assert:
1. `LANGCHAIN_TRACING_SAMPLING_RATE` key is present
2. Default value is `0.10` or `0.1`
3. A comment line above it mentions cost risk (contains "cost" or "WARNING" or "$")

---

### TEST 2.9 — `test_env_example_contains_max_memory_gb`
Read `.env.example`. Assert:
1. `PROFESSOR_MAX_MEMORY_GB` key is present
2. Default value is `6.0`
3. A comment mentions 8GB RAM or headroom

---

### TEST 2.10 — `test_cost_estimation_logged_after_run`
Run a complete pipeline (or mock it). Assert that `logger.info` was called with a message containing "$" and "cost" after the pipeline completes. The cost estimate must be logged at INFO level, not DEBUG (DEBUG gets filtered in production).

---

## BLOCK 3 — HITL PROMPT GENERATION (14 tests)
**Class:** `TestHITLPromptGeneration`

The bugs: prompt generated but not written to disk (JSON file missing). Interventions list not exactly 3 (edge case in unknown error class). Truncation missing from error message (500-char traceback becomes 50000-char JSON). MANUAL interventions triggering AUTO state changes. Resume restoring state but not resetting failure count.

---

### TEST 3.1 — `test_prompt_has_all_required_keys`
Call `generate_hitl_prompt()`. Assert all 9 keys present:
`session_id, failed_agent, failure_count, what_was_attempted, why_it_failed, error_class, interventions, resume_command, checkpoint_key, generated_at`.

---

### TEST 3.2 — `test_prompt_has_exactly_3_interventions`
For every error class (data_quality, model_failure, memory, api_timeout, unknown): assert `len(prompt["interventions"]) == 3`. The most common bug is the "unknown" class returning 2 or 4.

---

### TEST 3.3 — `test_each_intervention_has_required_keys`
For each intervention in the prompt: assert all 5 keys present: `id, label, action_type, risk, description`. `code_hint` is optional but must be present as a key (value may be None).

---

### TEST 3.4 — `test_intervention_ids_are_1_2_3`
Assert `[i["id"] for i in prompt["interventions"]] == [1, 2, 3]`. Not 0-indexed. Not skipping 2.

---

### TEST 3.5 — `test_error_classification_data_quality`
Raise `KeyError("target_column")` from `data_engineer`. Assert `error_class == "data_quality"`.

---

### TEST 3.6 — `test_error_classification_memory`
Raise `MemoryError()`. Assert `error_class == "memory"`.

---

### TEST 3.7 — `test_error_classification_api_timeout`
Raise `TimeoutError("groq request timed out")`. Assert `error_class == "api_timeout"`.

---

### TEST 3.8 — `test_error_classification_unknown_for_unexpected_type`
Raise `PermissionError("cannot write to /tmp")`. Assert `error_class == "unknown"`. No exception raised by the classifier.

---

### TEST 3.9 — `test_error_message_truncated_to_500_chars`
**Bug:** Long traceback (10000 chars) written verbatim to JSON → 10KB per HITL event.

Pass an exception whose `str()` is 2000 characters. Assert `prompt["why_it_failed"]` has length ≤ 500.

---

### TEST 3.10 — `test_prompt_written_to_disk`
After `generate_hitl_prompt()`: assert `outputs/{session_id}/hitl_prompt.json` exists. Load it. Assert JSON is valid and `session_id` key matches.

---

### TEST 3.11 — `test_prompt_write_failure_does_not_propagate`
**Bug:** Disk full → `_write_hitl_prompt()` raises `OSError` → propagates out of `generate_hitl_prompt()` → original HITL handling fails.

Monkeypatch `open()` to raise `OSError`. Assert `generate_hitl_prompt()` returns the prompt dict normally. The file write failure must be caught and logged, not raised.

---

### TEST 3.12 — `test_resume_from_checkpoint_resets_failure_count`
After `resume_from_checkpoint(session_id, intervention_id=1)`: assert `result["current_node_failure_count"] == 0`. Not 3. The re-entry state must be clean.

---

### TEST 3.13 — `test_resume_from_checkpoint_clears_hitl_required`
Assert `result["hitl_required"] == False` after resume. If still True, the graph halts again immediately on re-entry.

---

### TEST 3.14 — `test_resume_from_corrupt_checkpoint_returns_error_state`
Write malformed JSON to the Redis checkpoint key. Call `resume_from_checkpoint()`. Assert it returns an error state dict (with `"error"` key) rather than raising `json.JSONDecodeError`. The pipeline should fail gracefully on bad checkpoints, not crash the CLI.

---

## BLOCK 4 — FULL HITL INTEGRATION: 3x FAILURE SIMULATION (14 tests)
**Class:** `TestHITLFullIntegration`

The hardest tests. These prove the complete path: agent fails → inner retry (Day 9) → circuit breaker escalates → HITL prompt generated → state saved → pipeline halted → resume works → pipeline continues.

---

### TEST 4.1 — `test_3x_failure_triggers_hitl_not_macro`
**The primary 3x failure simulation.**

Inject a permanent failure into `data_engineer` (always raises `KeyError`). Run the pipeline. After 3 failures (failure_count reaches 3 in the circuit breaker): assert `state["hitl_required"] == True`. Assert the escalation level was HITL, not MACRO (which would be at failure_count == 2).

---

### TEST 4.2 — `test_state_saved_to_redis_on_hitl`
After the 3x failure triggers HITL: assert Redis contains a key matching `professor:hitl:{session_id}`. Load the key. Assert it's valid JSON containing `"state"` and `"agent_name"` fields.

---

### TEST 4.3 — `test_redis_checkpoint_contains_full_state`
The Redis checkpoint must contain enough state to resume. Assert at least these fields are present in `checkpoint["state"]`:
`session_id, competition_name, dag_version, current_node_failure_count, competition_fingerprint`.

---

### TEST 4.4 — `test_pipeline_halted_after_hitl_no_further_agents_run`
After HITL triggers: assert no agents run after the HITL flag is set. Specifically, `feature_factory`, `ml_optimizer`, and `ensemble_architect` must not appear in lineage after the HITL event. The pipeline is genuinely paused, not just flagged.

---

### TEST 4.5 — `test_hitl_prompt_generated_with_3_interventions`
After 3x failure: read `outputs/{session_id}/hitl_prompt.json`. Assert `len(prompt["interventions"]) == 3`. Assert `prompt["failed_agent"] == "data_engineer"`.

---

### TEST 4.6 — `test_hitl_prompt_error_class_matches_injected_error`
Inject `KeyError`. Assert `prompt["error_class"] == "data_quality"`. The error classification must read the actual exception type, not guess from the agent name.

---

### TEST 4.7 — `test_resume_with_auto_intervention_applies_state_change`
Resume with intervention 1 (data_quality: "Skip validation, proceed with raw features."). Assert `result_state["skip_data_validation"] == True`. The AUTO intervention must modify state, not just set `hitl_intervention_id`.

---

### TEST 4.8 — `test_resume_with_manual_intervention_does_not_change_state`
Resume with intervention 3 (any class: "MANUAL" action_type). Assert that no domain-specific flags were changed in state (skip_data_validation, lgbm_override, etc. remain at defaults). MANUAL means the engineer fixed it externally — state is unchanged.

---

### TEST 4.9 — `test_resume_with_invalid_intervention_id_returns_error`
`resume_from_checkpoint(session_id, intervention_id=5)`. Assert error state returned (not exception raised). Assert error message mentions valid range (1–3).

---

### TEST 4.10 — `test_resume_from_nonexistent_session_returns_error`
`resume_from_checkpoint("nonexistent-session-id", 1)`. Assert error state returned. Assert error mentions the missing key. Must not raise `ConnectionError` or `NoneType` error.

---

### TEST 4.11 — `test_pipeline_continues_after_successful_resume`
Full integration: inject 3 failures → HITL → call `resume_from_checkpoint(session_id, 1)` → inject the resumed state back into the pipeline → assert pipeline continues past `data_engineer` → assert at least one downstream agent runs (e.g. `eda_agent`).

---

### TEST 4.12 — `test_lineage_contains_hitl_and_resume_events`
After the full failure → resume cycle: read `lineage.jsonl`. Assert both:
- Entry with `action="hitl_escalation"` and `agent="data_engineer"`
- Entry with `action="hitl_resumed"` and `intervention_label` non-empty

These are the audit entries that prove what happened during the failure and what was done to recover.

---

### TEST 4.13 — `test_failure_count_resets_to_0_after_resume_not_3`
After resume, `data_engineer` runs again and succeeds. Assert `state["current_node_failure_count"] == 0` at this point. If it's still 3, the next failure immediately triggers HITL again (no recovery window).

---

### TEST 4.14 — `test_hitl_does_not_trigger_on_first_two_failures`
`failure_count=1` → MICRO, not HITL. `failure_count=2` → MACRO, not HITL. Assert `state["hitl_required"] == False` and `state["hitl_prompt"] == {}` after the first and second failures. HITL fires on 3rd failure only.

---

## THE 6 BUGS THAT WILL DEFINITELY BE PRESENT ON FIRST IMPLEMENTATION

**Bug 1 — Test 1.1:** `del models` inside `try` block. If the last fold raises, `models` is never deleted. The `finally` block is the only safe location for cleanup code in exception paths. Every LLM generates the `try/del/gc` pattern — not `try/finally/del/gc`.

**Bug 2 — Test 2.3:** `_disable_langsmith_tracing()` uses `try/except` not `try/finally`. When Optuna raises (e.g. study timeout), the `except` only handles the exception type it catches. Anything else propagates with tracing disabled permanently. `try/finally` is unconditional.

**Bug 3 — Test 2.5:** Restoring env var after it was absent. `original = os.environ.get("LANGCHAIN_TRACING_V2", "false")`. This stores `"false"` as the original even when the key didn't exist. After restore: `os.environ["LANGCHAIN_TRACING_V2"] = "false"` — the key is now set when it wasn't before. Correct: `original = os.environ.get(...)` then `if original is None: del os.environ[...] else: os.environ[...] = original`.

**Bug 4 — Test 3.9:** No truncation on `why_it_failed`. `str(error)` on a deep LightGBM stack trace can be 50,000 characters. The JSON file becomes 50KB per HITL event. Redis checkpoint bloats. Fix: `str(error)[:500]`.

**Bug 5 — Test 4.4:** Pipeline not actually halted after HITL. The `hitl_required=True` flag is set in state, but the LangGraph conditional edge that checks it is either missing or reads the wrong key. Nodes continue executing. The test catches this by checking lineage for agent executions after the HITL timestamp.

**Bug 6 — Test 1.3:** Memory check at end of trial, not per fold. The most catastrophic failure mode: fold 5 of a 5-fold CV OOMs mid-training with no `TrialPruned` raised. The kernel kills the process. No traceback. The check must happen after each fold while `models` list is still accessible.

---

## CONTRACT TEST EXTENSION

The existing `tests/contracts/test_circuit_breaker_contract.py` (Day 9, immutable) covers escalation logic. Day 12 adds two new contracts that must also be immutable:

**`test_hitl_prompt_contract.py`** (new, immutable after Day 12):
```python
# CONTRACT: generate_hitl_prompt()
#   Always returns dict with all 9 required keys
#   Always returns exactly 3 interventions
#   Never raises for any error type
#   Always truncates why_it_failed to ≤ 500 chars
#   Always writes hitl_prompt.json to disk (silently fails if disk unavailable)
```

**`test_resume_checkpoint_contract.py`** (new, immutable after Day 12):
```python
# CONTRACT: resume_from_checkpoint()
#   Returns error state (not exception) for missing session
#   Returns error state (not exception) for corrupt JSON
#   Returns error state (not exception) for invalid intervention_id
#   Sets current_node_failure_count=0 on success
#   Sets hitl_required=False on success
#   AUTO interventions modify state; MANUAL interventions do not
```

---

## TOTAL: 52 TESTS + 2 NEW CONTRACT FILES

| Block | Tests |
|---|---|
| 1 — Memory Management | 14 |
| 2 — LangSmith Cost Control | 10 |
| 3 — HITL Prompt Generation | 14 |
| 4 — Full HITL Integration | 14 |
| **Total quality tests** | **52** |
| New HITL prompt contract | ~6 |
| New resume checkpoint contract | ~6 |
| **Grand total** | **~64** |

---

## DEFINITION OF DONE

- [ ] 52 tests in `tests/test_day12_quality.py` — all pass
- [ ] `tests/contracts/test_hitl_prompt_contract.py` — new, immutable, all pass
- [ ] `tests/contracts/test_resume_checkpoint_contract.py` — new, immutable, all pass
- [ ] `pytest tests/contracts/` — green (including Day 9 circuit breaker contracts)
- [ ] `pytest tests/regression/` — green
- [ ] `.env.example` has both new vars with cost/memory risk comments
- [ ] Manual smoke test: inject 3 failures, read `hitl_prompt.json`, confirm 3 interventions displayed
- [ ] `git commit -m "Day 12: HITL human layer, OOM guardrails, tracing cost control — 52 tests green"`