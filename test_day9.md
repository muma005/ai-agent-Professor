# Professor Agent — Day 9 Test Specification
**For: Claude Code**
**Status: IMMUTABLE after Day 9 — never edit these tests once written**
**Philosophy: Day 9 is the resilience layer. Tests must prove the system survives adversity, not just runs in a clean environment.**

All tests live in `tests/test_day9_quality.py`.

```bash
pytest tests/test_day9_quality.py -v --tb=short 2>&1 | tee tests/logs/day9_quality.log
```

---

## BLOCK 1 — CIRCUIT BREAKER: ESCALATION PRECISION
**Class:** `TestCircuitBreakerEscalationPrecision`
**The bug this block catches:** An escalation level that fires at the wrong threshold, a HITL that doesn't actually halt the pipeline, a TRIAGE that ignores budget but respects time, a MACRO that increments the wrong counter.

These bugs are invisible in happy-path tests. They only surface when Professor is mid-competition and the Groq API goes down at 2am.

---

### TEST 1.1 — `test_failure_count_1_is_micro_not_macro`
**Bug:** Off-by-one — MACRO fires on the first failure, wasting a DAG rewrite.

Assert `get_escalation_level({..., "current_node_failure_count": 1}) == EscalationLevel.MICRO`.

---

### TEST 1.2 — `test_failure_count_2_is_macro_not_hitl`
Assert `get_escalation_level({..., "current_node_failure_count": 2}) == EscalationLevel.MACRO`.

---

### TEST 1.3 — `test_failure_count_3_is_hitl_not_macro`
Assert `get_escalation_level({..., "current_node_failure_count": 3}) == EscalationLevel.HITL`.

---

### TEST 1.4 — `test_triage_fires_at_5pct_budget_not_10pct`
**Bug:** Triage threshold set at 10% instead of 5% — wastes 5% of budget on doomed attempts.

Set `budget_remaining_usd = budget_limit_usd * 0.049` (just under 5%). Assert TRIAGE.
Set `budget_remaining_usd = budget_limit_usd * 0.051` (just above 5%). Assert NOT TRIAGE.

---

### TEST 1.5 — `test_triage_overrides_failure_count_1`
**Bug:** Triage not checked before failure count — circuit breaker starts a MICRO retry when budget is zero.

Set `current_node_failure_count=1` AND `budget_remaining_usd=0.001`. Assert TRIAGE, not MICRO.

---

### TEST 1.6 — `test_triage_fires_at_2_hours_remaining_not_3`
Set `competition_context["hours_remaining"] = 2`. Assert TRIAGE.
Set `competition_context["hours_remaining"] = 3`. Assert NOT TRIAGE.

---

### TEST 1.7 — `test_micro_error_context_contains_full_traceback`
**Bug:** Error context stores only the exception message, not the traceback. LLM cannot self-correct from a one-line error.

Call `handle_escalation(level=MICRO, traceback_str="Traceback (most recent call last):\n  File...")`. Assert `result["error_context"][-1]["traceback"]` starts with `"Traceback"`.

---

### TEST 1.8 — `test_micro_failure_count_increments_by_exactly_1`
Call MICRO on state with `current_node_failure_count=1`. Assert result has `current_node_failure_count=2`. Not 0. Not 3. Exactly 2.

---

### TEST 1.9 — `test_macro_dag_version_increments_from_current_value`
**Bug:** MACRO always resets `dag_version` to 2 instead of incrementing.

Set `dag_version=5`. Trigger MACRO. Assert `result["dag_version"] == 6`.

---

### TEST 1.10 — `test_macro_does_not_halt_pipeline`
**Bug:** MACRO sets `pipeline_halted=True` — this kills the pipeline instead of replanning.

Trigger MACRO. Assert `result.get("pipeline_halted") is not True`.

---

### TEST 1.11 — `test_hitl_reason_contains_session_id_and_resume_command`
**Bug:** HITL reason is vague — engineer doesn't know which session to resume.

Assert `hitl_reason` contains both `state["session_id"]` and the string `"resume"`.

---

### TEST 1.12 — `test_hitl_does_not_raise_when_redis_is_down`
**Bug:** HITL tries to checkpoint to Redis, Redis is unavailable, `_checkpoint_state_to_redis` raises, circuit breaker crashes.

Before calling HITL: monkeypatch Redis to always raise `ConnectionRefusedError`. Trigger HITL. Assert no exception. Assert `result["hitl_required"] is True`. The circuit breaker must never raise — it is the last line of defence.

---

### TEST 1.13 — `test_reset_failure_count_clears_error_context`
After 2 MICRO escalations, `error_context` has 2 entries. Call `reset_failure_count()`. Assert `error_context == []` and `current_node_failure_count == 0`.

---

### TEST 1.14 — `test_handle_escalation_never_raises_for_any_level`
**Bug:** Any one escalation level raises an exception — the entire resilience layer becomes a crash source.

Iterate over all 4 `EscalationLevel` values. Call `handle_escalation()` for each. Assert no exceptions raised. `handle_escalation` must be unconditionally safe.

---

## BLOCK 2 — SUBPROCESS SANDBOX: REAL ML CODE EXECUTION
**Class:** `TestSubprocessSandboxRealExecution`
**The bug this block catches:** Subprocess that works for `print("hello")` but fails on numpy, polars, or lightgbm. A timeout that fires at 605 seconds instead of 600. A returncode that is 0 even when the script crashed.

---

### TEST 2.1 — `test_numpy_executes_without_error`
**Bug:** RestrictedPython blocks numpy C-extensions.

```python
run_in_sandbox("import numpy as np; print(np.array([1,2,3]).mean())")
```
Assert `result["success"] is True` and `"2.0"` in `result["stdout"]`.

---

### TEST 2.2 — `test_polars_executes_without_error`
```python
run_in_sandbox("import polars as pl; df = pl.DataFrame({'x': [1,2,3]}); print(df['x'].mean())")
```
Assert `success is True` and `"2.0"` in stdout.

---

### TEST 2.3 — `test_lightgbm_trains_without_error`
**Bug:** LightGBM requires C-extensions that RestrictedPython blocks.

```python
run_in_sandbox("""
import lightgbm as lgb, numpy as np
X = np.random.rand(50, 3); y = np.random.randint(0, 2, 50)
ds = lgb.Dataset(X, label=y)
m = lgb.train({'objective': 'binary', 'verbosity': -1}, ds, num_boost_round=3)
print('preds:', m.predict(X[:2]))
""")
```
Assert `success is True` and `"preds:"` in stdout.

---

### TEST 2.4 — `test_sklearn_pipeline_executes`
```python
run_in_sandbox("""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
X = np.random.rand(50, 3); y = np.random.randint(0, 2, 50)
m = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
print('score:', m.score(X, y))
""")
```
Assert `success is True`.

---

### TEST 2.5 — `test_timeout_enforced_within_tolerance`
**Bug:** Timeout fires at `timeout + 30` seconds due to subprocess teardown delay.

```python
import time
start = time.time()
result = run_in_sandbox("import time; time.sleep(999)", timeout=3)
elapsed = time.time() - start
```
Assert `result["timed_out"] is True`.
Assert `elapsed < 10` — the timeout must enforce promptly, not after teardown overhead.

---

### TEST 2.6 — `test_returncode_nonzero_on_script_crash`
```python
result = run_in_sandbox("raise ValueError('deliberate crash')")
```
Assert `result["success"] is False`.
Assert `result["returncode"] != 0`.
Assert `"deliberate" in result["stderr"]`.

**Failure:** `result["success"] is True` with non-zero returncode — the success field is not based on returncode.

---

### TEST 2.7 — `test_stdout_captured_correctly`
```python
result = run_in_sandbox("print('line_one'); print('line_two')")
```
Assert `"line_one" in result["stdout"]` and `"line_two" in result["stdout"]`.

---

### TEST 2.8 — `test_extra_files_available_in_sandbox`
**Bug:** Extra files written to the tempdir but the subprocess's CWD is different — imports fail.

```python
result = run_in_sandbox(
    code="import json; d = json.load(open('config.json')); print(d['key'])",
    extra_files={"config.json": '{"key": "test_value"}'}
)
```
Assert `success is True` and `"test_value" in result["stdout"]`.

---

### TEST 2.9 — `test_memory_limit_kills_oom_process`
**Bug:** Memory limit not set — a trial that allocates 32GB crashes the host instead of the sandbox.

```python
result = run_in_sandbox("""
x = []
for _ in range(100):
    x.append(' ' * (100 * 1024 * 1024))  # 100MB per iteration → 10GB
""", timeout=30)
```
Assert either `result["success"] is False` or `result["timed_out"] is True`. The host process must not be affected — verify the test process itself remains responsive after this call.

---

### TEST 2.10 — `test_existing_sandbox_contract_tests_still_pass`
```bash
pytest tests/contracts/test_sandbox_contract.py -v
```
All existing contract tests must pass unchanged. Interface is frozen.

---

## BLOCK 3 — SERVICE HEALTH: FALLBACK QUALITY
**Class:** `TestServiceHealthFallbackQuality`
**The bug this block catches:** A fallback that silently activates without logging. A retry that uses linear delay instead of exponential. A Groq fallback that calls Groq again instead of Gemini.

---

### TEST 3.1 — `test_groq_fallback_activates_on_connection_error`
Monkeypatch `call_groq` to raise `ConnectionError`. Call `call_groq_safe(...)`. Assert:
- No exception raised
- Gemini was called (monkeypatch `call_gemini` to return `"gemini_response"`)
- Result is `"gemini_response"`

---

### TEST 3.2 — `test_groq_fallback_logs_warning_not_silence`
**Bug:** Fallback activates silently — engineer never knows Groq went down.

Monkeypatch `call_groq` to raise. Capture log output. Assert a `WARNING` log containing `"Groq"` and `"fallback"` was emitted.

---

### TEST 3.3 — `test_retry_delay_is_exponential_not_linear`
**Bug:** Exponential backoff implemented as `delay = base * attempt` (linear) instead of `delay = base * 2^(attempt-1)`.

Monkeypatch `time.sleep` to record call args. Make a function fail twice then succeed. Assert sleep calls are `[2.0, 4.0]` (base=2, doubles each time). Not `[2.0, 2.0]` (linear). Not `[2.0, 6.0]` (additive).

---

### TEST 3.4 — `test_service_unavailable_raised_when_no_fallback_and_all_retries_fail`
A service with `max_attempts=3` and no fallback that fails all 3 times must raise `ServiceUnavailable`. Assert the exception message names the service.

---

### TEST 3.5 — `test_kaggle_api_retry_uses_60s_base_delay`
**Bug:** Kaggle API retried with 2s base delay — Kaggle rate limits require 60s minimum.

Monkeypatch `time.sleep`. Make Kaggle call fail twice. Assert `sleep(60.0)` was called (not `sleep(2.0)`).

---

### TEST 3.6 — `test_chromadb_fallback_returns_empty_list_not_none`
**Bug:** ChromaDB fallback returns `None` — downstream agents do `len(results)` and crash on NoneType.

Monkeypatch `collection.query` to raise. Call `query_chromadb_safe(...)`. Assert result is `[]`, not `None`.

---

### TEST 3.7 — `test_redis_fallback_stores_and_retrieves_within_session`
**Bug:** Redis fallback `_memory_store` is not a shared singleton — each call creates a new empty dict.

Monkeypatch Redis client to raise `ConnectionRefusedError`. Call `redis_set_safe(key="k", value="v")`. Call `redis_get_safe(key="k")`. Assert result is `"v"`. The in-memory store must persist within the same session.

---

### TEST 3.8 — `test_successful_call_does_not_retry`
**Bug:** Retry decorator calls the function twice even on success (retry condition wrong).

Monkeypatch the underlying function to count calls and succeed on first. Call the wrapped version once. Assert call count is exactly 1.

---

## BLOCK 4 — DOCKER REDIS: PERSISTENCE GUARANTEES
**Class:** `TestDockerRedisPersistenceGuarantees`
**The bug this block catches:** Redis connected but `save_state/load_state` loses nested dicts, non-serialisable values crash the save, or loaded state has the wrong types.

---

### TEST 4.1 — `test_real_redis_connected_not_fakeredis`
**Bug:** Docker Redis not running — system silently uses fakeredis and state is lost on restart.

Call `get_redis_client()`. Assert `type(client).__name__ != "FakeRedis"`. If this assertion fails, print the exact `docker run` command to fix it.

---

### TEST 4.2 — `test_state_round_trip_preserves_floats`
Save `{"cv_mean": 0.882145678}`. Load it back. Assert loaded value equals original to 6 decimal places. JSON serialisation must not truncate floats.

---

### TEST 4.3 — `test_state_round_trip_preserves_nested_dicts`
Save `{"competition_context": {"strategy": "conservative", "days_remaining": 2}}`. Load it back. Assert `loaded["competition_context"]["strategy"] == "conservative"`.

---

### TEST 4.4 — `test_non_serialisable_values_excluded_not_crashed`
**Bug:** State contains a Polars DataFrame (not JSON-serialisable) — `save_state` crashes with TypeError.

Add a Polars DataFrame to state. Call `save_state()`. Assert no exception. Assert serialisable keys were saved. Assert the DataFrame key is absent from the loaded state (excluded, not crashed).

---

### TEST 4.5 — `test_ttl_is_set_on_saved_state`
Save state. Check TTL on the key: `client.ttl("professor:state:{session_id}")`. Assert TTL is between 1 and 604800 (7 days). TTL of -1 means no expiry — Redis will hold this key forever, filling disk.

---

### TEST 4.6 — `test_load_state_returns_none_for_missing_key`
Call `load_state("nonexistent-session-xyz")`. Assert result is `None`, not `{}` and not an exception.

---

### TEST 4.7 — `test_hitl_checkpoint_survives_client_reconnection`
**Bug:** State saved via one client instance but not persisted — second client reads nothing.

Save via `get_redis_client()`. Reset `_redis_client = None` (force reconnection). Load via new `get_redis_client()`. Assert state is still there. This is the core guarantee HITL depends on.

---

## BLOCK 5 — PARALLEL DAG: FAN-OUT AND FAN-JOIN CORRECTNESS
**Class:** `TestParallelDAGFanOutFanJoin`
**The bug this block catches:** A fan-out that dispatches both branches but waits for neither, a fan-join that proceeds when one branch is missing, a parallel_groups field that never updates from "pending" to "complete".**

---

### TEST 5.1 — `test_parallel_groups_field_in_initial_state`
Assert `initial_state()["parallel_groups"]` exists and contains keys `"intelligence"`, `"model_trials"`, `"critic"`.

---

### TEST 5.2 — `test_intelligence_group_has_correct_members`
Assert `parallel_groups["intelligence"]["members"]` contains both `"competition_intel"` and `"data_engineer"`.

---

### TEST 5.3 — `test_fan_join_raises_when_schema_missing`
Call `_intelligence_fan_join(state)` where `state` has no `schema_path`. Assert `ValueError` raised containing `"schema"`.

---

### TEST 5.4 — `test_fan_join_raises_when_competition_brief_missing`
Call `_intelligence_fan_join(state)` where `state` has no `competition_brief_path`. Assert `ValueError` raised containing `"competition_brief"`.

---

### TEST 5.5 — `test_fan_join_succeeds_when_both_branches_complete`
Provide state with valid `schema_path` (file exists) and valid `competition_brief_path` (file exists). Call `_intelligence_fan_join(state)`. Assert no exception. Assert `result["parallel_groups"]["intelligence"]["status"] == "complete"`.

---

### TEST 5.6 — `test_model_trial_fan_out_creates_three_sends`
Call `_fan_out_model_trials(state)`. Assert 3 `Send` objects returned, one for each of `"lgbm"`, `"xgb"`, `"catboost"`. Assert each `Send` has a unique `trial_model_type`.

---

### TEST 5.7 — `test_critic_fan_out_creates_four_sends`
Call `_fan_out_critic_vectors(state)`. Assert 4 `Send` objects, one per vector id `[1, 2, 3, 4]`.

---

### TEST 5.8 — `test_parallel_execution_faster_than_serial`
**Bug:** Parallel groups defined in the DAG schema but LangGraph still executes them serially.

Mock each branch to sleep for 1 second. Run the full intelligence fan-out+fan-join. Assert total elapsed time is less than 1.5 seconds (not 2+ seconds which would indicate serial execution).

---

## BLOCK 6 — INNER RETRY LOOP: SELF-CORRECTION QUALITY
**Class:** `TestInnerRetryLoopSelfCorrection`
**The bug this block catches:** A retry loop that retries without adding error context (LLM makes the same mistake), a loop that catches all exceptions including KeyboardInterrupt, a loop that escalates on attempt 2 instead of attempt 3.

---

### TEST 6.1 — `test_retry_loop_present_in_data_engineer`
```python
import inspect
from agents.data_engineer import run_data_engineer
src = inspect.getsource(run_data_engineer)
assert "MAX_INNER_ATTEMPTS" in src or "attempt" in src
assert "handle_escalation" in src or "circuit_breaker" in src
```

---

### TEST 6.2 — `test_retry_loop_present_in_all_required_agents`
Apply the same inspection check to all 8 required agents: `data_engineer, eda_agent, validation_architect, feature_factory, ml_optimizer, red_team_critic, ensemble_architect, competition_intel`.

---

### TEST 6.3 — `test_error_context_grows_on_each_retry`
Patch an agent's `_run_core_logic` to always raise. Run the agent. After 3 attempts: assert `state["error_context"]` has exactly 2 entries (attempts 1 and 2 — attempt 3 triggers escalation).

---

### TEST 6.4 — `test_previous_traceback_injected_into_second_attempt_prompt`
**Bug:** Retry calls `_run_core_logic(state, attempt=2)` but the LLM prompt is identical to attempt 1 — LLM repeats the same mistake.

Patch `_build_system_prompt`. On attempt 2, assert the prompt contains the string `"PREVIOUS ATTEMPT FAILED"`. On attempt 1, assert it does not.

---

### TEST 6.5 — `test_success_on_second_attempt_resets_failure_count`
Patch `_run_core_logic` to fail on attempt 1 and succeed on attempt 2. Run agent. Assert:
- `result["current_node_failure_count"] == 0` (reset on success)
- `result["error_context"] == []` (cleared on success)

---

### TEST 6.6 — `test_escalation_happens_only_after_max_attempts`
Patch `_run_core_logic` to always fail. Assert that `handle_escalation` is called exactly once, only after all 3 attempts are exhausted. Not after attempt 1. Not after attempt 2. Only after 3.

---

### TEST 6.7 — `test_keyboard_interrupt_not_swallowed`
**Bug:** Retry loop catches `Exception` which includes `KeyboardInterrupt` — Ctrl+C cannot stop Professor.

Patch `_run_core_logic` to raise `KeyboardInterrupt`. Assert `KeyboardInterrupt` propagates out of `run_agent()` — it must not be caught by the retry loop.

---

## END-TO-END ADVERSARIAL QUALITY GATE

This smoke test simulates real adversity. Run it last:

```bash
python -c "
import unittest.mock as mock

# Simulate: Groq goes down after 1 call, Redis unavailable, sandbox fails once then succeeds
with mock.patch('tools.llm_tools.call_groq', side_effect=[Exception('Groq rate limit'), 'ok']):
    from core.state import initial_state
    from agents.data_engineer import run_data_engineer

    state = initial_state('stress-test', 'data/spaceship_titanic/train.csv')
    # With retry loop + service health, this should succeed via fallback or retry
    try:
        result = run_data_engineer(state)
        print('Data engineer completed despite Groq failure:', result.get('schema_path'))
    except Exception as e:
        print('FAIL — unhandled exception despite resilience layer:', e)

print('[PASS] Adversarial smoke test complete')
"
```

---

## TOTAL TEST COUNT

| Block | Class | Tests |
|---|---|---|
| 1 — Circuit Breaker Precision | `TestCircuitBreakerEscalationPrecision` | 14 |
| 2 — Subprocess Sandbox | `TestSubprocessSandboxRealExecution` | 10 |
| 3 — Service Health Fallbacks | `TestServiceHealthFallbackQuality` | 8 |
| 4 — Docker Redis Persistence | `TestDockerRedisPersistenceGuarantees` | 7 |
| 5 — Parallel DAG | `TestParallelDAGFanOutFanJoin` | 8 |
| 6 — Inner Retry Loop | `TestInnerRetryLoopSelfCorrection` | 7 |
| **Total** | | **54** |

54 tests. Every one finds a specific named bug in the resilience layer. If all 54 are green, Professor can run overnight unattended without human intervention.

---

## DEFINITION OF DONE FOR TESTS

- [ ] All 54 tests written in `tests/test_day9_quality.py`
- [ ] All 54 tests pass
- [ ] `pytest tests/regression/` — still green (Phase 1 + Day 8 baselines unchanged)
- [ ] `pytest tests/contracts/` — still green (all existing contracts pass with new internals)
- [ ] Test run logged to `tests/logs/day9_quality.log`
- [ ] `git commit -m "Day 9: 54 adversarial resilience tests — all green"`