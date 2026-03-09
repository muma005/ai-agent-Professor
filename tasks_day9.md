# Professor Agent — Day 9 Implementation Guide
**For: Claude Code**
**Status: Day 8 COMPLETE — Phase 2 agents built, all contracts green.**
**Mission: Make Professor resilient. Day 8 gave it intelligence. Day 9 gives it armour.**

---

## ⚠️ NON-NEGOTIABLE RULES BEFORE YOU WRITE A SINGLE LINE

1. **Read the existing codebase first.** Read `tools/e2b_sandbox.py`, `memory/redis_state.py`, `core/professor.py`, and every file in `agents/` before touching anything. You are replacing internals — you must understand what every call site expects.
2. **Contract interfaces are frozen.** The sandbox contract, the Redis interface, and the agent function signatures do not change. Only internals swap. Every existing contract test must still pass after Day 9.
3. **No silent degradation.** Every fallback must log a `WARNING` with the service name, the error, and what fallback was activated. Silent degradation is indistinguishable from a bug.
4. **Regression suite runs after every task.** `pytest tests/regression/ tests/contracts/ -v` must be green before moving to the next task.
5. **Build order is mandatory.** The circuit breaker is a dependency for every other task. Build it first or the retry loops have nowhere to escalate.

---

## BUILD ORDER

```
Task 7  →  Build guards/circuit_breaker.py          (dependency for everything else)
Task 6  →  Add inner retry loop to all agents        (uses circuit_breaker)
Task 3  →  Build guards/service_health.py            (uses circuit_breaker, wraps all services)
Task 1  →  FIX: subprocess sandbox                   (standalone, replaces RestrictedPython)
Task 2  →  Upgrade to Docker Redis                   (standalone, interface unchanged)
Task 4  →  GAP 6: Parallel execution groups in DAG   (uses service_health for external calls)
Task 5  →  Write circuit breaker contract test       (validates all of the above)
            ── commit: "Day 9: resilience layer complete — circuit breaker, sandbox, Redis, parallel DAG" ──
```

---

## TASK 7 — Build `guards/circuit_breaker.py`

**File:** `guards/circuit_breaker.py`
**Priority:** CRITICAL — build first. Everything else in Day 9 escalates into this.
**Why it matters:** Without structured failure escalation, one bad LLM response crashes the entire pipeline. With it, Professor self-heals through 4 levels before asking for human help.

### The Four Escalation Levels

```
failure_count == 1  →  MICRO:   Patch the failing node only. Retry with added context.
failure_count == 2  →  MACRO:   Rewrite the entire DAG. dag_version increments.
failure_count == 3  →  HITL:    Save full state to Redis. Set hitl_required=True. Halt.
budget exhausted    →  TRIAGE:  Stop all non-essential work. Protect submission budget.
```

### Implementation

```python
# guards/circuit_breaker.py

import json
import logging
from enum import Enum
from typing import Optional
from core.state import ProfessorState
from core.lineage import log_event

logger = logging.getLogger(__name__)


class EscalationLevel(str, Enum):
    MICRO   = "micro"    # patch the failing node only
    MACRO   = "macro"    # rewrite the DAG
    HITL    = "hitl"     # pause, save state, alert human
    TRIAGE  = "triage"   # budget/time exhausted, protect rank


def get_escalation_level(state: ProfessorState) -> EscalationLevel:
    """
    Determines which escalation level applies given the current state.
    Called at the top of every agent that has a retry loop.
    """
    budget_remaining = state.get("budget_remaining_usd", float("inf"))
    budget_limit     = state.get("budget_limit_usd", float("inf"))
    time_remaining   = state.get("competition_context", {}).get("hours_remaining")

    # Triage overrides all — check it first
    if budget_remaining <= budget_limit * 0.05:
        return EscalationLevel.TRIAGE
    if time_remaining is not None and time_remaining <= 2:
        return EscalationLevel.TRIAGE

    failure_count = state.get("current_node_failure_count", 0)
    if failure_count >= 3:
        return EscalationLevel.HITL
    if failure_count == 2:
        return EscalationLevel.MACRO
    if failure_count == 1:
        return EscalationLevel.MICRO

    return EscalationLevel.MICRO  # first failure always starts at MICRO


def handle_escalation(
    state: ProfessorState,
    level: EscalationLevel,
    agent_name: str,
    error: Exception,
    traceback_str: str,
) -> ProfessorState:
    """
    Executes the correct response for each escalation level.
    Returns updated state. Never raises — this function must always complete.
    """
    logger.error(
        f"[CircuitBreaker] {agent_name} escalating to {level.value}. "
        f"Failure count: {state.get('current_node_failure_count', 0)}. "
        f"Error: {error}"
    )
    log_event(
        session_id=state["session_id"],
        agent="circuit_breaker",
        action=f"escalation_{level.value}",
        keys_read=["current_node_failure_count"],
        keys_written=["hitl_required", "dag_version"],
        values_changed={
            "level": level.value,
            "agent": agent_name,
            "error": str(error),
        },
    )

    if level == EscalationLevel.MICRO:
        # Append full traceback to the agent's context for next attempt
        error_context = state.get("error_context", [])
        error_context.append({
            "agent":     agent_name,
            "attempt":   state.get("current_node_failure_count", 1),
            "error":     str(error),
            "traceback": traceback_str,
        })
        return {
            **state,
            "error_context":              error_context,
            "current_node_failure_count": state.get("current_node_failure_count", 0) + 1,
        }

    elif level == EscalationLevel.MACRO:
        # Increment dag_version to force a full DAG rewrite on next Supervisor pass
        dag_version = state.get("dag_version", 0) + 1
        logger.warning(
            f"[CircuitBreaker] MACRO replan triggered. "
            f"DAG version incrementing to {dag_version}. "
            f"Supervisor will rewrite execution plan."
        )
        return {
            **state,
            "dag_version":                dag_version,
            "macro_replan_requested":     True,
            "macro_replan_reason":        f"{agent_name} failed twice: {error}",
            "current_node_failure_count": state.get("current_node_failure_count", 0) + 1,
        }

    elif level == EscalationLevel.HITL:
        # Save full state to Redis, pause pipeline, alert human
        _checkpoint_state_to_redis(state, agent_name, error)
        return {
            **state,
            "hitl_required":  True,
            "hitl_reason":    (
                f"Circuit breaker HITL: {agent_name} failed 3 times. "
                f"Last error: {error}. "
                f"Full state checkpointed to Redis. "
                f"Resume with: professor resume --session {state['session_id']}"
            ),
            "pipeline_halted": True,
        }

    elif level == EscalationLevel.TRIAGE:
        budget_remaining = state.get("budget_remaining_usd", 0)
        logger.warning(
            f"[CircuitBreaker] TRIAGE mode. "
            f"Budget remaining: ${budget_remaining:.4f}. "
            f"Stopping all non-essential work. Protecting submission."
        )
        return {
            **state,
            "triage_mode":    True,
            "triage_reason":  f"Budget/time exhausted. Protecting existing submission.",
            "pipeline_halted": True,
        }

    return state  # unreachable but satisfies type checker


def _checkpoint_state_to_redis(
    state: ProfessorState,
    agent_name: str,
    error: Exception,
) -> None:
    """
    Saves full ProfessorState to Redis for HITL resume.
    Fails silently with a warning — the HITL flag is already set.
    """
    try:
        from memory.redis_state import get_redis_client
        client = get_redis_client()
        key    = f"professor:hitl:{state['session_id']}"
        payload = json.dumps({
            "state":      {k: v for k, v in state.items() if _is_serialisable(v)},
            "agent":      agent_name,
            "error":      str(error),
            "checkpointed_at": __import__("datetime").datetime.utcnow().isoformat(),
        })
        client.set(key, payload, ex=86400 * 7)  # 7-day TTL
        logger.info(f"[CircuitBreaker] State checkpointed to Redis key: {key}")
    except Exception as redis_err:
        logger.warning(
            f"[CircuitBreaker] Could not checkpoint to Redis: {redis_err}. "
            f"HITL flag is set but state was not saved. "
            f"Manual recovery required from session logs."
        )


def _is_serialisable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def reset_failure_count(state: ProfessorState) -> ProfessorState:
    """Call this at the top of every agent on successful completion."""
    return {**state, "current_node_failure_count": 0, "error_context": []}
```

### Add to `ProfessorState` and `initial_state()`

```python
# ProfessorState additions:
current_node_failure_count: int
error_context:              list   # [{agent, attempt, error, traceback}]
dag_version:                int
macro_replan_requested:     bool
macro_replan_reason:        str
pipeline_halted:            bool
triage_mode:                bool

# initial_state() additions:
"current_node_failure_count": 0,
"error_context":              [],
"dag_version":                1,
"macro_replan_requested":     False,
"macro_replan_reason":        "",
"pipeline_halted":            False,
"triage_mode":                False,
```

### Verification

```bash
python -c "
from guards.circuit_breaker import get_escalation_level, handle_escalation, EscalationLevel
from core.state import initial_state

s = initial_state('test-cb', 'data/spaceship_titanic/train.csv')

# Test level detection
s2 = {**s, 'current_node_failure_count': 1}
assert get_escalation_level(s2) == EscalationLevel.MICRO

s3 = {**s, 'current_node_failure_count': 2}
assert get_escalation_level(s3) == EscalationLevel.MACRO

s4 = {**s, 'current_node_failure_count': 3}
assert get_escalation_level(s4) == EscalationLevel.HITL

s5 = {**s, 'budget_remaining_usd': 0.001, 'budget_limit_usd': 5.0}
assert get_escalation_level(s5) == EscalationLevel.TRIAGE

print('[PASS] Circuit breaker escalation levels correct')
"
pytest tests/regression/ -v
```

---

## TASK 6 — Add Inner Retry Loop to All Agents

**Files:** `agents/*.py` — every agent that executes LLM calls or runs code
**Priority:** HIGH — this is the most common failure mode. Without it, one bad LLM response kills the entire run.

### The Pattern

Every agent that calls an LLM or runs generated code must wrap its core logic in this exact pattern. Do not deviate from it:

```python
# agents/[any_agent].py

import traceback
from guards.circuit_breaker import (
    get_escalation_level, handle_escalation, reset_failure_count, EscalationLevel
)

MAX_INNER_ATTEMPTS = 3

def run_[agent_name](state: ProfessorState) -> ProfessorState:
    """LangGraph node — inner retry loop with circuit breaker escalation."""

    for attempt in range(1, MAX_INNER_ATTEMPTS + 1):
        try:
            result = _run_core_logic(state, attempt)
            # Success: reset failure count before returning
            return reset_failure_count(result)

        except Exception as e:
            tb = traceback.format_exc()
            print(
                f"[{AGENT_NAME}] Attempt {attempt}/{MAX_INNER_ATTEMPTS} failed. "
                f"Error: {e}"
            )

            if attempt == MAX_INNER_ATTEMPTS:
                # All attempts exhausted — escalate to circuit breaker
                level = get_escalation_level(state)
                return handle_escalation(
                    state=state,
                    level=level,
                    agent_name=AGENT_NAME,
                    error=e,
                    traceback_str=tb,
                )

            # Not yet exhausted — append error context and retry with it
            state = {
                **state,
                "current_node_failure_count": attempt,
                "error_context": state.get("error_context", []) + [{
                    "agent":     AGENT_NAME,
                    "attempt":   attempt,
                    "error":     str(e),
                    "traceback": tb,
                }],
            }

    return state  # unreachable


def _run_core_logic(state: ProfessorState, attempt: int) -> ProfessorState:
    """
    The actual agent logic, isolated so retry can call it cleanly.
    Receives attempt number so it can use error_context from previous attempts
    when constructing LLM prompts.
    """
    # On retry, inject previous error into system prompt:
    error_context = state.get("error_context", [])
    if error_context and attempt > 1:
        last_error = error_context[-1]
        # Prepend to system prompt: "Previous attempt failed with: {last_error['traceback']}"
        # This is how the LLM learns from its own mistakes within a single pipeline run.
        ...

    # ... normal agent logic here ...
```

### Which Agents Get the Retry Loop

Apply to every agent that calls an LLM or executes generated code:

- `agents/data_engineer.py`
- `agents/eda_agent.py`
- `agents/validation_architect.py`
- `agents/feature_factory.py`
- `agents/ml_optimizer.py`
- `agents/red_team_critic.py`
- `agents/ensemble_architect.py`
- `agents/competition_intel.py`

Do **not** apply to: `agents/semantic_router.py` (deterministic routing, no LLM), `agents/submission_strategist.py` (read-only).

### Key Implementation Detail: Error Context in LLM Prompt

The retry loop is only as good as the error information fed back to the LLM. On attempt 2 or 3, the agent's system prompt must include the full traceback from the previous failure. Without this, the LLM will make the same mistake again.

```python
def _build_system_prompt(base_prompt: str, error_context: list, attempt: int) -> str:
    if attempt == 1 or not error_context:
        return base_prompt

    error_block = "\n\n---\nPREVIOUS ATTEMPT FAILED. DO NOT REPEAT THE SAME MISTAKE.\n"
    for ctx in error_context[-2:]:  # last 2 errors max
        error_block += f"\nAttempt {ctx['attempt']} error:\n{ctx['traceback']}\n"
    error_block += "---\n\nRevise your approach based on the above errors.\n"

    return base_prompt + error_block
```

### Verification

```bash
python -c "
# Verify retry loop exists in data_engineer
import inspect
from agents.data_engineer import run_data_engineer
src = inspect.getsource(run_data_engineer)
assert 'MAX_INNER_ATTEMPTS' in src or 'attempt' in src, 'No retry loop found in data_engineer'
assert 'handle_escalation' in src or 'circuit_breaker' in src, 'No circuit breaker in data_engineer'
print('[PASS] Retry loop present in data_engineer')
"
pytest tests/contracts/ -v
```

---

## TASK 3 — Build `guards/service_health.py`

**File:** `guards/service_health.py`
**Priority:** HIGH — this is the single file that prevents cascading failures across all external services.

### The Problem This Solves

Currently, if Groq is rate-limited, the agent crashes. If Docker is unavailable, the sandbox crashes. If the Kaggle API hiccups, the scraper crashes. Each of these crashes uses a circuit breaker attempt. With `service_health.py`, transient failures are absorbed before they ever reach the circuit breaker.

### Implementation

```python
# guards/service_health.py

import time
import logging
import functools
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class ServiceUnavailable(Exception):
    """Raised when a service fails all retries and has no fallback."""
    pass


def with_retry(
    max_attempts: int,
    base_delay_s: float,
    service_name: str,
    fallback: Optional[Callable] = None,
):
    """
    Decorator factory. Wraps any function with exponential backoff retry.
    If all attempts fail and fallback is provided, calls fallback(*args, **kwargs).
    If all attempts fail and no fallback, raises ServiceUnavailable.

    Usage:
        @with_retry(max_attempts=3, base_delay_s=2.0, service_name="Groq API")
        def call_groq(prompt: str) -> str:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    delay = base_delay_s * (2 ** (attempt - 1))
                    if attempt < max_attempts:
                        logger.warning(
                            f"[ServiceHealth] {service_name} attempt {attempt}/{max_attempts} "
                            f"failed: {e}. Retrying in {delay:.1f}s."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[ServiceHealth] {service_name} failed all {max_attempts} attempts. "
                            f"Last error: {e}."
                        )

            if fallback is not None:
                logger.warning(
                    f"[ServiceHealth] {service_name} unavailable. "
                    f"Activating fallback: {fallback.__name__}."
                )
                return fallback(*args, **kwargs)

            raise ServiceUnavailable(
                f"{service_name} failed {max_attempts} attempts. Last error: {last_error}. "
                f"No fallback configured."
            ) from last_error

        return wrapper
    return decorator


# ── Pre-configured wrappers for every external service ────────────────────────

def _groq_fallback(*args, **kwargs):
    """Fall back to Gemini Flash if Groq is down."""
    logger.warning("[ServiceHealth] Falling back to Gemini Flash (Groq unavailable).")
    from tools.llm_tools import call_gemini
    return call_gemini(*args, **kwargs)


def _docker_sandbox_fallback(*args, **kwargs):
    """Fall back to local subprocess if Docker is unavailable."""
    logger.warning("[ServiceHealth] Docker unavailable. Using local subprocess sandbox.")
    from tools.e2b_sandbox import run_in_subprocess_sandbox
    return run_in_subprocess_sandbox(*args, **kwargs)


def _chromadb_fallback(*args, **kwargs):
    """Fall back to empty memory — log warning so engineer knows memory is cold."""
    logger.warning(
        "[ServiceHealth] ChromaDB unavailable. Returning empty memory. "
        "Optuna warm-start disabled for this session."
    )
    return []


def _redis_fallback(key: str, value=None, operation: str = "get"):
    """Fall back to in-memory dict for active session if Redis is unavailable."""
    logger.warning(
        "[ServiceHealth] Redis unavailable. Using in-memory state for this session. "
        "State will not survive process restart."
    )
    _memory_store = getattr(_redis_fallback, "_store", {})
    _redis_fallback._store = _memory_store
    if operation == "set":
        _memory_store[key] = value
        return True
    return _memory_store.get(key)


# ── Public retry-wrapped callables ─────────────────────────────────────────────

@with_retry(max_attempts=3, base_delay_s=2.0, service_name="Groq API", fallback=_groq_fallback)
def call_groq_safe(prompt: str, model: str, **kwargs) -> str:
    from tools.llm_tools import call_groq
    return call_groq(prompt, model, **kwargs)


@with_retry(max_attempts=2, base_delay_s=5.0, service_name="Docker Sandbox", fallback=_docker_sandbox_fallback)
def run_in_sandbox_safe(code: str, timeout: int = 600, **kwargs):
    from tools.e2b_sandbox import run_in_sandbox
    return run_in_sandbox(code, timeout=timeout, **kwargs)


@with_retry(max_attempts=3, base_delay_s=60.0, service_name="Kaggle API")
def call_kaggle_api_safe(fn: Callable, *args, **kwargs):
    """Wrap any kaggle.api call with 60s exponential backoff."""
    return fn(*args, **kwargs)


@with_retry(max_attempts=3, base_delay_s=1.0, service_name="ChromaDB", fallback=_chromadb_fallback)
def query_chromadb_safe(collection, query_texts: list, n_results: int = 5):
    return collection.query(query_texts=query_texts, n_results=n_results)


@with_retry(max_attempts=2, base_delay_s=0.5, service_name="Redis")
def redis_set_safe(client, key: str, value: str, **kwargs):
    return client.set(key, value, **kwargs)


@with_retry(max_attempts=2, base_delay_s=0.5, service_name="Redis")
def redis_get_safe(client, key: str):
    return client.get(key)
```

### Wire Into Codebase

After writing `service_health.py`, go through the codebase and replace every direct external service call:

| Replace this | With this |
|---|---|
| `call_groq(...)` | `call_groq_safe(...)` |
| `run_in_sandbox(...)` | `run_in_sandbox_safe(...)` |
| `kaggle.api.competition_list_files(...)` | `call_kaggle_api_safe(kaggle.api.competition_list_files, ...)` |
| `collection.query(...)` | `query_chromadb_safe(collection, ...)` |
| `redis_client.set(...)` | `redis_set_safe(redis_client, ...)` |
| `redis_client.get(...)` | `redis_get_safe(redis_client, ...)` |

Every external service call in the codebase must go through these wrappers. Search for direct calls: `grep -rn "call_groq\|kaggle.api\|\.query(" agents/ tools/` and replace each one.

---

## TASK 1 — FIX: Replace RestrictedPython with Subprocess Sandbox

**File:** `tools/e2b_sandbox.py`
**Priority:** CRITICAL — RestrictedPython cannot sandbox C-extension libraries (numpy, polars, lightgbm, sklearn). It either silently allows unsafe ops or blocks legitimate ML code with cryptic errors. Feature Factory will break on Day 10–12 without this fix.

### What Changes

Only the internals change. The function signatures, the inputs, the outputs, and all contract tests stay identical. Claude Code must not change any call sites.

```python
# tools/e2b_sandbox.py

import os
import sys
import json
import resource
import tempfile
import subprocess
import textwrap
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SANDBOX_TIMEOUT_S  = 600   # 10 minutes — covers full Optuna trial
MEMORY_LIMIT_BYTES = 6 * 1024 * 1024 * 1024   # 6 GB


def _set_memory_limit():
    """
    Called as preexec_fn in subprocess.run.
    Sets address space limit on the child process before exec.
    This is the only reliable way to enforce memory limits on ML workloads.
    """
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except ValueError:
        # Some systems don't support RLIMIT_AS — fall back to RLIMIT_DATA
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        except ValueError:
            pass  # Best effort — don't crash if limits can't be set


def run_in_sandbox(
    code: str,
    timeout: int = SANDBOX_TIMEOUT_S,
    extra_files: Optional[dict] = None,
    working_dir: Optional[str] = None,
) -> dict:
    """
    Executes Python code in an isolated subprocess.
    Replaces RestrictedPython — identical interface, subprocess internals.

    Args:
        code:        Python source code to execute.
        timeout:     Seconds before the subprocess is killed. Default 600.
        extra_files: Dict of {filename: content_str} to write alongside the script.
        working_dir: Working directory for the subprocess. Defaults to a fresh tempdir.

    Returns:
        {
          "stdout":      str,    # captured standard output
          "stderr":      str,    # captured standard error
          "returncode":  int,    # 0 = success
          "success":     bool,   # True iff returncode == 0
          "timed_out":   bool,   # True iff subprocess.TimeoutExpired was raised
        }

    Never raises — all errors are returned in the result dict.
    """
    with tempfile.TemporaryDirectory(prefix="professor_sandbox_") as tmpdir:
        work_dir = working_dir or tmpdir

        # Write extra files into the sandbox directory
        if extra_files:
            for fname, content in extra_files.items():
                fpath = os.path.join(work_dir, fname)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w") as f:
                    f.write(content)

        # Write the main script
        script_path = os.path.join(work_dir, "_professor_script.py")
        with open(script_path, "w") as f:
            f.write(code)

        logger.debug(f"[Sandbox] Executing script in {work_dir} (timeout={timeout}s)")

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
                preexec_fn=_set_memory_limit,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return {
                "stdout":     result.stdout,
                "stderr":     result.stderr,
                "returncode": result.returncode,
                "success":    result.returncode == 0,
                "timed_out":  False,
            }

        except subprocess.TimeoutExpired as e:
            logger.warning(
                f"[Sandbox] Script timed out after {timeout}s. "
                f"Partial stdout: {e.stdout[:500] if e.stdout else '(none)'}"
            )
            return {
                "stdout":     e.stdout or "",
                "stderr":     f"TimeoutExpired: script exceeded {timeout}s limit.",
                "returncode": -1,
                "success":    False,
                "timed_out":  True,
            }

        except Exception as e:
            logger.error(f"[Sandbox] Unexpected error: {e}")
            return {
                "stdout":     "",
                "stderr":     f"SandboxError: {e}",
                "returncode": -1,
                "success":    False,
                "timed_out":  False,
            }


def run_in_subprocess_sandbox(code: str, timeout: int = SANDBOX_TIMEOUT_S, **kwargs) -> dict:
    """Alias for run_in_sandbox — used as fallback target in service_health.py."""
    return run_in_sandbox(code, timeout=timeout, **kwargs)
```

### Verification

```bash
python -c "
from tools.e2b_sandbox import run_in_sandbox

# Test 1: numpy + polars work (RestrictedPython blocked these)
result = run_in_sandbox('''
import numpy as np
import polars as pl
arr = np.array([1, 2, 3])
df = pl.DataFrame({'x': arr})
print('numpy:', arr.mean())
print('polars:', df['x'].mean())
''')
assert result['success'], f'C-extension test failed: {result[\"stderr\"]}'
assert 'numpy: 2.0' in result['stdout']
assert 'polars: 2.0' in result['stdout']

# Test 2: Timeout enforcement
result = run_in_sandbox('import time; time.sleep(999)', timeout=2)
assert result['timed_out'], 'Timeout was not enforced'

# Test 3: Return code captured
result = run_in_sandbox('raise ValueError(\"deliberate error\")')
assert not result['success']
assert result['returncode'] != 0
assert 'deliberate error' in result['stderr']

print('[PASS] Subprocess sandbox working correctly')
"
pytest tests/contracts/ -v   # All existing sandbox contracts must still pass
```

---

## TASK 2 — Upgrade to Docker Redis for HITL Checkpointing

**File:** `memory/redis_state.py`
**Priority:** HIGH — HITL checkpointing introduced in Task 7 requires real persistence. fakeredis state dies when the process restarts. A paused pipeline must survive.

### Setup

```bash
docker run -d --name professor-redis -p 6379:6379 redis:7-alpine
```

Verify: `redis-cli ping` → `PONG`

### Implementation

Only the connection setup changes. Every call site uses the same client interface.

```python
# memory/redis_state.py

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_redis_client = None   # module-level singleton — built once, reused


def get_redis_client():
    """
    Returns a Redis client. Tries real Redis first, falls back to fakeredis.
    Fall back is logged as a WARNING — it is never silent.

    Call this once at startup. The module-level singleton means the connection
    is not re-established on every state read/write.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db   = int(os.getenv("REDIS_DB", "0"))

    # Try real Redis first
    try:
        import redis
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            socket_connect_timeout=3,   # fail fast if Docker is not running
            socket_timeout=5,
            decode_responses=True,
        )
        client.ping()   # validates the connection immediately
        logger.info(f"[Redis] Connected to real Redis at {redis_host}:{redis_port}")
        _redis_client = client
        return _redis_client

    except Exception as real_redis_err:
        logger.warning(
            f"[Redis] Real Redis unavailable at {redis_host}:{redis_port}: {real_redis_err}. "
            f"Falling back to fakeredis. "
            f"WARNING: State will not persist across process restarts. "
            f"HITL checkpointing is disabled for this session. "
            f"Fix: docker run -d -p 6379:6379 redis:7-alpine"
        )
        import fakeredis
        _redis_client = fakeredis.FakeRedis(decode_responses=True)
        return _redis_client


def save_state(session_id: str, state: dict, ttl_seconds: int = 86400 * 7) -> bool:
    """
    Serialises and saves ProfessorState to Redis.
    Returns True on success, False on failure (never raises).
    """
    import json
    from guards.service_health import redis_set_safe
    client = get_redis_client()
    key    = f"professor:state:{session_id}"
    try:
        payload = json.dumps({k: v for k, v in state.items() if _is_serialisable(v)})
        redis_set_safe(client, key, payload, ex=ttl_seconds)
        return True
    except Exception as e:
        logger.error(f"[Redis] Failed to save state for session {session_id}: {e}")
        return False


def load_state(session_id: str) -> Optional[dict]:
    """
    Loads ProfessorState from Redis. Returns None if not found.
    """
    import json
    from guards.service_health import redis_get_safe
    client = get_redis_client()
    key    = f"professor:state:{session_id}"
    try:
        raw = redis_get_safe(client, key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.error(f"[Redis] Failed to load state for session {session_id}: {e}")
        return None


def _is_serialisable(value) -> bool:
    import json
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
```

### Verification

```bash
# Verify Docker Redis is running
redis-cli ping

python -c "
from memory.redis_state import get_redis_client, save_state, load_state

# Test 1: Connection to real Redis
client = get_redis_client()
client.set('test_key', 'test_value', ex=60)
val = client.get('test_key')
assert val == 'test_value', f'Redis read failed: {val}'

# Test 2: State round-trip
state = {'session_id': 'test-redis-persist', 'cv_mean': 0.8821, 'task_type': 'tabular'}
save_state('test-redis-persist', state)
loaded = load_state('test-redis-persist')
assert loaded is not None
assert loaded['cv_mean'] == 0.8821
assert loaded['task_type'] == 'tabular'

print('[PASS] Docker Redis persistence verified — state survives serialisation round-trip')
"
```

---

## TASK 4 — GAP 6: Parallel Execution Groups in DAG

**Files:** `core/professor.py`, `agents/semantic_router.py`
**Priority:** HIGH — serial execution wastes wall-clock time. Parallel groups make Professor significantly faster on competition day.

### The Three Parallel Groups

```
Group 1 — Intelligence Gathering (fan-out immediately after semantic_router):
  Branch A: competition_intel    (needs only raw_data_path + competition_name)
  Branch B: data_engineer        (needs only raw_data_path)
  Fan-join: both complete → eda_agent (needs schema from data_engineer + brief from intel)

Group 2 — Model Trials (fan-out from validation_architect):
  Branch A: run_lgbm_trial       (separate E2B sandbox)
  Branch B: run_xgb_trial        (separate E2B sandbox)
  Branch C: run_catboost_trial   (separate E2B sandbox)
  Fan-join: all complete → ensemble_architect

Group 3 — Critic Vectors (fan-out from Feature Factory):
  Branch A: critic_vector_1      (leakage detection)
  Branch B: critic_vector_2      (preprocessing audit)
  Branch C: critic_vector_3      (overfitting signals)
  Branch D: critic_vector_4      (submission format)
  Fan-join: all complete → ensemble_architect
```

### LangGraph Send API Implementation

```python
# core/professor.py

from langgraph.graph import StateGraph, END
from langgraph.types import Send

def _fan_out_intelligence(state: ProfessorState) -> list:
    """
    Fan-out node: dispatches to competition_intel and data_engineer in parallel.
    Returns a list of Send objects — LangGraph executes them concurrently.
    """
    return [
        Send("competition_intel", state),
        Send("data_engineer",     state),
    ]


def _fan_out_model_trials(state: ProfessorState) -> list:
    """
    Fan-out node: dispatches one trial per model type.
    Each trial runs in its own E2B sandbox subprocess — true parallelism.
    """
    model_types = ["lgbm", "xgb", "catboost"]
    return [
        Send("run_model_trial", {**state, "trial_model_type": model})
        for model in model_types
    ]


def _fan_out_critic_vectors(state: ProfessorState) -> list:
    """
    Fan-out node: all 4 critic vectors are fully independent.
    Order does not matter. Slowest vector determines total critic time.
    """
    vectors = [1, 2, 3, 4]
    return [
        Send("run_critic_vector", {**state, "critic_vector_id": v})
        for v in vectors
    ]
```

### `parallel_groups` Field in ProfessorState

```python
# ProfessorState addition:
parallel_groups: dict   # {group_name: {status, started_at, completed_at, members}}

# initial_state() addition:
"parallel_groups": {
    "intelligence": {"status": "pending", "members": ["competition_intel", "data_engineer"]},
    "model_trials": {"status": "pending", "members": ["lgbm", "xgb", "catboost"]},
    "critic":       {"status": "pending", "members": ["vector_1", "vector_2", "vector_3", "vector_4"]},
},
```

### Fan-Join Guard

At each fan-join node, verify all expected branches completed before proceeding:

```python
def _intelligence_fan_join(state: ProfessorState) -> ProfessorState:
    """
    Fan-join after competition_intel + data_engineer.
    Verifies both outputs exist before handing off to eda_agent.
    """
    missing = []
    if not state.get("schema_path") or not os.path.exists(state.get("schema_path", "")):
        missing.append("schema.json (data_engineer)")
    if not state.get("competition_brief_path"):
        missing.append("competition_brief.json (competition_intel)")

    if missing:
        raise ValueError(
            f"[FanJoin:intelligence] Expected outputs missing: {missing}. "
            f"One or more parallel branches did not complete."
        )

    state["parallel_groups"]["intelligence"]["status"] = "complete"
    return state
```

### Verification

```bash
python -c "
from core.professor import build_graph
graph = build_graph()

# Verify parallel edges exist
nodes = list(graph.nodes)
assert 'competition_intel' in nodes
assert 'data_engineer' in nodes
print('Nodes:', nodes)

# Verify parallel_groups in initial_state
from core.state import initial_state
s = initial_state('test-parallel', 'data/spaceship_titanic/train.csv')
assert 'parallel_groups' in s
assert 'intelligence' in s['parallel_groups']
print('[PASS] Parallel execution groups wired in DAG')
"
pytest tests/regression/ -v
```

---

## TASK 5 — Contract Test: Circuit Breaker

**File:** `tests/contracts/test_circuit_breaker_contract.py`
**Status after writing: IMMUTABLE — do not edit after Day 9**

```python
# tests/contracts/test_circuit_breaker_contract.py
# ─────────────────────────────────────────────────────────────────────────────
# Written: Day 9   Status: IMMUTABLE
#
# CONTRACT: guards/circuit_breaker.py
#   failure_count=1 → MICRO (patch node, append error context)
#   failure_count=2 → MACRO (rewrite DAG, dag_version increments)
#   failure_count=3 → HITL  (hitl_required=True, pipeline_halted=True)
#   budget exhausted → TRIAGE (triage_mode=True, pipeline_halted=True)
#   success           → reset_failure_count returns count to 0
# ─────────────────────────────────────────────────────────────────────────────
import pytest
from core.state import initial_state
from guards.circuit_breaker import (
    get_escalation_level,
    handle_escalation,
    reset_failure_count,
    EscalationLevel,
)

FIXTURE_CSV = "tests/fixtures/tiny_train.csv"


@pytest.fixture
def base_state():
    return initial_state("test-cb-contract", FIXTURE_CSV)


class TestCircuitBreakerContract:

    # ── Level detection ──────────────────────────────────────────────────────

    def test_first_failure_is_micro(self, base_state):
        state = {**base_state, "current_node_failure_count": 1}
        assert get_escalation_level(state) == EscalationLevel.MICRO

    def test_second_failure_is_macro(self, base_state):
        state = {**base_state, "current_node_failure_count": 2}
        assert get_escalation_level(state) == EscalationLevel.MACRO

    def test_third_failure_is_hitl(self, base_state):
        state = {**base_state, "current_node_failure_count": 3}
        assert get_escalation_level(state) == EscalationLevel.HITL

    def test_budget_exhaustion_is_triage(self, base_state):
        state = {**base_state, "budget_remaining_usd": 0.01, "budget_limit_usd": 5.0}
        assert get_escalation_level(state) == EscalationLevel.TRIAGE

    def test_triage_overrides_failure_count(self, base_state):
        """TRIAGE takes priority even if failure_count is only 1."""
        state = {
            **base_state,
            "current_node_failure_count": 1,
            "budget_remaining_usd": 0.01,
            "budget_limit_usd": 5.0,
        }
        assert get_escalation_level(state) == EscalationLevel.TRIAGE

    def test_time_exhaustion_is_triage(self, base_state):
        state = {
            **base_state,
            "competition_context": {
                **base_state.get("competition_context", {}),
                "hours_remaining": 1,
            },
        }
        assert get_escalation_level(state) == EscalationLevel.TRIAGE

    # ── MICRO behaviour ──────────────────────────────────────────────────────

    def test_micro_appends_error_context(self, base_state):
        state = {**base_state, "current_node_failure_count": 1}
        result = handle_escalation(
            state=state,
            level=EscalationLevel.MICRO,
            agent_name="test_agent",
            error=ValueError("test error"),
            traceback_str="Traceback (most recent call last): ...",
        )
        assert len(result["error_context"]) == 1
        assert result["error_context"][0]["agent"] == "test_agent"
        assert "test error" in result["error_context"][0]["error"]

    def test_micro_increments_failure_count(self, base_state):
        state = {**base_state, "current_node_failure_count": 1}
        result = handle_escalation(
            state=state,
            level=EscalationLevel.MICRO,
            agent_name="test_agent",
            error=ValueError("micro error"),
            traceback_str="tb",
        )
        assert result["current_node_failure_count"] == 2

    def test_micro_does_not_set_hitl(self, base_state):
        state = {**base_state, "current_node_failure_count": 1}
        result = handle_escalation(
            state=state, level=EscalationLevel.MICRO,
            agent_name="a", error=ValueError("e"), traceback_str="t"
        )
        assert result.get("hitl_required") is not True

    # ── MACRO behaviour ──────────────────────────────────────────────────────

    def test_macro_increments_dag_version(self, base_state):
        state = {**base_state, "current_node_failure_count": 2, "dag_version": 1}
        result = handle_escalation(
            state=state,
            level=EscalationLevel.MACRO,
            agent_name="test_agent",
            error=RuntimeError("macro error"),
            traceback_str="tb",
        )
        assert result["dag_version"] == 2, (
            f"dag_version should have incremented to 2, got {result['dag_version']}"
        )

    def test_macro_sets_replan_flag(self, base_state):
        state = {**base_state, "current_node_failure_count": 2}
        result = handle_escalation(
            state=state, level=EscalationLevel.MACRO,
            agent_name="a", error=RuntimeError("e"), traceback_str="t"
        )
        assert result["macro_replan_requested"] is True

    def test_macro_replan_reason_names_agent(self, base_state):
        state = {**base_state, "current_node_failure_count": 2}
        result = handle_escalation(
            state=state, level=EscalationLevel.MACRO,
            agent_name="ml_optimizer", error=RuntimeError("OOM"), traceback_str="t"
        )
        assert "ml_optimizer" in result["macro_replan_reason"]

    def test_macro_does_not_halt_pipeline(self, base_state):
        """MACRO replans but does not halt — the Supervisor rewrites and continues."""
        state = {**base_state, "current_node_failure_count": 2}
        result = handle_escalation(
            state=state, level=EscalationLevel.MACRO,
            agent_name="a", error=RuntimeError("e"), traceback_str="t"
        )
        assert result.get("pipeline_halted") is not True

    # ── HITL behaviour ───────────────────────────────────────────────────────

    def test_hitl_sets_hitl_required_true(self, base_state):
        state = {**base_state, "current_node_failure_count": 3}
        result = handle_escalation(
            state=state,
            level=EscalationLevel.HITL,
            agent_name="feature_factory",
            error=Exception("persistent failure"),
            traceback_str="tb",
        )
        assert result["hitl_required"] is True

    def test_hitl_halts_pipeline(self, base_state):
        state = {**base_state, "current_node_failure_count": 3}
        result = handle_escalation(
            state=state, level=EscalationLevel.HITL,
            agent_name="a", error=Exception("e"), traceback_str="t"
        )
        assert result["pipeline_halted"] is True

    def test_hitl_reason_names_agent_and_session(self, base_state):
        state = {**base_state, "current_node_failure_count": 3}
        result = handle_escalation(
            state=state, level=EscalationLevel.HITL,
            agent_name="red_team_critic", error=Exception("critical fail"), traceback_str="t"
        )
        reason = result.get("hitl_reason", "")
        assert "red_team_critic" in reason, f"Agent name missing from hitl_reason: {reason}"
        assert base_state["session_id"] in reason, f"Session ID missing from hitl_reason: {reason}"

    # ── TRIAGE behaviour ─────────────────────────────────────────────────────

    def test_triage_sets_triage_mode(self, base_state):
        state = {**base_state, "budget_remaining_usd": 0.001, "budget_limit_usd": 5.0}
        result = handle_escalation(
            state=state, level=EscalationLevel.TRIAGE,
            agent_name="a", error=Exception("budget"), traceback_str="t"
        )
        assert result["triage_mode"] is True

    def test_triage_halts_pipeline(self, base_state):
        state = {**base_state, "budget_remaining_usd": 0.001, "budget_limit_usd": 5.0}
        result = handle_escalation(
            state=state, level=EscalationLevel.TRIAGE,
            agent_name="a", error=Exception("e"), traceback_str="t"
        )
        assert result["pipeline_halted"] is True

    # ── Reset ────────────────────────────────────────────────────────────────

    def test_reset_sets_failure_count_to_zero(self, base_state):
        state = {**base_state, "current_node_failure_count": 2, "error_context": [{"x": 1}]}
        result = reset_failure_count(state)
        assert result["current_node_failure_count"] == 0

    def test_reset_clears_error_context(self, base_state):
        state = {**base_state, "error_context": [{"agent": "x", "error": "y"}]}
        result = reset_failure_count(state)
        assert result["error_context"] == []

    def test_handle_escalation_never_raises(self, base_state):
        """The circuit breaker must never crash — it is the last line of defence."""
        for level in EscalationLevel:
            try:
                handle_escalation(
                    state=base_state,
                    level=level,
                    agent_name="test",
                    error=Exception("test"),
                    traceback_str="test traceback",
                )
            except Exception as e:
                pytest.fail(
                    f"handle_escalation raised an exception for level {level}: {e}. "
                    "The circuit breaker must never raise — it is the last line of defence."
                )
```

---

## END OF DAY CHECKLIST

```bash
# 1. Regression suite — must not break Phase 1 or Day 8
pytest tests/regression/ -v

# 2. All contract tests
pytest tests/contracts/ -v

# 3. Circuit breaker contract specifically
pytest tests/contracts/test_circuit_breaker_contract.py -v -s

# 4. Sandbox replaces RestrictedPython — C-extensions must work
python -c "
from tools.e2b_sandbox import run_in_sandbox
import lightgbm as lgb, numpy as np
result = run_in_sandbox('''
import lightgbm as lgb, numpy as np
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
ds = lgb.Dataset(X, label=y)
params = {'objective': 'binary', 'verbosity': -1}
model = lgb.train(params, ds, num_boost_round=5)
print('LightGBM score:', model.predict(X[:3]))
''')
assert result['success'], result['stderr']
print('[PASS] LightGBM runs in subprocess sandbox')
"

# 5. Redis persistence — state survives a save/load round-trip
python -c "
from memory.redis_state import save_state, load_state
save_state('test-persist', {'cv_mean': 0.8821, 'data_hash': 'abc123'})
s = load_state('test-persist')
assert s['cv_mean'] == 0.8821
print('[PASS] Redis state persistence verified')
"

# 6. Commit
git add .
git commit -m "Day 9: circuit breaker, subprocess sandbox, Docker Redis, service health, parallel DAG, inner retry loops — all contracts green"
git push origin phase-2
```

### Definition of Done for Day 9

- [ ] `guards/circuit_breaker.py` — all 4 levels implemented, never raises
- [ ] Inner retry loop in all 8 agents — 3 attempts, full traceback fed back on failure
- [ ] `guards/service_health.py` — all 5 services wrapped with retry + fallback
- [ ] `tools/e2b_sandbox.py` — subprocess replaces RestrictedPython, C-extensions work, timeout enforced
- [ ] `memory/redis_state.py` — Docker Redis connected, fakeredis fallback logs warning
- [ ] Parallel groups 1+2+3 wired in LangGraph via Send API
- [ ] `parallel_groups` field in ProfessorState
- [ ] Circuit breaker contract test — 22 tests written, all green, file immutable
- [ ] `pytest tests/regression/` — green (Phase 1 and Day 8 baselines unchanged)
- [ ] `pytest tests/contracts/` — green (all existing contracts pass with new internals)

---

## WHAT PODIUM WORK LOOKS LIKE ON THIS DAY

Day 9 is the day Professor stops being fragile. After today, you should be able to:

- Kill the Groq API mid-run and watch it seamlessly switch to Gemini
- Pull the Docker container while a trial is running and watch Redis preserve the checkpoint
- Inject a broken LLM response and watch the retry loop feed the error back for self-correction
- Run three model trials simultaneously and see all three complete in the time one used to take

If any of these break after Day 9, the resilience layer is incomplete. Professor entering a real competition must be able to run overnight unattended. That starts today.