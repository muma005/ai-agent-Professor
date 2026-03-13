# Professor Agent — Day 12 Implementation
**For: Claude Code**
**Theme: Podium-level hardening — the system must survive overnight unsupervised**

Day 9 built the circuit breaker skeleton (detection, levels, Redis checkpoint, flag).
Day 12 completes the human interaction layer and adds two critical production fixes.

Build order: Task 1 → Task 2 → Task 3 → Task 4

```
Task 1  →  Complete HITL human layer: generate_hitl_prompt() + resume_from_checkpoint()
Task 2  →  Fix OOM: ml_optimizer.py memory guardrails
Task 3  →  Fix cost: professor.py LangSmith tracing control + .env.example
Task 4  →  Write tests: tests/test_day12_quality.py
           commit: "Day 12: HITL human layer, OOM guardrails, tracing cost control"
```

---

## TASK 1 — Complete `guards/circuit_breaker.py` HITL Human Layer

**Status of Day 9 code:** Escalation detection ✓. Redis checkpoint ✓. HITL flag set ✓. Human interaction: one stub line — `"Resume with: professor resume --session {id}"`. That is what Day 12 fixes.

**Two functions to add:**

### 1a. `generate_hitl_prompt(state, agent_name, error) -> dict`

Called immediately after `_checkpoint_state_to_redis()` inside `handle_escalation()` when `level == EscalationLevel.HITL`.

```python
def generate_hitl_prompt(
    state: ProfessorState,
    agent_name: str,
    error: Exception,
) -> dict:
    """
    Builds the human-facing HITL prompt: what failed, why, 3 structured interventions.
    Writes to outputs/{session_id}/hitl_prompt.json for CLI pickup.
    Never raises — called in a failure path.
    """
    error_class = _classify_error(agent_name, error)
    interventions = _build_interventions(state, agent_name, error_class, error)  # always returns list of 3

    prompt = {
        "session_id":        state["session_id"],
        "failed_agent":      agent_name,
        "failure_count":     state.get("current_node_failure_count", 3),
        "what_was_attempted": _describe_attempt(state, agent_name),
        "why_it_failed":     str(error)[:500],   # truncate tracebacks
        "error_class":       error_class,
        "interventions":     interventions,
        "resume_command":    f"professor resume --session {state['session_id']}",
        "checkpoint_key":    f"professor:hitl:{state['session_id']}",
        "generated_at":      datetime.utcnow().isoformat(),
    }

    _write_hitl_prompt(state["session_id"], prompt)  # silently fails
    _print_hitl_banner(prompt)                        # terminal output for interactive sessions
    return prompt
```

### Error classification → `_classify_error(agent_name, error) -> str`

Returns one of 5 strings. Never raises.

```python
ERROR_CLASS_MAP = {
    "KeyError":            "data_quality",
    "ValueError":          "data_quality",
    "AttributeError":      "data_quality",
    "MemoryError":         "memory",
    "RuntimeError":        "model_failure",
    "optuna":              "model_failure",   # match in repr(error)
    "TimeoutError":        "api_timeout",
    "httpx":               "api_timeout",     # match in repr(type(error))
    "groq":                "api_timeout",
}
# Default: "unknown"
# Match on type name first, then repr of type for library errors
```

### Intervention structure

Each intervention is a dict. `_build_interventions()` must return **exactly 3**.

```python
{
    "id":          1,           # 1, 2, or 3. Displayed as "[1]", "[2]", "[3]" in terminal
    "label":       str,         # ≤ 60 chars, imperative. "Skip validation, use raw features."
    "action_type": "AUTO",      # "AUTO" = Professor applies it on resume. "MANUAL" = engineer does it first.
    "risk":        "LOW",       # "LOW" | "MEDIUM" | "HIGH"
    "description": str,         # 1-2 sentences. What this does, what it sacrifices.
    "code_hint":   str | None,  # Optional: exact command or line to change.
}
```

### Intervention templates per error class

```python
INTERVENTION_TEMPLATES = {
    "data_quality": [
        {"id": 1, "label": "Skip validation, proceed with raw features.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Bypasses all type-checking and missing-value validation. "
                        "Proceeds directly with whatever columns exist. May produce worse CV.",
         "code_hint": None},
        {"id": 2, "label": "Drop columns with > 30% nulls, fill rest with median.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Conservative imputation. Loses high-null features entirely. "
                        "Safe starting point for any tabular competition.",
         "code_hint": None},
        {"id": 3, "label": "Inspect data manually, then rerun.",
         "action_type": "MANUAL", "risk": "LOW",
         "description": "Open the raw CSV and check column names, dtypes, and sample rows. "
                        "Common cause: target column name differs from expected.",
         "code_hint": "import polars as pl; print(pl.read_csv('data/train.csv').head(5))"},
    ],
    "model_failure": [
        {"id": 1, "label": "Reduce LightGBM to 100 trees, disable early stopping.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Minimal model to prove the pipeline runs end-to-end. "
                        "CV score will be suboptimal but submission will complete.",
         "code_hint": None},
        {"id": 2, "label": "Switch to LogisticRegression fallback model.",
         "action_type": "AUTO", "risk": "MEDIUM",
         "description": "Replaces all boosting models with sklearn LogisticRegression. "
                        "Will underfit complex competitions but always converges.",
         "code_hint": None},
        {"id": 3, "label": "Check training data shape and feature count.",
         "action_type": "MANUAL", "risk": "LOW",
         "description": "Log X_train.shape before the failing model call to rule out "
                        "zero-feature or zero-row edge cases.",
         "code_hint": "print(f'X_train shape: {X_train.shape}, y shape: {y_train.shape}')"},
    ],
    "memory": [
        {"id": 1, "label": "Sample training data to 50% and retry.",
         "action_type": "AUTO", "risk": "MEDIUM",
         "description": "Randomly samples 50% of training rows. CV score will degrade "
                        "slightly. Prevents OOM for most datasets up to 2GB.",
         "code_hint": None},
        {"id": 2, "label": "Switch to n_jobs=1, reduce n_estimators to 200.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Forces single-threaded training and limits tree count. "
                        "Halves peak memory usage at ~15% CV score cost.",
         "code_hint": None},
        {"id": 3, "label": "Free memory manually, then rerun.",
         "action_type": "MANUAL", "risk": "LOW",
         "description": "Close all other applications. Restart the Python process to "
                        "clear fragmented memory before retrying.",
         "code_hint": "import gc; gc.collect()  # then restart: python -m professor run"},
    ],
    "api_timeout": [
        {"id": 1, "label": "Retry with 2x timeout and exponential backoff.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Doubles the API timeout and retries with 2s/4s/8s delays. "
                        "Handles transient Groq rate limits and network blips.",
         "code_hint": None},
        {"id": 2, "label": "Switch to local LLM (ollama/llama3) for this session.",
         "action_type": "AUTO", "risk": "MEDIUM",
         "description": "Falls back to a local model. Slower but zero API cost and "
                        "no rate limits. Requires ollama running on localhost:11434.",
         "code_hint": "ollama pull llama3; export PROFESSOR_LLM_PROVIDER=local"},
        {"id": 3, "label": "Check API status and keys, then resume.",
         "action_type": "MANUAL", "risk": "LOW",
         "description": "Verify GROQ_API_KEY in .env is valid and not expired. "
                        "Check status.groq.com for incidents.",
         "code_hint": "cat .env | grep GROQ_API_KEY"},
    ],
    "unknown": [
        {"id": 1, "label": "Retry this agent with extra debug logging.",
         "action_type": "AUTO", "risk": "LOW",
         "description": "Re-runs the failed agent with LOG_LEVEL=DEBUG to capture "
                        "the full traceback and intermediate state.",
         "code_hint": None},
        {"id": 2, "label": "Skip this agent and continue with defaults.",
         "action_type": "AUTO", "risk": "MEDIUM",
         "description": "Bypasses the failing agent entirely. Pipeline continues "
                        "with whatever state exists before the failure.",
         "code_hint": None},
        {"id": 3, "label": "Inspect session state and rerun manually.",
         "action_type": "MANUAL", "risk": "LOW",
         "description": "Load the Redis checkpoint and inspect state before the failure.",
         "code_hint": f"professor inspect --session <id>"},
    ],
}
```

### `_print_hitl_banner(prompt: dict) -> None`

Prints to stdout. Called in interactive terminal sessions. Does not raise.

```
╔══════════════════════════════════════════════════════════╗
║              ⚠  PROFESSOR PAUSED — HITL REQUIRED         ║
╚══════════════════════════════════════════════════════════╝

Agent:    data_engineer
Failure:  3 consecutive failures (KeyError: 'target_column')
Class:    data_quality
Session:  abc123

What was attempted:
  data_engineer attempted to validate column types and impute missing values.

Why it failed:
  KeyError: 'target_column' — the expected target column was not found in the CSV.

Suggested interventions:
  [1] AUTO  (LOW risk)  Skip validation, proceed with raw features.
             Common fix when target column has been renamed.

  [2] AUTO  (LOW risk)  Drop columns with > 30% nulls, fill rest with median.
             Conservative imputation. Proceeds without validation.

  [3] MANUAL (LOW risk) Inspect data manually, then rerun.
             HINT: import polars as pl; print(pl.read_csv('data/train.csv').head(5))

To resume: professor resume --session abc123
State saved to Redis key: professor:hitl:abc123

Progress will wait until you resume. Nothing will be lost.
```

### 1b. `resume_from_checkpoint(session_id, intervention_id) -> ProfessorState`

```python
def resume_from_checkpoint(session_id: str, intervention_id: int) -> ProfessorState:
    """
    Loads state from Redis checkpoint, applies intervention, returns clean state
    ready to re-enter the pipeline at the failed node.

    Called by: `professor resume --session <id>` CLI.
    Never raises — returns error state dict on failure instead.
    """
    key = f"professor:hitl:{session_id}"

    # 1. Load from Redis
    raw = redis_client.get(key)
    if raw is None:
        return _error_state(session_id, f"No checkpoint found for key: {key}")

    # 2. Validate checkpoint format
    try:
        checkpoint = json.loads(raw)
        state = checkpoint["state"]
        agent_name = checkpoint["agent_name"]
        error_class = checkpoint.get("error_class", "unknown")
    except (json.JSONDecodeError, KeyError) as e:
        return _error_state(session_id, f"Checkpoint corrupt: {e}")

    # 3. Validate intervention_id
    if intervention_id not in (1, 2, 3):
        return _error_state(session_id, f"intervention_id must be 1, 2, or 3. Got: {intervention_id}")

    # 4. Apply intervention
    intervention = INTERVENTION_TEMPLATES[error_class][intervention_id - 1]
    state = _apply_intervention(state, intervention, agent_name)

    # 5. Reset failure state for clean re-entry
    state = {
        **state,
        "current_node_failure_count": 0,
        "hitl_required":             False,
        "hitl_intervention_id":      intervention_id,
        "hitl_intervention_label":   intervention["label"],
        "replan_requested":          False,
    }

    # 6. Log to lineage
    log_event(
        state=state,
        action="hitl_resumed",
        agent="circuit_breaker",
        details={
            "session_id":         session_id,
            "intervention_id":    intervention_id,
            "intervention_label": intervention["label"],
            "action_type":        intervention["action_type"],
            "failed_agent":       agent_name,
        }
    )

    return state
```

### `_apply_intervention(state, intervention, agent_name) -> ProfessorState`

For `AUTO` interventions — modifies state before re-entry. For `MANUAL` interventions — returns state unchanged (engineer has already fixed the external issue).

```python
AUTO_INTERVENTION_EFFECTS = {
    # data_quality
    "Skip validation, proceed with raw features.":
        lambda s: {**s, "skip_data_validation": True},
    "Drop columns with > 30% nulls, fill rest with median.":
        lambda s: {**s, "null_threshold": 0.30, "impute_strategy": "median"},
    # model_failure
    "Reduce LightGBM to 100 trees, disable early stopping.":
        lambda s: {**s, "lgbm_override": {"n_estimators": 100, "early_stopping_rounds": None}},
    "Switch to LogisticRegression fallback model.":
        lambda s: {**s, "model_fallback": "logistic_regression"},
    # memory
    "Sample training data to 50% and retry.":
        lambda s: {**s, "data_sample_fraction": 0.50},
    "Switch to n_jobs=1, reduce n_estimators to 200.":
        lambda s: {**s, "lgbm_override": {"n_jobs": 1, "n_estimators": 200}},
    # api_timeout
    "Retry with 2x timeout and exponential backoff.":
        lambda s: {**s, "api_timeout_multiplier": 2.0, "api_backoff_enabled": True},
    "Switch to local LLM (ollama/llama3) for this session.":
        lambda s: {**s, "llm_provider": "local"},
    # unknown
    "Retry this agent with extra debug logging.":
        lambda s: {**s, "debug_logging": True},
    "Skip this agent and continue with defaults.":
        lambda s: {**s, f"skip_{agent_name}": True},
}
# MANUAL interventions: state unchanged — engineer fixed the issue externally
```

### Wire into existing `handle_escalation()`

In the `EscalationLevel.HITL` branch of `handle_escalation()`, after `_checkpoint_state_to_redis()`:

```python
# Add these two lines:
prompt = generate_hitl_prompt(state, agent_name, error)
# Store prompt in state for downstream reads:
return {
    **state,
    "hitl_required":      True,
    "hitl_prompt":        prompt,
    "hitl_checkpoint_key": prompt["checkpoint_key"],
    "pipeline_halted":    True,
    "hitl_message": (     # replaces the old stub message
        f"Pipeline paused after {agent_name} failed 3 times. "
        f"See outputs/{state['session_id']}/hitl_prompt.json for interventions. "
        f"Resume: professor resume --session {state['session_id']}"
    ),
}
```

### New ProfessorState fields (Day 12 additions)

```python
hitl_prompt: dict              # full prompt dict from generate_hitl_prompt()
hitl_checkpoint_key: str       # Redis key
hitl_intervention_id: int      # set on resume
hitl_intervention_label: str   # set on resume
skip_data_validation: bool     # intervention 1 for data_quality
null_threshold: float          # intervention 2 for data_quality
impute_strategy: str           # intervention 2 for data_quality
lgbm_override: dict            # model override params
model_fallback: str            # "logistic_regression"
data_sample_fraction: float    # 0.0-1.0, default 1.0
api_timeout_multiplier: float  # 1.0 = no change
api_backoff_enabled: bool
llm_provider: str              # "groq" | "local"
debug_logging: bool
memory_peak_gb: float          # set by ml_optimizer
memory_oom_risk: bool          # set by ml_optimizer
```

---

## TASK 2 — Fix OOM: `agents/ml_optimizer.py`

**The bug:** Each Optuna trial creates 5 fold models. Without explicit cleanup:
trial 1 = ~200MB, trial 50 = ~4GB, trial 80 = OOM kill with no traceback.

### Changes to `_objective()` — the Optuna trial function

```python
import gc
import psutil
import optuna

def _objective(trial: optuna.Trial, X, y, cv_folds, params, max_memory_gb: float = 6.0) -> float:
    models = []
    oof_scores = []
    
    try:
        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, ...)
            
            score = _evaluate(model, X_val, y_val)
            oof_scores.append(score)
            models.append(model)
            
            # ── Memory check after each fold, not just after the trial ──
            rss_gb = psutil.Process().memory_info().rss / 1e9
            if rss_gb > max_memory_gb:
                logger.warning(
                    f"[ml_optimizer] Trial {trial.number} fold {fold_idx}: "
                    f"RSS={rss_gb:.2f}GB exceeds limit {max_memory_gb}GB. "
                    f"Pruning trial to prevent OOM."
                )
                trial.set_user_attr("oom_risk", True)
                trial.set_user_attr("oom_at_fold", fold_idx)
                trial.set_user_attr("oom_rss_gb", round(rss_gb, 2))
                raise optuna.TrialPruned(f"Memory limit exceeded: {rss_gb:.2f}GB > {max_memory_gb}GB")
        
        return float(np.mean(oof_scores))
    
    finally:
        # ── Always runs — whether trial completed, pruned, or raised ──
        for model in models:
            del model
        del models
        gc.collect()
```

### Memory monitoring on the study

```python
def run_optimization(X, y, n_trials=200, max_memory_gb=6.0, n_jobs=1) -> optuna.Study:
    """
    n_jobs=1 is the default and should not be overridden on 8GB RAM.
    n_jobs=-1 means each worker holds its own model copy — instant OOM.
    """
    study = optuna.create_study(direction="maximize")
    
    # Callback: log memory stats after each trial
    def memory_callback(study, trial):
        rss_gb = psutil.Process().memory_info().rss / 1e9
        if trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"[ml_optimizer] Trial {trial.number} PRUNED (OOM). RSS={rss_gb:.2f}GB")
        else:
            logger.debug(f"[ml_optimizer] Trial {trial.number} complete. RSS={rss_gb:.2f}GB")
    
    study.optimize(
        lambda trial: _objective(trial, X, y, cv_folds, params, max_memory_gb),
        n_trials=n_trials,
        n_jobs=n_jobs,          # never change this default on 8GB
        callbacks=[memory_callback],
        gc_after_trial=True,    # Optuna's own GC flag — belt AND braces
    )
    
    return study
```

### Wire peak memory into state

After `study.optimize()` completes, update state with memory stats:

```python
peak_rss = max(
    t.user_attrs.get("oom_rss_gb", 0)
    for t in study.trials
    if t.user_attrs
) or psutil.Process().memory_info().rss / 1e9

state = {
    **state,
    "memory_peak_gb": round(peak_rss, 2),
    "memory_oom_risk": any(
        t.user_attrs.get("oom_risk") for t in study.trials
    ),
    "optuna_pruned_trials": sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ),
}
```

### Add `max_memory_gb` to `ProfessorConfig`

```python
# config.py or constants.py
MAX_MEMORY_GB: float = float(os.getenv("PROFESSOR_MAX_MEMORY_GB", "6.0"))
```

---

## TASK 3 — Fix Cost: LangSmith Tracing in `core/professor.py`

**The bug:** 200 Optuna trials × full state serialised per LangSmith node × 3000 tokens = $15–40 per overnight run. The trial function is inside the LangGraph node — every trial is traced.

### Three changes

**Change A — Disable tracing inside Optuna loop**

Wrap `study.optimize()` in a context manager:

```python
import contextlib

@contextlib.contextmanager
def _disable_langsmith_tracing():
    """Temporarily disables LangSmith tracing. Restores original value on exit, even on exception."""
    original = os.environ.get("LANGCHAIN_TRACING_V2", "false")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        yield
    finally:
        os.environ["LANGCHAIN_TRACING_V2"] = original

# Inside ml_optimizer node:
with _disable_langsmith_tracing():
    study = run_optimization(X, y, n_trials=n_trials, max_memory_gb=max_memory_gb)
```

**Change B — Sampling rate for outer pipeline**

Only trace 10% of outer pipeline runs (router, data_engineer, submit). Not every run needs full traces.

```python
# In professor.py startup:
os.environ.setdefault("LANGCHAIN_TRACING_SAMPLING_RATE", os.getenv("LANGCHAIN_TRACING_SAMPLING_RATE", "0.10"))
```

**Change C — Cost estimation logging**

After the full pipeline run completes, log estimated token cost so the engineer can track spend:

```python
def _log_estimated_cost(state: ProfessorState) -> None:
    """
    Rough cost estimate. Logged at end of each run.
    Formula: outer_nodes * avg_tokens_per_node * cost_per_1k_tokens
    """
    OUTER_NODES = ["semantic_router", "competition_intel", "data_engineer", "eda_agent",
                   "validation_architect", "feature_factory", "red_team_critic",
                   "ensemble_architect", "submission_strategist"]
    AVG_TOKENS_PER_NODE = 3000
    COST_PER_1K = 0.003  # claude-sonnet approximate

    estimated_cost = len(OUTER_NODES) * AVG_TOKENS_PER_NODE * COST_PER_1K / 1000
    logger.info(
        f"[professor] Estimated LLM cost this run: ${estimated_cost:.4f} "
        f"(outer pipeline only, Optuna tracing disabled). "
        f"Adjust LANGCHAIN_TRACING_SAMPLING_RATE in .env to change trace coverage."
    )
```

### `.env.example` additions

Add these lines with comments:

```bash
# LangSmith tracing
# WARNING: Do NOT set LANGCHAIN_TRACING_SAMPLING_RATE to 1.0 with Optuna.
# 200 trials × 3000 tokens × $0.003/1k = $1.80/run in tracing costs alone.
# 0.1 = trace 10% of outer pipeline runs. Optuna loop is always untraced.
LANGCHAIN_TRACING_SAMPLING_RATE=0.10
LANGCHAIN_TRACING_V2=true

# Memory guardrails
# Set lower if running on 8GB RAM. 6.0 leaves ~2GB headroom for OS + imports.
PROFESSOR_MAX_MEMORY_GB=6.0
```

---

## TASK 4 — Tests: `tests/test_day12_quality.py`

See `DAY_12_TESTS.md` for the full test specification.

---

## INTEGRATION CHECKLIST

- [ ] `generate_hitl_prompt()` called inside `handle_escalation()` at HITL level — not at MACRO or TRIAGE
- [ ] `resume_from_checkpoint()` wired to `professor resume --session <id>` CLI command
- [ ] `hitl_prompt.json` written to `outputs/{session_id}/hitl_prompt.json`
- [ ] `_apply_intervention()` only modifies state for `AUTO` interventions — `MANUAL` returns unchanged state
- [ ] `max_memory_gb` read from env var `PROFESSOR_MAX_MEMORY_GB`, default 6.0
- [ ] `gc_after_trial=True` set in `study.optimize()`
- [ ] `n_jobs=1` is the default in `run_optimization()` — NOT `-1`
- [ ] `_disable_langsmith_tracing()` is a context manager using `try/finally` — tracing always restored
- [ ] `.env.example` updated with both new vars and their cost/memory risk comments
- [ ] Existing Day 9 contract tests still pass — no changes to `get_escalation_level()`, `EscalationLevel`, or `handle_escalation()` signature

## NEW STATE FIELDS — Add to `core/state.py`

```python
# HITL human layer (Day 12)
hitl_prompt: dict              # full prompt dict; empty dict by default
hitl_checkpoint_key: str       # Redis key; "" by default
hitl_intervention_id: int      # set on resume; 0 = not yet resumed
hitl_intervention_label: str   # "" by default

# Intervention effects (Day 12)
skip_data_validation: bool     # False by default
null_threshold: float          # 1.0 by default (no threshold = keep all cols)
impute_strategy: str           # "default"
lgbm_override: dict            # {} by default
model_fallback: str            # "" by default (no fallback)
data_sample_fraction: float    # 1.0 by default (no sampling)
api_timeout_multiplier: float  # 1.0 by default
api_backoff_enabled: bool      # False by default
llm_provider: str              # "groq" by default
debug_logging: bool            # False by default

# Memory monitoring (Day 12)
memory_peak_gb: float          # 0.0 by default
memory_oom_risk: bool          # False by default
optuna_pruned_trials: int      # 0 by default
```

## GIT COMMIT MESSAGE

```
Day 12: HITL human layer + OOM guardrails + tracing cost control

- circuit_breaker: generate_hitl_prompt() — structured 3-intervention UI
- circuit_breaker: resume_from_checkpoint() — Redis → state → re-entry
- circuit_breaker: _apply_intervention() — AUTO effects on state
- ml_optimizer: memory guardrail — psutil RSS check per fold, TrialPruned on OOM
- ml_optimizer: gc_after_trial=True, del models in finally block
- professor: _disable_langsmith_tracing() context manager around Optuna
- professor: LANGCHAIN_TRACING_SAMPLING_RATE=0.10 default
- .env.example: cost risk comments for both new vars
- tests/test_day12_quality.py: 52 adversarial tests — all green
```