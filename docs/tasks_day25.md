# Day 25 — Pseudo-labeling, Session Isolation, Time-Series Routing
## Implementation Prompt for Qwen Code

---

## BEFORE YOU WRITE A SINGLE LINE

Read these files completely first:

```
CLAUDE.md
AGENTS.md
core/state.py
core/professor.py
agents/ml_optimizer.py
agents/ensemble_architect.py
agents/feature_factory.py
agents/validation_architect.py
memory/redis_state.py
```

After reading, answer before writing:
1. What is the current format of `session_id` in state? Where is it set?
2. What Redis key format do existing agents use? Quote one example.
3. What is `metric_contract.scorer` and where is it set in state?
4. Does `validation_architect.py` currently use `TimeSeriesSplit`? Quote the relevant line.
5. What does `feature_factory.py` currently do for lag features?

Do not proceed until you have answered all five from the actual code.

---

## TASK 1 — GAP 9: Pseudo-labeling agent (`agents/pseudo_label_agent.py`)

**Important:** This agent was partially specced in Day 18. Before writing anything, check whether `agents/pseudo_label_agent.py` already exists. If it does, read it completely and extend it rather than overwriting it.

### Three activation gates — ALL must pass before pseudo-labeling runs

```python
PROBABILITY_METRICS     = frozenset({"log_loss", "logloss", "cross_entropy", "brier_score", "auc"})
MIN_TEST_TO_TRAIN_RATIO = 2.0    # test set must have > 2x rows of train set
MIN_CALIBRATION_SCORE   = 0.80   # model calibration above this threshold required
HIGH_CONFIDENCE_THRESHOLD = 0.95  # probability threshold for pseudo-label selection
MAX_PSEUDO_LABEL_FRACTION = 0.30  # pseudo-labels never exceed 30% of training data
MAX_ITERATIONS          = 3
```

Gate check function:

```python
def _check_activation_gates(state: dict) -> tuple[bool, str]:
    """
    Returns (should_run, reason).
    ALL three gates must pass. Returns the first failing gate's reason.
    """
    metric = state.get("evaluation_metric", "")
    if metric not in PROBABILITY_METRICS:
        return False, f"metric '{metric}' is not probability-based"

    n_train = len(state.get("y_train", []))
    n_test  = state.get("n_test_rows", 0)
    if n_test == 0:
        n_test = _count_test_rows(state)
    if n_test <= n_train * MIN_TEST_TO_TRAIN_RATIO:
        return False, (
            f"test set ({n_test} rows) is not > {MIN_TEST_TO_TRAIN_RATIO}x "
            f"training set ({n_train} rows)"
        )

    calibration = _get_best_calibration_score(state)
    if calibration is None or calibration < MIN_CALIBRATION_SCORE:
        return False, (
            f"model calibration ({calibration}) below threshold {MIN_CALIBRATION_SCORE}"
        )

    return True, "all gates passed"
```

### The algorithm

```python
def run_pseudo_label_agent(state: dict) -> dict:
    should_run, reason = _check_activation_gates(state)
    if not should_run:
        logger.info(f"[pseudo_label] Skipped: {reason}")
        state["pseudo_labels_applied"]      = False
        state["pseudo_label_skip_reason"]   = reason
        state["pseudo_label_iterations"]    = 0
        return state

    # Load training data from disk — never from state (2GB DataFrames not in state)
    train_path = state.get("clean_train_path") or state.get("train_path")
    test_path  = state.get("clean_test_path")  or state.get("test_path")
    if not train_path or not test_path:
        logger.warning("[pseudo_label] No train/test path in state. Skipping.")
        state["pseudo_labels_applied"] = False
        return state

    X_train  = pl.read_csv(train_path)
    y_train  = X_train[state["target_column"]].to_numpy()
    X_train  = X_train.drop(state["target_column"])
    X_test   = pl.read_csv(test_path)

    feature_order = state.get("feature_order", [])
    if feature_order:
        available = [c for c in feature_order if c in X_train.columns]
        X_train   = X_train.select(available)
        X_test    = X_test.select([c for c in available if c in X_test.columns])
```

### Critic verification of pseudo-label confidence distribution

Before accepting any pseudo-labels, the critic must verify the confidence distribution is realistic:

```python
def _critic_verifies_confidence_distribution(
    confidences: np.ndarray,
    state: dict,
) -> tuple[bool, str]:
    """
    Critic check: pseudo-label confidence distribution must be realistic.

    Rejects if:
    1. > 50% of predictions are above HIGH_CONFIDENCE_THRESHOLD
       (model too overconfident — distribution collapse)
    2. Mean confidence < 0.55
       (model has no discriminative power at all)
    3. Std of confidences < 0.05
       (all predictions identical — constant predictor)

    Returns (accepted, reason).
    """
    high_conf_fraction = float(np.mean(confidences >= HIGH_CONFIDENCE_THRESHOLD))
    mean_conf          = float(np.mean(confidences))
    std_conf           = float(np.std(confidences))

    if high_conf_fraction > 0.50:
        return False, (
            f"distribution collapse: {high_conf_fraction:.1%} of predictions "
            f"above {HIGH_CONFIDENCE_THRESHOLD}. Model is overconfident."
        )
    if mean_conf < 0.55:
        return False, (
            f"mean confidence {mean_conf:.3f} too low. "
            "Model has insufficient discriminative power for pseudo-labeling."
        )
    if std_conf < 0.05:
        return False, (
            f"confidence std {std_conf:.4f} < 0.05. "
            "All predictions nearly identical — constant predictor."
        )

    return True, f"distribution OK (mean={mean_conf:.3f}, std={std_conf:.4f}, high_conf={high_conf_fraction:.1%})"
```

### Validation fold integrity — the critical rule

Pseudo-labels are added to TRAINING folds only. Validation fold always contains only real labels:

```python
def _run_cv_with_pseudo_labels(
    X_train_real: np.ndarray,
    y_train_real: np.ndarray,
    X_pseudo: np.ndarray,
    y_pseudo: np.ndarray,
    params: dict,
    n_folds: int = 5,
    metric: str = "accuracy",
    random_state: int = 42,
) -> list[float]:
    """
    Runs CV where pseudo-labels are added to TRAINING FOLDS ONLY.
    Validation fold sees ONLY real labeled samples. Always.

    This is the guard that prevents pseudo-labeling from inflating CV.
    """
    from sklearn.model_selection import StratifiedKFold, KFold

    is_classification = metric in ("accuracy", "auc", "logloss", "log_loss")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state) \
         if is_classification else \
         KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    ModelClass = _get_model_class(params.get("model_type", "lgbm"))
    fold_scores = []

    for train_idx, val_idx in cv.split(X_train_real, y_train_real if is_classification else None):
        # Training: real labels + pseudo-labels
        X_fold_train = np.vstack([X_train_real[train_idx], X_pseudo])
        y_fold_train = np.concatenate([y_train_real[train_idx], y_pseudo])

        # Validation: ONLY real labels
        X_fold_val = X_train_real[val_idx]
        y_fold_val = y_train_real[val_idx]

        model = ModelClass(**_strip_meta(params))
        model.fit(X_fold_train, y_fold_train)
        score = _score_model(model, X_fold_val, y_fold_val, metric)
        fold_scores.append(float(score))

        del model
        import gc; gc.collect()

    return fold_scores
```

### Wilcoxon gate before accepting each iteration

After computing CV with pseudo-labels, apply Wilcoxon gate:

```python
from tools.wilcoxon_gate import is_significantly_better

baseline_scores = best_model_entry.get("fold_scores", [])
pl_scores = _run_cv_with_pseudo_labels(...)

if not is_significantly_better(pl_scores, baseline_scores):
    logger.info(
        f"[pseudo_label] Iteration {iteration}: Wilcoxon gate failed. "
        "Pseudo-labels do not significantly improve CV. Stopping."
    )
    state["pseudo_label_halt_reason"] = "wilcoxon_gate_failed"
    break
```

### Max pseudo-label cap

Never let pseudo-labels exceed 30% of training data:

```python
max_pseudo = int(len(y_train_real) * MAX_PSEUDO_LABEL_FRACTION)
if len(X_pseudo_accumulated) >= max_pseudo:
    logger.info(
        f"[pseudo_label] Reached max pseudo-label fraction "
        f"({MAX_PSEUDO_LABEL_FRACTION:.0%} of training data). Stopping."
    )
    state["pseudo_label_halt_reason"] = "max_fraction_reached"
    break
```

### State outputs

```python
state["pseudo_labels_applied"]          # bool
state["pseudo_label_skip_reason"]       # str — "" if applied
state["pseudo_label_halt_reason"]       # str — reason for stopping early
state["pseudo_label_iterations"]        # int — iterations completed
state["pseudo_label_n_added"]           # int — total pseudo-labels added
state["pseudo_label_cv_improvement"]    # float — cv gain from pseudo-labeling
state["pseudo_label_confidence_mean"]   # float — mean confidence of selected samples
state["pseudo_label_confidence_std"]    # float — std of selected sample confidences
state["pseudo_label_critic_accepted"]   # bool — critic verified distribution
state["clean_train_with_pseudo_path"]   # str — path to augmented training CSV
```

### Never do

- Never read X_train as a DataFrame from state — it is too large
- Never add pseudo-labels to the validation fold under any circumstance
- Never run more than MAX_ITERATIONS = 3 iterations
- Never select pseudo-labels with confidence < HIGH_CONFIDENCE_THRESHOLD

---

## TASK 2 — GAP 8: Session ID namespace isolation (`core/state.py` + `core/professor.py`)

**The problem:** Without namespacing, three concurrent competition runs share Redis keys, ChromaDB namespaces, output directories, and MLflow experiments. Day 30 requires three concurrent runs — this must be fixed now.

### Session ID format

```python
import hashlib
from datetime import datetime

def generate_session_id(competition_name: str) -> str:
    """
    Generates a unique, namespaced session ID.
    Format: professor_{competition_slug}_{timestamp}_{short_hash}
    Example: professor_spaceship-titanic_20260301_142200_a3f9c2

    The short hash ensures uniqueness even if two runs start in the same second.
    """
    timestamp  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug       = competition_name.lower().replace(" ", "-").replace("_", "-")[:30]
    short_hash = hashlib.md5(f"{slug}{timestamp}".encode()).hexdigest()[:6]
    return f"professor_{slug}_{timestamp}_{short_hash}"
```

### All resources must use session_id as prefix

**Redis keys** — every key must start with `{session_id}:`:

```python
# In memory/redis_state.py:
def _key(session_id: str, suffix: str) -> str:
    return f"{session_id}:{suffix}"

# Examples:
redis.set(_key(session_id, "state"), ...)
redis.set(_key(session_id, "hitl"), ...)
redis.get(_key(session_id, "checkpoint"), ...)
```

**Output directory** — every file goes under `outputs/{session_id}/`:

```python
# Already correct if code uses state["session_id"] — verify it does
output_dir = Path(f"outputs/{state['session_id']}")
output_dir.mkdir(parents=True, exist_ok=True)
```

**ChromaDB namespace** — queries must filter by session_id in metadata where relevant. Competition-level patterns (professor_patterns_v2, professor_hpo_memories) are shared across sessions intentionally — do not namespace those. Session-specific data (HITL prompts, run logs) must be namespaced.

**MLflow experiment** — experiment name must include session_id:

```python
import mlflow
mlflow.set_experiment(f"professor_{state['session_id']}")
```

**Cost budget** — each session gets its own budget counter:

```python
# In core/state.py initial state:
"budget_limit_usd":    10.0,
"budget_spent_usd":    0.0,
"budget_session_id":   session_id,   # must match session_id to prevent cross-session sharing
```

### Changes to `core/professor.py`

```python
def run_professor(state: dict) -> dict:
    # Generate session_id if not already set
    if not state.get("session_id"):
        state["session_id"] = generate_session_id(
            state.get("competition_name", "unknown")
        )

    # Validate session_id format
    assert state["session_id"].startswith("professor_"), (
        f"session_id '{state['session_id']}' must start with 'professor_'. "
        "Use generate_session_id() to create session IDs."
    )

    # Create output directory for this session
    output_dir = Path(f"outputs/{state['session_id']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state["output_dir"] = str(output_dir)

    logger.info(f"[professor] Session: {state['session_id']}")
    logger.info(f"[professor] Output dir: {state['output_dir']}")

    return get_graph().invoke(state)
```

### Changes to `core/state.py`

Add these to the default initial state:

```python
def build_initial_state(competition_name: str, **kwargs) -> dict:
    """
    Factory function for building a valid initial ProfessorState.
    Always use this instead of building state dicts manually.
    """
    session_id = generate_session_id(competition_name)
    return {
        "competition_name":           competition_name,
        "session_id":                 session_id,
        "output_dir":                 f"outputs/{session_id}",
        "budget_session_id":          session_id,
        "budget_limit_usd":           10.0,
        "budget_spent_usd":           0.0,
        "dag_version":                1,
        "current_node_failure_count": 0,
        "hitl_required":              False,
        "replan_requested":           False,
        "critic_severity":            "unchecked",
        "model_registry":             {},
        "features_dropped":           [],
        "feature_order":              [],
        "external_data_allowed":      False,
        **kwargs,
    }
```

### What not to break

- Existing tests that build state manually must still work — `build_initial_state` is additive, not a replacement
- Redis fallback to fakeredis must still work when Docker Redis is unavailable
- ChromaDB shared collections (professor_patterns_v2, professor_hpo_memories) must not be namespaced — they are intentionally cross-session

---

## TASK 3 — Time-series routing (`agents/feature_factory.py` + `agents/validation_architect.py`)

### `agents/validation_architect.py` — enforce TimeSeriesSplit

Add at the start of the validation strategy selection:

```python
def _select_cv_strategy(state: dict) -> dict:
    task_type = state.get("task_type", "binary_classification")

    if task_type == "timeseries":
        # TimeSeriesSplit only — no random shuffle ever
        n_splits = state.get("cv_n_splits", 5)
        return {
            "cv_strategy":   "TimeSeriesSplit",
            "cv_class":      "sklearn.model_selection.TimeSeriesSplit",
            "cv_params":     {"n_splits": n_splits},
            "shuffle":       False,   # hard False — never shuffle time-series
            "stratify":      False,   # not applicable for time-series
            "rationale": (
                f"task_type=timeseries requires TimeSeriesSplit (n_splits={n_splits}). "
                "Random shuffle is prohibited — it causes temporal leakage."
            ),
        }

    # ... existing strategy selection for other task types unchanged
```

Add a guard that raises if someone attempts to use a random-shuffle CV on time-series data:

```python
def validate_cv_strategy(state: dict, cv_strategy: dict) -> None:
    """
    Raises ValueError if a random-shuffle CV is used with timeseries data.
    Called before any CV training begins.
    """
    if state.get("task_type") != "timeseries":
        return   # only enforced for time-series

    if cv_strategy.get("shuffle") is True:
        raise ValueError(
            "VALIDATION ERROR: shuffle=True is not allowed for task_type=timeseries. "
            "Temporal leakage detected. Use TimeSeriesSplit instead."
        )
    if cv_strategy.get("cv_strategy") in ("StratifiedKFold", "KFold"):
        raise ValueError(
            f"VALIDATION ERROR: {cv_strategy['cv_strategy']} is not allowed for timeseries. "
            "Use TimeSeriesSplit. Random fold splitting destroys temporal ordering."
        )
```

### `agents/feature_factory.py` — time-series feature routing

Add detection logic and route to a separate generation path when `task_type == "timeseries"`:

```python
def _generate_timeseries_features(
    schema: dict,
    competition_brief: dict,
) -> list[FeatureCandidate]:
    """
    Time-series specific features. Called instead of (not in addition to)
    Round 1 generic transforms when task_type == timeseries.

    Generates:
      - Lag features (lag_1, lag_2, lag_3, lag_7, lag_14, lag_28 for daily data)
      - Rolling statistics (rolling_mean_7, rolling_std_7, rolling_mean_28)
      - Seasonal decomposition indicators (day_of_week, month, quarter)
      - Trend features (days_since_start, row_index_normalised)

    All features are defined by name only — they are applied to actual data
    by _apply_timeseries_transforms() at transform time.
    """
    candidates = []
    columns    = schema.get("columns", [])

    # Find numeric columns that are candidates for lag/rolling features
    numeric_cols = [
        c["name"] for c in columns
        if _is_numeric(c) and not c.get("is_id") and not c.get("is_target")
    ]

    date_col = _find_date_column(schema)

    # Lag features — only generate for the most important numeric columns (top 5 by n_unique)
    top_numerics = sorted(
        numeric_cols,
        key=lambda n: next((c["n_unique"] for c in columns if c["name"] == n), 0),
        reverse=True
    )[:5]

    for col in top_numerics:
        for lag in [1, 2, 3, 7, 14, 28]:
            candidates.append(FeatureCandidate(
                name=f"{col}_lag_{lag}",
                source_columns=[col],
                transform_type="lag",
                description=f"{col} lagged by {lag} periods",
                round=1,
            ))

        for window in [7, 28]:
            candidates.append(FeatureCandidate(
                name=f"{col}_rolling_mean_{window}",
                source_columns=[col],
                transform_type="rolling_mean",
                description=f"Rolling mean of {col} over {window} periods",
                round=1,
            ))
            candidates.append(FeatureCandidate(
                name=f"{col}_rolling_std_{window}",
                source_columns=[col],
                transform_type="rolling_std",
                description=f"Rolling std of {col} over {window} periods",
                round=1,
            ))

    # Seasonal features from date column
    if date_col:
        for feat in ["day_of_week", "month", "quarter", "day_of_year", "week_of_year"]:
            candidates.append(FeatureCandidate(
                name=f"{feat}",
                source_columns=[date_col],
                transform_type=f"date_{feat}",
                description=f"{feat} extracted from {date_col}",
                round=1,
            ))

    logger.info(
        f"[feature_factory] Time-series mode: {len(candidates)} candidates generated "
        f"(lag, rolling, seasonal)."
    )
    return candidates
```

### Routing in `run_feature_factory()`

```python
def run_feature_factory(state: dict) -> dict:
    # ... existing setup code ...

    task_type = state.get("task_type", "binary_classification")

    if task_type == "timeseries":
        logger.info("[feature_factory] Time-series mode: using lag/rolling/seasonal features.")
        round1_candidates = _generate_timeseries_features(schema, competition_brief)
        # Skip Round 3 aggregations and Round 4 target encoding for time-series
        # Aggregations may use future data; target encoding requires careful time ordering
        round3_candidates = []
        round4_candidates = []
    else:
        round1_candidates = _generate_round1_features(schema)
        round3_candidates = _generate_round3_aggregation_features(schema)
        round4_candidates = _generate_round4_target_encoding_candidates(schema)

    # Rounds 2 and 5 run for all task types
    round2_candidates  = _generate_round2_features(schema, competition_brief, state)
    round5a_candidates = _generate_round5_hypothesis_features(schema, competition_brief, state)
    # ... rest unchanged
```

---

## COMMIT SEQUENCE

```
git commit -m "Day 25: agents/pseudo_label_agent.py — 3 activation gates, confidence gating, Wilcoxon gate, critic verification"
git commit -m "Day 25: core/state.py + core/professor.py — GAP 8 session_id namespace isolation"
git commit -m "Day 25: feature_factory.py + validation_architect.py — time-series routing, TimeSeriesSplit enforcement"
```

---

## VERIFICATION BEFORE EACH COMMIT

```bash
python -c "from agents.pseudo_label_agent import run_pseudo_label_agent; print('OK')"
python -c "from core.state import build_initial_state, generate_session_id; s = build_initial_state('test'); assert s['session_id'].startswith('professor_'); print('OK')"
python -c "from agents.feature_factory import run_feature_factory; print('OK')"
python -c "from agents.validation_architect import validate_cv_strategy; print('OK')"

pytest tests/contracts/ -v --tb=short
pytest tests/regression/ -v --tb=short
```

All must show zero failures.