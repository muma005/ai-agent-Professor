# Professor Agent — Day 13 Implementation
**Theme: Submission integrity — silent bugs that kill your LB score after correct CV**

Build order: Task 1 → Task 2 → Task 3 → Task 4

```
Task 1  →  FIX: feature_order saved at train time, enforced at submit time
           agents/ml_optimizer.py + tools/submit_tools.py
Task 2  →  FIX: data_hash validation before ensemble blend
           agents/ensemble_architect.py
Task 3  →  Build tools/wilcoxon_gate.py
Task 4  →  Plug Wilcoxon gate into ml_optimizer model selection
           agents/ml_optimizer.py
           commit: "Day 13: submission integrity — column order, hash guard, Wilcoxon gate"
```

---

## TASK 1 — FIX: Train/test column misalignment

**The bug:** If `test.csv` column order differs from `train.csv` (common in Kaggle when Polars reads CSVs with differing internal ordering), `X_test` columns are misaligned with what the model expects. Predictions are silently wrong. No error because shapes match. LightGBM silently uses the wrong features.

**Files:** `agents/ml_optimizer.py` + `tools/submit_tools.py`

### Change 1 — `agents/ml_optimizer.py`: save `feature_order` to `metrics.json`

At the end of training, after the model is fitted and metrics computed, append `feature_order` to the metrics dict before writing:

```python
# After computing oof_scores, before writing metrics.json:
feature_order = list(X_train.columns)   # Polars DataFrame — preserves insertion order

metrics = {
    "cv_mean":      float(np.mean(oof_scores)),
    "cv_std":       float(np.std(oof_scores)),
    "cv_folds":     cv_folds_n,
    "model_type":   model_type,
    "n_features":   len(feature_order),
    "feature_order": feature_order,       # ← NEW: exact ordered list
    "trained_at":   datetime.utcnow().isoformat(),
    "data_hash":    state["data_hash"],
}

metrics_path = Path(f"outputs/{state['session_id']}/metrics.json")
metrics_path.write_text(json.dumps(metrics, indent=2))
```

Also store in state:
```python
state = {**state, "feature_order": feature_order}
```

New `ProfessorState` field: `feature_order: list[str]  # [] by default`

### Change 2 — `tools/submit_tools.py`: enforce `feature_order` at prediction time

```python
def build_submission(state: ProfessorState, test_df: pl.DataFrame) -> pl.DataFrame:
    """
    Builds submission DataFrame from test predictions.
    Enforces feature_order from training — raises immediately on mismatch.
    """
    # 1. Load feature_order from metrics.json (source of truth)
    metrics_path = Path(f"outputs/{state['session_id']}/metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics.json not found at {metrics_path}. "
            "Cannot build submission without verified feature_order."
        )

    metrics = json.loads(metrics_path.read_text())
    feature_order = metrics.get("feature_order")
    if not feature_order:
        raise ValueError(
            "feature_order missing from metrics.json. "
            "Re-run ml_optimizer to regenerate metrics with column order."
        )

    # 2. Enforce exact column order — raises ColumnNotFoundError if missing
    try:
        test_subset = test_df.select(feature_order)   # Polars raises if column absent
    except pl.ColumnNotFoundError as e:
        raise ValueError(
            f"Test data is missing columns required by the trained model: {e}. "
            "Check that test.csv matches the training feature set."
        ) from e

    # 3. Hard assertion — belt AND braces
    assert list(test_subset.columns) == feature_order, (
        f"Column order mismatch after .select(). "
        f"Expected: {feature_order[:5]}... "
        f"Got: {list(test_subset.columns)[:5]}..."
    )

    # 4. Convert to numpy — now safe
    X_test = test_subset.to_numpy()

    # 5. Generate predictions via model_registry
    predictions = _generate_predictions(state, X_test)
    return _build_submission_df(state, test_df, predictions)
```

### Contract addition

Add to existing contract test or new contract file:
```python
# tests/contracts/test_submit_column_order_contract.py
def test_feature_order_saved_in_metrics_json():
    """feature_order must be a non-empty list in metrics.json after training."""
    ...

def test_submit_raises_on_missing_column():
    """submit_tools raises ValueError (not silent wrong prediction) when test column absent."""
    ...

def test_submit_raises_on_wrong_column_order():
    """submit_tools raises AssertionError when column order doesn't match feature_order."""
    ...
```

---

## TASK 2 — FIX: `data_hash` validation in `ensemble_architect.py`

**The bug:** Kaggle releases corrected data mid-competition. `model_registry` may contain models trained on old data (hash A) and new data (hash B). Ensemble blends them. The submission is partially trained on bad data. Predictions systematically wrong. No error.

**Note:** `data_hash` is already stored per `model_registry` entry (built Day 4). This is purely a validation check — no new data structures needed.

### Change — `agents/ensemble_architect.py`

Add at the top of the blending logic, before any model weights are computed:

```python
def _validate_data_hash_consistency(state: ProfessorState) -> None:
    """
    Ensures all models in registry were trained on the same data version.
    Raises ValueError if models are mixed across data versions.
    Logs WARNING if any filtering occurs.
    """
    registry = state.get("model_registry", {})
    if not registry:
        raise ValueError("model_registry is empty — no models to ensemble.")

    current_hash = state.get("data_hash")
    if not current_hash:
        logger.warning(
            "[ensemble_architect] state['data_hash'] is None. "
            "Cannot verify data version consistency. Proceeding without check."
        )
        return  # degrade gracefully if hash tracking wasn't set up

    # Extract hash from every registry entry
    hashes = {
        name: entry.get("data_hash")
        for name, entry in registry.items()
    }

    unique_hashes = set(h for h in hashes.values() if h is not None)

    if len(unique_hashes) > 1:
        logger.warning(
            f"[ensemble_architect] DATA VERSION MISMATCH DETECTED. "
            f"Registry contains models trained on {len(unique_hashes)} different data versions: "
            f"{unique_hashes}. "
            f"Filtering to only models matching current data_hash={current_hash}."
        )
        # Filter to models matching current data version only
        filtered_registry = {
            name: entry
            for name, entry in registry.items()
            if entry.get("data_hash") == current_hash
        }

        if not filtered_registry:
            raise ValueError(
                f"No models in registry match current data_hash={current_hash}. "
                f"All {len(registry)} models were trained on stale data versions. "
                f"Retrain required: run ml_optimizer from the beginning."
            )

        logger.info(
            f"[ensemble_architect] Filtered registry: "
            f"{len(filtered_registry)}/{len(registry)} models retained "
            f"(matching data_hash={current_hash})."
        )

        # Update state with filtered registry for downstream use
        state = {**state, "model_registry": filtered_registry}

    # All hashes match current — clean path
    log_event(
        state=state,
        action="data_hash_validated",
        agent="ensemble_architect",
        details={
            "data_hash": current_hash,
            "models_checked": len(registry),
            "models_retained": len(state["model_registry"]),
        }
    )
    return state


# Call at the START of blend_models(), before weight computation:
def blend_models(state: ProfessorState) -> ProfessorState:
    state = _validate_data_hash_consistency(state)
    # ... rest of blending logic
```

### New `ProfessorState` fields (none — uses existing `data_hash` and `model_registry`)

No new state fields required. The filter result updates `model_registry` in place within the function.

---

## TASK 3 — Build `tools/wilcoxon_gate.py`

**What it does:** Provides a statistically rigorous test for whether model A genuinely outperforms model B across CV folds, or whether the difference is within noise. Prevents selecting a more complex model on a lucky random seed.

**Why Wilcoxon and not t-test:** CV fold scores are not normally distributed and not independent. Wilcoxon signed-rank test is non-parametric — no normality assumption — and works on paired fold-level differences. Standard in ML model comparison literature.

```python
# tools/wilcoxon_gate.py

import logging
from typing import Optional
import numpy as np
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)

MIN_FOLDS_REQUIRED = 5   # Wilcoxon unreliable below 5 pairs
P_VALUE_THRESHOLD = 0.05


def is_significantly_better(
    fold_scores_a: list[float],
    fold_scores_b: list[float],
    p_threshold: float = P_VALUE_THRESHOLD,
    alternative: str = "greater",   # "greater" = test if A > B
) -> bool:
    """
    Returns True iff fold_scores_a is statistically significantly better
    than fold_scores_b at p < p_threshold using the Wilcoxon signed-rank test.

    Args:
        fold_scores_a: Per-fold scores for model A (candidate/challenger)
        fold_scores_b: Per-fold scores for model B (baseline/champion)
        p_threshold:   Significance threshold (default 0.05)
        alternative:   "greater" to test A > B, "less" to test A < B

    Returns:
        True  — A is significantly better than B (safe to select A)
        False — difference not significant (keep B, the simpler/existing model)

    Never raises — returns False on any error (conservative default).
    """
    if len(fold_scores_a) != len(fold_scores_b):
        logger.warning(
            f"[WilcoxonGate] fold count mismatch: "
            f"len(a)={len(fold_scores_a)}, len(b)={len(fold_scores_b)}. "
            f"Cannot compare — returning False (keep existing model)."
        )
        return False

    if len(fold_scores_a) < MIN_FOLDS_REQUIRED:
        logger.warning(
            f"[WilcoxonGate] Only {len(fold_scores_a)} folds — "
            f"minimum {MIN_FOLDS_REQUIRED} required for reliable Wilcoxon test. "
            f"Falling back to mean comparison."
        )
        return float(np.mean(fold_scores_a)) > float(np.mean(fold_scores_b))

    differences = np.array(fold_scores_a) - np.array(fold_scores_b)

    # If all differences are zero, models are identical
    if np.all(differences == 0):
        logger.info("[WilcoxonGate] All fold differences are zero — models identical.")
        return False

    try:
        stat, p_value = wilcoxon(differences, alternative=alternative, zero_method="wilcox")
    except Exception as e:
        logger.warning(
            f"[WilcoxonGate] scipy.stats.wilcoxon raised: {e}. "
            f"Falling back to mean comparison."
        )
        return float(np.mean(fold_scores_a)) > float(np.mean(fold_scores_b))

    result = p_value < p_threshold
    logger.info(
        f"[WilcoxonGate] stat={stat:.4f}, p={p_value:.4f}, "
        f"threshold={p_threshold}, significant={result}. "
        f"mean_a={np.mean(fold_scores_a):.5f}, mean_b={np.mean(fold_scores_b):.5f}, "
        f"delta={np.mean(differences):+.5f}"
    )
    return result


def gate_result(
    fold_scores_a: list[float],
    fold_scores_b: list[float],
    model_name_a: str = "challenger",
    model_name_b: str = "champion",
    p_threshold: float = P_VALUE_THRESHOLD,
) -> dict:
    """
    Returns a structured gate result dict — used for lineage logging.
    """
    significant = is_significantly_better(fold_scores_a, fold_scores_b, p_threshold)
    differences = np.array(fold_scores_a) - np.array(fold_scores_b)

    return {
        "gate_passed":    significant,
        "selected_model": model_name_a if significant else model_name_b,
        "mean_a":         round(float(np.mean(fold_scores_a)), 6),
        "mean_b":         round(float(np.mean(fold_scores_b)), 6),
        "mean_delta":     round(float(np.mean(differences)), 6),
        "p_threshold":    p_threshold,
        "n_folds":        len(fold_scores_a),
        "model_name_a":   model_name_a,
        "model_name_b":   model_name_b,
        "reason": (
            f"{model_name_a} significantly better (p<{p_threshold})"
            if significant else
            f"Difference not significant — keeping {model_name_b}"
        ),
    }
```

---

## TASK 4 — Plug Wilcoxon gate into `agents/ml_optimizer.py`

**Where it applies:** Every model comparison decision in the optimizer. Two specific places:

### 4a. Optuna trial selection — is the best trial genuinely better than the previous best?

In the Optuna study callback or after `study.optimize()`:

```python
from tools.wilcoxon_gate import gate_result, is_significantly_better
from core.lineage import log_event

def _select_best_trial_with_gate(
    study: optuna.Study,
    state: ProfessorState,
    previous_best_scores: Optional[list[float]] = None,
) -> optuna.Trial:
    """
    Selects the best Optuna trial, but only accepts it over the previous best
    if Wilcoxon confirms the improvement is significant.
    """
    candidate = study.best_trial
    candidate_scores = candidate.user_attrs.get("fold_scores", [])

    if not previous_best_scores or not candidate_scores:
        logger.info(
            "[ml_optimizer] No previous best or fold scores unavailable — "
            "accepting study best trial without gate."
        )
        return candidate

    result = gate_result(
        fold_scores_a=candidate_scores,
        fold_scores_b=previous_best_scores,
        model_name_a=f"trial_{candidate.number}",
        model_name_b="previous_best",
    )

    log_event(
        state=state,
        action="wilcoxon_gate_decision",
        agent="ml_optimizer",
        details=result,
    )

    if result["gate_passed"]:
        logger.info(
            f"[ml_optimizer] Wilcoxon gate PASSED — "
            f"accepting trial_{candidate.number} "
            f"(delta={result['mean_delta']:+.5f}, p<0.05)"
        )
        return candidate
    else:
        logger.info(
            f"[ml_optimizer] Wilcoxon gate FAILED — "
            f"keeping previous best. "
            f"trial_{candidate.number} not significantly better "
            f"(delta={result['mean_delta']:+.5f})"
        )
        return None   # caller keeps existing best
```

Store fold scores in each trial's user attrs during the objective function:
```python
# In _objective(), after computing oof_scores:
trial.set_user_attr("fold_scores", [float(s) for s in oof_scores])
trial.set_user_attr("mean_cv", float(np.mean(oof_scores)))
```

### 4b. Cross-model selection — LightGBM vs XGBoost vs CatBoost

After each model type's Optuna study completes, compare the winners:

```python
def _select_best_model_type(
    model_results: dict[str, dict],   # {"lgbm": {"fold_scores": [...], "study": ...}, ...}
    state: ProfessorState,
) -> str:
    """
    Selects the best model type using pairwise Wilcoxon gates.
    Returns the name of the significantly best model, or the simplest
    model if no significant differences found.

    Comparison order (complexity ascending): lgbm → xgb → catboost
    A more complex model must beat the simpler one to be selected.
    """
    MODEL_COMPLEXITY_ORDER = ["lgbm", "xgb", "catboost"]   # simplest first
    available = [m for m in MODEL_COMPLEXITY_ORDER if m in model_results]

    if not available:
        raise ValueError("No model results to compare.")

    champion = available[0]
    champion_scores = model_results[champion]["fold_scores"]

    for challenger in available[1:]:
        challenger_scores = model_results[challenger]["fold_scores"]

        result = gate_result(
            fold_scores_a=challenger_scores,
            fold_scores_b=champion_scores,
            model_name_a=challenger,
            model_name_b=champion,
        )

        log_event(
            state=state,
            action="wilcoxon_gate_decision",
            agent="ml_optimizer",
            details={**result, "comparison_type": "cross_model"},
        )

        if result["gate_passed"]:
            champion = challenger
            champion_scores = challenger_scores
            logger.info(
                f"[ml_optimizer] {challenger} significantly beats {result['model_name_b']} "
                f"— new champion."
            )
        else:
            logger.info(
                f"[ml_optimizer] {challenger} does NOT significantly beat {champion} "
                f"— keeping simpler model."
            )

    return champion
```

### Lineage format for all gate decisions

Every `wilcoxon_gate_decision` event in `lineage.jsonl` must include:
```json
{
  "action": "wilcoxon_gate_decision",
  "agent": "ml_optimizer",
  "gate_passed": true,
  "selected_model": "xgb",
  "model_name_a": "xgb",
  "model_name_b": "lgbm",
  "mean_a": 0.87234,
  "mean_b": 0.86891,
  "mean_delta": +0.00343,
  "p_threshold": 0.05,
  "n_folds": 5,
  "comparison_type": "cross_model",
  "reason": "xgb significantly better (p<0.05)"
}
```

---

## INTEGRATION CHECKLIST

- [ ] `feature_order` written to `metrics.json` at end of every training run
- [ ] `feature_order` stored in `state["feature_order"]`
- [ ] `build_submission()` loads `feature_order` from `metrics.json` (not from state — state can be stale)
- [ ] `test_df.select(feature_order)` called before `.to_numpy()` — always
- [ ] Hard `assert list(test_subset.columns) == feature_order` before numpy conversion
- [ ] `_validate_data_hash_consistency()` called at START of `blend_models()` before any weight computation
- [ ] Mixed-hash scenario logs WARNING and filters — does not silently proceed
- [ ] Empty filtered registry raises `ValueError` (not a warning)
- [ ] `wilcoxon_gate.py` — `is_significantly_better()` never raises — always returns bool
- [ ] Fold scores stored in `trial.user_attrs["fold_scores"]` inside `_objective()`
- [ ] Every model comparison in optimizer goes through `gate_result()`
- [ ] Every gate decision logged to lineage with full result dict
- [ ] Day 12 OOM guardrails (`del models` in `finally`) still in place — this task doesn't regress them

## NEW STATE FIELDS

```python
feature_order: list[str]   # [] by default — set by ml_optimizer after training
```

## GIT COMMIT MESSAGE

```
Day 13: submission integrity — column order, hash guard, Wilcoxon gate

- ml_optimizer: feature_order saved to metrics.json at training time
- submit_tools: test_df.select(feature_order) + assert before to_numpy()
- ensemble_architect: data_hash validation before blend, ValueError on stale models
- tools/wilcoxon_gate.py: is_significantly_better() — scipy Wilcoxon, never raises
- ml_optimizer: gate applied to Optuna trial selection + cross-model comparison
- lineage: every gate decision logged with full statistical result
- tests/test_day13_quality.py: 48 adversarial tests — all green
```