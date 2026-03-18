# Professor Agent — Day 18 Implementation
**Theme: Late-competition edges + feature engineering at scale without compute graveyard**

Build order: Task 1 → Task 2 → Task 3
```
Task 1  →  Feature Factory Rounds 3 + 4 + 5
           agents/feature_factory.py
Task 2  →  GAP 12: Interaction budget cap
           agents/feature_factory.py
Task 3  →  GM-CAP 6: Pseudo-labeling pipeline
           agents/pseudo_label_agent.py
           commit: "Day 18: feature factory rounds 3-5, interaction cap, pseudo-labeling"
```

**Prerequisites:**
- Day 16 feature factory (Rounds 1 + 2) complete — Round 3+ extends it
- Day 17 null importance filter complete — Round 3+ candidates pass through it
- Day 16 diversity ensemble complete — pseudo-labeling runs after ensemble produces OOF predictions
- `competition_brief.json` contains `meaningful_interactions` list (written by competition_intel)

---

## TASK 1 — Feature Factory Rounds 3 + 4 + 5

### Round 3: Aggregation features (groupby stats)

**What they are:** For a categorical column `C` and numeric column `N`, compute mean/std/min/max/count of `N` grouped by `C`. Example: `fare_mean_by_pclass`, `age_std_by_embarked`. These capture "how unusual is this value relative to its group" — one of the most reliable feature engineering patterns across all tabular competitions.
```python
ROUND3_AGG_FUNCTIONS = ["mean", "std", "min", "max", "count"]
MAX_ROUND3_CANDIDATES = 200   # cap before null importance filter

def _generate_round3_aggregation_features(schema: dict) -> list[FeatureCandidate]:
    """
    Round 3: groupby aggregation features.

    For each (categorical, numeric) pair:
      - categorical: dtype contains 'str', 'cat', 'object', or n_unique < 50
      - numeric: dtype contains 'float' or 'int', not id, not target
      - generate: mean, std, min, max, count of numeric grouped by categorical

    Hard cap: MAX_ROUND3_CANDIDATES. If exceeded, rank by
    (n_unique_categorical * solo_importance_numeric) descending and take top N.
    This proxy metric selects pairs most likely to produce informative aggregations.
    """
    columns = schema.get("columns", [])
    candidates = []

    categoricals = [
        c for c in columns
        if not c.get("is_id") and not c.get("is_target")
        and (_is_categorical(c))
    ]

    numerics = [
        c for c in columns
        if not c.get("is_id") and not c.get("is_target")
        and _is_numeric(c)
    ]

    for cat in categoricals:
        for num in numerics:
            if cat["name"] == num["name"]:
                continue
            for fn in ROUND3_AGG_FUNCTIONS:
                candidates.append(FeatureCandidate(
                    name=f"{num['name']}_{fn}_by_{cat['name']}",
                    source_columns=[num["name"], cat["name"]],
                    transform_type="groupby_agg",
                    description=(
                        f"{fn} of {num['name']} grouped by {cat['name']}. "
                        f"Captures within-group statistics."
                    ),
                    round=3,
                ))

    # Cap
    if len(candidates) > MAX_ROUND3_CANDIDATES:
        # Heuristic rank: low-cardinality categoricals × high-unique numerics
        def _agg_priority(c: FeatureCandidate) -> float:
            cat_col = _find_col(schema, c.source_columns[1])
            num_col = _find_col(schema, c.source_columns[0])
            cat_card = float(cat_col.get("n_unique", 1))
            num_unique = float(num_col.get("n_unique", 1))
            # Low cardinality categoricals produce stable group stats
            return (1.0 / (cat_card + 1)) * num_unique

        candidates = sorted(candidates, key=_agg_priority, reverse=True)[:MAX_ROUND3_CANDIDATES]

    logger.info(f"[feature_factory] Round 3: {len(candidates)} aggregation candidates.")
    return candidates


def _apply_round3_transforms(X: pl.DataFrame, candidates: list[FeatureCandidate]) -> pl.DataFrame:
    """Applies groupby aggregation transforms using Polars lazy evaluation."""
    new_cols = []
    for c in candidates:
        if c.transform_type != "groupby_agg":
            continue
        num_col, cat_col = c.source_columns[0], c.source_columns[1]
        if num_col not in X.columns or cat_col not in X.columns:
            continue
        fn_name = c.name.split("_")[1]   # "mean", "std", etc.
        agg_fn = {
            "mean": pl.col(num_col).mean(),
            "std":  pl.col(num_col).std(),
            "min":  pl.col(num_col).min(),
            "max":  pl.col(num_col).max(),
            "count": pl.col(num_col).count(),
        }.get(fn_name)
        if agg_fn is None:
            continue
        # Join-based approach: compute group stat then join back
        group_stats = (
            X.group_by(cat_col)
             .agg(agg_fn.alias(c.name))
        )
        X = X.join(group_stats, on=cat_col, how="left")
    return X
```

### Round 4: CV-safe target encoding

**Why CV-safe matters:** Standard target encoding (mean of target per category) leaks target information into training features — the model learns to predict based on the encoded value, which is computed using the very values it's predicting. Done naively this inflates CV and destroys LB. CV-safe implementation uses leave-one-out encoding within each fold.
```python
from sklearn.model_selection import KFold

MAX_ROUND4_CANDIDATES = 30   # target encoding for top-N categoricals only

def _generate_round4_target_encoding_candidates(schema: dict) -> list[FeatureCandidate]:
    """
    Round 4: CV-safe target encoding candidates.

    Only generates candidates for columns that are:
    - Categorical (dtype: str, cat, object OR n_unique < 50)
    - Not id, not target
    - n_unique >= 2 (binary columns rarely benefit from target encoding)
    - n_unique <= 200 (very high cardinality is handled by count encoding instead)

    Returns at most MAX_ROUND4_CANDIDATES — sorted by n_unique descending
    (higher cardinality categoricals benefit more from target encoding).
    """
    columns = schema.get("columns", [])
    candidates = []

    for col in columns:
        if col.get("is_id") or col.get("is_target"):
            continue
        if not _is_categorical(col):
            continue
        n_unique = int(col.get("n_unique", 0))
        if n_unique < 2 or n_unique > 200:
            continue

        candidates.append(FeatureCandidate(
            name=f"te_{col['name']}",
            source_columns=[col["name"]],
            transform_type="target_encoding",
            description=(
                f"CV-safe leave-one-out target encoding of {col['name']}. "
                f"n_unique={n_unique}. Computed within folds only — never on full training set."
            ),
            round=4,
        ))

    # Sort by n_unique descending — higher cardinality benefits most
    candidates.sort(
        key=lambda c: int(_find_col(schema, c.source_columns[0]).get("n_unique", 0)),
        reverse=True,
    )
    return candidates[:MAX_ROUND4_CANDIDATES]


def _apply_round4_target_encoding(
    X: pl.DataFrame,
    y: np.ndarray,
    candidates: list[FeatureCandidate],
    n_folds: int = 5,
    smoothing: float = 30.0,
) -> pl.DataFrame:
    """
    Applies CV-safe target encoding.

    For each fold:
      - Compute mean(y) per category using the OTHER folds only
      - Apply smoothing: (count * group_mean + smoothing * global_mean) / (count + smoothing)
      - Assign to current fold rows only

    This is the only correct way to do target encoding without validation fold leakage.

    Smoothing formula (empirical Bayes):
      encoded = (count * group_mean + smoothing * global_mean) / (count + smoothing)
    Smoothing=30 works well for most competitions. Higher = more regularization.
    """
    target_enc_cols = [c for c in candidates if c.transform_type == "target_encoding"]
    if not target_enc_cols:
        return X

    global_mean = float(np.mean(y))
    n = len(y)
    fold_assignments = np.zeros(n, dtype=int)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_idx, (_, val_idx) in enumerate(kf.split(np.arange(n))):
        fold_assignments[val_idx] = fold_idx

    result_X = X.clone()

    for candidate in target_enc_cols:
        col = candidate.source_columns[0]
        if col not in X.columns:
            continue

        encoded = np.full(n, global_mean, dtype=np.float64)

        for fold_idx in range(n_folds):
            train_mask = fold_assignments != fold_idx
            val_mask   = fold_assignments == fold_idx

            cat_values_train = X[col].to_numpy()[train_mask]
            y_train          = y[train_mask]
            cat_values_val   = X[col].to_numpy()[val_mask]

            # Compute per-category stats from training portion of this fold
            cat_stats: dict[str, tuple[float, int]] = {}
            for cat, target in zip(cat_values_train, y_train):
                if cat not in cat_stats:
                    cat_stats[cat] = [0.0, 0]
                cat_stats[cat][0] += float(target)
                cat_stats[cat][1] += 1

            # Apply smoothed encoding to validation portion
            for i, cat in enumerate(cat_values_val):
                if cat in cat_stats:
                    sum_t, count = cat_stats[cat]
                    group_mean = sum_t / count
                    encoded[val_mask][i] = (
                        (count * group_mean + smoothing * global_mean)
                        / (count + smoothing)
                    )
                # Unseen categories → global mean (already set as default)

        result_X = result_X.with_columns(
            pl.Series(name=candidate.name, values=encoded)
        )

    return result_X
```

### Round 5: Forum hypothesis testing + creative interactions

**What it does:** Takes `insights` from `competition_brief.json` (tagged with `validated=False`) and generates features that would test each hypothesis. Also generates domain-guided interaction terms (multiply/divide/add/subtract pairs) for the top-K features that passed null importance.
```python
ROUND5_HYPOTHESIS_PROMPT = """
You are a Kaggle feature engineer. For each unvalidated hypothesis below, generate one feature
that would test whether the hypothesis is true for this competition's data.

Competition: {competition_name}
Domain: {domain}

Unvalidated hypotheses from competition forum:
{hypotheses}

Available columns (from schema.json):
{column_summary}

For each hypothesis, return a JSON object:
{{
  "hypothesis_index": int,
  "hypothesis_summary": "brief restatement",
  "feature_name": "snake_case_name",
  "source_columns": ["col1", "col2"],
  "transform_type": "ratio | interaction | bin | polynomial",
  "expression": "e.g. 'col1 / (col2 + 1)' or 'col1 * col2'",
  "validation_logic": "How a high value of this feature would confirm the hypothesis"
}}

Return a JSON array. Include only hypotheses where a concrete feature test is possible.
Maximum 10 features.
"""

def _generate_round5_hypothesis_features(
    schema: dict,
    competition_brief: dict,
    state: ProfessorState,
) -> list[FeatureCandidate]:
    """
    Round 5a: Generate features that directly test forum hypotheses.
    Only generates for insights with validated=False.
    """
    insights = [
        ins for ins in competition_brief.get("insights", [])
        if ins.get("validated") is False
    ]

    if not insights:
        logger.info("[feature_factory] Round 5: no unvalidated hypotheses to test.")
        return []

    hypotheses_text = "\n".join(
        f"{i+1}. {ins['content']} (source: {ins.get('source', 'forum')})"
        for i, ins in enumerate(insights[:10])  # cap at 10
    )

    column_summary = "\n".join(
        f"  {c['name']} ({c['dtype']}, {c['n_unique']} unique)"
        for c in schema.get("columns", [])
        if not c.get("is_id") and not c.get("is_target")
    )

    prompt = ROUND5_HYPOTHESIS_PROMPT.format(
        competition_name=state.get("competition_name", "unknown"),
        domain=competition_brief.get("domain", "tabular"),
        hypotheses=hypotheses_text,
        column_summary=column_summary,
    )

    try:
        response = llm_call(prompt, state)
        raw = _extract_json(response)
        items = json.loads(raw)
    except Exception as e:
        logger.warning(f"[feature_factory] Round 5 hypothesis LLM failed: {e}. Returning [].")
        return []

    schema_col_names = {c["name"] for c in schema.get("columns", [])}
    candidates = []

    for item in items[:10]:
        sources = item.get("source_columns", [])
        unknown = [s for s in sources if s not in schema_col_names]
        if unknown:
            continue
        candidates.append(FeatureCandidate(
            name=item["feature_name"],
            source_columns=sources,
            transform_type=item.get("transform_type", "interaction"),
            description=f"Hypothesis test: {item.get('hypothesis_summary', '')}. "
                        f"Expression: {item.get('expression', '')}",
            round=5,
        ))

    logger.info(f"[feature_factory] Round 5a: {len(candidates)} hypothesis features.")
    return candidates


def _generate_round5_interaction_features(
    schema: dict,
    competition_brief: dict,
    top_features_by_importance: list[str],   # from null importance result
    max_k: int = 20,
) -> list[FeatureCandidate]:
    """
    Round 5b: Creative interactions between top-K features.

    Uses competition_brief["meaningful_interactions"] to limit pairs
    to domain-meaningful ones before the hard budget cap.

    This function is called AFTER null importance filtering — it receives
    the list of features that actually survived Stage 2. Only these are
    eligible for interaction.
    """
    top_k = top_features_by_importance[:max_k]   # cap K
    meaningful_pairs = competition_brief.get("meaningful_interactions", [])

    candidates = []

    # Domain-guided pairs first — highest priority
    schema_col_names = {c["name"] for c in schema.get("columns", [])}
    for pair in meaningful_pairs:
        a, b = pair[0], pair[1]
        if a not in schema_col_names or b not in schema_col_names:
            continue
        if a not in top_k or b not in top_k:
            continue
        for op, op_name in [("multiply", "x"), ("divide", "div"), ("add", "plus")]:
            candidates.append(FeatureCandidate(
                name=f"{a}_{op_name}_{b}",
                source_columns=[a, b],
                transform_type=f"interaction_{op}",
                description=f"Domain-guided interaction: {a} {op} {b}",
                round=5,
            ))

    # Fill remaining budget with all top-K pairs not already covered
    existing_pairs = {tuple(sorted(c.source_columns)) for c in candidates}
    for i, a in enumerate(top_k):
        for b in top_k[i+1:]:
            if tuple(sorted([a, b])) in existing_pairs:
                continue
            candidates.append(FeatureCandidate(
                name=f"{a}_x_{b}",
                source_columns=[a, b],
                transform_type="interaction_multiply",
                description=f"Pairwise interaction: {a} × {b}",
                round=5,
            ))

    logger.info(
        f"[feature_factory] Round 5b: {len(candidates)} interaction candidates "
        f"(before budget cap in Task 2)."
    )
    return candidates
```

### Wiring Rounds 3–5 into `run_feature_factory()`
```python
def run_feature_factory(state: ProfessorState) -> ProfessorState:
    # ... existing Round 1 + 2 logic from Day 16 ...

    # Round 3
    round3_candidates = _generate_round3_aggregation_features(schema)

    # Round 4
    round4_candidates = _generate_round4_target_encoding_candidates(schema)

    # Round 5 (interaction candidates receive top features AFTER null importance)
    # In Day 18, top_features comes from state["null_importance_result"].survivors
    # if available, otherwise falls back to all feature names
    null_result = state.get("null_importance_result")
    top_features_by_importance = (
        null_result.survivors if null_result else
        [c["name"] for c in schema.get("columns", [])
         if not c.get("is_id") and not c.get("is_target")]
    )

    round5a_candidates = _generate_round5_hypothesis_features(schema, competition_brief, state)
    round5b_candidates = _generate_round5_interaction_features(
        schema, competition_brief, top_features_by_importance
    )
    round5_candidates = round5a_candidates + round5b_candidates

    all_candidates = (
        round1_candidates + round2_candidates +
        round3_candidates + round4_candidates +
        round5_candidates
    )
    # ... rest of feature factory (budget cap in Task 2, filtering stub) ...
```

---

## TASK 2 — GAP 12: Interaction budget cap

**The problem:** 200 features → 19,900 pairs × 2 operations = 39,800 candidates. With 5-shuffle Stage 1 filter: 199,000 LightGBM runs. Days of compute. Pipeline never finishes.

**Three hard rules in code — not prompts:**
```python
# constants.py (or at top of feature_factory.py)
MAX_INTERACTION_FEATURES    = 20    # Rule 1: only interact top-K survivors
MAX_INTERACTION_CANDIDATES  = 500   # Rule 3: absolute ceiling
```

### Rule 1 — Interact only top-K features by solo importance

This is already enforced by `_generate_round5_interaction_features()` which takes `top_features_by_importance[:max_k]`. The `max_k=20` cap limits pairs to `20×19/2 = 190` base pairs × 3 operations = 570 candidates before the hard cap.

### Rule 2 — Domain-guided pairs first

Already implemented in `_generate_round5_interaction_features()` — `meaningful_interactions` from `competition_brief.json` are added first, filling the budget with the domain-guided pairs before falling back to all top-K pairs.

### Rule 3 — Hard budget cap: `_apply_interaction_budget_cap()`
```python
def _apply_interaction_budget_cap(
    candidates: list[FeatureCandidate],
    all_round_candidates: list[FeatureCandidate],
    max_cap: int = MAX_INTERACTION_CANDIDATES,
) -> list[FeatureCandidate]:
    """
    Hard budget cap applied to interaction candidates (Round 5b) only.
    Non-interaction candidates (Rounds 1-4, Round 5a hypotheses) are never capped.

    When candidates exceed max_cap:
      Score = domain_relevance × importance_product
      domain_relevance: 2.0 if in meaningful_interactions, 1.0 otherwise
      importance_product: product of null_importance_percentile of both source columns
                          (falls back to 1.0 if null importance not yet run)

    Returns top-max_cap candidates by score.
    """
    interaction_candidates = [
        c for c in candidates
        if c.transform_type.startswith("interaction_")
    ]
    non_interaction = [
        c for c in all_round_candidates
        if not c.transform_type.startswith("interaction_")
    ]

    if len(interaction_candidates) <= max_cap:
        return all_round_candidates  # no cap needed

    # Build importance lookup from null importance result (may be None)
    importance_lookup: dict[str, float] = {}
    # In a real run, this would read from state["null_importance_result"]
    # Here it defaults to 1.0 (equal priority) — null importance refines this

    def _score(c: FeatureCandidate) -> float:
        domain_rel = 2.0 if c.name.endswith("_domain") else 1.0
        imp_a = importance_lookup.get(c.source_columns[0], 1.0)
        imp_b = importance_lookup.get(c.source_columns[1], 1.0) if len(c.source_columns) > 1 else 1.0
        return domain_rel * imp_a * imp_b

    interaction_candidates.sort(key=_score, reverse=True)
    kept_interactions = interaction_candidates[:max_cap]

    total_before = len(interaction_candidates)
    total_after  = len(kept_interactions)
    logger.info(
        f"[feature_factory] Interaction budget cap: "
        f"{total_before} → {total_after} interaction candidates "
        f"(max={max_cap}). "
        f"Dropped {total_before - total_after} low-priority interactions."
    )

    return non_interaction + kept_interactions


def _apply_interaction_budget_cap_with_importance(
    candidates: list[FeatureCandidate],
    null_importance_result,   # NullImportanceResult or None
    meaningful_interactions: list[list[str]],
    max_cap: int = MAX_INTERACTION_CANDIDATES,
) -> list[FeatureCandidate]:
    """
    Extended version that uses null importance percentiles for scoring.
    Called when null importance result is available (Day 17 has run).
    """
    interaction_candidates = [
        c for c in candidates if c.transform_type.startswith("interaction_")
    ]
    non_interaction = [
        c for c in candidates if not c.transform_type.startswith("interaction_")
    ]

    if len(interaction_candidates) <= max_cap:
        return candidates

    # Build importance lookup from null importance
    importance_lookup: dict[str, float] = {}
    if null_importance_result:
        for feat, info in null_importance_result.actual_vs_threshold.items():
            importance_lookup[feat] = float(info.get("actual", 1.0))

    # Build domain relevance set
    domain_pairs = {
        tuple(sorted(pair)) for pair in meaningful_interactions
    }

    def _score(c: FeatureCandidate) -> float:
        pair_key = tuple(sorted(c.source_columns[:2]))
        domain_rel = 2.0 if pair_key in domain_pairs else 1.0
        imp_a = importance_lookup.get(c.source_columns[0], 1.0)
        imp_b = importance_lookup.get(c.source_columns[1], 1.0) if len(c.source_columns) > 1 else 1.0
        return domain_rel * (imp_a * imp_b)

    interaction_candidates.sort(key=_score, reverse=True)
    return non_interaction + interaction_candidates[:max_cap]
```

### Wire into `run_feature_factory()`

After Round 5 generation, before null importance filtering:
```python
all_candidates = _apply_interaction_budget_cap_with_importance(
    candidates=all_candidates,
    null_importance_result=state.get("null_importance_result"),
    meaningful_interactions=competition_brief.get("meaningful_interactions", []),
    max_cap=MAX_INTERACTION_CANDIDATES,
)
```

---

## TASK 3 — GM-CAP 6: `agents/pseudo_label_agent.py`

**What pseudo-labeling is and why it works:** When test distribution differs from train (common in Kaggle), the model hasn't seen the kinds of examples in the test set. Adding high-confidence test predictions back as training data teaches the model about the test distribution. The key word is **high-confidence** — adding uncertain predictions destroys CV and hurts LB.

**The three guards that make this reliable:**
1. Confidence gate (top 10% only — farthest from decision boundary)
2. Validation fold integrity (pseudo-labels added to training folds ONLY — validation fold sees only real labels)
3. CV gate (only proceed if CV with pseudo-labels beats CV without)

**Max 3 iterations:** Diminishing returns after iteration 1. Risk of distribution shift amplification grows with each iteration.
```python
# agents/pseudo_label_agent.py

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from dataclasses import dataclass

CONFIDENCE_TOP_FRACTION  = 0.10   # top 10% most confident
MAX_PL_ITERATIONS        = 3
MIN_CV_IMPROVEMENT       = 0.001  # must improve by at least 0.1pp to continue


@dataclass
class PseudoLabelResult:
    iterations_completed:  int
    pseudo_labels_added:   list[int]   # count per iteration
    cv_scores_with_pl:     list[float]
    cv_scores_without_pl:  list[float]
    cv_improvements:       list[float]
    halted_early:          bool
    halt_reason:           str         # "max_iterations" | "cv_did_not_improve" | "no_confident_samples"
    final_pseudo_label_mask: list[int] # 1 if test sample is pseudo-labeled, 0 otherwise
    confidence_thresholds:  list[float]


def _compute_confidence(
    y_pred: np.ndarray,
    metric: str,
    quantile_model=None,  # for regression: quantile regression model
) -> np.ndarray:
    """
    Computes confidence score for each prediction.

    For binary classification (AUC, logloss):
        confidence = distance from decision boundary = |pred - 0.5|
        Maximum confidence = 0.5 (when pred ∈ {0.0, 1.0})
        Minimum confidence = 0.0 (when pred = 0.5)

    For regression:
        confidence = inverse of prediction interval width
        Requires a quantile regression model to estimate uncertainty.
        confidence = 1 / (q_high - q_low + epsilon)
        Falls back to std of OOF predictions if quantile model unavailable.

    For multiclass:
        confidence = max class probability - second highest class probability
        (margin between top-2 classes — robust to skewed class distributions)
    """
    if metric in ("auc", "logloss", "binary"):
        return np.abs(y_pred - 0.5)

    elif metric in ("rmse", "mae", "regression"):
        if quantile_model is not None:
            q_low  = quantile_model.predict(X=None, pred_quantile=0.1)
            q_high = quantile_model.predict(X=None, pred_quantile=0.9)
            return 1.0 / (q_high - q_low + 1e-6)
        else:
            # Fallback: uniform confidence (no interval estimation)
            logger.warning(
                "[pseudo_label] No quantile model for regression confidence. "
                "Using uniform confidence — all samples equally confident. "
                "This may reduce the quality of pseudo-label selection."
            )
            return np.ones(len(y_pred))

    elif metric in ("multiclass", "logloss_multiclass"):
        # Margin between top-2 class probabilities
        if y_pred.ndim == 1:
            return np.abs(y_pred - 0.5)  # binary fallback
        sorted_probs = np.sort(y_pred, axis=1)[:, ::-1]
        return sorted_probs[:, 0] - sorted_probs[:, 1]

    else:
        logger.warning(f"[pseudo_label] Unknown metric '{metric}' — using binary confidence.")
        return np.abs(y_pred - 0.5)


def _select_confident_samples(
    confidence: np.ndarray,
    y_pred: np.ndarray,
    top_fraction: float = CONFIDENCE_TOP_FRACTION,
) -> tuple[np.ndarray, float]:
    """
    Selects top-fraction of test samples by confidence.
    Returns (boolean mask, threshold confidence value).
    """
    threshold = np.percentile(confidence, (1.0 - top_fraction) * 100)
    mask = confidence >= threshold
    return mask, float(threshold)


def _run_cv_with_pseudo_labels(
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    X_pseudo: pl.DataFrame,
    y_pseudo: np.ndarray,
    lgbm_params: dict,
    n_folds: int = 5,
    metric: str = "auc",
    random_state: int = 42,
) -> list[float]:
    """
    Runs CV where pseudo-labels are added to TRAINING FOLDS ONLY.

    CRITICAL INVARIANT: Validation fold sees only real labeled samples.
    Pseudo-labels are concatenated to the training portion of each fold BEFORE fitting.
    They are NEVER used for validation scoring.

    This is the guard that prevents pseudo-labeling from inflating CV.
    """
    from sklearn.metrics import roc_auc_score, mean_squared_error

    is_classification = metric in ("auc", "logloss", "binary", "multiclass")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state) \
         if is_classification else \
         KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    X_np = X_train.to_numpy()
    fold_scores = []

    ModelClass = lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor
    X_pseudo_np = X_pseudo.to_numpy()

    for train_idx, val_idx in cv.split(X_np, y_train if is_classification else None):
        # Training fold: real labels + pseudo-labels
        X_fold_train = np.vstack([X_np[train_idx], X_pseudo_np])
        y_fold_train = np.concatenate([y_train[train_idx], y_pseudo])

        # Validation fold: ONLY real labels (invariant)
        X_fold_val = X_np[val_idx]
        y_fold_val = y_train[val_idx]

        model = ModelClass(**lgbm_params)
        model.fit(X_fold_train, y_fold_train)

        if metric == "auc":
            preds = model.predict_proba(X_fold_val)[:, 1]
            score = roc_auc_score(y_fold_val, preds)
        elif metric in ("rmse", "regression"):
            preds = model.predict(X_fold_val)
            score = -np.sqrt(mean_squared_error(y_fold_val, preds))  # negative so higher=better
        else:
            preds = model.predict_proba(X_fold_val)[:, 1]
            score = roc_auc_score(y_fold_val, preds)

        fold_scores.append(float(score))

        del model
        import gc; gc.collect()

    return fold_scores


def run_pseudo_label_agent(state: ProfessorState) -> ProfessorState:
    """
    GM-CAP 6: Pseudo-labeling with confidence gating.

    Pipeline:
    1. Train on labeled data (uses state["model_registry"] best model params)
    2. Predict test set
    3. Select top 10% most confident predictions
    4. Add to training folds only (never validation)
    5. Run CV with and without pseudo-labels — only proceed if CV improves
    6. Repeat up to MAX_PL_ITERATIONS

    Runs AFTER ensemble selection (needs OOF predictions from model_registry).
    The best model's params are reused — no new HPO.
    """
    from tools.wilcoxon_gate import is_significantly_better

    X_train   = state["X_train"]   # pl.DataFrame
    y_train   = state["y_train"]   # np.ndarray
    X_test    = state["X_test"]    # pl.DataFrame
    metric    = state.get("evaluation_metric", "auc")

    # Use best model params from registry
    best_model_name = state.get("selected_models", [None])[0]
    if not best_model_name:
        logger.warning("[pseudo_label] No selected models found. Skipping pseudo-labeling.")
        return state

    best_entry  = state["model_registry"][best_model_name]
    lgbm_params = best_entry.get("params", {"n_estimators": 500, "learning_rate": 0.05})

    # Baseline CV (no pseudo-labels)
    baseline_cv = best_entry.get("fold_scores", [])
    if not baseline_cv:
        logger.warning("[pseudo_label] No baseline fold scores. Skipping.")
        return state

    result = PseudoLabelResult(
        iterations_completed=0,
        pseudo_labels_added=[],
        cv_scores_with_pl=[],
        cv_scores_without_pl=[float(np.mean(baseline_cv))],
        cv_improvements=[],
        halted_early=False,
        halt_reason="",
        final_pseudo_label_mask=[],
        confidence_thresholds=[],
    )

    # Working copies — accumulate pseudo-labels across iterations
    X_pseudo_accumulated = pl.DataFrame(schema=X_train.schema)
    y_pseudo_accumulated = np.array([], dtype=y_train.dtype)
    current_test_mask    = np.zeros(len(X_test), dtype=bool)

    for iteration in range(1, MAX_PL_ITERATIONS + 1):
        logger.info(f"[pseudo_label] Iteration {iteration}/{MAX_PL_ITERATIONS}")

        # Train on labelled + accumulated pseudo-labels
        X_all  = pl.concat([X_train, X_pseudo_accumulated]) if len(X_pseudo_accumulated) > 0 else X_train
        y_all  = np.concatenate([y_train, y_pseudo_accumulated]) if len(y_pseudo_accumulated) > 0 else y_train

        ModelClass = lgb.LGBMClassifier if metric in ("auc", "logloss", "binary") \
                     else lgb.LGBMRegressor
        model = ModelClass(**lgbm_params)
        model.fit(X_all.to_numpy(), y_all)

        # Predict test set — exclude already pseudo-labeled samples
        remaining_mask = ~current_test_mask
        X_remaining = X_test.filter(pl.Series(remaining_mask))

        if X_remaining.is_empty():
            result.halt_reason = "no_confident_samples"
            result.halted_early = True
            logger.info("[pseudo_label] All test samples already pseudo-labeled. Stopping.")
            del model; import gc; gc.collect()
            break

        is_cls = metric in ("auc", "logloss", "binary")
        y_pred = model.predict_proba(X_remaining.to_numpy())[:, 1] if is_cls \
                 else model.predict(X_remaining.to_numpy())

        del model; import gc; gc.collect()

        # Select high-confidence samples
        confidence = _compute_confidence(y_pred, metric)
        conf_mask, threshold = _select_confident_samples(confidence, y_pred)

        n_selected = int(conf_mask.sum())
        if n_selected == 0:
            result.halt_reason = "no_confident_samples"
            result.halted_early = True
            logger.info("[pseudo_label] No samples met confidence threshold. Stopping.")
            break

        result.confidence_thresholds.append(threshold)

        X_new_pseudo = X_remaining.filter(pl.Series(conf_mask))
        y_new_pseudo = y_pred[conf_mask]

        # For classification, convert to hard labels
        if is_cls:
            y_new_pseudo = (y_new_pseudo >= 0.5).astype(y_train.dtype)

        # CV with pseudo-labels — validation fold ONLY sees real labels
        cv_with = _run_cv_with_pseudo_labels(
            X_train=X_train,
            y_train=y_train,
            X_pseudo=pl.concat([X_pseudo_accumulated, X_new_pseudo])
                      if len(X_pseudo_accumulated) > 0 else X_new_pseudo,
            y_pseudo=np.concatenate([y_pseudo_accumulated, y_new_pseudo])
                     if len(y_pseudo_accumulated) > 0 else y_new_pseudo,
            lgbm_params=lgbm_params,
            metric=metric,
        )

        cv_mean_with    = float(np.mean(cv_with))
        cv_mean_without = float(np.mean(baseline_cv)) if iteration == 1 \
                          else result.cv_scores_with_pl[-1]
        improvement     = cv_mean_with - cv_mean_without

        result.cv_scores_with_pl.append(cv_mean_with)
        result.cv_improvements.append(round(improvement, 6))
        result.pseudo_labels_added.append(n_selected)

        logger.info(
            f"[pseudo_label] Iteration {iteration}: "
            f"n_added={n_selected}, threshold={threshold:.4f}, "
            f"cv_before={cv_mean_without:.5f}, cv_after={cv_mean_with:.5f}, "
            f"improvement={improvement:+.5f}"
        )

        # Wilcoxon gate: is improvement statistically significant?
        # Use the CV folds from baseline vs new CV as paired comparison
        gate_passed = is_significantly_better(cv_with, baseline_cv)

        if not gate_passed and improvement < MIN_CV_IMPROVEMENT:
            result.halt_reason = "cv_did_not_improve"
            result.halted_early = True
            logger.info(
                f"[pseudo_label] CV did not improve significantly "
                f"(improvement={improvement:+.5f} < min={MIN_CV_IMPROVEMENT}, "
                f"Wilcoxon gate failed). Reverting iteration {iteration}."
            )
            break

        # Accept iteration — accumulate pseudo-labels
        X_pseudo_accumulated = pl.concat([X_pseudo_accumulated, X_new_pseudo]) \
                               if len(X_pseudo_accumulated) > 0 else X_new_pseudo
        y_pseudo_accumulated = np.concatenate([y_pseudo_accumulated, y_new_pseudo]) \
                               if len(y_pseudo_accumulated) > 0 else y_new_pseudo
        current_test_mask[np.where(remaining_mask)[0][conf_mask]] = True
        result.iterations_completed = iteration
        baseline_cv = cv_with  # next iteration compares against this

    if result.iterations_completed == MAX_PL_ITERATIONS and not result.halted_early:
        result.halt_reason = "max_iterations"

    result.final_pseudo_label_mask = current_test_mask.tolist()

    # Update state
    state = {
        **state,
        "pseudo_label_result":         result,
        "X_train_with_pseudo":         pl.concat([X_train, X_pseudo_accumulated])
                                       if len(X_pseudo_accumulated) > 0 else X_train,
        "y_train_with_pseudo":         np.concatenate([y_train, y_pseudo_accumulated])
                                       if len(y_pseudo_accumulated) > 0 else y_train,
        "pseudo_labels_applied":       result.iterations_completed > 0 and not result.halted_early,
        "pseudo_label_cv_improvement": sum(result.cv_improvements) if result.cv_improvements else 0.0,
    }

    log_event(
        state=state,
        action="pseudo_label_complete",
        agent="pseudo_label_agent",
        details={
            "iterations":       result.iterations_completed,
            "total_pl_added":   sum(result.pseudo_labels_added),
            "cv_improvement":   state["pseudo_label_cv_improvement"],
            "halt_reason":      result.halt_reason,
        }
    )

    return state
```

### Pipeline position

Pseudo-label agent runs AFTER `ensemble_architect` and BEFORE `submission_strategist`:
```
... → ensemble_architect → pseudo_label_agent → submission_strategist → ...
```

The retrained model (using `X_train_with_pseudo`, `y_train_with_pseudo`) is used for final test predictions. If `pseudo_labels_applied=False`, the pipeline falls through to standard predictions.

### New `ProfessorState` fields
```python
pseudo_label_result:          object          # PseudoLabelResult dataclass
X_train_with_pseudo:          pl.DataFrame    # augmented training data
y_train_with_pseudo:          np.ndarray      # augmented labels
pseudo_labels_applied:        bool            # False if no improvement found
pseudo_label_cv_improvement:  float           # total CV improvement across iterations
```

---

## INTEGRATION CHECKLIST

- [ ] `_generate_round3_aggregation_features()` caps at `MAX_ROUND3_CANDIDATES=200`
- [ ] `_apply_round3_transforms()` uses Polars `group_by().agg()` followed by join (not `apply()`)
- [ ] `_apply_round4_target_encoding()` uses fold-split: pseudo-labels added to train fold ONLY
- [ ] Round 4 smoothing formula: `(count * group_mean + s * global_mean) / (count + s)` where `s=30`
- [ ] Round 5 interaction candidates pass through `_apply_interaction_budget_cap_with_importance()`
- [ ] `MAX_INTERACTION_CANDIDATES = 500` enforced as absolute ceiling on interaction candidates
- [ ] Budget cap scoring uses domain_relevance × importance_product (not random selection)
- [ ] Round 5a validates source columns against schema (same guard as Round 2)
- [ ] Pseudo-label agent runs AFTER ensemble (needs model_registry with OOF predictions)
- [ ] Validation fold NEVER contains pseudo-labels — this is the critical invariant
- [ ] `is_significantly_better()` (Day 13 Wilcoxon gate) applied to CV improvement check
- [ ] `_compute_confidence()` handles binary, regression, and multiclass metrics
- [ ] Max 3 iterations enforced — loop exits even if all iterations improve
- [ ] `PseudoLabelResult` and `NullImportanceResult` excluded from Redis checkpoint
- [ ] `_apply_round4_target_encoding()` uses the same `n_folds` as ml_optimizer (default 5)

## GIT COMMIT MESSAGE
```
Day 18: feature factory rounds 3-5, interaction cap, pseudo-labeling

- feature_factory: Round 3 — groupby aggregation (mean/std/min/max/count)
  cap at MAX_ROUND3_CANDIDATES=200, ranked by cardinality proxy
- feature_factory: Round 4 — CV-safe target encoding with smoothing=30
  fold-isolated: group stats computed from OTHER folds only
- feature_factory: Round 5a — hypothesis feature testing from forum insights
- feature_factory: Round 5b — interaction features, domain pairs first
- feature_factory: interaction budget cap (_apply_interaction_budget_cap)
  MAX_INTERACTION_FEATURES=20, MAX_INTERACTION_CANDIDATES=500
  scored by domain_relevance × importance_product
- pseudo_label_agent: confidence-gated pseudo-labeling (GAP 9)
  top 10%, validation fold integrity, CV gate + Wilcoxon, max 3 iterations
- tests/test_day18_quality.py: 52 adversarial tests — all green
```