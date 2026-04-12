# agents/feature_factory.py
# -------------------------------------------------------------------------
# Day 16: Feature Factory — Rounds 1 (generic) + 2 (domain/LLM)
# Day 17: Wilcoxon gate + null importance filtering.
# Day 18: Rounds 3 (aggregation) + 4 (target encoding) + 5 (hypothesis +
#          interactions) + interaction budget cap.
# Reads schema.json + competition_brief.json only — never raw data.
# Writes feature_manifest.json with per-feature metadata.
# -------------------------------------------------------------------------

import ast
import json
import math
import logging
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl
import numpy as np
from sklearn.model_selection import KFold

from core.state import ProfessorState
from core.lineage import log_event
from tools.llm_client import call_llm
from tools.wilcoxon_gate import feature_gate_result, is_feature_worth_adding
from tools.null_importance import run_null_importance_filter
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
MAX_ROUND3_CANDIDATES = 200
MAX_ROUND4_CANDIDATES = 30
MAX_INTERACTION_FEATURES = 20
MAX_INTERACTION_CANDIDATES = 500
ROUND3_AGG_FUNCTIONS = ["mean", "std", "min", "max", "count"]


# ── Feature candidate dataclass ──────────────────────────────────

@dataclass
class FeatureCandidate:
    name: str
    source_columns: list[str]
    transform_type: str       # "log", "sqrt", "missingness_flag", "ratio", "domain", etc.
    description: str
    round: int                # 1 or 2
    # Filled in after testing (Day 17 adds these):
    null_importance_percentile: float | None = None
    wilcoxon_p: float | None = None
    cv_delta: float | None = None
    verdict: str = "PENDING"  # PENDING | KEEP | DROP
    expression: str | None = None


# ── Round 1: generic transforms ─────────────────────────────────

def _generate_round1_features(schema: dict) -> list[FeatureCandidate]:
    """
    Round 1: generic transforms. Safe to apply to any tabular competition.
    Reads schema only — never touches raw data.

    Transforms:
      - log1p for positive-skewed numeric columns (min >= 0, n_unique > 10)
      - sqrt for non-negative numeric columns (different skew profile)
      - missingness flags for columns with null_fraction > 0.01
    """
    candidates = []
    columns = schema.get("columns", [])

    for col_name in columns:
        if isinstance(col_name, dict):
            # Fallback if an older schema version was used
            name = col_name.get("name", "")
            dtype = col_name.get("dtype", "")
            null_fraction = float(col_name.get("null_fraction", 0.0))
            min_val = col_name.get("min")
            n_unique = int(col_name.get("n_unique", 0))
            is_id = col_name.get("is_id", False)
            is_target = col_name.get("is_target", False)
        else:
            name = col_name
            dtype = str(schema.get("types", {}).get(name, ""))
            null_fraction = float(schema.get("missing_rates", {}).get(name, 0.0))
            min_val = 0  # Assuming non-negative for imputation/log1p fallback if strictly numeric
            n_unique = int(schema.get("cardinality", {}).get(name, 0))
            # Heuristics for ID and target
            is_id = any(kw in name.lower() for kw in ["id", "index"])
            is_target = name == schema.get("target_col", "unknown")

        if is_id or is_target:
            continue

        is_numeric = any(t in dtype for t in ("float", "int"))

        # log1p transform: applicable to non-negative numerics with reasonable variance
        if is_numeric and min_val is not None and float(min_val) >= 0 and n_unique > 10:
            candidates.append(FeatureCandidate(
                name=f"log1p_{name}",
                source_columns=[name],
                transform_type="log",
                description=f"log1p({name}) — natural log transform for skew reduction",
                round=1,
            ))

        # sqrt transform: different from log — handles moderate skew differently
        if is_numeric and min_val is not None and float(min_val) >= 0 and n_unique > 5:
            candidates.append(FeatureCandidate(
                name=f"sqrt_{name}",
                source_columns=[name],
                transform_type="sqrt",
                description=f"sqrt({name}) — square root transform for moderate skew",
                round=1,
            ))

        # Missingness flag: binary indicator that the value was missing
        if null_fraction > 0.01:
            candidates.append(FeatureCandidate(
                name=f"missing_{name}",
                source_columns=[name],
                transform_type="missingness_flag",
                description=f"Binary flag: 1 if {name} was null, 0 otherwise",
                round=1,
            ))

    return candidates


def _apply_round1_transforms(X: pl.DataFrame, candidates: list[FeatureCandidate]) -> pl.DataFrame:
    """
    Applies Round 1 transforms to the actual DataFrame.
    Returns a DataFrame with new columns appended.
    """
    new_cols = []

    for c in candidates:
        if c.transform_type == "log" and c.source_columns[0] in X.columns:
            src = c.source_columns[0]
            new_cols.append(
                (pl.col(src).cast(pl.Float64).fill_null(0.0) + 1.0)
                .log(base=math.e)
                .alias(c.name)
            )
        elif c.transform_type == "sqrt" and c.source_columns[0] in X.columns:
            src = c.source_columns[0]
            new_cols.append(
                pl.col(src).cast(pl.Float64).fill_null(0.0)
                .sqrt()
                .alias(c.name)
            )
        elif c.transform_type == "missingness_flag" and c.source_columns[0] in X.columns:
            src = c.source_columns[0]
            new_cols.append(
                pl.col(src).is_null().cast(pl.Int8).alias(c.name)
            )

    if not new_cols:
        return X

    return X.with_columns(new_cols)


# ── Round 2: domain features from competition_brief ──────────────

DOMAIN_FEATURE_PROMPT = """
You are an expert Kaggle feature engineer. Given the competition brief and column schema below,
generate domain-specific feature engineering ideas.

Competition domain: {domain}
Task type: {task_type}
Known winning features from similar competitions: {known_winning_features}

Available columns (from schema.json):
{column_summary}

Generate feature ideas that:
1. Use ONLY the available columns listed above
2. Are likely to improve model performance based on the domain
3. Can be expressed as valid Polars Python expressions

Return a JSON array of feature candidates:
[
  {{
    "name": "feature_name_snake_case",
    "source_columns": ["col_a", "col_b"],
    "transform_type": "ratio | interaction | groupby_mean | groupby_std | string_extract | bin",
    "expression": "pl.col('col_a') / (pl.col('col_b') + 1)",
    "domain_rationale": "Why this feature is likely predictive given the domain"
  }}
]

CRITICAL: The "expression" field MUST be a valid Polars Python expression string that can be evaluated with eval().
Examples of VALID expressions:
- "pl.col('feature_0') / (pl.col('feature_1') + 1)"
- "(pl.col('feature_0') + pl.col('feature_1')) / 2"
- "pl.col('feature_0') * pl.col('feature_1')"
- "(pl.col('feature_0').cast(pl.Float64) + 1.0).log()"

Examples of INVALID expressions (DO NOT USE):
- "feature_0 divided by feature_1" (natural language, not code)
- "Sum of all five features" (natural language)
- "feature_0 / feature_1" (missing pl.col() wrapper)
- "pl.col(feature_0)" (missing quotes around column name)

Rules:
- Maximum 15 candidates. Quality over quantity.
- Every source_column must exist in the schema.
- Do NOT suggest target encoding — this is handled separately.
- Do NOT suggest lag features for non-time-series competitions.
- expression MUST be a valid Polars expression string, not natural language.
"""


def _extract_json(text: str) -> str:
    """Extract JSON array from LLM response text."""
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        return text[start:end + 1]
    raise ValueError("No JSON array found in response")


def _generate_round2_features(
    schema: dict,
    competition_brief: dict,
    state: ProfessorState,
) -> list[FeatureCandidate]:
    """
    Round 2: domain-specific features generated by LLM from competition_brief.json.
    Only reads schema and competition_brief — never raw data.
    """
    if not competition_brief:
        logger.info("[feature_factory] No competition_brief — skipping Round 2.")
        return []

    column_summary = "\n".join(
        f"  {c['name']} ({c.get('dtype', 'unknown')}, {c.get('n_unique', '?')} unique, "
        f"null={float(c.get('null_fraction', 0)):.0%})"
        for c in schema.get("columns", [])
        if not c.get("is_id") and not c.get("is_target")
    )

    if not column_summary.strip():
        logger.info("[feature_factory] No eligible columns for Round 2.")
        return []

    prompt = DOMAIN_FEATURE_PROMPT.format(
        domain=competition_brief.get("domain", "tabular"),
        task_type=competition_brief.get("task_type", "binary_classification"),
        known_winning_features=json.dumps(
            competition_brief.get("known_winning_features", [])[:5]
        ),
        column_summary=column_summary,
    )

    try:
        response = call_llm(prompt, system="", model="deepseek")
        raw = _extract_json(response)
        candidates_raw = json.loads(raw)
    except Exception as e:
        logger.warning(f"[feature_factory] Round 2 LLM call failed: {e}. Returning no Round 2 candidates.")
        return []

    candidates = []
    schema_col_names = {c["name"] for c in schema.get("columns", [])}

    for item in candidates_raw[:15]:  # cap at 15
        # Validate all source columns exist in schema
        sources = item.get("source_columns", [])
        unknown = [s for s in sources if s not in schema_col_names]
        if unknown:
            logger.warning(
                f"[feature_factory] Round 2 candidate '{item.get('name', '?')}' "
                f"references unknown columns {unknown}. Skipping."
            )
            continue

        candidates.append(FeatureCandidate(
            name=item["name"],
            source_columns=sources,
            transform_type=item.get("transform_type", "domain"),
            description=item.get("expression", item.get("domain_rationale", "")),
            round=2,
            expression=item.get("expression", ""),
        ))

    logger.info(f"[feature_factory] Round 2: {len(candidates)} domain feature candidates generated.")
    return candidates


# ── Schema helper functions ──────────────────────────────────────

def _is_categorical(col: dict) -> bool:
    """Check if a schema column is categorical."""
    dtype = col.get("dtype", "").lower()
    n_unique = int(col.get("n_unique", 0))
    return any(t in dtype for t in ("str", "cat", "object")) or n_unique < 50


def _is_numeric(col: dict) -> bool:
    """Check if a schema column is numeric."""
    dtype = col.get("dtype", "").lower()
    return any(t in dtype for t in ("float", "int"))


def _find_col(schema: dict, name: str) -> dict:
    """Find a column definition in schema by name."""
    for c in schema.get("columns", []):
        if c["name"] == name:
            return c
    return {}


# ── Round 3: Aggregation features (groupby stats) ───────────────

def _generate_round3_aggregation_features(schema: dict) -> list[FeatureCandidate]:
    """
    Round 3: groupby aggregation features.

    For each (categorical, numeric) pair:
      - categorical: dtype contains 'str', 'cat', 'object', or n_unique < 50
      - numeric: dtype contains 'float' or 'int', not id, not target
      - generate: mean, std, min, max, count of numeric grouped by categorical

    Hard cap: MAX_ROUND3_CANDIDATES. If exceeded, rank by
    (1/(n_unique_categorical+1)) * n_unique_numeric descending and take top N.
    """
    columns = schema.get("columns", [])
    candidates = []

    categoricals = [
        c for c in columns
        if not c.get("is_id") and not c.get("is_target")
        and _is_categorical(c)
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
        def _agg_priority(c: FeatureCandidate) -> float:
            cat_col = _find_col(schema, c.source_columns[1])
            num_col = _find_col(schema, c.source_columns[0])
            cat_card = float(cat_col.get("n_unique", 1))
            num_unique = float(num_col.get("n_unique", 1))
            return (1.0 / (cat_card + 1)) * num_unique

        candidates = sorted(candidates, key=_agg_priority, reverse=True)[:MAX_ROUND3_CANDIDATES]

    logger.info(f"[feature_factory] Round 3: {len(candidates)} aggregation candidates.")
    return candidates


def _apply_round3_transforms(X: pl.DataFrame, candidates: list[FeatureCandidate]) -> pl.DataFrame:
    """Applies groupby aggregation transforms using Polars group_by+join."""
    for c in candidates:
        if c.transform_type != "groupby_agg":
            continue
        num_col, cat_col = c.source_columns[0], c.source_columns[1]
        if num_col not in X.columns or cat_col not in X.columns:
            continue
        # Extract fn name from candidate name: "{num}_{fn}_by_{cat}"
        parts = c.name.split("_")
        # Find the fn name that follows the numeric column name
        fn_name = None
        for fn in ROUND3_AGG_FUNCTIONS:
            if f"_{fn}_by_" in c.name:
                fn_name = fn
                break
        if fn_name is None:
            continue
        agg_fn = {
            "mean": pl.col(num_col).mean(),
            "std":  pl.col(num_col).std(),
            "min":  pl.col(num_col).min(),
            "max":  pl.col(num_col).max(),
            "count": pl.col(num_col).count(),
        }.get(fn_name)
        if agg_fn is None:
            continue
        group_stats = X.group_by(cat_col).agg(agg_fn.alias(c.name))
        X = X.join(group_stats, on=cat_col, how="left")
    return X


# ── Round 4: CV-safe target encoding ────────────────────────────

def _generate_round4_target_encoding_candidates(schema: dict) -> list[FeatureCandidate]:
    """
    Round 4: CV-safe target encoding candidates.

    Only for categorical columns with 2 <= n_unique <= 200.
    Sorted by n_unique descending. Capped at MAX_ROUND4_CANDIDATES.
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
                f"n_unique={n_unique}. Computed within folds only."
            ),
            round=4,
        ))

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

    Unseen categories get global_mean.
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
        col_values = X[col].to_numpy()

        for fold_idx in range(n_folds):
            train_mask = fold_assignments != fold_idx
            val_mask = fold_assignments == fold_idx

            cat_values_train = col_values[train_mask]
            y_train_fold = y[train_mask]

            # Compute per-category stats from training portion only
            cat_stats: dict = {}
            for cat, target in zip(cat_values_train, y_train_fold):
                key = str(cat)
                if key not in cat_stats:
                    cat_stats[key] = [0.0, 0]
                cat_stats[key][0] += float(target)
                cat_stats[key][1] += 1

            # Apply smoothed encoding to validation portion
            val_indices = np.where(val_mask)[0]
            for idx in val_indices:
                key = str(col_values[idx])
                if key in cat_stats:
                    sum_t, count = cat_stats[key]
                    group_mean = sum_t / count
                    encoded[idx] = (
                        (count * group_mean + smoothing * global_mean)
                        / (count + smoothing)
                    )
                # Unseen categories -> global mean (already set as default)

        result_X = result_X.with_columns(
            pl.Series(name=candidate.name, values=encoded)
        )

    return result_X


# ── Round 5: Hypothesis testing + creative interactions ──────────

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
        for i, ins in enumerate(insights[:10])
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
        response = call_llm(prompt, system="", model="deepseek")
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
            logger.warning(
                f"[feature_factory] Round 5a candidate '{item.get('feature_name', '?')}' "
                f"references unknown columns {unknown}. Skipping."
            )
            continue
        candidates.append(FeatureCandidate(
            name=item["feature_name"],
            source_columns=sources,
            transform_type=item.get("transform_type", "interaction"),
            description=f"Hypothesis test: {item.get('hypothesis_summary', '')}. "
                        f"Expression: {item.get('expression', '')}",
            round=5,
            expression=item.get("expression", ""),
        ))

    logger.info(f"[feature_factory] Round 5a: {len(candidates)} hypothesis features.")
    return candidates


def _generate_round5_interaction_features(
    schema: dict,
    competition_brief: dict,
    top_features_by_importance: list[str],
    max_k: int = MAX_INTERACTION_FEATURES,
) -> list[FeatureCandidate]:
    """
    Round 5b: Creative interactions between top-K features.

    Uses competition_brief["meaningful_interactions"] to limit pairs
    to domain-meaningful ones first, then fills with top-K pairs.
    """
    top_k = top_features_by_importance[:max_k]
    meaningful_pairs = competition_brief.get("meaningful_interactions", [])

    candidates = []

    # Domain-guided pairs first
    schema_col_names = {c["name"] for c in schema.get("columns", [])}
    for pair in meaningful_pairs:
        if len(pair) < 2:
            continue
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
                expression=f"{a} * {b}" if op == "multiply" else f"{a} / ({b} + 1)" if op == "divide" else f"{a} + {b}",
            ))

    # Fill remaining with all top-K pairs not already covered
    existing_pairs = {tuple(sorted(c.source_columns[:2])) for c in candidates}
    for i, a in enumerate(top_k):
        for b in top_k[i + 1:]:
            if tuple(sorted([a, b])) in existing_pairs:
                continue
            candidates.append(FeatureCandidate(
                name=f"{a}_x_{b}",
                source_columns=[a, b],
                transform_type="interaction_multiply",
                description=f"Pairwise interaction: {a} * {b}",
                round=5,
                expression=f"{a} * {b}",
            ))

    logger.info(
        f"[feature_factory] Round 5b: {len(candidates)} interaction candidates "
        f"(before budget cap)."
    )
    return candidates


# ── Interaction budget cap ───────────────────────────────────────

def _apply_interaction_budget_cap(
    candidates: list[FeatureCandidate],
    all_round_candidates: list[FeatureCandidate],
    max_cap: int = MAX_INTERACTION_CANDIDATES,
) -> list[FeatureCandidate]:
    """
    Hard budget cap on interaction candidates (Round 5b) only.
    Non-interaction candidates (Rounds 1-4, Round 5a) are never capped.
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
        return all_round_candidates

    def _score(c: FeatureCandidate) -> float:
        domain_rel = 2.0 if "_domain" in c.description.lower() else 1.0
        return domain_rel

    interaction_candidates.sort(key=_score, reverse=True)
    kept_interactions = interaction_candidates[:max_cap]

    total_before = len(interaction_candidates)
    total_after = len(kept_interactions)
    logger.info(
        f"[feature_factory] Interaction budget cap: "
        f"{total_before} -> {total_after} interaction candidates "
        f"(max={max_cap}). "
        f"Dropped {total_before - total_after} low-priority interactions."
    )

    return non_interaction + kept_interactions


def _apply_interaction_budget_cap_with_importance(
    candidates: list[FeatureCandidate],
    null_importance_result,
    meaningful_interactions: list[list[str]],
    max_cap: int = MAX_INTERACTION_CANDIDATES,
) -> list[FeatureCandidate]:
    """
    Extended budget cap that uses null importance percentiles for scoring.
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
    if null_importance_result and hasattr(null_importance_result, 'actual_vs_threshold'):
        for feat, info in null_importance_result.actual_vs_threshold.items():
            importance_lookup[feat] = float(info.get("actual", 1.0))

    # Build domain relevance set
    domain_pairs = {
        tuple(sorted(pair[:2])) for pair in meaningful_interactions if len(pair) >= 2
    }

    def _score(c: FeatureCandidate) -> float:
        pair_key = tuple(sorted(c.source_columns[:2]))
        domain_rel = 2.0 if pair_key in domain_pairs else 1.0
        imp_a = importance_lookup.get(c.source_columns[0], 1.0)
        imp_b = importance_lookup.get(c.source_columns[1], 1.0) if len(c.source_columns) > 1 else 1.0
        return domain_rel * (imp_a * imp_b)

    interaction_candidates.sort(key=_score, reverse=True)
    return non_interaction + interaction_candidates[:max_cap]


# ── Day 17: Statistical feature evaluation ──────────────────────

def _quick_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
    task_type: str = "binary",
) -> list[float]:
    """
    Quick K-fold CV for feature evaluation.
    Uses n_folds=3 (not 5) to keep each evaluation at ~1-3 seconds.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import accuracy_score, mean_squared_error

    if task_type == "binary":
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        ModelClass = lgb.LGBMClassifier
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        ModelClass = lgb.LGBMRegressor

    params = {"n_estimators": 100, "verbosity": -1, "n_jobs": 1}
    scores = []
    for train_idx, val_idx in kf.split(X, y):
        model = ModelClass(**params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        if task_type == "binary":
            score = accuracy_score(y[val_idx], preds)
        else:
            score = -float(mean_squared_error(y[val_idx], preds))
        scores.append(score)
        del model
    return scores


def _evaluate_candidate_feature(
    state: ProfessorState,
    X_base: pl.DataFrame,
    X_with_candidate: pl.DataFrame,
    y: np.ndarray,
    feature_name: str,
    skip_wilcoxon_gate: bool = False,
) -> bool:
    """
    Runs a quick 3-fold CV with and without the candidate feature.
    Returns True if the feature passes the Wilcoxon gate.
    """
    if skip_wilcoxon_gate:
        return True

    task_type = state.get("task_type", "binary")
    if task_type not in ("binary", "regression"):
        task_type = "binary"

    baseline_scores = _quick_cv(X_base.to_numpy(), y, n_folds=3, task_type=task_type)
    augmented_scores = _quick_cv(X_with_candidate.to_numpy(), y, n_folds=3, task_type=task_type)

    result = feature_gate_result(
        baseline_fold_scores=baseline_scores,
        augmented_fold_scores=augmented_scores,
        feature_name=feature_name,
    )

    log_event(
        session_id=state["session_id"],
        agent="feature_factory",
        action="wilcoxon_feature_gate",
        keys_read=["feature_candidates"],
        keys_written=["features_gate_passed", "features_gate_dropped"],
        values_changed={
            "feature_name": feature_name,
            "gate_passed": result["gate_passed"],
            "decision": result["decision"],
        },
    )

    return result["gate_passed"]


def _apply_null_importance_filter(
    state: ProfessorState,
    X: pl.DataFrame,
    y: np.ndarray,
) -> tuple[list[str], ProfessorState]:
    """
    Applies two-stage null importance filter to candidate feature set.
    Updates state with survivor list and dropped features.
    Returns (survivor_names, updated_state).
    """
    target_col = state.get("target_column", "")
    id_col = state.get("id_column", "")
    feature_names = [c for c in X.columns if c not in {target_col, id_col} and c]

    result = run_null_importance_filter(
        X=X, y=y,
        feature_names=feature_names,
        task_type=state.get("task_type", "binary"),
    )

    log_event(
        session_id=state["session_id"],
        agent="feature_factory",
        action="null_importance_filter_complete",
        keys_read=["feature_candidates"],
        keys_written=["features_dropped_stage1", "features_dropped_stage2",
                      "null_importance_result"],
        values_changed={
            "total_input":    result.total_features_input,
            "total_output":   result.total_features_output,
            "stage1_dropped": result.stage1_drop_count,
            "stage2_dropped": result.stage2_drop_count,
            "elapsed_s":      round(result.elapsed_seconds, 1),
        },
    )

    state = {
        **state,
        "null_importance_result": result,
        "features_dropped_stage1": result.dropped_stage1,
        "features_dropped_stage2": result.dropped_stage2,
    }

    return result.survivors, state


# ── Manifest builder ─────────────────────────────────────────────

def _build_feature_manifest(candidates: list[FeatureCandidate], schema: dict) -> dict:
    """
    Builds the feature_manifest.json structure.
    Per-feature: name, transform_type, source_columns, round, description,
                 null_importance_percentile, wilcoxon_p, cv_delta, verdict.
    """
    return {
        "total_candidates": len(candidates),
        "total_kept": sum(1 for c in candidates if c.verdict == "KEEP"),
        "total_dropped": sum(1 for c in candidates if c.verdict == "DROP"),
        "features": [
            {
                "name": c.name,
                "transform_type": c.transform_type,
                "source_columns": c.source_columns,
                "round": c.round,
                "description": c.description,
                "null_importance_percentile": c.null_importance_percentile,
                "wilcoxon_p": c.wilcoxon_p,
                "cv_delta": c.cv_delta,
                "verdict": c.verdict,
            }
            for c in candidates
        ],
        "schema_version": schema.get("session_id", "unknown"),
        "generated_at": datetime.utcnow().isoformat(),
    }


# ── Main entry point ─────────────────────────────────────────────

def _rewrite_llm_expression(expr_str: str, allowed_cols: set[str]) -> str:
    """Safely upgrades simple algebraic equations to valid Polars strings."""
    import ast
    class PolarsRewriter(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in allowed_cols:
                return ast.Call(
                    func=ast.Attribute(value=ast.Name(id="pl", ctx=ast.Load()), attr="col", ctx=ast.Load()),
                    args=[ast.Constant(value=node.id)],
                    keywords=[]
                )
            return node
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError:
        raise ValueError(f"Syntax error in expression: {expr_str}")

    rewritten = PolarsRewriter().visit(tree)
    ast.fix_missing_locations(rewritten)
    # Wrap entire output inside parens to be safe
    return f"({ast.unparse(rewritten)})"


# ── SECURITY FIX: Safe Polars Expression Evaluator ──────────────────

_ALLOWED_AST_NODES = {
    # Core
    ast.Expression, ast.Call, ast.Attribute, ast.Name, ast.Load,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare,
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or, ast.Not,
    # Functions
    ast.List, ast.Tuple,
}

_ALLOWED_POLARS_ATTRS = {
    'col', 'lit', 'select', 'with_columns', 'filter', 'group_by', 'agg',
    'mean', 'std', 'var', 'min', 'max', 'sum', 'count', 'median',
    'log', 'log10', 'exp', 'sqrt', 'abs', 'round', 'floor', 'ceil',
    'cast', 'fill_null', 'fill_nan', 'drop_nulls', 'is_null', 'is_not_null',
    'is_in', 'is_not_in', 'between', 'clip', 'replace',
    'str', 'dt', 'arr',  # namespaces
    'contains', 'starts_with', 'ends_with',  # string methods
    'year', 'month', 'day', 'hour', 'minute', 'second',  # datetime methods
}


def _safe_eval_polars_expr(expr_str: str, allowed_modules: dict) -> Any:
    """
    SECURITY FIX: Safely evaluate Polars expressions without using eval().
    
    Uses AST parsing and validation to prevent code injection attacks.
    
    Args:
        expr_str: Polars expression string (e.g., "pl.col('feature_0') / 2")
        allowed_modules: Dict of allowed modules (e.g., {"pl": polars, "np": numpy})
    
    Returns:
        Evaluated Polars expression object
    
    Raises:
        ValueError: If expression contains unsafe nodes or attributes
    """
    import ast
    
    # Parse expression
    try:
        tree = ast.parse(expr_str.strip(), mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in expression: {expr_str}. Error: {e}")
    
    # Validate all AST nodes are safe
    for node in ast.walk(tree):
        # Check node type is allowed
        if type(node) not in _ALLOWED_AST_NODES:
            raise ValueError(
                f"Unsafe AST node detected: {type(node).__name__}. "
                f"Expression: {expr_str}"
            )
        
        # Check attribute access is safe (for pl.col, pl.lit, etc.)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in allowed_modules:
                # pl.something - check if it's allowed
                if node.attr not in _ALLOWED_POLARS_ATTRS:
                    raise ValueError(
                        f"Unsafe Polars attribute: {node.attr}. "
                        f"Expression: {expr_str}"
                    )
    
    # Safe to evaluate - use restricted eval with only allowed modules
    safe_globals = {k: v for k, v in allowed_modules.items()}
    safe_globals["__builtins__"] = {}
    
    try:
        return eval(compile(tree, '<string>', 'eval'), safe_globals)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {expr_str}. Error: {e}")


# =========================================================================
# Day 25: Time-series feature generation
# =========================================================================

def _generate_timeseries_features(
    schema: dict,
    competition_brief: dict,
) -> list[FeatureCandidate]:
    """
    Time-series specific features. Called instead of Round 1 generic transforms
    when task_type == timeseries.

    Generates:
      - Lag features (lag_1, lag_2, lag_3, lag_7, lag_14, lag_28 for daily data)
      - Rolling statistics (rolling_mean_7, rolling_std_7, rolling_mean_28)
      - Seasonal decomposition indicators (day_of_week, month, quarter)
      - Trend features (days_since_start, row_index_normalised)

    All features are defined by name only — applied by _apply_timeseries_transforms().
    """
    candidates = []
    columns = schema.get("columns", [])

    # Find numeric columns that are candidates for lag/rolling features
    numeric_cols = [
        c["name"] for c in columns
        if _is_numeric(c) and not c.get("is_id") and not c.get("is_target")
    ]

    # Lag features — only for top 5 numeric columns by n_unique
    top_numerics = sorted(
        numeric_cols,
        key=lambda n: next((c.get("n_unique", 0) for c in columns if c["name"] == n), 0),
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
    date_col = _find_date_column(schema)
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


def _find_date_column(schema: dict) -> Optional[str]:
    """Find a date/datetime column in the schema."""
    for c in schema.get("columns", []):
        dtype = str(c.get("dtype", "")).lower()
        col_name = c.get("name", "")
        if any(kw in dtype for kw in ["date", "datetime", "time"]):
            return col_name
        if any(kw in col_name.lower() for kw in ["date", "time", "timestamp"]):
            return col_name
    return None


@timed_node
def run_feature_factory(state: ProfessorState) -> ProfessorState:
    """
    Feature Factory main node.
    Generates Rounds 1-5, applies AST transformations and Group aggregations, 
    evaluates via Null Importance and Wilcoxon CV, and solidifies survivors into the TabularPreprocessor.
    """
    import os
    import math
    import copy
    from pathlib import Path
    import polars as pl
    import numpy as np
    from core.preprocessor import TabularPreprocessor
    from tools.data_tools import write_json
    
    session_id = state["session_id"]
    output_dir = f"outputs/{session_id}"
    clean_path = state.get("clean_data_path", "")
    preprocessor_path = state.get("preprocessor_path", "")
    schema_path = state.get("schema_path", "")
    
    logger.info(f"[FeatureFactory] Starting — session: {session_id}")
    
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"clean_data_path missing: {clean_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"preprocessor_path missing: {preprocessor_path}")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema_path missing: {schema_path}")
        
    df = pl.read_parquet(clean_path)
    schema = json.loads(Path(schema_path).read_text())
    preprocessor = TabularPreprocessor.load(preprocessor_path)
    
    # ── Adapter: Convert modern flat schema into legacy hierarchical dicts ──
    target_col = schema.get("target_col", "")
    id_columns = schema.get("id_columns", [])
    flat_cols = schema.get("columns", [])
    if flat_cols and isinstance(flat_cols[0], str):
        dict_cols = []
        types_map = schema.get("types", {})
        n_unique_map = schema.get("n_unique", {})
        missing_map = schema.get("missing_rates", {})
        for c_str in flat_cols:
            dict_cols.append({
                "name": c_str,
                "dtype": types_map.get(c_str, "unknown"),
                "n_unique": n_unique_map.get(c_str, "?"),
                "missing_fraction": missing_map.get(c_str, 0.0),
                "is_target": c_str == target_col,
                "is_id": c_str in id_columns
            })
        schema["columns"] = dict_cols
    
    # Needs target for evaluation
    target_col = schema.get("target_col", "")
    y = df[target_col].to_numpy() if target_col in df.columns else None
    
    # Baseline dataframe (before any new features)
    X_base = preprocessor.transform(df)
    valid_cols = set(X_base.columns)

    # Day 25: Time-series routing — use lag/rolling features instead of generic Round 1
    task_type = state.get("task_type", "binary_classification")

    if task_type == "timeseries":
        logger.info("[feature_factory] Time-series mode: using lag/rolling/seasonal features.")
        round1_candidates = _generate_timeseries_features(schema, {})
        # Skip Round 3 aggregations and Round 4 target encoding for time-series
        round3_candidates = []
        round4_candidates = []
    else:
        round1_candidates = _generate_round1_features(schema)
        for c in round1_candidates:
            src = c.source_columns[0]
            if c.transform_type == "log":
                c.expression = f"(pl.col('{src}').cast(pl.Float64).fill_null(0.0) + 1.0).log()"
            elif c.transform_type == "sqrt":
                c.expression = f"pl.col('{src}').cast(pl.Float64).fill_null(0.0).sqrt()"
            elif c.transform_type == "missingness_flag":
                c.expression = f"pl.col('{src}').is_null().cast(pl.Int8)"
        round3_candidates = _generate_round3_aggregation_features(schema)
        round4_candidates = _generate_round4_target_encoding_candidates(schema)
            
    brief_path = state.get("competition_brief_path", "")
    competition_brief = {}
    if brief_path and os.path.exists(brief_path):
        competition_brief = json.loads(Path(brief_path).read_text())
        
    round2_candidates = _generate_round2_features(schema, competition_brief, state)
    # round3_candidates and round4_candidates already set above (empty for timeseries)
    round5a_candidates = _generate_round5_hypothesis_features(schema, competition_brief, state)
    
    schema_names = [c["name"] for c in schema.get("columns", []) if not c.get("is_id") and not c.get("is_target")]
    round5b_candidates = _generate_round5_interaction_features(schema, competition_brief, top_features_by_importance=schema_names[:50])
    
    all_candidates = round1_candidates + round2_candidates + round3_candidates + round4_candidates + round5a_candidates + round5b_candidates
    
    # AST Pipeline (Rounds 1, 2, 5)
    X_aug = X_base.clone()
    valid_candidates = []
    
    # Apply AST expressions one-at-a-time for fault tolerance
    for c in all_candidates:
        if c.round in (1, 2, 5) and c.expression:
            try:
                safe_ast = _rewrite_llm_expression(c.expression, valid_cols)
                # SECURITY FIX: Use safe evaluator instead of eval()
                expr_obj = _safe_eval_polars_expr(safe_ast, {"pl": pl, "np": np})
                # Test that the expression actually evaluates on the dataframe
                X_aug = X_aug.with_columns(expr_obj.alias(c.name))
                c.expression = safe_ast
                valid_candidates.append(c)
            except Exception as e:
                logger.warning(f"[FeatureFactory] Suppressed invalid AST round {c.round} feature {c.name}: {e}")
                c.verdict = "DROP"
        
    # Group Aggregation Pipeline (Round 3) - LEAKAGE FIX
    # DO NOT APPLY aggregations here - mark for CV-safe application in ml_optimizer
    c3_v = [c for c in round3_candidates if all(s in X_aug.columns for s in c.source_columns)]
    for c in c3_v:
        c.verdict = "PENDING_CV"  # Mark for CV-safe application
        valid_candidates.append(c)
    
    logger.info(f"[FeatureFactory] Round 3: {len(c3_v)} aggregation candidates marked for CV-safe application")
            
    # Target Encoding Pipeline (Round 4) - LEAKAGE FIX
    # DO NOT APPLY target encoding here - mark for CV-safe application in ml_optimizer
    c4_v = [c for c in round4_candidates if c.source_columns[0] in X_aug.columns]
    for c in c4_v:
        c.verdict = "PENDING_CV"  # Mark for CV-safe application
        valid_candidates.append(c)
    
    logger.info(f"[FeatureFactory] Round 4: {len(c4_v)} target encoding candidates marked for CV-safe application")
            
    # Budget Cap for R5 Interactions
    valid_candidates = _apply_interaction_budget_cap(valid_candidates, valid_candidates)
    aug_cols = {c.name for c in valid_candidates}
    final_aug_cols = [c for c in X_aug.columns if c in aug_cols]
    
    # Evaluation Gates
    survivors = set(final_aug_cols)
    if y is not None and len(survivors) > 0:
        survivor_list, temp_state = _apply_null_importance_filter(state, X_aug.select(final_aug_cols), y)
        state.update(temp_state)
        survivors = set(survivor_list)
        
    final_candidates = [c for c in valid_candidates if c.name in survivors]
    # Dynamically cap evaluation computation relative to dataset size 
    eval_cap = max(5, int(len(y) * 0.05)) if y is not None else len(valid_candidates)
    if len(final_candidates) > eval_cap:
        final_candidates = final_candidates[:eval_cap]
    
    # Wilcoxon Gate + Solidification
    added_features = []
    X_current = X_base.clone()
    
    for c in final_candidates:
        if y is None:
            c.verdict = "KEEP"
            added_features.append(c)
            X_current = X_current.hstack(X_aug.select([c.name]))
            continue
            
        passes = _evaluate_candidate_feature(state, X_current, X_current.hstack(X_aug.select([c.name])), y, c.name)
        if passes:
            c.verdict = "KEEP"
            added_features.append(c)
            X_current = X_current.hstack(X_aug.select([c.name]))
        else:
            c.verdict = "DROP"
            
    # Solidify Keepers into TabularPreprocessor
    for c in added_features:
        if c.round in (1, 2, 5):
            preprocessor.add_feature_expression(c.name, c.expression)
        elif c.round == 3:
            cat_col = c.source_columns[1]
            num_col = c.source_columns[0]
            fn_name = None
            for fn in ROUND3_AGG_FUNCTIONS:
                if f"_{fn}_by_" in c.name:
                    fn_name = fn
                    break
            if fn_name is None: continue
            
            agg_expr = {"mean": pl.col(num_col).mean(), "std": pl.col(num_col).std(), "min": pl.col(num_col).min(), "max": pl.col(num_col).max(), "count": pl.col(num_col).count()}.get(fn_name)
            if agg_expr is None: continue
            
            mapping_df = X_base.group_by(cat_col).agg(agg_expr.alias("val")).drop_nulls(cat_col)
            mapping_dict = dict(zip(mapping_df[cat_col].to_list(), mapping_df["val"].to_list()))
            default_val = float(mapping_df["val"].mean()) if len(mapping_df) > 0 else 0.0
            preprocessor.add_group_mapping(c.name, cat_col, mapping_dict, default_val)
            
        elif c.round == 4:
            col = c.source_columns[0]
            global_mean = float(np.mean(y)) if y is not None else 0.0
            smoothing = 30.0
            mapping_df = X_base.with_columns(pl.Series("y", y)).group_by(col).agg([
                pl.col("y").sum().alias("sum"), pl.col("y").count().alias("count")
            ]).drop_nulls(col)
            
            mapping_dict = {}
            for row in mapping_df.iter_rows(named=True):
                cat_val = row[col]
                count = row["count"]
                sm_y = row["sum"]
                sm_mean = sm_y / count if count > 0 else global_mean
                mapping_dict[cat_val] = float(((count * sm_mean) + (smoothing * global_mean)) / (count + smoothing))
            preprocessor.add_group_mapping(c.name, col, mapping_dict, global_mean)
            
    preprocessor.expected_columns = X_current.columns
    feature_parquet_path = f"{output_dir}/features.parquet"
    X_current.write_parquet(feature_parquet_path)
    preprocessor.save(preprocessor_path)

    manifest = _build_feature_manifest(all_candidates, schema)
    manifest_path = Path(f"{output_dir}/feature_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(f"[FeatureFactory] Wired {len(added_features)} robust tested features into Preprocessor across 5 Rounds.")

    from core.lineage import log_event
    log_event(
        session_id=session_id,
        agent="feature_factory",
        action="feature_factory_complete",
        keys_read=["clean_data_path", "preprocessor_path"],
        keys_written=["feature_data_path", "feature_manifest", "preprocessor_path", "feature_order"],
        values_changed={"added_features": len(added_features)},
    )

    return {
        **state,
        "feature_data_path": feature_parquet_path,
        "feature_manifest": manifest,
        "feature_candidates": added_features,
        "feature_order": list(X_current.columns),  # FIX: Write feature_order for pseudo_label_agent and submit
    }
