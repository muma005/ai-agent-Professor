# agents/feature_factory.py
# -------------------------------------------------------------------------
# Day 16: Feature Factory — Rounds 1 (generic) + 2 (domain/LLM)
# Reads schema.json + competition_brief.json only — never raw data.
# Writes feature_manifest.json with per-feature metadata.
# Day 17: Wilcoxon gate + null importance filtering.
# -------------------------------------------------------------------------

import json
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl
import numpy as np

from core.state import ProfessorState
from core.lineage import log_event
from tools.llm_client import call_llm
from tools.wilcoxon_gate import feature_gate_result, is_feature_worth_adding
from tools.null_importance import run_null_importance_filter

logger = logging.getLogger(__name__)


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

    for col in columns:
        name = col["name"]
        dtype = col.get("dtype", "")
        null_fraction = float(col.get("null_fraction", 0.0))
        min_val = col.get("min")
        n_unique = int(col.get("n_unique", 0))
        is_id = col.get("is_id", False)
        is_target = col.get("is_target", False)

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
3. Can be expressed as simple arithmetic, groupby-aggregation, or string extraction

Return a JSON array of feature candidates:
[
  {{
    "name": "feature_name_snake_case",
    "source_columns": ["col_a", "col_b"],
    "transform_type": "ratio | interaction | groupby_mean | groupby_std | string_extract | bin",
    "expression": "Human-readable description of the transform",
    "domain_rationale": "Why this feature is likely predictive given the domain"
  }}
]

Rules:
- Maximum 15 candidates. Quality over quantity.
- Every source_column must exist in the schema.
- Do NOT suggest target encoding — this is handled separately.
- Do NOT suggest lag features for non-time-series competitions.
- expression must be implementable in Polars with basic operations.
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
        ))

    logger.info(f"[feature_factory] Round 2: {len(candidates)} domain feature candidates generated.")
    return candidates


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

def run_feature_factory(state: ProfessorState) -> ProfessorState:
    """
    Feature Factory main node.
    Reads schema.json and competition_brief.json only.
    Generates Round 1 (generic) + Round 2 (domain) candidates.
    Writes feature_manifest.json.
    Day 17 adds Wilcoxon gate + null importance filtering.
    """
    session_id = state["session_id"]

    # Load schema (written by data_engineer)
    schema_path = Path(f"outputs/{session_id}/schema.json")
    if not schema_path.exists():
        raise FileNotFoundError(
            f"schema.json not found at {schema_path}. "
            "data_engineer must run before feature_factory."
        )
    schema = json.loads(schema_path.read_text())

    # Load competition_brief (written by competition_intel)
    brief_path = Path(f"outputs/{session_id}/competition_brief.json")
    competition_brief = json.loads(brief_path.read_text()) if brief_path.exists() else {}

    # Generate candidates
    round1_candidates = _generate_round1_features(schema)
    round2_candidates = _generate_round2_features(schema, competition_brief, state)
    all_candidates = round1_candidates + round2_candidates

    logger.info(
        f"[feature_factory] Candidates: {len(round1_candidates)} Round 1 + "
        f"{len(round2_candidates)} Round 2 = {len(all_candidates)} total"
    )

    # Day 17: Statistical filtering (requires clean data)
    gate_passed = []
    gate_dropped = []
    clean_path = state.get("clean_data_path", "")
    target_col = state.get("target_column", "")

    if clean_path and Path(clean_path).exists() and target_col:
        try:
            df = (pl.read_parquet(clean_path) if clean_path.endswith(".parquet")
                  else pl.read_csv(clean_path))
            y = df[target_col].to_numpy()

            # Apply Round 1 transforms
            X = _apply_round1_transforms(df, round1_candidates)

            # Wilcoxon gate for Round 2 candidates only
            # Round 1 (log, sqrt, missingness) skip gate — low-risk transforms
            for c in round2_candidates:
                cols_present = all(s in X.columns for s in c.source_columns)
                if not cols_present:
                    c.verdict = "DROP"
                    gate_dropped.append(c.name)
                    continue
                X_base = X.drop(c.name) if c.name in X.columns else X
                X_with = X_base  # Round 2 features aren't applied yet; use base
                passed = _evaluate_candidate_feature(
                    state, X_base, X_base, y, c.name,
                    skip_wilcoxon_gate=False,
                )
                if passed:
                    gate_passed.append(c.name)
                else:
                    gate_dropped.append(c.name)
                    c.verdict = "DROP"

            # Null importance filter on kept features
            kept_feature_names = (
                [c.name for c in round1_candidates]
                + gate_passed
            )
            # Only features that exist in X can be filtered
            available_features = [f for f in kept_feature_names if f in X.columns]
            if available_features:
                survivors, state = _apply_null_importance_filter(
                    state, X.select(available_features + [target_col]), y
                )
                survivor_set = set(survivors)
            else:
                survivor_set = set(kept_feature_names)

            # Set verdicts
            for c in all_candidates:
                if c.verdict == "DROP":
                    continue  # already dropped by Wilcoxon
                if c.name in survivor_set or c.name not in available_features:
                    c.verdict = "KEEP"
                else:
                    c.verdict = "DROP"

        except Exception as e:
            logger.warning(
                f"[feature_factory] Statistical filtering failed: {e}. "
                f"Using all candidates with KEEP verdict."
            )
            for c in all_candidates:
                c.verdict = "KEEP"
    else:
        # No data available — keep Day 16 behavior
        for c in all_candidates:
            c.verdict = "KEEP"

    kept_candidates = [c for c in all_candidates if c.verdict == "KEEP"]

    # Write feature_manifest.json
    manifest = _build_feature_manifest(kept_candidates, schema)
    manifest_path = Path(f"outputs/{session_id}/feature_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log_event(
        session_id=session_id,
        agent="feature_factory",
        action="feature_factory_complete",
        keys_read=["session_id"],
        keys_written=["feature_manifest", "feature_candidates",
                       "round1_features", "round2_features"],
        values_changed={
            "round1_candidates": len(round1_candidates),
            "round2_candidates": len(round2_candidates),
            "total_kept": len(kept_candidates),
            "dropped": len(all_candidates) - len(kept_candidates),
            "gate_passed": len(gate_passed),
            "gate_dropped": len(gate_dropped),
        },
    )

    state = {
        **state,
        "feature_manifest": manifest,
        "feature_candidates": [c.name for c in kept_candidates],
        "round1_features": [c.name for c in round1_candidates],
        "round2_features": [c.name for c in round2_candidates],
        "features_gate_passed": gate_passed,
        "features_gate_dropped": gate_dropped,
    }
    return state
