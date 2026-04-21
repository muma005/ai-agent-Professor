# agents/eda_agent.py

import os
import json
import logging
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from core.state import ProfessorState
from tools.data_tools import read_parquet, read_json
from core.lineage import log_event
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "eda_agent"

# ── Vector 1: Target distribution profiling ──────────────────────

def _analyze_target(df: pl.DataFrame, target_col: str) -> dict:
    """Profile the target column — distribution, skew, imbalance ratio."""
    if target_col not in df.columns:
        return {"skew": 0.0, "kurtosis": 0.0, "imbalance_ratio": 1.0,
                "is_multimodal": False, "recommended_transform": "none"}

    s = df[target_col].drop_nulls()
    n_unique = s.n_unique()
    dtype = s.dtype

    # Classification targets
    if dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Boolean) or n_unique <= 20:
        value_counts = s.value_counts().sort("count", descending=True)
        if len(value_counts) > 0:
            min_count = value_counts["count"].min()
            max_count = value_counts["count"].max()
            imbalance_ratio = float(min_count / max_count) if max_count > 0 else 1.0
        else:
            imbalance_ratio = 1.0
        return {
            "skew": 0.0, "kurtosis": 0.0,
            "imbalance_ratio": round(imbalance_ratio, 4),
            "n_classes": int(n_unique),
            "is_multimodal": False,
            "recommended_transform": "none",
        }

    # Continuous targets
    if len(s) < 3:
        return {"skew": 0.0, "kurtosis": 0.0, "imbalance_ratio": 1.0,
                "is_multimodal": False, "recommended_transform": "none"}

    skew = float(s.skew() or 0.0)
    kurt = float(s.kurtosis() or 0.0)

    transform = "none"
    if skew > 1.5:
        transform = "log"
    elif skew < -1.5:
        transform = "boxcox"

    return {
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "imbalance_ratio": 1.0,
        "is_multimodal": False,
        "recommended_transform": transform,
    }


# ── Vector 2: ID column validation ───────────────────────────────

def _validate_id_columns(df: pl.DataFrame, id_columns: list, target_col: str) -> dict:
    """Confirm data_engineer's ID detection and flag surprises."""
    n_rows = len(df)
    validated = []
    suspicious = []

    for col in df.columns:
        if col == target_col or col in id_columns:
            continue
        # Detect columns that LOOK like IDs but weren't flagged
        if df[col].n_unique() == n_rows and n_rows > 1:
            dtype = df[col].dtype
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Utf8, pl.String):
                suspicious.append(col)

    for col in id_columns:
        if col in df.columns:
            validated.append({
                "column": col,
                "n_unique": int(df[col].n_unique()),
                "is_unique": df[col].n_unique() == n_rows,
            })

    return {
        "confirmed_ids": [v["column"] for v in validated],
        "suspicious_untagged": suspicious,
        "details": validated,
    }


# ── Vector 3: Missing value profiling ────────────────────────────

def _analyze_missing(df: pl.DataFrame, target_col: str) -> list:
    """Detailed missing value analysis with imputation recommendations."""
    n_rows = len(df)
    results = []
    for col in df.columns:
        if col == target_col:
            continue
        null_count = df[col].null_count()
        null_rate = null_count / n_rows if n_rows > 0 else 0.0
        if null_count > 0:
            strategy = "median" if df[col].dtype in pl.NUMERIC_DTYPES else "mode"
            if null_rate > 0.5:
                strategy = "drop_column"
            results.append({
                "column": col,
                "null_count": int(null_count),
                "null_rate": round(float(null_rate), 4),
                "strategy": strategy,
            })
    return results


# ── Vector 4: Zero-variance detection ────────────────────────────

def _detect_zero_variance(df: pl.DataFrame, target_col: str) -> list:
    """Find columns with only 1 unique value (zero variance)."""
    drops = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].n_unique() <= 1:
            drops.append(col)
    return drops


# ── Vector 5: Cardinality profiling ──────────────────────────────

def _analyze_cardinality(df: pl.DataFrame, target_col: str) -> list:
    """Profile cardinality of all columns."""
    n_rows = len(df)
    results = []
    for col in df.columns:
        if col == target_col:
            continue
        n_unique = df[col].n_unique()
        ratio = n_unique / n_rows if n_rows > 0 else 0.0
        category = "constant" if n_unique <= 1 else \
                   "binary" if n_unique == 2 else \
                   "low" if n_unique <= 10 else \
                   "medium" if n_unique <= 50 else \
                   "high" if n_unique <= 1000 else "very_high"
        results.append({
            "column": col,
            "n_unique": int(n_unique),
            "uniqueness_ratio": round(float(ratio), 4),
            "category": category,
        })
    return results


# ── Vector 6: Feature-feature collinearity ───────────────────────

def _detect_collinearity(df: pl.DataFrame, target_col: str, threshold: float = 0.95) -> list:
    """Find pairs of features with Spearman correlation ≥ threshold."""
    num_cols = [c for c in df.columns
                if c != target_col and df[c].dtype in pl.NUMERIC_DTYPES]

    # Limit to top 100 numeric columns to avoid O(n²) explosion
    num_cols = num_cols[:100]
    pairs = []

    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col_a, col_b = num_cols[i], num_cols[j]
            try:
                pair_df = df.select([col_a, col_b]).drop_nulls()
                if len(pair_df) < 5:
                    continue
                corr = pair_df.select(
                    pl.corr(col_a, col_b, method="spearman")
                )[0, 0]
                if corr is not None and abs(float(corr)) >= threshold:
                    pairs.append({
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "spearman": round(float(corr), 4),
                    })
            except Exception:
                continue

    return pairs


# ── Vector 7: Leakage detection ──────────────────────────────────

def _detect_leakage(df: pl.DataFrame, target_col: str) -> Tuple[List[Dict], List[str]]:
    """Find features suspiciously correlated with target (>0.95)."""
    leakage = []
    drop_candidates = []
    num_cols = [c for c in df.columns if c != target_col and df[c].dtype in pl.NUMERIC_DTYPES]

    if target_col not in df.columns or df[target_col].dtype not in pl.NUMERIC_DTYPES:
        return leakage, drop_candidates

    for col in num_cols:
        try:
            pair = df.select([col, target_col]).drop_nulls()
            if len(pair) < 3:
                continue
            corr = pair.select(pl.corr(col, target_col, method="pearson"))[0, 0]
            corr = float(corr) if corr is not None else 0.0
        except Exception:
            corr = 0.0

        if str(corr).lower() in ("nan", "inf", "-inf"):
            corr = 0.0

        abs_c = abs(corr)
        if abs_c > 0.95:
            verdict = "FLAG"
            drop_candidates.append(col)
        elif abs_c > 0.80:
            verdict = "WATCH"
        else:
            verdict = "OK"

        if abs_c > 0.3:  # Only log meaningful correlations
            leakage.append({
                "feature": col,
                "target_correlation": round(corr, 4),
                "verdict": verdict,
            })

    return leakage, drop_candidates


# ── Vector 8: Outlier profiling ──────────────────────────────────

def _analyze_outliers(df: pl.DataFrame) -> list:
    """IQR-based outlier detection with strategy recommendations."""
    outlier_profile = []
    num_cols = [c for c in df.columns if df[c].dtype in pl.NUMERIC_DTYPES]
    total_rows = len(df)
    if total_rows == 0:
        return outlier_profile

    for col in num_cols:
        s = df[col].drop_nulls()
        if len(s) == 0:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        if q1 is None or q3 is None:
            continue
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = len(s.filter((s < lower) | (s > upper)))
        pct = (n_outliers / total_rows) * 100.0

        strategy = "keep" if pct < 1.0 else "winsorize" if pct <= 5.0 else \
                   "cap" if pct <= 10.0 else "remove"

        outlier_profile.append({
            "column": col, "strategy": strategy,
            "n_outliers": int(n_outliers), "pct_outliers": float(round(pct, 2)),
        })
    return outlier_profile


# ── Vector 9: Duplicate row detection ────────────────────────────

def _analyze_duplicates(df: pl.DataFrame) -> dict:
    """Count exact duplicate rows."""
    exact_count = len(df) - len(df.unique())
    return {"exact_count": exact_count, "pct": round(exact_count / max(len(df), 1) * 100, 2)}


# ── Vector 10: Temporal feature detection ────────────────────────

def _analyze_temporal(df: pl.DataFrame) -> dict:
    """Detect date/time columns by dtype and name patterns."""
    date_cols = []
    time_keywords = ["date", "time", "timestamp", "year", "month", "week"]
    for c in df.columns:
        if df[c].dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
            date_cols.append(c)
        elif any(kw in c.lower() for kw in time_keywords):
            date_cols.append(c)
    return {
        "has_dates": len(date_cols) > 0,
        "date_columns": date_cols,
        "train_test_drift_risk": len(date_cols) > 0,
    }


# ── Vector 11: Mixed-type detection ──────────────────────────────

def _detect_mixed_types(df: pl.DataFrame) -> list:
    """Find columns where dtype might not match content (e.g., numbers stored as strings)."""
    mixed = []
    for col in df.columns:
        if df[col].dtype in (pl.Utf8, pl.String):
            sample = df[col].drop_nulls().head(100)
            n_numeric = sum(1 for v in sample.to_list() if _is_numeric_str(v))
            if n_numeric > len(sample) * 0.8 and len(sample) > 0:
                mixed.append({"column": col, "likely_type": "numeric",
                             "pct_numeric": round(n_numeric / len(sample) * 100, 1)})
    return mixed


def _is_numeric_str(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ── Vector 12: High-missing drop recommendation ─────────────────

def _recommend_drops_from_missing(missing_profile: list, threshold: float = 0.5) -> list:
    """Columns with >50% missing are drop candidates."""
    return [m["column"] for m in missing_profile if m["null_rate"] > threshold]


# ── Main agent function ──────────────────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_eda_agent(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: EDA Agent — 12-vector comprehensive analysis.
    """
    # 1. Skip logic from config
    config = state.get("config")
    if config and config.agents.skip_eda:
        logger.info(f"[{AGENT_NAME}] Skipping per config.")
        return state
    
    session_id = state.get("session_id", "default")
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{AGENT_NAME}] Starting — session: {session_id}")

    clean_path = state.get("clean_data_path", "")
    if not clean_path or not os.path.exists(clean_path):
        raise ValueError(f"[{AGENT_NAME}] clean_data_path missing or not valid: {clean_path}")

    df = read_parquet(clean_path)
    
    # 2. Read from SCHEMA AUTHORITY
    target_col = state.get("target_col", "")
    id_columns = state.get("id_columns", [])
    if not target_col:
        raise ValueError(f"[{AGENT_NAME}] target_col not set in state.")

    # 3. Run all 12 analysis vectors (Local only for now, Lightning stubbed)
    target_dist    = _analyze_target(df, target_col)
    id_validation  = _validate_id_columns(df, id_columns, target_col)
    missing        = _analyze_missing(df, target_col)
    zero_var_drops = _detect_zero_variance(df, target_col)
    cardinality    = _analyze_cardinality(df, target_col)
    collinear      = _detect_collinearity(df, target_col)
    leakage, leak_drops = _detect_leakage(df, target_col)
    outliers       = _analyze_outliers(df)
    dupes          = _analyze_duplicates(df)
    temporal       = _analyze_temporal(df)
    mixed_types    = _detect_mixed_types(df)
    high_miss_drops = _recommend_drops_from_missing(missing)

    # 4. Build comprehensive drop manifest
    all_drops = set()
    all_drops.update(zero_var_drops)
    all_drops.update(leak_drops)
    all_drops.update(high_miss_drops)
    
    # Collinear pairs: drop the one with lower target correlation
    for pair in collinear:
        a, b = pair["feature_a"], pair["feature_b"]
        a_corr = abs(next((l["target_correlation"] for l in leakage if l["feature"] == a), 0.0))
        b_corr = abs(next((l["target_correlation"] for l in leakage if l["feature"] == b), 0.0))
        all_drops.add(b if a_corr >= b_corr else a)
        
    # Never drop target or IDs
    all_drops.discard(target_col)
    for id_col in id_columns:
        all_drops.discard(id_col)

    dropped_features = sorted(all_drops)

    # 5. Read Intel Brief to augment leaks
    brief = state.get("competition_brief", {})
    for leak in brief.get("known_leaks", []):
        if leak in df.columns and leak not in dropped_features:
            dropped_features.append(leak)
            leakage.append({"feature": leak, "target_correlation": 1.0, "verdict": "FLAG"})

    # 6. Build report
    summary_text = (
        f"Analyzed {len(df)} rows. Target='{target_col}'. "
        f"Drops: {len(dropped_features)} "
        f"(zero_var={len(zero_var_drops)}, leakage={len(leak_drops)}, "
        f"high_missing={len(high_miss_drops)}, collinear={len(collinear)})."
    )

    report = {
        "target_distribution":    target_dist,
        "id_validation":          id_validation,
        "missing_profile":        missing,
        "zero_variance_features": zero_var_drops,
        "cardinality_profile":    cardinality,
        "collinear_pairs":        collinear,
        "leakage_fingerprint":    leakage,
        "outlier_profile":        outliers,
        "duplicate_analysis":     dupes,
        "temporal_profile":       temporal,
        "mixed_types":            mixed_types,
        "drop_manifest":          dropped_features,
        "summary":                summary_text,
    }

    report_path = output_dir / "eda_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # 7. Update State
    updates = {
        "eda_report_path":  str(report_path),
        "eda_report":       report,
        "dropped_features": dropped_features,
    }

    log_event(
        session_id=session_id,
        agent=AGENT_NAME,
        action="eda_completed",
        keys_read=["clean_data_path", "target_col", "id_columns"],
        keys_written=["eda_report_path", "eda_report", "dropped_features"],
        values_changed={"dropped_features": dropped_features},
    )

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
