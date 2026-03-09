# agents/eda_agent.py

import os
import json
import polars as pl
from typing import Dict, Any, List, Optional
from core.state import ProfessorState
from tools.data_tools import read_parquet, read_json
from core.lineage import log_event
from guards.agent_retry import with_agent_retry


def _analyze_target(df: pl.DataFrame, target_col: str) -> dict:
    if target_col not in df.columns:
        return {
            "skew": 0.0,
            "kurtosis": 0.0,
            "is_multimodal": False,
            "recommended_transform": "none"
        }
        
    dtype = df[target_col].dtype
    if dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Boolean):
        return {
            "skew": 0.0,
            "kurtosis": 0.0,
            "is_multimodal": False,
            "recommended_transform": "none"
        }
    
    # Check if actually continuous vs just binary encoded as int
    s = df[target_col].drop_nulls()
    valid_count = len(s)
    if valid_count < 3 or s.n_unique() <= 2:
        return {
            "skew": 0.0,
            "kurtosis": 0.0,
            "is_multimodal": False,
            "recommended_transform": "none"
        }
    
    # It's continuous
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
        "is_multimodal": False,
        "recommended_transform": transform
    }


def _analyze_correlations_and_leakage(df: pl.DataFrame, target_col: str) -> tuple[list, list, list]:
    correlations = []
    leakage = []
    drop_candidates = []
    
    num_cols = [c for c in df.columns if df[c].dtype in pl.NUMERIC_DTYPES]
    
    # 1. Zero variance check
    for c in num_cols:
        if df[c].n_unique() <= 1 and c != target_col:
            drop_candidates.append(c)

    if target_col not in df.columns or df[target_col].dtype not in pl.NUMERIC_DTYPES:
        return correlations, leakage, drop_candidates

    target_s = df[target_col].drop_nulls()
    if target_s.n_unique() <= 1:
        return correlations, leakage, drop_candidates

    # 2. Compute correlations
    for col in num_cols:
        if col == target_col or col in drop_candidates: 
            continue
            
        try:
            # We must drop nulls pairwise
            pair = df.select([col, target_col]).drop_nulls()
            if len(pair) < 3:
                corr = 0.0
            else:
                corr_val = pair.select(pl.corr(col, target_col, method="pearson"))[0, 0]
                corr = float(corr_val) if corr_val is not None else 0.0
        except Exception:
            corr = 0.0
            
        if str(corr).lower() in ("nan", "inf", "-inf"):
            corr = 0.0
            
        abs_c = abs(corr)
        rel_type = "linear" if abs_c > 0.3 else ("none" if abs_c < 0.1 else "nonlinear")
        
        correlations.append({
            "feature": col,
            "correlation": round(corr, 4),
            "relationship_type": rel_type
        })
        
        verdict = "OK"
        if abs_c > 0.95:
            verdict = "FLAG"
            drop_candidates.append(col)
        elif abs_c > 0.80:
            verdict = "WATCH"
            
        leakage.append({
            "feature": col,
            "target_correlation": round(corr, 4),
            "verdict": verdict
        })
        
    return correlations, leakage, drop_candidates


def _analyze_outliers(df: pl.DataFrame) -> list:
    outlier_profile = []
    num_cols = [c for c in df.columns if df[c].dtype in pl.NUMERIC_DTYPES]
    total_rows = len(df)
    
    if total_rows == 0:
        return outlier_profile

    for col in num_cols:
        s = df[col].drop_nulls()
        n_valid = len(s)
        if n_valid == 0: 
            continue
        
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        if q1 is None or q3 is None:
            continue
            
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        n_outliers = len(s.filter((s < lower) | (s > upper)))
        pct = (n_outliers / total_rows) * 100.0 if total_rows > 0 else 0.0
        
        if pct < 1.0:
            strategy = "keep"
        elif pct <= 5.0:
            strategy = "winsorize"
        elif pct <= 10.0:
            strategy = "cap"
        else:
            strategy = "remove"
            
        outlier_profile.append({
            "column": col,
            "strategy": strategy,
            "n_outliers": int(n_outliers),
            "pct_outliers": float(round(pct, 2))
        })
            
    return outlier_profile


def _analyze_duplicates(df: pl.DataFrame, target_col: str, schema: dict) -> dict:
    exact_count = len(df) - len(df.unique())
    near_dup = 0 # Simplified for v0
    
    # Find group columns that could be IDs
    id_keywords = ["id", "patient", "user", "customer", "store", "site", "group"]
    possible_ids = [c for c in df.columns if any(kw in c.lower() for kw in id_keywords)]
    
    id_conflicts = []
    id_conflict_count = 0
    
    if target_col in df.columns and possible_ids:
        for gid in possible_ids:
            # Check if this ID appears with DIFFERENT targets
            check_df = df.group_by(gid).agg(pl.col(target_col).n_unique().alias("nunique"))
            n_conflicts = len(check_df.filter(pl.col("nunique") > 1))
            
            if n_conflicts > 0:
                id_conflicts.append(gid)
                id_conflict_count += n_conflicts

    return {
        "exact_count": exact_count,
        "near_duplicate_count": near_dup,
        "id_conflict_count": id_conflict_count,
        "id_conflict_columns": id_conflicts
    }


def _analyze_temporal(df: pl.DataFrame) -> dict:
    date_cols = []
    time_keywords = ["date", "time", "timestamp", "year", "month", "week"]
    
    for c in df.columns:
        if df[c].dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration) or any(kw in c.lower() for kw in time_keywords):
            date_cols.append(c)
                
    return {
        "has_dates": len(date_cols) > 0,
        "date_columns": date_cols,
        "seasonality_detected": False,
        "train_test_drift_risk": len(date_cols) > 0
    }


@with_agent_retry("EDAAgent")
def run_eda_agent(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: EDA Agent.
    
    Reads:  state["clean_data_path"]
            state["schema_path"]
            state["competition_brief_path"] (optional)
    Writes: eda_report.json
            state["eda_report_path"]
            state["eda_report"]
    """
    session_id = state["session_id"]
    output_dir = f"outputs/{session_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[EDAAgent] Starting — session: {session_id}")
    
    if not state.get("clean_data_path") or not os.path.exists(state["clean_data_path"]):
        raise ValueError("[EDAAgent] clean_data_path missing or not valid.")
        
    df = read_parquet(state["clean_data_path"])
    schema = read_json(state["schema_path"]) if state.get("schema_path") else {}
    
    target_col = state.get("target_col") or schema.get("target_col")
    if not target_col:
        cols = schema.get("columns", df.columns)
        target_col = cols[-1] if cols else ""
        
    # Analysis
    target_dist = _analyze_target(df, target_col)
    corrs, leakage, drops = _analyze_correlations_and_leakage(df, target_col)
    outliers = _analyze_outliers(df)
    dupes = _analyze_duplicates(df, target_col, schema)
    temporal = _analyze_temporal(df)
    
    # Read Intel Brief to augment leaks
    brief_path = state.get("competition_brief_path", "")
    if brief_path and os.path.exists(brief_path):
        brief = read_json(brief_path)
        known_leaks = brief.get("known_leaks", [])
        for leak in known_leaks:
            if leak in df.columns and leak not in drops:
                drops.append(leak)
                leakage.append({
                    "feature": leak,
                    "target_correlation": 1.0, 
                    "verdict": "FLAG"
                })

    summary_text = (
        f"Analyzed {len(df)} rows. Target={target_col}. "
        f"Found {len(drops)} drop candidates due to leakage or zero variance. "
        f"{dupes['id_conflict_count']} ID conflicts detected. "
        f"{'Has temporal drift risk' if temporal['has_dates'] else 'No dates detected'}."
    )

    report = {
        "target_distribution": target_dist,
        "feature_correlations": corrs,
        "outlier_profile": outliers,
        "duplicate_analysis": dupes,
        "temporal_profile": temporal,
        "leakage_fingerprint": leakage,
        "drop_candidates": list(set(drops)),
        "summary": summary_text
    }
    
    report_path = f"{output_dir}/eda_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    n_flags = len([x for x in leakage if x["verdict"] == "FLAG"])
    print(f"[EDAAgent] EDA complete. Flags: {n_flags} | Drops: {len(drops)}")
    
    log_event(
        session_id=session_id,
        agent="eda_agent",
        action="eda_completed",
        keys_read=["clean_data_path"],
        keys_written=["eda_report_path", "eda_report"],
        values_changed={},
    )
    
    return {
        **state,
        "eda_report_path": report_path,
        "eda_report": report
    }
