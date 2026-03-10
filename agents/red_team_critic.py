# agents/red_team_critic.py
# -------------------------------------------------------------------------
# Day 10 — Red Team Critic: 6-vector quality gate
# Catches: target leakage, ID ordering leakage, train/test drift,
#          preprocessing leakage, majority-class-only models, temporal leakage
# -------------------------------------------------------------------------

import os
import re
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional

import polars as pl
import numpy as np

from core.state import ProfessorState
from core.lineage import log_event
from guards.circuit_breaker import get_escalation_level, handle_escalation, reset_failure_count

logger       = logging.getLogger(__name__)
AGENT_NAME   = "red_team_critic"
MAX_ATTEMPTS = 3

_SEVERITY_ORDER = {"OK": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


# =========================================================================
# VECTOR 1A — Shuffled Target Test
# =========================================================================

def _check_shuffled_target(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    target_type: str,
) -> dict:
    """
    Trains a simple model on shuffled targets.
    If AUC is meaningfully above 0.5, leakage is present.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    y_shuffled = y_train.sample(fraction=1.0, shuffle=True, seed=42).to_numpy()

    # Select only numeric columns
    numeric_dtypes = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                      pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    numeric_cols = [c for c in X_train.columns if X_train[c].dtype in numeric_dtypes]
    if not numeric_cols:
        return {"verdict": "OK", "auc_shuffled": None, "note": "No numeric features to test"}

    X_np = X_train.select(numeric_cols).fill_null(0).to_numpy()

    if target_type in ("binary", "multiclass"):
        model  = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42, n_jobs=-1)
        try:
            scores = cross_val_score(model, X_np, y_shuffled, cv=3, scoring="roc_auc")
        except ValueError:
            # fallback for multiclass without proper AUC support
            scores = cross_val_score(model, X_np, y_shuffled, cv=3, scoring="accuracy")
    else:
        model  = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X_np, y_shuffled, cv=3, scoring="r2")

    mean_score = float(np.mean(scores))
    threshold  = 0.55 if target_type in ("binary", "multiclass") else 0.10

    if target_type in ("binary", "multiclass") and mean_score > threshold:
        return {
            "verdict":      "CRITICAL",
            "auc_shuffled": round(mean_score, 4),
            "threshold":    threshold,
            "evidence":     f"Model trained on shuffled targets achieved AUC {mean_score:.4f} > {threshold}. Leakage confirmed.",
            "action":       "Inspect features for any direct or indirect encoding of the target. Remove suspect features and retrain.",
            "replan_instructions": {
                "remove_features": [],
                "rerun_nodes": ["feature_factory", "ml_optimizer"],
            },
        }

    return {"verdict": "OK", "auc_shuffled": round(mean_score, 4)}


# =========================================================================
# VECTOR 1B — ID-Only Model Test
# =========================================================================

def _check_id_only_model(
    df: pl.DataFrame,
    target_col: str,
    target_type: str,
    schema: dict,
) -> dict:
    """
    Trains a model using only ID/index columns.
    Meaningful AUC = data ordering encodes the target.
    """
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    id_keywords = ["id", "index", "row", "num", "_no", "number"]
    id_cols = [
        col for col in df.columns
        if col != target_col
        and any(kw in col.lower() for kw in id_keywords)
    ]

    if not id_cols:
        return {"verdict": "OK", "note": "No ID columns detected"}

    X_id = df.select(id_cols).with_columns([
        pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in id_cols
    ]).to_numpy()

    y = df[target_col].to_numpy()

    if target_type in ("binary", "multiclass"):
        model  = GradientBoostingClassifier(n_estimators=20, max_depth=2, random_state=42)
        try:
            scores = cross_val_score(model, X_id, y, cv=3, scoring="roc_auc")
        except ValueError:
            scores = cross_val_score(model, X_id, y, cv=3, scoring="accuracy")
    else:
        model  = GradientBoostingRegressor(n_estimators=20, max_depth=2, random_state=42)
        scores = cross_val_score(model, X_id, y, cv=3, scoring="r2")

    mean_score = float(np.mean(scores))
    threshold  = 0.65

    if target_type in ("binary", "multiclass") and mean_score > threshold:
        return {
            "verdict":    "CRITICAL",
            "auc_id_only": round(mean_score, 4),
            "id_columns": id_cols,
            "evidence":   f"ID-only model AUC {mean_score:.4f} > {threshold}. Data is sorted by target or ID encodes ordering.",
            "action":     f"Sort training data randomly before splitting. Drop leaking ID columns: {id_cols}.",
            "replan_instructions": {
                "remove_features": id_cols,
                "rerun_nodes": ["data_engineer", "ml_optimizer"],
            },
        }

    return {"verdict": "OK", "auc_id_only": round(mean_score, 4), "id_columns_tested": id_cols}


# =========================================================================
# VECTOR 1C — Adversarial Train/Test Classifier
# =========================================================================

def _check_adversarial_classifier(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_col: str,
) -> dict:
    """
    Binary classifier: train row = 0, test row = 1.
    AUC > 0.6 = feature distributions differ between train and test.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    if test_df is None or len(test_df) == 0:
        return {"verdict": "OK", "note": "No test data available for adversarial check"}

    numeric_dtypes = (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
    numeric_cols = [
        c for c in train_df.columns
        if c != target_col
        and train_df[c].dtype in numeric_dtypes
    ]
    if not numeric_cols:
        return {"verdict": "OK", "note": "No shared numeric columns for adversarial test"}

    shared_cols = [c for c in numeric_cols if c in test_df.columns]
    if not shared_cols:
        return {"verdict": "OK", "note": "No shared numeric columns between train and test"}

    train_X = train_df.select(shared_cols).fill_null(0).to_numpy()
    test_X  = test_df.select(shared_cols).fill_null(0).to_numpy()

    # Subsample for speed -- 5k rows max per split
    n_train = min(5000, len(train_X))
    n_test  = min(5000, len(test_X))
    rng     = np.random.default_rng(42)
    train_X = train_X[rng.choice(len(train_X), n_train, replace=False)]
    test_X  = test_X[rng.choice(len(test_X),  n_test,  replace=False)]

    X = np.vstack([train_X, test_X])
    y = np.array([0] * n_train + [1] * n_test)

    model  = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    auc    = float(np.mean(scores))

    if auc > 0.6:
        model.fit(X, y)
        importances = dict(zip(shared_cols, model.feature_importances_))
        top_drift   = sorted(importances, key=importances.get, reverse=True)[:5]
        severity    = "HIGH" if auc < 0.75 else "CRITICAL"

        return {
            "verdict":             severity,
            "adversarial_auc":     round(auc, 4),
            "top_drift_features":  top_drift,
            "evidence":            f"Train/test classifier AUC {auc:.4f} > 0.60. Features {top_drift} have different distributions in train vs test.",
            "action":              f"Inspect feature distributions for {top_drift}. Consider removing or transforming these features.",
            "replan_instructions": {
                "remove_features": top_drift,
                "rerun_nodes": ["feature_factory", "ml_optimizer"],
            },
        }

    return {"verdict": "OK", "adversarial_auc": round(auc, 4)}


# =========================================================================
# VECTOR 1D — Preprocessing Leakage Code Audit (GAP 2)
# =========================================================================

_LEAKAGE_PATTERNS = [
    (
        re.compile(r"fit_transform\s*\(\s*(?:X|train|df|data)\b"),
        "fit_transform called on full dataset (variable name suggests pre-split data)",
    ),
    (
        re.compile(r"(?:SimpleImputer|KNNImputer|IterativeImputer).*\.fit\s*\(\s*(?:X|train|data)\b"),
        "Imputer fitted on full dataset before CV",
    ),
    (
        re.compile(r"(?:TargetEncoder|OrdinalEncoder|LabelEncoder).*\.fit\s*\(\s*(?:X|train|data)\b"),
        "Encoder fitted on full dataset before CV",
    ),
    (
        re.compile(r"PCA.*\.fit(?:_transform)?\s*\(\s*(?:X|train|data)\b"),
        "PCA fitted on full dataset before CV",
    ),
]


def _check_preprocessing_leakage(data_engineer_code: str) -> dict:
    """
    Static analysis of Data Engineer generated code.
    Looks for preprocessing steps fitted on full dataset before CV split.
    """
    if not data_engineer_code or len(data_engineer_code) < 10:
        return {"verdict": "OK", "note": "No Data Engineer code available to audit"}

    findings = []
    lines = data_engineer_code.split("\n")
    split_line = None

    # Find where the actual CV fold loop starts (not the import line)
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip import lines — only match actual usage
        if stripped.startswith(("import ", "from ")):
            continue
        if any(kw in stripped for kw in ["train_test_split(", "KFold(", "StratifiedKFold(", "GroupKFold(", ".split("]):
            split_line = i
            break

    for pattern, description in _LEAKAGE_PATTERNS:
        matches = list(pattern.finditer(data_engineer_code))
        for match in matches:
            match_line = data_engineer_code[:match.start()].count("\n")
            # Only flag if this appears BEFORE the split line (or split_line not found)
            if split_line is None or match_line < split_line:
                findings.append({
                    "line":        match_line + 1,
                    "pattern":     description,
                    "code_snippet": lines[max(0, match_line)][:120].strip(),
                })

    if findings:
        return {
            "verdict":  "CRITICAL",
            "findings": findings,
            "evidence": f"{len(findings)} preprocessing leakage pattern(s) found before CV split.",
            "action":   "Move all fit() calls inside the CV fold loop. Never fit on data that includes validation rows.",
            "replan_instructions": {
                "remove_features": [],
                "rerun_nodes": ["data_engineer", "ml_optimizer"],
            },
        }

    return {"verdict": "OK", "audited_lines": len(lines)}


# =========================================================================
# VECTOR 1E — PR Curve Audit for Imbalanced Datasets
# =========================================================================

def _check_pr_curve_imbalance(
    y_true,
    y_prob,
    imbalance_ratio: float,
    target_type: str,
) -> dict:
    """
    For imbalanced binary classification: audit Precision-Recall curve.
    High precision + near-zero recall = model predicts majority class only.
    """
    if target_type != "binary":
        return {"verdict": "OK", "note": "PR curve audit only applies to binary classification"}

    if imbalance_ratio >= 0.15:
        return {"verdict": "OK", "note": f"Dataset not sufficiently imbalanced ({imbalance_ratio:.1%}) — PR audit skipped"}

    if y_prob is None or len(y_prob) == 0:
        return {"verdict": "OK", "note": "No OOF probabilities available for PR audit"}

    from sklearn.metrics import precision_recall_curve, auc as sk_auc

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = float(sk_auc(recall, precision))

    # Find recall at the threshold that maximises F1
    f1_scores    = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx     = int(np.argmax(f1_scores))
    best_recall  = float(recall[best_idx])
    best_prec    = float(precision[best_idx])
    best_f1      = float(f1_scores[best_idx])

    # Baseline: random classifier on imbalanced data
    random_pr_auc = imbalance_ratio

    result = {
        "pr_auc":           round(pr_auc, 4),
        "best_recall":      round(best_recall, 4),
        "best_precision":   round(best_prec, 4),
        "best_f1":          round(best_f1, 4),
        "random_baseline":  round(random_pr_auc, 4),
        "imbalance_ratio":  round(imbalance_ratio, 4),
    }

    if best_recall < 0.50:
        result.update({
            "verdict":  "CRITICAL",
            "evidence": (
                f"Model achieves best-F1 recall of only {best_recall:.1%} on minority class "
                f"({imbalance_ratio:.1%} prevalence). Model learned to predict majority class only. "
                f"PR-AUC: {pr_auc:.4f} vs random baseline: {random_pr_auc:.4f}."
            ),
            "action": (
                "Model is useless on the minority class despite good accuracy. "
                "Fix: add class_weight='balanced', use SMOTE oversampling, "
                "lower decision threshold, or use PR-AUC as optimisation metric instead of ROC-AUC."
            ),
            "replan_instructions": {
                "remove_features": [],
                "rerun_nodes": ["ml_optimizer"],
            },
        })
    elif pr_auc < random_pr_auc * 1.5:
        result.update({
            "verdict":  "HIGH",
            "evidence": f"PR-AUC {pr_auc:.4f} is only {pr_auc/random_pr_auc:.1f}x above random baseline {random_pr_auc:.4f}.",
            "action":   "Model barely beats random on the minority class. Tune class weights and decision threshold.",
            "replan_instructions": {"remove_features": [], "rerun_nodes": ["ml_optimizer"]},
        })
    else:
        result["verdict"] = "OK"

    return result


# =========================================================================
# VECTOR 1F — Temporal Leakage Check
# =========================================================================

def _check_temporal_leakage(
    df: pl.DataFrame,
    target_col: str,
    temporal_profile: dict,
) -> dict:
    """
    If temporal features exist, check for features that are suspiciously
    monotonically correlated with row order (proxy for time ordering).
    """
    if not temporal_profile.get("has_dates"):
        return {"verdict": "OK", "note": "No temporal features -- check skipped"}

    n = min(len(df), 1000)
    suspect = []

    numeric_cols = [
        c for c in df.columns
        if c != target_col
        and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]

    for col in numeric_cols[:50]:
        series = df[col].head(n).drop_nulls().to_numpy()
        if len(series) < 10:
            continue
        row_idx = np.arange(len(series))
        corr    = float(np.corrcoef(row_idx, series)[0, 1])
        if abs(corr) > 0.90:
            suspect.append({"feature": col, "row_order_correlation": round(corr, 4)})

    if suspect:
        return {
            "verdict":         "HIGH",
            "suspect_features": suspect,
            "evidence":         f"{len(suspect)} feature(s) highly correlated with row order ({[s['feature'] for s in suspect]}). These may be cumulative aggregates computed after the split.",
            "action":           "Verify these aggregates are computed within the CV fold only, not on the full dataset before splitting.",
            "replan_instructions": {
                "remove_features": [s["feature"] for s in suspect],
                "rerun_nodes": ["feature_factory", "ml_optimizer"],
            },
        }

    return {"verdict": "OK", "temporal_features_checked": temporal_profile.get("date_columns", [])}


# =========================================================================
# ORCHESTRATOR
# =========================================================================

def _overall_severity(findings: list) -> str:
    if not findings:
        return "OK"
    return max(
        (f.get("severity", "OK") for f in findings),
        key=lambda s: _SEVERITY_ORDER.get(s, 0),
    )


def run_red_team_critic(state: ProfessorState) -> ProfessorState:
    """LangGraph node -- inner retry loop."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = _run_core_logic(state, attempt)
            return reset_failure_count(result)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[{AGENT_NAME}] Attempt {attempt}/{MAX_ATTEMPTS} failed: {e}")
            if attempt == MAX_ATTEMPTS:
                level = get_escalation_level(state)
                return handle_escalation(state, level, AGENT_NAME, e, tb)
            state = {
                **state,
                "current_node_failure_count": attempt,
                "error_context": state.get("error_context", []) + [
                    {"agent": AGENT_NAME, "attempt": attempt, "error": str(e), "traceback": tb}
                ],
            }
    return state


def _run_core_logic(state: ProfessorState, attempt: int) -> ProfessorState:
    session_id = state["session_id"]
    output_dir = f"outputs/{session_id}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"[{AGENT_NAME}] Starting -- session: {session_id}, attempt: {attempt}")

    # -- Load data ---------------------------------------------------------------
    raw_data_path = state.get("raw_data_path", "")
    if not raw_data_path or not os.path.exists(raw_data_path):
        raise ValueError(f"[{AGENT_NAME}] raw_data_path missing or not found: {raw_data_path}")

    df = pl.read_csv(raw_data_path, infer_schema_length=10000)

    schema      = {}
    schema_path = state.get("schema_path", "")
    if schema_path and os.path.exists(schema_path):
        schema = json.load(open(schema_path))

    target_col  = state.get("target_col") or schema.get("target_col") or df.columns[-1]
    vs          = state.get("validation_strategy", {})
    target_type = vs.get("target_type", "binary")
    eda_report  = state.get("eda_report", {})

    # -- Load OOF predictions if available ---------------------------------------
    y_true = df[target_col].to_numpy()
    y_prob = None
    oof_path = state.get("oof_predictions_path", "")
    if oof_path and os.path.exists(oof_path):
        try:
            oof_df = pl.read_csv(oof_path)
            prob_col = [c for c in oof_df.columns if "prob" in c.lower() or "pred" in c.lower()]
            if prob_col:
                y_prob = oof_df[prob_col[0]].to_numpy()
        except Exception as e:
            logger.warning(f"[{AGENT_NAME}] Could not load OOF predictions: {e}")

    # -- Load test data if available ---------------------------------------------
    test_df = None
    test_path = state.get("test_data_path", "")
    if test_path and os.path.exists(test_path):
        try:
            test_df = pl.read_csv(test_path, infer_schema_length=10000)
        except Exception:
            pass

    # -- Load Data Engineer code if available ------------------------------------
    de_code = ""
    de_code_path = state.get("data_engineer_code_path", "")
    if de_code_path and os.path.exists(de_code_path):
        with open(de_code_path) as f:
            de_code = f.read()

    X_train = df.drop(target_col)

    # -- Run all 6 vectors -------------------------------------------------------
    vectors_checked = []
    findings        = []

    def _run_vector(name: str, result: dict):
        vectors_checked.append(name)
        verdict = result.get("verdict", "OK")
        logger.info(f"[{AGENT_NAME}] Vector {name}: {verdict}")
        if verdict != "OK":
            findings.append({"severity": verdict, "vector": name, **result})

    _run_vector("shuffled_target",       _check_shuffled_target(X_train, df[target_col], target_type))
    _run_vector("id_only_model",         _check_id_only_model(df, target_col, target_type, schema))
    _run_vector("adversarial_classifier", _check_adversarial_classifier(df, test_df or pl.DataFrame(), target_col))
    _run_vector("preprocessing_audit",   _check_preprocessing_leakage(de_code))

    imbalance_ratio = eda_report.get("target_distribution", {}).get("imbalance_ratio", 1.0)
    _run_vector("pr_curve_imbalance",    _check_pr_curve_imbalance(y_true, y_prob, imbalance_ratio, target_type))
    _run_vector("temporal_leakage",      _check_temporal_leakage(df, target_col, eda_report.get("temporal_profile", {})))

    # -- Compute overall severity ------------------------------------------------
    overall = _overall_severity(findings)

    verdict = {
        "overall_severity": overall,
        "vectors_checked":  vectors_checked,
        "findings":         findings,
        "clean":            overall == "OK",
        "checked_at":       datetime.now(timezone.utc).isoformat(),
    }

    verdict_path = f"{output_dir}/critic_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)

    logger.info(f"[{AGENT_NAME}] Verdict: {overall}. Findings: {len(findings)}. Path: {verdict_path}")

    log_event(
        session_id=session_id, agent=AGENT_NAME, action="verdict_issued",
        keys_read=["raw_data_path", "eda_report"],
        keys_written=["critic_verdict"],
        values_changed={"severity": overall, "findings_count": len(findings)},
    )

    updated = {
        **state,
        "critic_verdict":      verdict,
        "critic_verdict_path": verdict_path,
        "critic_severity":     overall,
    }

    # -- CRITICAL: halt pipeline -------------------------------------------------
    if overall == "CRITICAL":
        critical = [f for f in findings if f["severity"] == "CRITICAL"]
        rerun    = list({n for f in critical for n in f.get("replan_instructions", {}).get("rerun_nodes", [])})
        remove   = list({c for f in critical for c in f.get("replan_instructions", {}).get("remove_features", [])})
        updated.update({
            "hitl_required":   True,
            "hitl_reason": (
                f"Red Team Critic CRITICAL finding(s): "
                + "; ".join(f['evidence'] for f in critical if 'evidence' in f)
            ),
            "replan_requested": True,
            "replan_remove_features": remove,
            "replan_rerun_nodes":     rerun,
        })
        logger.error(f"[{AGENT_NAME}] CRITICAL -- pipeline halted. Rerun nodes: {rerun}")

    return updated
