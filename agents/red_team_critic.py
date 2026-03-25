# agents/red_team_critic.py
# -------------------------------------------------------------------------
# Day 11 — Red Team Critic: 7-vector quality gate
# Catches: target leakage, ID ordering leakage, train/test drift,
#          preprocessing leakage, majority-class-only models, temporal leakage,
#          robustness (noise injection, slice audit, calibration)
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
from tools.performance_monitor import timed_node

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
        except ValueError as e:
            # fallback for multiclass without proper AUC support
            try:
                scores = cross_val_score(model, X_np, y_shuffled, cv=3, scoring="accuracy")
            except Exception as e_inner:
                return {"verdict": "CRITICAL", "auc_shuffled": None, "evidence": f"Crash in shuffled target eval: {e_inner}"}
    else:
        model  = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42, n_jobs=-1)
        try:
            scores = cross_val_score(model, X_np, y_shuffled, cv=3, scoring="r2")
        except Exception as e_inner:
            return {"verdict": "CRITICAL", "auc_shuffled": None, "evidence": f"Crash in shuffled target eval: {e_inner}"}

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
            try:
                scores = cross_val_score(model, X_id, y, cv=3, scoring="accuracy")
            except Exception as e_inner:
                return {"verdict": "CRITICAL", "auc_id_only": None, "evidence": f"Crash in ID eval: {e_inner}"}
    else:
        model  = GradientBoostingRegressor(n_estimators=20, max_depth=2, random_state=42)
        try:
            scores = cross_val_score(model, X_id, y, cv=3, scoring="r2")
        except Exception as e_inner:
            return {"verdict": "CRITICAL", "auc_id_only": None, "evidence": f"Crash in ID eval: {e_inner}"}

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
# VECTOR 4 — Robustness: Noise Injection + Slice Audit + Calibration
# =========================================================================

def _noise_injection_check(
    X_train: pl.DataFrame,
    y_true,
    model_registry: list,
) -> dict:
    """
    Sub-check A: Add Gaussian noise (σ = 10% of feature stddev) to top-k
    features and re-score. Degradation > 20% → CRITICAL, 10-20% → HIGH.
    """
    if not model_registry:
        return {"verdict": "OK", "note": "No model in registry — noise injection skipped"}

    import pickle

    model_path = model_registry[0].get("model_path", "")
    if not model_path or not os.path.exists(model_path):
        return {"verdict": "OK", "note": "Model file not found — noise injection skipped"}

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return {"verdict": "CRITICAL", "evidence": f"Could not load model: {e}"}

    numeric_dtypes = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                      pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    numeric_cols = [c for c in X_train.columns if X_train[c].dtype in numeric_dtypes]
    if not numeric_cols:
        return {"verdict": "OK", "note": "No numeric features for noise injection"}

    X_np = X_train.select(numeric_cols).fill_null(0).to_numpy().astype(np.float64)

    # Get feature importances if available
    try:
        importances = model.feature_importances_
        top_k = min(5, len(importances))
        top_indices = np.argsort(importances)[-top_k:]
    except AttributeError:
        top_k = min(5, X_np.shape[1])
        top_indices = list(range(top_k))

    from sklearn.metrics import roc_auc_score

    # Clean score
    try:
        clean_probs = model.predict_proba(X_np)[:, 1]
        clean_auc = roc_auc_score(y_true, clean_probs)
    except Exception as e:
        return {"verdict": "CRITICAL", "evidence": f"Could not compute clean AUC: {e}"}

    # Noisy score
    rng = np.random.default_rng(42)
    X_noisy = X_np.copy()
    for idx in top_indices:
        col_std = np.std(X_np[:, idx])
        if col_std > 0:
            noise = rng.normal(0, 0.10 * col_std, size=X_np.shape[0])
            X_noisy[:, idx] += noise

    try:
        noisy_probs = model.predict_proba(X_noisy)[:, 1]
        noisy_auc = roc_auc_score(y_true, noisy_probs)
    except Exception as e:
        return {"verdict": "CRITICAL", "evidence": f"Could not compute noisy AUC: {e}"}

    if clean_auc > 0:
        degradation = (clean_auc - noisy_auc) / clean_auc
    else:
        degradation = 0.0

    result = {
        "clean_auc": round(clean_auc, 4),
        "noisy_auc": round(noisy_auc, 4),
        "degradation_pct": round(degradation * 100, 2),
        "top_features_perturbed": [numeric_cols[i] for i in top_indices if i < len(numeric_cols)],
    }

    if degradation > 0.20:
        result["verdict"] = "CRITICAL"
        result["evidence"] = (
            f"Noise injection caused {degradation:.1%} AUC degradation "
            f"(clean: {clean_auc:.4f} → noisy: {noisy_auc:.4f}). "
            f"Model is overfit to noise."
        )
    elif degradation > 0.10:
        result["verdict"] = "HIGH"
        result["evidence"] = (
            f"Noise injection caused {degradation:.1%} AUC degradation. "
            f"Model may be fragile."
        )
    else:
        result["verdict"] = "OK"

    return result


def _slice_performance_check(
    X_train: pl.DataFrame,
    y_true,
    model_registry: list,
) -> dict:
    """
    Sub-check B: Per-slice AUC for categoricals (2-10 unique) and
    quartile splits for numerics. Max-min spread > 0.15 → HIGH.
    """
    if not model_registry:
        return {"verdict": "OK", "note": "No model in registry — slice audit skipped"}

    import pickle
    from sklearn.metrics import roc_auc_score

    model_path = model_registry[0].get("model_path", "")
    if not model_path or not os.path.exists(model_path):
        return {"verdict": "OK", "note": "Model file not found — slice audit skipped"}

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return {"verdict": "CRITICAL", "evidence": f"Could not load model: {e}"}

    numeric_dtypes = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                      pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    numeric_cols = [c for c in X_train.columns if X_train[c].dtype in numeric_dtypes]
    if not numeric_cols:
        return {"verdict": "OK", "note": "No numeric features for slice audit"}

    X_np = X_train.select(numeric_cols).fill_null(0).to_numpy().astype(np.float64)

    try:
        y_prob = model.predict_proba(X_np)[:, 1]
    except Exception as e:
        return {"verdict": "CRITICAL", "evidence": f"Could not get predictions for slice audit: {e}"}

    y_arr = np.array(y_true)
    slices = []

    # Categorical slices
    cat_dtypes = (pl.Utf8, pl.Categorical, pl.String)
    for col in X_train.columns:
        if X_train[col].dtype in cat_dtypes:
            n_unique = X_train[col].n_unique()
            if 2 <= n_unique <= 10:
                for val in X_train[col].unique().to_list():
                    mask = (X_train[col] == val).to_numpy()
                    if mask.sum() < 20 or len(np.unique(y_arr[mask])) < 2:
                        continue
                    try:
                        auc = roc_auc_score(y_arr[mask], y_prob[mask])
                        slices.append({"feature": col, "value": str(val), "auc": round(auc, 4), "n": int(mask.sum())})
                    except ValueError:
                        pass

    # Numeric quartile slices
    for col in numeric_cols[:20]:
        series = X_train[col].to_numpy()
        median = np.nanmedian(series)
        for label, mask in [("bottom_half", series <= median), ("top_half", series > median)]:
            if mask.sum() < 20 or len(np.unique(y_arr[mask])) < 2:
                continue
            try:
                auc = roc_auc_score(y_arr[mask], y_prob[mask])
                slices.append({"feature": col, "value": label, "auc": round(auc, 4), "n": int(mask.sum())})
            except ValueError:
                pass

    if not slices:
        return {"verdict": "OK", "note": "No valid slices for audit"}

    best = max(slices, key=lambda s: s["auc"])
    worst = min(slices, key=lambda s: s["auc"])
    spread = best["auc"] - worst["auc"]

    result = {
        "best_slice": best,
        "worst_slice": worst,
        "spread": round(spread, 4),
        "slices_checked": len(slices),
    }

    if spread > 0.15:
        result["verdict"] = "HIGH"
        result["evidence"] = (
            f"Slice performance spread {spread:.4f} > 0.15. "
            f"Worst: {worst['feature']}={worst['value']} (AUC {worst['auc']}). "
            f"Best: {best['feature']}={best['value']} (AUC {best['auc']})."
        )
    else:
        result["verdict"] = "OK"

    return result


def _calibration_check(
    y_true,
    y_prob,
) -> dict:
    """
    Sub-check C: OOF calibration — ECE (10 bins) and Brier Score.
    ECE > 0.10 → HIGH.  Brier > 2× random baseline → HIGH.
    """
    if y_prob is None or len(y_prob) == 0:
        return {"verdict": "OK", "note": "No OOF probabilities — calibration skipped"}

    y_arr = np.array(y_true).astype(float)
    p_arr = np.array(y_prob).astype(float)

    # ECE — Expected Calibration Error (10 equal-width bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_arr)
    for i in range(n_bins):
        mask = (p_arr >= bin_edges[i]) & (p_arr < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = np.mean(y_arr[mask])
        bin_conf = np.mean(p_arr[mask])
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)

    # Brier Score
    brier = float(np.mean((p_arr - y_arr) ** 2))
    prevalence = float(np.mean(y_arr))
    random_brier = prevalence * (1 - prevalence)

    result = {
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "random_brier": round(random_brier, 4),
    }

    verdicts = []
    if ece > 0.10:
        verdicts.append(f"ECE {ece:.4f} > 0.10 threshold")
    if random_brier > 0 and brier > 2 * random_brier:
        verdicts.append(f"Brier {brier:.4f} > 2× random {random_brier:.4f}")

    if verdicts:
        result["verdict"] = "HIGH"
        result["evidence"] = "Calibration issues: " + "; ".join(verdicts)
    else:
        result["verdict"] = "OK"

    return result


def _check_calibration_quality(state) -> dict:
    """
    Checks calibration for all models in registry when metric is probability-based.
    """
    from agents.ml_optimizer import PROBABILITY_METRICS

    metric = state.get("evaluation_metric", "")
    if metric not in PROBABILITY_METRICS:
        return {"verdict": "OK", "note": "Non-probability metric — calibration check skipped."}

    registry = state.get("model_registry", {})
    if isinstance(registry, list):
        registry = {str(i): e for i, e in enumerate(registry)}
    warnings = []

    for model_name, entry in registry.items():
        if not entry.get("is_calibrated", False):
            warnings.append({
                "model":   model_name,
                "issue":   "Model not calibrated despite probability metric",
                "action":  "Check that calibration step ran for this model.",
            })

        brier = entry.get("calibration_score")
        if brier is not None and brier > 0.25:
            warnings.append({
                "model":  model_name,
                "issue":  f"Poor calibration: Brier score = {brier:.4f} (> 0.25 threshold)",
                "action": "Consider recalibration with more samples or a different method.",
            })

    if not warnings:
        return {"verdict": "OK", "note": f"All {len(registry)} models calibrated for {metric}."}

    return {
        "verdict":  "HIGH",
        "warnings": warnings,
        "note": (
            f"Calibration issues found for probability metric '{metric}'. "
            "Poor calibration directly harms log-loss and Brier score."
        ),
    }


def _check_robustness(
    X_train: pl.DataFrame,
    y_true,
    y_prob,
    eda_report: dict,
    model_registry: list,
) -> dict:
    """
    Vector 4: Robustness. Runs 3 sub-checks. Overall verdict = max severity.
    """
    sub_results = {}

    try:
        sub_results["noise_injection"] = _noise_injection_check(X_train, y_true, model_registry)
    except Exception as e:
        logger.error(f"[{AGENT_NAME}] noise_injection sub-check failed: {e}")
        sub_results["noise_injection"] = {"verdict": "CRITICAL", "note": f"Error: {e}", "evidence": f"Crash in noise_injection_check: {e}"}

    try:
        sub_results["slice_audit"] = _slice_performance_check(X_train, y_true, model_registry)
    except Exception as e:
        logger.error(f"[{AGENT_NAME}] slice_audit sub-check failed: {e}")
        sub_results["slice_audit"] = {"verdict": "CRITICAL", "note": f"Error: {e}", "evidence": f"Crash in slice_performance_check: {e}"}

    try:
        sub_results["calibration"] = _calibration_check(y_true, y_prob)
    except Exception as e:
        logger.error(f"[{AGENT_NAME}] calibration sub-check failed: {e}")
        sub_results["calibration"] = {"verdict": "CRITICAL", "note": f"Error: {e}", "evidence": f"Crash in calibration_check: {e}"}

    # Overall = max severity across sub-checks
    overall = max(
        (r.get("verdict", "OK") for r in sub_results.values()),
        key=lambda s: _SEVERITY_ORDER.get(s, 0),
    )

    return {
        "verdict": overall,
        "sub_checks": sub_results,
    }


# =========================================================================
# VECTOR 8 — Historical Failures (GM-CAP 4: compounding memory)
# =========================================================================

MIN_FEATURE_LEN_FOR_SUBSTRING = 4  # short names like "id" mustn't substring-match everything


def _check_historical_failures(state: ProfessorState) -> dict:
    """
    Vector 8: Retrieves top-5 historical failure patterns similar to this
    competition from the critic_failure_patterns ChromaDB collection. Flags
    any patterns where the flagged feature or failure mode is present in the
    current feature set.

    Severity logic:
        confidence >= 0.85 AND feature present  →  CRITICAL
        confidence >= 0.70 AND feature present  →  HIGH
        confidence >= 0.50 AND feature present  →  MEDIUM
        no matches or collection empty          →  OK

    Never raises — returns OK with diagnostic note on any failure.
    """
    from memory.memory_schema import query_critic_failure_patterns

    fingerprint = state.get("competition_fingerprint", {})
    feature_names = state.get("feature_names", [])

    # 1. Query ChromaDB for similar failure patterns
    try:
        patterns = query_critic_failure_patterns(
            fingerprint=fingerprint,
            n_results=5,
            max_distance=0.75,
        )
    except Exception as e:
        return {
            "verdict": "OK",
            "note": f"ChromaDB query failed ({e}). Historical check skipped.",
            "patterns_retrieved": 0,
            "findings": [],
        }

    if not patterns:
        return {
            "verdict": "OK",
            "note": "No similar historical failure patterns found in memory.",
            "patterns_retrieved": 0,
            "findings": [],
        }

    # 2. For each retrieved pattern, check if the failure mode is present now
    findings = []
    for pattern in patterns:
        feature_flagged    = pattern.get("feature_flagged", "")
        failure_mode       = pattern.get("failure_mode", "")
        confidence         = float(pattern.get("confidence", 0.0))
        cv_lb_gap          = float(pattern.get("cv_lb_gap", 0.0))
        competition_source = pattern.get("competition_name", pattern.get("competition", "unknown"))
        distance           = float(pattern.get("distance", 1.0))

        if not feature_flagged:
            continue  # no feature to match against

        # Match: exact name OR substring (min length guard to prevent false positives)
        # Short feature names (< 4 chars) require exact match only
        if len(feature_flagged) < MIN_FEATURE_LEN_FOR_SUBSTRING:
            feature_present = feature_flagged in feature_names
        else:
            feature_present = (
                feature_flagged in feature_names
                or any(feature_flagged in f for f in feature_names)
                or any(f in feature_flagged for f in feature_names if len(f) >= MIN_FEATURE_LEN_FOR_SUBSTRING)
            )

        if not feature_present:
            continue   # pattern doesn't apply to current feature set

        # Determine severity
        if confidence >= 0.85:
            severity = "CRITICAL"
        elif confidence >= 0.70:
            severity = "HIGH"
        elif confidence >= 0.50:
            severity = "MEDIUM"
        else:
            continue   # too low confidence to flag

        findings.append({
            "severity":          severity,
            "vector":            "historical_failures",
            "feature_flagged":   feature_flagged,
            "failure_mode":      failure_mode,
            "confidence":        round(confidence, 3),
            "cv_lb_gap_history": round(cv_lb_gap, 4),
            "competition_source": competition_source,
            "similarity_distance": round(distance, 3),
            "evidence": (
                f"In {competition_source} (similar competition profile, "
                f"distance={distance:.2f}), {failure_mode} caused "
                f"CV/LB gap={cv_lb_gap:.3f}. Confidence: {confidence:.2f}. "
                f"Feature '{feature_flagged}' is present in current feature set."
            ),
            "action": f"Investigate '{feature_flagged}' for {failure_mode}.",
            "replan_instructions": {
                "remove_features":  [feature_flagged] if severity == "CRITICAL" else [],
                "rerun_nodes":      ["feature_factory"] if severity == "CRITICAL" else [],
            },
        })

    if not findings:
        return {
            "verdict": "OK",
            "note": (
                f"Retrieved {len(patterns)} historical patterns. "
                f"None matched current feature set."
            ),
            "patterns_retrieved": len(patterns),
            "findings": [],
        }

    overall_severity = max(
        findings,
        key=lambda f: _SEVERITY_ORDER.get(f["severity"], 0),
    )["severity"]

    return {
        "verdict":           overall_severity,
        "patterns_retrieved": len(patterns),
        "patterns_matched":  len(findings),
        "findings":          findings,
    }


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


@timed_node
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
    # -- Load data ---------------------------------------------------------------
    feature_data_path = state.get("feature_data_path", "")
    if not feature_data_path or not os.path.exists(feature_data_path):
        raise ValueError(f"[{AGENT_NAME}] feature_data_path missing or not found: {feature_data_path}")

    df = pl.read_parquet(feature_data_path)

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
            if str(oof_path).endswith(".npy"):
                y_prob = np.load(oof_path)
            else:
                oof_df = pl.read_csv(oof_path)
                prob_col = [c for c in oof_df.columns if "prob" in c.lower() or "pred" in c.lower()]
                if prob_col:
                    y_prob = oof_df[prob_col[0]].to_numpy()
        except Exception as e:
            logger.error(f"[{AGENT_NAME}] Could not load OOF predictions from {oof_path}: {e}")

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

    _run_vector("robustness",            _check_robustness(
        X_train=X_train,
        y_true=y_true,
        y_prob=y_prob,
        eda_report=eda_report,
        model_registry=list(state.get("model_registry") or []),
    ))
    _run_vector("historical_failures",  _check_historical_failures(state))
    _run_vector("calibration_quality",   _check_calibration_quality(state))

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

    # -- CRITICAL: supervisor replan first, not hitl directly ------------------
    if overall == "CRITICAL":
        critical = [f for f in findings if f["severity"] == "CRITICAL"]
        rerun    = list({n for f in critical for n in f.get("replan_instructions", {}).get("rerun_nodes", [])})
        remove   = list({c for f in critical for c in f.get("replan_instructions", {}).get("remove_features", [])})
        updated.update({
            "hitl_required":   False,         # Day 11: supervisor gets first shot
            "replan_requested": True,
            "replan_remove_features": remove,
            "replan_rerun_nodes":     rerun,
        })
        logger.error(f"[{AGENT_NAME}] CRITICAL -- replan requested. Rerun nodes: {rerun}")

    return updated
