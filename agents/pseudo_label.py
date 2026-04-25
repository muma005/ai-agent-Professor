# agents/pseudo_label.py

import os
import json
import logging
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import wilcoxon

from core.state import ProfessorState
from tools.llm_provider import llm_call, _safe_json_loads
from tools.sandbox import run_in_sandbox
from core.lineage import log_event
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node
from tools.operator_channel import emit_to_operator

logger = logging.getLogger(__name__)

AGENT_NAME = "pseudo_label"

# ── Safety Caps — NON-NEGOTIABLE ─────────────────────────────────────────────
MAX_PSEUDO_ROUNDS = 2
MAX_PSEUDO_FRACTION = 0.30

# ── Core Logic ──────────────────────────────────────────────────────────────

def _run_pseudo_round(
    state: ProfessorState,
    round_num: int,
    current_k_pct: int,
    train_path: str,
    test_path: str,
    preds_path: str
) -> Tuple[bool, float, List[float], Dict]:
    """Execute a single pseudo-labeling round in the sandbox."""
    
    val_strat = state.get("validation_strategy") or {}
    cv_class = val_strat.get("cv_type", "StratifiedKFold")
    target_col = state.get("target_col", "target")
    task_type = state.get("task_type", "classification")
    n_splits = val_strat.get("n_splits", 5)

    code = f"""
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import {cv_class}, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error

# 1. Load data
train = pl.read_parquet("{train_path}")
test = pl.read_parquet("{test_path}")
test_preds = pl.read_parquet("{preds_path}")

# 2. Compute confidence
if "{task_type}" in ("binary", "multiclass"):
    confidence = test_preds["pred"].to_numpy()
    confidence = np.maximum(confidence, 1 - confidence)
else:
    confidence = np.ones(len(test_preds))

# 3. Filter top K%
K = {current_k_pct}
n_pseudo = int(len(test) * K / 100)
n_pseudo = min(n_pseudo, int(len(train) * {MAX_PSEUDO_FRACTION}))

top_indices = np.argsort(confidence)[-n_pseudo:]
pseudo_test = test[top_indices.tolist()]
pseudo_labels = test_preds["pred"][top_indices.tolist()]

if "{task_type}" in ("binary", "multiclass"):
    pseudo_labels = pseudo_labels.round().cast(pl.Int64)

# Tag
pseudo_test = pseudo_test.with_columns([
    pseudo_labels.alias("{target_col}"),
    pl.lit(True).alias("_is_pseudo_label"),
])

# 4. Augment
train_orig = train.with_columns(pl.lit(False).alias("_is_pseudo_label"))
train_aug = pl.concat([train_orig, pseudo_test])

# 5. Train with same CV
X_aug = train_aug.drop("{target_col}").to_numpy()
y_aug = train_aug["{target_col}"].to_numpy()

if "{task_type}" == "regression":
    cv = KFold(n_splits={n_splits}, shuffle=True, random_state=42)
else:
    cv = {cv_class}(n_splits={n_splits}, shuffle=True, random_state=42)

fold_scores = []
for tr_idx, val_idx in cv.split(train_orig.to_numpy(), train_orig["{target_col}"].to_numpy()):
    pseudo_indices = np.arange(len(train_orig), len(train_aug))
    tr_aug_idx = np.concatenate([tr_idx, pseudo_indices])
    
    X_tr, y_tr = X_aug[tr_aug_idx], y_aug[tr_aug_idx]
    X_val, y_val = X_aug[val_idx], y_aug[val_idx]
    
    if "{task_type}" == "regression":
        model = lgb.LGBMRegressor(n_estimators=100, verbosity=-1)
        model.fit(X_tr, y_tr)
        s = -mean_squared_error(y_val, model.predict(X_val))
    else:
        model = lgb.LGBMClassifier(n_estimators=100, verbosity=-1)
        model.fit(X_tr, y_tr)
        s = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    fold_scores.append(float(s))

print(f"PSEUDO_RESULT:{{\"mean_cv\": {{np.mean(fold_scores)}}, \"scores\": {{fold_scores}}, \"n_pseudo\": {{n_pseudo}}}}")
"""
    res = run_in_sandbox(code, agent_name=AGENT_NAME, purpose=f"Pseudo-label round {round_num}")
    
    if res["success"]:
        try:
            line = [l for l in res["stdout"].split("\n") if "PSEUDO_RESULT:" in l][0]
            data = json.loads(line.split("PSEUDO_RESULT:")[1])
            return True, data["mean_cv"], data["scores"], data
        except:
            return False, 0.0, [], {}
    return False, 0.0, [], {}

@timed_node
@with_agent_retry(AGENT_NAME)
def pseudo_label_architect(state: ProfessorState) -> ProfessorState:
    """
    Semi-supervised learning with safety-gated pseudo-labeling.
    """
    try:
        # 1. Activation Checks
        if state.get("pipeline_depth") == "sprint" or "pseudo_label" in (state.get("agents_skipped") or []):
            emit_to_operator("⏭️ Pseudo-Labels skipped (SPRINT mode)", level="STATUS")
            return state

        train_rows = state.get("canonical_train_rows", 0)
        test_rows = state.get("canonical_test_rows", 0)
        if test_rows < train_rows:
            emit_to_operator(f"⏭️ Pseudo-Labels skipped: test ({test_rows}) < train ({train_rows})", level="STATUS")
            return ProfessorState.validated_update(state, AGENT_NAME, {"pseudo_label_activated": False})

        critic_severity = state.get("critic_severity", "CLEAR")
        if critic_severity in ("CRITICAL", "CONFIRMED_CRITICAL"):
            emit_to_operator(f"⏭️ Pseudo-Labels skipped: Critic severity={critic_severity}", level="STATUS")
            return ProfessorState.validated_update(state, AGENT_NAME, {"pseudo_label_activated": False})

        emit_to_operator("🧪 Pseudo-Label Architect activated", level="STATUS")

        # 2. Setup paths
        train_path = state.get("feature_data_path") or state.get("clean_data_path")
        test_path = state.get("test_data_path")
        preds_path = state.get("test_predictions_path")
        
        if not all([train_path, test_path, preds_path]):
            return state

        original_cv = state.get("cv_mean", 0.0)
        original_scores = state.get("cv_scores", [])
        if not original_scores: original_scores = [original_cv] * 5

        # 3. Round 1
        k1 = min(30, int(100 * train_rows / test_rows) if test_rows > 0 else 30)
        success1, cv1, scores1, data1 = _run_pseudo_round(state, 1, k1, train_path, test_path, preds_path)
        
        if not success1:
            return ProfessorState.validated_update(state, AGENT_NAME, {"pseudo_label_activated": False})

        # Wilcoxon Gate
        wilcoxon_p_threshold = (state.get("gate_config") or {}).get("wilcoxon_p", 0.05)
        try:
            _, p_val = wilcoxon(scores1, original_scores, alternative="greater")
        except:
            p_val = 1.0

        round1_accepted = (p_val < wilcoxon_p_threshold and cv1 > original_cv)
        
        if not round1_accepted:
            emit_to_operator(f"🧪 Pseudo-Labels Round 1 REJECTED: p={p_val:.4f}", level="STATUS")
            return ProfessorState.validated_update(state, AGENT_NAME, {"pseudo_label_activated": False})

        emit_to_operator(f"🧪 Pseudo-Labels Round 1 ACCEPTED: +{cv1 - original_cv:.4f}", level="STATUS")

        # 4. Round 2
        k2 = k1 // 2
        success2, cv2, scores2, data2 = _run_pseudo_round(state, 2, k2, train_path, test_path, preds_path)
        
        final_cv = cv1
        final_fraction = data1["n_pseudo"] / train_rows if train_rows > 0 else 0.0
        
        if success2 and cv2 > cv1:
            try:
                _, p_val2 = wilcoxon(scores2, scores1, alternative="greater")
                if p_val2 < wilcoxon_p_threshold:
                    final_cv = cv2
                    final_fraction = data2["n_pseudo"] / train_rows if train_rows > 0 else 0.0
                    emit_to_operator(f"🧪 Pseudo-Labels Round 2 ACCEPTED: +{cv2 - cv1:.4f}", level="STATUS")
            except:
                pass

        # 5. Update State
        updates = {
            "pseudo_label_activated": True,
            "pseudo_label_fraction": float(final_fraction),
            "pseudo_label_cv_delta": float(final_cv - original_cv)
        }

        log_event(
            session_id=state.get("session_id", "default"),
            agent=AGENT_NAME,
            action="pseudo_label_complete",
            keys_written=list(updates.keys())
        )

        return ProfessorState.validated_update(state, AGENT_NAME, updates)

    except Exception as e:
        logger.error(f"[{AGENT_NAME}] Unexpected error: {e}")
        # Never halt pipeline, just return inactive state
        return ProfessorState.validated_update(state, AGENT_NAME, {"pseudo_label_activated": False})
