# Professor Agent — Day 10 Implementation Guide
**For: Claude Code**
**Status: Day 9 COMPLETE — resilience layer complete, all contracts green.**
**Mission: Build Professor's quality conscience. Day 10 is when Professor learns to catch its own mistakes.**

---

## ⚠️ NON-NEGOTIABLE RULES

1. **Read the existing codebase first.** Read `memory/chroma_client.py`, `agents/red_team_critic.py` (if it exists), `core/state.py`, `agents/eda_agent.py`, `agents/ml_optimizer.py`, and `tools/data_tools.py` in full before touching anything.
2. **The critic must never be bypassed.** On CRITICAL verdict, `hitl_required=True` and `replan_requested=True` are set. The pipeline must not reach the ensemble architect. If it does, the guard is missing.
3. **Memory schema migration must be non-destructive.** The new `professor_patterns_v2` collection coexists with any existing collection. Do not delete old data.
4. **All 6 critic detection vectors must run.** A critic that runs 3 vectors and silently skips the other 3 is worse than no critic — it creates false confidence.
5. **Regression suite runs after every task.** `pytest tests/regression/ tests/contracts/ -v` must be green before moving to the next task.

---

## BUILD ORDER

```
Task 1  →  memory/memory_schema.py       (standalone, no agent dependencies)
            ── commit: "Day 10: memory schema v2 — patterns not params" ──
Task 2a →  agents/red_team_critic.py     (core structure + Vector 1a: shuffled target)
Task 2b →  agents/red_team_critic.py     (Vector 1b: ID-only + adversarial classifier)
Task 2c →  agents/red_team_critic.py     (Vector 1c: GAP 2 — preprocessing code audit)
Task 2d →  agents/red_team_critic.py     (Vector 1d: PR curve audit — imbalanced datasets)
Task 2e →  agents/red_team_critic.py     (Vector 1e: temporal leakage check)
            ── commit: "Day 10: red_team_critic — all 6 vectors" ──
Task 3  →  tests/contracts/test_critic_contract.py  (IMMUTABLE after today)
            ── commit: "Day 10: critic contract test — immutable" ──
```

---

## TASK 1 — GM-CAP 2: Redesign ChromaDB Memory Schema

**File:** `memory/memory_schema.py`
**Priority:** HIGH — this makes memory useful. The current schema stores `cv_score + hyperparams` which never transfer between competitions. After this change, Professor starts every competition with validated priors from structurally similar past competitions.

### Why Hyperparams Don't Transfer (and Patterns Do)

A model trained on a 10-row dataset with imbalance 0.5 will have completely different optimal hyperparams than one trained on 1M rows with imbalance 0.05. Sharing hyperparams between them is noise.

But patterns transfer exactly: "competitions with high cardinality categoricals + temporal features respond better to target encoding inside the CV fold than one-hot encoding" is true regardless of exact dataset size. These structural insights are what Professor needs to warm-start from.

### Competition Fingerprint Structure

```python
FINGERPRINT_SCHEMA = {
    "task_type":                    str,    # "tabular" | "timeseries" | "nlp" | "image"
    "imbalance_ratio":              float,  # minority_class_count / total. 1.0 = balanced
    "n_categorical_high_cardinality": int,  # columns with n_unique > 50
    "n_rows_bucket":                str,    # "tiny"<1k | "small"<10k | "medium"<100k | "large"<1M | "huge">=1M
    "has_temporal_feature":         bool,
    "n_features_bucket":            str,    # "narrow"<10 | "medium"<50 | "wide"<200 | "very_wide">=200
    "target_type":                  str,    # "binary" | "multiclass" | "continuous"
    "cv_lb_gap_typical":            float,  # populated post-competition by post_mortem_agent
}
```

### Pattern Entry Structure

```python
PATTERN_SCHEMA = {
    "pattern_id":               str,    # UUID
    "competition_fingerprint":  dict,   # matches FINGERPRINT_SCHEMA above
    "validated_approaches": [           # things that improved the score
        {
            "approach":         str,    # human-readable description
            "cv_improvement":   float,  # positive = improvement
            "competitions":     list,   # which competitions this was validated on
        }
    ],
    "failed_approaches": [              # things that hurt the score
        {
            "approach":         str,
            "cv_degradation":   float,  # positive = degradation
            "competitions":     list,
        }
    ],
    "confidence":               float,  # 0.0–1.0. Increases with each validation.
    "competitions_validated_on": list,  # list of competition names
    "created_at":               str,    # ISO timestamp
    "last_updated":             str,    # ISO timestamp
}
```

### How ChromaDB Stores Patterns

ChromaDB works on text embeddings. The pattern JSON is **not** what gets embedded — a natural language description of the fingerprint is. This is the critical design decision that makes retrieval semantic:

```python
def fingerprint_to_text(fingerprint: dict) -> str:
    """
    Converts a competition fingerprint to a natural language description
    that will be embedded by all-MiniLM-L6-v2.
    
    This text representation is what makes semantic retrieval work.
    A query about "binary classification with highly imbalanced data"
    should match a stored pattern about "fraud detection with 3% positive rate".
    """
    parts = [
        f"{fingerprint['task_type']} task",
        f"target type: {fingerprint['target_type']}",
        f"dataset size: {fingerprint['n_rows_bucket']} ({fingerprint['n_rows_bucket']} rows)",
        f"feature space: {fingerprint['n_features_bucket']} features",
    ]

    if fingerprint["imbalance_ratio"] < 0.15:
        parts.append(f"highly imbalanced dataset ({fingerprint['imbalance_ratio']:.1%} minority)")
    elif fingerprint["imbalance_ratio"] < 0.35:
        parts.append(f"moderately imbalanced ({fingerprint['imbalance_ratio']:.1%} minority)")
    else:
        parts.append("balanced dataset")

    if fingerprint["n_categorical_high_cardinality"] > 5:
        parts.append(f"many high-cardinality categorical features ({fingerprint['n_categorical_high_cardinality']})")
    elif fingerprint["n_categorical_high_cardinality"] > 0:
        parts.append(f"some high-cardinality categoricals ({fingerprint['n_categorical_high_cardinality']})")

    if fingerprint["has_temporal_feature"]:
        parts.append("time-series or temporal structure present")

    return ". ".join(parts)
```

### Full Implementation

```python
# memory/memory_schema.py

import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from memory.chroma_client import build_chroma_client, get_or_create_collection
from core.state import ProfessorState

logger = logging.getLogger(__name__)

PATTERNS_COLLECTION = "professor_patterns_v2"  # v2 — do NOT touch v1


def build_competition_fingerprint(state: ProfessorState) -> dict:
    """
    Builds a competition fingerprint from the current pipeline state.
    Called after EDA completes — requires eda_report and validation_strategy in state.

    Never raises. Returns a fingerprint with sensible defaults if data is missing.
    """
    eda    = state.get("eda_report", {})
    schema = {}
    if state.get("schema_path"):
        import os, json as _json
        if os.path.exists(state["schema_path"]):
            try:
                schema = _json.load(open(state["schema_path"]))
            except Exception:
                pass

    # ── Row count bucket ───────────────────────────────────────────────────────
    n_rows = schema.get("n_rows", 0)
    if n_rows < 1_000:
        n_rows_bucket = "tiny"
    elif n_rows < 10_000:
        n_rows_bucket = "small"
    elif n_rows < 100_000:
        n_rows_bucket = "medium"
    elif n_rows < 1_000_000:
        n_rows_bucket = "large"
    else:
        n_rows_bucket = "huge"

    # ── Feature count bucket ───────────────────────────────────────────────────
    n_features = len(schema.get("columns", [])) - 1  # exclude target
    if n_features < 10:
        n_features_bucket = "narrow"
    elif n_features < 50:
        n_features_bucket = "medium"
    elif n_features < 200:
        n_features_bucket = "wide"
    else:
        n_features_bucket = "very_wide"

    # ── Imbalance ratio ────────────────────────────────────────────────────────
    target_dist   = eda.get("target_distribution", {})
    imbalance_ratio = target_dist.get("imbalance_ratio", 1.0)  # 1.0 = balanced

    # ── High-cardinality categorical count ────────────────────────────────────
    types = schema.get("types", {})
    n_unique_map = schema.get("n_unique", {})
    target_col = state.get("target_col", "")
    n_categorical_high_cardinality = sum(
        1 for col, dtype in types.items()
        if col != target_col
        and str(dtype) in {"Utf8", "Categorical", "str"}
        and n_unique_map.get(col, 0) > 50
    )

    # ── Temporal feature ──────────────────────────────────────────────────────
    temporal = eda.get("temporal_profile", {})
    has_temporal_feature = temporal.get("has_dates", False)

    # ── Task and target types ──────────────────────────────────────────────────
    task_type   = state.get("task_type", "unknown")
    vs          = state.get("validation_strategy", {})
    target_type = vs.get("target_type", "unknown")

    fingerprint = {
        "task_type":                      task_type,
        "imbalance_ratio":                round(float(imbalance_ratio), 4),
        "n_categorical_high_cardinality": int(n_categorical_high_cardinality),
        "n_rows_bucket":                  n_rows_bucket,
        "has_temporal_feature":           bool(has_temporal_feature),
        "n_features_bucket":              n_features_bucket,
        "target_type":                    target_type,
        "cv_lb_gap_typical":              0.0,  # populated post-competition
    }

    logger.info(f"[MemorySchema] Competition fingerprint built: {fingerprint}")
    return fingerprint


def fingerprint_to_text(fingerprint: dict) -> str:
    """Converts fingerprint dict to embeddable natural language text."""
    imbalance = fingerprint.get("imbalance_ratio", 1.0)
    parts = [
        f"{fingerprint.get('task_type', 'tabular')} machine learning task",
        f"target: {fingerprint.get('target_type', 'unknown')}",
        f"dataset size: {fingerprint.get('n_rows_bucket', 'medium')}",
        f"feature count: {fingerprint.get('n_features_bucket', 'medium')}",
    ]
    if imbalance < 0.05:
        parts.append(f"severely imbalanced dataset {imbalance:.1%} minority class fraud-like")
    elif imbalance < 0.15:
        parts.append(f"highly imbalanced {imbalance:.1%} minority class medical or anomaly detection")
    elif imbalance < 0.35:
        parts.append(f"moderately imbalanced {imbalance:.1%} minority class")
    else:
        parts.append("balanced or near-balanced class distribution")

    n_hc = fingerprint.get("n_categorical_high_cardinality", 0)
    if n_hc > 5:
        parts.append(f"many high-cardinality categorical features {n_hc} columns over 50 unique values")
    elif n_hc > 0:
        parts.append(f"some high-cardinality categoricals {n_hc} columns")
    else:
        parts.append("no high-cardinality categorical features")

    if fingerprint.get("has_temporal_feature"):
        parts.append("temporal structure present time-series date features")

    return ". ".join(parts) + "."


def store_pattern(
    fingerprint: dict,
    validated_approaches: list,
    failed_approaches: list,
    competition_name: str,
    confidence: float = 0.5,
    cv_lb_gap: float = 0.0,
) -> str:
    """
    Stores a pattern in ChromaDB v2 collection.
    Called by post_mortem_agent (Day 11) after a competition closes.
    Returns the pattern_id.
    """
    client     = build_chroma_client()
    collection = get_or_create_collection(client, PATTERNS_COLLECTION)

    pattern_id = str(uuid.uuid4())
    fingerprint["cv_lb_gap_typical"] = cv_lb_gap

    pattern = {
        "pattern_id":               pattern_id,
        "competition_fingerprint":  fingerprint,
        "validated_approaches":     validated_approaches,
        "failed_approaches":        failed_approaches,
        "confidence":               confidence,
        "competitions_validated_on": [competition_name],
        "created_at":               datetime.now(timezone.utc).isoformat(),
        "last_updated":             datetime.now(timezone.utc).isoformat(),
    }

    document_text = (
        fingerprint_to_text(fingerprint)
        + " Validated approaches: "
        + "; ".join(a["approach"] for a in validated_approaches[:3])
    )

    collection.add(
        documents=[document_text],
        metadatas=[{"pattern_json": json.dumps(pattern), "competition": competition_name}],
        ids=[pattern_id],
    )
    logger.info(f"[MemorySchema] Pattern stored: {pattern_id} for {competition_name}")
    return pattern_id


def query_similar_competitions(
    fingerprint: dict,
    n_results: int = 5,
) -> list:
    """
    Returns up to n_results patterns from structurally similar past competitions.
    Sorted by semantic similarity — most similar first.

    Returns list of dicts, each containing the full pattern_json and distance.
    Returns [] (not None) if ChromaDB is unavailable or no patterns exist.
    """
    try:
        client     = build_chroma_client()
        collection = get_or_create_collection(client, PATTERNS_COLLECTION)

        count = collection.count()
        if count == 0:
            logger.info("[MemorySchema] Pattern collection is empty — no warm-start priors available.")
            return []

        query_text = fingerprint_to_text(fingerprint)
        results    = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

        patterns = []
        for i, metadata in enumerate(results["metadatas"][0]):
            try:
                pattern = json.loads(metadata["pattern_json"])
                pattern["_similarity_distance"] = results["distances"][0][i]
                patterns.append(pattern)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[MemorySchema] Skipping malformed pattern: {e}")

        logger.info(f"[MemorySchema] Retrieved {len(patterns)} similar patterns.")
        return patterns

    except Exception as e:
        logger.warning(f"[MemorySchema] Pattern query failed: {e}. Returning empty priors.")
        return []


def get_warm_start_priors(state: ProfessorState) -> list:
    """
    High-level entry point called by ml_optimizer at the start of each session.
    Builds fingerprint from state, queries memory, returns top validated approaches.
    """
    fingerprint = build_competition_fingerprint(state)
    patterns    = query_similar_competitions(fingerprint, n_results=3)

    priors = []
    for pattern in patterns:
        distance = pattern.get("_similarity_distance", 1.0)
        if distance > 0.8:
            continue  # too dissimilar — don't trust it
        for approach in pattern.get("validated_approaches", []):
            priors.append({
                "approach":    approach["approach"],
                "confidence":  pattern["confidence"] * (1.0 - distance),
                "source":      pattern["competitions_validated_on"],
                "improvement": approach.get("cv_improvement", 0.0),
            })

    priors.sort(key=lambda x: x["confidence"], reverse=True)
    logger.info(f"[MemorySchema] Warm-start priors: {len(priors)} approaches")
    return priors[:10]  # top 10 only
```

### Wire into Pipeline

**In `agents/ml_optimizer.py`**, at the start of `run_ml_optimizer()`:
```python
from memory.memory_schema import get_warm_start_priors

priors = get_warm_start_priors(state)
state["warm_start_priors"] = priors
if priors:
    print(f"[MLOptimizer] Warm-start: {len(priors)} priors from similar competitions")
    for p in priors[:3]:
        print(f"  → {p['approach']} (confidence: {p['confidence']:.2f})")
```

**Add to `ProfessorState` and `initial_state()`:**
```python
# ProfessorState:
competition_fingerprint: dict   # built from EDA + schema
warm_start_priors:       list   # retrieved from memory

# initial_state():
"competition_fingerprint": {},
"warm_start_priors":       [],
```

### Verification

```bash
python -c "
from memory.memory_schema import (
    build_competition_fingerprint, fingerprint_to_text,
    store_pattern, query_similar_competitions
)
from core.state import initial_state
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect

state = initial_state('test-memory', 'data/spaceship_titanic/train.csv')
state = run_data_engineer(state)
state = run_eda_agent(state)
state = run_validation_architect(state)

fp = build_competition_fingerprint(state)
print('Fingerprint:', fp)

text = fingerprint_to_text(fp)
print('Embedded text:', text)

# Store a test pattern
pid = store_pattern(
    fingerprint=fp,
    validated_approaches=[{'approach': 'LightGBM with log-transform', 'cv_improvement': 0.02, 'competitions': ['test']}],
    failed_approaches=[],
    competition_name='spaceship-titanic',
    confidence=0.6,
)
print('Pattern stored:', pid)

# Query it back
results = query_similar_competitions(fp, n_results=3)
print('Retrieved patterns:', len(results))
assert len(results) >= 1
assert results[0]['pattern_id'] == pid
print('[PASS] Memory schema v2 round-trip verified')
"
```

---

## TASK 2 — Build `agents/red_team_critic.py` — Vector 1 (All Leakage Detection)

**File:** `agents/red_team_critic.py`
**Priority:** CRITICAL — this is Professor's quality gate. Without the critic, Professor is a system that trains models confidently and submits garbage.

### Pipeline Position

```
feature_factory → red_team_critic → ensemble_architect
```

If `critic_verdict["overall_severity"] == "CRITICAL"` → set `hitl_required=True`, `replan_requested=True`. Do NOT proceed to ensemble_architect.

### `critic_verdict.json` Output Schema

This is the contract. Every key is required. The contract test enforces all of them.

```json
{
  "overall_severity": "CRITICAL | HIGH | MEDIUM | OK",
  "vectors_checked":  ["shuffled_target", "id_only_model", "adversarial_classifier",
                       "preprocessing_audit", "pr_curve_imbalance", "temporal_leakage"],
  "findings": [
    {
      "severity":   "CRITICAL | HIGH | MEDIUM | OK",
      "vector":     "one of vectors_checked",
      "evidence":   {},
      "action":     "specific remediation instruction",
      "replan_instructions": {
        "remove_features": [],
        "rerun_nodes":     []
      }
    }
  ],
  "clean": true,
  "checked_at": "ISO timestamp"
}
```

`clean: true` means `overall_severity == "OK"`. No findings in the list = clean.

### State Additions

```python
# ProfessorState:
critic_verdict:      dict   # full verdict dict
critic_verdict_path: str
critic_severity:     str    # top-level severity string
replan_requested:    bool

# initial_state():
"critic_verdict":      {},
"critic_verdict_path": "",
"critic_severity":     "unchecked",
"replan_requested":    False,
```

---

### VECTOR 1A — Shuffled Target Test

**Bug it catches:** Any form of data leakage. If a model trained on *randomly shuffled targets* gets meaningful AUC, the features contain information that is not supposed to be there.

**Threshold:** AUC > 0.55 on shuffled targets = leakage. (Random = 0.50, threshold gives a 5% buffer.)

```python
def _check_shuffled_target(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    target_type: str,
) -> dict:
    """
    Trains a simple model on shuffled targets.
    If AUC is meaningfully above 0.5, leakage is present.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import polars as pl

    y_shuffled = y_train.sample(fraction=1.0, shuffle=True, seed=42).to_numpy()
    X_np = X_train.select(pl.col(pl.NUMERIC_DTYPES)).to_numpy()

    if X_np.shape[1] == 0:
        return {"verdict": "OK", "auc_shuffled": None, "note": "No numeric features to test"}

    if target_type in ("binary", "multiclass"):
        model  = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X_np, y_shuffled, cv=3, scoring="roc_auc")
    else:
        model  = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X_np, y_shuffled, cv=3, scoring="r2")

    mean_score = float(np.mean(scores))
    threshold  = 0.55 if target_type in ("binary", "multiclass") else 0.10

    if (target_type in ("binary", "multiclass") and mean_score > threshold):
        return {
            "verdict":      "CRITICAL",
            "auc_shuffled": round(mean_score, 4),
            "threshold":    threshold,
            "evidence":     f"Model trained on shuffled targets achieved AUC {mean_score:.4f} > {threshold}. Leakage confirmed.",
            "action":       "Inspect features for any direct or indirect encoding of the target. Remove suspect features and retrain.",
            "replan_instructions": {
                "remove_features": [],  # populated by deeper analysis
                "rerun_nodes": ["feature_factory", "ml_optimizer"],
            },
        }

    return {"verdict": "OK", "auc_shuffled": round(mean_score, 4)}
```

---

### VECTOR 1B — ID-Only Model Test

**Bug it catches:** Row ordering leakage. If a model trained using only identifier columns (IDs, row numbers) can predict the target, the training data is sorted by target or the ID encodes temporal ordering.

```python
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
    import numpy as np
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
        scores = cross_val_score(model, X_id, y, cv=3, scoring="roc_auc")
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
```

---

### VECTOR 1C — Adversarial Train/Test Classifier

**Bug it catches:** Distribution shift between train and test. If a classifier can tell "is this row from train or test?", the feature distributions are different — features that look predictive in CV will degrade on the LB.

```python
def _check_adversarial_classifier(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_col: str,
) -> dict:
    """
    Binary classifier: train row = 0, test row = 1.
    AUC > 0.6 = feature distributions differ between train and test.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import polars as pl

    if test_df is None or len(test_df) == 0:
        return {"verdict": "OK", "note": "No test data available for adversarial check"}

    numeric_cols = [
        c for c in train_df.columns
        if c != target_col
        and train_df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
    ]
    if not numeric_cols:
        return {"verdict": "OK", "note": "No shared numeric columns for adversarial test"}

    shared_cols = [c for c in numeric_cols if c in test_df.columns]

    train_X = train_df.select(shared_cols).fill_null(0).to_numpy()
    test_X  = test_df.select(shared_cols).fill_null(0).to_numpy()

    # Subsample for speed — 5k rows max per split
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
        # Find which features drive the split
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
```

---

### VECTOR 1D — Preprocessing Leakage Code Audit (GAP 2)

**Bug it catches:** StandardScaler, imputers, target encoders, PCA fitted on the full training dataset before the CV split. This is the most common silent failure in Kaggle — it inflates CV by 2–5% and collapses on the private LB.

**Method:** Static analysis of the Data Engineer's generated Python code. Regex is sufficient — we are looking for known anti-patterns.

```python
import re

# Patterns that indicate a preprocessing step fitted on full data before split
_LEAKAGE_PATTERNS = [
    (
        re.compile(r"(\w+)\s*=\s*StandardScaler\(\).*?\n.*?\1\.fit(?:_transform)?\s*\((?!.*fold|.*train_fold|.*X_train\b)"),
        "StandardScaler fitted before CV split",
    ),
    (
        re.compile(r"fit_transform\s*\(\s*(?:X|train|df|data)\s*\)"),
        "fit_transform called on full dataset (variable name suggests pre-split data)",
    ),
    (
        re.compile(r"(?:SimpleImputer|KNNImputer|IterativeImputer)\s*\(.*\).*\.fit\s*\(\s*(?:X|train|data)\s*\)"),
        "Imputer fitted on full dataset before CV",
    ),
    (
        re.compile(r"(?:TargetEncoder|OrdinalEncoder|LabelEncoder)\s*\(.*\).*\.fit\s*\(\s*(?:X|train|data)\s*\)"),
        "Encoder fitted on full dataset before CV",
    ),
    (
        re.compile(r"PCA\s*\(.*\).*\.fit(?:_transform)?\s*\(\s*(?:X|train|data)\s*\)"),
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

    # Find where train/test split or CV fold creation happens
    for i, line in enumerate(lines):
        if any(kw in line for kw in ["train_test_split", "KFold", "StratifiedKFold", "GroupKFold", "fold"]):
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
```

---

### VECTOR 1E — PR Curve Audit for Imbalanced Datasets

**Bug it catches:** A model that learned to predict the majority class only. This is catastrophic in fraud detection, medical diagnosis, and churn — the model achieves high accuracy by ignoring the minority class entirely. ROC-AUC looks fine because it's not recall-sensitive. Only the PR curve reveals this failure.

**Trigger:** `imbalance_ratio < 0.15` (minority class is less than 15% of dataset).

```python
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
    import numpy as np

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
    random_pr_auc = imbalance_ratio  # random classifier PR-AUC = prevalence

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
```

---

### VECTOR 1F — Temporal Leakage Check

**Bug it catches:** Features that encode future information relative to the row's timestamp.

```python
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
        return {"verdict": "OK", "note": "No temporal features — check skipped"}

    import numpy as np

    # Check for features that are monotonically increasing with row index
    # (proxy for future leakage — cumulative aggregates computed post-split)
    n = min(len(df), 1000)
    suspect = []

    numeric_cols = [
        c for c in df.columns
        if c != target_col
        and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]

    for col in numeric_cols[:50]:  # check first 50 numeric columns
        series = df[col].head(n).drop_nulls().to_numpy()
        if len(series) < 10:
            continue
        # Spearman correlation with row index
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
```

---

### Main Orchestrator

```python
# agents/red_team_critic.py

import os
import json
import logging
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


def _overall_severity(findings: list) -> str:
    if not findings:
        return "OK"
    return max(
        (f.get("severity", "OK") for f in findings),
        key=lambda s: _SEVERITY_ORDER.get(s, 0),
    )


def run_red_team_critic(state: ProfessorState) -> ProfessorState:
    """LangGraph node — inner retry loop."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = _run_core_logic(state, attempt)
            return reset_failure_count(result)
        except Exception as e:
            import traceback
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

    logger.info(f"[{AGENT_NAME}] Starting — session: {session_id}, attempt: {attempt}")

    # ── Load data ──────────────────────────────────────────────────────────────
    raw_data_path = state.get("raw_data_path", "")
    if not raw_data_path or not os.path.exists(raw_data_path):
        raise ValueError(f"[{AGENT_NAME}] raw_data_path missing or not found: {raw_data_path}")

    df = pl.read_csv(raw_data_path, infer_schema_length=10000)

    schema      = {}
    schema_path = state.get("schema_path", "")
    if schema_path and os.path.exists(schema_path):
        schema = json.load(open(schema_path))

    target_col = state.get("target_col") or schema.get("target_col") or df.columns[-1]
    vs         = state.get("validation_strategy", {})
    target_type = vs.get("target_type", "binary")
    eda_report  = state.get("eda_report", {})

    # ── Load OOF predictions if available ─────────────────────────────────────
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

    # ── Load test data if available ────────────────────────────────────────────
    test_df = None
    test_path = state.get("test_data_path", "")
    if test_path and os.path.exists(test_path):
        try:
            test_df = pl.read_csv(test_path, infer_schema_length=10000)
        except Exception:
            pass

    # ── Load Data Engineer code if available ───────────────────────────────────
    de_code = ""
    de_code_path = state.get("data_engineer_code_path", "")
    if de_code_path and os.path.exists(de_code_path):
        with open(de_code_path) as f:
            de_code = f.read()

    X_train = df.drop(target_col)

    # ── Run all 6 vectors ──────────────────────────────────────────────────────
    vectors_checked = []
    findings        = []

    def _run_vector(name: str, result: dict):
        vectors_checked.append(name)
        verdict = result.get("verdict", "OK")
        logger.info(f"[{AGENT_NAME}] Vector {name}: {verdict}")
        if verdict != "OK":
            findings.append({"severity": verdict, "vector": name, **result})

    _run_vector("shuffled_target",     _check_shuffled_target(X_train, df[target_col], target_type))
    _run_vector("id_only_model",       _check_id_only_model(df, target_col, target_type, schema))
    _run_vector("adversarial_classifier", _check_adversarial_classifier(df, test_df or pl.DataFrame(), target_col))
    _run_vector("preprocessing_audit", _check_preprocessing_leakage(de_code))

    imbalance_ratio = eda_report.get("target_distribution", {}).get("imbalance_ratio", 1.0)
    _run_vector("pr_curve_imbalance",  _check_pr_curve_imbalance(y_true, y_prob, imbalance_ratio, target_type))
    _run_vector("temporal_leakage",    _check_temporal_leakage(df, target_col, eda_report.get("temporal_profile", {})))

    # ── Compute overall severity ───────────────────────────────────────────────
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

    # ── CRITICAL: halt pipeline ────────────────────────────────────────────────
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
        logger.error(f"[{AGENT_NAME}] CRITICAL — pipeline halted. Rerun nodes: {rerun}")

    return updated
```

### Wire into LangGraph

```python
# core/professor.py

from agents.red_team_critic import run_red_team_critic

graph.add_node("red_team_critic", run_red_team_critic)
graph.add_edge("feature_factory", "red_team_critic")
graph.add_conditional_edges(
    "red_team_critic",
    lambda s: "hitl_handler" if s.get("hitl_required") else "ensemble_architect",
    {"hitl_handler": "hitl_handler", "ensemble_architect": "ensemble_architect"},
)
```

---

## TASK 3 — Contract Test: Red Team Critic

**File:** `tests/contracts/test_critic_contract.py`
**Status after writing: IMMUTABLE**

```python
# tests/contracts/test_critic_contract.py
# ──────────────────────────────────────────────────────────────────────────────
# Written: Day 10   Status: IMMUTABLE
#
# CONTRACT: run_red_team_critic()
#   INPUT:   state["raw_data_path"], state["eda_report"], state["validation_strategy"]
#   OUTPUT:  critic_verdict.json on disk — overall_severity/vectors_checked/findings/clean/checked_at
#   MUST CATCH: injected target leakage → CRITICAL
#               majority-class-only model (recall < 0.5 on imbalanced) → CRITICAL
#               preprocessing leakage code pattern → CRITICAL
#   NEVER:   Proceed to ensemble when overall_severity == CRITICAL
#            Return verdict without all required keys
#            Silently skip any of the 6 vectors
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import pytest
import numpy as np
import polars as pl

from core.state import initial_state
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect
from agents.red_team_critic import run_red_team_critic

FIXTURE_CSV   = "tests/fixtures/tiny_train.csv"
TITANIC_CSV   = "data/spaceship_titanic/train.csv"


@pytest.fixture(scope="module")
def clean_state():
    """Clean pipeline state — critic should return OK verdict."""
    s = initial_state("test-critic-clean", FIXTURE_CSV)
    s = run_data_engineer(s)
    s = run_eda_agent(s)
    s = run_validation_architect(s)
    return run_red_team_critic(s)


class TestCriticContractOutputSchema:

    def test_runs_without_error(self, clean_state):
        assert clean_state is not None

    def test_critic_verdict_key_in_state(self, clean_state):
        assert "critic_verdict" in clean_state
        assert isinstance(clean_state["critic_verdict"], dict)

    def test_critic_verdict_json_written_to_disk(self, clean_state):
        path = clean_state.get("critic_verdict_path")
        assert path is not None
        assert os.path.exists(path), f"critic_verdict.json not found at {path}"
        loaded = json.load(open(path))
        assert isinstance(loaded, dict)

    def test_overall_severity_is_valid_value(self, clean_state):
        s = clean_state["critic_verdict"]["overall_severity"]
        assert s in ("CRITICAL", "HIGH", "MEDIUM", "OK"), f"Invalid severity: {s}"

    def test_vectors_checked_contains_all_six(self, clean_state):
        vc = clean_state["critic_verdict"]["vectors_checked"]
        required = {
            "shuffled_target", "id_only_model", "adversarial_classifier",
            "preprocessing_audit", "pr_curve_imbalance", "temporal_leakage",
        }
        missing = required - set(vc)
        assert not missing, f"Vectors not checked: {missing}. All 6 must run."

    def test_findings_is_a_list(self, clean_state):
        assert isinstance(clean_state["critic_verdict"]["findings"], list)

    def test_clean_flag_matches_severity(self, clean_state):
        v = clean_state["critic_verdict"]
        if v["overall_severity"] == "OK":
            assert v["clean"] is True
        else:
            assert v["clean"] is False

    def test_checked_at_is_iso_timestamp(self, clean_state):
        from datetime import datetime
        ts = clean_state["critic_verdict"]["checked_at"]
        assert ts is not None and len(ts) > 0
        datetime.fromisoformat(ts)  # raises if not valid ISO

    def test_critic_severity_in_state_matches_verdict(self, clean_state):
        assert clean_state["critic_severity"] == clean_state["critic_verdict"]["overall_severity"]

    def test_clean_data_does_not_trigger_hitl(self, clean_state):
        assert clean_state.get("hitl_required") is not True, (
            f"HITL triggered on clean data. Reason: {clean_state.get('hitl_reason', 'none')}"
        )


class TestCriticCatchesInjectedLeakage:

    def test_injected_target_leakage_triggers_critical(self):
        """
        Inject a copy of the target into features.
        Shuffled target test must detect this as CRITICAL.
        """
        import tempfile, polars as pl
        df = pl.read_csv(FIXTURE_CSV)
        target_col = df.columns[-1]

        # Inject: add target as a feature column (pure leakage)
        df_leaked = df.with_columns(pl.col(target_col).alias("leaked_target_feature"))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df_leaked.write_csv(f.name)
            leaked_path = f.name

        s = initial_state("test-critic-leak", leaked_path)
        s = run_data_engineer(s)
        s = run_eda_agent(s)
        s = run_validation_architect(s)
        result = run_red_team_critic(s)

        os.unlink(leaked_path)

        verdict = result["critic_verdict"]
        assert verdict["overall_severity"] == "CRITICAL", (
            f"Injected target leakage should produce CRITICAL verdict. Got: {verdict['overall_severity']}. "
            f"Findings: {verdict['findings']}"
        )

    def test_critical_verdict_sets_hitl_required(self):
        """CRITICAL verdict must halt the pipeline."""
        import tempfile, polars as pl
        df = pl.read_csv(FIXTURE_CSV)
        target_col = df.columns[-1]
        df_leaked = df.with_columns(pl.col(target_col).alias("leaked_feature_2"))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df_leaked.write_csv(f.name)
            p = f.name

        s = initial_state("test-critic-halt", p)
        s = run_data_engineer(s)
        s = run_eda_agent(s)
        s = run_validation_architect(s)
        result = run_red_team_critic(s)
        os.unlink(p)

        assert result.get("hitl_required") is True, (
            "CRITICAL verdict must set hitl_required=True to halt the pipeline."
        )

    def test_critical_findings_have_replan_instructions(self):
        """CRITICAL findings must specify which nodes to rerun."""
        import tempfile, polars as pl
        df = pl.read_csv(FIXTURE_CSV)
        target_col = df.columns[-1]
        df_leaked = df.with_columns(pl.col(target_col).alias("leaked_feature_3"))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df_leaked.write_csv(f.name)
            p = f.name

        s = initial_state("test-critic-replan", p)
        s = run_data_engineer(s)
        s = run_eda_agent(s)
        s = run_validation_architect(s)
        result = run_red_team_critic(s)
        os.unlink(p)

        for finding in result["critic_verdict"]["findings"]:
            if finding["severity"] == "CRITICAL":
                ri = finding.get("replan_instructions", {})
                assert "rerun_nodes" in ri, "CRITICAL finding missing replan_instructions.rerun_nodes"
                assert isinstance(ri["rerun_nodes"], list)
                assert len(ri["rerun_nodes"]) > 0, "rerun_nodes must name at least one node to rerun"
                assert "remove_features" in ri


class TestCriticPreprocessingAudit:

    def test_preprocessing_leakage_code_triggers_critical(self):
        """
        Feed code containing a pre-split scaler fit into the critic.
        The code audit vector must flag it as CRITICAL.
        """
        from agents.red_team_critic import _check_preprocessing_leakage

        # Code that fits a scaler before the train/val split
        bad_code = """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

X = df.drop('target', axis=1)
y = df['target']

# BUG: scaler fitted on full X before CV loop
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # <- leakage

kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
"""
        result = _check_preprocessing_leakage(bad_code)
        assert result["verdict"] == "CRITICAL", (
            f"Pre-split scaler fit should trigger CRITICAL. Got: {result['verdict']}. "
            f"Evidence: {result.get('findings', [])}"
        )

    def test_clean_preprocessing_code_passes(self):
        """Code that correctly fits preprocessors inside the fold must not trigger."""
        from agents.red_team_critic import _check_preprocessing_leakage

        good_code = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    # Correct: scaler fitted only on training fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
"""
        result = _check_preprocessing_leakage(good_code)
        assert result["verdict"] == "OK", (
            f"Correct preprocessing code should not trigger. Got: {result['verdict']}. "
            f"Findings: {result.get('findings', [])}"
        )


class TestCriticPRCurveAudit:

    def test_majority_class_model_triggers_critical_on_imbalanced(self):
        """
        Simulate a model that always predicts the majority class:
        all predicted probabilities are near 0.
        On an imbalanced dataset, this should trigger CRITICAL.
        """
        from agents.red_team_critic import _check_pr_curve_imbalance

        n        = 1000
        minority = int(n * 0.05)  # 5% minority
        y_true   = np.array([1] * minority + [0] * (n - minority))
        # Model that always predicts 0 (majority class)
        y_prob   = np.zeros(n) + 0.01

        result = _check_pr_curve_imbalance(y_true, y_prob, imbalance_ratio=0.05, target_type="binary")
        assert result["verdict"] == "CRITICAL", (
            f"Majority-class-only model on imbalanced data must produce CRITICAL. Got: {result['verdict']}. "
            f"Best recall: {result.get('best_recall')}"
        )

    def test_good_minority_recall_passes_pr_audit(self):
        """A model with >= 50% recall on minority class must not trigger."""
        from agents.red_team_critic import _check_pr_curve_imbalance

        n          = 1000
        minority   = int(n * 0.08)
        y_true     = np.array([1] * minority + [0] * (n - minority))
        # Good model: high probability for true positives
        y_prob     = np.array([0.9] * minority + [0.1] * (n - minority))

        result = _check_pr_curve_imbalance(y_true, y_prob, imbalance_ratio=0.08, target_type="binary")
        assert result["verdict"] == "OK", (
            f"Good recall model should not trigger PR audit. Got: {result['verdict']}"
        )

    def test_pr_audit_skipped_for_balanced_data(self):
        """PR audit must not run on balanced datasets (imbalance_ratio >= 0.15)."""
        from agents.red_team_critic import _check_pr_curve_imbalance

        y_true = np.array([0, 1] * 500)
        y_prob = np.random.rand(1000)
        result = _check_pr_curve_imbalance(y_true, y_prob, imbalance_ratio=0.5, target_type="binary")
        assert result["verdict"] == "OK"
        assert "skipped" in result.get("note", "").lower()

    def test_pr_audit_skipped_for_non_binary_target(self):
        """PR audit only applies to binary classification."""
        from agents.red_team_critic import _check_pr_curve_imbalance

        result = _check_pr_curve_imbalance(
            np.array([0, 1, 2] * 100), None, imbalance_ratio=0.05, target_type="multiclass"
        )
        assert result["verdict"] == "OK"


class TestCriticMemorySchemaIntegration:

    def test_fingerprint_built_from_state(self):
        """After running full pipeline, competition_fingerprint must be populated."""
        s = initial_state("test-fp", FIXTURE_CSV)
        s = run_data_engineer(s)
        s = run_eda_agent(s)
        s = run_validation_architect(s)

        from memory.memory_schema import build_competition_fingerprint
        fp = build_competition_fingerprint(s)

        required_keys = {
            "task_type", "imbalance_ratio", "n_categorical_high_cardinality",
            "n_rows_bucket", "has_temporal_feature", "n_features_bucket", "target_type"
        }
        missing = required_keys - set(fp.keys())
        assert not missing, f"Fingerprint missing keys: {missing}"

    def test_fingerprint_text_is_non_empty_and_semantic(self):
        from memory.memory_schema import fingerprint_to_text, build_competition_fingerprint
        s  = initial_state("test-fptext", FIXTURE_CSV)
        s  = run_data_engineer(s)
        s  = run_eda_agent(s)
        s  = run_validation_architect(s)
        fp = build_competition_fingerprint(s)
        text = fingerprint_to_text(fp)
        assert len(text) > 50, "Fingerprint text too short to be useful for embedding"
        assert "task" in text.lower() or "classif" in text.lower() or "tabular" in text.lower()

    def test_pattern_store_and_retrieve_round_trip(self):
        from memory.memory_schema import (
            build_competition_fingerprint, store_pattern, query_similar_competitions
        )
        s  = initial_state("test-mem-rt", FIXTURE_CSV)
        s  = run_data_engineer(s)
        s  = run_eda_agent(s)
        s  = run_validation_architect(s)
        fp = build_competition_fingerprint(s)

        pid = store_pattern(
            fingerprint=fp,
            validated_approaches=[{"approach": "LGBM + log-transform", "cv_improvement": 0.02, "competitions": ["test"]}],
            failed_approaches=[],
            competition_name="test-round-trip",
            confidence=0.65,
        )
        assert pid is not None and len(pid) > 0

        results = query_similar_competitions(fp, n_results=3)
        assert len(results) >= 1, "Stored pattern must be retrievable"
        ids = [r["pattern_id"] for r in results]
        assert pid in ids, f"Stored pattern {pid} not in query results: {ids}"

    def test_query_returns_empty_list_not_none_when_no_patterns(self, tmp_path):
        """query_similar_competitions must return [] not None when collection is empty."""
        from memory.memory_schema import query_similar_competitions
        fp = {"task_type": "tabular", "imbalance_ratio": 0.5, "n_categorical_high_cardinality": 0,
              "n_rows_bucket": "medium", "has_temporal_feature": False,
              "n_features_bucket": "medium", "target_type": "binary"}
        results = query_similar_competitions(fp, n_results=5)
        assert isinstance(results, list), f"Expected list, got {type(results)}"
```

---

## END OF DAY CHECKLIST

```bash
# 1. Regression — Phase 1 + Day 8 + Day 9 baselines unchanged
pytest tests/regression/ -v

# 2. All contracts
pytest tests/contracts/ -v

# 3. Critic contract specifically
pytest tests/contracts/test_critic_contract.py -v -s

# 4. Memory schema verification
python -c "
from memory.memory_schema import fingerprint_to_text, query_similar_competitions
fp = {'task_type': 'tabular', 'imbalance_ratio': 0.05, 'n_categorical_high_cardinality': 3,
      'n_rows_bucket': 'medium', 'has_temporal_feature': False, 'n_features_bucket': 'medium', 'target_type': 'binary'}
print('Fingerprint text:', fingerprint_to_text(fp))
results = query_similar_competitions(fp)
print('Warm-start patterns:', len(results))
print('[PASS] Memory schema v2 operational')
"

# 5. End-to-end with critic in pipeline
python -c "
from core.state import initial_state
from agents.competition_intel import run_competition_intel
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect
from agents.red_team_critic import run_red_team_critic

state = initial_state('spaceship-titanic', 'data/spaceship_titanic/train.csv')
state = run_competition_intel(state)
state = run_data_engineer(state)
state = run_eda_agent(state)
state = run_validation_architect(state)
state = run_red_team_critic(state)

print('Critic severity:', state['critic_severity'])
print('Vectors checked:', state['critic_verdict']['vectors_checked'])
print('Findings:', len(state['critic_verdict']['findings']))
print('Pipeline halted:', state.get('hitl_required', False))
assert state['critic_severity'] != 'unchecked'
assert len(state['critic_verdict']['vectors_checked']) == 6
print('[PASS] Critic in full pipeline')
"

# 6. Commit
git add .
git commit -m 'Day 10: memory schema v2 patterns, red_team_critic all 6 vectors, critic contract — all green'
git push origin phase-2
```

### Definition of Done for Day 10

- [ ] `memory/memory_schema.py` — `build_competition_fingerprint()`, `fingerprint_to_text()`, `store_pattern()`, `query_similar_competitions()`, `get_warm_start_priors()`
- [ ] `competition_fingerprint` and `warm_start_priors` in ProfessorState
- [ ] `agents/red_team_critic.py` — all 6 vectors run, none skipped
- [ ] CRITICAL verdict sets `hitl_required=True`, `replan_requested=True`, halts pipeline
- [ ] Critic contract test — all tests green, file immutable
- [ ] `pytest tests/regression/` — green
- [ ] `pytest tests/contracts/` — green

---

## WHAT PODIUM WORK LOOKS LIKE ON THIS DAY

Day 10 is when Professor develops a conscience. After today, Professor should be able to tell you:

- *"Your preprocessing leakage check failed: scaler was fitted on the full dataset before the CV split. CV is inflated by approximately 3–5%. Rerunning data_engineer and ml_optimizer with corrected preprocessing."*
- *"Model achieves 94% accuracy but 12% recall on the minority class. This model is completely useless for fraud detection. Optimising for PR-AUC instead."*
- *"I found a similar competition in memory: Home Credit Default Risk (binary, severe imbalance, temporal features). Top validated approach: log-transform on exposure features + LGBM with class_weight='balanced'. Confidence: 0.71."*
- *"Train/test adversarial classifier AUC 0.74 — feature distributions differ significantly. Top drift features: ['age_bin', 'transaction_count', 'device_type']. These will degrade on the LB."*

If Professor cannot say any of those things after Day 10, something is missing.