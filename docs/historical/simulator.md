You are building the Professor Agent historical competition harness.
This is a benchmarking system that measures Professor's real performance against known competition outcomes without waiting for live competitions.

Read CLAUDE.md, AGENTS.md, and core/professor.py before making any changes.

---

## WHAT YOU ARE BUILDING

A harness that:
1. Downloads historical Kaggle competition data
2. Holds out 20% as a "private test set" (Professor never sees these labels)
3. Gives Professor the remaining 80% as its training data
4. Runs the full Professor pipeline
5. Scores Professor's predictions against the 20% held-out
6. Maps that score to a simulated leaderboard percentile and medal
7. Writes a benchmark_report.json with all metrics

This is the primary way we measure Professor's improvement over time.

---

## FILES TO CREATE

Create all of the following files exactly as specified.

### FILE 1: tools/harness/__init__.py
Empty file.

### FILE 2: tools/harness/competition_registry.py

```python
"""
Registry of historical competitions with metadata and approximate LB percentile curves.
Percentile curve: maps a raw score -> approximate percentile rank (0-100, 100 = top of LB).
Medal thresholds (approximate):
  Gold:   top 3%  -> percentile >= 97
  Silver: top 10% -> percentile >= 90
  Bronze: top 20% -> percentile >= 80
"""

from dataclasses import dataclass, field
import numpy as np

BETTER_IS_HIGHER = {"accuracy", "auc", "f1"}
BETTER_IS_LOWER  = {"rmse", "rmsle", "mae", "logloss", "log_loss", "brier_score"}


@dataclass
class LeaderboardCurve:
    breakpoints: list        # [(score, percentile), ...] sorted by score ascending
    higher_is_better: bool

    def score_to_percentile(self, score: float) -> float:
        if not self.breakpoints:
            return 50.0
        scores = [b[0] for b in self.breakpoints]
        pcts   = [b[1] for b in self.breakpoints]
        return float(np.interp(score, scores, pcts))


@dataclass
class CompetitionSpec:
    competition_id:       str
    display_name:         str
    task_type:            str   # "binary_classification" | "regression" | "multiclass"
    target_column:        str
    id_column:            str
    evaluation_metric:    str   # "accuracy" | "rmsle" | "auc" | "logloss"
    train_file:           str
    test_file:            str
    sample_submission_file: str
    lb_curve:             LeaderboardCurve
    known_winning_features: list = field(default_factory=list)
    known_pitfalls:       list  = field(default_factory=list)
    gold_threshold:       float = 0.0
    silver_threshold:     float = 0.0
    bronze_threshold:     float = 0.0


COMPETITION_REGISTRY = {

    "spaceship-titanic": CompetitionSpec(
        competition_id="spaceship-titanic",
        display_name="Spaceship Titanic",
        task_type="binary_classification",
        target_column="Transported",
        id_column="PassengerId",
        evaluation_metric="accuracy",
        train_file="train.csv",
        test_file="test.csv",
        sample_submission_file="sample_submission.csv",
        lb_curve=LeaderboardCurve(
            breakpoints=[
                (0.770, 5.0), (0.785, 20.0), (0.795, 40.0), (0.803, 60.0),
                (0.810, 80.0), (0.815, 90.0), (0.820, 95.0), (0.825, 99.0),
            ],
            higher_is_better=True,
        ),
        known_winning_features=["CryoSleep", "Cabin_deck", "total_spend", "GroupSize"],
        known_pitfalls=["Target-encoding without fold isolation", "Ignoring group structure in PassengerId"],
        gold_threshold=0.820,
        silver_threshold=0.813,
        bronze_threshold=0.805,
    ),

    "titanic": CompetitionSpec(
        competition_id="titanic",
        display_name="Titanic — Machine Learning from Disaster",
        task_type="binary_classification",
        target_column="Survived",
        id_column="PassengerId",
        evaluation_metric="accuracy",
        train_file="train.csv",
        test_file="test.csv",
        sample_submission_file="gender_submission.csv",
        lb_curve=LeaderboardCurve(
            breakpoints=[
                (0.750, 10.0), (0.770, 30.0), (0.780, 50.0),
                (0.790, 70.0), (0.800, 85.0), (0.810, 92.0), (0.820, 96.0),
            ],
            higher_is_better=True,
        ),
        known_winning_features=["Title", "FamilySize", "IsAlone", "Deck"],
        known_pitfalls=["Overfitting on 891 training rows", "Name-based leakage via titles"],
        gold_threshold=0.820,
        silver_threshold=0.800,
        bronze_threshold=0.780,
    ),

    "house-prices-advanced-regression-techniques": CompetitionSpec(
        competition_id="house-prices-advanced-regression-techniques",
        display_name="House Prices — Advanced Regression Techniques",
        task_type="regression",
        target_column="SalePrice",
        id_column="Id",
        evaluation_metric="rmsle",
        train_file="train.csv",
        test_file="test.csv",
        sample_submission_file="sample_submission.csv",
        lb_curve=LeaderboardCurve(
            breakpoints=[
                (0.160, 5.0), (0.140, 20.0), (0.130, 40.0), (0.125, 60.0),
                (0.120, 80.0), (0.115, 90.0), (0.110, 95.0), (0.105, 99.0),
            ],
            higher_is_better=False,
        ),
        known_winning_features=["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageArea"],
        known_pitfalls=["Not log-transforming SalePrice", "Outliers in GrLivArea"],
        gold_threshold=0.110,
        silver_threshold=0.118,
        bronze_threshold=0.125,
    ),
}
```

### FILE 3: tools/harness/data_downloader.py

```python
"""
Downloads competition data via Kaggle API and creates the 80/20 split.
The 20% held-out set is the private test — Professor never sees these labels.
Split is stratified for classification, random for regression.
Fixed seed=42. Split is saved to disk so the same split is reused across runs.
"""

import json, hashlib, subprocess
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split

HARNESS_DATA_DIR = Path("tests/harness/data")
SPLIT_SEED       = 42
HOLDOUT_FRACTION = 0.20


def download_competition(competition_id: str, force: bool = False) -> Path:
    """Downloads via Kaggle CLI. Skips if data already present."""
    target_dir = HARNESS_DATA_DIR / competition_id / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(target_dir.iterdir()):
        print(f"[harness] Data already present at {target_dir}. Skipping download.")
        return target_dir

    print(f"[harness] Downloading {competition_id}...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition_id,
         "-p", str(target_dir), "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed for '{competition_id}':\n{result.stderr}\n"
            "Ensure KAGGLE_USERNAME and KAGGLE_KEY are in .env and you have "
            "accepted the competition rules on kaggle.com."
        )
    print(f"[harness] Downloaded to {target_dir}")
    return target_dir


def make_split(competition_id: str, spec, force: bool = False):
    """
    Loads full training data, splits 80/20.
    Returns (professor_train, private_test, original_test) as Polars DataFrames.
    Saves to disk so the same split is reused.
    """
    split_dir = HARNESS_DATA_DIR / competition_id / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    prof_path    = split_dir / "professor_train.csv"
    private_path = split_dir / "private_test.csv"
    test_path    = split_dir / "original_test.csv"

    if not force and prof_path.exists() and private_path.exists():
        print(f"[harness] Split already exists. Reusing.")
        return pl.read_csv(prof_path), pl.read_csv(private_path), pl.read_csv(test_path)

    raw_dir    = HARNESS_DATA_DIR / competition_id / "raw"
    full_train = pl.read_csv(raw_dir / spec.train_file)
    orig_test  = pl.read_csv(raw_dir / spec.test_file)

    y        = full_train[spec.target_column].to_numpy()
    stratify = y if spec.task_type in ("binary_classification", "multiclass") else None

    train_idx, holdout_idx = train_test_split(
        np.arange(len(full_train)),
        test_size=HOLDOUT_FRACTION,
        random_state=SPLIT_SEED,
        stratify=stratify,
    )

    professor_train = full_train[train_idx.tolist()]
    private_test    = full_train[holdout_idx.tolist()]

    professor_train.write_csv(prof_path)
    private_test.write_csv(private_path)
    orig_test.write_csv(test_path)

    # Audit trail
    meta = {
        "competition_id":     competition_id,
        "split_seed":         SPLIT_SEED,
        "holdout_fraction":   HOLDOUT_FRACTION,
        "n_professor_train":  len(professor_train),
        "n_private_test":     len(private_test),
        "data_hash":          hashlib.md5(full_train.write_csv().encode()).hexdigest(),
    }
    (split_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[harness] Split: {len(professor_train)} train rows, "
        f"{len(private_test)} private test rows."
    )
    return professor_train, private_test, orig_test
```

### FILE 4: tools/harness/scorer.py

```python
"""
Metric-aware scorer: computes the official competition metric
between Professor's predictions and the private held-out set.
"""

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error,
    log_loss, mean_absolute_error,
)


def score_predictions(private_test: pl.DataFrame, predictions: pl.DataFrame, spec) -> dict:
    """
    Aligns predictions with private_test on id_column, computes official metric.
    predictions must have columns: [id_column, "prediction"].
    """
    merged = private_test.join(
        predictions.rename({"prediction": "prof_pred"}),
        on=spec.id_column,
        how="left",
    )

    n_missing = int(merged["prof_pred"].is_null().sum())
    if n_missing > 0:
        print(f"[scorer] WARNING: {n_missing} IDs missing. Filling with baseline.")
        if spec.task_type == "regression":
            fill = float(merged[spec.target_column].mean())
        else:
            vals, counts = np.unique(
                merged[spec.target_column].drop_nulls().to_numpy(), return_counts=True
            )
            fill = float(vals[counts.argmax()])
        merged = merged.with_columns(pl.col("prof_pred").fill_null(fill))

    y_true = merged[spec.target_column].to_numpy()
    y_pred = merged["prof_pred"].to_numpy()
    metric = spec.evaluation_metric.lower().replace("-", "_")

    return {
        "private_score":    round(float(_compute(y_true, y_pred, metric)), 6),
        "metric":           metric,
        "n_scored":         len(merged),
        "n_missing_ids":    n_missing,
        "higher_is_better": metric in {"accuracy", "auc", "f1"},
    }


def _compute(y_true, y_pred, metric: str) -> float:
    if metric == "accuracy":
        return accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    elif metric == "auc":
        return roc_auc_score(y_true, y_pred)
    elif metric in ("rmsle", "root_mean_squared_log_error"):
        yt = np.log1p(np.clip(y_true.astype(float), 0, None))
        yp = np.log1p(np.clip(y_pred.astype(float), 0, None))
        return float(np.sqrt(mean_squared_error(yt, yp)))
    elif metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    elif metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    elif metric in ("logloss", "log_loss"):
        return float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7)))
    else:
        raise ValueError(f"Unknown metric: '{metric}'")
```

### FILE 5: tools/harness/leaderboard_comparator.py

```python
"""Maps a private score to a simulated leaderboard percentile and medal."""


def compare_to_leaderboard(private_score: float, spec) -> dict:
    percentile = spec.lb_curve.score_to_percentile(private_score)
    medal      = _medal(percentile)
    hib        = spec.lb_curve.higher_is_better

    gap_to_bronze = (private_score - spec.bronze_threshold) if hib else (spec.bronze_threshold - private_score)
    gap_to_gold   = (private_score - spec.gold_threshold)   if hib else (spec.gold_threshold   - private_score)

    return {
        "simulated_percentile": round(percentile, 1),
        "simulated_medal":      medal,
        "gap_to_bronze":        round(gap_to_bronze, 5),
        "gap_to_gold":          round(gap_to_gold, 5),
        "bronze_threshold":     spec.bronze_threshold,
        "silver_threshold":     spec.silver_threshold,
        "gold_threshold":       spec.gold_threshold,
    }


def _medal(percentile: float) -> str:
    if percentile >= 97:  return "Gold"
    if percentile >= 90:  return "Silver"
    if percentile >= 80:  return "Bronze"
    return "None"
```

### FILE 6: tools/harness/harness_runner.py

```python
"""
Main harness orchestrator. Runs all 5 steps and writes benchmark_report.json.
"""

import os, json, time, traceback
from pathlib import Path
from datetime import datetime

from tools.harness.competition_registry import COMPETITION_REGISTRY
from tools.harness.data_downloader import download_competition, make_split
from tools.harness.scorer import score_predictions
from tools.harness.leaderboard_comparator import compare_to_leaderboard

RESULTS_DIR = Path("tests/harness/results")


def run_harness(
    competition_id: str,
    session_id: str = None,
    fast_mode: bool = False,
    force_download: bool = False,
    force_split: bool = False,
) -> dict:
    if competition_id not in COMPETITION_REGISTRY:
        raise ValueError(
            f"Unknown competition: '{competition_id}'. "
            f"Available: {list(COMPETITION_REGISTRY.keys())}"
        )

    spec       = COMPETITION_REGISTRY[competition_id]
    session_id = session_id or f"harness_{competition_id}_{int(time.time())}"
    result_dir = RESULTS_DIR / session_id
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PROFESSOR HARNESS: {spec.display_name}")
    print(f"Session: {session_id} | Fast: {fast_mode}")
    print(f"{'='*60}\n")

    t_total = time.time()

    # Step 1: Download
    print("[1/5] Downloading competition data...")
    download_competition(competition_id, force=force_download)

    # Step 2: Split
    print("[2/5] Creating 80/20 split...")
    professor_train, private_test, original_test = make_split(
        competition_id, spec, force=force_split
    )

    # Step 3: Run Professor
    print("[3/5] Running Professor pipeline...")
    if fast_mode:
        os.environ["PROFESSOR_FAST_MODE"] = "1"

    import polars as pl
    train_path = result_dir / "professor_train.csv"
    test_path  = result_dir / "professor_test.csv"
    professor_train.write_csv(train_path)
    private_test.drop(spec.target_column).write_csv(test_path)

    state = {
        "competition_name":          spec.competition_id,
        "session_id":                session_id,
        "train_path":                str(train_path),
        "test_path":                 str(test_path),
        "target_column":             spec.target_column,
        "id_column":                 spec.id_column,
        "evaluation_metric":         spec.evaluation_metric,
        "task_type":                 spec.task_type,
        "budget_limit_usd":          5.0,
        "budget_remaining_usd":      5.0,
        "dag_version":               1,
        "external_data_allowed":     False,
        "hitl_required":             False,
        "replan_requested":          False,
        "critic_severity":           "unchecked",
        "model_registry":            {},
        "features_dropped":          [],
        "feature_order":             [],
        "current_node_failure_count": 0,
    }

    t_pipeline    = time.time()
    final_state   = None
    pipeline_error = None

    try:
        from core.professor import run_professor
        final_state = run_professor(state)
    except Exception as e:
        pipeline_error = f"{type(e).__name__}: {str(e)[:500]}"
        print(f"[harness] Pipeline error: {pipeline_error}")
        traceback.print_exc()

    pipeline_seconds = time.time() - t_pipeline

    # Step 4: Score
    print("[4/5] Scoring against private held-out set...")
    score_result = {"private_score": None, "error": "pipeline_failed"}

    if final_state is not None:
        # Check state for submission_df first
        if "submission_df" in final_state:
            try:
                score_result = score_predictions(private_test, final_state["submission_df"], spec)
            except Exception as e:
                score_result = {"private_score": None, "error": str(e)[:200]}
        else:
            # Fall back to submission CSV on disk
            sub_path = Path(f"outputs/{session_id}/submission.csv")
            if sub_path.exists():
                try:
                    preds = pl.read_csv(sub_path)
                    score_result = score_predictions(private_test, preds, spec)
                except Exception as e:
                    score_result = {"private_score": None, "error": str(e)[:200]}

    # Step 5: Compare to LB
    print("[5/5] Comparing to leaderboard...")
    lb_result = {}
    if score_result.get("private_score") is not None:
        lb_result = compare_to_leaderboard(score_result["private_score"], spec)

    # Build report
    cv_score = final_state.get("cv_mean") if final_state else None
    private  = score_result.get("private_score")
    cv_gap   = round(abs(cv_score - private), 6) if (cv_score and private) else None

    report = {
        "session_id":             session_id,
        "competition_id":         competition_id,
        "competition_name":       spec.display_name,
        "run_timestamp":          datetime.utcnow().isoformat(),
        "private_score":          private,
        "cv_score":               cv_score,
        "cv_lb_gap":              cv_gap,
        "metric":                 spec.evaluation_metric,
        **lb_result,
        "total_runtime_seconds":  round(time.time() - t_total, 1),
        "pipeline_seconds":       round(pipeline_seconds, 1),
        "pipeline_error":         pipeline_error,
        "winning_model_type":     _get(final_state, "winning_model_type"),
        "n_features_final":       _get(final_state, "n_features_final"),
        "n_features_dropped":     (_get(final_state, "stage1_drop_count", 0)
                                   + _get(final_state, "stage2_drop_count", 0)),
        "n_pseudo_label_iterations": _get(final_state, "pseudo_label_iterations", 0),
        "estimated_llm_cost_usd": _estimate_cost(_get(final_state, "total_tokens_used", 0)),
        "optimisation_flags": {
            "PROFESSOR_FAST_MODE":    os.getenv("PROFESSOR_FAST_MODE", "0"),
            "PROFESSOR_DEVICE":       os.getenv("PROFESSOR_DEVICE", "cpu"),
        },
        "n_professor_train_rows": len(professor_train),
        "n_private_test_rows":    len(private_test),
    }

    report_path = result_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    return report


def _get(state, key, default=None):
    return state.get(key, default) if state else default

def _estimate_cost(tokens: int) -> float:
    return round((tokens * 0.80 * 0.05 + tokens * 0.20 * 0.59) / 1_000_000, 4)

def _print_summary(r: dict):
    print(f"\n{'='*60}")
    print(f"HARNESS COMPLETE: {r['competition_name']}")
    print(f"{'='*60}")
    print(f"  Private score:        {r.get('private_score', 'N/A')}")
    print(f"  CV score:             {r.get('cv_score', 'N/A')}")
    print(f"  CV/Private gap:       {r.get('cv_lb_gap', 'N/A')}")
    print(f"  Simulated percentile: {r.get('simulated_percentile', 'N/A')}%")
    print(f"  Simulated medal:      {r.get('simulated_medal', 'N/A')}")
    print(f"  Gap to gold:          {r.get('gap_to_gold', 'N/A')}")
    print(f"  Runtime:              {r.get('total_runtime_seconds')}s")
    print(f"  Winning model:        {r.get('winning_model_type', 'N/A')}")
    print(f"  Features (final):     {r.get('n_features_final', 'N/A')}")
    print(f"  Est. LLM cost:        ${r.get('estimated_llm_cost_usd', 0)}")
    if r.get("pipeline_error"):
        print(f"\n  ERROR: {r['pipeline_error']}")
    print(f"{'='*60}\n")
```

### FILE 7: tests/harness/__init__.py
Empty file.

### FILE 8: tests/harness/run_harness.py

```python
"""
CLI entry point for the historical competition harness.

Usage:
  python tests/harness/run_harness.py --list
  python tests/harness/run_harness.py -c spaceship-titanic --fast
  python tests/harness/run_harness.py -c spaceship-titanic -s my_run_01
  python tests/harness/run_harness.py -c titanic
  python tests/harness/run_harness.py -c house-prices-advanced-regression-techniques
"""

import argparse, sys
from tools.harness.harness_runner import run_harness
from tools.harness.competition_registry import COMPETITION_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Professor Agent — Historical Competition Harness")
    parser.add_argument("--competition", "-c", type=str)
    parser.add_argument("--session",     "-s", type=str)
    parser.add_argument("--fast",        action="store_true", help="Enable FAST_MODE (skip null importance Stage 2)")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-split",    action="store_true")
    parser.add_argument("--list", "-l",  action="store_true", help="List available competitions")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable competitions:")
        for cid, spec in COMPETITION_REGISTRY.items():
            print(f"  {cid:<55} metric={spec.evaluation_metric}")
        return

    if not args.competition:
        parser.print_help()
        sys.exit(1)

    report = run_harness(
        competition_id=args.competition,
        session_id=args.session,
        fast_mode=args.fast,
        force_download=args.force_download,
        force_split=args.force_split,
    )

    sys.exit(0 if report.get("simulated_medal", "None") != "None" else 1)


if __name__ == "__main__":
    main()
```

---

## REQUIREMENTS

After creating all 8 files:

1. Create the directory structure:
   mkdir -p tools/harness tests/harness/data tests/harness/results

2. Add to requirements.txt if not already present:
   scikit-learn
   polars
   kaggle

3. Verify imports work:
   python -c "from tools.harness.competition_registry import COMPETITION_REGISTRY; print(list(COMPETITION_REGISTRY.keys()))"
   python -c "from tools.harness.harness_runner import run_harness; print('OK')"

4. Verify the CLI works:
   python tests/harness/run_harness.py --list

5. DO NOT run a full harness yet. Just confirm the files are created and imports work.

---

## IMPORTANT CONSTRAINTS

- DO NOT modify any agent files (data_engineer, ml_optimizer, etc.)
- DO NOT run `kaggle competitions download` unless explicitly asked
- DO NOT run the full pipeline (run_harness) unless explicitly asked
- The harness is READ-ONLY with respect to all Professor agent code
- Professor's submission must be written to outputs/{session_id}/submission.csv OR
  returned in state["submission_df"] as a Polars DataFrame with columns [id_column, "prediction"]
  Check which convention Professor currently uses in core/professor.py and submission_strategist.py
  and make sure harness_runner.py reads from the right place

---

## AFTER CREATING FILES

Run this verification and report the output:

  python -c "
  from tools.harness.competition_registry import COMPETITION_REGISTRY
  from tools.harness.data_downloader import HARNESS_DATA_DIR
  from tools.harness.scorer import score_predictions
  from tools.harness.leaderboard_comparator import compare_to_leaderboard
  from tools.harness.harness_runner import run_harness
  print('All imports OK')
  print('Competitions registered:', list(COMPETITION_REGISTRY.keys()))
  "

If all imports succeed, report: "Harness ready. Run with: python tests/harness/run_harness.py --list"
If any import fails, fix it before reporting completion.