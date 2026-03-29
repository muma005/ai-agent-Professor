# ⚡ Cloud Simulation Infrastructure — Implementation Guide

## Modal Setup for Professor Private Leaderboard Simulator

---

> **Purpose:** Run Professor benchmarks in 12 minutes instead of 2 hours, in the cloud, for free.
> **Platform:** Modal (serverless Python compute)
> **Cost:** $0 — fits within $30/month free tier
> **Setup time:** 30 minutes
> **Prerequisite:** Professor v1 pipeline functional, simulator components built

---

## Phase 0 — Environment Setup (5 minutes)

### 0.1 Install Modal CLI

```bash
pip install modal
```

### 0.2 Authenticate

```bash
modal setup
```

This opens a browser window. Log in (or create account). Token is saved locally at `~/.modal.toml`. One-time operation.

### 0.3 Verify Installation

```bash
modal --version
# Should print modal version

python -c "import modal; print('Modal imported successfully')"
```

### 0.4 Store API Keys as Modal Secrets

Professor needs Fireworks AI, Google AI Studio, and Kaggle API keys. Store them encrypted in Modal — never in code, never in git.

```bash
modal secret create professor-keys \
    FIREWORKS_API_KEY="fw_your_fireworks_key_here" \
    FIREWORKS_GLM_API_KEY="fw_your_fireworks_glm_key_here" \
    GOOGLE_API_KEY="AIza_your_google_key_here" \
    KAGGLE_USERNAME="your_kaggle_username" \
    KAGGLE_KEY="your_kaggle_api_key"
```

Verify:

```bash
modal secret list
# Should show: professor-keys
```

---

## Phase 1 — Project Structure (10 minutes)

### 1.1 Create Simulator Directory

Inside the Professor project root:

```bash
mkdir -p simulator/data
mkdir -p simulator/results
mkdir -p simulator/tests
```

### 1.2 Final File Structure

```
professor-agent/
├── simulator/
│   ├── __init__.py
│   ├── competition_registry.py      # Competition metadata + LB curves
│   ├── data_splitter.py             # Deterministic train/public/private splits
│   ├── leaderboard.py               # Simulated Kaggle LB (scores submissions)
│   ├── scorers.py                   # Metric implementations (match Kaggle exactly)
│   ├── report_generator.py          # Aggregate benchmark reports
│   ├── cloud_benchmark.py           # ★ Modal app — the cloud entry point
│   ├── data/                        # Downloaded + split competition data (gitignored)
│   │   ├── spaceship-titanic/
│   │   ├── titanic/
│   │   └── ...
│   ├── results/                     # Benchmark reports (tracked in git)
│   │   ├── benchmark_v2.0.json
│   │   └── ...
│   └── tests/
│       ├── test_splitter.py
│       ├── test_leaderboard.py
│       └── test_scorers.py
└── ...
```

### 1.3 Add to .gitignore

```bash
echo "simulator/data/" >> .gitignore
```

Competition data is cached on Modal's persistent volume, not in git.

---

## Phase 2 — Build Competition Registry (30 minutes)

### 2.1 Create `simulator/competition_registry.py`

```python
"""
Competition Registry — metadata for all benchmark competitions.

Each entry contains everything needed to simulate a competition:
- Task type, target column, evaluation metric
- LB percentile curves for medal calculation
- Split strategy matching the real competition's structure
- Domain classification for Domain Research Engine testing

LB percentile curves are scraped from actual Kaggle leaderboards
after competitions close. These are public data.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompetitionEntry:
    # ── Identity ──
    slug: str                    # Kaggle competition slug
    title: str                   # Human-readable name
    
    # ── Task Definition ──
    task_type: str               # binary | multiclass | regression
    target_column: str           # Column name of the target
    id_column: str               # Row identifier column
    metric: str                  # Evaluation metric name
    metric_direction: str        # maximize | minimize
    
    # ── LB Calibration ──
    # Keys = percentile (lower = better rank), Values = score threshold
    # Example: {10: 0.810} means top 10% scored >= 0.810
    lb_percentiles: dict = field(default_factory=dict)
    gold_threshold: float = 0.0   # Score for top ~10% (gold)
    silver_threshold: float = 0.0 # Score for top ~5%
    bronze_threshold: float = 0.0 # Score for top ~25%
    total_teams: int = 0
    
    # ── Split Configuration ──
    split_strategy: str = "stratified"  # stratified | temporal | group
    split_column: Optional[str] = None  # Column for temporal/group splits
    test_ratio: float = 0.40            # Fraction held out as test
    public_ratio: float = 0.30          # Fraction of test → public LB
    random_seed: int = 42               # Deterministic, reproducible
    
    # ── Data Source ──
    download_method: str = "kaggle_api"  # kaggle_api | manual | cached
    cached_path: Optional[str] = None
    requires_join: bool = False
    join_instructions: Optional[str] = None
    
    # ── Domain ──
    primary_domain: str = "general"
    sub_domain: str = ""


# ════════════════════════════════════════════════════════════
# REGISTRY — Start with 10 competitions, expand over time
# ════════════════════════════════════════════════════════════

REGISTRY = [
    # ── Binary Classification ─────────────────────────────
    CompetitionEntry(
        slug="spaceship-titanic",
        title="Spaceship Titanic",
        task_type="binary",
        target_column="Transported",
        id_column="PassengerId",
        metric="accuracy",
        metric_direction="maximize",
        lb_percentiles={5: 0.815, 10: 0.810, 25: 0.795, 50: 0.780, 75: 0.760, 90: 0.740},
        gold_threshold=0.810,
        silver_threshold=0.815,
        bronze_threshold=0.795,
        total_teams=2500,
        split_strategy="stratified",
        primary_domain="transport",
        sub_domain="passenger survival prediction",
    ),
    
    CompetitionEntry(
        slug="titanic",
        title="Titanic - Machine Learning from Disaster",
        task_type="binary",
        target_column="Survived",
        id_column="PassengerId",
        metric="accuracy",
        metric_direction="maximize",
        lb_percentiles={5: 0.810, 10: 0.800, 25: 0.790, 50: 0.775, 75: 0.755, 90: 0.730},
        gold_threshold=0.800,
        silver_threshold=0.810,
        bronze_threshold=0.790,
        total_teams=15000,
        split_strategy="stratified",
        primary_domain="transport",
        sub_domain="passenger survival prediction",
    ),

    # ── Regression ────────────────────────────────────────
    CompetitionEntry(
        slug="house-prices-advanced-regression-techniques",
        title="House Prices: Advanced Regression",
        task_type="regression",
        target_column="SalePrice",
        id_column="Id",
        metric="rmsle",
        metric_direction="minimize",
        lb_percentiles={5: 0.115, 10: 0.120, 25: 0.130, 50: 0.145, 75: 0.165, 90: 0.200},
        gold_threshold=0.120,
        silver_threshold=0.115,
        bronze_threshold=0.130,
        total_teams=5000,
        split_strategy="stratified",
        primary_domain="real_estate",
        sub_domain="residential house price prediction",
    ),
]


def get_competition(slug: str) -> CompetitionEntry:
    """Retrieve a competition entry by slug. Raises ValueError if not found."""
    for entry in REGISTRY:
        if entry.slug == slug:
            return entry
    available = [e.slug for e in REGISTRY]
    raise ValueError(
        f"Competition '{slug}' not in registry. "
        f"Available: {available}"
    )


def list_competitions() -> list:
    """Return list of all registered competition slugs."""
    return [e.slug for e in REGISTRY]
```

### 2.2 Why These Specific Competitions

| Competition | Why Included | What It Tests |
|---|---|---|
| spaceship-titanic | Daily v1 test dataset, baseline comparison available | Standard binary, tabular |
| titanic | Most well-known, huge competitor pool for calibration | Binary, small data, feature engineering heavy |
| house-prices | Regression, different metric (RMSLE), domain-specific | Regression, target transform, domain knowledge |

**Expand to 10 competitions by adding:** a multiclass problem (Playground Series), an imbalanced binary problem (ICR), a time-series problem, a healthcare domain problem, a finance domain problem, and 2 more diverse tabular problems. Add these iteratively as the simulator proves stable on the initial 3.

---

## Phase 3 — Build Data Splitter (30 minutes)

### 3.1 Create `simulator/data_splitter.py`

```python
"""
Data Splitter — deterministic, reproducible train/public/private splits.

CRITICAL INVARIANTS:
1. Same data + same seed = exact same split every time
2. No row appears in more than one partition
3. Target distribution is preserved across partitions
4. Test file has NO target column (same as real Kaggle)
5. Public/private labels are hidden files (dotfiles)
"""

import polars as pl
import numpy as np
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from simulator.competition_registry import CompetitionEntry


@dataclass
class SplitResult:
    train_path: str
    test_path: str                # Features only (public + private rows, shuffled)
    sample_submission_path: str
    public_labels_path: str       # HIDDEN — only leaderboard.py reads this
    private_labels_path: str      # HIDDEN — only leaderboard.py reads this
    n_train: int
    n_public: int
    n_private: int
    split_hash: str               # SHA-256 of split for reproducibility verification


def split_competition_data(
    data_path: str,
    entry: CompetitionEntry,
    output_dir: str,
) -> SplitResult:
    """
    Split competition data into train / public test / private test.
    
    Split ratios:
    - Train: 60% of total data
    - Public test: 12% of total data (30% of the 40% test)
    - Private test: 28% of total data (70% of the 40% test)
    
    Professor receives: train.csv (with target), test.csv (without target),
    sample_submission.csv. Identical to real Kaggle experience.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    df = pl.read_csv(data_path)
    target = entry.target_column
    id_col = entry.id_column
    seed = entry.random_seed
    
    # ── Validate ──
    assert target in df.columns, f"Target '{target}' not in columns: {df.columns}"
    assert id_col in df.columns, f"ID '{id_col}' not in columns: {df.columns}"
    
    # ── Step 1: Split into train (60%) and test (40%) ──
    if entry.split_strategy == "stratified":
        train_df, test_df = _stratified_split(df, target, entry.test_ratio, seed)
    elif entry.split_strategy == "temporal":
        train_df, test_df = _temporal_split(df, entry.split_column, entry.test_ratio)
    elif entry.split_strategy == "group":
        train_df, test_df = _group_split(df, entry.split_column, entry.test_ratio, seed)
    else:
        raise ValueError(f"Unknown split strategy: {entry.split_strategy}")
    
    # ── Step 2: Split test into public (30%) and private (70%) ──
    public_test, private_test = _stratified_split(test_df, target, 0.70, seed + 1)
    
    # ── Step 3: Validate distributions ──
    _validate_distribution(df, train_df, target, "train")
    
    # ── Step 4: Save partitions ──
    train_path = str(output / "train.csv")
    test_path = str(output / "test.csv")
    sample_sub_path = str(output / "sample_submission.csv")
    public_labels_path = str(output / ".public_labels.csv")
    private_labels_path = str(output / ".private_labels.csv")
    
    # Train: full data with target
    train_df.write_csv(train_path)
    
    # Test: public + private combined, shuffled, WITHOUT target
    full_test = pl.concat([public_test, private_test]).sample(
        fraction=1.0, shuffle=True, seed=seed + 2
    )
    full_test_features = full_test.drop(target)
    full_test_features.write_csv(test_path)
    
    # Hidden labels (dotfiles)
    public_test.select([id_col, target]).write_csv(public_labels_path)
    private_test.select([id_col, target]).write_csv(private_labels_path)
    
    # Sample submission
    default_pred = _default_prediction(df, target, entry.task_type)
    sample = full_test_features.select(id_col).with_columns(
        pl.lit(default_pred).alias(target)
    )
    sample.write_csv(sample_sub_path)
    
    # Compute split hash for reproducibility verification
    split_hash = _compute_split_hash(train_path, public_labels_path, private_labels_path)
    
    # Save metadata
    metadata = {
        "competition": entry.slug,
        "strategy": entry.split_strategy,
        "seed": seed,
        "n_total": len(df),
        "n_train": len(train_df),
        "n_public": len(public_test),
        "n_private": len(private_test),
        "split_hash": split_hash,
        "train_target_mean": float(train_df[target].mean()) if entry.task_type != "multiclass" else None,
    }
    (output / "split_metadata.json").write_text(json.dumps(metadata, indent=2))
    
    return SplitResult(
        train_path=train_path,
        test_path=test_path,
        sample_submission_path=sample_sub_path,
        public_labels_path=public_labels_path,
        private_labels_path=private_labels_path,
        n_train=len(train_df),
        n_public=len(public_test),
        n_private=len(private_test),
        split_hash=split_hash,
    )


def _stratified_split(df, target, test_ratio, seed):
    """Stratified split preserving target distribution."""
    n = len(df)
    n_test = int(n * test_ratio)
    
    # For classification: stratify by target
    # For regression: stratify by target quantile bins
    if df[target].dtype in [pl.Utf8, pl.Categorical] or df[target].n_unique() < 20:
        # Classification — stratify by class
        indices = np.arange(n)
        rng = np.random.RandomState(seed)
        
        target_values = df[target].to_numpy()
        test_indices = []
        
        for cls in np.unique(target_values):
            cls_indices = indices[target_values == cls]
            rng.shuffle(cls_indices)
            n_cls_test = max(1, int(len(cls_indices) * test_ratio))
            test_indices.extend(cls_indices[:n_cls_test].tolist())
        
        test_indices = sorted(test_indices)
        train_indices = sorted(set(range(n)) - set(test_indices))
    else:
        # Regression — stratify by quantile bins
        quantiles = df[target].quantile([0.2, 0.4, 0.6, 0.8]).to_list()
        bins = [-float("inf")] + quantiles + [float("inf")]
        bin_labels = list(range(len(bins) - 1))
        
        target_np = df[target].to_numpy()
        bin_assignments = np.digitize(target_np, bins[1:-1])
        
        indices = np.arange(n)
        rng = np.random.RandomState(seed)
        test_indices = []
        
        for b in np.unique(bin_assignments):
            b_indices = indices[bin_assignments == b]
            rng.shuffle(b_indices)
            n_b_test = max(1, int(len(b_indices) * test_ratio))
            test_indices.extend(b_indices[:n_b_test].tolist())
        
        test_indices = sorted(test_indices)
        train_indices = sorted(set(range(n)) - set(test_indices))
    
    return df[train_indices], df[test_indices]


def _temporal_split(df, time_column, test_ratio):
    """Temporal split — last N% as test. No shuffling."""
    df_sorted = df.sort(time_column)
    split_idx = int(len(df_sorted) * (1 - test_ratio))
    return df_sorted[:split_idx], df_sorted[split_idx:]


def _group_split(df, group_column, test_ratio, seed):
    """Group split — no group appears in both train and test."""
    groups = df[group_column].unique().to_list()
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    n_test_groups = max(1, int(len(groups) * test_ratio))
    test_groups = set(groups[:n_test_groups])
    
    test_df = df.filter(pl.col(group_column).is_in(list(test_groups)))
    train_df = df.filter(~pl.col(group_column).is_in(list(test_groups)))
    return train_df, test_df


def _validate_distribution(full_df, train_df, target, name):
    """Validate that split didn't distort target distribution."""
    from scipy.stats import ks_2samp
    
    full_vals = full_df[target].drop_nulls().to_numpy().astype(float)
    split_vals = train_df[target].drop_nulls().to_numpy().astype(float)
    
    stat, pvalue = ks_2samp(full_vals, split_vals)
    if pvalue < 0.01:
        print(f"[WARNING] {name} split has significantly different target "
              f"distribution (KS p={pvalue:.4f}). Consider re-seeding.")


def _default_prediction(df, target, task_type):
    """Generate a default prediction for sample_submission."""
    if task_type == "binary":
        return 0
    elif task_type == "multiclass":
        return df[target].mode().to_list()[0]
    elif task_type == "regression":
        return float(df[target].median())
    return 0


def _compute_split_hash(train_path, public_path, private_path):
    """SHA-256 hash of all three split files — for reproducibility verification."""
    sha = hashlib.sha256()
    for path in [train_path, public_path, private_path]:
        with open(path, "rb") as f:
            sha.update(f.read())
    return sha.hexdigest()[:16]


def ensure_data_cached(entry: CompetitionEntry, cache_dir: str):
    """
    Download competition data if not already cached.
    Uses Kaggle API. Falls back to manual instruction if API fails.
    """
    cache_path = Path(cache_dir) / entry.slug
    full_data = cache_path / "full_data.csv"
    
    if full_data.exists():
        return  # already cached
    
    cache_path.mkdir(parents=True, exist_ok=True)
    
    if entry.download_method == "kaggle_api":
        import subprocess
        try:
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", entry.slug, "-p", str(cache_path)],
                check=True, capture_output=True, text=True, timeout=120,
            )
            # Unzip if needed
            import zipfile
            for zf in cache_path.glob("*.zip"):
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(cache_path)
                zf.unlink()
            
            # Find and rename the training file
            # Most competitions have train.csv at the root
            candidates = list(cache_path.glob("*train*.csv"))
            if candidates:
                # For simulation, we need to combine train + test (if test has labels)
                # For most closed competitions, we only use train.csv
                candidates[0].rename(full_data)
            else:
                raise FileNotFoundError(f"No train*.csv found in {cache_path}")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to download '{entry.slug}' via Kaggle API: {e}\n"
                f"Manual fix: download from kaggle.com/competitions/{entry.slug}/data\n"
                f"Place train.csv at: {full_data}"
            )
    
    elif entry.download_method == "cached":
        if entry.cached_path and Path(entry.cached_path).exists():
            import shutil
            shutil.copy(entry.cached_path, full_data)
        else:
            raise FileNotFoundError(
                f"Cached path '{entry.cached_path}' not found for {entry.slug}"
            )
```

---

## Phase 4 — Build Scorers (20 minutes)

### 4.1 Create `simulator/scorers.py`

```python
"""
Metric scorers — must match Kaggle's evaluation exactly.

Every scorer takes (true_labels_df, predictions_df, id_column, target_column)
and returns a float score.

CRITICAL: These must produce IDENTICAL results to Kaggle's evaluation.
Verify each scorer against known Kaggle submissions when possible.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, log_loss,
    mean_squared_error, mean_absolute_error
)


def score_accuracy(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy()
    y_pred = merged[f"{target_col}_pred"].to_numpy()
    return float(accuracy_score(y_true, y_pred))


def score_auc(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy()
    y_pred = merged[f"{target_col}_pred"].to_numpy().astype(float)
    return float(roc_auc_score(y_true, y_pred))


def score_f1_binary(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy()
    y_pred = merged[f"{target_col}_pred"].to_numpy()
    return float(f1_score(y_true, y_pred, average="binary"))


def score_macro_f1(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy()
    y_pred = merged[f"{target_col}_pred"].to_numpy()
    return float(f1_score(y_true, y_pred, average="macro"))


def score_log_loss(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy()
    y_pred = merged[f"{target_col}_pred"].to_numpy().astype(float)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return float(log_loss(y_true, y_pred))


def score_rmse(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy().astype(float)
    y_pred = merged[f"{target_col}_pred"].to_numpy().astype(float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def score_rmsle(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy().astype(float)
    y_pred = merged[f"{target_col}_pred"].to_numpy().astype(float)
    y_pred = np.maximum(y_pred, 0)  # RMSLE requires non-negative
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def score_mae(labels_df, preds_df, id_col, target_col):
    merged = labels_df.join(preds_df, on=id_col, suffix="_pred")
    y_true = merged[target_col].to_numpy().astype(float)
    y_pred = merged[f"{target_col}_pred"].to_numpy().astype(float)
    return float(mean_absolute_error(y_true, y_pred))


SCORERS = {
    "accuracy":     score_accuracy,
    "auc":          score_auc,
    "f1":           score_f1_binary,
    "macro_f1":     score_macro_f1,
    "log_loss":     score_log_loss,
    "rmse":         score_rmse,
    "rmsle":        score_rmsle,
    "mae":          score_mae,
}


def get_scorer(metric: str):
    """Get scorer function by metric name. Raises ValueError if unknown."""
    if metric not in SCORERS:
        raise ValueError(
            f"Unknown metric '{metric}'. Available: {list(SCORERS.keys())}. "
            f"Add new scorers to simulator/scorers.py"
        )
    return SCORERS[metric]
```

---

## Phase 5 — Build Simulated Leaderboard (30 minutes)

### 5.1 Create `simulator/leaderboard.py`

```python
"""
Simulated Leaderboard — behaves identically to Kaggle's LB.

Professor submits → gets public score back.
Private score computed but NOT revealed until competition_end().

This forces Professor's Submission Strategist to operate under
the same constraints as a real competition.
"""

import polars as pl
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

from simulator.competition_registry import CompetitionEntry
from simulator.data_splitter import SplitResult
from simulator.scorers import get_scorer


@dataclass
class SubmissionRecord:
    submission_id: int
    path: str
    public_score: float
    private_score: float        # stored but NOT returned to Professor
    day: int
    timestamp: str


@dataclass
class SubmissionResult:
    success: bool
    public_score: Optional[float] = None
    private_score: Optional[float] = None  # ALWAYS None until competition_end
    submission_id: Optional[int] = None
    public_rank_estimate: Optional[float] = None
    submissions_today: int = 0
    submissions_remaining: int = 0
    error: Optional[str] = None


@dataclass
class CompetitionResult:
    slug: str = ""
    best_public_score: Optional[float] = None
    best_private_score: Optional[float] = None
    selected_submission_1: Optional[int] = None
    selected_submission_2: Optional[int] = None
    public_rank_pct: float = 100.0
    private_rank_pct: float = 100.0
    shakeup_positions: float = 0.0
    medal: str = "none"
    total_submissions: int = 0
    days_used: int = 0
    all_submissions: list = field(default_factory=list)
    error: Optional[str] = None


class SimulatedLeaderboard:
    
    def __init__(
        self,
        entry: CompetitionEntry,
        split: SplitResult,
        daily_limit: int = 5,
    ):
        self.entry = entry
        self.split = split
        self.scorer = get_scorer(entry.metric)
        self.maximize = entry.metric_direction == "maximize"
        
        # Load hidden labels
        self.public_labels = pl.read_csv(split.public_labels_path)
        self.private_labels = pl.read_csv(split.private_labels_path)
        self.public_ids = set(self.public_labels[entry.id_column].to_list())
        self.private_ids = set(self.private_labels[entry.id_column].to_list())
        
        # Submission tracking
        self.submissions: List[SubmissionRecord] = []
        self.daily_limit = daily_limit
        self.current_day = 1
    
    def submit(self, submission_path: str) -> SubmissionResult:
        """
        Score a submission. Returns PUBLIC score only.
        Private score is computed and stored but NOT returned.
        """
        # Validate format
        try:
            sub_df = pl.read_csv(submission_path)
        except Exception as e:
            return SubmissionResult(success=False, error=f"Cannot read submission: {e}")
        
        required_cols = {self.entry.id_column, self.entry.target_column}
        if not required_cols.issubset(set(sub_df.columns)):
            return SubmissionResult(
                success=False,
                error=f"Missing columns. Required: {required_cols}. Got: {set(sub_df.columns)}"
            )
        
        # Check daily limit
        today_count = sum(1 for s in self.submissions if s.day == self.current_day)
        if today_count >= self.daily_limit:
            return SubmissionResult(
                success=False,
                error=f"Daily limit reached ({self.daily_limit}/day). "
                      f"Call leaderboard.advance_day() to proceed."
            )
        
        # Validate row count
        expected_rows = len(self.public_ids) + len(self.private_ids)
        if len(sub_df) != expected_rows:
            return SubmissionResult(
                success=False,
                error=f"Expected {expected_rows} rows, got {len(sub_df)}"
            )
        
        # Split submission into public and private
        public_preds = sub_df.filter(
            pl.col(self.entry.id_column).is_in(list(self.public_ids))
        )
        private_preds = sub_df.filter(
            pl.col(self.entry.id_column).is_in(list(self.private_ids))
        )
        
        # Score
        public_score = self.scorer(
            self.public_labels, public_preds,
            self.entry.id_column, self.entry.target_column
        )
        private_score = self.scorer(
            self.private_labels, private_preds,
            self.entry.id_column, self.entry.target_column
        )
        
        # Record
        record = SubmissionRecord(
            submission_id=len(self.submissions) + 1,
            path=submission_path,
            public_score=public_score,
            private_score=private_score,
            day=self.current_day,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.submissions.append(record)
        
        return SubmissionResult(
            success=True,
            public_score=public_score,
            private_score=None,  # HIDDEN
            submission_id=record.submission_id,
            public_rank_estimate=self._estimate_rank(public_score),
            submissions_today=today_count + 1,
            submissions_remaining=self.daily_limit - today_count - 1,
        )
    
    def advance_day(self):
        """Simulate passage of one competition day."""
        self.current_day += 1
    
    def competition_end(self) -> CompetitionResult:
        """Reveal private scores. Calculate final standing."""
        if not self.submissions:
            return CompetitionResult(error="No submissions made.")
        
        # Select best public score submission
        if self.maximize:
            best_public = max(self.submissions, key=lambda s: s.public_score)
        else:
            best_public = min(self.submissions, key=lambda s: s.public_score)
        
        # Best private score from selected submissions
        best_private_score = best_public.private_score
        
        # Percentile ranks
        public_pct = self._estimate_rank(best_public.public_score)
        private_pct = self._estimate_rank(best_private_score)
        shakeup = private_pct - public_pct  # positive = dropped rank
        
        return CompetitionResult(
            slug=self.entry.slug,
            best_public_score=best_public.public_score,
            best_private_score=best_private_score,
            selected_submission_1=best_public.submission_id,
            public_rank_pct=public_pct,
            private_rank_pct=private_pct,
            shakeup_positions=shakeup,
            medal=self._compute_medal(best_private_score),
            total_submissions=len(self.submissions),
            days_used=self.current_day,
            all_submissions=[
                {
                    "id": s.submission_id,
                    "public": round(s.public_score, 6),
                    "private": round(s.private_score, 6),
                    "day": s.day,
                }
                for s in self.submissions
            ],
        )
    
    def _estimate_rank(self, score: float) -> float:
        """Estimate percentile using LB curves. Lower = better."""
        percentiles = self.entry.lb_percentiles
        for pct in sorted(percentiles.keys()):
            threshold = percentiles[pct]
            if self.maximize and score >= threshold:
                return float(pct)
            elif not self.maximize and score <= threshold:
                return float(pct)
        return 90.0
    
    def _compute_medal(self, score: float) -> str:
        if self.maximize:
            if score >= self.entry.silver_threshold: return "silver"
            if score >= self.entry.gold_threshold: return "gold"
            if score >= self.entry.bronze_threshold: return "bronze"
        else:
            if score <= self.entry.silver_threshold: return "silver"
            if score <= self.entry.gold_threshold: return "gold"
            if score <= self.entry.bronze_threshold: return "bronze"
        return "none"
```

---

## Phase 6 — Build Modal Cloud App (30 minutes)

### 6.1 Create `simulator/cloud_benchmark.py`

This is the main Modal application. It orchestrates parallel execution.

```python
"""
Professor Cloud Benchmark — Modal App

Run all competitions in parallel on Modal's serverless infrastructure.
Each competition runs in its own container. Wall clock = slowest single competition.

Usage:
    modal run simulator/cloud_benchmark.py                              # fast, all comps
    modal run simulator/cloud_benchmark.py --mode deep                  # deep, all comps
    modal run simulator/cloud_benchmark.py --competition spaceship-titanic  # single comp
"""

import modal
import json
import time
from pathlib import Path
from datetime import datetime

# ── Modal Image: built once, cached ──
professor_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "lightgbm>=4.0", "xgboost>=2.0", "catboost>=1.2",
        "scikit-learn>=1.3", "optuna>=3.0", "polars>=1.0",
        "pyarrow>=14.0", "scipy>=1.11", "numpy>=1.24",
        "chromadb>=0.5", "sentence-transformers>=2.0",
        "openai>=1.0", "google-generativeai>=0.5",
        "kaggle>=1.6", "psutil>=5.9", "mlflow>=2.0",
    )
    .copy_local_dir(".", "/root/professor-agent")
)

# ── Persistent Volume: competition data cached across runs ──
data_volume = modal.Volume.from_name("professor-benchmark-data", create_if_missing=True)

# ── Secrets ──
secrets = modal.Secret.from_name("professor-keys")

# ── App ──
app = modal.App("professor-benchmark")


@app.function(
    image=professor_image,
    volumes={"/data": data_volume},
    secrets=[secrets],
    cpu=4.0,
    memory=16384,
    timeout=3600,
)
def run_single_competition(competition_slug: str, mode: str = "fast") -> dict:
    """Run Professor against one competition. Returns result dict."""
    import sys
    sys.path.insert(0, "/root/professor-agent")
    
    from simulator.competition_registry import get_competition
    from simulator.data_splitter import split_competition_data, ensure_data_cached
    from simulator.leaderboard import SimulatedLeaderboard
    
    entry = get_competition(competition_slug)
    
    # Cache data on persistent volume
    cache_dir = "/data/competitions"
    ensure_data_cached(entry, cache_dir)
    
    # Split (deterministic)
    split_dir = f"/data/splits/{entry.slug}"
    split = split_competition_data(
        data_path=f"{cache_dir}/{entry.slug}/full_data.csv",
        entry=entry,
        output_dir=split_dir,
    )
    
    # Create leaderboard
    lb = SimulatedLeaderboard(entry, split)
    
    # Configure for benchmark mode
    fast_config = {
        "optuna_trials": 30,
        "null_importance_shuffles": 5,
        "max_submissions": 1,
        "skip_forum_scrape": True,
    }
    deep_config = {
        "optuna_trials": 200,
        "null_importance_shuffles": 50,
        "max_submissions": 3,
        "skip_forum_scrape": False,
    }
    config = fast_config if mode == "fast" else deep_config
    
    # Run Professor
    start = time.time()
    
    try:
        # Import Professor's pipeline
        from core.professor import run_professor_pipeline
        
        state = run_professor_pipeline(
            train_path=split.train_path,
            test_path=split.test_path,
            sample_submission_path=split.sample_submission_path,
            competition_name=entry.slug,
            config=config,
        )
        
        runtime = time.time() - start
        
        # Submit
        submission_path = state.get("submission_path")
        if submission_path:
            result = lb.submit(submission_path)
        else:
            result = None
        
        # Reveal scores
        final = lb.competition_end()
        
        return {
            "slug": entry.slug,
            "task_type": entry.task_type,
            "domain": entry.primary_domain,
            "metric": entry.metric,
            "cv_score": state.get("cv_mean"),
            "public_score": final.best_public_score,
            "private_score": final.best_private_score,
            "public_percentile": final.public_rank_pct,
            "private_percentile": final.private_rank_pct,
            "shakeup": final.shakeup_positions,
            "medal": final.medal,
            "total_submissions": final.total_submissions,
            "runtime_seconds": round(runtime, 1),
            "mode": mode,
            "error": None,
        }
    
    except Exception as e:
        import traceback
        return {
            "slug": entry.slug,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "runtime_seconds": round(time.time() - start, 1),
            "mode": mode,
        }
    
    finally:
        data_volume.commit()


@app.function(
    image=professor_image,
    timeout=7200,
)
def run_full_benchmark(mode: str = "fast", slugs: list = None) -> dict:
    """Orchestrate all competitions in parallel. Return aggregate report."""
    import sys
    sys.path.insert(0, "/root/professor-agent")
    from simulator.competition_registry import REGISTRY
    
    if not slugs:
        slugs = [e.slug for e in REGISTRY]
    
    # Launch all in parallel
    results = list(run_single_competition.map(
        slugs, [mode] * len(slugs)
    ))
    
    # Aggregate
    successful = [r for r in results if r.get("error") is None]
    failed = [r for r in results if r.get("error") is not None]
    
    private_scores = [r["private_percentile"] for r in successful if r.get("private_percentile")]
    medals = [r["medal"] for r in successful]
    
    import numpy as np
    
    report = {
        "run_id": f"benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "professor_version": "v2.0",  # update as versions change
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "n_competitions": len(slugs),
        "n_successful": len(successful),
        "n_failed": len(failed),
        "aggregate_metrics": {
            "median_percentile": float(np.median(private_scores)) if private_scores else None,
            "mean_percentile": float(np.mean(private_scores)) if private_scores else None,
            "gold_rate": medals.count("gold") / len(medals) if medals else 0,
            "silver_rate": medals.count("silver") / len(medals) if medals else 0,
            "bronze_rate": medals.count("bronze") / len(medals) if medals else 0,
            "medal_rate": sum(1 for m in medals if m != "none") / len(medals) if medals else 0,
        },
        "per_competition": results,
    }
    
    return report


@app.local_entrypoint()
def main(
    mode: str = "fast",
    competition: str = "",
):
    """
    CLI entry point.
    
    Usage:
        modal run simulator/cloud_benchmark.py
        modal run simulator/cloud_benchmark.py --mode deep
        modal run simulator/cloud_benchmark.py --competition spaceship-titanic
    """
    if competition:
        result = run_single_competition.remote(competition, mode)
        print(json.dumps(result, indent=2))
    else:
        report = run_full_benchmark.remote(mode)
        
        agg = report["aggregate_metrics"]
        print(f"\n{'='*60}")
        print(f"  PROFESSOR BENCHMARK — {report['professor_version']}")
        print(f"  Mode: {mode} | Competitions: {report['n_competitions']}")
        print(f"{'='*60}")
        
        if agg["median_percentile"]:
            print(f"  Median percentile:  {agg['median_percentile']:.1f}%")
            print(f"  Gold rate:          {agg['gold_rate']:.0%}")
            print(f"  Medal rate:         {agg['medal_rate']:.0%}")
        
        print(f"\n  Per competition:")
        emojis = {"gold": "🥇", "silver": "🥈", "bronze": "🥉", "none": "  "}
        for c in report["per_competition"]:
            if c.get("error"):
                print(f"  ❌ {c['slug']}: {c['error'][:80]}")
            else:
                m = emojis.get(c.get("medal", "none"), "  ")
                print(f"  {m} {c['slug']:<40} "
                      f"private={c.get('private_score', 'N/A')}  "
                      f"pct={c.get('private_percentile', 'N/A')}%  "
                      f"{c.get('runtime_seconds', 0):.0f}s")
        
        # Save locally
        Path("simulator/results").mkdir(parents=True, exist_ok=True)
        out = f"simulator/results/{report['run_id']}.json"
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"\n  Report saved: {out}")
```

---

## Phase 7 — Verify (15 minutes)

### 7.1 Smoke Test — Local (no Modal)

```bash
# Test the registry
python -c "
from simulator.competition_registry import list_competitions, get_competition
print('Registered:', list_competitions())
entry = get_competition('spaceship-titanic')
print(f'  Target: {entry.target_column}, Metric: {entry.metric}')
print('[PASS] Registry works')
"
```

### 7.2 Smoke Test — Modal (single competition)

```bash
# This uploads code, builds image (first time ~3 min), runs one competition
modal run simulator/cloud_benchmark.py --competition spaceship-titanic
```

**Expected output (first run):**
```
Building image... (2-3 minutes, cached after)
Running spaceship-titanic in fast mode...
{
  "slug": "spaceship-titanic",
  "private_score": 0.8089,
  "private_percentile": 10.0,
  "medal": "gold",
  "runtime_seconds": 285,
  ...
}
```

### 7.3 Full Benchmark

```bash
modal run simulator/cloud_benchmark.py
```

**Expected: all competitions run in parallel, report in ~12 minutes.**

---

## Phase 8 — Scheduled Runs (Optional, 5 minutes)

### 8.1 Nightly Cron on Modal

Add to `cloud_benchmark.py`:

```python
@app.function(
    schedule=modal.Cron("0 6 * * *"),  # 6 AM UTC daily
    image=professor_image,
    volumes={"/data": data_volume},
    secrets=[secrets],
    timeout=7200,
)
def nightly_benchmark():
    report = run_full_benchmark.remote(mode="fast")
    # Results saved to volume automatically
    return report
```

Deploy:

```bash
modal deploy simulator/cloud_benchmark.py
```

Now benchmarks run automatically every night. Check results anytime:

```bash
modal volume get professor-benchmark-data results/
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  PROFESSOR CLOUD BENCHMARK — QUICK REFERENCE                │
│                                                              │
│  Setup (one-time):                                          │
│    pip install modal && modal setup                         │
│    modal secret create professor-keys FIREWORKS_API_KEY=... │
│                                                              │
│  Run:                                                       │
│    modal run simulator/cloud_benchmark.py                   │
│    modal run simulator/cloud_benchmark.py --mode deep       │
│    modal run simulator/cloud_benchmark.py --competition X   │
│                                                              │
│  Deploy nightly:                                            │
│    modal deploy simulator/cloud_benchmark.py                │
│                                                              │
│  Cost:                                                      │
│    Fast (10 comps, 12 min): ~$0.52                         │
│    Deep (10 comps, 60 min): ~$3.17                         │
│    Free tier: $30/month = ~57 fast runs                    │
│                                                              │
│  Results: simulator/results/benchmark_*.json                │
└─────────────────────────────────────────────────────────────┘
```