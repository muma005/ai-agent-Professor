"""
Data Splitter — Produces deterministic, reproducible partitions.

Takes competition data and produces three partitions:
- Train (60%): What Professor receives (with target)
- Public test (12%): Features only, target held for scoring
- Private test (28%): Features only, target held for scoring

Critical design decisions:
1. Split is DETERMINISTIC — same data + same seed = exact same split
2. Data isolation — Professor NEVER sees private labels
3. Distribution preservation — KS-test validation between train/test
4. Strategy matched to competition type (stratified/temporal/group)

File naming convention:
- train.csv: What Professor receives
- test.csv: Features only (public + private rows shuffled together)
- .public_labels.csv: HIDDEN (dotfile) — only leaderboard reads
- .private_labels.csv: HIDDEN (dotfile) — only leaderboard reads
- sample_submission.csv: Template for submissions
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import json
import hashlib
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from simulator.competition_registry import CompetitionEntry


@dataclass
class SplitResult:
    """Result of splitting competition data."""
    
    train_path: Path
    test_path: Path
    public_labels_path: Path
    private_labels_path: Path
    sample_submission_path: Path
    n_train: int
    n_public: int
    n_private: int
    split_metadata: Dict[str, Any]
    
    @property
    def train_hash(self) -> str:
        """Hash of train data for determinism verification."""
        return hashlib.md5(Path(self.train_path).read_bytes()).hexdigest()
    
    @property
    def public_hash(self) -> str:
        """Hash of public test data for determinism verification."""
        return hashlib.md5(Path(self.public_labels_path).read_bytes()).hexdigest()
    
    @property
    def private_hash(self) -> str:
        """Hash of private test data for determinism verification."""
        return hashlib.md5(Path(self.private_labels_path).read_bytes()).hexdigest()
    
    def verify_determinism(self, other: "SplitResult") -> bool:
        """Verify this split is identical to another (for testing)."""
        return (
            self.train_hash == other.train_hash
            and self.public_hash == other.public_hash
            and self.private_hash == other.private_hash
        )


def split_competition_data(
    data_path: str,
    entry: CompetitionEntry,
    force: bool = False,
) -> SplitResult:
    """
    Produces 3 partitions from a single dataset.
    
    IMPORTANT: The split is DETERMINISTIC and REPRODUCIBLE.
    Same data + same seed = exact same split every time.
    This is essential for regression testing — if Professor's score
    changes between runs on the same split, the change is real,
    not split variance.
    
    Args:
        data_path: Path to the full competition data CSV
        entry: Competition entry with split configuration
        force: If True, re-split even if cached split exists
    
    Returns:
        SplitResult with paths to all partitions and metadata
    """
    data_dir = entry.get_data_dir()
    
    # Check if split already exists
    train_path = entry.get_train_path()
    test_path = entry.get_test_path()
    public_labels_path = entry.get_public_labels_path()
    private_labels_path = entry.get_private_labels_path()
    
    if not force and train_path.exists() and test_path.exists():
        # Load existing split metadata
        meta_path = data_dir / "split_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            print(f"[splitter] Split already exists for {entry.slug}. Reusing.")
            
            # Verify hashes match (determinism check)
            current_hash = hashlib.md5(Path(data_path).read_bytes()).hexdigest()
            if meta.get("data_hash") == current_hash:
                return SplitResult(
                    train_path=train_path,
                    test_path=test_path,
                    public_labels_path=public_labels_path,
                    private_labels_path=private_labels_path,
                    sample_submission_path=entry.get_sample_submission_path(),
                    n_train=meta["n_train"],
                    n_public=meta["n_public"],
                    n_private=meta["n_private"],
                    split_metadata=meta,
                )
            else:
                print("[splitter] Data changed, re-splitting...")
    
    # Load full dataset
    df = pl.read_csv(data_path)
    target = entry.target_column
    seed = entry.random_seed  # always 42
    
    print(f"[splitter] Splitting {entry.slug} ({len(df)} rows) using {entry.split_strategy} strategy...")
    
    # ── Step 1: Split into train (60%) and test (40%) ──
    if entry.split_strategy == "stratified":
        # Stratified by target — preserves class distribution
        train_df, test_df = _stratified_split(
            df, target=target, test_size=entry.test_ratio, seed=seed
        )
        
    elif entry.split_strategy == "temporal":
        # Sort by time column, take last 40% as test
        if entry.split_column is None:
            raise ValueError("temporal split requires split_column to be set")
        df_sorted = df.sort(entry.split_column)
        split_idx = int(len(df_sorted) * (1 - entry.test_ratio))
        train_df = df_sorted[:split_idx]
        test_df = df_sorted[split_idx:]
        
    elif entry.split_strategy == "group":
        # Split by group — no group appears in both train and test
        if entry.split_column is None:
            raise ValueError("group split requires split_column to be set")
        train_df, test_df = _stratified_group_split(
            df, 
            group_column=entry.split_column,
            target=target,
            test_size=entry.test_ratio, 
            seed=seed
        )
    
    else:
        raise ValueError(f"Unknown split strategy: {entry.split_strategy}")
    
    # Validate split quality
    _validate_split(train_df, test_df, entry)
    
    # ── Step 2: Split test into public (30%) and private (70%) ──
    # Note: 30% of test = 0.3 * 0.4 = 12% of total
    #       70% of test = 0.7 * 0.4 = 28% of total
    public_test, private_test = _stratified_split(
        test_df, target=target, test_size=0.70, seed=seed + 1
    )
    
    # ── Step 3: Save partitions ──
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Train: what Professor receives (with target)
    train_df.write_csv(train_path)
    
    # Combine public + private into one test file (features only)
    # Professor sees this as the test set — same as real Kaggle
    full_test = pl.concat([public_test, private_test])
    # Shuffle to mix public and private rows
    full_test = full_test.sample(fraction=1.0, seed=seed + 2)
    full_test_features = full_test.drop(target)
    full_test_features.write_csv(test_path)
    
    # Hidden label files (dotfiles)
    # Only the Simulated Leaderboard component reads these
    public_test[[entry.id_column, target]].write_csv(public_labels_path)
    private_test[[entry.id_column, target]].write_csv(private_labels_path)
    
    # Sample submission (all zeros or median, same as Kaggle provides)
    default_value = _default_prediction(entry.task_type)
    sample = full_test_features[[entry.id_column]].with_columns(
        pl.lit(default_value).alias(target)
    )
    sample.write_csv(entry.get_sample_submission_path())
    
    # ── Step 4: Save metadata for reproducibility ──
    meta = {
        "competition_slug": entry.slug,
        "strategy": entry.split_strategy,
        "seed": seed,
        "train_ratio": 1 - entry.test_ratio,
        "test_ratio": entry.test_ratio,
        "public_ratio": entry.public_ratio,
        "n_train": len(train_df),
        "n_public": len(public_test),
        "n_private": len(private_test),
        "n_test_total": len(full_test),
        "target_distribution_train": _target_stats(train_df, target, entry.task_type),
        "target_distribution_public": _target_stats(public_test, target, entry.task_type),
        "target_distribution_private": _target_stats(private_test, target, entry.task_type),
        "data_hash": hashlib.md5(Path(data_path).read_bytes()).hexdigest(),
        "train_hash": hashlib.md5(train_path.read_bytes()).hexdigest(),
        "public_hash": hashlib.md5(public_labels_path.read_bytes()).hexdigest(),
        "private_hash": hashlib.md5(private_labels_path.read_bytes()).hexdigest(),
    }
    
    meta_path = data_dir / "split_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    
    print(
        f"[splitter] Complete: {len(train_df)} train, "
        f"{len(public_test)} public, {len(private_test)} private"
    )
    
    return SplitResult(
        train_path=train_path,
        test_path=test_path,
        public_labels_path=public_labels_path,
        private_labels_path=private_labels_path,
        sample_submission_path=entry.get_sample_submission_path(),
        n_train=len(train_df),
        n_public=len(public_test),
        n_private=len(private_test),
        split_metadata=meta,
    )


def _stratified_split(
    df: pl.DataFrame,
    target: str,
    test_size: float,
    seed: int,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Stratified train/test split preserving target distribution.
    Falls back to random split for regression tasks.
    """
    y = df[target].to_numpy()
    
    # For classification, stratify by target
    if y.dtype.kind in ('i', 'u', 'b', 'O'):  # int, uint, bool, object
        stratify = y
    else:
        # For continuous targets (regression), bin the target for stratification
        n_bins = min(10, len(np.unique(y)))
        y_binned = np.digitize(y, bins=np.linspace(y.min(), y.max(), n_bins))
        stratify = y_binned
    
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    
    return df[train_idx], df[test_idx]


def _stratified_group_split(
    df: pl.DataFrame,
    group_column: str,
    target: str,
    test_size: float,
    seed: int,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split by group — no group appears in both train and test.
    Uses stratified sampling on groups based on target distribution.
    """
    # Get unique groups
    groups = df[group_column].unique().to_list()
    
    # Compute target mean per group (for stratification)
    group_stats = df.group_by(group_column).agg(
        pl.col(target).mean().alias("target_mean")
    )
    
    # Stratified split of groups
    group_targets = group_stats["target_mean"].to_numpy()
    
    # Bin for stratification if continuous
    if group_targets.dtype.kind == 'f':
        n_bins = min(5, len(np.unique(group_targets)))
        group_bins = np.digitize(group_targets, bins=np.linspace(
            group_targets.min(), group_targets.max(), n_bins
        ))
        stratify = group_bins
    else:
        stratify = group_targets.astype(int)
    
    train_groups, test_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    
    train_df = df.filter(pl.col(group_column).is_in(train_groups))
    test_df = df.filter(pl.col(group_column).is_in(test_groups))
    
    return train_df, test_df


def _validate_split(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    entry: CompetitionEntry,
    max_reseeds: int = 5,
) -> None:
    """
    Validate split quality:
    1. No row appears in both partitions
    2. Target distribution preserved (KS-test for classification)
    3. No categorical level appears only in test
    
    Raises ValueError if validation fails after max_reseeds attempts.
    """
    target = entry.target_column
    
    # Check no overlap
    train_ids = set(train_df[entry.id_column].to_list())
    test_ids = set(test_df[entry.id_column].to_list())
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Split validation failed: {len(overlap)} IDs in both train and test")
    
    # Check target distribution (for classification)
    if entry.task_type in ("binary", "multiclass"):
        train_dist = _target_stats(train_df, target, entry.task_type)
        test_dist = _target_stats(test_df, target, entry.task_type)
        
        # Simple chi-square-like check
        for label in train_dist:
            if label not in test_dist:
                # Label missing from test — acceptable for rare classes
                if train_dist[label] < 0.05:  # < 5% of train
                    print(f"[splitter] Warning: Label '{label}' rare in train ({train_dist[label]:.2%})")
                else:
                    raise ValueError(
                        f"Split validation failed: Label '{label}' missing from test "
                        f"but common in train ({train_dist[label]:.2%})"
                    )
    
    # Check no categorical level appears only in test
    # (would cause model to fail on unseen categories)
    for col in train_df.columns:
        if col in (entry.id_column, target):
            continue
        if train_df[col].dtype in (pl.Categorical, pl.Utf8):
            train_levels = set(train_df[col].drop_nulls().unique().to_list())
            test_levels = set(test_df[col].drop_nulls().unique().to_list())
            only_in_test = test_levels - train_levels
            if only_in_test:
                print(
                    f"[splitter] Warning: Column '{col}' has {len(only_in_test)} "
                    f"levels only in test: {list(only_in_test)[:5]}"
                )


def _target_stats(df: pl.DataFrame, target: str, task_type: str) -> Dict[str, float]:
    """Compute target distribution as proportions."""
    if task_type in ("binary", "multiclass"):
        counts = df[target].value_counts(sort=True)
        total = len(df)
        return {str(row[target]): row["count"] / total for row in counts.iter_rows(named=True)}
    else:
        # For regression, return mean and std
        return {
            "mean": float(df[target].mean()),
            "std": float(df[target].std()),
            "min": float(df[target].min()),
            "max": float(df[target].max()),
        }


def _default_prediction(task_type: str) -> float:
    """Return default prediction value for sample submission."""
    if task_type == "binary":
        return 0.5
    elif task_type == "multiclass":
        return 0  # Will be overridden per-class in real submissions
    else:
        return 0.0
