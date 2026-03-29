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
    try:
        from scipy.stats import ks_2samp

        full_vals = full_df[target].drop_nulls().to_numpy().astype(float)
        split_vals = train_df[target].drop_nulls().to_numpy().astype(float)

        stat, pvalue = ks_2samp(full_vals, split_vals)
        if pvalue < 0.01:
            print(f"[WARNING] {name} split has significantly different target "
                  f"distribution (KS p={pvalue:.4f}). Consider re-seeding.")
    except Exception:
        pass  # scipy not available, skip validation


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
