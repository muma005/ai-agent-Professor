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
    import sys, os
    kaggle_exe = os.path.join(os.path.dirname(sys.executable), "kaggle.exe" if os.name == "nt" else "kaggle")
    
    result = subprocess.run(
        [kaggle_exe, "competitions", "download", "-c", competition_id,
         "-p", str(target_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed for '{competition_id}':\n{result.stderr}\n"
            "Ensure KAGGLE_USERNAME and KAGGLE_KEY are in .env and you have "
            "accepted the competition rules on kaggle.com."
        )

    import zipfile
    for p in target_dir.glob("*.zip"):
        with zipfile.ZipFile(p, 'r') as zf:
            zf.extractall(target_dir)

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
