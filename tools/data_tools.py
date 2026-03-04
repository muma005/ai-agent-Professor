# tools/data_tools.py

import os
import json
import hashlib
import polars as pl
import polars.selectors as cs
from datetime import datetime
from pathlib import Path


# ── Read ──────────────────────────────────────────────────────────

def read_csv(path: str, infer_schema_length: int = 10000) -> pl.DataFrame:
    """
    Read a CSV file into a Polars DataFrame.
    Always use this — never pl.read_csv() directly in agents.
    Validates the file exists before reading.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pl.read_csv(path, infer_schema_length=infer_schema_length)


def read_parquet(path: str) -> pl.DataFrame:
    """
    Read a Parquet file into a Polars DataFrame.
    Always use this — never pl.read_parquet() directly in agents.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pl.read_parquet(path)
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected Polars DataFrame, got {type(df)}")
    return df


# ── Write ─────────────────────────────────────────────────────────

def write_parquet(df: pl.DataFrame, path: str) -> str:
    """
    Write a Polars DataFrame to Parquet.
    Creates parent directories if they don't exist.
    Returns the path written.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"write_parquet expects a Polars DataFrame, got {type(df)}. "
            "If you have a Pandas DataFrame, convert with pl.from_pandas(df) first."
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.write_parquet(path)
    return path


def write_json(data: dict, path: str) -> str:
    """
    Write a dict to JSON. Creates parent directories if needed.
    Returns the path written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def read_json(path: str) -> dict:
    """Read a JSON file and return as dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path) as f:
        return json.load(f)


# ── Profile ───────────────────────────────────────────────────────

def profile_data(df: pl.DataFrame) -> dict:
    """
    Profile a Polars DataFrame and return a schema dict.
    This is what gets written to schema.json by the Data Engineer.

    Returns:
        {
            columns:       [list of column names],
            types:         {col: dtype_str},
            missing_rates: {col: float 0-1},
            missing_counts:{col: int},
            shape:         [rows, cols],
            numeric_cols:  [list],
            categorical_cols: [list],
            boolean_cols:  [list],
            cardinality:   {col: n_unique} for categorical cols,
            profiled_at:   ISO timestamp
        }
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"profile_data expects Polars DataFrame, got {type(df)}")

    n_rows = len(df)

    # Column types
    column_types = {col: str(df[col].dtype) for col in df.columns}

    # Missing rates and counts
    missing_counts = {col: int(df[col].null_count()) for col in df.columns}
    missing_rates  = {
        col: round(missing_counts[col] / n_rows, 4) if n_rows > 0 else 0.0
        for col in df.columns
    }

    # Categorise columns by type
    numeric_cols     = df.select(cs.numeric()).columns
    categorical_cols = df.select(cs.string()).columns
    boolean_cols     = df.select(cs.boolean()).columns

    # Cardinality for categoricals (useful for encoding decisions)
    cardinality = {
        col: int(df[col].n_unique())
        for col in categorical_cols
    }

    return {
        "columns":          df.columns,
        "types":            column_types,
        "missing_rates":    missing_rates,
        "missing_counts":   missing_counts,
        "shape":            list(df.shape),
        "numeric_cols":     numeric_cols,
        "categorical_cols": categorical_cols,
        "boolean_cols":     boolean_cols,
        "cardinality":      cardinality,
        "profiled_at":      datetime.utcnow().isoformat(),
    }


# ── Hash ──────────────────────────────────────────────────────────

def hash_file(path: str, length: int = 16) -> str:
    """
    SHA-256 hash of a file. Returns first `length` hex chars.
    Used to detect if the dataset changes mid-competition.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot hash — file not found: {path}")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:length]


def hash_dataframe(df: pl.DataFrame, length: int = 16) -> str:
    """
    SHA-256 hash of a Polars DataFrame's content.
    Deterministic — same data always produces same hash.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"hash_dataframe expects Polars DataFrame, got {type(df)}")
    content = df.write_csv().encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:length]


# ── Validate ──────────────────────────────────────────────────────

def validate_submission(
    submission: pl.DataFrame,
    sample_submission: pl.DataFrame
) -> dict:
    """
    Validate a submission DataFrame against the sample submission format.

    Returns:
        {"valid": bool, "errors": [list of error strings]}
    """
    errors = []

    # Column names must match exactly
    if set(submission.columns) != set(sample_submission.columns):
        errors.append(
            f"Column mismatch. Expected: {sample_submission.columns}. "
            f"Got: {submission.columns}"
        )

    # Row count must match
    if len(submission) != len(sample_submission):
        errors.append(
            f"Row count mismatch. Expected: {len(sample_submission)}. "
            f"Got: {len(submission)}"
        )

    # No nulls in submission
    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        errors.append(f"Submission contains {null_count} null values")

    return {
        "valid":  len(errors) == 0,
        "errors": errors
    }


# ── Ensure output dir ─────────────────────────────────────────────

def ensure_session_dirs(session_id: str) -> dict:
    """
    Create all output subdirectories for a session.
    Returns dict of all paths.
    """
    base = f"outputs/{session_id}"
    dirs = {
        "base":        base,
        "models":      f"{base}/models",
        "predictions": f"{base}/predictions",
        "charts":      f"{base}/charts",
        "logs":        f"{base}/logs",
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs
