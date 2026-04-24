# tests/contracts/test_submission_validator_contract.py

import pytest
import os
import polars as pl
from tools.submission_validator import validate_submission

@pytest.fixture
def sample_submission(tmp_path):
    path = str(tmp_path / "sample_submission.csv")
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "target": [0.5, 0.5, 0.5]
    })
    df.write_csv(path)
    return path

def test_perfect_match_passes(sample_submission, tmp_path):
    gen_path = str(tmp_path / "perfect.csv")
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "target": [0.1, 0.9, 0.4]
    })
    df.write_csv(gen_path)
    res = validate_submission(sample_submission, gen_path)
    assert res["is_valid"] is True

def test_missing_rows_fails(sample_submission, tmp_path):
    gen_path = str(tmp_path / "missing_rows.csv")
    df = pl.DataFrame({
        "id": [1, 2],
        "target": [0.1, 0.9]
    })
    df.write_csv(gen_path)
    res = validate_submission(sample_submission, gen_path)
    assert res["is_valid"] is False
    assert any("Row count mismatch" in err for err in res["errors"])

def test_extra_rows_fails(sample_submission, tmp_path):
    gen_path = str(tmp_path / "extra_rows.csv")
    df = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "target": [0.1, 0.9, 0.4, 0.5]
    })
    df.write_csv(gen_path)
    res = validate_submission(sample_submission, gen_path)
    assert res["is_valid"] is False
    assert any("Row count mismatch" in err for err in res["errors"])

def test_missing_column_fails(sample_submission, tmp_path):
    gen_path = str(tmp_path / "missing_col.csv")
    df = pl.DataFrame({
        "target": [0.1, 0.9, 0.4]
    })
    df.write_csv(gen_path)
    res = validate_submission(sample_submission, gen_path)
    assert res["is_valid"] is False
    assert any("Missing columns" in err for err in res["errors"])

def test_extra_column_fails(sample_submission, tmp_path):
    gen_path = str(tmp_path / "extra_col.csv")
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "target": [0.1, 0.9, 0.4],
        "extra": [1, 1, 1]
    })
    df.write_csv(gen_path)
    res = validate_submission(sample_submission, gen_path)
    assert res["is_valid"] is False
    assert any("Extra columns" in err for err in res["errors"])

def test_type_mismatch_fails(sample_submission, tmp_path):
    gen_path = str(tmp_path / "type_mismatch.csv")
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "target": ["a", "b", "c"]
    })
    df.write_csv(gen_path)
    res = validate_submission(sample_submission, gen_path)
    assert res["is_valid"] is False
    assert any("Dtype mismatch" in err for err in res["errors"])

def test_float32_float64_passes(tmp_path):
    sample = str(tmp_path / "sample.parquet")
    df_sample = pl.DataFrame({"id": [1, 2], "target": pl.Series([0.5, 0.5], dtype=pl.Float32)})
    df_sample.write_parquet(sample)

    gen = str(tmp_path / "gen.parquet")
    df_gen = pl.DataFrame({"id": [1, 2], "target": pl.Series([0.1, 0.9], dtype=pl.Float64)})
    df_gen.write_parquet(gen)

    res = validate_submission(sample, gen)
    assert res["is_valid"] is True

def test_int32_int64_passes(tmp_path):
    sample = str(tmp_path / "sample.parquet")
    df_sample = pl.DataFrame({"id": pl.Series([1, 2], dtype=pl.Int32), "target": [0.5, 0.5]})
    df_sample.write_parquet(sample)

    gen = str(tmp_path / "gen.parquet")
    df_gen = pl.DataFrame({"id": pl.Series([1, 2], dtype=pl.Int64), "target": [0.1, 0.9]})
    df_gen.write_parquet(gen)

    res = validate_submission(sample, gen)
    assert res["is_valid"] is True
