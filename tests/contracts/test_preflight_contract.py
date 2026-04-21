# tests/contracts/test_preflight_contract.py

import pytest
import os
import polars as pl
from shields.preflight import run_preflight_checks, _profile_columns
from core.state import ProfessorState

def test_audit_missing_files():
    """Verify missing files are detected."""
    state = ProfessorState(
        raw_data_path="non_existent.csv",
        test_data_path="missing.csv",
        sample_submission_path="empty.csv"
    )
    
    new_state = run_preflight_checks(state)
    assert new_state.preflight_passed is False
    assert any("File not found" in w for w in new_state.preflight_warnings)

def test_column_profiling(tmp_path):
    """Verify heuristic detection of column types."""
    path = tmp_path / "train.csv"
    df = pl.DataFrame({
        "text": ["This is a very long text " * 10, "another long one " * 10],
        "image": ["cat.jpg", "dog.png"],
        "data": ["{\"a\": 1}", "[1, 2, 3]"],
        "normal": [1, 2]
    })
    df.write_csv(path)
    
    profile = _profile_columns(str(path))
    assert "nlp" in profile
    assert "image" in profile
    assert "json" in profile
    assert "text" in profile["nlp"]
    assert "image" in profile["image"]
    assert "data" in profile["json"]

def test_preflight_node_pass(tmp_path):
    """Verify pass when files exist."""
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    sub = tmp_path / "sub.csv"
    
    pl.DataFrame({"a": [1]}).write_csv(train)
    pl.DataFrame({"a": [1]}).write_csv(test)
    pl.DataFrame({"a": [1]}).write_csv(sub)
    
    state = ProfessorState(
        raw_data_path=str(train),
        test_data_path=str(test),
        sample_submission_path=str(sub)
    )
    
    new_state = run_preflight_checks(state)
    assert new_state.preflight_passed is True
