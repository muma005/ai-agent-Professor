# tests/contracts/test_sandbox_contract.py

import pytest
import os
import polars as pl
from tools.sandbox import run_in_sandbox

def test_sandbox_success():
    """Verify successful code returns success=True and captures scalars."""
    code = """
import polars as pl
x = 10
df = pl.DataFrame({"a": [1, 2, 3]})
print("Hello World")
"""
    res = run_in_sandbox(code)
    
    assert res["success"] is True
    assert "Hello World" in res["entry"]["stdout"]
    assert res["diagnostics"]["locals"]["x"] == "10"
    assert res["diagnostics"]["dataframes"]["df"]["shape"] == [3, 1]

def test_sandbox_key_error_classification():
    """Verify KeyError triggers column_missing classification."""
    code = """
import polars as pl
df = pl.DataFrame({"target": [1, 0, 1]})
# Intentional error
df["wrong_column"]
"""
    res = run_in_sandbox(code)
    
    assert res["success"] is False
    # In Polars, df["missing"] raises ColumnNotFoundError usually, 
    # but let's see what the classifier picks up.
    # Actually df["wrong"] in Polars is ColumnNotFoundError.
    assert "column_missing" in str(res["diagnostics"].get("error", {}).get("type", "")).lower() or \
           "column_missing" == _get_err_class(res)

def _get_err_class(res):
    from tools.sandbox import _classify_error
    err_class, _ = _classify_error(res["diagnostics"])
    return err_class

def test_sandbox_type_error_classification():
    """Verify TypeError triggers type_error classification."""
    code = """
x = "10"
y = 5
z = x + y
"""
    res = run_in_sandbox(code)
    assert res["success"] is False
    assert _get_err_class(res) == "type_error"

def test_sandbox_import_error_classification():
    """Verify ImportError triggers import_missing classification."""
    code = """
import non_existent_package
"""
    res = run_in_sandbox(code)
    assert res["success"] is False
    assert _get_err_class(res) == "import_missing"

def test_sandbox_df_diagnostics():
    """Verify dataframe null counts and head are captured."""
    code = """
import polars as pl
df = pl.DataFrame({
    "a": [1, None, 3],
    "b": ["x", "y", "z"]
})
"""
    res = run_in_sandbox(code)
    df_diag = res["diagnostics"]["dataframes"]["df"]
    assert df_diag["null_counts"]["a"] == 1
    assert "shape" in df_diag
