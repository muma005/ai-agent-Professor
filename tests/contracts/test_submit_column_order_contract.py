# tests/contracts/test_submit_column_order_contract.py
# Day 13 Contract: Column order enforcement in build_submission
# 3 contract tests — IMMUTABLE after Day 13

import json
import pytest
import polars as pl
from pathlib import Path
from core.state import initial_state
from tools.submit_tools import build_submission


SESSION_ID = "test_col_order_contract"


def _setup(feature_order):
    out_dir = Path(f"outputs/{SESSION_ID}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"cv_mean": 0.85, "feature_order": feature_order}
    (out_dir / "metrics.json").write_text(json.dumps(metrics))
    state = initial_state("test_comp", "data/test.csv")
    state["session_id"] = SESSION_ID
    return state


def _teardown():
    import shutil
    out_dir = Path(f"outputs/{SESSION_ID}")
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)


class TestSubmitColumnOrderContract:

    def setup_method(self):
        _teardown()

    def teardown_method(self):
        _teardown()

    def test_feature_order_saved_in_metrics_json(self):
        """Contract: metrics.json must contain feature_order list."""
        state = _setup(["col_a", "col_b", "col_c"])
        metrics = json.loads(Path(f"outputs/{SESSION_ID}/metrics.json").read_text())
        assert "feature_order" in metrics
        assert isinstance(metrics["feature_order"], list)
        assert len(metrics["feature_order"]) == 3

    def test_raises_on_missing_column(self):
        """Contract: build_submission raises ValueError when test_df lacks a column."""
        state = _setup(["col_a", "col_b", "col_c"])
        test_df = pl.DataFrame({"col_a": [1], "col_b": [2]})  # missing col_c
        with pytest.raises(ValueError, match="missing columns"):
            build_submission(state, test_df)

    def test_raises_on_wrong_column_order(self):
        """Contract: assert fires when .select() produces wrong order."""
        from unittest.mock import patch
        state = _setup(["b_col", "a_col"])
        test_df = pl.DataFrame({"b_col": [1], "a_col": [2]})

        def bad_select(self_df, *args, **kwargs):
            return pl.DataFrame({"a_col": [2], "b_col": [1]})

        with patch.object(pl.DataFrame, "select", bad_select):
            with pytest.raises(AssertionError):
                build_submission(state, test_df)
