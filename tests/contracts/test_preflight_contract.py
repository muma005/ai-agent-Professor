# tests/contracts/test_preflight_contract.py

import pytest
import os
import json
import polars as pl
from pathlib import Path
from core.state import ProfessorState, initial_state
from shields.preflight import run_preflight_checks

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_tabular_dir(tmp_path):
    """Standard tabular competition — should pass cleanly."""
    train = pl.DataFrame({
        "id": range(100),
        "age": [25 + i % 50 for i in range(100)],
        "target": [0, 1] * 50,
    })
    test = train.drop("target").head(20)
    sample_sub = pl.DataFrame({"id": range(20), "target": [0.5] * 20})
    
    train.write_csv(tmp_path / "train.csv")
    test.write_csv(tmp_path / "test.csv")
    sample_sub.write_csv(tmp_path / "sample_submission.csv")
    return tmp_path

@pytest.fixture
def nlp_heavy_dir(tmp_path):
    """NLP features detected."""
    train = pl.DataFrame({
        "id": range(50),
        "text": ["This is a very long text string that should trigger the NLP flag for preflight" * 2] * 50
    })
    train.write_csv(tmp_path / "train.csv")
    pl.DataFrame({"id": range(50), "target": [0.5] * 50}).write_csv(tmp_path / "sample_submission.csv")
    return tmp_path

@pytest.fixture
def image_path_dir(tmp_path):
    """Image paths detected."""
    train = pl.DataFrame({
        "id": range(20),
        "img": [f"path/to/image_{i}.jpg" for i in range(20)]
    })
    train.write_csv(tmp_path / "train.csv")
    pl.DataFrame({"id": range(20), "target": [0.5] * 20}).write_csv(tmp_path / "sample_submission.csv")
    return tmp_path

# ── Tests ───────────────────────────────────────────────────────────────────

class TestPreflightContract:
    """
    Contract: Pre-flight Checks (Shield 6)
    Ensures data profiling, modality detection, and resource checks.
    """

    def test_clean_tabular_passes(self, clean_tabular_dir):
        """Verify standard data passes without blockers."""
        state_dict = initial_state(
            raw_data_path=str(clean_tabular_dir / "train.csv"),
            session_id="test-preflight-pass"
        )
        state = ProfessorState(**state_dict)
        
        final_state = run_preflight_checks(state)
        assert final_state.preflight_passed is True
        assert final_state.preflight_data_size_mb > 0

    def test_text_column_flagged(self, nlp_heavy_dir):
        """Verify long text triggers possible_nlp flag."""
        state_dict = initial_state(
            raw_data_path=str(nlp_heavy_dir / "train.csv")
        )
        state = ProfessorState(**state_dict)
        final_state = run_preflight_checks(state)
        
        nlp_warnings = [w for w in final_state.preflight_warnings if w["type"] == "possible_nlp"]
        assert len(nlp_warnings) > 0

    def test_image_paths_flagged(self, image_path_dir):
        """Verify .jpg strings trigger image modality flag."""
        state_dict = initial_state(
            raw_data_path=str(image_path_dir / "train.csv")
        )
        state = ProfessorState(**state_dict)
        final_state = run_preflight_checks(state)
        
        assert "image" in final_state.preflight_unsupported_modalities

    def test_submission_json_blocks(self, tmp_path):
        """Verify JSON submission blocks pipeline."""
        train = pl.DataFrame({"id": [1], "target": [0]})
        train.write_csv(tmp_path / "train.csv")
        
        with open(tmp_path / "sample_submission.json", "w") as f:
            json.dump({"test": "data"}, f)
            
        state_dict = initial_state(raw_data_path=str(tmp_path / "train.csv"))
        state = ProfessorState(**state_dict)
        final_state = run_preflight_checks(state)
        
        assert final_state.preflight_passed is False
        assert final_state.preflight_submission_format["format"] == "json"

    def test_mostly_null_flagged(self, tmp_path):
        """Verify high null percentage triggers warning."""
        train = pl.DataFrame({
            "id": range(100),
            "mostly_null": [None] * 95 + [1.0] * 5
        })
        train.write_csv(tmp_path / "train.csv")
        pl.DataFrame({"id": [1], "target": [0.5]}).write_csv(tmp_path / "sample_submission.csv")
        
        state_dict = initial_state(raw_data_path=str(tmp_path / "train.csv"))
        state = ProfessorState(**state_dict)
        final_state = run_preflight_checks(state)
        
        null_warnings = [w for w in final_state.preflight_warnings if w["type"] == "mostly_null"]
        assert len(null_warnings) > 0
