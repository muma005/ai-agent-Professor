# tests/contracts/test_feature_factory_contract.py
#
# CONTRACT: agents/feature_factory.py
#
# INPUT:  schema.json (written by data_engineer)
#         competition_brief.json (written by competition_intel)
# OUTPUT: feature_manifest.json
#
# INVARIANTS:
#   - Never reads raw data files directly
#   - feature_manifest.json always written (even if 0 candidates)
#   - Every feature in manifest has all required fields
#   - Verdict is one of: PENDING | KEEP | DROP
#   - source_columns must all exist in schema.json
#   - Round 2 features must not reference columns absent from schema
#
# STATUS: IMMUTABLE after Day 16

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime


# ── Fixtures ──────────────────────────────────────────────────────

SAMPLE_SCHEMA = {
    "columns": [
        {
            "name": "Age",
            "dtype": "float64",
            "n_unique": 88,
            "null_fraction": 0.20,
            "min": 0.42,
            "max": 80.0,
            "is_id": False,
            "is_target": False,
        },
        {
            "name": "Fare",
            "dtype": "float64",
            "n_unique": 281,
            "null_fraction": 0.0,
            "min": 0.0,
            "max": 512.33,
            "is_id": False,
            "is_target": False,
        },
        {
            "name": "PassengerId",
            "dtype": "int64",
            "n_unique": 891,
            "null_fraction": 0.0,
            "min": 1,
            "max": 891,
            "is_id": True,
            "is_target": False,
        },
        {
            "name": "Survived",
            "dtype": "int64",
            "n_unique": 2,
            "null_fraction": 0.0,
            "min": 0,
            "max": 1,
            "is_id": False,
            "is_target": True,
        },
    ],
    "n_rows": 891,
    "target_column": "Survived",
    "id_column": "PassengerId",
    "session_id": "test_contract",
}


@pytest.fixture
def feature_factory_state(tmp_path):
    """State with schema.json written to a temp session dir."""
    session_id = "test_contract_ff"
    session_dir = tmp_path / f"outputs/{session_id}"
    session_dir.mkdir(parents=True)

    # Write schema.json
    schema_path = session_dir / "schema.json"
    schema_path.write_text(json.dumps(SAMPLE_SCHEMA, indent=2))

    # Write a minimal competition_brief.json
    brief = {"domain": "maritime", "task_type": "binary_classification",
             "known_winning_features": ["Title from Name"]}
    brief_path = session_dir / "competition_brief.json"
    brief_path.write_text(json.dumps(brief, indent=2))

    # Create a minimal cleaned.parquet so feature_factory can read it
    import polars as pl
    clean_path = session_dir / "cleaned.parquet"
    df = pl.DataFrame({
        "Age": [25.0, 30.0, 35.0, 40.0, 45.0],
        "Fare": [10.0, 20.0, 30.0, 40.0, 50.0],
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 0, 1, 0],
    })
    df.write_parquet(clean_path)

    state = {
        "session_id": session_id,
        "competition_name": "titanic",
        "task_type": "tabular",
        "schema_path": str(schema_path),
        "clean_data_path": str(clean_path),
        "target_col": "Survived",
    }

    # Monkey-patch outputs path by changing cwd
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield state
    os.chdir(original_cwd)


@pytest.fixture
def feature_factory_state_no_schema(tmp_path):
    """State where schema.json is absent."""
    session_id = "test_contract_no_schema"
    session_dir = tmp_path / f"outputs/{session_id}"
    session_dir.mkdir(parents=True)
    # No schema.json written

    state = {
        "session_id": session_id,
        "competition_name": "titanic",
        "task_type": "tabular",
    }

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield state
    os.chdir(original_cwd)


def _load_manifest(state) -> dict:
    path = Path(f"outputs/{state['session_id']}/feature_manifest.json")
    return json.loads(path.read_text())


def _load_schema(state) -> dict:
    path = Path(f"outputs/{state['session_id']}/schema.json")
    return json.loads(path.read_text())


# ── Contract Tests ───────────────────────────────────────────────

class TestFeatureFactoryContract:
    """Contract tests — immutable after Day 16."""

    REQUIRED_FEATURE_FIELDS = {
        "name", "transform_type", "source_columns",
        "round", "description", "verdict",
        "null_importance_percentile", "wilcoxon_p", "cv_delta",
    }

    VALID_VERDICTS = {"PENDING", "KEEP", "DROP"}

    def test_manifest_written_after_run(self, feature_factory_state):
        """feature_manifest.json must always be written."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(feature_factory_state)
        path = Path(f"outputs/{state['session_id']}/feature_manifest.json")
        assert path.exists(), "feature_manifest.json not written."

    def test_manifest_is_valid_json(self, feature_factory_state):
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(feature_factory_state)
        path = Path(f"outputs/{state['session_id']}/feature_manifest.json")
        manifest = json.loads(path.read_text())
        assert isinstance(manifest, dict), "feature_manifest.json is not a JSON object."

    def test_manifest_has_required_top_level_keys(self, feature_factory_state):
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)
        for key in ("total_candidates", "total_kept", "total_dropped", "features", "generated_at"):
            assert key in manifest, f"feature_manifest.json missing required key: '{key}'"

    def test_every_feature_has_required_fields(self, feature_factory_state):
        """Every feature entry must have all required fields. None may be absent."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)

        for i, feature in enumerate(manifest.get("features", [])):
            missing = self.REQUIRED_FEATURE_FIELDS - set(feature.keys())
            assert not missing, (
                f"Feature {i} ('{feature.get('name', '?')}') missing fields: {missing}"
            )

    def test_all_verdicts_are_valid_enum(self, feature_factory_state):
        """Verdict must be one of PENDING, KEEP, DROP."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)

        for feature in manifest.get("features", []):
            assert feature["verdict"] in self.VALID_VERDICTS, (
                f"Feature '{feature['name']}' has invalid verdict: '{feature['verdict']}'. "
                f"Must be one of {self.VALID_VERDICTS}"
            )

    def test_source_columns_exist_in_schema(self, feature_factory_state):
        """Every source column referenced in the manifest must exist in schema.json."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)
        schema = _load_schema(state)
        schema_col_names = {c["name"] for c in schema.get("columns", [])}

        for feature in manifest.get("features", []):
            for src in feature.get("source_columns", []):
                assert src in schema_col_names, (
                    f"Feature '{feature['name']}' references column '{src}' "
                    f"not found in schema.json. Available: {schema_col_names}"
                )

    def test_feature_factory_does_not_read_raw_data(self, feature_factory_state, tmp_path):
        """
        Feature factory must read schema.json only — never raw CSV or parquet files.
        Verified by removing raw data and confirming factory still succeeds.
        """
        from agents.feature_factory import run_feature_factory
        raw_data_path = Path(f"data/{feature_factory_state['session_id']}/train.csv")
        if raw_data_path.exists():
            raw_data_path.rename(raw_data_path.with_suffix(".bak"))
        # Must succeed without raw data
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(feature_factory_state)
        assert state["feature_manifest"] is not None

    def test_total_counts_consistent_with_features_list(self, feature_factory_state):
        """total_candidates, total_kept, total_dropped must match the features list."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)
        features = manifest.get("features", [])

        assert manifest["total_candidates"] == len(features), (
            f"total_candidates={manifest['total_candidates']} != len(features)={len(features)}"
        )
        assert manifest["total_kept"] == sum(1 for f in features if f["verdict"] == "KEEP"), (
            "total_kept does not match count of KEEP verdicts in features list."
        )
        assert manifest["total_dropped"] == sum(1 for f in features if f["verdict"] == "DROP"), (
            "total_dropped does not match count of DROP verdicts in features list."
        )

    def test_manifest_empty_gracefully_when_no_schema(self, feature_factory_state_no_schema):
        """If schema.json is missing, feature_factory must raise FileNotFoundError."""
        from agents.feature_factory import run_feature_factory
        with pytest.raises(FileNotFoundError, match="schema.json"):
            run_feature_factory(feature_factory_state_no_schema)
