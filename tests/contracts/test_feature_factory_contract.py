# tests/contracts/test_feature_factory_contract.py

import pytest
import os
import json
import polars as pl
from pathlib import Path
from core.state import initial_state
from agents.feature_factory import run_feature_factory

FIXTURE_TRAIN = "data/spaceship_titanic/train.parquet"
FIXTURE_PREP = "outputs/test-ff/preprocessor.pkl"
FIXTURE_SCHEMA = "outputs/test-ff/schema.json"

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def feature_factory_state():
    """Valid initial state for FeatureFactory."""
    os.makedirs("outputs/test-ff", exist_ok=True)
    
    # 1. Ensure Parquet
    if not os.path.exists(FIXTURE_TRAIN):
        df = pl.read_csv("data/spaceship_titanic/train.csv")
        df.write_parquet(FIXTURE_TRAIN)
    
    # 2. Mock Preprocessor
    from core.preprocessor import TabularPreprocessor
    prep = TabularPreprocessor(target_col="Transported")
    prep.save(FIXTURE_PREP)
    
    # 3. Mock Schema
    schema = {
        "target_col": "Transported",
        "id_columns": ["PassengerId"],
        "columns": ["PassengerId", "HomePlanet", "Transported"],
        "types": {"PassengerId": "String", "HomePlanet": "String", "Transported": "Boolean"},
        "cardinality": {"HomePlanet": 3}
    }
    with open(FIXTURE_SCHEMA, "w") as f:
        json.dump(schema, f)

    return initial_state(
        session_id="test-ff",
        clean_data_path=FIXTURE_TRAIN,
        preprocessor_path=FIXTURE_PREP,
        schema_path=FIXTURE_SCHEMA,
        target_col="Transported"
    )

# ── Tests ───────────────────────────────────────────────────────────────────

class TestFeatureFactoryContract:
    """
    Contract: Feature Factory Agent
    Ensures multi-round feature generation and persistence.
    """

    def test_feature_manifest_written_and_valid(self, feature_factory_state):
        """Verify feature_manifest.json exists and contains candidates."""
        state = run_feature_factory(feature_factory_state)
        path = Path(state["feature_manifest_path"] if "feature_manifest_path" in state else f"outputs/{state['session_id']}/feature_manifest.json")
        assert path.exists()
        
        manifest = json.loads(path.read_text())
        assert "features" in manifest
        assert isinstance(manifest["features"], list)

    def test_feature_data_parquet_exists(self, feature_factory_state):
        """Verify augmented data is persisted as Parquet."""
        state = run_feature_factory(feature_factory_state)
        path = Path(state["feature_data_path"])
        assert path.exists()
        assert path.suffix == ".parquet"

    def test_feature_order_populated(self, feature_factory_state):
        """Verify feature_order key is written to state."""
        state = run_feature_factory(feature_factory_state)
        assert "feature_order" in state
        assert isinstance(state["feature_order"], list)
        assert len(state["feature_order"]) > 0

    def test_preprocessor_updated(self, feature_factory_state):
        """Verify preprocessor_path is updated with augmented version."""
        state = run_feature_factory(feature_factory_state)
        assert "preprocessor_path" in state
        assert "preprocessor_ff.pkl" in state["preprocessor_path"]
