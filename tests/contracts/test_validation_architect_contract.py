# tests/contracts/test_validation_architect_contract.py
# ─────────────────────────────────────────────────────────────────────────────
# Written: Day 8
# Status:  IMMUTABLE — never edit this file after today
#
# CONTRACT: run_validation_architect()
#   INPUT:   state["schema_path"]            — must exist
#   OUTPUT:  validation_strategy.json        — must have cv_type/n_splits/group_col
#            metric_contract.json            — must have scorer_fn/direction/forbidden_metrics
#   BLOCKER: CV/LB mismatch must set hitl_required=True and NOT proceed to optimizer
#   NEVER:   Return unknown scorer name. Use forbidden metrics. Proceed past mismatch.
# ─────────────────────────────────────────────────────────────────────────────
import pytest
import os
import json
from core.state import initial_state
from agents.data_engineer import run_data_engineer
from agents.validation_architect import run_validation_architect

FIXTURE_CSV = "tests/fixtures/tiny_train.csv"


@pytest.fixture(scope="module")
def validated_state():
    state = initial_state("test-validation", FIXTURE_CSV, budget_usd=2.0)
    state["target_col"] = "Transported"  # Required for data_engineer schema authority
    state = run_data_engineer(state)
    state = run_validation_architect(state)
    return state


class TestValidationArchitectContract:

    def test_runs_without_error(self, validated_state):
        assert validated_state is not None

    def test_validation_strategy_json_exists(self, validated_state):
        path = validated_state.get("validation_strategy_path")
        assert path is not None, "validation_strategy_path must be in state"
        assert os.path.exists(path), f"validation_strategy.json not found at {path}"

    def test_validation_strategy_has_cv_type(self, validated_state):
        vs = validated_state["validation_strategy"]
        assert "cv_type" in vs
        assert vs["cv_type"] in (
            "StratifiedKFold", "KFold", "GroupKFold", "TimeSeriesSplit"
        ), f"cv_type '{vs['cv_type']}' is not a recognised CV strategy"

    def test_validation_strategy_has_n_splits(self, validated_state):
        vs = validated_state["validation_strategy"]
        assert "n_splits" in vs
        assert isinstance(vs["n_splits"], int)
        assert 2 <= vs["n_splits"] <= 10

    def test_validation_strategy_has_group_col_key(self, validated_state):
        vs = validated_state["validation_strategy"]
        assert "group_col" in vs  # may be None — that's fine

    def test_metric_contract_json_exists(self, validated_state):
        path = validated_state.get("metric_contract_path")
        assert path is not None
        assert os.path.exists(path)

    def test_metric_contract_has_scorer_name(self, validated_state):
        path = validated_state["metric_contract_path"]
        mc   = json.load(open(path))
        assert "scorer_name" in mc
        assert isinstance(mc["scorer_name"], str)
        assert len(mc["scorer_name"]) > 0

    def test_metric_contract_has_direction(self, validated_state):
        path = validated_state["metric_contract_path"]
        mc   = json.load(open(path))
        assert mc["direction"] in ("maximize", "minimize")

    def test_metric_contract_has_forbidden_metrics(self, validated_state):
        path = validated_state["metric_contract_path"]
        mc   = json.load(open(path))
        assert "forbidden_metrics" in mc
        assert isinstance(mc["forbidden_metrics"], list)

    def test_task_type_written_to_state(self, validated_state):
        assert "task_type" in validated_state
        assert validated_state["task_type"] in (
            "tabular", "timeseries", "nlp", "image",
            "classification", "regression",
            "binary", "multiclass",  # data_engineer detects these directly
        )

    def test_no_hitl_on_clean_tabular_data(self, validated_state):
        # The fixture is clean tabular data — should not trigger HITL
        assert validated_state.get("hitl_required") is False, (
            "HITL triggered on clean tabular fixture — check mismatch detection logic"
        )

    def test_mismatch_triggers_hitl(self):
        """Inject a datetime column into schema and verify HITL triggers."""
        import tempfile, copy
        from core.state import initial_state
        from agents.data_engineer import run_data_engineer

        state = initial_state("test-mismatch", FIXTURE_CSV)
        state["target_col"] = "Transported"  # Required for data_engineer
        state = run_data_engineer(state)

        # Patch the schema to include a datetime column
        schema = json.load(open(state["schema_path"]))
        schema["columns"].append("transaction_date")
        schema["types"]["transaction_date"] = "Date"

        patched_path = state["schema_path"].replace(".json", "_patched.json")
        with open(patched_path, "w") as f:
            json.dump(schema, f)

        state["schema_path"] = patched_path
        # Force StratifiedKFold to be selected by removing any hints
        result = run_validation_architect(state)

        assert result["hitl_required"] is True, (
            "Datetime column should trigger CV/LB mismatch detection and set hitl_required=True"
        )
        assert "hitl_reason" in result
        assert len(result["hitl_reason"]) > 0

    def test_mismatch_halts_before_writing_metric_contract(self):
        """When mismatch is detected, validation_strategy.json must be written but
        the pipeline must not continue to ML Optimizer."""
        import tempfile
        from core.state import initial_state
        from agents.data_engineer import run_data_engineer

        state = initial_state("test-halt", FIXTURE_CSV)
        state["target_col"] = "Transported"  # Required for data_engineer
        state = run_data_engineer(state)

        schema = json.load(open(state["schema_path"]))
        schema["columns"].append("patient_id")
        schema["types"]["patient_id"] = "Utf8"
        patched_path = state["schema_path"].replace(".json", "_group_patched.json")
        with open(patched_path, "w") as f:
            json.dump(schema, f)

        state["schema_path"] = patched_path
        result = run_validation_architect(state)

        assert result["hitl_required"] is True
        # validation_strategy.json must exist (for debugging)
        assert os.path.exists(result["validation_strategy_path"])
        # metric_contract_path should NOT be in state when halted
        assert result.get("metric_contract_path") is None or not os.path.exists(
            result.get("metric_contract_path", "")
        ), "metric_contract.json must not be written when mismatch halts the pipeline"
