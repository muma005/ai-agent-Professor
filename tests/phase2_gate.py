# tests/phase2_gate.py
#
# PHASE 2 GATE — Run once after Day 14. All 3 conditions must pass.
# If any fail, Phase 2 is NOT complete. Fix the failure, re-run.
#
# Record passing commit hash in tests/regression/test_phase2_regression.py header.
#
# DO NOT run this file as part of regular pytest suite:
#   pytest tests/phase2_gate.py -v  ← run explicitly only

import json
import os
import sys
import tempfile

import pytest
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import initial_state, ProfessorState


# ── Helpers ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "spaceship_titanic")


def _build_pipeline_state(tmp_path=None) -> ProfessorState:
    """Build a minimal valid state with Spaceship Titanic data."""
    state = initial_state("spaceship-titanic", os.path.join(DATA_DIR, "train.csv"))
    state["target_col"] = "Transported"
    state["test_data_path"] = os.path.join(DATA_DIR, "test.csv")
    state["validation_strategy"] = {"target_type": "binary"}
    state["eda_report"] = {"target_distribution": {"imbalance_ratio": 0.50}}
    state["competition_fingerprint"] = {
        "task_type": "tabular", "target_type": "binary",
        "n_rows_bucket": "medium", "n_features_bucket": "medium",
        "imbalance_ratio": 0.50, "n_categorical_high_cardinality": 2,
        "has_temporal_feature": False,
    }
    state["feature_names"] = ["HomePlanet", "CryoSleep", "Cabin", "Destination",
                              "Age", "VIP", "RoomService", "FoodCourt",
                              "ShoppingMall", "Spa", "VRDeck", "Name"]
    if tmp_path:
        state["session_id"] = "phase2_gate"
        output_dir = os.path.join(str(tmp_path), "outputs", "phase2_gate")
        os.makedirs(output_dir, exist_ok=True)
    return state


def inject_leaky_feature(state: ProfessorState,
                         feature_type: str = "target_derived") -> ProfessorState:
    """Inject a leaky feature into the training data."""
    raw_path = state["raw_data_path"]
    df = pl.read_csv(raw_path, infer_schema_length=10000)
    target_col = state.get("target_col", df.columns[-1])

    if feature_type == "target_derived":
        # Shift target by 1 — directly derived from target
        target = df[target_col].cast(pl.Float64)
        leaky = target.shift(1).fill_null(0.0).alias("leaky_feature")
        df = df.with_columns(leaky)
    elif feature_type == "row_id":
        # Row index as feature — perfect predictor on train
        df = df.with_columns(pl.Series("row_id", list(range(len(df)))))

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.write_csv(tmp.name)
    return {**state, "raw_data_path": tmp.name}


def inject_legitimate_features(state: ProfessorState) -> ProfessorState:
    """Return state with only legitimate features (no leakage)."""
    return dict(state)  # original data is clean


def run_red_team_critic(state: ProfessorState) -> ProfessorState:
    """Run the red team critic agent."""
    from agents.red_team_critic import run_red_team_critic as _run
    return _run(state)


def run_validation_architect(state: ProfessorState) -> ProfessorState:
    """Run the validation architect agent."""
    from agents.validation_architect import run_validation_architect as _run
    return _run(state)


def get_nodes_executed(result: ProfessorState) -> list:
    """Get list of nodes that were executed from lineage."""
    lineage_path = result.get("lineage_log_path", "")
    nodes = []
    if lineage_path and os.path.exists(lineage_path):
        with open(lineage_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    agent = entry.get("agent", "")
                    if agent:
                        nodes.append(agent)
                except json.JSONDecodeError:
                    pass
    return nodes


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def full_pipeline_state(tmp_path):
    """A fully initialized pipeline state with real Spaceship Titanic data."""
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        pytest.skip("Spaceship Titanic train.csv not found — cannot run gate")
    return _build_pipeline_state(tmp_path)


# ─── CONDITION 1: Critic catches injected leakage ─────────────────────────────

class TestPhase2Condition1_CriticCatchesLeakage:
    """
    Inject a feature that is directly derived from the target column.
    Critic must return CRITICAL via the shuffled_target or id_only_model vector.
    Gate fails if critic returns OK or HIGH on this input.
    """

    def test_critic_catches_target_derived_feature(self, full_pipeline_state):
        state = inject_leaky_feature(full_pipeline_state, feature_type="target_derived")
        result = run_red_team_critic(state)
        assert result["critic_verdict"]["overall_severity"] == "CRITICAL", (
            f"GATE FAIL: Critic returned {result['critic_verdict']['overall_severity']} "
            f"on a target-derived feature. Expected CRITICAL. "
            f"Findings: {result['critic_verdict']['findings']}"
        )

    def test_critic_catches_id_as_feature(self, full_pipeline_state):
        state = inject_leaky_feature(full_pipeline_state, feature_type="row_id")
        result = run_red_team_critic(state)
        assert result["critic_verdict"]["overall_severity"] == "CRITICAL", (
            f"GATE FAIL: Critic returned {result['critic_verdict']['overall_severity']} "
            f"on row_id-as-feature. Expected CRITICAL."
        )

    def test_critic_clean_on_legitimate_features(self, full_pipeline_state):
        state = inject_legitimate_features(full_pipeline_state)
        result = run_red_team_critic(state)
        assert result["critic_verdict"]["overall_severity"] != "CRITICAL", (
            f"GATE FAIL: Critic returned CRITICAL on clean features (false positive). "
            f"Findings: {result['critic_verdict']['findings']}"
        )


# ─── CONDITION 2: Validation Architect blocks wrong metric ────────────────────

class TestPhase2Condition2_ValidationArchitectBlocksWrongMetric:
    """
    Set competition metric to AUC but inject a regression dataset (continuous target).
    Validation Architect must block this with an error before any training occurs.
    """

    def test_validation_architect_blocks_auc_on_regression(self, full_pipeline_state):
        state = {
            **full_pipeline_state,
            "metric": "auc",
            "target_type": "continuous",
            "validation_strategy": {"target_type": "continuous"},
        }
        result = run_validation_architect(state)
        # Validation architect should detect mismatch and set hitl_required
        has_error = (
            result.get("validation_error") is not None
            or result.get("hitl_required") is True
        )
        assert has_error, (
            "GATE FAIL: Validation Architect did not block/flag AUC metric "
            "on continuous target."
        )

    def test_validation_architect_blocks_rmse_on_binary(self, full_pipeline_state):
        state = {
            **full_pipeline_state,
            "metric": "rmse",
            "target_type": "binary",
            "validation_strategy": {"target_type": "binary"},
        }
        result = run_validation_architect(state)
        has_error = (
            result.get("validation_error") is not None
            or result.get("hitl_required") is True
        )
        assert has_error, (
            "GATE FAIL: Validation Architect did not block RMSE on binary target."
        )

    def test_validation_architect_passes_correct_metric(self, full_pipeline_state):
        state = {
            **full_pipeline_state,
            "metric": "auc",
            "target_type": "binary",
            "validation_strategy": {"target_type": "binary"},
        }
        result = run_validation_architect(state)
        assert result.get("validation_error") is None, (
            "GATE FAIL: Validation Architect blocked a correct metric/target combination."
        )


# ─── CONDITION 3: End-to-end CV better than Phase 1 baseline ─────────────────

class TestPhase2Condition3_CVBetterThanPhase1Baseline:
    """
    Phase 2 CV must exceed Phase 1 baseline by at least 0.005.
    Requires tests/regression/phase1_baseline.json to exist.
    """

    MINIMUM_IMPROVEMENT = 0.005

    def test_phase2_cv_beats_phase1_baseline(self):
        from pathlib import Path

        baseline_path = Path("tests/regression/phase1_baseline.json")
        if not baseline_path.exists():
            pytest.skip(
                "tests/regression/phase1_baseline.json not found. "
                "Run Phase 1 gate first and record the baseline CV score."
            )

        baseline = json.loads(baseline_path.read_text())
        phase1_cv = float(baseline["cv_mean"])

        # Phase 2 baseline must be recorded by a prior full run
        phase2_path = Path("tests/regression/phase2_baseline.json")
        if not phase2_path.exists():
            pytest.skip(
                "tests/regression/phase2_baseline.json not found. "
                "Run a full pipeline first to record Phase 2 CV."
            )

        phase2_data = json.loads(phase2_path.read_text())
        phase2_cv = float(phase2_data["cv_mean"])
        improvement = phase2_cv - phase1_cv

        assert improvement >= self.MINIMUM_IMPROVEMENT, (
            f"GATE FAIL: Phase 2 CV ({phase2_cv:.5f}) does not beat "
            f"Phase 1 baseline ({phase1_cv:.5f}) by the required {self.MINIMUM_IMPROVEMENT}. "
            f"Improvement: {improvement:+.5f}."
        )


# ─── GATE SUMMARY ─────────────────────────────────────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    """Print gate summary on completion."""
    if exitstatus == 0:
        print("\n" + "=" * 60)
        print("PHASE 2 GATE: ALL CONDITIONS PASSED")
        print("Next step: freeze tests/regression/test_phase2_regression.py")
        print("Record this commit hash in the regression file header.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("PHASE 2 GATE: FAILED — do not freeze regression test yet")
        print("Fix failures above and re-run: pytest tests/phase2_gate.py")
        print("=" * 60)
