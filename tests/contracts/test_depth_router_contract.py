# tests/contracts/test_depth_router_contract.py

import pytest
from graph.depth_router import classify_pipeline_depth

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_competition():
    """1000 rows, 14 features, binary, standard metric — should be SPRINT."""
    return {
        "preflight_data_files": [{"name": "train.csv", "size_mb": 0.5}],
        "preflight_warnings": [],
        "preflight_target_type": "binary",
        "preflight_data_size_mb": 0.8,
        "n_rows": 1000,
        "n_features": 14,
        "metric_name": "roc_auc",
    }

@pytest.fixture
def complex_competition():
    """200K rows, 300 features, custom metric — should be MARATHON."""
    return {
        "preflight_data_files": [{"name": "train.csv", "size_mb": 800}],
        "preflight_warnings": [],
        "preflight_target_type": "regression",
        "preflight_data_size_mb": 900,
        "n_rows": 200000,
        "n_features": 300,
        "metric_name": "custom_weighted_f1",
    }

@pytest.fixture
def ambiguous_competition():
    """15K rows, 25 features — ambiguous, should default to STANDARD."""
    return {
        "preflight_data_files": [{"name": "train.csv", "size_mb": 50}],
        "preflight_warnings": [],
        "preflight_target_type": "binary",
        "preflight_data_size_mb": 55,
        "n_rows": 15000,
        "n_features": 25,
        "metric_name": "roc_auc",
    }

# ── Tests ───────────────────────────────────────────────────────────────────

class TestDepthRouterContract:
    """
    Contract: Complexity-Gated Pipeline Depth (Component 7)
    """

    def test_simple_gets_sprint(self, simple_competition):
        res = classify_pipeline_depth(**simple_competition)
        assert res["depth"] == "sprint"

    def test_complex_gets_marathon(self, complex_competition):
        res = classify_pipeline_depth(**complex_competition)
        assert res["depth"] == "marathon"

    def test_ambiguous_gets_standard(self, ambiguous_competition):
        res = classify_pipeline_depth(**ambiguous_competition)
        assert res["depth"] == "standard"

    def test_sprint_skips_correct_agents(self, simple_competition):
        res = classify_pipeline_depth(**simple_competition)
        expected = ["competition_intel", "domain_researcher", "shift_detector", "creative_hypothesis", "problem_reframer", "pseudo_label_agent", "post_processor"]
        for agent in expected:
            assert agent in res["agents_skipped"]

    def test_sprint_optuna_50(self, simple_competition):
        res = classify_pipeline_depth(**simple_competition)
        assert res["optuna_trials"] == 50

    def test_marathon_optuna_200(self, complex_competition):
        res = classify_pipeline_depth(**complex_competition)
        assert res["optuna_trials"] == 200

    def test_standard_optuna_100(self, ambiguous_competition):
        res = classify_pipeline_depth(**ambiguous_competition)
        assert res["optuna_trials"] == 100

    def test_sprint_critic_4_vectors(self, simple_competition):
        res = classify_pipeline_depth(**simple_competition)
        assert res["critic_vectors"] == 4

    def test_standard_critic_9_vectors(self, ambiguous_competition):
        res = classify_pipeline_depth(**ambiguous_competition)
        assert res["critic_vectors"] == 9

    def test_sprint_feature_rounds_2(self, simple_competition):
        res = classify_pipeline_depth(**simple_competition)
        assert res["feature_rounds"] == 2

    def test_marathon_feature_rounds_7(self, complex_competition):
        res = classify_pipeline_depth(**complex_competition)
        assert res["feature_rounds"] == 7

    def test_operator_override_respected(self, complex_competition):
        res = classify_pipeline_depth(operator_override="sprint", **complex_competition)
        assert res["depth"] == "sprint"
        assert res["auto_detected"] is False

    def test_custom_metric_triggers_marathon(self, simple_competition):
        simple_competition["metric_name"] = "custom_xyz"
        res = classify_pipeline_depth(**simple_competition)
        assert res["depth"] == "marathon"

    def test_reason_string_populated(self, simple_competition):
        res = classify_pipeline_depth(**simple_competition)
        assert "rows=" in res["reason"]
        assert "features=" in res["reason"]

    def test_blocking_warnings_prevent_sprint(self, simple_competition):
        simple_competition["preflight_warnings"] = [{"type": "blocking", "description": "Too many nulls"}]
        res = classify_pipeline_depth(**simple_competition)
        assert res["depth"] != "sprint"

    def test_depth_result_has_all_fields(self, simple_competition):
        res = classify_pipeline_depth(**simple_competition)
        keys = ["depth", "auto_detected", "reason", "agents_skipped", "optuna_trials", "feature_rounds", "critic_vectors"]
        for k in keys:
            assert k in res
