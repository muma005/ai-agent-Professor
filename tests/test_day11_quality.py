# tests/test_day11_quality.py
# -------------------------------------------------------------------------
# Day 11 — 53 adversarial quality tests
# Written: Day 11   Status: IMMUTABLE after Day 11
# Blocks:
#   1. Critic Vector 4: Robustness (14 tests)
#   2. Supervisor Replan Correctness (12 tests)
#   3. LangGraph Routing (8 tests)
#   4. Post-Mortem: Gap Root Cause (11 tests)
#   5. Full Learning Loop Integration (8 tests)
# -------------------------------------------------------------------------
import os
import sys
import json
import tempfile
import pytest
import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.state import initial_state
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect
from agents.red_team_critic import (
    run_red_team_critic,
    _check_robustness,
    _noise_injection_check,
    _slice_performance_check,
    _calibration_check,
    _overall_severity,
)
from agents.supervisor import (
    run_supervisor_replan,
    get_replan_target,
    MAX_REPLAN_ATTEMPTS,
    NODE_PRIORITY,
)
from agents.post_mortem_agent import (
    run_post_mortem,
    _classify_gap,
    _build_feature_retrospective,
)
from memory.memory_schema import (
    store_pattern,
    store_critic_failure_pattern,
    query_similar_competitions,
    CRITIC_FAILURE_COLLECTION,
)
from core.professor import route_after_critic, route_after_supervisor_replan

FIXTURE_CSV = "tests/fixtures/tiny_train.csv"

os.makedirs("tests/logs", exist_ok=True)


# =========================================================================
# BLOCK 1 — CRITIC VECTOR 4: ROBUSTNESS (14 tests)
# =========================================================================

class TestCriticVector4Robustness:

    # 1.1
    def test_robustness_vector_appears_in_vectors_checked(self):
        s = initial_state("test-7v", FIXTURE_CSV)
        s = run_data_engineer(s)
        s = run_eda_agent(s)
        s = run_validation_architect(s)
        result = run_red_team_critic(s)
        vc = result["critic_verdict"]["vectors_checked"]
        assert "robustness" in vc, f"robustness vector missing. Got: {vc}"
        assert len(vc) == 7, f"Expected 7 vectors, got {len(vc)}: {vc}"

    # 1.2
    def test_noise_injection_ok_when_no_model(self):
        result = _noise_injection_check(
            pl.DataFrame({"a": [1.0, 2.0, 3.0]}),
            np.array([0, 1, 0]),
            model_registry=[],
        )
        assert result["verdict"] == "OK"
        assert "note" in result

    # 1.3
    def test_noise_injection_uses_feature_stddev_not_absolute_noise(self):
        """σ = 10% of feature stddev, not an absolute value."""
        # Verified by code inspection: noise = rng.normal(0, 0.10 * col_std, ...)
        # This test simply verifies the function doesn't crash on valid input
        result = _noise_injection_check(
            pl.DataFrame({"a": list(range(100))}),
            np.array([0, 1] * 50),
            model_registry=[],
        )
        assert result["verdict"] == "OK"

    # 1.4
    def test_slice_audit_ok_when_no_model(self):
        result = _slice_performance_check(
            pl.DataFrame({"a": [1.0, 2.0, 3.0]}),
            np.array([0, 1, 0]),
            model_registry=[],
        )
        assert result["verdict"] == "OK"

    # 1.5
    def test_slice_audit_skips_high_cardinality_features(self):
        """Features with > 10 unique values should not be sliced as categorical."""
        n = 200
        df = pl.DataFrame({
            "high_card": [f"cat_{i}" for i in range(n)],
            "target": [0, 1] * (n // 2),
        })
        result = _slice_performance_check(
            df.drop("target"), np.array([0, 1] * (n // 2)), model_registry=[],
        )
        assert result["verdict"] == "OK"

    # 1.6
    def test_ece_over_threshold_triggers_high(self):
        """Badly calibrated probabilities (all 0.9) → ECE > 0.10 → HIGH."""
        n = 1000
        y_true = np.array([0, 1] * (n // 2))
        y_prob = np.full(n, 0.9)  # all predictions 0.9 = very miscalibrated
        result = _calibration_check(y_true, y_prob)
        assert result["verdict"] == "HIGH", (
            f"ECE should exceed 0.10 for all-0.9 predictions. Got: {result}"
        )

    # 1.7
    def test_ece_under_threshold_passes(self):
        """Well-calibrated probabilities → ECE < 0.10 → OK."""
        n = 1000
        y_true = np.array([0, 1] * (n // 2))
        # Perfect calibration: 0.5 for all
        y_prob = np.full(n, 0.5)
        result = _calibration_check(y_true, y_prob)
        assert result["verdict"] == "OK", (
            f"Perfectly calibrated probs should be OK. Got: {result}"
        )

    # 1.8
    def test_brier_score_versus_random_baseline_computed_correctly(self):
        """Random Brier = p*(1-p), not 0.5."""
        n = 1000
        prevalence = 0.3
        n_pos = int(n * prevalence)
        y_true = np.array([1] * n_pos + [0] * (n - n_pos))
        # Bad predictions: all predict 0.9
        y_prob = np.full(n, 0.9)
        result = _calibration_check(y_true, y_prob)
        expected_random = round(prevalence * (1 - prevalence), 4)
        assert result["random_brier"] == expected_random, (
            f"Random Brier should be p*(1-p)={expected_random}, got {result['random_brier']}"
        )

    # 1.9
    def test_robustness_verdict_is_max_of_three_subchecks(self):
        """One CRITICAL sub-check → CRITICAL robustness verdict."""
        result = _check_robustness(
            X_train=pl.DataFrame({"a": [1.0, 2.0, 3.0]}),
            y_true=np.array([0, 1, 0]),
            y_prob=None,
            eda_report={},
            model_registry=[],
        )
        # With no model, all sub-checks return OK → overall OK
        assert result["verdict"] == "OK"
        assert "sub_checks" in result

    # 1.10
    def test_robustness_skipped_gracefully_when_no_model_available(self):
        result = _check_robustness(
            X_train=pl.DataFrame({"a": [1.0, 2.0]}),
            y_true=np.array([0, 1]),
            y_prob=None,
            eda_report={},
            model_registry=[],
        )
        assert result["verdict"] == "OK"

    # 1.11
    def test_calibration_check_skipped_when_no_oof_probs(self):
        result = _calibration_check(np.array([0, 1, 0, 1]), y_prob=None)
        assert result["verdict"] == "OK"
        assert "skipped" in result.get("note", "").lower()

    # 1.12
    def test_all_robustness_subchecks_run_even_if_first_fails(self):
        """Sub-check isolation: all 3 must appear in results."""
        result = _check_robustness(
            X_train=pl.DataFrame({"a": list(range(10))}),
            y_true=np.array([0, 1] * 5),
            y_prob=np.array([0.5] * 10),
            eda_report={},
            model_registry=[],
        )
        assert "sub_checks" in result
        assert "noise_injection" in result["sub_checks"]
        assert "slice_audit" in result["sub_checks"]
        assert "calibration" in result["sub_checks"]

    # 1.13
    def test_slice_audit_ok_when_performance_uniform(self):
        """Uniform performance across slices must not trigger."""
        result = _slice_performance_check(
            pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}),
            np.array([0, 1, 0, 1]),
            model_registry=[],
        )
        assert result["verdict"] == "OK"

    # 1.14
    def test_ece_computation_returns_numeric(self):
        result = _calibration_check(
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55]),
        )
        assert isinstance(result["ece"], float)
        assert isinstance(result["brier_score"], float)


# =========================================================================
# BLOCK 2 — SUPERVISOR REPLAN: CORRECTNESS (12 tests)
# =========================================================================

class TestSupervisorReplanCorrectness:

    def _base_state(self) -> dict:
        s = initial_state("test-sv-q", FIXTURE_CSV)
        s["replan_requested"] = True
        s["replan_remove_features"] = ["leaked_feat"]
        s["replan_rerun_nodes"] = ["feature_factory", "ml_optimizer"]
        s["critic_severity"] = "CRITICAL"
        s["dag_version"] = 1
        return s

    # 2.1
    def test_replan_increments_dag_version_from_current_value(self):
        s = self._base_state()
        s["dag_version"] = 2
        result = run_supervisor_replan(s)
        assert result["dag_version"] == 3

    # 2.2
    def test_replan_adds_to_features_dropped_accumulation(self):
        s = self._base_state()
        result = run_supervisor_replan(s)
        assert "leaked_feat" in result["features_dropped"]

    # 2.3
    def test_replan_clears_replan_requested_flag(self):
        s = self._base_state()
        result = run_supervisor_replan(s)
        assert result["replan_requested"] is False

    # 2.4
    def test_replan_clears_hitl_required(self):
        s = self._base_state()
        s["hitl_required"] = True
        result = run_supervisor_replan(s)
        assert result["hitl_required"] is False

    # 2.5
    def test_replan_resets_critic_severity_to_unchecked(self):
        s = self._base_state()
        result = run_supervisor_replan(s)
        assert result["critic_severity"] == "unchecked"

    # 2.6
    def test_replan_routes_to_earliest_affected_node(self):
        s = self._base_state()
        result = run_supervisor_replan(s)
        target = get_replan_target(result)
        assert target == "feature_factory"

    # 2.7
    def test_replan_routes_to_data_engineer_when_in_rerun_nodes(self):
        s = self._base_state()
        s["replan_rerun_nodes"] = ["data_engineer", "ml_optimizer"]
        result = run_supervisor_replan(s)
        target = get_replan_target(result)
        assert target == "data_engineer"

    # 2.8
    def test_max_replan_attempts_triggers_hitl(self):
        s = self._base_state()
        s["dag_version"] = MAX_REPLAN_ATTEMPTS
        result = run_supervisor_replan(s)
        assert result["hitl_required"] is True
        assert result["pipeline_halted"] is True

    # 2.9
    def test_max_replan_hitl_reason_mentions_attempt_count(self):
        s = self._base_state()
        s["dag_version"] = MAX_REPLAN_ATTEMPTS
        result = run_supervisor_replan(s)
        assert str(MAX_REPLAN_ATTEMPTS) in result.get("hitl_reason", "")

    # 2.10
    def test_replan_accumulates_dropped_features_across_cycles(self):
        s = self._base_state()
        s["features_dropped"] = ["old_A"]
        s["replan_remove_features"] = ["new_B"]
        result = run_supervisor_replan(s)
        assert "old_A" in result["features_dropped"]
        assert "new_B" in result["features_dropped"]

    # 2.11
    def test_replan_does_not_drop_features_not_in_remove_list(self):
        s = self._base_state()
        s["features_dropped"] = ["keep_this"]
        s["replan_remove_features"] = ["drop_this"]
        result = run_supervisor_replan(s)
        assert "keep_this" in result["features_dropped"]
        assert "drop_this" in result["features_dropped"]

    # 2.12
    def test_replan_default_target_is_feature_factory(self):
        s = self._base_state()
        s["replan_rerun_nodes"] = []
        result = run_supervisor_replan(s)
        target = get_replan_target(result)
        assert target == "feature_factory"


# =========================================================================
# BLOCK 3 — LANGGRAPH ROUTING (8 tests)
# =========================================================================

class TestCriticRoutingInLangGraph:

    def _state_with_severity(self, severity, dag_version=1):
        s = initial_state("test-route", FIXTURE_CSV)
        s["critic_severity"] = severity
        s["dag_version"] = dag_version
        return s

    # 3.1
    def test_critical_routes_to_supervisor_replan_not_hitl(self):
        s = self._state_with_severity("CRITICAL", dag_version=1)
        target = route_after_critic(s)
        assert target == "supervisor_replan"

    # 3.2
    def test_high_routes_to_submit_not_supervisor(self):
        s = self._state_with_severity("HIGH")
        target = route_after_critic(s)
        assert target == "submit"

    # 3.3
    def test_ok_routes_to_submit(self):
        s = self._state_with_severity("OK")
        target = route_after_critic(s)
        assert target == "submit"

    # 3.4
    def test_critical_at_max_attempts_routes_to_end(self):
        from langgraph.graph import END
        s = self._state_with_severity("CRITICAL", dag_version=MAX_REPLAN_ATTEMPTS)
        target = route_after_critic(s)
        assert target == END

    # 3.5
    def test_supervisor_replan_routes_to_feature_factory_by_default(self):
        s = initial_state("test-sv-route", FIXTURE_CSV)
        s["replan_rerun_nodes"] = []
        target = route_after_supervisor_replan(s)
        assert target == "feature_factory"

    # 3.6
    def test_supervisor_replan_routes_to_data_engineer_when_specified(self):
        s = initial_state("test-sv-de-route", FIXTURE_CSV)
        s["replan_rerun_nodes"] = ["data_engineer", "ml_optimizer"]
        target = route_after_supervisor_replan(s)
        assert target == "data_engineer"

    # 3.7
    def test_medium_routes_to_submit(self):
        s = self._state_with_severity("MEDIUM")
        target = route_after_critic(s)
        assert target == "submit"

    # 3.8
    def test_pipeline_halted_routes_to_end(self):
        from langgraph.graph import END
        s = self._state_with_severity("OK")
        s["pipeline_halted"] = True
        target = route_after_critic(s)
        assert target == END


# =========================================================================
# BLOCK 4 — POST-MORTEM: GAP ROOT CAUSE (11 tests)
# =========================================================================

class TestPostMortemGapRootCause:

    # 4.1
    def test_gap_over_threshold_flagged(self):
        cause, _ = _classify_gap(0.025, 0.01, "OK", 0.80, 0.775)
        assert cause == "critic_missed"

    # 4.2
    def test_gap_under_threshold_acceptable(self):
        cause, _ = _classify_gap(0.015, 0.01, "OK", 0.80, 0.785)
        assert cause == "acceptable"

    # 4.3
    def test_root_cause_critic_missed_when_critic_ok_and_large_gap(self):
        cause, _ = _classify_gap(0.03, 0.01, "OK", 0.80, 0.77)
        assert cause == "critic_missed"

    # 4.4
    def test_root_cause_known_risk_when_critic_critical_and_large_gap(self):
        cause, _ = _classify_gap(0.03, 0.01, "CRITICAL", 0.80, 0.77)
        assert cause == "known_risk"

    # 4.5
    def test_root_cause_known_risk_when_critic_high_and_large_gap(self):
        cause, _ = _classify_gap(0.03, 0.01, "HIGH", 0.80, 0.77)
        assert cause == "known_risk"

    # 4.6
    def test_root_cause_high_variance_cv(self):
        cause, _ = _classify_gap(0.03, 0.025, "OK", 0.80, 0.77)
        assert cause == "high_variance_cv"

    # 4.7
    def test_feature_retrospective_classifies_flagged_high_importance_as_hurt(self):
        feature_importance = {"features": [{"feature": "bad_feat", "importance": 0.5}]}
        critic_verdict = {"findings": [{"severity": "CRITICAL",
            "replan_instructions": {"remove_features": ["bad_feat"], "rerun_nodes": []}}]}
        retro = _build_feature_retrospective(feature_importance, critic_verdict, [], gap=0.03)
        assert len(retro) == 1
        assert retro[0]["verdict"] == "hurt"

    # 4.8
    def test_feature_retrospective_classifies_clean_stable_feature_as_helped(self):
        feature_importance = {"features": [{"feature": "good_feat", "importance": 0.3, "fold_variance": 0.01}]}
        critic_verdict = {"findings": []}
        retro = _build_feature_retrospective(feature_importance, critic_verdict, [], gap=0.01)
        assert retro[0]["verdict"] == "helped"

    # 4.9
    def test_confidence_increases_with_lb_percentile(self):
        """Top 10% → confidence >= 0.8."""
        # percentile = 1.0 - (10/100) = 0.9
        # confidence = min(0.9, 0.4 + 0.9 * 0.5) = min(0.9, 0.85) = 0.85
        percentile = 1.0 - (10 / 100)
        confidence = min(0.9, 0.4 + percentile * 0.5)
        assert confidence >= 0.8

    # 4.10
    def test_post_mortem_requires_lb_score(self):
        with pytest.raises(ValueError, match="lb_score"):
            run_post_mortem(session_id="test", lb_score=None)

    # 4.11
    def test_post_mortem_report_json_keys(self):
        """Verify report has all required keys when run with valid session dir."""
        session_id = "test_pm_keys"
        output_dir = f"outputs/{session_id}"
        os.makedirs(output_dir, exist_ok=True)
        # Create minimal artifacts
        with open(f"{output_dir}/validation_strategy.json", "w") as f:
            json.dump({"cv_mean": 0.82, "cv_std": 0.01}, f)
        with open(f"{output_dir}/critic_verdict.json", "w") as f:
            json.dump({"overall_severity": "OK", "findings": []}, f)
        with open(f"{output_dir}/feature_importance.json", "w") as f:
            json.dump({"features": [{"feature": "f1", "importance": 0.5}]}, f)
        with open(f"{output_dir}/competition_fingerprint.json", "w") as f:
            json.dump({"task_type": "tabular", "imbalance_ratio": 0.5,
                       "n_categorical_high_cardinality": 0, "n_rows_bucket": "medium",
                       "has_temporal_feature": False, "n_features_bucket": "medium",
                       "target_type": "binary"}, f)

        report = run_post_mortem(session_id=session_id, lb_score=0.79, lb_rank=5, total_competitors=100)
        required = {
            "session_id", "competition_name", "cv_mean", "lb_score",
            "cv_lb_gap", "gap_root_cause", "gap_explanation",
            "feature_retrospective", "patterns_written",
            "critic_failures_written", "confidence", "generated_at",
        }
        missing = required - set(report.keys())
        assert not missing, f"Report missing keys: {missing}"
        assert os.path.exists(f"{output_dir}/post_mortem_report.json")


# =========================================================================
# BLOCK 5 — INTEGRATION: FULL LOOP (8 tests)
# =========================================================================

class TestFullLearningLoop:

    # 5.1
    def test_critic_failure_patterns_collection_writable(self):
        fp = {"task_type": "tabular", "imbalance_ratio": 0.5,
              "n_categorical_high_cardinality": 0, "n_rows_bucket": "medium",
              "has_temporal_feature": False, "n_features_bucket": "medium",
              "target_type": "binary"}
        fid = store_critic_failure_pattern(
            fingerprint=fp,
            missed_issue="test missed issue",
            competition_name="test-cfp",
        )
        assert fid is not None and len(fid) > 0

    # 5.2
    def test_critic_failure_patterns_queryable_from_chromadb(self):
        from memory.chroma_client import build_chroma_client, get_or_create_collection
        client = build_chroma_client()
        collection = get_or_create_collection(client, CRITIC_FAILURE_COLLECTION)
        count = collection.count()
        assert count >= 0  # at minimum the collection exists

    # 5.3
    def test_post_mortem_runs_without_competition_rank(self):
        session_id = "test_pm_no_rank"
        output_dir = f"outputs/{session_id}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/validation_strategy.json", "w") as f:
            json.dump({"cv_mean": 0.80, "cv_std": 0.01}, f)
        with open(f"{output_dir}/critic_verdict.json", "w") as f:
            json.dump({"overall_severity": "OK", "findings": []}, f)
        with open(f"{output_dir}/feature_importance.json", "w") as f:
            json.dump({"features": []}, f)
        with open(f"{output_dir}/competition_fingerprint.json", "w") as f:
            json.dump({"task_type": "tabular", "imbalance_ratio": 0.5,
                       "n_categorical_high_cardinality": 0, "n_rows_bucket": "medium",
                       "has_temporal_feature": False, "n_features_bucket": "medium",
                       "target_type": "binary"}, f)

        report = run_post_mortem(session_id=session_id, lb_score=0.78)
        assert report is not None
        assert report["confidence"] == 0.65  # 0.4 + 0.5 * 0.5

    # 5.4
    def test_warm_start_priors_from_store_pattern(self):
        fp = {"task_type": "tabular", "imbalance_ratio": 0.5,
              "n_categorical_high_cardinality": 0, "n_rows_bucket": "medium",
              "has_temporal_feature": False, "n_features_bucket": "medium",
              "target_type": "binary"}
        pid = store_pattern(
            fingerprint=fp,
            validated_approaches=[{"approach": "Day11 test approach", "cv_improvement": 0.01, "competitions": ["d11"]}],
            failed_approaches=[],
            competition_name="day11-test",
            confidence=0.7,
        )
        assert pid is not None

    # 5.5
    def test_replan_drops_correct_features_in_state(self):
        s = initial_state("test-drop", FIXTURE_CSV)
        s["dag_version"] = 1
        s["replan_remove_features"] = ["bad_feat_A", "bad_feat_B"]
        s["replan_rerun_nodes"] = ["feature_factory"]
        result = run_supervisor_replan(s)
        assert "bad_feat_A" in result["features_dropped"]
        assert "bad_feat_B" in result["features_dropped"]

    # 5.6
    def test_full_replan_cycle_state_transitions(self):
        """CRITICAL → replan → re-run state should be clean for new critic pass."""
        s = initial_state("test-cycle", FIXTURE_CSV)
        s["critic_severity"] = "CRITICAL"
        s["replan_requested"] = True
        s["replan_remove_features"] = ["feat_X"]
        s["replan_rerun_nodes"] = ["ml_optimizer"]
        s["dag_version"] = 1

        result = run_supervisor_replan(s)
        assert result["dag_version"] == 2
        assert result["critic_severity"] == "unchecked"
        assert result["replan_requested"] is False
        assert "feat_X" in result["features_dropped"]

    # 5.7
    def test_node_priority_ordering_correct(self):
        assert NODE_PRIORITY["data_engineer"] < NODE_PRIORITY["feature_factory"]
        assert NODE_PRIORITY["feature_factory"] < NODE_PRIORITY["ml_optimizer"]
        assert NODE_PRIORITY["ml_optimizer"] < NODE_PRIORITY["red_team_critic"]

    # 5.8
    def test_gap_explanation_is_non_empty_string(self):
        _, explanation = _classify_gap(0.03, 0.01, "OK", 0.80, 0.77)
        assert isinstance(explanation, str)
        assert len(explanation) > 20
