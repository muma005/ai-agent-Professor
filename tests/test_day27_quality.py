# tests/test_day27_quality.py
# Day 27 quality tests — post_mortem_agent correctness.
#
# CONTRACT: agents/post_mortem_agent.py
#   INPUTS:  session outputs directory with JSON files
#   OUTPUTS: competition_memory.json + ChromaDB writes
#   SECTIONS: solution_autopsy, strategy_evaluation, memory_writes, critic_calibration

import pytest
import json
import numpy as np
from pathlib import Path


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def tmp_outputs(tmp_path, monkeypatch):
    """Changes cwd so outputs/{session_id}/ resolves correctly."""
    (tmp_path / "outputs").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def session_dir_with_all_files(tmp_outputs):
    """
    Complete session directory with all expected files.
    Simulates a successful Professor run on Spaceship Titanic.
    """
    session_id = "test_post_mortem_session"
    session_dir = tmp_outputs / "outputs" / session_id
    session_dir.mkdir(parents=True)

    # metrics.json
    (session_dir / "metrics.json").write_text(json.dumps({
        "cv_mean": 0.820,
        "cv_std": 0.010,
        "winning_model_type": "lgbm",
        "feature_order": ["CryoSleep", "Cabin_deck", "total_spend",
                          "GroupSize", "leaked_target"],
        "n_features_final": 5,
    }))

    # null_importance_result.json
    (session_dir / "null_importance_result.json").write_text(json.dumps({
        "dropped_stage1": ["noise_feature_1", "noise_feature_2"],
        "dropped_stage2": ["marginal_feature_1"],
        "survivors": ["CryoSleep", "Cabin_deck", "total_spend",
                      "GroupSize", "leaked_target"],
    }))

    # critic_verdict.json — CRITICAL for leaked_target
    (session_dir / "critic_verdict.json").write_text(json.dumps({
        "overall_severity": "CRITICAL",
        "findings": [
            {
                "vector": "shuffled_target",
                "severity": "CRITICAL",
                "feature_flagged": "leaked_target",
                "evidence": "Correlation with target > 0.95 after shuffle",
            }
        ],
        "vectors_checked": [
            "shuffled_target", "id_only_model", "adversarial_classifier",
            "preprocessing_audit", "pr_curve_imbalance", "temporal_leakage",
            "robustness", "historical_failures", "prediction_audit",
            "permutation_importance",
        ],
    }))

    # feature_importance.json
    (session_dir / "feature_importance.json").write_text(json.dumps({
        "CryoSleep": 0.312,
        "Cabin_deck": 0.187,
        "total_spend": 0.145,
        "GroupSize": 0.098,
        "leaked_target": 0.002,
    }))

    # competition_brief.json
    (session_dir / "competition_brief.json").write_text(json.dumps({
        "competition_name": "spaceship-titanic",
        "domain": "spaceship_titanic",
        "fingerprint": {
            "task_type": "binary_classification",
            "n_rows_bucket": "small",
        },
    }))

    # ensemble_selection.json
    (session_dir / "ensemble_selection.json").write_text(json.dumps({
        "ensemble_accepted": True,
        "ensemble_holdout_score": 0.817,
        "selected_models": ["lgbm_001", "xgb_002"],
    }))

    # submission_log.json
    (session_dir / "submission_log.json").write_text(json.dumps([
        {
            "submission_number": 1, "cv_score": 0.820,
            "lb_score": None, "cv_lb_gap": None,
            "model_used": "lgbm_001",
        }
    ]))

    return session_dir


@pytest.fixture
def session_dir_minimal(tmp_outputs):
    """Session with only metrics.json — all optional files missing."""
    session_id = "minimal_session"
    session_dir = tmp_outputs / "outputs" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "metrics.json").write_text(json.dumps({
        "cv_mean": 0.810, "cv_std": 0.012,
        "winning_model_type": "lgbm",
        "feature_order": ["f1", "f2", "f3"],
        "n_features_final": 3,
    }))
    return session_dir


@pytest.fixture
def session_dir_with_ok_critic(session_dir_with_all_files):
    """Same as full session but critic returned OK."""
    (session_dir_with_all_files / "critic_verdict.json").write_text(json.dumps({
        "overall_severity": "OK",
        "findings": [],
        "vectors_checked": ["shuffled_target"],
    }))
    return session_dir_with_all_files


@pytest.fixture
def session_dir_with_critical_verdict(session_dir_with_all_files):
    """Session dir explicitly tagged with CRITICAL critic verdict."""
    return session_dir_with_all_files  # already has CRITICAL verdict


# =========================================================================
# BLOCK 1 — Solution Autopsy (8 tests)
# =========================================================================


class TestSolutionAutopsy:

    def test_autopsy_classifies_high_importance_feature_as_contributed(
        self, session_dir_with_all_files
    ):
        """Feature with importance > 0.01 and not flagged by critic → 'contributed'."""
        from agents.post_mortem_agent import _build_solution_autopsy
        result = _build_solution_autopsy(session_dir_with_all_files, lb_score=0.807)
        audit = {item["feature"]: item for item in result["feature_audit"]}

        # CryoSleep is in feature_order, high importance, not flagged by critic
        assert "CryoSleep" in audit
        assert audit["CryoSleep"]["classification"] == "contributed", (
            f"CryoSleep classified as '{audit['CryoSleep']['classification']}', "
            "expected 'contributed'."
        )

    def test_autopsy_classifies_critic_flagged_feature_as_noise(
        self, session_dir_with_all_files
    ):
        """Feature flagged by critic → 'noise' regardless of importance."""
        from agents.post_mortem_agent import _build_solution_autopsy
        result = _build_solution_autopsy(session_dir_with_all_files, lb_score=0.807)
        audit = {item["feature"]: item for item in result["feature_audit"]}

        # leaked_target is in critic findings
        assert "leaked_target" in audit
        assert audit["leaked_target"]["classification"] == "noise", (
            f"Critic-flagged feature classified as '{audit['leaked_target']['classification']}', "
            "expected 'noise'."
        )

    def test_autopsy_classifies_dropped_stage1_as_correctly_pruned(
        self, session_dir_with_all_files
    ):
        from agents.post_mortem_agent import _build_solution_autopsy
        result = _build_solution_autopsy(session_dir_with_all_files, lb_score=0.807)
        audit = {item["feature"]: item for item in result["feature_audit"]}

        assert "noise_feature_1" in audit
        assert audit["noise_feature_1"]["status"] == "dropped_stage1"
        assert audit["noise_feature_1"]["classification"] == "correctly_pruned"

    def test_autopsy_counts_match_file_contents(self, session_dir_with_all_files):
        from agents.post_mortem_agent import _build_solution_autopsy
        import json
        metrics = json.loads((session_dir_with_all_files / "metrics.json").read_text())
        null = json.loads((session_dir_with_all_files / "null_importance_result.json").read_text())

        result = _build_solution_autopsy(session_dir_with_all_files, lb_score=0.807)

        assert result["total_features_trained"] == len(metrics["feature_order"])
        assert result["total_dropped_stage1"] == len(null["dropped_stage1"])
        assert result["total_dropped_stage2"] == len(null["dropped_stage2"])

    def test_autopsy_handles_missing_null_importance_file(self, session_dir_minimal):
        """Session without null_importance_result.json must not crash."""
        from agents.post_mortem_agent import _build_solution_autopsy
        result = _build_solution_autopsy(session_dir_minimal, lb_score=0.807)
        assert "feature_audit" in result
        assert result["total_dropped_stage1"] == 0
        assert result["total_dropped_stage2"] == 0

    def test_autopsy_handles_missing_critic_verdict(self, session_dir_minimal):
        """Session without critic_verdict.json must not crash."""
        from agents.post_mortem_agent import _build_solution_autopsy
        result = _build_solution_autopsy(session_dir_minimal, lb_score=0.807)
        assert "feature_audit" in result
        assert result["features_flagged_by_critic"] == 0

    def test_all_survived_features_in_audit(self, session_dir_with_all_files):
        """Every feature in feature_order must appear in feature_audit."""
        import json
        from agents.post_mortem_agent import _build_solution_autopsy
        metrics = json.loads((session_dir_with_all_files / "metrics.json").read_text())
        result = _build_solution_autopsy(session_dir_with_all_files, lb_score=0.807)
        audited = {item["feature"] for item in result["feature_audit"]}
        for feat in metrics["feature_order"]:
            assert feat in audited, f"Feature '{feat}' in feature_order not in audit."

    def test_feature_audit_has_all_required_keys(self, session_dir_with_all_files):
        from agents.post_mortem_agent import _build_solution_autopsy
        result = _build_solution_autopsy(session_dir_with_all_files, lb_score=0.807)
        required = {"feature", "status", "importance", "flagged_by_critic", "classification"}
        for item in result["feature_audit"]:
            missing = required - set(item.keys())
            assert not missing, f"Feature audit item missing keys: {missing}"


# =========================================================================
# BLOCK 2 — Strategy Evaluation (6 tests)
# =========================================================================


class TestStrategyEvaluation:

    def test_cv_lb_gap_computed_correctly(self, session_dir_with_all_files):
        from agents.post_mortem_agent import _build_strategy_evaluation
        result = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.807,
            lb_rank=None, total_teams=None
        )
        expected_gap = abs(result["cv_mean"] - 0.807)
        assert abs(result["cv_lb_gap"] - expected_gap) < 1e-6

    def test_percentile_computed_when_rank_given(self, session_dir_with_all_files):
        from agents.post_mortem_agent import _build_strategy_evaluation
        result = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.807,
            lb_rank=150, total_teams=1000
        )
        expected = round(100 * (1 - 150 / 1000), 2)
        assert result["percentile"] == expected, (
            f"Expected percentile {expected}, got {result['percentile']}."
        )

    def test_percentile_none_when_rank_not_given(self, session_dir_with_all_files):
        from agents.post_mortem_agent import _build_strategy_evaluation
        result = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.807,
            lb_rank=None, total_teams=None
        )
        assert result["percentile"] is None

    def test_gap_root_cause_critic_missed_on_large_gap_ok_critic(
        self, session_dir_with_ok_critic
    ):
        """Large gap + critic returned OK → gap_root_cause = 'critic_missed'."""
        from agents.post_mortem_agent import _build_strategy_evaluation
        result = _build_strategy_evaluation(
            session_dir_with_ok_critic, lb_score=0.790,  # large gap from cv=0.820
            lb_rank=None, total_teams=None
        )
        assert result["gap_root_cause"] == "critic_missed", (
            f"Expected 'critic_missed', got '{result['gap_root_cause']}'."
        )

    def test_gap_root_cause_acceptable_on_small_gap(self, session_dir_with_all_files):
        from agents.post_mortem_agent import _build_strategy_evaluation
        # cv_mean in metrics.json is 0.820, lb=0.819 → gap=0.001
        result = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.819,
            lb_rank=None, total_teams=None
        )
        assert result["gap_root_cause"] == "acceptable"

    def test_all_required_keys_present(self, session_dir_with_all_files):
        from agents.post_mortem_agent import _build_strategy_evaluation
        result = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.807,
            lb_rank=100, total_teams=500
        )
        required = {
            "cv_mean", "cv_std", "lb_score", "cv_lb_gap", "gap_root_cause",
            "lb_rank", "total_teams", "percentile", "ensemble_accepted",
            "ensemble_holdout_score", "n_models_in_ensemble", "winning_model_type",
        }
        missing = required - set(result.keys())
        assert not missing, f"Strategy evaluation missing keys: {missing}"


# =========================================================================
# BLOCK 3 — Structured Memory Writes (6 tests)
# =========================================================================


class TestStructuredMemoryWrites:

    def test_high_importance_features_generate_memory_writes(
        self, session_dir_with_all_files
    ):
        from agents.post_mortem_agent import (
            _build_solution_autopsy, _build_strategy_evaluation, _build_memory_writes
        )
        autopsy = _build_solution_autopsy(session_dir_with_all_files, lb_score=0.807)
        strategy = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.807, lb_rank=None, total_teams=None
        )
        writes = _build_memory_writes(session_dir_with_all_files, strategy, autopsy, {})

        feature_writes = [w for w in writes if w["finding_type"] == "feature"]
        assert len(feature_writes) >= 1, "No feature memory writes generated."

    def test_pitfall_writes_generated_for_critic_flagged_features(
        self, session_dir_with_all_files
    ):
        from agents.post_mortem_agent import (
            _build_solution_autopsy, _build_strategy_evaluation, _build_memory_writes
        )
        autopsy = _build_solution_autopsy(session_dir_with_all_files, lb_score=0.807)
        strategy = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.807, lb_rank=None, total_teams=None
        )
        writes = _build_memory_writes(session_dir_with_all_files, strategy, autopsy, {})

        pitfalls = [w for w in writes if w["finding_type"] == "pitfall"]
        assert len(pitfalls) >= 1, "No pitfall writes generated for critic-flagged features."

    def test_every_write_has_required_fields(self, session_dir_with_all_files):
        from agents.post_mortem_agent import (
            _build_solution_autopsy, _build_strategy_evaluation, _build_memory_writes
        )
        autopsy = _build_solution_autopsy(session_dir_with_all_files, lb_score=0.807)
        strategy = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.807, lb_rank=None, total_teams=None
        )
        writes = _build_memory_writes(session_dir_with_all_files, strategy, autopsy, {})

        required = {"domain", "feature", "cv_delta", "private_lb_delta",
                    "validated", "reusable", "confidence", "finding_type"}
        for write in writes:
            missing = required - set(write.keys())
            assert not missing, f"Memory write missing fields: {missing}. Write: {write}"

    def test_validated_field_false_when_large_gap(self, session_dir_with_all_files):
        """When cv_lb_gap > 0.010, findings should not be marked validated=True."""
        from agents.post_mortem_agent import (
            _build_solution_autopsy, _build_strategy_evaluation, _build_memory_writes
        )
        autopsy = _build_solution_autopsy(session_dir_with_all_files, lb_score=0.790)
        strategy = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.790,  # gap=0.030
            lb_rank=None, total_teams=None
        )
        writes = _build_memory_writes(session_dir_with_all_files, strategy, autopsy, {})

        # When gap is large, feature findings should not be validated
        feature_writes = [w for w in writes if w["finding_type"] == "feature"]
        for w in feature_writes:
            assert w["validated"] is False or w["validated"] == False, (
                f"Feature '{w['feature']}' marked validated=True despite large cv_lb_gap."
            )

    def test_confidence_higher_for_small_gap_than_large_gap(
        self, session_dir_with_all_files
    ):
        from agents.post_mortem_agent import _compute_confidence, _build_strategy_evaluation

        strategy_small_gap = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.819, lb_rank=50, total_teams=1000
        )
        strategy_large_gap = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.790, lb_rank=400, total_teams=1000
        )

        conf_small = _compute_confidence(strategy_small_gap)
        conf_large = _compute_confidence(strategy_large_gap)

        assert conf_small > conf_large, (
            f"Confidence for small gap ({conf_small}) should exceed "
            f"confidence for large gap ({conf_large})."
        )

    def test_confidence_range_0_to_1(self, session_dir_with_all_files):
        from agents.post_mortem_agent import _compute_confidence, _build_strategy_evaluation
        strategy = _build_strategy_evaluation(
            session_dir_with_all_files, lb_score=0.807, lb_rank=100, total_teams=1000
        )
        conf = _compute_confidence(strategy)
        assert 0.0 <= conf <= 1.0


# =========================================================================
# BLOCK 4 — Critic Calibration (5 tests)
# =========================================================================


class TestCriticCalibration:

    def test_true_positive_when_critic_fired_and_gap_large(
        self, session_dir_with_critical_verdict
    ):
        """Critic fired CRITICAL + gap > 0.010 → true_positive."""
        from agents.post_mortem_agent import _build_critic_calibration
        result = _build_critic_calibration(
            session_dir_with_critical_verdict,
            lb_score=0.790, cv_mean=0.820  # gap = 0.030
        )
        assert result["calibration_verdict"] == "true_positive", (
            f"Expected true_positive, got '{result['calibration_verdict']}'."
        )
        assert result["threshold_recommendation"] == "thresholds_appropriate"

    def test_false_positive_when_critic_fired_but_gap_small(
        self, session_dir_with_critical_verdict
    ):
        """Critic fired CRITICAL + gap <= 0.005 → false_positive."""
        from agents.post_mortem_agent import _build_critic_calibration
        result = _build_critic_calibration(
            session_dir_with_critical_verdict,
            lb_score=0.818, cv_mean=0.820  # gap = 0.002
        )
        assert result["calibration_verdict"] == "false_positive"
        assert result["threshold_recommendation"] == "consider_raising_thresholds"

    def test_false_negative_when_critic_ok_and_gap_large(
        self, session_dir_with_ok_critic
    ):
        """Critic returned OK + gap > 0.010 → false_negative."""
        from agents.post_mortem_agent import _build_critic_calibration
        result = _build_critic_calibration(
            session_dir_with_ok_critic,
            lb_score=0.790, cv_mean=0.820  # gap = 0.030
        )
        assert result["calibration_verdict"] == "false_negative"
        assert result["threshold_recommendation"] == "consider_lowering_thresholds"

    def test_true_negative_when_critic_ok_and_gap_small(
        self, session_dir_with_ok_critic
    ):
        """Critic returned OK + gap <= 0.005 → true_negative."""
        from agents.post_mortem_agent import _build_critic_calibration
        result = _build_critic_calibration(
            session_dir_with_ok_critic,
            lb_score=0.818, cv_mean=0.820  # gap = 0.002
        )
        assert result["calibration_verdict"] == "true_negative"

    def test_all_required_keys_present(self, session_dir_with_all_files):
        from agents.post_mortem_agent import _build_critic_calibration
        result = _build_critic_calibration(
            session_dir_with_all_files,
            lb_score=0.807, cv_mean=0.820
        )
        required = {
            "overall_severity", "n_critical_findings", "n_high_findings",
            "cv_lb_gap", "calibration_verdict", "threshold_recommendation",
            "threshold_note", "vectors_checked",
        }
        missing = required - set(result.keys())
        assert not missing, f"Critic calibration missing keys: {missing}"


# =========================================================================
# BLOCK 5 — Full post_mortem_agent integration (5 tests)
# =========================================================================


class TestPostMortemIntegration:

    def test_run_produces_competition_memory_json(self, session_dir_with_all_files):
        from agents.post_mortem_agent import run_post_mortem_agent
        run_post_mortem_agent(
            session_id=session_dir_with_all_files.name,
            lb_score=0.807, lb_rank=150, total_teams=1000,
        )
        output = session_dir_with_all_files / "competition_memory.json"
        assert output.exists(), "competition_memory.json not written."

    def test_competition_memory_json_has_four_sections(self, session_dir_with_all_files):
        from agents.post_mortem_agent import run_post_mortem_agent
        report = run_post_mortem_agent(
            session_id=session_dir_with_all_files.name,
            lb_score=0.807, lb_rank=None, total_teams=None,
        )
        for section in ["solution_autopsy", "strategy_evaluation",
                        "memory_writes", "critic_calibration"]:
            assert section in report, f"Section '{section}' missing from report."

    def test_run_returns_error_dict_on_missing_session(self, tmp_path):
        """Missing session directory must return error dict, not raise."""
        from agents.post_mortem_agent import run_post_mortem_agent
        result = run_post_mortem_agent(
            session_id="nonexistent_session_xyz",
            lb_score=0.807,
        )
        assert "error" in result, (
            "run_post_mortem_agent should return error dict for missing session."
        )

    def test_run_returns_error_dict_on_missing_metrics(self, tmp_path, monkeypatch):
        """Session dir exists but metrics.json missing → error dict."""
        session_dir = tmp_path / "outputs" / "empty_session"
        session_dir.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        from agents.post_mortem_agent import run_post_mortem_agent
        result = run_post_mortem_agent(session_id="empty_session", lb_score=0.807)
        assert "error" in result

    def test_n_memory_writes_matches_writes_list(self, session_dir_with_all_files):
        from agents.post_mortem_agent import run_post_mortem_agent
        report = run_post_mortem_agent(
            session_id=session_dir_with_all_files.name,
            lb_score=0.807, lb_rank=None, total_teams=None,
        )
        assert report["n_memory_writes"] == len(report["memory_writes"])
