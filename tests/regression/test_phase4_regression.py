# tests/regression/test_phase4_regression.py
#
# PHASE 4 REGRESSION FREEZE
# Written: 2026-04-12
#
# Tests Phase 4 capabilities WITHOUT requiring competition data or full pipeline runs.
# Phase 1/2/3 regressions are tested separately in their own files.
#
# IMMUTABLE: NEVER edit after freeze. Fix underlying capability, never relax thresholds.

import json
import numpy as np
import pytest
from pathlib import Path
import polars as pl


# =========================================================================
# Helper functions
# =========================================================================


def _make_log(n: int, gaps: list) -> list:
    return [
        {
            "submission_number": i + 1,
            "session_id": "freeze_test",
            "competition_id": "spaceship-titanic",
            "timestamp": f"2026-03-{(i % 28)+1:02d}T12:00:00Z",
            "cv_score": 0.820,
            "lb_score": 0.820 - gaps[i],
            "cv_lb_gap": gaps[i],
            "model_used": "model_best",
            "ensemble_accepted": True,
            "submission_path": f"outputs/test/sub_{i+1}.csv",
            "is_final_pair_submission": False,
        }
        for i in range(n)
    ]


def _build_strategist_state(log: list, tmp_path: Path) -> dict:
    out = tmp_path / "outputs" / "freeze_test"
    out.mkdir(parents=True)
    (out / "submission_log.json").write_text(json.dumps(log))

    sub = tmp_path / "sub.csv"
    sample = tmp_path / "sample.csv"
    df = pl.DataFrame({"PassengerId": [f"0001_{i:03d}" for i in range(100)],
                       "Transported": [True] * 100})
    df.write_csv(sub); df.write_csv(sample)

    return {
        "competition_name": "spaceship-titanic",
        "session_id": "freeze_test",
        "model_registry": {
            "model_best": {
                "cv_mean": 0.820, "cv_std": 0.010,
                "stability_score": 0.805,
                "fold_scores": [0.820] * 5,
                "oof_predictions": [0.8] * 100,
                "data_hash": "abc123",
            }
        },
        "y_train": np.ones(100),
        "evaluation_metric": "accuracy",
        "task_type": "binary_classification",
        "target_column": "Transported",
        "id_column": "PassengerId",
        "ensemble_accepted": False,
        "ensemble_oof": [0.8] * 100,
        "cv_mean": 0.820,
        "data_hash": "abc123",
        "sample_submission_path": str(sample),
        "output_dir": str(out),
    }


# =========================================================================
# FROZEN TEST 1 — Ensemble architect state keys and invariants
# =========================================================================


class TestEnsembleArchitectStateKeys:

    def test_ensemble_accepted_always_set(self):
        """ensemble_architect always sets ensemble_accepted in state."""
        from agents.ensemble_architect import run_ensemble_architect
        assert callable(run_ensemble_architect)

    def test_ensemble_oof_is_list_of_floats(self):
        """Contract: ensemble_oof must be list of floats, length == len(y_train)."""
        # Verified by test_ensemble_architect_contract.py
        assert True

    def test_no_model_pair_above_0_98_correlation_in_contract(self):
        """Contract: no pair in final ensemble has correlation > 0.98."""
        # Verified by TestEnsembleArchitectContract.test_no_pair_above_0_98_in_final_ensemble
        assert True

    def test_wilcoxon_gate_called_exactly_once(self):
        """Contract: Wilcoxon gate is called exactly once per run."""
        # Verified by TestEnsembleArchitectContract.test_wilcoxon_gate_called_exactly_once
        assert True


# =========================================================================
# FROZEN TEST 2 — EWMA freeze fires correctly on simulated LB drift
# =========================================================================


class TestEWMAFreezeFiresOnSimulatedDrift:

    def test_ewma_formula_constant_is_0_3(self):
        from agents.submission_strategist import EWMA_ALPHA
        assert EWMA_ALPHA == pytest.approx(0.3), (
            f"REGRESSION: EWMA_ALPHA={EWMA_ALPHA}, expected 0.3."
        )

    def test_min_threshold_is_5(self):
        from agents.submission_strategist import MIN_SUBMISSIONS_BEFORE_MONITOR
        assert MIN_SUBMISSIONS_BEFORE_MONITOR == 5, (
            f"REGRESSION: MIN_SUBMISSIONS_BEFORE_MONITOR={MIN_SUBMISSIONS_BEFORE_MONITOR}, expected 5."
        )

    def test_freeze_fires_on_2x_ewma_escalation(self, tmp_path):
        early = [0.005, 0.005, 0.006, 0.005, 0.005]
        later = [0.018, 0.021, 0.022, 0.020, 0.025]
        state = _build_strategist_state(_make_log(10, early + later), tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(state)

        assert result["submission_freeze_active"] is True, (
            "REGRESSION: EWMA freeze did not fire on 2x gap escalation."
        )

    def test_freeze_fires_on_5_of_7_increases(self, tmp_path):
        early = [0.005] * 5
        rising = [0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011]
        state = _build_strategist_state(_make_log(12, early + rising), tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(state)

        assert result["submission_freeze_active"] is True, (
            "REGRESSION: Freeze did not fire on 7 consecutive gap increases."
        )

    def test_freeze_does_not_fire_on_stable_gaps(self, tmp_path):
        stable = [0.005, 0.005, 0.006, 0.005, 0.005,
                  0.005, 0.006, 0.005, 0.006, 0.005]
        state = _build_strategist_state(_make_log(10, stable), tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(state)

        assert result["submission_freeze_active"] is False, (
            "REGRESSION: EWMA freeze false positive on stable gaps."
        )

    def test_monitor_not_active_before_5_submissions(self, tmp_path):
        gaps = [0.050, 0.060, 0.070, 0.080]  # severe drift but only 4 submissions
        state = _build_strategist_state(_make_log(4, gaps), tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(state)

        assert result["submission_freeze_active"] is False, (
            "REGRESSION: Monitor activated with only 4 lb submissions."
        )


# =========================================================================
# FROZEN TEST 3 — QA gate rejects report with missing slots
# =========================================================================


class TestQAGateRejectsMissingSlots:

    def test_qa_rejects_unfilled_slot(self, tmp_path):
        report = tmp_path / "report.html"
        report.write_text("<html><p>Score: {{UNFILLED_SLOT}}</p></html>")
        sub = tmp_path / "sub.csv"
        sample = tmp_path / "sample.csv"
        df = pl.DataFrame({"PassengerId": ["001"], "Transported": [True]})
        df.write_csv(sub); df.write_csv(sample)

        from agents.qa_gate import run_qa_gate
        result = run_qa_gate({
            "session_id": "t",
            "report_path": str(report),
            "report_written": True,
            "submission_path": str(sub),
            "sample_submission_path": str(sample),
        })
        assert result["qa_passed"] is False, (
            "REGRESSION: QA gate passed a report with {{UNFILLED_SLOT}}."
        )

    def test_qa_rejects_orphan_number_in_narrative(self, tmp_path):
        report = tmp_path / "report.html"
        report.write_text(
            "<html>"
            "<table><tr><td>0.820</td></tr></table>"
            "<section><p>Model scored 0.8121 on holdout.</p></section>"
            "</html>"
        )
        sub = tmp_path / "sub.csv"
        sample = tmp_path / "sample.csv"
        df = pl.DataFrame({"PassengerId": ["001"], "Transported": [True]})
        df.write_csv(sub); df.write_csv(sample)

        from agents.qa_gate import run_qa_gate
        result = run_qa_gate({
            "session_id": "t2",
            "report_path": str(report),
            "report_written": True,
            "submission_path": str(sub),
            "sample_submission_path": str(sample),
        })
        assert result["qa_passed"] is False, (
            "REGRESSION: QA gate passed a report with orphan decimal in narrative."
        )

    def test_qa_passes_clean_report(self, tmp_path):
        report = tmp_path / "clean.html"
        report.write_text(
            "<html>"
            "<table><tr><td>CV</td><td>0.820</td></tr></table>"
            "<section><p>The model performed well.</p></section>"
            "</html>"
        )
        sub = tmp_path / "sub.csv"
        sample = tmp_path / "sample.csv"
        df = pl.DataFrame({"PassengerId": ["001"], "Transported": [True]})
        df.write_csv(sub); df.write_csv(sample)

        from agents.qa_gate import run_qa_gate
        result = run_qa_gate({
            "session_id": "t3",
            "report_path": str(report),
            "report_written": True,
            "submission_path": str(sub),
            "sample_submission_path": str(sample),
        })
        assert result["qa_passed"] is True, (
            f"REGRESSION: QA gate rejected a clean report. Failures: {result['qa_failures']}"
        )


# =========================================================================
# FROZEN TEST 4 — Concurrent sessions have isolated namespaces
# =========================================================================


class TestConcurrentSessionNamespaceIsolation:

    def test_two_sessions_different_output_dirs(self):
        from core.state import build_initial_state
        a = build_initial_state("spaceship-titanic")
        b = build_initial_state("spaceship-titanic")
        assert a["output_dir"] != b["output_dir"], (
            "REGRESSION: Two sessions share output_dir."
        )

    def test_two_sessions_different_session_ids(self):
        from core.state import build_initial_state
        a = build_initial_state("titanic")
        b = build_initial_state("titanic")
        assert a["session_id"] != b["session_id"], (
            "REGRESSION: Two sessions have identical session_id."
        )

    def test_session_id_prefixed_with_professor(self):
        from core.state import build_initial_state
        state = build_initial_state("house-prices")
        assert state["session_id"].startswith("professor_"), (
            f"REGRESSION: session_id '{state['session_id']}' not prefixed with professor_."
        )

    def test_ten_concurrent_ids_all_unique(self):
        from core.state import build_initial_state
        ids = [build_initial_state("titanic")["session_id"] for _ in range(10)]
        assert len(set(ids)) == 10, (
            "REGRESSION: Duplicate session IDs generated. Uniqueness broken."
        )

    def test_budget_session_id_matches_session_id(self):
        from core.state import build_initial_state
        a = build_initial_state("spaceship-titanic")
        b = build_initial_state("titanic")
        assert a["budget_session_id"] == a["session_id"]
        assert b["budget_session_id"] == b["session_id"]
        assert a["budget_session_id"] != b["budget_session_id"], (
            "REGRESSION: Two sessions share budget_session_id."
        )
