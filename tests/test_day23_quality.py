# tests/test_day23_quality.py
# Day 23 quality tests — algorithmic correctness for all three agents.

import json
import pytest
import numpy as np
import polars as pl
from pathlib import Path


# ── Fixtures (reuse from contract tests) ──────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def y_train(rng):
    return rng.integers(0, 2, 500).astype(np.float32)


@pytest.fixture
def oof_a(y_train, rng):
    return np.clip(y_train + rng.normal(0, 0.25, len(y_train)), 0, 1).tolist()


@pytest.fixture
def oof_b(oof_a, rng):
    return np.clip(rng.normal(0.5, 0.35, len(oof_a)), 0, 1).tolist()


@pytest.fixture
def oof_c(oof_a, rng):
    return np.clip(rng.normal(0.45, 0.4, len(oof_a)), 0, 1).tolist()


@pytest.fixture
def base_registry(oof_a, oof_b, oof_c):
    return {
        "model_best": {
            "cv_mean": 0.820, "cv_std": 0.010,
            "stability_score": 0.805,
            "fold_scores": [0.820] * 5,
            "oof_predictions": oof_a,
            "data_hash": "abc123",
        },
        "model_mid": {
            "cv_mean": 0.810, "cv_std": 0.012,
            "stability_score": 0.792,
            "fold_scores": [0.810] * 5,
            "oof_predictions": oof_b,
            "data_hash": "abc123",
        },
        "model_low": {
            "cv_mean": 0.800, "cv_std": 0.015,
            "stability_score": 0.778,
            "fold_scores": [0.800] * 5,
            "oof_predictions": oof_c,
            "data_hash": "abc123",
        },
    }


@pytest.fixture
def sample_submission(tmp_path, y_train):
    ids   = [f"0001_{i:03d}" for i in range(len(y_train))]
    preds = [True] * len(y_train)
    df    = pl.DataFrame({"PassengerId": ids, "Transported": preds})
    path  = tmp_path / "sample_submission.csv"
    df.write_csv(path)
    return path


@pytest.fixture
def base_state(base_registry, y_train, sample_submission, tmp_path):
    session_id = "test_session_23"
    out_dir    = tmp_path / "outputs" / session_id
    out_dir.mkdir(parents=True)
    return {
        "competition_name":        "spaceship-titanic",
        "session_id":              session_id,
        "model_registry":          base_registry,
        "y_train":                 y_train,
        "evaluation_metric":       "accuracy",
        "task_type":               "binary_classification",
        "target_column":           "Transported",
        "id_column":               "PassengerId",
        "ensemble_accepted":       True,
        "ensemble_oof":            base_registry["model_best"]["oof_predictions"],
        "cv_mean":                 0.820,
        "data_hash":               "abc123",
        "sample_submission_path":  str(sample_submission),
        "output_dir":              str(out_dir),
    }


# ── BLOCK 1 — EWMA computation correctness (5 tests) ──────────────

class TestEWMAComputation:

    def test_ewma_formula_correct(self):
        """EWMA formula: ewma = alpha * gap + (1-alpha) * ewma_prev."""
        from agents.submission_strategist import compute_ewma_gap, EWMA_ALPHA
        gaps = [0.005, 0.010, 0.008]
        # Manual calculation
        ewma = gaps[0]
        for g in gaps[1:]:
            ewma = EWMA_ALPHA * g + (1 - EWMA_ALPHA) * ewma
        result = compute_ewma_gap(gaps)
        assert abs(result - ewma) < 1e-9, (
            f"EWMA formula wrong. Expected {ewma:.8f}, got {result:.8f}."
        )

    def test_single_value_ewma_equals_that_value(self):
        from agents.submission_strategist import compute_ewma_gap
        assert compute_ewma_gap([0.007]) == pytest.approx(0.007)

    def test_ewma_alpha_is_0_3(self):
        from agents.submission_strategist import EWMA_ALPHA
        assert EWMA_ALPHA == pytest.approx(0.3), (
            f"EWMA_ALPHA is {EWMA_ALPHA}, expected 0.3."
        )

    def test_increasing_gaps_raise_ewma(self):
        from agents.submission_strategist import compute_ewma_gap
        stable    = compute_ewma_gap([0.005, 0.005, 0.005, 0.005, 0.005])
        escalating = compute_ewma_gap([0.005, 0.010, 0.015, 0.020, 0.025])
        assert escalating > stable

    def test_ewma_more_responsive_to_recent_gaps(self):
        """Recent high gap must raise EWMA more than early high gap."""
        from agents.submission_strategist import compute_ewma_gap
        early_spike  = compute_ewma_gap([0.050, 0.005, 0.005, 0.005, 0.005])
        recent_spike = compute_ewma_gap([0.005, 0.005, 0.005, 0.005, 0.050])
        assert recent_spike > early_spike


# ── BLOCK 2 — Submission pair selection (4 tests) ─────────────────

class TestSubmissionPairSelection:

    def test_submission_a_is_highest_stability_not_highest_mean(
        self, base_state, rng
    ):
        """
        stability_score = mean - 1.5*std. A model with lower mean but
        lower std can have higher stability_score. A must use stability,
        not raw mean.
        """
        oof_stable = np.clip(rng.normal(0.81, 0.1, 500), 0, 1).tolist()
        oof_peaky  = np.clip(rng.normal(0.82, 0.3, 500), 0, 1).tolist()

        registry = {
            "model_stable": {
                "cv_mean": 0.810, "cv_std": 0.001, "stability_score": 0.808,
                "fold_scores": [0.810] * 5, "oof_predictions": oof_stable,
                "data_hash": "abc123",
            },
            "model_peaky": {
                "cv_mean": 0.820, "cv_std": 0.020, "stability_score": 0.790,
                "fold_scores": [0.820] * 5, "oof_predictions": oof_peaky,
                "data_hash": "abc123",
            },
        }
        state = {**base_state, "model_registry": registry,
                 "ensemble_accepted": False}
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(state)
        assert result["submission_a_model"] == "model_stable", (
            "Submission A selected highest mean CV, not highest stability_score. "
            "stability_score = mean - 1.5*std must be used."
        )

    def test_submission_b_is_not_same_as_a(self, base_state):
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert result["submission_a_model"] != result["submission_b_model"]

    def test_submission_b_different_from_a_in_content(self, base_state):
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        a_content = Path(result["submission_a_path"]).read_text()
        b_content = Path(result["submission_b_path"]).read_text()
        assert a_content != b_content, (
            "submission_a.csv and submission_b.csv are identical. "
            "They must represent different models."
        )

    def test_single_model_registry_uses_same_model_for_both(self, base_state, oof_a):
        registry = {
            "only_model": {
                "cv_mean": 0.820, "cv_std": 0.010, "stability_score": 0.805,
                "fold_scores": [0.820] * 5, "oof_predictions": oof_a,
                "data_hash": "abc123",
            }
        }
        state = {**base_state, "model_registry": registry, "ensemble_accepted": False}
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(state)
        assert result["submission_a_model"] == "only_model"
        assert result["submission_b_model"] == "only_model"


# ── BLOCK 3 — Publisher template correctness (5 tests) ─────────────

class TestPublisherTemplate:

    def test_all_numeric_slots_filled(self, base_state):
        """After fill_numeric_slots, no {{SLOT}} patterns remain for numeric slots."""
        import re
        from agents.publisher import fill_numeric_slots, TEMPLATE, NUMERIC_SLOTS
        metrics = {"cv_mean": 0.820, "cv_std": 0.011}
        filled  = fill_numeric_slots(TEMPLATE, metrics, base_state)
        remaining = re.findall(r"\{\{([A-Z_]+)\}\}", filled)
        numeric_remaining = [s for s in remaining if s in NUMERIC_SLOTS]
        assert not numeric_remaining, (
            f"Numeric slots still unfilled after fill_numeric_slots: {numeric_remaining}"
        )

    def test_numeric_slots_use_values_from_metrics_not_hardcoded(self, base_state):
        from agents.publisher import fill_numeric_slots, TEMPLATE
        metrics_a = {"cv_mean": 0.820, "cv_std": 0.010}
        metrics_b = {"cv_mean": 0.750, "cv_std": 0.030}
        filled_a  = fill_numeric_slots(TEMPLATE, metrics_a, base_state)
        filled_b  = fill_numeric_slots(TEMPLATE, metrics_b, base_state)
        assert "0.82000" in filled_a
        assert "0.75000" in filled_b

    def test_missing_metric_injected_as_na(self, base_state):
        from agents.publisher import fill_numeric_slots, TEMPLATE
        metrics = {}  # all missing
        filled  = fill_numeric_slots(TEMPLATE, metrics, base_state)
        assert "N/A" in filled

    def test_report_written_to_disk(self, base_state):
        from agents.publisher import run_publisher
        result = run_publisher(base_state)
        assert Path(result["report_path"]).exists()
        assert result["report_written"] is True

    def test_report_written_even_if_llm_fails(self, base_state, monkeypatch):
        """LLM failure must not prevent report from being written."""
        import agents.publisher as pub
        monkeypatch.setattr(pub, "_call_llm_for_narrative", lambda *a, **k: "[Narrative unavailable]")
        from agents.publisher import run_publisher
        result = run_publisher(base_state)
        assert Path(result["report_path"]).exists()
        assert result["report_written"] is True
        html = Path(result["report_path"]).read_text()
        assert "Narrative unavailable" in html


# ── BLOCK 4 — QA Gate checks (6 tests) ────────────────────────────

class TestQAGateChecks:

    def test_passes_with_valid_report_and_submission(self, base_state):
        """Full happy path — QA gate must pass."""
        from agents.submission_strategist import run_submission_strategist
        from agents.publisher import run_publisher
        from agents.qa_gate import run_qa_gate

        state = run_submission_strategist(base_state)
        state = run_publisher(state)
        result = run_qa_gate(state)
        assert result["qa_passed"] is True, (
            f"QA failed unexpectedly. Failures: {result['qa_failures']}"
        )

    def test_fails_with_unfilled_slot_in_report(self, base_state, tmp_path):
        """Report with {{UNFILLED}} slot must fail QA."""
        from agents.submission_strategist import run_submission_strategist
        from agents.qa_gate import run_qa_gate

        report_path = tmp_path / "report.html"
        report_path.write_text("<html><p>{{UNFILLED_SLOT}}</p></html>")

        state = run_submission_strategist(base_state)
        state["report_path"]    = str(report_path)
        state["report_written"] = True
        result = run_qa_gate(state)
        assert result["qa_passed"] is False
        assert any("UNFILLED_SLOT" in f for f in result["qa_failures"])

    def test_fails_with_orphan_number_in_narrative(self, base_state, tmp_path):
        """Narrative paragraph containing 0.8121 must fail QA."""
        from agents.submission_strategist import run_submission_strategist
        from agents.qa_gate import run_qa_gate

        report_path = tmp_path / "report.html"
        report_path.write_text(
            "<html><table><tr><td>CV</td><td>0.820</td></tr></table>"
            "<section id='narrative'><p>The model achieved 0.8121 accuracy.</p></section>"
            "</html>"
        )
        state = run_submission_strategist(base_state)
        state["report_path"]    = str(report_path)
        state["report_written"] = True
        result = run_qa_gate(state)
        assert result["qa_passed"] is False
        assert any("0.8121" in f for f in result["qa_failures"])

    def test_numbers_in_table_do_not_trigger_orphan_check(self, base_state, tmp_path):
        """Numbers inside <table> must not be flagged as orphans."""
        from agents.submission_strategist import run_submission_strategist
        from agents.qa_gate import run_qa_gate

        report_path = tmp_path / "report.html"
        report_path.write_text(
            "<html><table><tr><td>0.8200</td></tr></table>"
            "<section id='narrative'><p>The model performed well.</p></section>"
            "</html>"
        )
        state = run_submission_strategist(base_state)
        state["report_path"]    = str(report_path)
        state["report_written"] = True
        result = run_qa_gate(state)
        orphan_failures = [f for f in result["qa_failures"] if "0.8200" in f]
        assert not orphan_failures, (
            "Number inside <table> was incorrectly flagged as an orphan."
        )

    def test_fails_when_submission_csv_missing(self, base_state, tmp_path):
        from agents.submission_strategist import run_submission_strategist
        from agents.publisher import run_publisher
        from agents.qa_gate import run_qa_gate

        state = run_submission_strategist(base_state)
        state = run_publisher(state)
        state["submission_path"] = str(tmp_path / "nonexistent.csv")
        result = run_qa_gate(state)
        assert result["qa_passed"] is False
        assert any("not found" in f.lower() for f in result["qa_failures"])

    def test_qa_gate_never_raises(self, base_state):
        """QA gate must not raise — always returns state with qa_passed set."""
        from agents.qa_gate import run_qa_gate
        state = {"session_id": "empty", "competition_name": "test"}
        result = run_qa_gate(state)
        assert "qa_passed" in result
        assert isinstance(result["qa_passed"], bool)
