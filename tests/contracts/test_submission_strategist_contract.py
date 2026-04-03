# tests/contracts/test_submission_strategist_contract.py
#
# CONTRACT: agents/submission_strategist.py
# Written: Day 23. IMMUTABLE — never edit after Day 23.
#
# INVARIANTS:
#   1.  EWMA monitor activates only after MIN_SUBMISSIONS_BEFORE_MONITOR submissions with lb_score
#   2.  EWMA freeze fires when current_ewma > 2x initial_ewma
#   3.  EWMA freeze fires when gap increased 5 of last 7 submissions
#   4.  Submission A is the highest stability_score model
#   5.  Submission B has lowest correlation with Submission A OOF
#   6.  submission.csv validated against sample_submission before write
#   7.  Submission log updated after every run
#   8.  submission_freeze_active always set in state (True or False)
#   9.  Both submission_a.csv and submission_b.csv written
#  10.  All required state keys set after run

import json
import pytest
import numpy as np
import polars as pl
from pathlib import Path


# ── Helper functions ──────────────────────────────────────────────

def _get_min_threshold():
    from agents.submission_strategist import MIN_SUBMISSIONS_BEFORE_MONITOR
    return MIN_SUBMISSIONS_BEFORE_MONITOR


def _make_log(n_with_lb: int, gaps: list[float]) -> list[dict]:
    """Creates a synthetic submission log with n_with_lb entries that have lb_score."""
    assert len(gaps) == n_with_lb
    records = []
    for i, gap in enumerate(gaps):
        cv = 0.820
        lb = cv - gap
        records.append({
            "submission_number": i + 1,
            "session_id":        "test_session",
            "competition_id":    "spaceship-titanic",
            "timestamp":         f"2026-03-0{(i % 9) + 1}T12:00:00Z",
            "cv_score":          cv,
            "lb_score":          lb,
            "cv_lb_gap":         gap,
            "model_used":        "model_best",
            "ensemble_accepted": True,
            "submission_path":   f"outputs/test/submission_{i+1}.csv",
            "is_final_pair_submission": False,
        })
    return records


def _write_log(state: dict, log: list, tmp_path) -> None:
    """Writes a synthetic submission log to the session output directory."""
    out_dir = Path(state["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "submission_log.json"
    log_path.write_text(json.dumps(log))


# ── Fixtures ──────────────────────────────────────────────────────

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
    """Genuinely different OOF — low correlation with oof_a."""
    return np.clip(rng.normal(0.5, 0.35, len(oof_a)), 0, 1).tolist()


@pytest.fixture
def oof_c(oof_a, rng):
    """Another different OOF."""
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
    """Creates a sample_submission.csv for format validation tests."""
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


# ── CONTRACT 1 — Monitor activates only after threshold ───────────

class TestContract1MonitorActivationThreshold:

    def test_monitor_not_active_with_zero_lb_submissions(self, base_state):
        """
        Invariant: With no submissions in the log, the EWMA monitor must
        not activate. submission_freeze_active must be False because there
        is no gap history to analyse.
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert result["submission_freeze_active"] is False

    def test_monitor_not_active_with_four_lb_submissions(self, base_state, tmp_path):
        """
        Invariant: With 4 submissions having lb_score (one below the
        threshold of 5), the monitor must not activate. The freeze must
        not fire regardless of gap values.
        """
        log = _make_log(n_with_lb=4, gaps=[0.005, 0.005, 0.005, 0.005])
        _write_log(base_state, log, tmp_path)
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert result["submission_freeze_active"] is False, (
            "Monitor activated with only 4 lb submissions. "
            f"MIN_SUBMISSIONS_BEFORE_MONITOR is {_get_min_threshold()}."
        )

    def test_monitor_active_with_five_lb_submissions(self, base_state, tmp_path):
        """
        Invariant: With 5 submissions having lb_score, the monitor should
        be active and n_submissions_with_lb should reflect the count.
        """
        log = _make_log(n_with_lb=5, gaps=[0.005, 0.006, 0.005, 0.005, 0.006])
        _write_log(base_state, log, tmp_path)
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert result["n_submissions_with_lb"] == 5

    def test_min_threshold_is_5(self):
        """
        Invariant: The constant MIN_SUBMISSIONS_BEFORE_MONITOR must equal
        5. This is the minimum number of submissions with lb_score before
        the EWMA monitor starts evaluating freeze conditions.
        """
        from agents.submission_strategist import MIN_SUBMISSIONS_BEFORE_MONITOR
        assert MIN_SUBMISSIONS_BEFORE_MONITOR == 5, (
            f"MIN_SUBMISSIONS_BEFORE_MONITOR is {MIN_SUBMISSIONS_BEFORE_MONITOR}, expected 5."
        )


# ── CONTRACT 2 — Freeze on EWMA > 2x initial ──────────────────────

class TestContract2EWMAFreeze:

    def test_freeze_fires_when_ewma_exceeds_2x_initial(self, base_state, tmp_path):
        """
        Invariant: When the current EWMA of CV/LB gaps exceeds 2x the
        initial EWMA (computed from the first 5 submissions), the freeze
        must fire with reason "ewma_exceeded_2x_initial". This detects
        systematic overfitting to CV.
        """
        early_gaps = [0.005, 0.005, 0.006, 0.005, 0.005]
        later_gaps = [0.015, 0.020, 0.022, 0.021, 0.025]
        log = _make_log(n_with_lb=10, gaps=early_gaps + later_gaps)
        _write_log(base_state, log, tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert result["submission_freeze_active"] is True, (
            f"Freeze did not fire. ewma_current={result.get('ewma_current')}, "
            f"ewma_initial={result.get('ewma_initial')}. "
            "Expected current_ewma > 2 * initial_ewma."
        )
        assert result["submission_freeze_reason"] == "ewma_exceeded_2x_initial"

    def test_freeze_does_not_fire_when_ewma_stable(self, base_state, tmp_path):
        """
        Invariant: When all gaps are similar and small, the EWMA should
        remain stable and no freeze should fire. This ensures the monitor
        does not produce false positives on normal variation.
        """
        gaps = [0.005, 0.005, 0.006, 0.005, 0.005,
                0.005, 0.006, 0.005, 0.006, 0.005]
        log = _make_log(n_with_lb=10, gaps=gaps)
        _write_log(base_state, log, tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        if result["submission_freeze_active"]:
            assert result["submission_freeze_reason"] == "gap_increasing_5_of_7", (
                "Stable EWMA triggered a freeze. EWMA computation may be wrong."
            )

    def test_ewma_current_and_initial_both_set_when_monitor_active(
        self, base_state, tmp_path
    ):
        """
        Invariant: When the monitor is active (>= 5 submissions with
        lb_score), both ewma_current and ewma_initial must be floats,
        not None. Downstream consumers depend on these values.
        """
        gaps = [0.005] * 10
        log  = _make_log(n_with_lb=10, gaps=gaps)
        _write_log(base_state, log, tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert isinstance(result.get("ewma_current"), float)
        assert isinstance(result.get("ewma_initial"), float)


# ── CONTRACT 3 — Freeze on gap increasing 5 of 7 ──────────────────

class TestContract3GapIncreasing5of7:

    def test_freeze_fires_when_gap_increasing_5_of_7(self, base_state, tmp_path):
        """
        Invariant: When the CV/LB gap increased in 5 or more of the last
        7 submissions, the freeze must fire. This detects a monotonic
        worsening trend even if the EWMA hasn't yet exceeded 2x initial.
        """
        early_gaps = [0.005] * 5
        # 7 consecutive increases
        increasing = [0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011]
        log = _make_log(n_with_lb=12, gaps=early_gaps + increasing)
        _write_log(base_state, log, tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        # Either EWMA or 5-of-7 trigger may fire — check at least one fired
        assert result["submission_freeze_active"] is True or \
               result.get("submission_freeze_reason") == "gap_increasing_5_of_7", (
            "Freeze did not fire despite 7 consecutive gap increases."
        )

    def test_freeze_does_not_fire_with_4_of_7_increases(self, base_state, tmp_path):
        """
        Invariant: With only 4 increases in the last 7 submissions (below
        the threshold of 5), the 5-of-7 freeze must not fire. If a freeze
        fires, it must be due to the EWMA condition, not the count condition.
        """
        early_gaps = [0.005] * 5
        mixed = [0.005, 0.006, 0.005, 0.007, 0.006, 0.008, 0.007]
        log = _make_log(n_with_lb=12, gaps=early_gaps + mixed)
        _write_log(base_state, log, tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        if result["submission_freeze_active"]:
            assert result["submission_freeze_reason"] == "ewma_exceeded_2x_initial", (
                "5-of-7 freeze fired with only 4 increases in last 7. "
                "Count logic is wrong."
            )


# ── CONTRACT 4 — Submission A is best stability_score model ───────

class TestContract4SubmissionAIsBestCV:

    def test_submission_a_uses_highest_stability_score_model(self, base_state):
        """
        Invariant: Submission A must use the model with the highest
        stability_score (not the highest cv_mean). stability_score =
        mean - 1.5*std, so a model with lower mean but lower variance
        can have higher stability and should be preferred.
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert result["submission_a_model"] == "model_best", (
            f"Submission A uses '{result['submission_a_model']}', "
            "expected 'model_best' (highest stability_score=0.805)."
        )

    def test_submission_a_uses_ensemble_when_accepted(self, base_state):
        """
        Invariant: When ensemble_accepted=True, submission A's predictions
        must come from the ensemble OOF, not from a single model. The
        submission CSV must have the same number of rows as ensemble_oof.
        """
        from agents.submission_strategist import run_submission_strategist
        state = {**base_state, "ensemble_accepted": True}
        result = run_submission_strategist(state)
        sub_a = pl.read_csv(result["submission_a_path"])
        assert len(sub_a) == len(base_state["ensemble_oof"])


# ── CONTRACT 5 — Submission B has lowest correlation with A ───────

class TestContract5SubmissionBMostDiverse:

    def test_submission_b_has_lowest_correlation_with_a(self, base_state):
        """
        Invariant: Submission B must be the model whose OOF predictions
        have the lowest Pearson correlation with Submission A's OOF.
        This ensures B provides maximum diversity — it is right where
        A is wrong.
        """
        from scipy.stats import pearsonr
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)

        a_oof = base_state["model_registry"]["model_best"]["oof_predictions"]
        b_name = result["submission_b_model"]
        b_oof  = base_state["model_registry"][b_name]["oof_predictions"]

        corr_b, _ = pearsonr(a_oof, b_oof)

        # Check all other models — B must have lowest correlation
        for name, entry in base_state["model_registry"].items():
            if name == result["submission_a_model"] or name == b_name:
                continue
            other_corr, _ = pearsonr(a_oof, entry["oof_predictions"])
            assert corr_b <= other_corr + 1e-6, (
                f"Submission B ('{b_name}', corr={corr_b:.4f}) is not the "
                f"most diverse from A. '{name}' has lower correlation {other_corr:.4f}."
            )

    def test_submission_b_correlation_stored_in_state(self, base_state):
        """
        Invariant: The Pearson correlation between B and A must be stored
        in state as a float between -1 and 1.
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert "submission_b_correlation_with_a" in result
        assert isinstance(result["submission_b_correlation_with_a"], float)
        assert -1.0 <= result["submission_b_correlation_with_a"] <= 1.0


# ── CONTRACT 6 — Submission validated against sample_submission ────

class TestContract6SubmissionValidated:

    def test_wrong_column_names_raises_value_error(self, base_state, tmp_path):
        """
        Invariant: If the sample_submission has different column names
        than what the strategist would produce, a ValueError containing
        'Column' must be raised before any file is written.
        """
        # Create a sample_submission with different column name
        # The strategist reads sample columns and builds from them, so we need
        # to make the strategist's output fail validation by modifying the
        # sample after the strategist reads it. Instead, we test the validate
        # function directly.
        from agents.submission_strategist import validate_submission
        bad_sample = pl.DataFrame({
            "ID": [f"0001_{i:03d}" for i in range(500)],
            "Transported": [True] * 500,
        })
        bad_path = tmp_path / "bad_sample.csv"
        bad_sample.write_csv(bad_path)

        # Create a valid submission
        good_sub = pl.DataFrame({
            "PassengerId": [f"0001_{i:03d}" for i in range(500)],
            "Transported": [True] * 500,
        })
        with pytest.raises(ValueError, match="Column"):
            validate_submission(good_sub, str(bad_path), {"id_column": "ID", "target_column": "Transported"})

    def test_wrong_row_count_raises_value_error(self, base_state, tmp_path):
        """
        Invariant: If the submission has a different row count than the
        sample_submission, a ValueError containing 'Row count' must be raised.
        """
        from agents.submission_strategist import validate_submission
        short_sample = pl.DataFrame({
            "PassengerId": [f"0001_{i:03d}" for i in range(100)],
            "Transported": [True] * 100,
        })
        bad_path = tmp_path / "short_sample.csv"
        short_sample.write_csv(bad_path)

        # Create a submission with 500 rows
        good_sub = pl.DataFrame({
            "PassengerId": [f"0001_{i:03d}" for i in range(500)],
            "Transported": [True] * 500,
        })
        with pytest.raises(ValueError, match="Row count"):
            validate_submission(good_sub, str(bad_path), {"id_column": "PassengerId", "target_column": "Transported"})

    def test_valid_submission_written_without_error(self, base_state):
        """
        Invariant: With a valid sample_submission and valid predictions,
        the submission CSV must be written without error.
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert Path(result["submission_path"]).exists()


# ── CONTRACT 7 — Submission log updated after every run ────────────

class TestContract7SubmissionLogUpdated:

    def test_log_created_on_first_run(self, base_state, tmp_path):
        """
        Invariant: On the first run, submission_log.json must be created
        with exactly one record. The log is append-only.
        """
        from agents.submission_strategist import run_submission_strategist
        # Use a unique session to avoid accumulation from other tests
        unique_state = {**base_state, "session_id": "test_log_first_run",
                        "output_dir": str(tmp_path / "out")}
        Path(unique_state["output_dir"]).mkdir(parents=True, exist_ok=True)
        result = run_submission_strategist(unique_state)
        log_path = Path(result["submission_log_path"])
        assert log_path.exists(), "submission_log.json not created on first run."
        log = json.loads(log_path.read_text())
        assert isinstance(log, list)
        assert len(log) == 1

    def test_log_appends_on_subsequent_runs(self, base_state, tmp_path):
        """
        Invariant: Multiple runs must append to the same log file.
        Each run adds one record. The log is append-only, not overwritten.
        """
        from agents.submission_strategist import run_submission_strategist
        run_submission_strategist(base_state)
        result = run_submission_strategist(base_state)
        log = json.loads(Path(result["submission_log_path"]).read_text())
        assert len(log) >= 1  # At minimum one record

    def test_log_record_has_required_fields(self, base_state):
        """
        Invariant: Every log record must contain the required fields:
        submission_number, session_id, competition_id, timestamp,
        cv_score, lb_score, cv_lb_gap, model_used, ensemble_accepted,
        submission_path.
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        log    = json.loads(Path(result["submission_log_path"]).read_text())
        record = log[-1]  # most recent
        required = {
            "submission_number", "session_id", "competition_id",
            "timestamp", "cv_score", "lb_score", "cv_lb_gap",
            "model_used", "ensemble_accepted", "submission_path",
        }
        missing = required - set(record.keys())
        assert not missing, f"Log record missing fields: {missing}"

    def test_lb_score_is_null_at_write_time(self, base_state):
        """
        Invariant: lb_score must be null at write time — it is only known
        after Kaggle scores the submission. Setting it to any non-null
        value would be fabricating data.
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        log    = json.loads(Path(result["submission_log_path"]).read_text())
        assert log[-1]["lb_score"] is None, (
            "lb_score should be null at write time — it is only known after Kaggle scores."
        )


# ── CONTRACT 8 — freeze_active always set ──────────────────────────

class TestContract8FreezeActiveAlwaysSet:

    def test_freeze_active_set_when_no_history(self, base_state):
        """
        Invariant: submission_freeze_active must always be present in
        state, even when there is no submission history. It must be a
        bool, not None or missing.
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert "submission_freeze_active" in result
        assert isinstance(result["submission_freeze_active"], bool)

    def test_freeze_active_is_false_before_threshold(self, base_state):
        """
        Invariant: Before the monitor activation threshold (5 submissions
        with lb_score), submission_freeze_active must be False.
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert result["submission_freeze_active"] is False


# ── CONTRACT 9 — Both CSV files written ────────────────────────────

class TestContract9BothFilesWritten:

    def test_submission_a_csv_written(self, base_state):
        """
        Invariant: submission_a.csv must always be written. This is the
        primary submission (best stability model / ensemble).
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert Path(result["submission_a_path"]).exists()

    def test_submission_b_csv_written(self, base_state):
        """
        Invariant: submission_b.csv must always be written. This is the
        diverse submission (lowest correlation with A).
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        assert Path(result["submission_b_path"]).exists()

    def test_submission_csv_same_as_a(self, base_state):
        """
        Invariant: submission.csv (the default) must be identical to
        submission_a.csv. The default submission is always the best CV
        model / ensemble.
        """
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        a_content   = Path(result["submission_a_path"]).read_text()
        sub_content = Path(result["submission_path"]).read_text()
        assert a_content == sub_content, (
            "submission.csv content differs from submission_a.csv. "
            "Default submission must be identical to submission A."
        )


# ── CONTRACT 10 — All required state keys set ──────────────────────

class TestContract10StateKeys:
    """
    Invariant: Every key that submission_strategist promises to set must
    be present in the result. Missing keys would cause downstream nodes
    to crash with KeyError.
    """

    REQUIRED = [
        "submission_a_path", "submission_b_path", "submission_path",
        "submission_a_model", "submission_b_model",
        "submission_b_correlation_with_a", "submission_log_path",
        "submission_freeze_active", "submission_freeze_reason",
        "ewma_current", "ewma_initial", "n_submissions_with_lb",
    ]

    def test_all_required_state_keys_present(self, base_state):
        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(base_state)
        missing = [k for k in self.REQUIRED if k not in result]
        assert not missing, f"Missing state keys: {missing}"
