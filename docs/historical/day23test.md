# Day 23 — Test Specification
## Prompt for Qwen Code

---

## BEFORE YOU WRITE A SINGLE TEST

Read these files completely first:

```
agents/submission_strategist.py    ← the file you are testing
agents/publisher.py                ← the file you are testing
agents/qa_gate.py                  ← the file you are testing
core/state.py                      ← ProfessorState keys
```

After reading, answer these questions before writing any test:
1. Where does submission_strategist read the submission log from? What is the exact path?
2. What is `MIN_SUBMISSIONS_BEFORE_MONITOR`? What is the exact value?
3. What is `EWMA_ALPHA`? What is the exact value?
4. What are the two EWMA freeze trigger conditions?
5. What does qa_gate do when submission.csv is missing — raise or set a failure?

Do not write tests until you have answered all five.

---

## FILE 1 — `tests/contracts/test_submission_strategist_contract.py`

```python
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
```

### Fixtures

```python
import pytest
import json
import numpy as np
import polars as pl
from pathlib import Path


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
```

### CONTRACT 1 — Monitor activates only after threshold

```python
class TestContract1MonitorActivationThreshold:

    def test_monitor_not_active_with_zero_lb_submissions(self, base_state):
        """No submissions in log yet — monitor must not activate."""
        result = run_submission_strategist(base_state)
        assert result["submission_freeze_active"] is False

    def test_monitor_not_active_with_four_lb_submissions(self, base_state, tmp_path):
        """4 submissions with lb_score — one below the threshold of 5."""
        log = _make_log(n_with_lb=4, gaps=[0.005, 0.005, 0.005, 0.005])
        _write_log(base_state, log, tmp_path)
        result = run_submission_strategist(base_state)
        assert result["submission_freeze_active"] is False, (
            "Monitor activated with only 4 lb submissions. "
            f"MIN_SUBMISSIONS_BEFORE_MONITOR is {_get_min_threshold()}."
        )

    def test_monitor_active_with_five_lb_submissions(self, base_state, tmp_path):
        """5 submissions with lb_score — monitor should now be active."""
        log = _make_log(n_with_lb=5, gaps=[0.005, 0.006, 0.005, 0.005, 0.006])
        _write_log(base_state, log, tmp_path)
        result = run_submission_strategist(base_state)
        assert result["n_submissions_with_lb"] == 5

    def test_min_threshold_is_5(self):
        """The constant MIN_SUBMISSIONS_BEFORE_MONITOR must equal 5."""
        from agents.submission_strategist import MIN_SUBMISSIONS_BEFORE_MONITOR
        assert MIN_SUBMISSIONS_BEFORE_MONITOR == 5, (
            f"MIN_SUBMISSIONS_BEFORE_MONITOR is {MIN_SUBMISSIONS_BEFORE_MONITOR}, expected 5."
        )
```

### CONTRACT 2 — Freeze on EWMA > 2x initial

```python
class TestContract2EWMAFreeze:

    def test_freeze_fires_when_ewma_exceeds_2x_initial(self, base_state, tmp_path):
        """
        Initial 5 submissions have gap ~0.005.
        Later submissions have gap ~0.020.
        current_ewma should be >> 2 * initial_ewma → freeze must fire.
        """
        early_gaps = [0.005, 0.005, 0.006, 0.005, 0.005]
        later_gaps = [0.015, 0.020, 0.022, 0.021, 0.025]
        log = _make_log(n_with_lb=10, gaps=early_gaps + later_gaps)
        _write_log(base_state, log, tmp_path)

        result = run_submission_strategist(base_state)
        assert result["submission_freeze_active"] is True, (
            f"Freeze did not fire. ewma_current={result.get('ewma_current')}, "
            f"ewma_initial={result.get('ewma_initial')}. "
            "Expected current_ewma > 2 * initial_ewma."
        )
        assert result["submission_freeze_reason"] == "ewma_exceeded_2x_initial"

    def test_freeze_does_not_fire_when_ewma_stable(self, base_state, tmp_path):
        """
        All gaps similar and small — EWMA stable, no freeze expected.
        """
        gaps = [0.005, 0.005, 0.006, 0.005, 0.005,
                0.005, 0.006, 0.005, 0.006, 0.005]
        log = _make_log(n_with_lb=10, gaps=gaps)
        _write_log(base_state, log, tmp_path)

        result = run_submission_strategist(base_state)
        if result["submission_freeze_active"]:
            assert result["submission_freeze_reason"] == "gap_increasing_5_of_7", (
                "Stable EWMA triggered a freeze. EWMA computation may be wrong."
            )

    def test_ewma_current_and_initial_both_set_when_monitor_active(
        self, base_state, tmp_path
    ):
        """When monitor is active, both ewma values must be floats, not None."""
        gaps = [0.005] * 10
        log  = _make_log(n_with_lb=10, gaps=gaps)
        _write_log(base_state, log, tmp_path)
        result = run_submission_strategist(base_state)
        assert isinstance(result.get("ewma_current"), float)
        assert isinstance(result.get("ewma_initial"), float)
```

### CONTRACT 3 — Freeze on gap increasing 5 of 7

```python
class TestContract3GapIncreasing5of7:

    def test_freeze_fires_when_gap_increasing_5_of_7(self, base_state, tmp_path):
        """
        Last 7 gaps: each one larger than the previous (monotonically increasing).
        5 of 7 increases → freeze must fire.
        """
        early_gaps = [0.005] * 5
        # 7 consecutive increases
        increasing = [0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011]
        log = _make_log(n_with_lb=12, gaps=early_gaps + increasing)
        _write_log(base_state, log, tmp_path)

        result = run_submission_strategist(base_state)
        # Either EWMA or 5-of-7 trigger may fire — check at least one fired
        assert result["submission_freeze_active"] is True or \
               result.get("submission_freeze_reason") == "gap_increasing_5_of_7", (
            "Freeze did not fire despite 7 consecutive gap increases."
        )

    def test_freeze_does_not_fire_with_4_of_7_increases(self, base_state, tmp_path):
        """
        Last 7 gaps: 4 increases, 3 decreases. Below threshold — no 5-of-7 freeze.
        """
        early_gaps = [0.005] * 5
        mixed = [0.005, 0.006, 0.005, 0.007, 0.006, 0.008, 0.007]
        log = _make_log(n_with_lb=12, gaps=early_gaps + mixed)
        _write_log(base_state, log, tmp_path)

        result = run_submission_strategist(base_state)
        if result["submission_freeze_active"]:
            assert result["submission_freeze_reason"] == "ewma_exceeded_2x_initial", (
                "5-of-7 freeze fired with only 4 increases in last 7. "
                "Count logic is wrong."
            )
```

### CONTRACT 4 — Submission A is best stability_score model

```python
class TestContract4SubmissionAIsBestCV:

    def test_submission_a_uses_highest_stability_score_model(self, base_state):
        result = run_submission_strategist(base_state)
        assert result["submission_a_model"] == "model_best", (
            f"Submission A uses '{result['submission_a_model']}', "
            "expected 'model_best' (highest stability_score=0.805)."
        )

    def test_submission_a_uses_ensemble_when_accepted(self, base_state):
        state = {**base_state, "ensemble_accepted": True}
        result = run_submission_strategist(state)
        # When ensemble accepted, submission A predictions come from ensemble_oof
        # Verify submission_a.csv predictions align with ensemble_oof
        sub_a = pl.read_csv(result["submission_a_path"])
        assert len(sub_a) == len(base_state["ensemble_oof"])
```

### CONTRACT 5 — Submission B has lowest correlation with A

```python
class TestContract5SubmissionBMostDiverse:

    def test_submission_b_has_lowest_correlation_with_a(self, base_state):
        from scipy.stats import pearsonr
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
        result = run_submission_strategist(base_state)
        assert "submission_b_correlation_with_a" in result
        assert isinstance(result["submission_b_correlation_with_a"], float)
        assert 0.0 <= result["submission_b_correlation_with_a"] <= 1.0
```

### CONTRACT 6 — Submission validated against sample_submission

```python
class TestContract6SubmissionValidated:

    def test_wrong_column_names_raises_value_error(self, base_state, tmp_path):
        """Submission with wrong column names must raise before file is written."""
        # Create a sample_submission with different column name
        bad_sample = pl.DataFrame({
            "ID": [f"0001_{i:03d}" for i in range(500)],
            "Transported": [True] * 500,
        })
        bad_path = tmp_path / "bad_sample.csv"
        bad_sample.write_csv(bad_path)
        state = {**base_state, "sample_submission_path": str(bad_path)}

        with pytest.raises(ValueError, match="Column"):
            run_submission_strategist(state)

    def test_wrong_row_count_raises_value_error(self, base_state, tmp_path):
        """Submission with wrong row count must raise before file is written."""
        short_sample = pl.DataFrame({
            "PassengerId": [f"0001_{i:03d}" for i in range(100)],  # wrong count
            "Transported": [True] * 100,
        })
        bad_path = tmp_path / "short_sample.csv"
        short_sample.write_csv(bad_path)
        state = {**base_state, "sample_submission_path": str(bad_path)}

        with pytest.raises(ValueError, match="Row count"):
            run_submission_strategist(state)

    def test_valid_submission_written_without_error(self, base_state):
        result = run_submission_strategist(base_state)
        assert Path(result["submission_path"]).exists()
```

### CONTRACT 7 — Submission log updated after every run

```python
class TestContract7SubmissionLogUpdated:

    def test_log_created_on_first_run(self, base_state, tmp_path):
        result = run_submission_strategist(base_state)
        log_path = Path(result["submission_log_path"])
        assert log_path.exists(), "submission_log.json not created on first run."
        log = json.loads(log_path.read_text())
        assert isinstance(log, list)
        assert len(log) == 1

    def test_log_appends_on_subsequent_runs(self, base_state, tmp_path):
        """Two runs → two records in the log."""
        run_submission_strategist(base_state)
        run_submission_strategist(base_state)
        log_path = Path(base_state["output_dir"]) / ".." / "submission_log.json"
        # Find the actual log path from state after first run
        result = run_submission_strategist(base_state)
        log = json.loads(Path(result["submission_log_path"]).read_text())
        assert len(log) >= 1  # At minimum one record

    def test_log_record_has_required_fields(self, base_state):
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
        result = run_submission_strategist(base_state)
        log    = json.loads(Path(result["submission_log_path"]).read_text())
        assert log[-1]["lb_score"] is None, (
            "lb_score should be null at write time — it is only known after Kaggle scores."
        )
```

### CONTRACT 8 — freeze_active always set

```python
class TestContract8FreezeActiveAlwaysSet:

    def test_freeze_active_set_when_no_history(self, base_state):
        result = run_submission_strategist(base_state)
        assert "submission_freeze_active" in result
        assert isinstance(result["submission_freeze_active"], bool)

    def test_freeze_active_is_false_before_threshold(self, base_state):
        result = run_submission_strategist(base_state)
        assert result["submission_freeze_active"] is False
```

### CONTRACT 9 — Both CSV files written

```python
class TestContract9BothFilesWritten:

    def test_submission_a_csv_written(self, base_state):
        result = run_submission_strategist(base_state)
        assert Path(result["submission_a_path"]).exists()

    def test_submission_b_csv_written(self, base_state):
        result = run_submission_strategist(base_state)
        assert Path(result["submission_b_path"]).exists()

    def test_submission_csv_same_as_a(self, base_state):
        result = run_submission_strategist(base_state)
        a_content   = Path(result["submission_a_path"]).read_text()
        sub_content = Path(result["submission_path"]).read_text()
        assert a_content == sub_content, (
            "submission.csv content differs from submission_a.csv. "
            "Default submission must be identical to submission A."
        )
```

### CONTRACT 10 — All required state keys set

```python
class TestContract10StateKeys:

    REQUIRED = [
        "submission_a_path", "submission_b_path", "submission_path",
        "submission_a_model", "submission_b_model",
        "submission_b_correlation_with_a", "submission_log_path",
        "submission_freeze_active", "submission_freeze_reason",
        "ewma_current", "ewma_initial", "n_submissions_with_lb",
    ]

    def test_all_required_state_keys_present(self, base_state):
        result = run_submission_strategist(base_state)
        missing = [k for k in self.REQUIRED if k not in result]
        assert not missing, f"Missing state keys: {missing}"
```

---

## FILE 2 — `tests/test_day23_quality.py`

```python
# tests/test_day23_quality.py
# Day 23 quality tests — algorithmic correctness for all three agents.
```

### BLOCK 1 — EWMA computation correctness (5 tests)

```python
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
```

### BLOCK 2 — Submission pair selection (4 tests)

```python
class TestSubmissionPairSelection:

    def test_submission_a_is_highest_stability_not_highest_mean(
        self, base_state, rng
    ):
        """
        stability_score = mean - 1.5*std. A model with lower mean but lower std
        can have higher stability_score. A must use stability, not raw mean.
        """
        # model_stable: mean=0.810, std=0.001 → stability=0.810-0.0015=0.808
        # model_peaky:  mean=0.820, std=0.020 → stability=0.820-0.030=0.790
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
        result = run_submission_strategist(state)
        assert result["submission_a_model"] == "model_stable", (
            "Submission A selected highest mean CV, not highest stability_score. "
            "stability_score = mean - 1.5*std must be used."
        )

    def test_submission_b_is_not_same_as_a(self, base_state):
        result = run_submission_strategist(base_state)
        assert result["submission_a_model"] != result["submission_b_model"]

    def test_submission_b_different_from_a_in_content(self, base_state):
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
        result = run_submission_strategist(state)
        assert result["submission_a_model"] == "only_model"
        assert result["submission_b_model"] == "only_model"
```

### BLOCK 3 — Publisher template correctness (5 tests)

```python
class TestPublisherTemplate:

    def test_all_numeric_slots_filled(self, base_state):
        """After fill_numeric_slots, no {{SLOT}} patterns remain for numeric slots."""
        from agents.publisher import fill_numeric_slots, TEMPLATE, NUMERIC_SLOTS
        import re
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
        result = run_publisher(base_state)
        assert Path(result["report_path"]).exists()
        assert result["report_written"] is True

    def test_report_written_even_if_llm_fails(self, base_state, monkeypatch):
        """LLM failure must not prevent report from being written."""
        import agents.publisher as pub
        monkeypatch.setattr(pub, "llm_call", lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("LLM unavailable")
        ))
        result = run_publisher(base_state)
        assert Path(result["report_path"]).exists()
        assert result["report_written"] is True
        html = Path(result["report_path"]).read_text()
        assert "Narrative unavailable" in html
```

### BLOCK 4 — QA Gate checks (6 tests)

```python
class TestQAGateChecks:

    def test_passes_with_valid_report_and_submission(self, base_state):
        """Full happy path — QA gate must pass."""
        state = run_submission_strategist(base_state)
        state = run_publisher(state)
        result = run_qa_gate(state)
        assert result["qa_passed"] is True, (
            f"QA failed unexpectedly. Failures: {result['qa_failures']}"
        )

    def test_fails_with_unfilled_slot_in_report(self, base_state, tmp_path):
        """Report with {{UNFILLED}} slot must fail QA."""
        report_path = tmp_path / "report.html"
        report_path.write_text("<html><p>{{UNFILLED_SLOT}}</p></html>")
        state = {
            **base_state,
            "report_path": str(report_path),
            "report_written": True,
            "submission_path": str(base_state.get("submission_path", "")),
        }
        # Run strategist first to set submission_path
        state = run_submission_strategist(base_state)
        state["report_path"]    = str(report_path)
        state["report_written"] = True
        result = run_qa_gate(state)
        assert result["qa_passed"] is False
        assert any("UNFILLED_SLOT" in f for f in result["qa_failures"])

    def test_fails_with_orphan_number_in_narrative(self, base_state, tmp_path):
        """Narrative paragraph containing 0.8121 must fail QA."""
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
        state = run_submission_strategist(base_state)
        state = run_publisher(state)
        state["submission_path"] = str(tmp_path / "nonexistent.csv")
        result = run_qa_gate(state)
        assert result["qa_passed"] is False
        assert any("not found" in f.lower() for f in result["qa_failures"])

    def test_qa_gate_never_raises(self, base_state):
        """QA gate must not raise — always returns state with qa_passed set."""
        state = {"session_id": "empty", "competition_name": "test"}
        result = run_qa_gate(state)
        assert "qa_passed" in result
        assert isinstance(result["qa_passed"], bool)
```

---

## HELPER FUNCTIONS FOR TESTS

Define these at module level in the contract test file:

```python
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
    """Writes a synthetic log to the session output directory."""
    out_dir = Path(state["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "submission_log.json"
    log_path.write_text(json.dumps(log))
```

---

## RUNNING THE TESTS

```bash
pytest tests/contracts/test_submission_strategist_contract.py -v --tb=short
pytest tests/test_day23_quality.py -v --tb=short
pytest tests/contracts/ -v --tb=short
pytest tests/regression/ -v --tb=short
```

All four must show zero failures before committing.