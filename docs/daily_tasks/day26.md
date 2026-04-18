
# Day 26 — Phase 4 Regression Freeze
## Prompt for Qwen Code

---

## BEFORE YOU WRITE A SINGLE LINE

Read these files completely first:

```
tests/regression/test_phase1_regression.py
tests/regression/test_phase2_regression.py
tests/regression/test_phase3_regression.py
agents/ensemble_architect.py
agents/submission_strategist.py
agents/qa_gate.py
core/state.py
core/professor.py
```

After reading, answer before writing:
1. What is the Phase 3 gate CV score? Read from `tests/phase3_gate_results/` or `tests/regression/phase2_baseline.json`.
2. Does `ensemble_architect.py` set `state["ensemble_holdout_score"]`? Quote the line.
3. Does `submission_strategist.py` implement EWMA freeze? Quote the two trigger conditions.
4. Does `qa_gate.py` check for unfilled `{{SLOT}}` patterns? Quote the regex.
5. Does `generate_session_id()` in `core/state.py` include a hash suffix? Quote the line.

Do not write a single test until you have answered all five from the actual code.
If any answer reveals the underlying feature is not built, stop and report it before writing anything.

---

## GATE CHECK — DO THIS FIRST

```python
from pathlib import Path
import json

gate_results = list(Path("tests/phase4_gate_results").glob("*/gate_result.json"))
if not gate_results:
    print("Phase 4 gate has not passed. Write the placeholder file only.")
else:
    gate_result = json.loads(sorted(gate_results)[-1].read_text())
    print(f"Gate passed. CV: {gate_result.get('cv_score')}")
    print(f"Public score: {gate_result.get('public_score')}")
```

If the gate has NOT passed: create only the placeholder file shown below and stop.
If the gate HAS passed: create the full freeze file using the actual gate values.

### Placeholder (gate not yet passed)

```python
# tests/regression/test_phase4_regression.py
#
# PLACEHOLDER — Phase 4 gate has not yet passed.
# Replace this entire file after gate_result.json exists in tests/phase4_gate_results/.
# Gate requirement: 3 competitions, at least 1 scores top 30% on public leaderboard.
```

---

## THE FULL FREEZE FILE (only if gate has passed)

```python
# tests/regression/test_phase4_regression.py
#
# PHASE 4 REGRESSION FREEZE
# Written: [DATE]
# Commit hash at freeze: [git rev-parse HEAD]
# Gate session: [session_id from gate_result.json]
# Gate CV score: [cv_score from gate_result.json]
# Gate public score: [public_score from gate_result.json]
# Competitions at gate: [competition names from gate_result.json]
#
# IMMUTABLE. Never edit. Never relax thresholds.
# If a test fails: fix the underlying capability. Never fix the test.

import json
import numpy as np
import pytest
from pathlib import Path
```

---

### FROZEN TEST 1 — All Phase 1 + 2 + 3 floors still hold

```python
class TestAllPreviousFloorsStillHold:

    def test_phase1_regression_still_passes(self):
        import subprocess
        r = subprocess.run(
            ["pytest", "tests/regression/test_phase1_regression.py", "-q", "--tb=short"],
            capture_output=True, text=True
        )
        assert r.returncode == 0, (
            f"REGRESSION: Phase 1 tests failed.\n{r.stdout[-2000:]}"
        )

    def test_phase2_regression_still_passes(self):
        import subprocess
        r = subprocess.run(
            ["pytest", "tests/regression/test_phase2_regression.py", "-q", "--tb=short"],
            capture_output=True, text=True
        )
        assert r.returncode == 0, (
            f"REGRESSION: Phase 2 tests failed.\n{r.stdout[-2000:]}"
        )

    def test_phase3_regression_still_passes(self):
        import subprocess
        r = subprocess.run(
            ["pytest", "tests/regression/test_phase3_regression.py", "-q", "--tb=short"],
            capture_output=True, text=True
        )
        assert r.returncode == 0, (
            f"REGRESSION: Phase 3 tests failed.\n{r.stdout[-2000:]}"
        )

    def test_cv_above_phase3_floor(self, benchmark_state):
        """CV must not drop below Phase 3 gate CV minus 0.020."""
        gate    = _load_phase4_gate()
        p3_cv   = float(gate.get("phase3_cv_score") or _load_phase2_baseline()["cv_mean"])
        current = float(benchmark_state["cv_mean"])
        floor   = p3_cv - 0.020

        assert current >= floor, (
            f"REGRESSION: CV {current:.5f} dropped below Phase 3 floor {floor:.5f}. "
            f"(phase3_cv={p3_cv:.5f})"
        )
```

---

### FROZEN TEST 2 — Ensemble beats best single model on holdout

```python
class TestEnsembleBeatsSingleModelOnHoldout:

    def test_ensemble_holdout_score_in_state(self, benchmark_state):
        assert "ensemble_holdout_score" in benchmark_state, (
            "REGRESSION: ensemble_holdout_score missing from state."
        )
        assert isinstance(benchmark_state["ensemble_holdout_score"], float)

    def test_ensemble_accepted_in_state(self, benchmark_state):
        assert "ensemble_accepted" in benchmark_state, (
            "REGRESSION: ensemble_accepted missing from state."
        )

    def test_ensemble_holdout_beats_best_single_when_accepted(self, benchmark_state):
        if not benchmark_state.get("ensemble_accepted", False):
            pytest.skip("Ensemble not accepted — Wilcoxon gate failed.")

        registry        = benchmark_state.get("model_registry", {})
        best_single_cv  = max(float(e.get("cv_mean", 0)) for e in registry.values())
        ensemble_score  = float(benchmark_state["ensemble_holdout_score"])

        assert ensemble_score >= best_single_cv - 0.002, (
            f"REGRESSION: Ensemble holdout {ensemble_score:.5f} is more than 0.002 "
            f"below best single model CV {best_single_cv:.5f}."
        )

    def test_no_model_pair_above_0_98_correlation(self, benchmark_state):
        corr = benchmark_state.get("ensemble_correlation_matrix", {})
        for pair, c in corr.items():
            assert c <= 0.98, (
                f"REGRESSION: Pair {pair} correlation {c:.4f} > 0.98. "
                "Diversity pruning broken."
            )
```

---

### FROZEN TEST 3 — EWMA freeze fires correctly on simulated LB drift

```python
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
        early    = [0.005] * 5
        rising   = [0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011]
        state    = _build_strategist_state(_make_log(12, early + rising), tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(state)

        assert result["submission_freeze_active"] is True, (
            "REGRESSION: Freeze did not fire on 7 consecutive gap increases."
        )

    def test_freeze_does_not_fire_on_stable_gaps(self, tmp_path):
        stable = [0.005, 0.005, 0.006, 0.005, 0.005,
                  0.005, 0.006, 0.005, 0.006, 0.005]
        state  = _build_strategist_state(_make_log(10, stable), tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(state)

        assert result["submission_freeze_active"] is False, (
            "REGRESSION: EWMA freeze false positive on stable gaps."
        )

    def test_monitor_not_active_before_5_submissions(self, tmp_path):
        gaps  = [0.050, 0.060, 0.070, 0.080]   # severe drift but only 4 submissions
        state = _build_strategist_state(_make_log(4, gaps), tmp_path)

        from agents.submission_strategist import run_submission_strategist
        result = run_submission_strategist(state)

        assert result["submission_freeze_active"] is False, (
            "REGRESSION: Monitor activated with only 4 lb submissions."
        )
```

---

### FROZEN TEST 4 — QA gate rejects report with missing slots

```python
class TestQAGateRejectsMissingSlots:

    def test_qa_rejects_unfilled_slot(self, tmp_path):
        report = tmp_path / "report.html"
        report.write_text(
            "<html><p>Score: {{UNFILLED_SLOT}}</p></html>"
        )
        import polars as pl
        sub    = tmp_path / "sub.csv"
        sample = tmp_path / "sample.csv"
        df = pl.DataFrame({"PassengerId": ["001"], "Transported": [True]})
        df.write_csv(sub); df.write_csv(sample)

        from agents.qa_gate import run_qa_gate
        result = run_qa_gate({
            "session_id":             "t",
            "report_path":            str(report),
            "report_written":         True,
            "submission_path":        str(sub),
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
        import polars as pl
        sub    = tmp_path / "sub.csv"
        sample = tmp_path / "sample.csv"
        df = pl.DataFrame({"PassengerId": ["001"], "Transported": [True]})
        df.write_csv(sub); df.write_csv(sample)

        from agents.qa_gate import run_qa_gate
        result = run_qa_gate({
            "session_id":             "t2",
            "report_path":            str(report),
            "report_written":         True,
            "submission_path":        str(sub),
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
        import polars as pl
        sub    = tmp_path / "sub.csv"
        sample = tmp_path / "sample.csv"
        df = pl.DataFrame({"PassengerId": ["001"], "Transported": [True]})
        df.write_csv(sub); df.write_csv(sample)

        from agents.qa_gate import run_qa_gate
        result = run_qa_gate({
            "session_id":             "t3",
            "report_path":            str(report),
            "report_written":         True,
            "submission_path":        str(sub),
            "sample_submission_path": str(sample),
        })
        assert result["qa_passed"] is True, (
            f"REGRESSION: QA gate rejected a clean report. Failures: {result['qa_failures']}"
        )
```

---

### FROZEN TEST 5 — Concurrent sessions have isolated namespaces

```python
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
```

---

## HELPER FUNCTIONS

```python
def _load_phase4_gate() -> dict:
    results = list(Path("tests/phase4_gate_results").glob("*/gate_result.json"))
    return json.loads(sorted(results)[-1].read_text()) if results else {}


def _load_phase2_baseline() -> dict:
    p = Path("tests/regression/phase2_baseline.json")
    return json.loads(p.read_text()) if p.exists() else {"cv_mean": 0.0}


def _make_log(n: int, gaps: list) -> list:
    return [
        {
            "submission_number": i + 1,
            "session_id":        "freeze_test",
            "competition_id":    "spaceship-titanic",
            "timestamp":         f"2026-03-{(i % 28)+1:02d}T12:00:00Z",
            "cv_score":          0.820,
            "lb_score":          0.820 - gaps[i],
            "cv_lb_gap":         gaps[i],
            "model_used":        "model_best",
            "ensemble_accepted": True,
            "submission_path":   f"outputs/test/sub_{i+1}.csv",
            "is_final_pair_submission": False,
        }
        for i in range(n)
    ]


def _build_strategist_state(log: list, tmp_path: Path) -> dict:
    import polars as pl, json as _json

    out = tmp_path / "outputs" / "freeze_test"
    out.mkdir(parents=True)
    (out / "submission_log.json").write_text(_json.dumps(log))

    sub    = tmp_path / "sub.csv"
    sample = tmp_path / "sample.csv"
    df = pl.DataFrame({"PassengerId": [f"0001_{i:03d}" for i in range(100)],
                        "Transported": [True] * 100})
    df.write_csv(sub); df.write_csv(sample)

    return {
        "competition_name":        "spaceship-titanic",
        "session_id":              "freeze_test",
        "model_registry": {
            "model_best": {
                "cv_mean": 0.820, "cv_std": 0.010,
                "stability_score": 0.805,
                "fold_scores": [0.820] * 5,
                "oof_predictions": [0.8] * 100,
                "data_hash": "abc123",
            }
        },
        "y_train":                 np.ones(100),
        "evaluation_metric":       "accuracy",
        "task_type":               "binary_classification",
        "target_column":           "Transported",
        "id_column":               "PassengerId",
        "ensemble_accepted":       False,
        "ensemble_oof":            [0.8] * 100,
        "cv_mean":                 0.820,
        "data_hash":               "abc123",
        "sample_submission_path":  str(sample),
        "output_dir":              str(out),
    }
```

---

## COMMIT

```
git commit -m "Day 26: tests/regression/test_phase4_regression.py — frozen after Phase 4 gate"
```

## VERIFICATION

```bash
pytest tests/regression/test_phase4_regression.py -v --tb=short
pytest tests/regression/ -v --tb=short
pytest tests/contracts/ -v --tb=short
```

All three must show zero failures.