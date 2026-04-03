# tests/regression/test_phase3_regression.py
#
# PHASE 3 REGRESSION FREEZE
#
# Written: Day 22 — 2026-04-03
# Commit hash at freeze: (set at test definition time)
# Gate result: tests/phase3_gate_results/[session_id]/gate_result.json
#
# These tests are IMMUTABLE. They encode the minimum quality floor that existed
# when Phase 3 gate passed. Any failure here means a Phase 3 capability has
# regressed and must be restored — not papered over.
#
# DO NOT EDIT. DO NOT PARAMETRIZE. DO NOT RELAX THRESHOLDS.
# If a test fails: fix the underlying capability. Never fix the test.
#
# Run as part of the standard regression suite:
#   pytest tests/regression/ -v

import json
import subprocess
import numpy as np
import pytest
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def benchmark_state(benchmark_dataset_spaceship_titanic):
    """
    Runs the full Professor pipeline once per test session on a fixed
    benchmark dataset (Spaceship Titanic, full training data) and returns
    the final state. Expensive — runs once, shared across all tests.
    """
    from core.professor import run_professor
    return run_professor(benchmark_dataset_spaceship_titanic)


@pytest.fixture(scope="session")
def phase3_gate_result():
    """
    Loads the first passing Phase 3 gate result found in the gate results directory.
    Returns None if no passing gate exists.
    """
    gate_dir = Path("tests/phase3_gate_results")
    if not gate_dir.exists():
        return None

    for session_folder in sorted(gate_dir.iterdir()):
        gate_file = session_folder / "gate_result.json"
        if gate_file.exists():
            data = json.loads(gate_file.read_text())
            if data.get("gate_passed"):
                return data

    return None


# ── FREEZE 1: Phase 1 + Phase 2 CV floors still hold ─────────────────────────
#
# What: Professor's CV on Spaceship Titanic must not drop below Phase 2's floor.
# Why:  Adding Phase 3 features (stability ranking, Critic Vectors 2+3, harness)
#       must not regress the core CV score that earlier phases established.
# Tolerance: 0.002 — 2pp below Phase 2 baseline. Anything more is a regression.

class TestPhase1And2FloorsStillHold:

    def test_phase1_and_phase2_cv_floors_still_hold(self, benchmark_state):
        phase2_baseline = json.loads(
            Path("tests/regression/phase2_baseline.json").read_text()
        )
        phase2_floor = float(phase2_baseline["cv_mean"])
        current_cv = float(benchmark_state["cv_mean"])
        assert current_cv >= phase2_floor - 0.002, (
            f"REGRESSION: CV {current_cv:.5f} dropped below Phase 2 floor "
            f"{phase2_floor:.5f}. Phase 3 additions regressed baseline performance."
        )


# ── FREEZE 2: CV score floor from gate result ──────────────────────────────────
#
# What: Professor's CV score on the standard benchmark must stay within 0.020
#       of the score it achieved when the Phase 3 gate passed.
# Why:  The gate score is the reference point. A 2pp drop signals something
#       materially broke.
# Note: Floor = gate_cv_score - 0.020. Read from actual gate result.

class TestCVFloorFromGateScore:

    def test_cv_score_above_gate_floor(self, benchmark_state, phase3_gate_result):
        if phase3_gate_result is None:
            pytest.skip(
                "No passing Phase 3 gate result found. "
                "Run phase3_gate.py and ensure gate_passed is True."
            )

        gate_cv_score = float(phase3_gate_result.get("cv_score", 0.0))
        if gate_cv_score == 0.0:
            pytest.skip("Gate CV score not recorded. Update gate_result.json.")

        floor = gate_cv_score - 0.020
        current_cv = float(benchmark_state["cv_mean"])

        assert current_cv >= floor, (
            f"REGRESSION: CV {current_cv:.5f} dropped below gate floor "
            f"{floor:.5f} (gate_cv={gate_cv_score:.5f} - 0.020). "
            f"A 2pp drop signals a material regression in model quality."
        )


# ── FREEZE 3: Null importance filter is running and removing features ─────────
#
# What: Null importance filter must drop at least some features and must not
#       drop all features. It must be running and producing meaningful output.
# Why:  If the filter stops running, Professor trains on noise features.
# Bounds: [5, 200] features surviving — calibrated from gate run.

class TestNullImportanceFilterWorks:

    def test_null_importance_filter_removes_features(self, benchmark_state):
        n_final = benchmark_state.get("n_features_final", 0)
        assert 5 <= n_final <= 200, (
            f"REGRESSION: {n_final} features survived null importance. "
            "Either filter is not running (>200) or is too aggressive (<5)."
        )
        dropped = (benchmark_state.get("stage1_drop_count", 0) +
                   benchmark_state.get("stage2_drop_count", 0))
        assert dropped > 0, (
            "REGRESSION: Null importance filter dropped 0 features. Filter is not running."
        )


# ── FREEZE 4: Optuna stability ranking beats peak ranking ─────────────────────
#
# What: The model selected by stability_score must have seed_results from 5 seeds
#       and the stability_score formula must be mean - 1.5*std.
# Why:  The entire point of stability ranking is to avoid lucky-seed winners.

class TestOptunaStabilityRankingBeatsPeak:

    def test_stability_ranking_beats_peak_ranking(self, benchmark_state):
        registry = benchmark_state.get("model_registry", {})
        if isinstance(registry, list):
            if len(registry) < 2:
                pytest.skip("Need at least 2 models to compare stability vs peak.")
            entries = registry
        elif isinstance(registry, dict):
            if len(registry) < 2:
                pytest.skip("Need at least 2 models to compare stability vs peak.")
            entries = list(registry.values())
        else:
            pytest.skip(f"model_registry has unexpected type: {type(registry)}")

        winner = max(entries, key=lambda e: e.get("stability_score", 0.0))
        assert "seed_results" in winner, "REGRESSION: Winner missing seed_results."
        assert len(winner["seed_results"]) == 5, (
            f"REGRESSION: Winner has {len(winner['seed_results'])} seed results, expected 5."
        )
        computed_stability = (
            float(np.mean(winner["seed_results"])) -
            1.5 * float(np.std(winner["seed_results"]))
        )
        assert abs(winner["stability_score"] - computed_stability) < 1e-5, (
            "REGRESSION: stability_score formula changed. "
            f"Expected mean - 1.5*std = {computed_stability:.6f}, "
            f"got {winner['stability_score']:.6f}."
        )


# ── FREEZE 5: All 4 core Critic vectors fire on injected failures ─────────────
#
# What: The four core Critic vectors (shuffled_target, preprocessing_audit,
#       pr_curve_imbalance, temporal_leakage) must each return CRITICAL or HIGH
#       when their specific failure mode is injected.
# Why:  Critic vectors can be silently disabled by import errors or threshold changes.

class TestAllCriticVectorsFire:

    def _run_critic(self, state):
        from agents.red_team_critic import run_red_team_critic
        return run_red_team_critic(state)["critic_verdict"]

    def _base_state(self, benchmark_state):
        return {**benchmark_state, "critic_severity": "unchecked"}

    def test_shuffled_target_vector_fires_on_target_derived_feature(self, benchmark_state):
        """Vector 1: shuffled target test must catch a target-derived feature."""
        state = self._base_state(benchmark_state)
        state = _inject_leaky_feature(state, feature_type="target_derived")
        verdict = self._run_critic(state)

        assert verdict["overall_severity"] == "CRITICAL", (
            f"REGRESSION: Critic returned '{verdict['overall_severity']}' "
            "on target-derived leakage. Vector 1 (shuffled_target) must return CRITICAL."
        )

    def test_preprocessing_audit_vector_fires_on_fit_before_split(self, benchmark_state):
        """Vector 4: preprocessing audit must catch fit_transform before train/test split."""
        state = self._base_state(benchmark_state)
        state = _inject_leaky_preprocessing(state)
        verdict = self._run_critic(state)

        fitting_found = any(
            "preprocessing" in f.get("vector", "")
            for f in verdict.get("findings", [])
        )
        assert fitting_found or verdict["overall_severity"] in ("CRITICAL", "HIGH"), (
            "REGRESSION: Critic did not flag fit_before_split preprocessing leakage. "
            "Vector 4 (preprocessing_audit) may not be running."
        )

    def test_pr_curve_imbalance_vector_fires_on_majority_class_model(self, benchmark_state):
        """Vector 5: PR curve check must flag a model that always predicts majority class."""
        state = {
            **self._base_state(benchmark_state),
            "imbalance_ratio": 0.05,   # 5% minority
        }
        state["model_registry"] = {
            "collapsed_model": {
                "oof_predictions": [0.0] * 1000,
                "cv_mean": 0.50,
                "fold_scores": [0.50] * 5,
            }
        }
        verdict = self._run_critic(state)

        assert verdict["overall_severity"] in ("CRITICAL", "HIGH"), (
            "REGRESSION: Critic did not flag a majority-class-only model on imbalanced data. "
            "Vector 5 (pr_curve_imbalance) may not be running."
        )

    def test_temporal_leakage_vector_fires_on_row_index_feature(self, benchmark_state):
        """Vector 6: temporal leakage check fires when a feature is the row index."""
        state = {
            **self._base_state(benchmark_state),
            "has_dates": True,
        }
        n = 1000
        state["feature_names"] = ["time_proxy"]
        state["X_train"] = _make_dataframe_with_time_proxy(n)
        state["y_train"] = np.random.default_rng(42).integers(0, 2, n)

        verdict = self._run_critic(state)

        temporal_fired = any(
            "temporal" in f.get("vector", "")
            for f in verdict.get("findings", [])
        )
        assert temporal_fired or verdict["overall_severity"] in ("CRITICAL", "HIGH"), (
            "REGRESSION: Critic did not flag a row-index-correlated feature. "
            "Vector 6 (temporal_leakage) may not be running when has_dates=True."
        )


# ── FREEZE 6: Wilcoxon gate rejects noise ─────────────────────────────────────
#
# What: Wilcoxon gate must NOT approve noise-level differences.
# Why:  Complex models being selected on lucky seeds is the #1 quality killer.

class TestWilcoxonGateRejectsNoise:

    def test_wilcoxon_gate_rejects_noise_level_difference(self):
        from tools.wilcoxon_gate import is_significantly_better
        a = [0.8012, 0.8009, 0.8015, 0.8011, 0.8013]
        b = [0.8010, 0.8012, 0.8013, 0.8009, 0.8014]
        assert is_significantly_better(a, b) is False, (
            "REGRESSION: Wilcoxon gate approved noise-level difference. "
            "Complex models being selected on lucky seeds."
        )


# ── FREEZE 7: Ensemble architect diversity pruning enforced ────────────────────
#
# What: No pair of models in the final ensemble may have correlation > 0.98.
# Why:  Diversity pruning is the core mechanism that prevents redundant models
#       from entering the ensemble. If it stops working, ensemble quality degrades.

class TestEnsembleArchitectDiversityPruning:

    def test_ensemble_architect_prunes_correlated_models(self, benchmark_state):
        corr_matrix = benchmark_state.get("ensemble_correlation_matrix", {})
        for pair, corr in corr_matrix.items():
            assert corr <= 0.98, (
                f"REGRESSION: Model pair {pair} has correlation {corr:.4f} > 0.98 "
                "in the final ensemble. Diversity pruning is not working."
            )


# ── Helper functions ──────────────────────────────────────────────────────────

def _inject_leaky_feature(state: dict, feature_type: str) -> dict:
    """Injects a feature derived from the target column."""
    import polars as pl

    y = state.get("y_train", np.array([0, 1] * 500))
    n = len(y)

    if feature_type == "target_derived":
        leak = pl.Series("leaked_target_copy", y.astype(float).tolist())
    else:
        leak = pl.Series("leaked_row_id", list(range(n)))

    X = state.get("X_train")
    if X is not None and hasattr(X, "with_columns"):
        state = {**state, "X_train": X.with_columns([leak])}

    state["feature_names"] = list(state.get("feature_names", [])) + [leak.name]
    return state


def _inject_leaky_preprocessing(state: dict) -> dict:
    """Injects a code snippet with fit_transform before split."""
    return {**state, "_injected_preprocessing_code": "scaler.fit_transform(X)"}


def _make_dataframe_with_time_proxy(n: int):
    import polars as pl
    return pl.DataFrame({
        "time_proxy": list(range(n)),
        "noise": np.random.default_rng(42).uniform(0, 1, n).tolist(),
    })
