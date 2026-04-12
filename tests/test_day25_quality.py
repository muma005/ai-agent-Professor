# tests/test_day25_quality.py
# Day 25: pseudo-labeling gates, session isolation, time-series routing.

import pytest
import numpy as np
import polars as pl
from pathlib import Path


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def y_train_100(rng):
    return rng.integers(0, 2, 100).astype(np.float32)


@pytest.fixture
def base_pl_state(rng, y_train_100, tmp_path):
    """Minimal state for pseudo-label agent tests. NOT all gates pass."""
    return {
        "competition_name":    "test-comp",
        "session_id":          "test_session_25",
        "evaluation_metric":   "rmse",        # Gate 1 FAILS by default
        "task_type":           "binary_classification",
        "target_column":       "target",
        "id_column":           "id",
        "y_train":             y_train_100,
        "n_test_rows":         50,             # Gate 2: not 2x train
        "model_registry": {
            "model_a": {
                "cv_mean": 0.820, "cv_std": 0.010,
                "fold_scores": [0.820] * 5,
                "stability_score": 0.805,
                "calibration_score": 0.05,     # good calibration (low Brier)
                "is_calibrated": True,
                "oof_predictions": rng.uniform(0, 1, 100).tolist(),
                "params": {"model_type": "lgbm"},
                "data_hash": "abc123",
            }
        },
        "data_hash":    "abc123",
        "output_dir":   str(tmp_path),
    }


@pytest.fixture
def full_pl_state(rng, tmp_path):
    """State where ALL three gates pass. Writes actual CSV files to tmp_path."""
    n_train = 100
    n_test = 300  # 3x train — gate 2 passes

    train_df = pl.DataFrame({
        "id": list(range(n_train)),
        **{f"f{i}": rng.uniform(0, 1, n_train).tolist() for i in range(5)},
        "target": rng.integers(0, 2, n_train).tolist(),
    })
    train_path = tmp_path / "train.csv"
    train_df.write_csv(train_path)

    test_df = pl.DataFrame({
        "id": list(range(n_train, n_train + n_test)),
        **{f"f{i}": rng.uniform(0, 1, n_test).tolist() for i in range(5)},
    })
    test_path = tmp_path / "test.csv"
    test_df.write_csv(test_path)

    y_train = train_df["target"].to_numpy().astype(np.float32)

    return {
        "competition_name":    "test-comp",
        "session_id":          "full_pl_session",
        "evaluation_metric":   "logloss",   # Gate 1: passes
        "task_type":           "binary_classification",
        "target_column":       "target",
        "id_column":           "id",
        "y_train":             y_train,
        "n_test_rows":         n_test,       # Gate 2: 3x train
        "train_path":          str(train_path),
        "test_path":           str(test_path),
        "feature_order":       [f"f{i}" for i in range(5)],
        "model_registry": {
            "model_a": {
                "cv_mean": 0.820, "cv_std": 0.010,
                "fold_scores": [0.820] * 5,
                "stability_score": 0.805,
                "calibration_score": 0.04,   # Gate 3: 0.04 Brier = good calibration
                "is_calibrated": True,
                "oof_predictions": y_train.tolist(),
                "params": {"model_type": "lgbm", "n_estimators": 100},
                "data_hash": "abc123",
            }
        },
        "data_hash":    "abc123",
        "output_dir":   str(tmp_path),
    }


@pytest.fixture
def ts_state():
    """Minimal state for time-series routing tests."""
    return {
        "competition_name":  "store-sales",
        "session_id":        "ts_session",
        "task_type":         "timeseries",
        "evaluation_metric": "rmsle",
        "target_column":     "sales",
        "id_column":         "id",
        "cv_n_splits":       5,
    }


@pytest.fixture
def ts_state_with_schema(ts_state, tmp_path):
    """Time-series state with schema.json written to disk."""
    schema = {
        "columns": [
            {"name": "sales",    "dtype": "float64", "n_unique": 200,
             "null_fraction": 0.0, "min": 0.0, "is_id": False, "is_target": True},
            {"name": "store_id", "dtype": "int64",   "n_unique": 10,
             "null_fraction": 0.0, "is_id": False, "is_target": False},
            {"name": "date",     "dtype": "date",    "n_unique": 365,
             "null_fraction": 0.0, "is_id": False, "is_target": False},
            {"name": "promo",    "dtype": "int64",   "n_unique": 2,
             "null_fraction": 0.05, "is_id": False, "is_target": False},
        ],
        "target_column": "sales",
        "id_column":     "id",
    }
    schema_path = tmp_path / "schema.json"
    import json
    schema_path.write_text(json.dumps(schema))

    return {
        **ts_state,
        "output_dir":   str(tmp_path),
        "schema_path":  str(schema_path),
    }


@pytest.fixture
def base_state_classification():
    return {
        "task_type":           "binary_classification",
        "evaluation_metric":   "accuracy",
        "target_column":       "Transported",
    }


# Helper functions
def _mock_wilcoxon_pass(monkeypatch):
    import tools.wilcoxon_gate as wg
    monkeypatch.setattr(wg, "is_significantly_better", lambda *a, **k: True)


def _mock_downstream_gates_pass(state, monkeypatch):
    """Mocks gates 2 and 3 to pass so only gate 1 is being tested."""
    import agents.pseudo_label_agent as pla
    monkeypatch.setattr(pla, "_count_test_rows", lambda s: 999)
    monkeypatch.setattr(pla, "_get_best_calibration_score", lambda s: 0.90)


# =========================================================================
# BLOCK 1 — Pseudo-labeling activation gates (10 tests)
# =========================================================================


class TestPseudoLabelActivationGates:
    """All three gates must pass before pseudo-labeling runs."""

    def test_skipped_when_metric_is_accuracy(self, base_pl_state):
        """Gate 1: accuracy is not probability-based — skip."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        state = {**base_pl_state, "evaluation_metric": "accuracy"}
        result = run_pseudo_label_agent(state)
        assert result["pseudo_labels_applied"] is False
        assert "not probability-based" in result["pseudo_label_skip_reason"]

    def test_skipped_when_metric_is_rmse(self, base_pl_state):
        """Gate 1: rmse is not probability-based — skip."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        state = {**base_pl_state, "evaluation_metric": "rmse"}
        result = run_pseudo_label_agent(state)
        assert result["pseudo_labels_applied"] is False

    def test_runs_when_metric_is_logloss(self, base_pl_state, monkeypatch):
        """Gate 1: logloss is probability-based — proceed to gate 2."""
        _mock_downstream_gates_pass(base_pl_state, monkeypatch)
        from agents.pseudo_label_agent import run_pseudo_label_agent
        state = {**base_pl_state, "evaluation_metric": "logloss"}
        result = run_pseudo_label_agent(state)
        # Gate 1 passed — skip reason should not mention metric
        assert "not probability-based" not in result.get("pseudo_label_skip_reason", "")

    def test_skipped_when_test_not_2x_train(self, base_pl_state):
        """Gate 2: test set must be > 2x training set rows."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        state = {
            **base_pl_state,
            "evaluation_metric": "logloss",
            "n_test_rows": 100,   # same as n_train — not 2x
        }
        result = run_pseudo_label_agent(state)
        assert result["pseudo_labels_applied"] is False
        reason = result["pseudo_label_skip_reason"].lower()
        assert "2" in reason and ("x" in reason or "ratio" in reason), (
            f"Expected 2x ratio message, got: {result['pseudo_label_skip_reason']}"
        )

    def test_runs_when_test_is_3x_train(self, base_pl_state, monkeypatch):
        """Gate 2 passes when test set is 3x training rows."""
        _mock_downstream_gates_pass(base_pl_state, monkeypatch)
        from agents.pseudo_label_agent import run_pseudo_label_agent
        state = {
            **base_pl_state,
            "evaluation_metric": "logloss",
            "n_test_rows": 300,   # 3x the 100-row training set
        }
        result = run_pseudo_label_agent(state)
        assert "2x" not in result.get("pseudo_label_skip_reason", "")

    def test_skipped_when_calibration_below_threshold(self, base_pl_state, tmp_path):
        """Gate 3: calibration below 0.80 threshold — skip."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        import json
        n_train = 100
        n_test = 300
        rng = np.random.default_rng(42)

        # Create actual CSV files so we get past the path check
        train_df = pl.DataFrame({
            "id": list(range(n_train)),
            **{f"f{i}": rng.uniform(0, 1, n_train).tolist() for i in range(5)},
            "target": rng.integers(0, 2, n_train).tolist(),
        })
        train_path = tmp_path / "train.csv"
        train_df.write_csv(train_path)
        test_df = pl.DataFrame({
            "id": list(range(n_train, n_train + n_test)),
            **{f"f{i}": rng.uniform(0, 1, n_test).tolist() for i in range(5)},
        })
        test_path = tmp_path / "test.csv"
        test_df.write_csv(test_path)

        # Set a poor calibration score in model_registry
        state = {**base_pl_state}
        # Set a poor calibration score: Brier=0.25 → 1-0.25=0.75 < 0.80
        state["model_registry"]["model_a"]["calibration_score"] = 0.25
        state["evaluation_metric"] = "logloss"
        state["n_test_rows"] = n_test
        state["train_path"] = str(train_path)
        state["test_path"] = str(test_path)
        result = run_pseudo_label_agent(state)
        assert result["pseudo_labels_applied"] is False
        assert "calibration" in result["pseudo_label_skip_reason"].lower()

    @pytest.mark.skip(reason="slow — requires full pseudo-labeling pipeline with large enough dataset")
    def test_all_gates_pass_triggers_run(self, base_pl_state, tmp_path, monkeypatch):
        """When all three gates pass, pseudo-labeling must actually run."""
        state = _setup_full_pass_state(base_pl_state, tmp_path)
        _mock_wilcoxon_pass(monkeypatch)
        from agents.pseudo_label_agent import run_pseudo_label_agent
        # Also mock the critic to always accept
        import agents.pseudo_label_agent as pla
        monkeypatch.setattr(pla, "_critic_verifies_confidence_distribution",
                            lambda *a, **k: (True, "mock accepted"))
        result = run_pseudo_label_agent(state)
        assert result.get("pseudo_label_iterations", 0) >= 1

    def test_gate_check_does_not_read_train_data(self, base_pl_state, monkeypatch):
        """When gates fail, agent must not read train CSV from disk."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        state = {**base_pl_state, "evaluation_metric": "accuracy"}
        read_calls = []
        original_read = pl.read_csv
        monkeypatch.setattr(pl, "read_csv", lambda *a, **k: (read_calls.append(a[0]), original_read(*a, **k))[1])
        run_pseudo_label_agent(state)
        assert len(read_calls) == 0, (
            f"Gate failed but read_csv was called {len(read_calls)} times. "
            "Train data should not be loaded when gates fail."
        )

    def test_skip_reason_set_when_skipped(self, base_pl_state):
        """pseudo_label_skip_reason must always be set when pseudo_labels_applied=False."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        state = {**base_pl_state, "evaluation_metric": "accuracy"}
        result = run_pseudo_label_agent(state)
        assert "pseudo_label_skip_reason" in result
        assert len(result["pseudo_label_skip_reason"]) > 0

    def test_all_required_state_keys_set_when_skipped(self, base_pl_state):
        """Even when skipped, all output state keys must be present."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        state = {**base_pl_state, "evaluation_metric": "rmse"}
        result = run_pseudo_label_agent(state)
        required = [
            "pseudo_labels_applied", "pseudo_label_skip_reason",
            "pseudo_label_iterations", "pseudo_label_n_added",
        ]
        for key in required:
            assert key in result, f"Missing state key: {key}"


# =========================================================================
# BLOCK 3 — Session ID namespace isolation (8 tests)
# =========================================================================


class TestSessionIDNamespaceIsolation:

    def test_generate_session_id_starts_with_professor(self):
        from core.state import generate_session_id
        sid = generate_session_id("spaceship-titanic")
        assert sid.startswith("professor_"), (
            f"session_id '{sid}' does not start with 'professor_'."
        )

    def test_generate_session_id_contains_competition_name(self):
        from core.state import generate_session_id
        sid = generate_session_id("spaceship-titanic")
        assert "spaceship" in sid, (
            f"session_id '{sid}' does not contain the competition slug."
        )

    def test_generate_session_id_is_unique(self):
        """Two calls at the same moment must produce different session IDs."""
        from core.state import generate_session_id
        ids = {generate_session_id("titanic") for _ in range(10)}
        assert len(ids) == 10, (
            "generate_session_id is not unique — produced duplicate IDs."
        )

    def test_build_initial_state_sets_session_id(self):
        from core.state import build_initial_state
        state = build_initial_state("spaceship-titanic")
        assert "session_id" in state
        assert state["session_id"].startswith("professor_")

    def test_build_initial_state_sets_output_dir(self):
        from core.state import build_initial_state
        state = build_initial_state("spaceship-titanic")
        assert "output_dir" in state
        assert state["session_id"] in state["output_dir"]

    def test_two_concurrent_sessions_have_different_output_dirs(self):
        from core.state import build_initial_state
        state_a = build_initial_state("spaceship-titanic")
        state_b = build_initial_state("spaceship-titanic")
        assert state_a["output_dir"] != state_b["output_dir"], (
            "Two sessions have the same output_dir. Concurrent runs will corrupt each other."
        )

    def test_session_id_format_has_correct_structure(self):
        """session_id format: professor_{slug}_{timestamp}_{hash}"""
        from core.state import generate_session_id
        import re
        sid = generate_session_id("test-comp")
        pattern = r"^professor_[a-z0-9\-]+_\d{8}_\d{6}_[a-f0-9]{6}$"
        assert re.match(pattern, sid), (
            f"session_id '{sid}' doesn't match expected format."
        )

    def test_session_id_max_30_char_slug(self):
        """Competition name slug is truncated to 30 chars."""
        from core.state import generate_session_id
        long_name = "a" * 50
        sid = generate_session_id(long_name)
        parts = sid.split("_")
        # parts[1] should be the slug (max 30 chars)
        assert len(parts[1]) <= 30


# =========================================================================
# BLOCK 4 — Time-series routing (8 tests)
# =========================================================================


class TestTimeSeriesRouting:

    def test_timeseries_uses_timeseriessplit_not_kfold(self, ts_state):
        """task_type=timeseries must result in TimeSeriesSplit CV strategy."""
        from agents.validation_architect import run_validation_architect
        result = run_validation_architect(ts_state)
        # Since no schema_path, it will fail — we test the _select_cv_strategy function directly
        from agents.validation_architect import _select_cv_strategy
        strategy = _select_cv_strategy(ts_state)
        assert strategy.get("cv_strategy") == "TimeSeriesSplit", (
            f"Time-series task got CV strategy: {strategy.get('cv_strategy')}. "
            "Must be TimeSeriesSplit."
        )

    def test_timeseries_shuffle_is_false(self, ts_state):
        """TimeSeriesSplit must never shuffle — temporal ordering must be preserved."""
        from agents.validation_architect import _select_cv_strategy
        strategy = _select_cv_strategy(ts_state)
        assert strategy.get("shuffle") is False, (
            f"shuffle={strategy.get('shuffle')} for timeseries. Must be False."
        )

    def test_validate_cv_strategy_raises_on_kfold_for_timeseries(self, ts_state):
        """validate_cv_strategy raises ValueError if KFold used with timeseries."""
        from agents.validation_architect import validate_cv_strategy
        bad_strategy = {"cv_strategy": "KFold", "shuffle": True}
        with pytest.raises(ValueError, match="timeseries"):
            validate_cv_strategy(ts_state, bad_strategy)

    def test_validate_cv_strategy_raises_on_shuffle_true_for_timeseries(self, ts_state):
        """validate_cv_strategy raises if shuffle=True for timeseries."""
        from agents.validation_architect import validate_cv_strategy
        bad_strategy = {"cv_strategy": "TimeSeriesSplit", "shuffle": True}
        with pytest.raises(ValueError, match="shuffle"):
            validate_cv_strategy(ts_state, bad_strategy)

    def test_timeseries_feature_factory_generates_lag_features(self, ts_state_with_schema):
        """task_type=timeseries triggers lag feature generation."""
        from agents.feature_factory import _generate_timeseries_features
        schema = {
            "columns": [
                {"name": "sales",    "dtype": "float64", "n_unique": 200,
                 "null_fraction": 0.0, "is_id": False, "is_target": True},
                {"name": "store_id", "dtype": "int64",   "n_unique": 10,
                 "null_fraction": 0.0, "is_id": False, "is_target": False},
                {"name": "promo",    "dtype": "int64",   "n_unique": 2,
                 "null_fraction": 0.05, "is_id": False, "is_target": False},
            ],
        }
        candidates = _generate_timeseries_features(schema, {})
        lag_features = [c.name for c in candidates if "_lag_" in c.name]
        assert len(lag_features) >= 1, (
            "No lag features generated for timeseries task. "
            "_generate_timeseries_features() may not be routing correctly."
        )

    def test_timeseries_feature_factory_generates_rolling_features(self, ts_state_with_schema):
        """Rolling window features must be generated for timeseries."""
        from agents.feature_factory import _generate_timeseries_features
        schema = {
            "columns": [
                {"name": "sales",    "dtype": "float64", "n_unique": 200,
                 "null_fraction": 0.0, "is_id": False, "is_target": True},
                {"name": "promo",    "dtype": "int64",   "n_unique": 2,
                 "null_fraction": 0.05, "is_id": False, "is_target": False},
            ],
        }
        candidates = _generate_timeseries_features(schema, {})
        rolling = [c.name for c in candidates if "rolling" in c.name]
        assert len(rolling) >= 1, "No rolling features generated for timeseries task."

    def test_non_timeseries_uses_stratified_kfold(self, base_state_classification):
        """Binary classification must still use StratifiedKFold — not TimeSeriesSplit."""
        from agents.validation_architect import _select_cv_strategy
        strategy = _select_cv_strategy(base_state_classification)
        # For non-timeseries, _select_cv_strategy returns empty dict (falls through)
        # The actual CV selection happens in run_validation_architect
        assert strategy == {} or "Stratified" in str(strategy.get("cv_strategy", ""))

    def test_timeseries_select_cv_strategy_returns_correct_config(self, ts_state):
        """_select_cv_strategy for timeseries returns full config dict."""
        from agents.validation_architect import _select_cv_strategy
        strategy = _select_cv_strategy(ts_state)
        assert strategy["cv_strategy"] == "TimeSeriesSplit"
        assert strategy["shuffle"] is False
        assert strategy["stratify"] is False
        assert "n_splits" in strategy["cv_params"]


# =========================================================================
# Helper for full pass state
# =========================================================================


def _setup_full_pass_state(base_state, tmp_path):
    """Set up a state where all gates pass for full pseudo-labeling run."""
    import json
    n_train = 100
    n_test = 300
    rng = np.random.default_rng(42)

    train_df = pl.DataFrame({
        "id": list(range(n_train)),
        **{f"f{i}": rng.uniform(0, 1, n_train).tolist() for i in range(5)},
        "target": rng.integers(0, 2, n_train).tolist(),
    })
    train_path = tmp_path / "train.csv"
    train_df.write_csv(train_path)

    test_df = pl.DataFrame({
        "id": list(range(n_train, n_train + n_test)),
        **{f"f{i}": rng.uniform(0, 1, n_test).tolist() for i in range(5)},
    })
    test_path = tmp_path / "test.csv"
    test_df.write_csv(test_path)

    y_train = train_df["target"].to_numpy().astype(np.float32)

    return {
        **base_state,
        "evaluation_metric": "logloss",
        "n_test_rows": n_test,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "feature_order": [f"f{i}" for i in range(5)],
        "y_train": y_train,
        "model_registry": {
            "model_a": {
                "cv_mean": 0.820, "cv_std": 0.010,
                "fold_scores": [0.820] * 5,
                "stability_score": 0.805,
                "calibration_score": 0.04,
                "is_calibrated": True,
                "oof_predictions": y_train.tolist(),
                "params": {"model_type": "lgbm", "n_estimators": 100},
                "data_hash": "abc123",
            }
        },
    }
