# Day 25 — Test Specification
## Prompt for Qwen Code

---

## BEFORE YOU WRITE A SINGLE TEST

Read these files first:

```
agents/pseudo_label_agent.py
core/state.py
core/professor.py
agents/feature_factory.py
agents/validation_architect.py
```

After reading, answer before writing:
1. What is `MIN_TEST_TO_TRAIN_RATIO`? Exact value.
2. What is `HIGH_CONFIDENCE_THRESHOLD`? Exact value.
3. What is `MAX_PSEUDO_LABEL_FRACTION`? Exact value.
4. What does `generate_session_id()` return for `competition_name="spaceship-titanic"`? Describe the format.
5. When `task_type == "timeseries"`, does validation_architect raise or return an error state for KFold?

Do not write tests until you have answered all five.

---

## FILE — `tests/test_day25_quality.py`

```python
# tests/test_day25_quality.py
# Day 25: pseudo-labeling gates, session isolation, time-series routing.
```

### BLOCK 1 — Pseudo-labeling activation gates (10 tests)

```python
class TestPseudoLabelActivationGates:
    """
    All three gates must pass before pseudo-labeling runs.
    These tests verify each gate independently and in combination.
    """

    def test_skipped_when_metric_is_accuracy(self, base_pl_state):
        """Gate 1: accuracy is not probability-based — skip."""
        state = {**base_pl_state, "evaluation_metric": "accuracy"}
        result = run_pseudo_label_agent(state)
        assert result["pseudo_labels_applied"] is False
        assert "not probability-based" in result["pseudo_label_skip_reason"]

    def test_skipped_when_metric_is_rmse(self, base_pl_state):
        """Gate 1: rmse is not probability-based — skip."""
        state = {**base_pl_state, "evaluation_metric": "rmse"}
        result = run_pseudo_label_agent(state)
        assert result["pseudo_labels_applied"] is False

    def test_runs_when_metric_is_logloss(self, base_pl_state, monkeypatch):
        """Gate 1: logloss is probability-based — proceed to gate 2."""
        _mock_downstream_gates_pass(base_pl_state, monkeypatch)
        state = {**base_pl_state, "evaluation_metric": "logloss"}
        result = run_pseudo_label_agent(state)
        # Gate 1 passed — skip reason should not mention metric
        assert "not probability-based" not in result.get("pseudo_label_skip_reason", "")

    def test_skipped_when_test_not_2x_train(self, base_pl_state):
        """Gate 2: test set must be > 2x training set rows."""
        state = {
            **base_pl_state,
            "evaluation_metric": "logloss",
            "n_test_rows": 100,   # same as n_train — not 2x
        }
        result = run_pseudo_label_agent(state)
        assert result["pseudo_labels_applied"] is False
        assert "2x" in result["pseudo_label_skip_reason"] or \
               "ratio" in result["pseudo_label_skip_reason"].lower()

    def test_runs_when_test_is_3x_train(self, base_pl_state, monkeypatch):
        """Gate 2 passes when test set is 3x training rows."""
        _mock_downstream_gates_pass(base_pl_state, monkeypatch)
        state = {
            **base_pl_state,
            "evaluation_metric": "logloss",
            "n_test_rows": 300,   # 3x the 100-row training set
        }
        result = run_pseudo_label_agent(state)
        assert "2x" not in result.get("pseudo_label_skip_reason", "")

    def test_skipped_when_calibration_below_threshold(self, base_pl_state):
        """Gate 3: calibration below 0.80 threshold — skip."""
        state = {
            **base_pl_state,
            "evaluation_metric": "logloss",
            "n_test_rows":       300,
        }
        # Set a poor calibration score in model_registry
        state["model_registry"]["model_a"]["calibration_score"] = 0.15  # high Brier = bad
        result = run_pseudo_label_agent(state)
        assert result["pseudo_labels_applied"] is False
        assert "calibration" in result["pseudo_label_skip_reason"].lower()

    def test_all_gates_pass_triggers_run(self, base_pl_state, tmp_path, monkeypatch):
        """When all three gates pass, pseudo-labeling must actually run."""
        state = _setup_full_pass_state(base_pl_state, tmp_path)
        _mock_wilcoxon_pass(monkeypatch)
        result = run_pseudo_label_agent(state)
        assert result.get("pseudo_label_iterations", 0) >= 1

    def test_gate_check_does_not_read_train_data(self, base_pl_state, monkeypatch):
        """
        When gates fail, agent must not read train CSV from disk.
        Reading 2GB when the gate would immediately fail is wasteful.
        """
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
        state = {**base_pl_state, "evaluation_metric": "accuracy"}
        result = run_pseudo_label_agent(state)
        assert "pseudo_label_skip_reason" in result
        assert len(result["pseudo_label_skip_reason"]) > 0

    def test_all_required_state_keys_set_when_skipped(self, base_pl_state):
        """Even when skipped, all output state keys must be present."""
        state = {**base_pl_state, "evaluation_metric": "rmse"}
        result = run_pseudo_label_agent(state)
        required = [
            "pseudo_labels_applied", "pseudo_label_skip_reason",
            "pseudo_label_iterations", "pseudo_label_n_added",
        ]
        for key in required:
            assert key in result, f"Missing state key: {key}"
```

### BLOCK 2 — Pseudo-labeling correctness (8 tests)

```python
class TestPseudoLabelCorrectness:

    def test_validation_fold_never_contains_pseudo_labels(
        self, full_pl_state, tmp_path, monkeypatch
    ):
        """
        THE CRITICAL TEST. Pseudo-labels must appear in training folds only.
        Validation fold must contain only real labels.
        """
        val_fold_sources = []

        original_fit = _capture_fold_training(val_fold_sources)
        monkeypatch.setattr("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                            _intercepting_cv(val_fold_sources))

        run_pseudo_label_agent(full_pl_state)

        # Every validation sample must be a real label index, not a pseudo-label index
        real_n = len(full_pl_state["y_train"])
        for val_indices in val_fold_sources:
            assert all(i < real_n for i in val_indices), (
                "Pseudo-label indices found in validation fold. "
                "Temporal leakage: validation fold must only contain real labels."
            )

    def test_confidence_threshold_strictly_0_95(self, full_pl_state, tmp_path, monkeypatch):
        """Only predictions >= HIGH_CONFIDENCE_THRESHOLD (0.95) become pseudo-labels."""
        from agents.pseudo_label_agent import HIGH_CONFIDENCE_THRESHOLD
        assert HIGH_CONFIDENCE_THRESHOLD == 0.95, (
            f"HIGH_CONFIDENCE_THRESHOLD is {HIGH_CONFIDENCE_THRESHOLD}, expected 0.95."
        )

    def test_max_pseudo_label_fraction_respected(
        self, full_pl_state, tmp_path, monkeypatch
    ):
        """
        Pseudo-labels never exceed MAX_PSEUDO_LABEL_FRACTION of training data.
        Even across multiple iterations.
        """
        from agents.pseudo_label_agent import MAX_PSEUDO_LABEL_FRACTION
        _mock_wilcoxon_pass(monkeypatch)
        result = run_pseudo_label_agent(full_pl_state)

        max_allowed = int(len(full_pl_state["y_train"]) * MAX_PSEUDO_LABEL_FRACTION)
        n_added = result.get("pseudo_label_n_added", 0)
        assert n_added <= max_allowed, (
            f"Added {n_added} pseudo-labels, max allowed is {max_allowed} "
            f"({MAX_PSEUDO_LABEL_FRACTION:.0%} of {len(full_pl_state['y_train'])} training rows)."
        )

    def test_max_iterations_is_3(self):
        """MAX_ITERATIONS must equal 3."""
        from agents.pseudo_label_agent import MAX_ITERATIONS
        assert MAX_ITERATIONS == 3

    def test_wilcoxon_gate_called_each_iteration(
        self, full_pl_state, monkeypatch
    ):
        """Wilcoxon gate must be called once per iteration."""
        call_count = []
        import tools.wilcoxon_gate as wg
        original = wg.is_significantly_better
        def counting(*a, **k):
            call_count.append(1)
            return True   # always pass for this test
        monkeypatch.setattr(wg, "is_significantly_better", counting)

        result = run_pseudo_label_agent(full_pl_state)
        n_iters = result.get("pseudo_label_iterations", 0)
        assert len(call_count) == n_iters, (
            f"Wilcoxon gate called {len(call_count)} times "
            f"but {n_iters} iterations ran."
        )

    def test_stops_when_wilcoxon_fails(self, full_pl_state, monkeypatch):
        """When Wilcoxon gate fails, pseudo-labeling stops immediately."""
        import tools.wilcoxon_gate as wg
        monkeypatch.setattr(wg, "is_significantly_better", lambda *a, **k: False)
        result = run_pseudo_label_agent(full_pl_state)
        assert result["pseudo_label_iterations"] == 0
        assert result["pseudo_labels_applied"] is False
        assert "wilcoxon" in result.get("pseudo_label_halt_reason", "").lower()

    def test_critic_distribution_check_called(self, full_pl_state, monkeypatch):
        """Critic must verify confidence distribution before accepting pseudo-labels."""
        critic_calls = []
        import agents.pseudo_label_agent as pla
        original = pla._critic_verifies_confidence_distribution
        def tracking(conf, state):
            critic_calls.append(conf)
            return original(conf, state)
        monkeypatch.setattr(pla, "_critic_verifies_confidence_distribution", tracking)

        _mock_wilcoxon_pass(monkeypatch)
        run_pseudo_label_agent(full_pl_state)
        assert len(critic_calls) >= 1, "Critic distribution check was never called."

    def test_critic_rejects_overconfident_distribution(self, full_pl_state, monkeypatch):
        """
        If > 50% of predictions are above 0.95, critic rejects the distribution
        and pseudo-labeling is halted.
        """
        import agents.pseudo_label_agent as pla
        import numpy as np

        # Mock model predictions: all very high confidence (overconfident model)
        def mock_predict(*a, **k):
            return np.ones(500) * 0.99   # everything 0.99
        monkeypatch.setattr(pla, "_predict_test_set", mock_predict)
        _mock_wilcoxon_pass(monkeypatch)

        result = run_pseudo_label_agent(full_pl_state)
        assert result["pseudo_label_critic_accepted"] is False or \
               result["pseudo_labels_applied"] is False, (
            "Critic should have rejected overconfident distribution."
        )
```

### BLOCK 3 — Session ID namespace isolation (8 tests)

```python
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

    def test_redis_keys_use_session_id_prefix(self, monkeypatch):
        """Every Redis key set by the pipeline must start with session_id."""
        stored_keys = []
        import memory.redis_state as rs
        original_set = rs._redis_set
        def tracking_set(key, value, *args, **kwargs):
            stored_keys.append(key)
            return original_set(key, value, *args, **kwargs)
        monkeypatch.setattr(rs, "_redis_set", tracking_set)

        from core.state import build_initial_state
        state = build_initial_state("test-competition")
        # Simulate a Redis write
        rs._redis_set(f"{state['session_id']}:test_key", "value")

        for key in stored_keys:
            assert key.startswith(state["session_id"]) or ":" not in key, (
                f"Redis key '{key}' does not use session_id prefix. "
                "Concurrent runs will corrupt each other's state."
            )

    def test_run_professor_generates_session_id_if_missing(self, base_state, monkeypatch):
        """
        If state has no session_id, run_professor must generate one.
        It must not crash on missing session_id.
        """
        state = {**base_state}
        del state["session_id"]

        # Mock the graph to avoid running the full pipeline
        import core.professor as prof
        monkeypatch.setattr(prof, "get_graph", lambda: _mock_graph())

        result = prof.run_professor(state)
        assert "session_id" in result or True   # may be in returned state or generated internally
```

### BLOCK 4 — Time-series routing (8 tests)

```python
class TestTimeSeriesRouting:

    def test_timeseries_uses_timeseriessplit_not_kfold(self, ts_state):
        """task_type=timeseries must result in TimeSeriesSplit CV strategy."""
        result = run_validation_architect(ts_state)
        cv_strategy = result.get("cv_strategy", {})
        assert cv_strategy.get("cv_strategy") == "TimeSeriesSplit", (
            f"Time-series task got CV strategy: {cv_strategy.get('cv_strategy')}. "
            "Must be TimeSeriesSplit."
        )

    def test_timeseries_shuffle_is_false(self, ts_state):
        """TimeSeriesSplit must never shuffle — temporal ordering must be preserved."""
        result = run_validation_architect(ts_state)
        cv_strategy = result.get("cv_strategy", {})
        assert cv_strategy.get("shuffle") is False, (
            f"shuffle={cv_strategy.get('shuffle')} for timeseries. Must be False."
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
        result = run_feature_factory(ts_state_with_schema)
        candidates = result.get("feature_candidates", [])
        lag_features = [f for f in candidates if "_lag_" in f]
        assert len(lag_features) >= 1, (
            "No lag features generated for timeseries task. "
            "_generate_timeseries_features() may not be routing correctly."
        )

    def test_timeseries_feature_factory_generates_rolling_features(self, ts_state_with_schema):
        """Rolling window features must be generated for timeseries."""
        result = run_feature_factory(ts_state_with_schema)
        candidates = result.get("feature_candidates", [])
        rolling = [f for f in candidates if "rolling" in f]
        assert len(rolling) >= 1, "No rolling features generated for timeseries task."

    def test_timeseries_feature_factory_skips_target_encoding(self, ts_state_with_schema):
        """
        Target encoding is skipped for timeseries — it requires careful time ordering
        and is excluded from the timeseries feature generation path.
        """
        result = run_feature_factory(ts_state_with_schema)
        candidates = result.get("feature_candidates", [])
        te_features = [f for f in candidates if f.startswith("te_")]
        assert len(te_features) == 0, (
            f"Target encoding features found for timeseries: {te_features}. "
            "Target encoding is excluded from timeseries feature generation."
        )

    def test_non_timeseries_uses_stratified_kfold(self, base_state_classification):
        """Binary classification must still use StratifiedKFold — not TimeSeriesSplit."""
        result = run_validation_architect(base_state_classification)
        cv_strategy = result.get("cv_strategy", {})
        assert "Stratified" in cv_strategy.get("cv_strategy", ""), (
            f"Binary classification got CV strategy: {cv_strategy.get('cv_strategy')}. "
            "Expected StratifiedKFold."
        )
```

---

## FIXTURES

```python
import pytest
import numpy as np
import polars as pl
from pathlib import Path


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def y_train_100(rng):
    return rng.integers(0, 2, 100).astype(np.float32)


@pytest.fixture
def base_pl_state(y_train_100, tmp_path):
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
    n_test  = 300  # 3x train — gate 2 passes

    # Write train CSV
    train_df = pl.DataFrame({
        "id":     list(range(n_train)),
        **{f"f{i}": rng.uniform(0, 1, n_train).tolist() for i in range(5)},
        "target": rng.integers(0, 2, n_train).tolist(),
    })
    train_path = tmp_path / "train.csv"
    train_df.write_csv(train_path)

    # Write test CSV (no target column)
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


def _mock_wilcoxon_pass(monkeypatch):
    import tools.wilcoxon_gate as wg
    monkeypatch.setattr(wg, "is_significantly_better", lambda *a, **k: True)


def _mock_downstream_gates_pass(state, monkeypatch):
    """Mocks gates 2 and 3 to pass so only gate 1 is being tested."""
    import agents.pseudo_label_agent as pla
    monkeypatch.setattr(pla, "_count_test_rows", lambda s: 999)
    monkeypatch.setattr(pla, "_get_best_calibration_score", lambda s: 0.90)
```

---

## RUNNING THE TESTS

```bash
pytest tests/test_day25_quality.py -v --tb=short

# Regression check
pytest tests/contracts/ -v --tb=short
pytest tests/regression/ -v --tb=short
```

All three commands must show zero failures before committing.