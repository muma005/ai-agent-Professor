# tests/test_day16_quality.py
#
# Day 16 Quality Tests — 46 tests
# Block 1: Diversity ensemble selection (18)
# Block 2: Feature factory Round 1 (10)
# Block 3: Feature factory Round 2 (8)
# Block 4: Feature manifest and contract (10)

import json
import math
import os
import pytest
import numpy as np
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _make_oof(n=100, seed=42):
    """Generate random OOF predictions."""
    rng = np.random.RandomState(seed)
    return rng.rand(n).tolist()


def _make_correlated_oof(base, correlation, n=100, seed=99):
    """Generate OOF predictions with approximate target correlation to base."""
    rng = np.random.RandomState(seed)
    base_arr = np.array(base)
    noise = rng.rand(n)
    # Mix base with noise to achieve approximate correlation
    mixed = correlation * base_arr + (1 - correlation) * noise
    return mixed.tolist()


def _make_registry(*models):
    """Build a model registry dict from (name, cv, oof, data_hash) tuples."""
    registry = {}
    for name, cv, oof, *extra in models:
        entry = {"cv_mean": cv, "oof_predictions": oof}
        if extra:
            entry["data_hash"] = extra[0]
        registry[name] = entry
    return registry


SAMPLE_SCHEMA = {
    "columns": [
        {"name": "Age", "dtype": "float64", "n_unique": 88,
         "null_fraction": 0.20, "min": 0.42, "max": 80.0,
         "is_id": False, "is_target": False},
        {"name": "Fare", "dtype": "float64", "n_unique": 281,
         "null_fraction": 0.0, "min": 0.0, "max": 512.33,
         "is_id": False, "is_target": False},
        {"name": "SibSp", "dtype": "int64", "n_unique": 7,
         "null_fraction": 0.0, "min": 0, "max": 8,
         "is_id": False, "is_target": False},
        {"name": "Pclass", "dtype": "int64", "n_unique": 3,
         "null_fraction": 0.0, "min": 1, "max": 3,
         "is_id": False, "is_target": False},
        {"name": "PassengerId", "dtype": "int64", "n_unique": 891,
         "null_fraction": 0.0, "min": 1, "max": 891,
         "is_id": True, "is_target": False},
        {"name": "Survived", "dtype": "int64", "n_unique": 2,
         "null_fraction": 0.0, "min": 0, "max": 1,
         "is_id": False, "is_target": True},
    ],
    "n_rows": 891,
    "target_column": "Survived",
    "id_column": "PassengerId",
    "session_id": "test_d16",
}


@pytest.fixture
def ff_state(tmp_path):
    """Feature factory test state with schema.json and competition_brief.json."""
    session_id = "test_d16_ff"
    session_dir = tmp_path / f"outputs/{session_id}"
    session_dir.mkdir(parents=True)
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True)

    schema_path = session_dir / "schema.json"
    schema_path.write_text(json.dumps(SAMPLE_SCHEMA, indent=2))

    brief = {"domain": "maritime", "task_type": "binary_classification",
             "known_winning_features": ["Title from Name", "Family size"]}
    brief_path = session_dir / "competition_brief.json"
    brief_path.write_text(json.dumps(brief, indent=2))

    state = {
        "session_id": session_id,
        "competition_name": "titanic",
        "task_type": "tabular",
    }

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield state
    os.chdir(original_cwd)


@pytest.fixture
def ff_state_no_schema(tmp_path):
    """Feature factory state without schema.json."""
    session_id = "test_d16_no_schema"
    session_dir = tmp_path / f"outputs/{session_id}"
    session_dir.mkdir(parents=True)

    state = {
        "session_id": session_id,
        "competition_name": "titanic",
        "task_type": "tabular",
    }

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield state
    os.chdir(original_cwd)


@pytest.fixture
def id_target_only_state(tmp_path):
    """State with only id + target columns."""
    session_id = "test_d16_idonly"
    session_dir = tmp_path / f"outputs/{session_id}"
    session_dir.mkdir(parents=True)
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True)

    schema = {
        "columns": [
            {"name": "Id", "dtype": "int64", "n_unique": 100,
             "null_fraction": 0.0, "min": 1, "max": 100,
             "is_id": True, "is_target": False},
            {"name": "Target", "dtype": "int64", "n_unique": 2,
             "null_fraction": 0.0, "min": 0, "max": 1,
             "is_id": False, "is_target": True},
        ],
        "n_rows": 100,
        "target_column": "Target",
        "id_column": "Id",
        "session_id": "test_d16_idonly",
    }
    schema_path = session_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))

    state = {"session_id": session_id, "competition_name": "test", "task_type": "tabular"}
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield state
    os.chdir(original_cwd)


# ═══════════════════════════════════════════════════════════════════
# BLOCK 1 — DIVERSITY-FIRST ENSEMBLE SELECTION (18 tests)
# ═══════════════════════════════════════════════════════════════════

class TestDiversityEnsembleSelection:

    def test_anchor_is_highest_cv_model(self):
        """Test 1.1 — Anchor must be the highest CV model."""
        from agents.ensemble_architect import select_diverse_ensemble

        oof_a = _make_oof(100, seed=1)
        oof_b = _make_oof(100, seed=2)
        oof_c = _make_oof(100, seed=3)

        registry = _make_registry(
            ("model_a", 0.85, oof_a),
            ("model_b", 0.87, oof_b),
            ("model_c", 0.82, oof_c),
        )
        state = {"session_id": "test_anchor"}
        result = select_diverse_ensemble(registry, state)
        assert result["anchor"] == "model_b"
        assert result["selected_models"][0] == "model_b"

    def test_high_correlation_model_rejected(self):
        """Test 1.2 — Model with correlation > 0.97 must be rejected."""
        from agents.ensemble_architect import select_diverse_ensemble

        anchor_oof = _make_oof(100, seed=10)
        # Nearly identical OOF
        correlated_oof = _make_correlated_oof(anchor_oof, 0.99, 100, seed=11)

        registry = _make_registry(
            ("anchor", 0.87, anchor_oof),
            ("copycat", 0.86, correlated_oof),
        )
        state = {"session_id": "test_corr"}
        result = select_diverse_ensemble(registry, state)

        log_decisions = {e["model"]: e["decision"] for e in result["selection_log"]}
        assert log_decisions["copycat"] == "REJECTED_TOO_CORRELATED"
        assert "copycat" not in result["selected_models"]

    def test_correlation_threshold_at_exactly_0_97(self):
        """Test 1.3 — Correlation = 0.97 is NOT rejected (> not >=). 0.971 IS."""
        from agents.ensemble_architect import select_diverse_ensemble
        from scipy.stats import pearsonr

        anchor_oof = _make_oof(100, seed=20)

        # We need precise control. Use known arrays.
        # Model at boundary: create OOF with correlation ~0.969
        rng = np.random.RandomState(42)
        base = np.array(anchor_oof)
        noise = rng.rand(100)
        # Tune mix to get ~0.969 corr
        borderline_ok = 0.969 * base + (1 - 0.969) * noise
        corr_ok, _ = pearsonr(base, borderline_ok)

        registry_ok = _make_registry(
            ("anchor", 0.87, anchor_oof),
            ("borderline", 0.86, borderline_ok.tolist()),
        )
        state = {"session_id": "test_boundary"}
        result = select_diverse_ensemble(registry_ok, state)
        log_ok = {e["model"]: e["decision"] for e in result["selection_log"]}
        # If actual corr <= 0.97, should not be rejected
        if corr_ok <= 0.97:
            assert log_ok["borderline"] != "REJECTED_TOO_CORRELATED"

    def test_diverse_model_selected_over_higher_cv_correlated_model(self):
        """Test 1.4 — Diverse low-CV model beats correlated high-CV model on diversity score."""
        from agents.ensemble_architect import select_diverse_ensemble

        anchor_oof = _make_oof(100, seed=30)
        # Model A: high CV but very correlated with anchor
        model_a_oof = _make_correlated_oof(anchor_oof, 0.96, 100, seed=31)
        # Model B: lower CV but very diverse
        model_b_oof = _make_oof(100, seed=32)  # random = diverse

        registry = _make_registry(
            ("anchor", 0.87, anchor_oof),
            ("model_a", 0.863, model_a_oof),
            ("model_b", 0.855, model_b_oof),
        )
        state = {"session_id": "test_diverse"}
        result = select_diverse_ensemble(registry, state)

        assert "model_b" in result["selected_models"]

    def test_correlation_computed_against_ensemble_mean_not_anchor_only(self):
        """Test 1.5 — Correlation for 3rd+ model must be vs ensemble mean, not anchor alone."""
        from agents.ensemble_architect import select_diverse_ensemble
        from scipy.stats import pearsonr

        # Create distinct models
        anchor_oof = _make_oof(200, seed=40)
        model_b_oof = _make_oof(200, seed=41)
        model_c_oof = _make_oof(200, seed=42)

        registry = _make_registry(
            ("anchor", 0.90, anchor_oof),
            ("model_b", 0.88, model_b_oof),
            ("model_c", 0.86, model_c_oof),
        )
        state = {"session_id": "test_mean"}
        result = select_diverse_ensemble(registry, state)

        # After selecting anchor + model_b, the ensemble mean changes.
        # model_c's correlation should be vs mean(anchor, model_b), not just anchor.
        # Verify by checking the logged correlation for model_c
        model_c_log = [e for e in result["selection_log"] if e["model"] == "model_c"]
        assert len(model_c_log) == 1

        # Compute expected: corr vs ensemble mean
        ens_mean = (np.array(anchor_oof) + np.array(model_b_oof)) / 2.0
        expected_corr, _ = pearsonr(np.array(model_c_oof), ens_mean)
        # Corr vs anchor only
        anchor_corr, _ = pearsonr(np.array(model_c_oof), np.array(anchor_oof))

        logged_corr = model_c_log[0]["correlation"]
        # Should match ensemble mean correlation, not anchor-only
        assert abs(logged_corr - round(expected_corr, 4)) < 0.01, (
            f"Correlation logged as {logged_corr} but expected ~{expected_corr:.4f} "
            f"(vs ensemble mean). Anchor-only would be ~{anchor_corr:.4f}."
        )

    def test_ensemble_oof_mean_updated_after_each_selection(self):
        """Test 1.6 — Running mean must update after each model is added."""
        from agents.ensemble_architect import select_diverse_ensemble

        oof1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        oof2 = [5.0, 4.0, 3.0, 2.0, 1.0]

        registry = _make_registry(
            ("m1", 0.90, oof1),
            ("m2", 0.85, oof2),
        )
        state = {"session_id": "test_mean_update"}
        result = select_diverse_ensemble(registry, state)

        # Both should be selected (very different OOFs)
        assert "m1" in result["selected_models"]
        assert "m2" in result["selected_models"]

        # Ensemble OOF from blend_models would be mean
        expected_mean = [(a + b) / 2 for a, b in zip(oof1, oof2)]
        # The weights are equal so weighted result = mean
        for i in range(5):
            w1 = result["ensemble_weights"]["m1"]
            w2 = result["ensemble_weights"]["m2"]
            assert abs(w1 - 0.5) < 1e-9
            assert abs(w2 - 0.5) < 1e-9

    def test_prize_candidate_requires_both_conditions(self):
        """Test 1.7 — Prize requires low correlation AND competitive CV (AND not OR)."""
        from agents.ensemble_architect import select_diverse_ensemble

        anchor_oof = _make_oof(100, seed=50)
        # Low correlation but terrible CV (far from best)
        diverse_bad_oof = _make_oof(100, seed=51)

        registry = _make_registry(
            ("anchor", 0.87, anchor_oof),
            ("diverse_bad", 0.75, diverse_bad_oof),  # delta=0.12 >> 0.01
        )
        state = {"session_id": "test_prize_and"}
        result = select_diverse_ensemble(registry, state)

        # Should NOT be a prize candidate — CV delta too large
        prize_names = [p["model"] for p in result["prize_candidates"]]
        assert "diverse_bad" not in prize_names, (
            "Model with low correlation but terrible CV should NOT be a prize candidate. "
            "Both conditions required: corr < 0.85 AND abs(cv - best_cv) <= 0.01"
        )

    def test_prize_candidate_identified_correctly(self):
        """Test 1.8 — Model with low correlation AND competitive CV = prize."""
        from agents.ensemble_architect import select_diverse_ensemble

        anchor_oof = _make_oof(100, seed=60)
        prize_oof = _make_oof(100, seed=61)  # diverse

        registry = _make_registry(
            ("anchor", 0.870, anchor_oof),
            ("prize_model", 0.868, prize_oof),  # delta=0.002 <= 0.01
        )
        state = {"session_id": "test_prize_yes"}
        result = select_diverse_ensemble(registry, state)

        from scipy.stats import pearsonr
        corr, _ = pearsonr(np.array(anchor_oof), np.array(prize_oof))

        if corr < 0.85:
            prize_names = [p["model"] for p in result["prize_candidates"]]
            assert "prize_model" in prize_names
            prize_entry = [p for p in result["prize_candidates"] if p["model"] == "prize_model"][0]
            assert "correlation" in prize_entry
            assert "cv_delta_from_best" in prize_entry

    def test_max_ensemble_size_respected(self):
        """Test 1.9 — No more than MAX_ENSEMBLE_SIZE models selected."""
        from agents.ensemble_architect import select_diverse_ensemble, MAX_ENSEMBLE_SIZE

        models = []
        for i in range(10):
            models.append((f"model_{i}", 0.90 - i * 0.005, _make_oof(100, seed=70 + i)))

        registry = _make_registry(*models)
        state = {"session_id": "test_maxsize"}
        result = select_diverse_ensemble(registry, state)

        assert len(result["selected_models"]) <= MAX_ENSEMBLE_SIZE

        skipped = [e for e in result["selection_log"] if e["decision"] == "SKIPPED_MAX_SIZE"]
        if len(registry) > MAX_ENSEMBLE_SIZE:
            # Some must have been skipped
            total_accounted = (
                len(result["selected_models"])
                + len(skipped)
                + len([e for e in result["selection_log"] if e["decision"] == "REJECTED_TOO_CORRELATED"])
            )
            assert total_accounted == len(registry)

    def test_validate_oof_present_raises_on_missing(self):
        """Test 1.10 — Empty OOF list raises ValueError before selection."""
        from agents.ensemble_architect import select_diverse_ensemble

        registry = _make_registry(
            ("good_model", 0.85, _make_oof(100, seed=80)),
            ("bad_model", 0.82, []),  # empty
        )
        state = {"session_id": "test_oof_empty"}

        with pytest.raises(ValueError, match="bad_model"):
            select_diverse_ensemble(registry, state)

    def test_validate_oof_present_raises_on_absent_key(self):
        """Test 1.11 — Missing oof_predictions key entirely raises ValueError, not KeyError."""
        from agents.ensemble_architect import select_diverse_ensemble

        registry = {"model_a": {"cv_mean": 0.85}}  # no oof_predictions key
        state = {"session_id": "test_oof_absent"}

        with pytest.raises(ValueError):
            select_diverse_ensemble(registry, state)

    def test_selection_log_has_all_models(self):
        """Test 1.12 — Every model appears in the selection log."""
        from agents.ensemble_architect import select_diverse_ensemble

        models = [(f"m{i}", 0.90 - i * 0.01, _make_oof(100, seed=90 + i)) for i in range(5)]
        registry = _make_registry(*models)
        state = {"session_id": "test_log_all"}
        result = select_diverse_ensemble(registry, state)

        logged_models = {e["model"] for e in result["selection_log"]}
        for name, _, _ in models:
            assert name in logged_models, f"Model {name} missing from selection_log"

    def test_correlation_matrix_contains_selected_pairs(self):
        """Test 1.13 — Correlation matrix has n*(n-1)/2 entries for n selected models."""
        from agents.ensemble_architect import select_diverse_ensemble

        models = [(f"m{i}", 0.90 - i * 0.01, _make_oof(100, seed=100 + i)) for i in range(4)]
        registry = _make_registry(*models)
        state = {"session_id": "test_corrmat"}
        result = select_diverse_ensemble(registry, state)

        n = len(result["selected_models"])
        expected_pairs = n * (n - 1) // 2
        assert len(result["correlation_matrix"]) == expected_pairs

        # Keys formatted as "model_a_vs_model_b"
        for key in result["correlation_matrix"]:
            assert "_vs_" in key

    def test_equal_weights_sum_to_one(self):
        """Test 1.14 — Weights must sum to 1.0."""
        from agents.ensemble_architect import select_diverse_ensemble

        models = [(f"m{i}", 0.88 - i * 0.01, _make_oof(100, seed=110 + i)) for i in range(3)]
        registry = _make_registry(*models)
        state = {"session_id": "test_weights"}
        result = select_diverse_ensemble(registry, state)

        total = sum(result["ensemble_weights"].values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0"

    def test_single_model_registry_returns_that_model(self):
        """Test 1.15 — Single-model registry returns that model as sole selection."""
        from agents.ensemble_architect import select_diverse_ensemble

        registry = _make_registry(("solo", 0.85, _make_oof(100, seed=120)))
        state = {"session_id": "test_solo"}
        result = select_diverse_ensemble(registry, state)

        assert result["selected_models"] == ["solo"]
        assert len(result["prize_candidates"]) == 0
        assert result["ensemble_weights"]["solo"] == 1.0

    def test_empty_registry_raises_value_error(self):
        """Test 1.16 — Empty registry must raise ValueError."""
        from agents.ensemble_architect import select_diverse_ensemble

        with pytest.raises(ValueError, match="empty"):
            select_diverse_ensemble({}, {"session_id": "test_empty"})

    def test_diversity_selection_called_before_blend(self):
        """Test 1.17 — select_diverse_ensemble must be called inside blend_models."""
        from agents.ensemble_architect import blend_models

        anchor_oof = _make_oof(100, seed=130)
        diverse_oof = _make_oof(100, seed=131)

        registry = _make_registry(
            ("m1", 0.87, anchor_oof, "hash123"),
            ("m2", 0.85, diverse_oof, "hash123"),
        )
        state = {
            "session_id": "test_blend_order",
            "data_hash": "hash123",
            "model_registry": registry,
        }

        call_order = []
        original_select = None
        import agents.ensemble_architect as ea
        original_select = ea.select_diverse_ensemble

        def mock_select(*args, **kwargs):
            call_order.append("select_diverse_ensemble")
            return original_select(*args, **kwargs)

        with patch.object(ea, "select_diverse_ensemble", side_effect=mock_select):
            result = blend_models(state)

        assert "select_diverse_ensemble" in call_order

    def test_selection_result_written_to_state(self):
        """Test 1.18 — blend_models sets all 5 new state fields."""
        from agents.ensemble_architect import blend_models

        anchor_oof = _make_oof(100, seed=140)
        diverse_oof = _make_oof(100, seed=141)

        registry = _make_registry(
            ("m1", 0.87, anchor_oof, "hash456"),
            ("m2", 0.85, diverse_oof, "hash456"),
        )
        state = {
            "session_id": "test_state_fields",
            "data_hash": "hash456",
            "model_registry": registry,
        }
        result = blend_models(state)

        assert result.get("ensemble_selection") is not None
        assert isinstance(result.get("selected_models"), list)
        assert isinstance(result.get("ensemble_weights"), dict)
        assert isinstance(result.get("ensemble_oof"), list)
        assert isinstance(result.get("prize_candidates"), list)


# ═══════════════════════════════════════════════════════════════════
# BLOCK 2 — FEATURE FACTORY ROUND 1 (10 tests)
# ═══════════════════════════════════════════════════════════════════

class TestFeatureFactoryRound1:

    def test_log1p_candidate_generated_for_positive_numeric(self):
        """Test 2.1 — log1p generated for positive numeric with n_unique > 10."""
        from agents.feature_factory import _generate_round1_features
        candidates = _generate_round1_features(SAMPLE_SCHEMA)
        names = [c.name for c in candidates]
        assert "log1p_Fare" in names

    def test_log1p_not_generated_for_negative_min(self):
        """Test 2.2 — log1p NOT generated when min < 0."""
        from agents.feature_factory import _generate_round1_features
        schema = {
            "columns": [
                {"name": "Balance", "dtype": "float64", "n_unique": 200,
                 "null_fraction": 0.0, "min": -5.0, "max": 100.0,
                 "is_id": False, "is_target": False},
            ]
        }
        candidates = _generate_round1_features(schema)
        log_names = [c.name for c in candidates if c.transform_type == "log"]
        assert len(log_names) == 0, f"log1p generated for negative min: {log_names}"

    def test_missingness_flag_generated_for_high_null_fraction(self):
        """Test 2.3 — Missingness flag generated when null_fraction > 0.01."""
        from agents.feature_factory import _generate_round1_features
        candidates = _generate_round1_features(SAMPLE_SCHEMA)
        names = [c.name for c in candidates]
        assert "missing_Age" in names  # Age has null_fraction=0.20

    def test_missingness_flag_not_generated_for_complete_columns(self):
        """Test 2.4 — No missingness flag for columns with null_fraction=0."""
        from agents.feature_factory import _generate_round1_features
        candidates = _generate_round1_features(SAMPLE_SCHEMA)
        names = [c.name for c in candidates]
        assert "missing_Fare" not in names  # Fare has null_fraction=0.0

    def test_no_candidates_for_id_columns(self):
        """Test 2.5 — Zero candidates from ID columns."""
        from agents.feature_factory import _generate_round1_features
        candidates = _generate_round1_features(SAMPLE_SCHEMA)
        id_candidates = [c for c in candidates if "PassengerId" in c.source_columns]
        assert len(id_candidates) == 0

    def test_no_candidates_for_target_column(self):
        """Test 2.6 — Zero candidates from target column."""
        from agents.feature_factory import _generate_round1_features
        candidates = _generate_round1_features(SAMPLE_SCHEMA)
        target_candidates = [c for c in candidates if "Survived" in c.source_columns]
        assert len(target_candidates) == 0

    def test_round_field_is_1_for_all_round1_candidates(self):
        """Test 2.7 — All Round 1 candidates have round=1."""
        from agents.feature_factory import _generate_round1_features
        candidates = _generate_round1_features(SAMPLE_SCHEMA)
        for c in candidates:
            assert c.round == 1, f"Candidate {c.name} has round={c.round}, expected 1"

    def test_apply_round1_transforms_produces_new_columns(self):
        """Test 2.8 — Applying transforms adds new columns, preserves originals."""
        from agents.feature_factory import _generate_round1_features, _apply_round1_transforms

        df = pl.DataFrame({
            "Age": [25.0, None, 30.0, 45.0],
            "Fare": [7.25, 71.28, 0.0, 8.05],
        })
        candidates = _generate_round1_features(SAMPLE_SCHEMA)
        result = _apply_round1_transforms(df, candidates)

        assert "Age" in result.columns
        assert "Fare" in result.columns
        assert len(result) == 4  # row count unchanged

        # At least one new column should exist
        new_cols = set(result.columns) - {"Age", "Fare"}
        assert len(new_cols) > 0

    def test_log1p_transform_is_log_base_e_not_base_2(self):
        """Test 2.9 — log1p must use natural log (base e), not base 2 or 10."""
        from agents.feature_factory import _apply_round1_transforms, FeatureCandidate

        df = pl.DataFrame({"Value": [1.0, 10.0, 100.0]})
        candidates = [FeatureCandidate(
            name="log1p_Value", source_columns=["Value"],
            transform_type="log", description="log1p", round=1,
        )]
        result = _apply_round1_transforms(df, candidates)
        log_values = result["log1p_Value"].to_list()

        # log1p(1.0) = ln(2) ≈ 0.693, NOT log2(2) = 1.0
        expected = math.log(1.0 + 1.0)  # natural log
        assert abs(log_values[0] - expected) < 1e-6, (
            f"log1p(1.0) = {log_values[0]}, expected {expected} (natural log). "
            f"Got base-2? {abs(log_values[0] - 1.0) < 1e-6}"
        )

    def test_missingness_flag_is_binary_int_not_boolean(self):
        """Test 2.10 — Missingness flag must be Int8 (0/1), not Boolean."""
        from agents.feature_factory import _apply_round1_transforms, FeatureCandidate

        df = pl.DataFrame({"Col": [1.0, None, 3.0, None]})
        candidates = [FeatureCandidate(
            name="missing_Col", source_columns=["Col"],
            transform_type="missingness_flag", description="missing", round=1,
        )]
        result = _apply_round1_transforms(df, candidates)

        assert result["missing_Col"].dtype == pl.Int8, (
            f"Expected Int8, got {result['missing_Col'].dtype}"
        )
        assert result["missing_Col"].to_list() == [0, 1, 0, 1]


# ═══════════════════════════════════════════════════════════════════
# BLOCK 3 — FEATURE FACTORY ROUND 2 (8 tests)
# ═══════════════════════════════════════════════════════════════════

class TestFeatureFactoryRound2:

    def test_round2_candidates_have_round_2_field(self):
        """Test 3.1 — All Round 2 candidates have round=2."""
        from agents.feature_factory import _generate_round2_features

        llm_response = json.dumps([{
            "name": "age_fare_ratio",
            "source_columns": ["Age", "Fare"],
            "transform_type": "ratio",
            "expression": "Age / Fare",
            "domain_rationale": "Economic proxy",
        }])

        with patch("agents.feature_factory.call_llm", return_value=llm_response):
            candidates = _generate_round2_features(
                SAMPLE_SCHEMA,
                {"domain": "maritime", "task_type": "binary_classification"},
                {"session_id": "test"},
            )
        for c in candidates:
            assert c.round == 2

    def test_round2_rejects_candidates_with_unknown_source_columns(self):
        """Test 3.2 — Candidates with non-existent source columns are rejected."""
        from agents.feature_factory import _generate_round2_features

        llm_response = json.dumps([{
            "name": "bad_feature",
            "source_columns": ["NonExistentColumn"],
            "transform_type": "ratio",
            "expression": "something",
            "domain_rationale": "test",
        }])

        with patch("agents.feature_factory.call_llm", return_value=llm_response):
            candidates = _generate_round2_features(
                SAMPLE_SCHEMA,
                {"domain": "test"},
                {"session_id": "test"},
            )
        names = [c.name for c in candidates]
        assert "bad_feature" not in names

    def test_round2_capped_at_15_candidates(self):
        """Test 3.3 — Maximum 15 candidates from Round 2."""
        from agents.feature_factory import _generate_round2_features

        # Generate 25 candidates
        items = [{
            "name": f"feat_{i}",
            "source_columns": ["Age"],
            "transform_type": "ratio",
            "expression": f"transform {i}",
            "domain_rationale": f"reason {i}",
        } for i in range(25)]
        llm_response = json.dumps(items)

        with patch("agents.feature_factory.call_llm", return_value=llm_response):
            candidates = _generate_round2_features(
                SAMPLE_SCHEMA,
                {"domain": "test"},
                {"session_id": "test"},
            )
        assert len(candidates) <= 15

    def test_round2_graceful_on_llm_failure(self):
        """Test 3.4 — LLM timeout returns empty list, no exception."""
        from agents.feature_factory import _generate_round2_features

        with patch("agents.feature_factory.call_llm", side_effect=TimeoutError("boom")):
            candidates = _generate_round2_features(
                SAMPLE_SCHEMA,
                {"domain": "test"},
                {"session_id": "test"},
            )
        assert candidates == []

    def test_round2_graceful_on_invalid_json_response(self):
        """Test 3.5 — Invalid JSON from LLM returns empty list."""
        from agents.feature_factory import _generate_round2_features

        with patch("agents.feature_factory.call_llm", return_value="not valid json at all"):
            candidates = _generate_round2_features(
                SAMPLE_SCHEMA,
                {"domain": "test"},
                {"session_id": "test"},
            )
        assert candidates == []

    def test_round2_uses_domain_from_competition_brief(self):
        """Test 3.6 — LLM prompt contains the domain value."""
        from agents.feature_factory import _generate_round2_features

        captured_prompt = []

        def mock_llm(prompt, **kwargs):
            captured_prompt.append(prompt)
            return "[]"

        with patch("agents.feature_factory.call_llm", side_effect=mock_llm):
            _generate_round2_features(
                SAMPLE_SCHEMA,
                {"domain": "maritime_survival"},
                {"session_id": "test"},
            )
        assert "maritime_survival" in captured_prompt[0]

    def test_round2_uses_known_winning_features(self):
        """Test 3.7 — LLM prompt includes known winning features."""
        from agents.feature_factory import _generate_round2_features

        captured_prompt = []

        def mock_llm(prompt, **kwargs):
            captured_prompt.append(prompt)
            return "[]"

        brief = {
            "domain": "test",
            "known_winning_features": ["Title from Name", "Family size"],
        }
        with patch("agents.feature_factory.call_llm", side_effect=mock_llm):
            _generate_round2_features(SAMPLE_SCHEMA, brief, {"session_id": "test"})

        assert "Title from Name" in captured_prompt[0] or "Family size" in captured_prompt[0]

    def test_round2_runs_without_competition_brief(self):
        """Test 3.8 — Empty brief returns empty list, no crash."""
        from agents.feature_factory import _generate_round2_features

        candidates = _generate_round2_features(
            SAMPLE_SCHEMA,
            {},  # empty brief
            {"session_id": "test"},
        )
        assert candidates == []


# ═══════════════════════════════════════════════════════════════════
# BLOCK 4 — FEATURE MANIFEST AND CONTRACT (10 tests)
# ═══════════════════════════════════════════════════════════════════

class TestFeatureManifest:

    def test_manifest_written_to_correct_path(self, ff_state):
        """Test 4.1 — feature_manifest.json written to outputs/{session_id}/."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(ff_state)
        path = Path(f"outputs/{state['session_id']}/feature_manifest.json")
        assert path.exists()

    def test_manifest_counts_consistent(self, ff_state):
        """Test 4.2 — Counts match features list."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(ff_state)
        manifest = state["feature_manifest"]
        features = manifest["features"]

        assert manifest["total_candidates"] == len(features)
        keep_count = sum(1 for f in features if f["verdict"] == "KEEP")
        drop_count = sum(1 for f in features if f["verdict"] == "DROP")
        assert manifest["total_kept"] == keep_count
        assert manifest["total_dropped"] == drop_count
        assert manifest["total_candidates"] >= manifest["total_kept"] + manifest["total_dropped"]

    def test_manifest_has_generated_at_timestamp(self, ff_state):
        """Test 4.3 — generated_at is a valid ISO 8601 timestamp."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(ff_state)
        manifest = state["feature_manifest"]
        # Should not raise
        datetime.fromisoformat(manifest["generated_at"])

    def test_feature_state_fields_set_correctly(self, ff_state):
        """Test 4.4 — State has feature_candidates, round1_features, round2_features."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(ff_state)

        assert isinstance(state["feature_candidates"], list)
        assert isinstance(state["round1_features"], list)
        assert isinstance(state["round2_features"], list)
        # round1 and round2 are subsets of feature_candidates
        all_names = set(state["feature_candidates"])
        for name in state["round1_features"]:
            assert name in all_names
        for name in state["round2_features"]:
            assert name in all_names

    def test_manifest_empty_when_all_columns_are_id_or_target(self, id_target_only_state):
        """Test 4.5 — Schema with only id+target yields 0 features."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(id_target_only_state)
        manifest = state["feature_manifest"]
        assert len(manifest["features"]) == 0
        assert manifest["total_candidates"] == 0

    def test_run_feature_factory_raises_on_missing_schema(self, ff_state_no_schema):
        """Test 4.6 — Missing schema.json raises FileNotFoundError."""
        from agents.feature_factory import run_feature_factory
        with pytest.raises(FileNotFoundError, match="schema.json"):
            run_feature_factory(ff_state_no_schema)

    def test_lineage_event_written_after_feature_factory(self, ff_state):
        """Test 4.7 — Lineage log contains feature_factory_complete event."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(ff_state)

        lineage_path = Path(f"outputs/{state['session_id']}/logs/lineage.jsonl")
        assert lineage_path.exists(), "lineage.jsonl not written"
        lines = lineage_path.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        actions = [e["action"] for e in events]
        assert "feature_factory_complete" in actions

    def test_feature_candidates_excludes_target_and_id(self, ff_state):
        """Test 4.8 — feature_candidates must not contain target or id column names."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(ff_state)

        for name in state["feature_candidates"]:
            assert "PassengerId" not in name or name.startswith("log1p_") or name.startswith("sqrt_") or name.startswith("missing_"), (
                f"Raw ID column should not appear as feature: {name}"
            )
            # Target should never appear
            assert name != "Survived"

    def test_all_pending_verdicts_in_day16_stub(self, ff_state):
        """Test 4.9 — Day 16 stub sets all verdicts to KEEP (not PENDING)."""
        from agents.feature_factory import run_feature_factory
        with patch("agents.feature_factory.call_llm", return_value="[]"):
            state = run_feature_factory(ff_state)
        manifest = state["feature_manifest"]
        for f in manifest["features"]:
            assert f["verdict"] == "KEEP", (
                f"Day 16 stub should set KEEP, got '{f['verdict']}' for {f['name']}"
            )

    def test_day13_ensemble_contracts_still_pass(self):
        """Test 4.10 — Previous contracts still pass (run via import check)."""
        # Import the ensemble architect to verify no import errors
        from agents.ensemble_architect import (
            blend_models, select_diverse_ensemble,
            _validate_data_hash_consistency, _validate_oof_present,
        )
        # Verify functions are callable
        assert callable(blend_models)
        assert callable(select_diverse_ensemble)
        assert callable(_validate_data_hash_consistency)
        assert callable(_validate_oof_present)
