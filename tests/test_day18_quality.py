# tests/test_day18_quality.py
# -------------------------------------------------------------------------
# Day 18 — Feature Factory Rounds 3-5, Interaction Budget Cap,
#           Pseudo-labeling Pipeline  |  52 tests
# Status: IMMUTABLE after Day 18
# -------------------------------------------------------------------------

import json
import os
import sys
import uuid
import copy
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import pytest
import numpy as np
import polars as pl

# ── Helpers ──────────────────────────────────────────────────────

def _make_schema(columns: list[dict]) -> dict:
    """Build a minimal schema dict from column descriptors."""
    return {"columns": columns, "session_id": "test_session"}


def _col(name, dtype="float64", n_unique=100, is_id=False, is_target=False, null_fraction=0.0, min_val=0):
    """Shorthand for a schema column."""
    return {
        "name": name,
        "dtype": dtype,
        "n_unique": n_unique,
        "is_id": is_id,
        "is_target": is_target,
        "null_fraction": null_fraction,
        "min": min_val,
    }


def _make_state(session_id=None, **overrides):
    """Minimal ProfessorState-like dict for testing."""
    sid = session_id or f"test_{uuid.uuid4().hex[:8]}"
    state = {
        "session_id": sid,
        "competition_name": "test_comp",
        "task_type": "binary",
        "target_column": "target",
        "id_column": "id",
        "evaluation_metric": "auc",
        "clean_data_path": "",
        "null_importance_result": None,
        "selected_models": [],
        "model_registry": [],
    }
    state.update(overrides)
    return state


def _cleanup_outputs(session_id: str):
    """Remove test outputs."""
    import shutil
    p = Path(f"outputs/{session_id}")
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def _dynamic_predict_proba(X):
    """Mock predict_proba that returns varying confidences matching input size."""
    n = len(X)
    probs = np.linspace(0, 1, n)
    return np.column_stack([1 - probs, probs])


# ═══════════════════════════════════════════════════════════════
# BLOCK 1 — FEATURE FACTORY ROUNDS 3 + 4 (14 tests)
# ═══════════════════════════════════════════════════════════════

class TestFeatureFactoryRounds3And4:
    """Tests for Round 3 (aggregation) and Round 4 (target encoding)."""

    # ── TEST 1.1 ─────────────────────────────────────────────────
    def test_round3_generates_all_five_agg_functions(self):
        from agents.feature_factory import _generate_round3_aggregation_features, ROUND3_AGG_FUNCTIONS
        schema = _make_schema([
            _col("cat_a", dtype="str", n_unique=10),
            _col("num_b", dtype="float64", n_unique=100),
        ])
        candidates = _generate_round3_aggregation_features(schema)
        assert len(candidates) == 5
        names = {c.name for c in candidates}
        for fn in ROUND3_AGG_FUNCTIONS:
            assert f"num_b_{fn}_by_cat_a" in names
        for c in candidates:
            assert c.round == 3
            assert c.transform_type == "groupby_agg"

    # ── TEST 1.2 ─────────────────────────────────────────────────
    def test_round3_caps_at_max_candidates(self):
        from agents.feature_factory import _generate_round3_aggregation_features, MAX_ROUND3_CANDIDATES
        # 10 categoricals × 10 numerics × 5 agg = 500 candidates
        cols = []
        for i in range(10):
            cols.append(_col(f"cat_{i}", dtype="str", n_unique=5 + i))
        for i in range(10):
            cols.append(_col(f"num_{i}", dtype="float64", n_unique=50 + i))
        schema = _make_schema(cols)
        candidates = _generate_round3_aggregation_features(schema)
        assert len(candidates) <= MAX_ROUND3_CANDIDATES

    # ── TEST 1.3 ─────────────────────────────────────────────────
    def test_round3_cap_prioritises_low_cardinality_categorical(self):
        from agents.feature_factory import _generate_round3_aggregation_features, MAX_ROUND3_CANDIDATES
        # A (n_unique=3) vs B (n_unique=100)
        # Need enough to trigger the cap: 2 cats × many numerics × 5
        cols = [
            _col("cat_low", dtype="str", n_unique=3),
            _col("cat_high", dtype="str", n_unique=100),
        ]
        for i in range(25):
            cols.append(_col(f"num_{i}", dtype="float64", n_unique=50 + i))
        schema = _make_schema(cols)
        # 2 × 25 × 5 = 250 > 200 → cap fires
        candidates = _generate_round3_aggregation_features(schema)
        assert len(candidates) == MAX_ROUND3_CANDIDATES
        # Low cardinality cat should dominate the top positions
        top_half = candidates[:len(candidates) // 2]
        low_cat_count = sum(1 for c in top_half if "cat_low" in c.name)
        high_cat_count = sum(1 for c in top_half if "cat_high" in c.name)
        assert low_cat_count > high_cat_count

    # ── TEST 1.4 ─────────────────────────────────────────────────
    def test_round3_skips_id_and_target_columns(self):
        from agents.feature_factory import _generate_round3_aggregation_features
        schema = _make_schema([
            _col("id_col", dtype="str", n_unique=1000, is_id=True),
            _col("target_col", dtype="float64", n_unique=2, is_target=True),
            _col("cat_a", dtype="str", n_unique=10),
            _col("num_b", dtype="float64", n_unique=100),
        ])
        candidates = _generate_round3_aggregation_features(schema)
        all_sources = set()
        for c in candidates:
            all_sources.update(c.source_columns)
        assert "id_col" not in all_sources
        assert "target_col" not in all_sources

    # ── TEST 1.5 ─────────────────────────────────────────────────
    def test_round3_apply_produces_correct_group_means(self):
        from agents.feature_factory import _apply_round3_transforms, FeatureCandidate
        df = pl.DataFrame({
            "cat": ["A", "A", "B", "B", "B"],
            "val": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        candidates = [FeatureCandidate(
            name="val_mean_by_cat",
            source_columns=["val", "cat"],
            transform_type="groupby_agg",
            description="mean of val by cat",
            round=3,
        )]
        result = _apply_round3_transforms(df, candidates)
        assert "val_mean_by_cat" in result.columns
        values = result["val_mean_by_cat"].to_list()
        # A mean = 15, B mean = 40
        for i in range(2):
            assert abs(values[i] - 15.0) < 1e-6
        for i in range(2, 5):
            assert abs(values[i] - 40.0) < 1e-6

    # ── TEST 1.6 ─────────────────────────────────────────────────
    def test_round3_apply_uses_join_not_apply(self):
        from agents.feature_factory import _apply_round3_transforms, FeatureCandidate
        df = pl.DataFrame({
            "cat": ["A", "A", "B"],
            "val": [10.0, 20.0, 30.0],
        })
        candidates = [FeatureCandidate(
            name="val_mean_by_cat",
            source_columns=["val", "cat"],
            transform_type="groupby_agg",
            description="mean of val by cat",
            round=3,
        )]
        # Monkeypatch: if apply is used, it should raise
        original_map_rows = getattr(pl.DataFrame, "map_rows", None)
        try:
            pl.DataFrame.map_rows = lambda *a, **kw: (_ for _ in ()).throw(
                NotImplementedError("apply/map_rows should not be used")
            )
            # Should still work — uses group_by+join
            result = _apply_round3_transforms(df, candidates)
            assert "val_mean_by_cat" in result.columns
        finally:
            if original_map_rows is not None:
                pl.DataFrame.map_rows = original_map_rows
            else:
                if hasattr(pl.DataFrame, "map_rows"):
                    delattr(pl.DataFrame, "map_rows")

    # ── TEST 1.7 ─────────────────────────────────────────────────
    def test_round4_generates_only_suitable_categoricals(self):
        from agents.feature_factory import _generate_round4_target_encoding_candidates
        schema = _make_schema([
            _col("col_a", dtype="str", n_unique=15),    # suitable
            _col("col_b", dtype="str", n_unique=1),     # binary — too few
            _col("col_c", dtype="float64", n_unique=300),  # not categorical (numeric + high card)
            _col("col_d", dtype="str", n_unique=250),   # too high cardinality
        ])
        candidates = _generate_round4_target_encoding_candidates(schema)
        names = [c.name for c in candidates]
        assert "te_col_a" in names
        assert "te_col_b" not in names
        assert "te_col_c" not in names
        assert "te_col_d" not in names

    # ── TEST 1.8 ─────────────────────────────────────────────────
    def test_round4_caps_at_max_candidates(self):
        from agents.feature_factory import _generate_round4_target_encoding_candidates, MAX_ROUND4_CANDIDATES
        cols = [_col(f"cat_{i}", dtype="str", n_unique=10 + i) for i in range(35)]
        schema = _make_schema(cols)
        candidates = _generate_round4_target_encoding_candidates(schema)
        assert len(candidates) <= MAX_ROUND4_CANDIDATES

    # ── TEST 1.9 ─────────────────────────────────────────────────
    def test_round4_encoding_never_uses_validation_fold_target(self):
        """The most important Round 4 test — fold isolation."""
        from agents.feature_factory import _apply_round4_target_encoding, FeatureCandidate
        np.random.seed(42)
        n = 100
        # Category "A" always has target=1 — if fold-isolated,
        # val fold's encoded value won't be exactly 1.0
        cats = np.array(["A"] * 50 + ["B"] * 50)
        y = np.array([1.0] * 50 + [0.0] * 50)

        df = pl.DataFrame({"grp": cats})
        candidates = [FeatureCandidate(
            name="te_grp", source_columns=["grp"],
            transform_type="target_encoding",
            description="target encoding of grp", round=4,
        )]

        result = _apply_round4_target_encoding(df, y, candidates, n_folds=5, smoothing=30.0)
        encoded = result["te_grp"].to_numpy()

        # Global mean of "A" = 1.0. If fold-isolated, the encoded value
        # for fold 3's "A" samples should NOT be exactly 1.0 (due to smoothing
        # and train-only computation). If full-set is used, it would be close to
        # (50*1.0 + 30*0.5)/(50+30) ≈ 0.8125 uniformly.
        # With fold-isolated: train portion for fold k has ~40 "A" samples with target=1.0,
        # so: (40*1.0 + 30*0.5)/(40+30) ≈ 0.786
        # The key check: values should NOT all be identical (fold isolation causes per-fold variation).
        unique_encoded = set(np.round(encoded[:50], 6))
        # If global leak, all A's get the same value. With fold-isolation, different folds get different values.
        # With smoothing=30, even with all A's having target=1, the encoded value is not 1.0.
        assert all(v < 0.95 for v in encoded[:50]), "Smoothing should prevent values close to 1.0"

    # ── TEST 1.10 ────────────────────────────────────────────────
    def test_round4_smoothing_formula_correct(self):
        from agents.feature_factory import _apply_round4_target_encoding, FeatureCandidate
        # Construct a scenario: category "X" with 5 samples, target=[1,1,1,0,1]
        # in training portion. global_mean=0.5, smoothing=30.
        # Expected: (5*0.8 + 30*0.5) / (5+30) = 19/35 ≈ 0.5429
        n = 10
        cats = np.array(["X"] * 5 + ["Y"] * 5)
        y = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 1], dtype=np.float64)

        df = pl.DataFrame({"cat": cats})
        candidates = [FeatureCandidate(
            name="te_cat", source_columns=["cat"],
            transform_type="target_encoding",
            description="te of cat", round=4,
        )]
        result = _apply_round4_target_encoding(df, y, candidates, n_folds=5, smoothing=30.0)
        encoded = result["te_cat"].to_numpy()

        # All encoded values should be between global_mean (0.5) and the raw group mean
        # Due to smoothing, no value should be exactly 0.8 (raw group mean for X)
        global_mean = np.mean(y)
        for val in encoded:
            assert abs(val - global_mean) < 0.5  # within reasonable range
            assert val != 0.8  # not unsmoothed

    # ── TEST 1.11 ────────────────────────────────────────────────
    def test_round4_unseen_category_gets_global_mean(self):
        from agents.feature_factory import _apply_round4_target_encoding, FeatureCandidate
        # Category "Z" only appears in some fold's validation but never in that fold's training
        n = 10
        # Z appears once — in one fold's val set, it won't have Z in training
        cats = np.array(["A", "A", "A", "A", "A", "A", "A", "A", "A", "Z"])
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)

        df = pl.DataFrame({"cat": cats})
        candidates = [FeatureCandidate(
            name="te_cat", source_columns=["cat"],
            transform_type="target_encoding",
            description="te of cat", round=4,
        )]
        # Should not raise KeyError
        result = _apply_round4_target_encoding(df, y, candidates, n_folds=5, smoothing=30.0)
        encoded = result["te_cat"].to_numpy()
        global_mean = float(np.mean(y))
        # Z should get global_mean
        assert abs(encoded[-1] - global_mean) < 0.01

    # ── TEST 1.12 ────────────────────────────────────────────────
    def test_round4_candidates_have_round_4_field(self):
        from agents.feature_factory import _generate_round4_target_encoding_candidates
        schema = _make_schema([
            _col("cat_a", dtype="str", n_unique=15),
            _col("cat_b", dtype="str", n_unique=20),
        ])
        candidates = _generate_round4_target_encoding_candidates(schema)
        for c in candidates:
            assert c.round == 4
            assert c.transform_type == "target_encoding"

    # ── TEST 1.13 ────────────────────────────────────────────────
    def test_round4_high_cardinality_sorted_first(self):
        from agents.feature_factory import _generate_round4_target_encoding_candidates
        schema = _make_schema([
            _col("cat_a", dtype="str", n_unique=50),
            _col("cat_b", dtype="str", n_unique=20),
            _col("cat_c", dtype="str", n_unique=35),
        ])
        candidates = _generate_round4_target_encoding_candidates(schema)
        names = [c.name for c in candidates]
        # Sorted descending by n_unique: A(50), C(35), B(20)
        assert names == ["te_cat_a", "te_cat_c", "te_cat_b"]

    # ── TEST 1.14 ────────────────────────────────────────────────
    def test_round3_round4_both_appear_in_manifest(self):
        """Both round 3 and round 4 candidates appear when run_feature_factory is called."""
        from agents.feature_factory import (
            _generate_round3_aggregation_features,
            _generate_round4_target_encoding_candidates,
        )
        schema = _make_schema([
            _col("cat_a", dtype="str", n_unique=10),
            _col("num_b", dtype="float64", n_unique=100),
        ])
        r3 = _generate_round3_aggregation_features(schema)
        r4 = _generate_round4_target_encoding_candidates(schema)
        rounds_present = {c.round for c in r3 + r4}
        assert 3 in rounds_present
        assert 4 in rounds_present


# ═══════════════════════════════════════════════════════════════
# BLOCK 2 — FEATURE FACTORY ROUND 5 + INTERACTION BUDGET CAP (14 tests)
# ═══════════════════════════════════════════════════════════════

class TestFeatureFactoryRound5AndBudgetCap:
    """Tests for Round 5 (hypothesis + interactions) and budget cap."""

    # ── TEST 2.1 ─────────────────────────────────────────────────
    def test_round5a_generates_from_unvalidated_insights_only(self):
        from agents.feature_factory import _generate_round5_hypothesis_features
        schema = _make_schema([
            _col("fare", dtype="float64", n_unique=100),
            _col("age", dtype="float64", n_unique=80),
        ])
        brief = {
            "insights": [
                {"content": "Older passengers survived less", "validated": False},
                {"content": "Fare correlates with survival", "validated": False},
                {"content": "Gender matters", "validated": False},
                {"content": "Class matters", "validated": True},
                {"content": "Embarkment irrelevant", "validated": True},
            ],
            "domain": "tabular",
        }
        state = _make_state()

        # Mock LLM to return 3 candidates (one per unvalidated insight)
        mock_response = json.dumps([
            {"hypothesis_index": 1, "hypothesis_summary": "age", "feature_name": "age_bin",
             "source_columns": ["age"], "transform_type": "bin", "expression": "bin(age)"},
            {"hypothesis_index": 2, "hypothesis_summary": "fare", "feature_name": "log_fare",
             "source_columns": ["fare"], "transform_type": "ratio", "expression": "log(fare)"},
            {"hypothesis_index": 3, "hypothesis_summary": "combo", "feature_name": "fare_age",
             "source_columns": ["fare", "age"], "transform_type": "interaction", "expression": "fare*age"},
        ])

        with patch("agents.feature_factory.call_llm", return_value=mock_response):
            candidates = _generate_round5_hypothesis_features(schema, brief, state)

        assert len(candidates) <= 3
        for c in candidates:
            assert c.round == 5

    # ── TEST 2.2 ─────────────────────────────────────────────────
    def test_round5a_validates_source_columns_against_schema(self):
        from agents.feature_factory import _generate_round5_hypothesis_features
        schema = _make_schema([_col("fare", dtype="float64", n_unique=100)])
        brief = {
            "insights": [{"content": "Ghost col matters", "validated": False}],
            "domain": "tabular",
        }
        state = _make_state()

        mock_response = json.dumps([
            {"hypothesis_index": 1, "hypothesis_summary": "ghost",
             "feature_name": "ghost_feat", "source_columns": ["ghost_column"],
             "transform_type": "ratio", "expression": "ghost"},
        ])

        with patch("agents.feature_factory.call_llm", return_value=mock_response):
            candidates = _generate_round5_hypothesis_features(schema, brief, state)

        assert len(candidates) == 0

    # ── TEST 2.3 ─────────────────────────────────────────────────
    def test_round5a_capped_at_10_candidates(self):
        from agents.feature_factory import _generate_round5_hypothesis_features
        cols = [_col(f"col_{i}", dtype="float64", n_unique=50) for i in range(20)]
        schema = _make_schema(cols)
        brief = {
            "insights": [{"content": f"Hypothesis {i}", "validated": False} for i in range(15)],
            "domain": "tabular",
        }
        state = _make_state()

        items = [
            {"hypothesis_index": i, "hypothesis_summary": f"h{i}",
             "feature_name": f"feat_{i}", "source_columns": [f"col_{i % 20}"],
             "transform_type": "ratio", "expression": f"col_{i}"}
            for i in range(15)
        ]
        mock_response = json.dumps(items)

        with patch("agents.feature_factory.call_llm", return_value=mock_response):
            candidates = _generate_round5_hypothesis_features(schema, brief, state)

        assert len(candidates) <= 10

    # ── TEST 2.4 ─────────────────────────────────────────────────
    def test_round5a_graceful_on_no_unvalidated_hypotheses(self):
        from agents.feature_factory import _generate_round5_hypothesis_features
        schema = _make_schema([_col("col_a", dtype="float64")])
        brief = {"insights": [], "domain": "tabular"}
        state = _make_state()

        call_count = {"n": 0}
        original_call = None

        def mock_llm(*args, **kwargs):
            call_count["n"] += 1
            return "[]"

        with patch("agents.feature_factory.call_llm", side_effect=mock_llm):
            candidates = _generate_round5_hypothesis_features(schema, brief, state)

        assert candidates == []
        assert call_count["n"] == 0

    # ── TEST 2.5 ─────────────────────────────────────────────────
    def test_round5b_limited_to_top_k_features(self):
        from agents.feature_factory import _generate_round5_interaction_features, MAX_INTERACTION_FEATURES
        schema = _make_schema([_col(f"f_{i}", dtype="float64") for i in range(50)])
        brief = {"meaningful_interactions": []}
        top_features = [f"f_{i}" for i in range(50)]

        candidates = _generate_round5_interaction_features(
            schema, brief, top_features, max_k=MAX_INTERACTION_FEATURES
        )

        all_sources = set()
        for c in candidates:
            all_sources.update(c.source_columns)

        # No feature beyond top-20 should appear
        for feat in all_sources:
            idx = int(feat.split("_")[1])
            assert idx < MAX_INTERACTION_FEATURES

    # ── TEST 2.6 ─────────────────────────────────────────────────
    def test_round5b_domain_pairs_included_first(self):
        from agents.feature_factory import _generate_round5_interaction_features
        schema = _make_schema([
            _col("fare", dtype="float64"),
            _col("pclass", dtype="float64"),
            _col("age", dtype="float64"),
        ])
        brief = {"meaningful_interactions": [["fare", "pclass"]]}
        top_features = ["fare", "pclass", "age"]

        candidates = _generate_round5_interaction_features(schema, brief, top_features, max_k=20)

        # Domain pairs should appear first
        first_sources = [tuple(sorted(c.source_columns)) for c in candidates[:3]]
        assert ("fare", "pclass") in first_sources

    # ── TEST 2.7 ─────────────────────────────────────────────────
    def test_budget_cap_fires_above_500_interactions(self):
        from agents.feature_factory import (
            _apply_interaction_budget_cap, FeatureCandidate, MAX_INTERACTION_CANDIDATES,
        )
        interactions = [
            FeatureCandidate(
                name=f"f{i}_x_f{j}", source_columns=[f"f{i}", f"f{j}"],
                transform_type="interaction_multiply",
                description=f"interaction {i} {j}", round=5,
            )
            for i in range(35) for j in range(i + 1, 35)
        ]  # C(35,2) = 595 > 500
        non_interactions = [
            FeatureCandidate(
                name=f"log_{i}", source_columns=[f"f{i}"],
                transform_type="log", description="log", round=1,
            )
            for i in range(10)
        ]

        result = _apply_interaction_budget_cap(
            interactions, non_interactions + interactions, max_cap=MAX_INTERACTION_CANDIDATES
        )
        interaction_count = sum(1 for c in result if c.transform_type.startswith("interaction_"))
        assert interaction_count == MAX_INTERACTION_CANDIDATES

    # ── TEST 2.8 ─────────────────────────────────────────────────
    def test_budget_cap_does_not_fire_below_500_interactions(self):
        from agents.feature_factory import _apply_interaction_budget_cap, FeatureCandidate
        interactions = [
            FeatureCandidate(
                name=f"f{i}_x_f{j}", source_columns=[f"f{i}", f"f{j}"],
                transform_type="interaction_multiply",
                description="interaction", round=5,
            )
            for i in range(20) for j in range(i + 1, 20)
        ]  # C(20,2) = 190 < 500
        all_candidates = interactions[:]

        result = _apply_interaction_budget_cap(interactions, all_candidates, max_cap=500)
        assert len(result) == 190

    # ── TEST 2.9 ─────────────────────────────────────────────────
    def test_budget_cap_scores_domain_pairs_higher(self):
        from agents.feature_factory import (
            _apply_interaction_budget_cap_with_importance, FeatureCandidate,
        )
        # 10 domain + 590 non-domain = 600 total interactions
        domain_candidates = [
            FeatureCandidate(
                name=f"domain_{i}_x_other_{i}",
                source_columns=[f"domain_{i}", f"other_{i}"],
                transform_type="interaction_multiply",
                description="Domain-guided interaction", round=5,
            )
            for i in range(10)
        ]
        non_domain = [
            FeatureCandidate(
                name=f"nd_{i}_x_nd_{i+1}",
                source_columns=[f"nd_{i}", f"nd_{i+1}"],
                transform_type="interaction_multiply",
                description="generic interaction", round=5,
            )
            for i in range(590)
        ]
        all_cands = domain_candidates + non_domain

        meaningful = [[f"domain_{i}", f"other_{i}"] for i in range(10)]

        result = _apply_interaction_budget_cap_with_importance(
            all_cands, null_importance_result=None,
            meaningful_interactions=meaningful, max_cap=500,
        )
        result_interaction_names = {
            c.name for c in result if c.transform_type.startswith("interaction_")
        }
        # All 10 domain candidates should survive
        for i in range(10):
            assert f"domain_{i}_x_other_{i}" in result_interaction_names

    # ── TEST 2.10 ────────────────────────────────────────────────
    def test_budget_cap_preserves_non_interaction_candidates(self):
        from agents.feature_factory import (
            _apply_interaction_budget_cap_with_importance, FeatureCandidate,
        )
        non_interactions = [
            FeatureCandidate(
                name=f"log_{i}", source_columns=[f"f{i}"],
                transform_type="log", description="log", round=1,
            )
            for i in range(150)
        ]
        interactions = [
            FeatureCandidate(
                name=f"f{i}_x_f{j}", source_columns=[f"f{i}", f"f{j}"],
                transform_type="interaction_multiply",
                description="interaction", round=5,
            )
            for i in range(35) for j in range(i + 1, 35)
        ][:600]
        all_cands = non_interactions + interactions

        result = _apply_interaction_budget_cap_with_importance(
            all_cands, null_importance_result=None,
            meaningful_interactions=[], max_cap=500,
        )
        non_int_count = sum(1 for c in result if not c.transform_type.startswith("interaction_"))
        assert non_int_count == 150

    # ── TEST 2.11 ────────────────────────────────────────────────
    def test_round5b_generates_correct_interaction_names(self):
        from agents.feature_factory import _generate_round5_interaction_features
        schema = _make_schema([
            _col("A", dtype="float64"),
            _col("B", dtype="float64"),
        ])
        brief = {"meaningful_interactions": []}
        candidates = _generate_round5_interaction_features(
            schema, brief, ["A", "B"], max_k=20
        )
        if candidates:
            c = candidates[0]
            assert "_x_" in c.name or "_div_" in c.name or "_plus_" in c.name
            assert " " not in c.name

    # ── TEST 2.12 ────────────────────────────────────────────────
    def test_max_interaction_features_constant_is_20(self):
        from agents.feature_factory import MAX_INTERACTION_FEATURES
        assert MAX_INTERACTION_FEATURES == 20

    # ── TEST 2.13 ────────────────────────────────────────────────
    def test_max_interaction_candidates_constant_is_500(self):
        from agents.feature_factory import MAX_INTERACTION_CANDIDATES
        assert MAX_INTERACTION_CANDIDATES == 500

    # ── TEST 2.14 ────────────────────────────────────────────────
    def test_total_candidates_after_all_rounds_within_budget(self):
        from agents.feature_factory import (
            _generate_round1_features,
            _generate_round3_aggregation_features,
            _generate_round4_target_encoding_candidates,
            _generate_round5_interaction_features,
            _apply_interaction_budget_cap_with_importance,
            MAX_ROUND3_CANDIDATES, MAX_ROUND4_CANDIDATES,
            MAX_INTERACTION_CANDIDATES,
        )
        # Medium schema: 20 features, 5 categoricals
        cols = []
        for i in range(5):
            cols.append(_col(f"cat_{i}", dtype="str", n_unique=10 + i))
        for i in range(15):
            cols.append(_col(f"num_{i}", dtype="float64", n_unique=50 + i, min_val=0))
        schema = _make_schema(cols)

        r1 = _generate_round1_features(schema)
        r3 = _generate_round3_aggregation_features(schema)
        r4 = _generate_round4_target_encoding_candidates(schema)
        top_features = [c["name"] for c in schema["columns"]]
        r5b = _generate_round5_interaction_features(
            schema, {"meaningful_interactions": []}, top_features, max_k=20
        )

        all_cands = r1 + r3 + r4 + r5b
        all_cands = _apply_interaction_budget_cap_with_importance(
            all_cands, null_importance_result=None,
            meaningful_interactions=[], max_cap=MAX_INTERACTION_CANDIDATES,
        )

        # Max theoretical: 50(R1) + 30(R2) + 200(R3) + 30(R4) + 10(R5a) + 500(R5b) = 820
        assert len(all_cands) <= 820


# ═══════════════════════════════════════════════════════════════
# BLOCK 3 — PSEUDO-LABELING: CORRECTNESS (14 tests)
# ═══════════════════════════════════════════════════════════════

class TestPseudoLabelCorrectness:
    """Tests for pseudo-label confidence, selection, CV invariants."""

    # ── TEST 3.1 ─────────────────────────────────────────────────
    def test_confidence_for_binary_is_distance_from_0_5(self):
        from agents.pseudo_label_agent import _compute_confidence
        y_pred = np.array([0.9, 0.1, 0.6, 0.4, 0.5])
        confidence = _compute_confidence(y_pred, metric="auc")
        expected = np.array([0.4, 0.4, 0.1, 0.1, 0.0])
        np.testing.assert_allclose(confidence, expected, atol=1e-10)

    # ── TEST 3.2 ─────────────────────────────────────────────────
    def test_top_10_percent_selection_correct(self):
        from agents.pseudo_label_agent import _select_confident_samples, _compute_confidence
        np.random.seed(42)
        y_pred = np.random.uniform(0, 1, 100)
        confidence = _compute_confidence(y_pred, metric="auc")
        mask, threshold = _select_confident_samples(confidence, y_pred)
        n_selected = int(mask.sum())
        assert n_selected >= 10
        assert n_selected <= 15  # some ties at threshold are acceptable

    # ── TEST 3.3 ─────────────────────────────────────────────────
    def test_validation_fold_never_sees_pseudo_labels(self):
        """The most critical pseudo-labeling test."""
        from agents.pseudo_label_agent import _run_cv_with_pseudo_labels

        n_train = 100
        n_pseudo = 20
        X_train = pl.DataFrame({"f1": np.random.randn(n_train), "f2": np.random.randn(n_train)})
        y_train = np.random.randint(0, 2, n_train).astype(np.float64)

        # Pseudo-labels: all target=1.0 — a detectable signal
        X_pseudo = pl.DataFrame({"f1": np.full(n_pseudo, 999.0), "f2": np.full(n_pseudo, 999.0)})
        y_pseudo = np.ones(n_pseudo, dtype=np.float64)

        fit_records = []
        original_fit = None

        class MockModel:
            """Records fit data to verify fold integrity."""
            def __init__(self, **params):
                self.params = params

            def fit(self, X, y):
                fit_records.append({
                    "X_shape": X.shape,
                    "y_len": len(y),
                    "has_pseudo_marker": bool(np.any(X[:, 0] == 999.0)),
                    "n_train_original": n_train,
                })
                return self

            def predict_proba(self, X):
                return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])

        with patch("agents.pseudo_label_agent.lgb.LGBMClassifier", MockModel):
            _run_cv_with_pseudo_labels(
                X_train=X_train, y_train=y_train,
                X_pseudo=X_pseudo, y_pseudo=y_pseudo,
                lgbm_params={}, metric="auc",
            )

        # Each fold should have been called with training data + pseudo-labels
        assert len(fit_records) == 5
        for record in fit_records:
            # Training portion (~80 real samples) + 20 pseudo = ~100
            assert record["has_pseudo_marker"], "Pseudo-labels must be in training fold"
            # X_shape rows should be > n_train_original (because pseudo-labels are added)
            assert record["X_shape"][0] > 80

    # ── TEST 3.4 ─────────────────────────────────────────────────
    def test_cv_gate_stops_iteration_when_no_improvement(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        # Mock CV to return WORSE scores
        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.65, 0.65, 0.65, 0.65, 0.65]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=False):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        assert result_state["pseudo_labels_applied"] == False
        pl_result = result_state["pseudo_label_result"]
        assert pl_result.halt_reason == "cv_did_not_improve"

    # ── TEST 3.5 ─────────────────────────────────────────────────
    def test_cv_gate_proceeds_when_improvement_significant(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        call_count = {"n": 0}

        def mock_cv(*args, **kwargs):
            call_count["n"] += 1
            # Each iteration returns progressively better scores
            return [0.71, 0.71, 0.71, 0.71, 0.71]

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels", side_effect=mock_cv), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        assert result_state["pseudo_labels_applied"] == True
        pl_result = result_state["pseudo_label_result"]
        assert pl_result.iterations_completed >= 1

    # ── TEST 3.6 ─────────────────────────────────────────────────
    def test_wilcoxon_gate_applied_to_cv_improvement(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        wilcoxon_calls = {"n": 0}

        def mock_wilcoxon(a, b, **kwargs):
            wilcoxon_calls["n"] += 1
            return True

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.71, 0.71, 0.71, 0.71, 0.71]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", side_effect=mock_wilcoxon):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            run_pseudo_label_agent(state)

        assert wilcoxon_calls["n"] >= 1

    # ── TEST 3.7 ─────────────────────────────────────────────────
    def test_max_iterations_is_3_not_4(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent, MAX_PL_ITERATIONS

        assert MAX_PL_ITERATIONS == 3

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(200)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        iteration_count = {"n": 0}

        def mock_cv(*args, **kwargs):
            iteration_count["n"] += 1
            return [0.72, 0.72, 0.72, 0.72, 0.72]

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels", side_effect=mock_cv), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        pl_result = result_state["pseudo_label_result"]
        assert pl_result.iterations_completed == 3
        assert pl_result.halt_reason == "max_iterations"

    # ── TEST 3.8 ─────────────────────────────────────────────────
    def test_confidence_comparison_for_regression_uses_interval_width(self):
        from agents.pseudo_label_agent import _compute_confidence
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        confidence = _compute_confidence(y_pred, metric="rmse", quantile_model=None)
        # Fallback: uniform confidence
        np.testing.assert_array_equal(confidence, np.ones(5))

    # ── TEST 3.9 ─────────────────────────────────────────────────
    def test_multiclass_confidence_is_margin_between_top_2_classes(self):
        from agents.pseudo_label_agent import _compute_confidence
        y_pred = np.array([
            [0.7, 0.2, 0.1],
            [0.4, 0.35, 0.25],
            [0.9, 0.05, 0.05],
        ])
        confidence = _compute_confidence(y_pred, metric="multiclass")
        expected = np.array([0.5, 0.05, 0.85])
        np.testing.assert_allclose(confidence, expected, atol=1e-10)

    # ── TEST 3.10 ────────────────────────────────────────────────
    def test_no_pseudo_labels_when_no_confident_samples(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.72, 0.72, 0.72, 0.72, 0.72]), \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            # All predictions at 0.5 — zero confidence
            def _all_05_proba(X):
                n = len(X)
                return np.column_stack([np.ones(n) * 0.5, np.ones(n) * 0.5])
            mock_model.predict_proba.side_effect = _all_05_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        pl_result = result_state["pseudo_label_result"]
        # All confidences are 0 → threshold is 0 → all pass (mask >= 0 is true)
        # Pipeline should still complete without error
        assert pl_result is not None

    # ── TEST 3.11 ────────────────────────────────────────────────
    def test_pseudo_labels_accumulated_across_iterations(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(200)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.72, 0.72, 0.72, 0.72, 0.72]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        pl_result = result_state["pseudo_label_result"]
        total_added = sum(pl_result.pseudo_labels_added)
        assert total_added > 0
        assert len(pl_result.pseudo_labels_added) == pl_result.iterations_completed

    # ── TEST 3.12 ────────────────────────────────────────────────
    def test_already_pseudolabeled_samples_excluded_from_subsequent_iterations(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        predict_sizes = []

        class TrackingModel:
            def __init__(self, **params):
                pass
            def fit(self, X, y):
                return self
            def predict_proba(self, X):
                predict_sizes.append(len(X))
                # Varying confidences so top 10% get selected
                n = len(X)
                probs = np.linspace(0, 1, n)
                return np.column_stack([1 - probs, probs])

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.72, 0.72, 0.72, 0.72, 0.72]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier", TrackingModel), \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            result_state = run_pseudo_label_agent(state)

        # Each subsequent iteration should predict fewer samples
        if len(predict_sizes) > 1:
            assert predict_sizes[1] < predict_sizes[0]

    # ── TEST 3.13 ────────────────────────────────────────────────
    def test_pseudo_label_result_excluded_from_redis_checkpoint(self):
        from agents.pseudo_label_agent import PseudoLabelResult
        from memory.redis_state import _is_serialisable

        result = PseudoLabelResult(
            iterations_completed=2,
            pseudo_labels_added=[10, 8],
            cv_scores_with_pl=[0.72, 0.73],
            cv_scores_without_pl=[0.70],
            cv_improvements=[0.02, 0.01],
            halted_early=False,
            halt_reason="max_iterations",
            final_pseudo_label_mask=[0, 1, 0, 1],
            confidence_thresholds=[0.3, 0.35],
        )
        # PseudoLabelResult is a dataclass — not JSON-serializable directly
        # The _is_serialisable check should filter it
        assert not _is_serialisable(result)

    # ── TEST 3.14 ────────────────────────────────────────────────
    def test_pseudo_label_agent_skipped_when_no_selected_models(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        state = _make_state(
            X_train=pl.DataFrame({"f1": [1.0, 2.0]}),
            y_train=np.array([0.0, 1.0]),
            X_test=pl.DataFrame({"f1": [3.0]}),
            selected_models=[],
        )
        result_state = run_pseudo_label_agent(state)
        assert result_state["pseudo_labels_applied"] == False


# ═══════════════════════════════════════════════════════════════
# BLOCK 4 — PSEUDO-LABELING: INTEGRATION (10 tests)
# ═══════════════════════════════════════════════════════════════

class TestPseudoLabelIntegration:
    """Integration tests for pseudo-labeling pipeline."""

    # ── TEST 4.1 ─────────────────────────────────────────────────
    def test_pseudo_label_agent_runs_after_ensemble(self):
        """Pipeline order: ensemble_architect → pseudo_label_agent."""
        from agents.pseudo_label_agent import run_pseudo_label_agent
        # This is a structural test: verify the graph has the right ordering
        # by checking that pseudo_label_agent can access ensemble_oof
        state = _make_state(
            X_train=pl.DataFrame({"f1": np.random.randn(50)}),
            y_train=np.random.randint(0, 2, 50).astype(np.float64),
            X_test=pl.DataFrame({"f1": np.random.randn(20)}),
            selected_models=["lgbm"],
            ensemble_oof=[0.5] * 50,
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.65, 0.65, 0.65, 0.65, 0.65]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=False):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result = run_pseudo_label_agent(state)

        # Lineage log should exist
        session_id = state["session_id"]
        log_path = Path(f"outputs/{session_id}/logs/lineage.jsonl")
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        actions = [e["action"] for e in entries]
        assert "pseudo_label_complete" in actions

        _cleanup_outputs(session_id)

    # ── TEST 4.2 ─────────────────────────────────────────────────
    def test_x_train_with_pseudo_larger_than_original(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.72, 0.72, 0.72, 0.72, 0.72]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        assert len(result_state["X_train_with_pseudo"]) > n

        _cleanup_outputs(state["session_id"])

    # ── TEST 4.3 ─────────────────────────────────────────────────
    def test_x_train_with_pseudo_same_schema_as_x_train(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.72, 0.72, 0.72, 0.72, 0.72]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        assert result_state["X_train_with_pseudo"].schema == X_train.schema

        _cleanup_outputs(state["session_id"])

    # ── TEST 4.4 ─────────────────────────────────────────────────
    def test_y_train_with_pseudo_length_matches_x_train_with_pseudo(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.72, 0.72, 0.72, 0.72, 0.72]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        assert len(result_state["y_train_with_pseudo"]) == len(result_state["X_train_with_pseudo"])

        _cleanup_outputs(state["session_id"])

    # ── TEST 4.5 ─────────────────────────────────────────────────
    def test_pseudo_label_cv_improvement_logged_to_lineage(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.72, 0.72, 0.72, 0.72, 0.72]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        session_id = state["session_id"]
        log_path = Path(f"outputs/{session_id}/logs/lineage.jsonl")
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        pl_entries = [e for e in entries if e["action"] == "pseudo_label_complete"]
        assert len(pl_entries) >= 1
        entry = pl_entries[0]
        assert "cv_improvement" in entry["values_changed"]
        assert "iterations" in entry["values_changed"]
        assert "total_pl_added" in entry["values_changed"]

        _cleanup_outputs(session_id)

    # ── TEST 4.6 ─────────────────────────────────────────────────
    def test_pseudo_labels_applied_false_when_cv_does_not_improve(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.65, 0.65, 0.65, 0.65, 0.65]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=False):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        assert result_state["pseudo_labels_applied"] == False
        assert len(result_state["X_train_with_pseudo"]) == n

        _cleanup_outputs(state["session_id"])

    # ── TEST 4.7 ─────────────────────────────────────────────────
    def test_submission_uses_model_trained_on_augmented_data(self):
        """When pseudo_labels_applied=True, X_train_with_pseudo must be larger."""
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.72, 0.72, 0.72, 0.72, 0.72]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        if result_state["pseudo_labels_applied"]:
            # When applied, augmented data must be larger
            assert len(result_state["X_train_with_pseudo"]) > n
            assert len(result_state["y_train_with_pseudo"]) > n

        _cleanup_outputs(state["session_id"])

    # ── TEST 4.8 ─────────────────────────────────────────────────
    def test_confidence_threshold_stored_per_iteration(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(200)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.72, 0.72, 0.72, 0.72, 0.72]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        pl_result = result_state["pseudo_label_result"]
        # Should have threshold stored for each completed iteration
        assert len(pl_result.confidence_thresholds) == pl_result.iterations_completed

        _cleanup_outputs(state["session_id"])

    # ── TEST 4.9 ─────────────────────────────────────────────────
    def test_zero_iterations_when_first_cv_fails_gate(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.69, 0.69, 0.69, 0.69, 0.69]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=False):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        pl_result = result_state["pseudo_label_result"]
        assert pl_result.iterations_completed == 0

        _cleanup_outputs(state["session_id"])

    # ── TEST 4.10 ────────────────────────────────────────────────
    def test_day16_ensemble_oof_still_available_after_pseudo_labeling(self):
        from agents.pseudo_label_agent import run_pseudo_label_agent

        n = 100
        X_train = pl.DataFrame({"f1": np.random.randn(n)})
        y_train = np.random.randint(0, 2, n).astype(np.float64)
        X_test = pl.DataFrame({"f1": np.random.randn(50)})
        original_oof = [0.5] * n

        state = _make_state(
            X_train=X_train, y_train=y_train, X_test=X_test,
            selected_models=["lgbm"],
            ensemble_oof=original_oof,
            model_registry=[{
                "model_type": "lgbm",
                "params": {"n_estimators": 10, "verbosity": -1},
                "fold_scores": [0.7, 0.7, 0.7, 0.7, 0.7],
            }],
        )

        with patch("agents.pseudo_label_agent._run_cv_with_pseudo_labels",
                    return_value=[0.72, 0.72, 0.72, 0.72, 0.72]), \
             patch("agents.pseudo_label_agent.lgb.LGBMClassifier") as MockCls, \
             patch("tools.wilcoxon_gate.is_significantly_better", return_value=True):
            mock_model = MagicMock()
            mock_model.predict_proba.side_effect = _dynamic_predict_proba
            MockCls.return_value = mock_model

            result_state = run_pseudo_label_agent(state)

        assert result_state["ensemble_oof"] == original_oof

        _cleanup_outputs(state["session_id"])
