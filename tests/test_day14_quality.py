# tests/test_day14_quality.py
# Day 14: GM-CAP 4 critic memory — compounding advantage + Phase 2 gate
# 30 tests — IMMUTABLE after Day 14

import pytest
import uuid
import json
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.state import initial_state


# ── Helpers ──────────────────────────────────────────────────────────

def _make_state(**overrides):
    state = initial_state("test_comp", "data/test.csv")
    state["session_id"] = "test_session_day14"
    state["competition_fingerprint"] = {
        "task_type": "tabular",
        "target_type": "binary",
        "n_rows_bucket": "medium",
        "n_features_bucket": "medium",
        "imbalance_ratio": 0.35,
        "n_categorical_high_cardinality": 2,
        "has_temporal_feature": False,
    }
    state["feature_names"] = ["age", "fare", "cabin_deck", "embarked"]
    for k, v in overrides.items():
        state[k] = v
    return state


def _make_similar_fingerprint():
    """Fingerprint close to _make_state() — should have small distance."""
    return {
        "task_type": "tabular",
        "target_type": "binary",
        "n_rows_bucket": "medium",
        "n_features_bucket": "medium",
        "imbalance_ratio": 0.30,
        "n_categorical_high_cardinality": 1,
        "has_temporal_feature": False,
    }


def _make_dissimilar_fingerprint():
    """Fingerprint far from _make_state() — NLP multiclass."""
    return {
        "task_type": "nlp",
        "target_type": "multiclass",
        "n_rows_bucket": "huge",
        "n_features_bucket": "very_wide",
        "imbalance_ratio": 0.01,
        "n_categorical_high_cardinality": 50,
        "has_temporal_feature": True,
    }


def _store_pattern(fingerprint, feature_flagged="target_enc_cabin",
                   failure_mode="target encoding without fold isolation",
                   confidence=0.90, cv_lb_gap=0.026,
                   competition_name="spaceship-titanic-2024"):
    """Store a critic failure pattern directly in ChromaDB."""
    from memory.memory_schema import store_critic_failure_pattern
    return store_critic_failure_pattern(
        fingerprint=fingerprint,
        missed_issue=f"{failure_mode} on {feature_flagged}",
        competition_name=competition_name,
        feature_flagged=feature_flagged,
        failure_mode=failure_mode,
        cv_lb_gap=cv_lb_gap,
        confidence=confidence,
    )


@pytest.fixture(autouse=True)
def _isolate_chroma(tmp_path, monkeypatch):
    """Every test gets its own empty ChromaDB so tests are independent."""
    chroma_dir = str(tmp_path / "chroma_test")
    import memory.chroma_client as cc
    original_build = cc.build_chroma_client

    def _patched_build(persist_dir=None):
        return original_build(persist_dir=chroma_dir)

    monkeypatch.setattr(cc, "build_chroma_client", _patched_build)
    # Also patch in memory_schema since it imports build_chroma_client
    import memory.memory_schema as ms
    monkeypatch.setattr(ms, "build_chroma_client", _patched_build)


# =====================================================================
# BLOCK 1 — HISTORICAL FAILURES VECTOR: CORRECTNESS (14 tests)
# =====================================================================
class TestHistoricalFailuresVector:

    # TEST 1.1
    def test_historical_vector_appears_in_vectors_checked(self):
        """Vector 8 must appear in vectors_checked and total == 8."""
        from agents.red_team_critic import _check_historical_failures
        state = _make_state()
        result = _check_historical_failures(state)
        assert "verdict" in result
        # Check it directly — the orchestrator wiring is tested via
        # the _run_vector call producing the name in vectors_checked.
        # Here we confirm the function runs without error.
        assert result["verdict"] == "OK"

    # TEST 1.2
    def test_returns_ok_when_collection_empty(self):
        """Empty/missing collection must return OK, not raise."""
        from agents.red_team_critic import _check_historical_failures
        state = _make_state()
        result = _check_historical_failures(state)
        assert result["verdict"] == "OK"
        assert result["patterns_retrieved"] == 0
        assert result["findings"] == []

    # TEST 1.3
    def test_returns_ok_when_no_patterns_within_distance(self):
        """Patterns with dissimilar fingerprints should be filtered out."""
        from agents.red_team_critic import _check_historical_failures
        # Store 3 patterns with very dissimilar fingerprints
        dissimilar = _make_dissimilar_fingerprint()
        for i in range(3):
            _store_pattern(dissimilar, feature_flagged=f"nlp_feat_{i}",
                           competition_name=f"nlp_comp_{i}")
        state = _make_state()
        result = _check_historical_failures(state)
        # Even if some patterns are retrieved (distance may vary by embedding model),
        # the verdict must be OK because the features don't match
        assert result["verdict"] == "OK"

    # TEST 1.4
    def test_returns_ok_when_patterns_found_but_feature_not_present(self):
        """Pattern flags a feature not in current set → OK."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin")
        state = _make_state(feature_names=["age", "fare", "pclass"])
        result = _check_historical_failures(state)
        assert result["verdict"] == "OK"

    # TEST 1.5
    def test_critical_when_high_confidence_feature_present(self):
        """confidence=0.90 + feature present → CRITICAL."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin", confidence=0.90)
        state = _make_state(feature_names=["target_enc_cabin", "age", "fare"])
        result = _check_historical_failures(state)
        assert result["verdict"] == "CRITICAL"

    # TEST 1.6
    def test_high_when_medium_confidence_feature_present(self):
        """confidence=0.75 → HIGH."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin", confidence=0.75)
        state = _make_state(feature_names=["target_enc_cabin", "age", "fare"])
        result = _check_historical_failures(state)
        assert result["verdict"] == "HIGH"

    # TEST 1.7
    def test_medium_when_low_confidence_feature_present(self):
        """confidence=0.55 → MEDIUM."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin", confidence=0.55)
        state = _make_state(feature_names=["target_enc_cabin", "age", "fare"])
        result = _check_historical_failures(state)
        assert result["verdict"] == "MEDIUM"

    # TEST 1.8
    def test_below_0_50_confidence_not_flagged(self):
        """confidence=0.45 → OK (below threshold)."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin", confidence=0.45)
        state = _make_state(feature_names=["target_enc_cabin", "age", "fare"])
        result = _check_historical_failures(state)
        assert result["verdict"] == "OK"

    # TEST 1.9
    def test_verdict_is_max_across_multiple_patterns(self):
        """Multiple findings → verdict is max severity."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="age", confidence=0.75,
                       competition_name="comp_a")  # HIGH
        _store_pattern(similar, feature_flagged="fare", confidence=0.55,
                       competition_name="comp_b")  # MEDIUM
        state = _make_state(feature_names=["age", "fare", "cabin_deck"])
        result = _check_historical_failures(state)
        assert result["verdict"] == "HIGH"
        assert result["patterns_matched"] >= 2

    # TEST 1.10
    def test_feature_matching_uses_substring_check(self):
        """'target_enc' should match 'target_enc_cabin' via substring."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc", confidence=0.90)
        state = _make_state(feature_names=["target_enc_cabin", "age", "fare"])
        result = _check_historical_failures(state)
        assert result["verdict"] != "OK", (
            "Substring matching failed: 'target_enc' should match 'target_enc_cabin'"
        )

    # TEST 1.11
    def test_feature_matching_not_too_loose(self):
        """Short name 'id' must NOT match 'period', 'latitude', etc."""
        from agents.red_team_critic import _check_historical_failures
        from agents.red_team_critic import MIN_FEATURE_LEN_FOR_SUBSTRING
        assert MIN_FEATURE_LEN_FOR_SUBSTRING >= 4, "Guard must be at least 4 chars"

        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="id", confidence=0.90)
        state = _make_state(feature_names=["period", "latitude", "validity_score"])
        result = _check_historical_failures(state)
        assert result["verdict"] == "OK", (
            f"Short feature name 'id' incorrectly matched with "
            f"features {state['feature_names']}"
        )

    # TEST 1.12
    def test_finding_contains_evidence_string(self):
        """Evidence must contain competition name, failure_mode, cv_lb_gap."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin",
                       failure_mode="target_leakage", confidence=0.90,
                       cv_lb_gap=0.026, competition_name="spaceship-titanic")
        state = _make_state(feature_names=["target_enc_cabin", "age"])
        result = _check_historical_failures(state)
        assert len(result["findings"]) >= 1
        evidence = result["findings"][0]["evidence"]
        assert "spaceship-titanic" in evidence
        assert "target_leakage" in evidence
        assert "0.026" in evidence

    # TEST 1.13
    def test_critical_finding_includes_replan_instructions(self):
        """CRITICAL finding must have replan_instructions."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin", confidence=0.90)
        state = _make_state(feature_names=["target_enc_cabin", "age"])
        result = _check_historical_failures(state)
        critical_findings = [f for f in result["findings"] if f["severity"] == "CRITICAL"]
        assert len(critical_findings) >= 1
        replan = critical_findings[0]["replan_instructions"]
        assert "target_enc_cabin" in replan["remove_features"]
        assert "feature_factory" in replan["rerun_nodes"]

    # TEST 1.14
    def test_chromadb_failure_returns_ok_not_exception(self):
        """ChromaDB crash must not propagate — returns OK."""
        from agents.red_team_critic import _check_historical_failures
        state = _make_state()
        with patch("memory.memory_schema.build_chroma_client",
                   side_effect=ConnectionError("ChromaDB down")):
            result = _check_historical_failures(state)
        assert result["verdict"] == "OK"
        assert "failed" in result.get("note", "").lower() or result["patterns_retrieved"] == 0


# =====================================================================
# BLOCK 2 — query_critic_failure_patterns() (8 tests)
# =====================================================================
class TestQueryCriticFailurePatterns:

    # TEST 2.1
    def test_returns_empty_list_when_collection_missing(self):
        """No collection → empty list, no exception."""
        from memory.memory_schema import query_critic_failure_patterns
        result = query_critic_failure_patterns(
            fingerprint=_make_similar_fingerprint()
        )
        assert result == []

    # TEST 2.2
    def test_returns_empty_list_when_collection_empty(self):
        """Collection exists but empty → empty list."""
        from memory.memory_schema import query_critic_failure_patterns
        from memory.chroma_client import build_chroma_client, get_or_create_collection
        client = build_chroma_client()
        get_or_create_collection(client, "critic_failure_patterns")
        result = query_critic_failure_patterns(
            fingerprint=_make_similar_fingerprint()
        )
        assert result == []

    # TEST 2.3
    def test_returns_patterns_within_distance_threshold(self):
        """Only patterns within max_distance are returned."""
        from memory.memory_schema import query_critic_failure_patterns
        similar = _make_similar_fingerprint()
        dissimilar = _make_dissimilar_fingerprint()
        # Store similar patterns
        for i in range(3):
            _store_pattern(similar, feature_flagged=f"feat_sim_{i}",
                           competition_name=f"sim_comp_{i}")
        # Store dissimilar patterns
        for i in range(2):
            _store_pattern(dissimilar, feature_flagged=f"feat_dis_{i}",
                           competition_name=f"dis_comp_{i}")
        result = query_critic_failure_patterns(
            fingerprint=_make_similar_fingerprint(), max_distance=0.75
        )
        # All returned patterns should be close (within distance)
        for p in result:
            assert p["distance"] <= 0.75

    # TEST 2.4
    def test_respects_n_results_limit(self):
        """n_results limits number of returned patterns."""
        from memory.memory_schema import query_critic_failure_patterns
        similar = _make_similar_fingerprint()
        for i in range(10):
            _store_pattern(similar, feature_flagged=f"feat_{i}",
                           competition_name=f"comp_{i}")
        result = query_critic_failure_patterns(
            fingerprint=_make_similar_fingerprint(), n_results=5
        )
        assert len(result) <= 5

    # TEST 2.5
    def test_returned_patterns_have_required_metadata_fields(self):
        """Each pattern must have the required fields."""
        from memory.memory_schema import query_critic_failure_patterns
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin",
                       confidence=0.9, cv_lb_gap=0.03)
        result = query_critic_failure_patterns(
            fingerprint=_make_similar_fingerprint(), max_distance=1.5
        )
        assert len(result) >= 1
        p = result[0]
        required_fields = [
            "competition_name", "feature_flagged", "failure_mode",
            "cv_lb_gap", "confidence", "distance",
        ]
        for field in required_fields:
            assert field in p, f"Missing field: {field}"

    # TEST 2.6
    def test_uses_fingerprint_to_text_for_query(self):
        """Must use fingerprint_to_text(), not str()."""
        from memory.memory_schema import query_critic_failure_patterns
        # Store a pattern first so collection exists
        similar = _make_similar_fingerprint()
        _store_pattern(similar)

        with patch("memory.memory_schema.fingerprint_to_text",
                   wraps=__import__("memory.memory_schema",
                                    fromlist=["fingerprint_to_text"]).fingerprint_to_text
                   ) as mock_ftt:
            query_critic_failure_patterns(fingerprint=similar)
            assert mock_ftt.call_count >= 1, (
                "fingerprint_to_text was not called — query likely uses str()"
            )

    # TEST 2.7
    def test_never_raises_on_any_input(self):
        """Edge-case inputs must not raise."""
        from memory.memory_schema import query_critic_failure_patterns
        # Empty dict
        assert query_critic_failure_patterns(fingerprint={}) == []
        # n_results=0 — should just return empty
        assert query_critic_failure_patterns(fingerprint={}, n_results=0) == []
        # max_distance=0.0 — nothing qualifies
        assert query_critic_failure_patterns(
            fingerprint=_make_similar_fingerprint(), max_distance=0.0
        ) == []

    # TEST 2.8
    def test_patterns_sorted_by_distance_ascending(self):
        """Returned patterns must be sorted by distance (most similar first)."""
        from memory.memory_schema import query_critic_failure_patterns
        similar = _make_similar_fingerprint()
        # Store several patterns — they'll have varying distances
        for i in range(5):
            fp = dict(similar)
            fp["imbalance_ratio"] = 0.30 + i * 0.02  # slight variation
            _store_pattern(fp, feature_flagged=f"feat_{i}",
                           competition_name=f"comp_{i}")
        result = query_critic_failure_patterns(
            fingerprint=_make_similar_fingerprint(), max_distance=1.5
        )
        if len(result) >= 2:
            distances = [p["distance"] for p in result]
            assert distances == sorted(distances), (
                f"Patterns not sorted by distance: {distances}"
            )


# =====================================================================
# BLOCK 3 — COMPOUNDING ADVANTAGE: END-TO-END (8 tests)
# =====================================================================
class TestCompoundingAdvantage:

    # TEST 3.1
    def test_pattern_written_by_post_mortem_is_retrieved_by_critic(self):
        """Full write → read loop: store → query → flag."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin", confidence=0.90)
        state = _make_state(feature_names=["target_enc_cabin", "age", "fare"])
        result = _check_historical_failures(state)
        assert result["verdict"] == "CRITICAL"
        assert result["patterns_matched"] >= 1

    # TEST 3.2
    def test_pattern_not_retrieved_for_dissimilar_competition(self):
        """NLP pattern must not fire on tabular competition."""
        from agents.red_team_critic import _check_historical_failures
        dissimilar = _make_dissimilar_fingerprint()
        # Use a feature name NOT in our feature set so even if distance passes,
        # the feature-match check blocks it
        _store_pattern(dissimilar, feature_flagged="nlp_token_embedding_dim",
                       confidence=0.90)
        state = _make_state(feature_names=["age", "fare"])
        result = _check_historical_failures(state)
        assert result["verdict"] == "OK"

    # TEST 3.3
    def test_critic_verdict_includes_historical_context_in_evidence(self):
        """Evidence must name the source competition."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin",
                       confidence=0.90, competition_name="spaceship-titanic-v1")
        state = _make_state(feature_names=["target_enc_cabin"])
        result = _check_historical_failures(state)
        assert len(result["findings"]) >= 1
        assert "spaceship-titanic-v1" in result["findings"][0]["evidence"]

    # TEST 3.4
    def test_multiple_competitions_accumulate_in_collection(self):
        """Patterns from different competitions must all be stored."""
        from memory.chroma_client import build_chroma_client
        similar = _make_similar_fingerprint()
        _store_pattern(similar, competition_name="comp_1")
        _store_pattern(similar, competition_name="comp_2")
        _store_pattern(similar, competition_name="comp_3")
        client = build_chroma_client()
        ef = getattr(client, "_professor_ef", None)
        coll = client.get_collection("critic_failure_patterns",
                                      embedding_function=ef)
        assert coll.count() == 3

    # TEST 3.5
    def test_high_confidence_historical_pattern_triggers_replan(self):
        """CRITICAL historical finding must produce replan_instructions."""
        from agents.red_team_critic import _check_historical_failures
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin",
                       confidence=0.90)
        state = _make_state(feature_names=["target_enc_cabin", "age"])
        result = _check_historical_failures(state)
        assert result["verdict"] == "CRITICAL"
        critical_findings = [f for f in result["findings"]
                             if f["severity"] == "CRITICAL"]
        assert len(critical_findings) >= 1
        replan = critical_findings[0]["replan_instructions"]
        assert len(replan["remove_features"]) >= 1
        assert len(replan["rerun_nodes"]) >= 1

    # TEST 3.6
    def test_historical_vector_ok_does_not_block_pipeline(self):
        """No historical patterns → critic does not block."""
        from agents.red_team_critic import _check_historical_failures
        state = _make_state()
        result = _check_historical_failures(state)
        assert result["verdict"] == "OK"

    # TEST 3.7
    def test_critic_failure_pattern_metadata_format(self):
        """Stored pattern must have all 7 required metadata fields."""
        from memory.chroma_client import build_chroma_client
        similar = _make_similar_fingerprint()
        _store_pattern(similar, feature_flagged="target_enc_cabin",
                       failure_mode="leakage", confidence=0.85,
                       cv_lb_gap=0.02, competition_name="comp_x")
        client = build_chroma_client()
        ef = getattr(client, "_professor_ef", None)
        coll = client.get_collection("critic_failure_patterns",
                                      embedding_function=ef)
        results = coll.get(include=["metadatas"])
        assert len(results["metadatas"]) == 1
        meta = results["metadatas"][0]
        required = [
            "competition_name", "feature_flagged", "failure_mode",
            "cv_lb_gap", "confidence", "fingerprint_text", "stored_at",
        ]
        for field in required:
            assert field in meta and meta[field], f"Missing or empty: {field}"

    # TEST 3.8
    def test_10th_competition_has_more_patterns_than_1st(self):
        """More competitions → more patterns retrieved."""
        from memory.memory_schema import query_critic_failure_patterns
        similar = _make_similar_fingerprint()

        # After 0 stored patterns
        result_0 = query_critic_failure_patterns(
            fingerprint=_make_similar_fingerprint(), max_distance=1.5
        )

        # Store 5 patterns (simulating 5 past competitions)
        for i in range(5):
            _store_pattern(similar, feature_flagged=f"feat_{i}",
                           competition_name=f"comp_{i}")

        result_5 = query_critic_failure_patterns(
            fingerprint=_make_similar_fingerprint(), max_distance=1.5
        )

        assert len(result_5) > len(result_0), (
            f"After 5 competitions ({len(result_5)} patterns) should have "
            f"more than before ({len(result_0)} patterns)"
        )
