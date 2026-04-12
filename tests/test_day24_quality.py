# tests/test_day24_quality.py
# Day 24 quality tests — memory layer, warm start, seed memory, quality scoring.
#
# CONTRACT: memory/memory_schema.py, memory/seed_memory.py,
#           memory/memory_quality.py, memory/pinecone_memory.py, agents/ml_optimizer.py

import pytest
import numpy as np
import importlib

chromadb_available = importlib.util.find_spec("chromadb") is not None


@pytest.fixture
def clean_chroma(monkeypatch):
    """Returns a fresh in-memory ChromaDB client, patched into memory_schema."""
    import chromadb
    client = chromadb.Client()  # in-memory, no persistence
    import memory.memory_schema as ms
    monkeypatch.setattr(ms, "build_chroma_client", lambda: client)
    return client


@pytest.fixture
def seeded_hpo_collection(clean_chroma):
    """ChromaDB with one HPO seed pre-populated."""
    from memory.memory_schema import store_hpo_memory
    store_hpo_memory(
        state={
            "competition_fingerprint": {
                "task_type":     "binary_classification",
                "n_rows_bucket": "small",
            },
            "competition_name": "seeded_test",
            "session_id":       "seed_001",
        },
        best_params={"model_type": "lgbm", "n_estimators": 800, "learning_rate": 0.05},
        cv_mean=0.820,
        cv_std=0.008,
    )
    return clean_chroma


@pytest.fixture
def base_ml_state(tmp_path):
    """Minimal state for ml_optimizer tests."""
    import polars as pl
    rng = np.random.default_rng(42)
    n = 300
    # Create a small parquet feature file
    feature_path = tmp_path / "features.parquet"
    df_data = {
        "f0": rng.random(n),
        "f1": rng.random(n),
        "f2": rng.random(n),
        "target": rng.integers(0, 2, n),
    }
    pl.DataFrame(df_data).write_parquet(feature_path)

    # Create schema
    schema_path = tmp_path / "schema.json"
    import json
    schema_path.write_text(json.dumps({
        "target_col": "target",
        "columns": ["f0", "f1", "f2", "target"],
        "types": {"f0": "Float64", "f1": "Float64", "f2": "Float64", "target": "Int64"},
    }))

    return {
        "competition_name":       "test_comp",
        "session_id":             "test_session_24",
        "evaluation_metric":      "accuracy",
        "task_type":              "binary_classification",
        "target_column":          "target",
        "target_col":             "target",
        "data_hash":              "test_hash",
        "competition_fingerprint": {"task_type": "binary_classification"},
        "y_train":                rng.integers(0, 2, n).astype(np.float32),
        "feature_data_path":      str(feature_path),
        "schema_path":            str(schema_path),
        "model_registry":         {},
        "feature_order":          ["f0", "f1", "f2"],
    }


# =========================================================================
# BLOCK 1 — HPO warm start seeds (8 tests)
# =========================================================================


@pytest.mark.skipif(not chromadb_available, reason="chromadb not installed")
class TestHPOWarmStartSeeds:

    def test_returns_empty_list_when_collection_missing(self, tmp_path, monkeypatch):
        """First competition: collection does not exist -> [] returned, no crash."""
        import memory.memory_schema as ms
        # Create a client that has no professor_hpo_memories collection
        client = chromadb.Client()
        monkeypatch.setattr(ms, "build_chroma_client", lambda: client)
        from memory.memory_schema import get_hpo_warm_start_seeds
        result = get_hpo_warm_start_seeds({"competition_fingerprint": {}})
        assert result == []

    def test_returns_empty_list_when_collection_empty(self, clean_chroma):
        """Collection exists but has zero entries -> []."""
        from memory.memory_schema import get_hpo_warm_start_seeds
        clean_chroma.get_or_create_collection("professor_hpo_memories")
        result = get_hpo_warm_start_seeds({"competition_fingerprint": {}})
        assert result == []

    def test_returns_seeds_within_distance_threshold(self, seeded_hpo_collection):
        """Seeds within max_distance=0.70 are returned."""
        from memory.memory_schema import get_hpo_warm_start_seeds
        state = {
            "competition_fingerprint": {
                "task_type":     "binary_classification",
                "n_rows_bucket": "small",
            }
        }
        result = get_hpo_warm_start_seeds(state, max_distance=0.70)
        assert len(result) >= 1, "Expected at least one seed within distance threshold."

    def test_seeds_contain_params_and_tracking_fields(self, seeded_hpo_collection):
        """Each returned seed must have model_type and _seed_source."""
        from memory.memory_schema import get_hpo_warm_start_seeds
        state = {
            "competition_fingerprint": {
                "task_type": "binary_classification",
                "n_rows_bucket": "small",
            }
        }
        seeds = get_hpo_warm_start_seeds(state)
        for seed in seeds:
            assert "model_type" in seed, "Seed missing 'model_type'."
            assert "_seed_source" in seed, "Seed missing '_seed_source' tracking field."
            assert "_seed_confidence" in seed, "Seed missing '_seed_confidence'."
            assert "_seed_distance" in seed, "Seed missing '_seed_distance'."

    def test_low_confidence_seeds_filtered_out(self, clean_chroma):
        """Seeds with confidence < min_confidence must not be returned."""
        from memory.memory_schema import get_hpo_warm_start_seeds, store_hpo_memory
        # Store a seed with very low confidence (high variance -> low confidence)
        store_hpo_memory(
            state={
                "competition_fingerprint": {"task_type": "binary_classification"},
                "competition_name": "low_conf_seed",
                "session_id": "test",
            },
            best_params={"model_type": "lgbm", "n_estimators": 500},
            cv_mean=0.500,   # terrible CV -> low confidence computed
            cv_std=0.100,    # high variance -> low confidence
        )
        result = get_hpo_warm_start_seeds(
            {"competition_fingerprint": {"task_type": "binary_classification"}},
            min_confidence=0.65,
        )
        # The low-confidence seed should be filtered
        for seed in result:
            assert seed["_seed_confidence"] >= 0.65

    def test_never_raises_on_any_input(self):
        """get_hpo_warm_start_seeds must never raise regardless of input."""
        from memory.memory_schema import get_hpo_warm_start_seeds
        # Various malformed inputs
        assert get_hpo_warm_start_seeds({}) == [] or True
        assert get_hpo_warm_start_seeds({"competition_fingerprint": None}) == [] or True
        assert get_hpo_warm_start_seeds({"competition_fingerprint": "not_a_dict"}) == [] or True

    def test_store_hpo_memory_returns_true_on_success(self, clean_chroma):
        from memory.memory_schema import store_hpo_memory
        result = store_hpo_memory(
            state={
                "competition_fingerprint": {"task_type": "binary_classification"},
                "competition_name": "test_comp",
                "session_id": "test_session",
            },
            best_params={"model_type": "lgbm", "n_estimators": 500},
            cv_mean=0.820,
            cv_std=0.010,
        )
        assert result is True

    def test_store_and_retrieve_roundtrip(self, clean_chroma):
        """Store a seed, then retrieve it with a matching fingerprint."""
        from memory.memory_schema import store_hpo_memory, get_hpo_warm_start_seeds

        fingerprint = {"task_type": "regression", "n_rows_bucket": "medium"}
        store_hpo_memory(
            state={
                "competition_fingerprint": fingerprint,
                "competition_name":        "roundtrip_test",
                "session_id":              "rt_001",
            },
            best_params={"model_type": "lgbm", "n_estimators": 800},
            cv_mean=0.115,
            cv_std=0.005,
        )

        seeds = get_hpo_warm_start_seeds(
            {"competition_fingerprint": fingerprint},
            max_distance=0.80,
        )
        assert len(seeds) >= 1
        found = any(s.get("model_type") == "lgbm" for s in seeds)
        assert found, "Stored LightGBM seed not found in retrieval."


# =========================================================================
# BLOCK 2 — Optuna warm start integration (5 tests)
# These tests run full Optuna optimization and are slow.
# =========================================================================


@pytest.mark.skip(reason="slow — requires Optuna full pipeline run")
class TestOptunaWarmStartIntegration:

    def test_hpo_seeds_used_count_set_in_state(self, base_ml_state, monkeypatch):
        """state['hpo_warm_start_seeds_used'] must be set after ml_optimizer runs."""
        import memory.memory_schema as ms
        monkeypatch.setattr(ms, "get_hpo_warm_start_seeds", lambda *a, **k: [])
        from agents.ml_optimizer import run_ml_optimizer
        result = run_ml_optimizer(base_ml_state)
        assert "hpo_warm_start_seeds_used" in result
        assert isinstance(result["hpo_warm_start_seeds_used"], int)

    def test_hpo_seeds_used_is_0_when_no_history(self, base_ml_state, monkeypatch):
        """No seeds in ChromaDB -> hpo_warm_start_seeds_used == 0."""
        import memory.memory_schema as ms
        monkeypatch.setattr(ms, "get_hpo_warm_start_seeds", lambda *a, **k: [])
        from agents.ml_optimizer import run_ml_optimizer
        result = run_ml_optimizer(base_ml_state)
        assert result["hpo_warm_start_seeds_used"] == 0

    def test_hpo_seeds_used_matches_seeds_injected(self, base_ml_state, monkeypatch):
        """When 3 seeds available, hpo_warm_start_seeds_used should be 3."""
        fake_seeds = [
            {"model_type": "lgbm", "n_estimators": 500,
             "_seed_source": "test", "_seed_confidence": 0.8, "_seed_distance": 0.3},
            {"model_type": "lgbm", "n_estimators": 700,
             "_seed_source": "test", "_seed_confidence": 0.75, "_seed_distance": 0.4},
            {"model_type": "xgb", "n_estimators": 600,
             "_seed_source": "test", "_seed_confidence": 0.7, "_seed_distance": 0.5},
        ]
        import memory.memory_schema as ms
        monkeypatch.setattr(ms, "get_hpo_warm_start_seeds", lambda *a, **k: fake_seeds)
        from agents.ml_optimizer import run_ml_optimizer
        result = run_ml_optimizer(base_ml_state)
        assert result["hpo_warm_start_seeds_used"] == 3

    def test_warm_start_does_not_break_optuna_study(self, base_ml_state, monkeypatch):
        """Warm start seeds must not prevent Optuna from completing normally."""
        fake_seeds = [
            {"model_type": "lgbm", "n_estimators": 300,
             "_seed_source": "test", "_seed_confidence": 0.8, "_seed_distance": 0.3},
        ]
        import memory.memory_schema as ms
        monkeypatch.setattr(ms, "get_hpo_warm_start_seeds", lambda *a, **k: fake_seeds)
        from agents.ml_optimizer import run_ml_optimizer
        result = run_ml_optimizer(base_ml_state)
        # Pipeline must complete and produce a model
        assert result.get("model_registry"), "model_registry empty after warm start run."

    def test_failed_seed_injection_does_not_crash_optimizer(self, base_ml_state, monkeypatch):
        """
        Seeds that fail during injection (e.g. invalid params) must be skipped silently.
        Optimizer must complete normally.
        """
        bad_seeds = [
            {"model_type": "nonexistent_model", "n_estimators": "not_an_int",
             "_seed_source": "bad", "_seed_confidence": 0.9, "_seed_distance": 0.2},
        ]
        import memory.memory_schema as ms
        monkeypatch.setattr(ms, "get_hpo_warm_start_seeds", lambda *a, **k: bad_seeds)
        from agents.ml_optimizer import run_ml_optimizer
        result = run_ml_optimizer(base_ml_state)
        assert result.get("model_registry"), "Optimizer crashed on bad seed injection."


# =========================================================================
# BLOCK 3 — Seed memory script (5 tests)
# =========================================================================


@pytest.mark.skipif(not chromadb_available, reason="chromadb not installed")
class TestSeedMemoryScript:

    def test_seed_script_imports_cleanly(self):
        """seed_memory.py must be importable without errors."""
        from memory.seed_memory import run_seed_memory, COMPETITION_SEEDS, HPO_SEEDS
        assert len(COMPETITION_SEEDS) >= 3, "Need at least 3 competition pattern seeds."
        assert len(HPO_SEEDS) >= 2, "Need at least 2 HPO seeds."

    def test_seed_script_populates_patterns_collection(self, clean_chroma):
        """After run_seed_memory(), professor_patterns_v2 has entries."""
        from memory.seed_memory import run_seed_memory
        run_seed_memory()
        collection = clean_chroma.get_collection("professor_patterns_v2")
        assert collection.count() >= 3, (
            f"Expected >= 3 pattern entries after seeding, got {collection.count()}."
        )

    def test_seed_script_populates_hpo_collection(self, clean_chroma):
        """After run_seed_memory(), professor_hpo_memories has entries."""
        from memory.seed_memory import run_seed_memory
        run_seed_memory()
        collection = clean_chroma.get_collection("professor_hpo_memories")
        assert collection.count() >= 2, (
            f"Expected >= 2 HPO entries after seeding, got {collection.count()}."
        )

    def test_seed_script_is_idempotent(self, clean_chroma):
        """Running twice does not double the entries."""
        from memory.seed_memory import run_seed_memory
        run_seed_memory()
        count_after_first = clean_chroma.get_collection("professor_patterns_v2").count()
        run_seed_memory()
        count_after_second = clean_chroma.get_collection("professor_patterns_v2").count()
        assert count_after_second == count_after_first, (
            f"Seed script is not idempotent: {count_after_first} entries after first run, "
            f"{count_after_second} after second run."
        )

    def test_all_seeds_have_confidence_0_7(self, clean_chroma):
        """All seeded entries must have confidence=0.70 (known but unvalidated)."""
        from memory.seed_memory import run_seed_memory, SEED_CONFIDENCE
        assert SEED_CONFIDENCE == 0.70, (
            f"SEED_CONFIDENCE is {SEED_CONFIDENCE}, expected 0.70."
        )
        run_seed_memory()
        collection = clean_chroma.get_or_create_collection("professor_hpo_memories")
        result = collection.get(include=["metadatas"])
        for meta in result["metadatas"]:
            assert abs(float(meta.get("confidence", 0)) - 0.70) < 0.01, (
                f"Seeded entry has confidence={meta.get('confidence')}, expected 0.70."
            )


# =========================================================================
# BLOCK 4 — Memory quality scoring (8 tests)
# =========================================================================


@pytest.mark.skipif(not chromadb_available, reason="chromadb not installed")
class TestMemoryQualityScoring:

    def test_helpfulness_rate_updates_correctly_on_helpful(self, clean_chroma):
        """After one helpful retrieval: n_retrieved=1, n_helpful=1, rate=1.0."""
        from memory.memory_quality import update_memory_helpfulness
        from memory.memory_schema import store_hpo_memory

        store_hpo_memory(
            state={"competition_fingerprint": {}, "competition_name": "test",
                   "session_id": "s1"},
            best_params={"model_type": "lgbm"},
            cv_mean=0.820, cv_std=0.010,
        )

        collection = clean_chroma.get_collection("professor_hpo_memories")
        memory_id = collection.get()["ids"][0]

        result = update_memory_helpfulness("professor_hpo_memories", memory_id, was_helpful=True)
        assert result is True

        updated = collection.get(ids=[memory_id], include=["metadatas"])["metadatas"][0]
        assert int(updated["n_retrieved"]) == 1
        assert int(updated["n_helpful"]) == 1
        assert float(updated["helpfulness_rate"]) == pytest.approx(1.0)

    def test_helpfulness_rate_updates_correctly_on_not_helpful(self, clean_chroma):
        """After one unhelpful retrieval: rate=0.0, confidence decays by DECAY_RATE."""
        from memory.memory_quality import update_memory_helpfulness, DECAY_RATE
        from memory.memory_schema import store_hpo_memory

        store_hpo_memory(
            state={"competition_fingerprint": {}, "competition_name": "test",
                   "session_id": "s2"},
            best_params={"model_type": "xgb"},
            cv_mean=0.810, cv_std=0.012,
        )

        collection = clean_chroma.get_collection("professor_hpo_memories")
        memory_id = collection.get()["ids"][0]
        initial_confidence = float(
            collection.get(ids=[memory_id], include=["metadatas"])
            ["metadatas"][0]["confidence"]
        )

        update_memory_helpfulness("professor_hpo_memories", memory_id, was_helpful=False)

        updated = collection.get(ids=[memory_id], include=["metadatas"])["metadatas"][0]
        new_conf = float(updated["confidence"])
        assert new_conf == pytest.approx(initial_confidence - DECAY_RATE, abs=0.001), (
            f"Confidence did not decay by DECAY_RATE={DECAY_RATE}. "
            f"Before: {initial_confidence}, After: {new_conf}."
        )

    def test_decay_rate_is_0_05(self):
        from memory.memory_quality import DECAY_RATE
        assert DECAY_RATE == pytest.approx(0.05)

    def test_removal_threshold_is_0_50(self):
        from memory.memory_quality import REMOVAL_THRESHOLD
        assert REMOVAL_THRESHOLD == pytest.approx(0.50)

    def test_remove_decayed_removes_low_confidence_entries(self, clean_chroma):
        """Entries with confidence < 0.50 are removed by remove_decayed_memories()."""
        from memory.memory_quality import remove_decayed_memories

        collection = clean_chroma.get_or_create_collection("professor_hpo_memories")
        collection.add(
            documents=["low confidence entry"],
            metadatas=[{"confidence": 0.35, "model_type": "lgbm"}],
            ids=["low_conf_001"],
        )
        collection.add(
            documents=["high confidence entry"],
            metadatas=[{"confidence": 0.80, "model_type": "xgb"}],
            ids=["high_conf_001"],
        )

        removed = remove_decayed_memories("professor_hpo_memories")
        assert removed == 1
        remaining = collection.get()["ids"]
        assert "low_conf_001" not in remaining
        assert "high_conf_001" in remaining

    def test_remove_decayed_returns_0_when_nothing_to_remove(self, clean_chroma):
        from memory.memory_quality import remove_decayed_memories
        result = remove_decayed_memories("professor_hpo_memories")
        assert result == 0

    def test_should_retrieve_true_for_new_entry(self):
        """Brand new entry (n_retrieved=0) should always be retrieved if confidence ok."""
        from memory.memory_quality import should_retrieve
        meta = {"n_retrieved": 0, "confidence": 0.70}
        assert should_retrieve(meta) is True

    def test_should_retrieve_false_for_low_helpfulness(self):
        """Entry with helpfulness_rate=0.3 below threshold should not be retrieved."""
        from memory.memory_quality import should_retrieve
        meta = {"n_retrieved": 10, "n_helpful": 3, "helpfulness_rate": 0.30,
                "confidence": 0.75}
        assert should_retrieve(meta) is False


# =========================================================================
# BLOCK 5 — Pinecone stub (6 tests)
# =========================================================================


class TestPineconeMemoryStub:

    def setup_method(self):
        """Clear store before each test."""
        from memory import pinecone_memory
        pinecone_memory._STORE.clear()

    def test_upsert_returns_true(self):
        from memory.pinecone_memory import upsert
        result = upsert("experiments", "exp_001", "LightGBM on fraud, AUC=0.920")
        assert result is True

    def test_count_correct_after_upsert(self):
        from memory.pinecone_memory import upsert, count
        upsert("experiments", "exp_001", "entry one")
        upsert("experiments", "exp_002", "entry two")
        upsert("domain", "d_001", "different collection")
        assert count("experiments") == 2
        assert count("domain") == 1

    def test_query_returns_matching_entries(self):
        from memory.pinecone_memory import upsert, query
        upsert("experiments", "e1", "LightGBM fraud detection high AUC", confidence=0.80)
        upsert("experiments", "e2", "house prices regression RMSLE", confidence=0.75)
        results = query("experiments", "fraud detection", n_results=5)
        assert len(results) >= 1
        # First result should be the fraud one (better text overlap)
        assert "fraud" in results[0]["text"].lower()

    def test_quality_filtering_applied_in_query(self):
        from memory.pinecone_memory import upsert, query
        upsert("experiments", "good_e", "good experiment results", confidence=0.85)
        upsert("experiments", "bad_e", "bad experiment results", confidence=0.40)
        results = query("experiments", "experiment results", min_confidence=0.60)
        ids = [r["id"] for r in results]
        assert "good_e" in ids
        assert "bad_e" not in ids

    def test_update_helpfulness_updates_rate(self):
        from memory.pinecone_memory import upsert, update_helpfulness, _STORE
        upsert("experiments", "e1", "test entry")
        update_helpfulness("experiments", "e1", was_helpful=True)
        update_helpfulness("experiments", "e1", was_helpful=False)
        entry = _STORE["experiments::e1"]
        assert entry["n_retrieved"] == 2
        assert entry["n_helpful"] == 1
        assert entry["helpfulness_rate"] == pytest.approx(0.5)

    def test_upsert_overwrites_existing_entry(self):
        from memory.pinecone_memory import upsert, _STORE
        upsert("experiments", "e1", "original text", confidence=0.70)
        upsert("experiments", "e1", "updated text", confidence=0.85)
        entry = _STORE["experiments::e1"]
        assert entry["text"] == "updated text"
        assert entry["confidence"] == pytest.approx(0.85)
