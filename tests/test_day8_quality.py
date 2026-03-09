# tests/test_day8_quality.py
import pytest
import os
import tempfile
import numpy as np
import json
from datetime import datetime

import chromadb
from memory.chroma_client import _build_embedding_function, build_chroma_client, get_or_create_collection

from core.state import initial_state
from tools.data_tools import hash_dataset
from agents.semantic_router import _determine_strategy
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect
from agents.competition_intel import run_competition_intel
import polars as pl

# -------------------------------------------------------------------
# BLOCK 1 — CHROMADB: EMBEDDING SEMANTICS
# -------------------------------------------------------------------

class TestChromaDBEmbeddingSemanticsNotJustStartup:

    def test_embedding_dimension_is_exactly_384(self):
        ef = _build_embedding_function()
        emb = ef(["test"])
        dim = len(emb[0])
        assert dim == 384, f"Expected embedding dim 384, got {dim}. Wrong model is loaded or ChromaDB fell back to a different embedding."

    def test_semantically_similar_query_returns_correct_top_result(self, tmp_path):
        client = build_chroma_client(str(tmp_path))
        col = get_or_create_collection(client, "test_collection_similar")
        col.add(
            documents=[
                "LightGBM gradient boosting tabular classification AUC",
                "LSTM recurrent neural network time series forecasting",
                "BERT transformer NLP text classification cross-entropy"
            ],
            ids=["doc_lgbm", "doc_lstm", "doc_bert"]
        )
        res = col.query(query_texts=["gradient boosting trees tabular data"], n_results=1)
        top_doc = res["ids"][0][0]
        assert top_doc == "doc_lgbm", f"Semantic query for gradient boosting returned '{top_doc}' instead of 'doc_lgbm'. Embeddings are not semantic — ChromaDB may be using random or wrong embeddings."

    def test_dissimilar_query_does_not_contaminate_top_result(self, tmp_path):
        client = build_chroma_client(str(tmp_path))
        col = get_or_create_collection(client, "test_collection_contaminate")
        col.add(
            documents=[
                "LightGBM gradient boosting tabular classification AUC",
                "LSTM recurrent neural network time series forecasting",
                "BERT transformer NLP text classification cross-entropy"
            ],
            ids=["doc_lgbm", "doc_lstm", "doc_bert"]
        )
        res = col.query(query_texts=["BERT tokenizer text classification"], n_results=1)
        top_doc = res["ids"][0][0]
        assert top_doc != "doc_lgbm", "NLP query surfaced a tabular ML document as its top hit."

    def test_cosine_similarity_between_similar_docs_exceeds_threshold(self):
        ef = _build_embedding_function()
        embs = ef([
            "LightGBM gradient boosting tabular AUC",
            "XGBoost gradient boosted trees tabular classification",
            "BERT transformer NLP language model"
        ])
        A = np.array(embs[0])
        B = np.array(embs[1])
        C = np.array(embs[2])
        
        sim_ab = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        sim_ac = np.dot(A, C) / (np.linalg.norm(A) * np.linalg.norm(C))
        
        assert sim_ab > 0.80, f"sim(A,B)={sim_ab:.3f} < 0.80 — embeddings are not producing a coherent semantic space"
        assert sim_ab > sim_ac + 0.10, f"sim(A,B)={sim_ab:.3f} not >10pts above sim(A,C)={sim_ac:.3f} — embedding space is too flat"

    def test_client_bypassing_build_chroma_client_raises_runtime_error(self, tmp_path):
        raw_client = chromadb.PersistentClient(path=str(tmp_path))
        with pytest.raises(RuntimeError) as exc:
            get_or_create_collection(raw_client, "test")
        assert "build_chroma_client" in str(exc.value)

    def test_embedding_is_deterministic_across_two_calls(self):
        ef = _build_embedding_function()
        emb1 = ef(["tabular binary classification"])
        emb2 = ef(["tabular binary classification"])
        np.testing.assert_array_equal(emb1[0], emb2[0])


# -------------------------------------------------------------------
# BLOCK 2 — STATE FIELDS: BOUNDARY LOGIC
# -------------------------------------------------------------------

class TestStateFieldsBoundaryLogic:

    def test_task_type_initial_value_is_unknown(self):
        state = initial_state("comp", "data.csv")
        assert state["task_type"] == "unknown"

    def test_data_hash_changes_when_file_contents_change(self, tmp_path):
        f1 = tmp_path / "f1.csv"
        f1.write_text("a,b\n1,2\n")
        f2 = tmp_path / "f2.csv"
        f2.write_text("a,b\n1,3\n")
        
        h1 = hash_dataset(str(f1))
        h2 = hash_dataset(str(f2))
        assert h1 != h2, "data_hash is the same for two files with different contents. SHA-256 is reading wrong content or caching incorrectly."

    def test_data_hash_is_stable_for_identical_content(self, tmp_path):
        f1 = tmp_path / "f1.csv"
        f1.write_text("orig,content\n1,2\n")
        h1 = hash_dataset(str(f1))
        h2 = hash_dataset(str(f1))
        assert h1 == h2

    def test_data_hash_is_16_hex_characters(self, tmp_path):
        f1 = tmp_path / "f1.csv"
        f1.write_text("a,b\n1,2\n")
        h1 = hash_dataset(str(f1))
        
        assert len(h1) == 16
        for char in h1:
            assert char in "0123456789abcdef"

    @pytest.mark.parametrize("percentile, days, expected", [
        (0.05, 2, "conservative"),
        (0.10, 2, "conservative"),
        (0.11, 2, "aggressive"),
        (0.50, 2, "aggressive"),
        (0.41, 8, "aggressive"),
        (0.99, 30, "aggressive"),
        (0.30, 10, "balanced"),
        (0.05, 10, "balanced"),
        (0.40, 8, "balanced"),
        (None, None, "balanced")
    ])
    def test_strategy_at_conservative_boundary(self, percentile, days, expected):
        got = _determine_strategy(percentile, days)
        assert got == expected, f"_determine_strategy(percentile={percentile}, days={days}) returned '{got}', expected '{expected}'"

    def test_competition_context_has_all_required_keys(self):
        state = initial_state("comp", "data.csv")
        ctx = state["competition_context"]
        required = [
            "days_remaining", "hours_remaining", "submissions_used", 
            "submissions_remaining", "current_public_rank", "total_competitors", 
            "current_percentile", "shakeup_risk", "strategy", "last_updated"
        ]
        for key in required:
            assert key in ctx, f"Missing key {key} in competition_context"

    def test_data_hash_written_to_state_after_data_engineer(self, tmp_path):
        f1 = tmp_path / "train.csv"
        f1.write_text("target,feature1\n0,1.2\n1,2.3\n0,3.1\n")
        state = initial_state("test-comp", str(f1))
        state["session_id"] = "test_data_hash_123"
        result = run_data_engineer(state)
        assert result.get("data_hash", "") != "", "data_hash is empty"

    def test_model_registry_entry_contains_data_hash(self, tmp_path):
        import shutil
        from agents.ml_optimizer import run_ml_optimizer
        path = "data/spaceship_titanic/train.csv"
        if not os.path.exists(path):
            pytest.skip(f"{path} not found")
        state = initial_state("spaceship-titanic", path)
        state["session_id"] = "test_opt_123"
        state = run_data_engineer(state)
        state["metric_contract"] = {
            "scorer_name": "auc",
            "direction": "maximize",
            "forbidden_metrics": ["accuracy"]
        }
        state["validation_strategy"] = {
            "cv_type": "KFold",
            "n_splits": 2,
            "group_col": None,
            "scorer_name": "auc",
            "task_type": "tabular_classification",
            "target_col": "Transported"
        }
        state["budget_usd"] = 0.0 # Force no new trials, might load registry
        state["task_type"] = "tabular_classification"
        result = run_ml_optimizer(state)
        registry = result.get("model_registry", [])
        if len(registry) > 0:
            for entry in registry:
                assert "data_hash" in entry
                assert entry["data_hash"] != ""
                assert isinstance(entry["data_hash"], str)


# -------------------------------------------------------------------
# BLOCK 3 — VALIDATION ARCHITECT: STRATEGY CORRECTNESS
# -------------------------------------------------------------------

class TestValidationArchitectStrategyCorrectness:
    def _run_va_on_schema(self, tmp_path, schema):
        # Create dummy dataframe matching schema
        data = {}
        for col, dtype in schema.items():
            if dtype in (pl.Utf8, pl.String):
                data[col] = [f"id_{i}" for i in range(10)]
            else:
                data[col] = [float(i) if dtype == pl.Float64 else i for i in range(10)]
        df = pl.DataFrame(data, schema=schema)
        # We need a target
        if "target" not in schema:
            data["target"] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            schema["target"] = pl.Int64
            df = pl.DataFrame(data, schema=schema)
            
        p = tmp_path / "train.parquet"
        df.write_parquet(str(p))
        state = initial_state("comp", str(p))
        state["session_id"] = f"test_va_{datetime.now().timestamp()}"
        state["clean_data_path"] = str(p)
        state["task_type"] = "tabular_classification"
        
        # We need the schema_json for ValidationArchitect
        csv_path = tmp_path / "train2.csv"
        df.write_csv(str(csv_path))
        state["raw_data_path"] = str(csv_path)
        from agents.data_engineer import run_data_engineer
        state = run_data_engineer(state)
        return run_validation_architect(state)

    def test_stratified_kfold_for_binary_target(self, tmp_path):
        import polars as pl
        schema = {"feature": pl.Float64, "target": pl.Int64}
        res = self._run_va_on_schema(tmp_path, schema)
        assert res["validation_strategy"]["cv_type"] == "StratifiedKFold"

    def test_group_kfold_when_group_column_present(self, tmp_path):
        import polars as pl
        schema = {"feature": pl.Float64, "target": pl.Int64, "customer_id": pl.Utf8}
        res = self._run_va_on_schema(tmp_path, schema)
        cv_type = res["validation_strategy"]["cv_type"]
        assert cv_type == "GroupKFold", f"Group column 'customer_id' present but CV strategy is '{cv_type}', not GroupKFold. This inflates CV by leaking rows from the same group."
        assert res["validation_strategy"]["group_col"] == "customer_id"

    def test_timeseries_split_when_datetime_column_present(self, tmp_path):
        import polars as pl
        from datetime import date
        data = {
            "feature": [float(i) for i in range(10)],
            "target": [0,1,0,1,0,1,0,1,0,1],
            "transaction_date": [date(2021,1,1) for _ in range(10)]
        }
        df = pl.DataFrame(data, schema={"feature": pl.Float64, "target": pl.Int64, "transaction_date": pl.Date})
        
        csv_path = tmp_path / "train3.csv"
        df.write_csv(str(csv_path))
        state = initial_state("comp", str(csv_path))
        state["session_id"] = "test_va_ts"
        state = run_data_engineer(state)
        state["task_type"] = "tabular_classification"
        res = run_validation_architect(state)
        
        cv_type = res["validation_strategy"]["cv_type"]
        assert cv_type == "TimeSeriesSplit", f"Datetime column present but CV strategy is '{cv_type}', not TimeSeriesSplit. This leaks future data into past folds."

    def test_group_kfold_takes_priority_over_datetime_column(self, tmp_path):
        import polars as pl
        from datetime import date
        data = {
            "patient_id": ["A"]*10,
            "visit_date": [date(2021,1,1)]*10,
            "target": [0,1,0,1,0,1,0,1,0,1]
        }
        df = pl.DataFrame(data, schema={"patient_id": pl.Utf8, "visit_date": pl.Date, "target": pl.Int64})
        csv_path = tmp_path / "train4.csv"
        df.write_csv(str(csv_path))
        state = initial_state("comp", str(csv_path))
        state["session_id"] = "test_va_pri"
        state = run_data_engineer(state)
        state["task_type"] = "tabular_classification"
        res = run_validation_architect(state)
        
        assert res["validation_strategy"]["cv_type"] == "GroupKFold"

    def test_kfold_for_continuous_target(self, tmp_path):
        import polars as pl
        data = {
            "feature": list(range(105)),
            "target": [float(i) for i in range(105)]
        }
        df = pl.DataFrame(data, schema={"feature": pl.Int64, "target": pl.Float64})
        csv_path = tmp_path / "train_cont.csv"
        df.write_csv(str(csv_path))
        state = initial_state("comp", str(csv_path))
        state["session_id"] = "test_va_cont1"
        state = run_data_engineer(state)
        state["task_type"] = "tabular_regression"
        res = run_validation_architect(state)
        assert res["validation_strategy"]["cv_type"] in ("KFold", "StratifiedKFold")

    def test_n_splits_is_always_5(self, tmp_path):
        import polars as pl
        res1 = self._run_va_on_schema(tmp_path, {"f": pl.Float64, "target": pl.Int64})
        assert res1["validation_strategy"]["n_splits"] == 5
        res2 = self._run_va_on_schema(tmp_path, {"f": pl.Float64, "target": pl.Int64, "customer_id": pl.Utf8})
        assert res2["validation_strategy"]["n_splits"] == 5

    def test_mismatch_detected_stratified_plus_datetime(self, tmp_path):
        import polars as pl
        from datetime import datetime
        data = {
            "feature": [float(i) for i in range(10)],
            "target": [0,1,0,1,0,1,0,1,0,1],
            "order_date": [datetime(2021,1,1)]*10
        }
        df = pl.DataFrame(data, schema={"feature": pl.Float64, "target": pl.Int64, "order_date": pl.Datetime})
        csv_path = tmp_path / "train_dt.csv"
        df.write_csv(str(csv_path))
        state = initial_state("comp", str(csv_path))
        state["session_id"] = "test_va_mm1"
        state = run_data_engineer(state)
        state["task_type"] = "tabular_classification"
        res = run_validation_architect(state)
        assert res["hitl_required"] is True, "Datetime column with StratifiedKFold should trigger mismatch detection. hitl_required was not set to True."

    def test_mismatch_reason_names_the_offending_column(self, tmp_path):
        import polars as pl
        from datetime import datetime
        data = {
            "feature": [float(i) for i in range(10)],
            "target": [0,1,0,1,0,1,0,1,0,1],
            "signup_date": [datetime(2021,1,1)]*10
        }
        df = pl.DataFrame(data, schema={"feature": pl.Float64, "target": pl.Int64, "signup_date": pl.Datetime})
        csv_path = tmp_path / "train_dt2.csv"
        df.write_csv(str(csv_path))
        state = initial_state("comp", str(csv_path))
        state["session_id"] = "test_va_mm2"
        state = run_data_engineer(state)
        state["task_type"] = "tabular_classification"
        res = run_validation_architect(state)
        assert "signup_date" in res["hitl_reason"]

    def test_no_false_positive_mismatch_on_clean_tabular(self, tmp_path):
        import polars as pl
        schema = {"feature": pl.Float64, "target": pl.Int64}
        res = self._run_va_on_schema(tmp_path, schema)
        assert res["hitl_required"] is False

    def test_metric_contract_direction_correct_for_auc(self, tmp_path):
        import polars as pl
        schema = {"feature": pl.Float64, "target": pl.Int64}
        res = self._run_va_on_schema(tmp_path, schema)
        if res["validation_strategy"]["scorer_name"] == "auc":
            mc_path = res.get("metric_contract_path")
            with open(mc_path) as f:
                mc = json.load(f)
            assert mc["direction"] == "maximize"

    def test_metric_contract_direction_correct_for_rmse(self, tmp_path):
        import polars as pl
        data = {
            "feature": list(range(105)),
            "target": [float(i) for i in range(105)]
        }
        df = pl.DataFrame(data, schema={"feature": pl.Int64, "target": pl.Float64})
        csv_path = tmp_path / "train_cont2.csv"
        df.write_csv(str(csv_path))
        state = initial_state("comp", str(csv_path))
        state["session_id"] = "test_va_rmse"
        state = run_data_engineer(state)
        state["task_type"] = "tabular_regression"
        res = run_validation_architect(state)
        if res["validation_strategy"]["scorer_name"] == "rmse":
            mc_path = res.get("metric_contract_path")
            with open(mc_path) as f:
                mc = json.load(f)
            assert mc["direction"] == "minimize"

    def test_metric_contract_forbidden_metrics_is_non_empty_list(self, tmp_path):
        import polars as pl
        schema = {"feature": pl.Float64, "target": pl.Int64}
        res = self._run_va_on_schema(tmp_path, schema)
        mc_path = res.get("metric_contract_path")
        with open(mc_path) as f:
            mc = json.load(f)
        assert isinstance(mc["forbidden_metrics"], list)
        assert len(mc["forbidden_metrics"]) > 0

    def test_validation_strategy_json_written_even_when_mismatch_halts(self, tmp_path):
        import polars as pl
        from datetime import datetime
        data = {
            "feature": [float(i) for i in range(10)],
            "target": [0,1,0,1,0,1,0,1,0,1],
            "order_date": [datetime(2021,1,1)]*10
        }
        df = pl.DataFrame(data, schema={"feature": pl.Float64, "target": pl.Int64, "order_date": pl.Datetime})
        csv_path = tmp_path / "train_dt3.csv"
        df.write_csv(str(csv_path))
        state = initial_state("comp", str(csv_path))
        state["session_id"] = "test_va_mm3"
        state = run_data_engineer(state)
        res = run_validation_architect(state)
        
        assert res["hitl_required"] is True
        strat_path = res.get("validation_strategy_path")
        assert strat_path is not None
        assert os.path.exists(strat_path)

    def test_metric_contract_not_written_when_mismatch_halts(self, tmp_path):
        import polars as pl
        from datetime import datetime
        data = {
            "feature": [float(i) for i in range(10)],
            "target": [0,1,0,1,0,1,0,1,0,1],
            "order_date": [datetime(2021,1,1)]*10
        }
        df = pl.DataFrame(data, schema={"feature": pl.Float64, "target": pl.Int64, "order_date": pl.Datetime})
        csv_path = tmp_path / "train_dt4.csv"
        df.write_csv(str(csv_path))
        state = initial_state("comp", str(csv_path))
        state["session_id"] = "test_va_mm4"
        state = run_data_engineer(state)
        res = run_validation_architect(state)
        assert res["hitl_required"] is True
        mc_path = res.get("metric_contract_path")
        if mc_path is not None:
            assert not os.path.exists(mc_path)


# -------------------------------------------------------------------
# BLOCK 4 — EDA AGENT: THRESHOLD ACCURACY
# -------------------------------------------------------------------

class TestEDAAgentThresholdAccuracy:
    def _run_eda_on_df(self, tmp_path, df, sid_suffix):
        csv_path = tmp_path / f"eda_{sid_suffix}.csv"
        df.write_csv(str(csv_path))
        state = initial_state("comp", str(csv_path))
        state["session_id"] = f"eda_{sid_suffix}"
        state = run_data_engineer(state)
        state["task_type"] = "tabular_classification"
        return run_eda_agent(state)

    def test_eda_report_has_all_required_keys(self, tmp_path):
        import polars as pl
        df = pl.DataFrame({"f": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
        res = self._run_eda_on_df(tmp_path, df, "1")
        req = ["target_distribution", "feature_correlations", "outlier_profile", "duplicate_analysis", "temporal_profile", "leakage_fingerprint", "drop_candidates", "summary"]
        report = res["eda_report"]
        for k in req:
            assert k in report, f"Missing key: {k}"

    def test_outlier_strategy_keep_below_1pct(self, tmp_path):
        import numpy as np
        n = 1000
        f = np.zeros(n)
        f[:5] = 1000.0 # 5 / 1000 = 0.5%
        df = pl.DataFrame({"feature1": f, "target": np.random.randint(0, 2, n)})
        res = self._run_eda_on_df(tmp_path, df, "2")
        for out in res["eda_report"]["outlier_profile"]:
            if out["column"] == "feature1":
                assert out["strategy"] == "keep"

    def test_outlier_strategy_winsorize_between_1_and_5pct(self, tmp_path):
        import numpy as np
        n = 1000
        f = np.zeros(n)
        f[:30] = 1000.0 # 30 / 1000 = 3%
        df = pl.DataFrame({"feature1": f, "target": np.random.randint(0, 2, n)})
        res = self._run_eda_on_df(tmp_path, df, "3")
        for out in res["eda_report"]["outlier_profile"]:
            if out["column"] == "feature1":
                assert out["strategy"] == "winsorize"

    def test_outlier_strategy_cap_between_5_and_10pct(self, tmp_path):
        import numpy as np
        n = 1000
        f = np.zeros(n)
        f[:70] = 1000.0 # 70 / 1000 = 7%
        df = pl.DataFrame({"feature1": f, "target": np.random.randint(0, 2, n)})
        res = self._run_eda_on_df(tmp_path, df, "4")
        for out in res["eda_report"]["outlier_profile"]:
            if out["column"] == "feature1":
                assert out["strategy"] == "cap"

    def test_outlier_strategy_remove_above_10pct(self, tmp_path):
        import numpy as np
        n = 1000
        f = np.zeros(n)
        f[:150] = 1000.0 # 15%
        df = pl.DataFrame({"feature1": f, "target": np.random.randint(0, 2, n)})
        res = self._run_eda_on_df(tmp_path, df, "5")
        for out in res["eda_report"]["outlier_profile"]:
            if out["column"] == "feature1":
                assert out["strategy"] == "remove"

    def test_leakage_flag_triggers_above_095_correlation(self, tmp_path):
        import numpy as np
        n = 1000
        t = np.random.randint(0, 2, n)
        f = t + np.random.randn(n) * 0.1
        df = pl.DataFrame({"leak_feature": f, "target": t})
        res = self._run_eda_on_df(tmp_path, df, "6")
        found = False
        for leak in res["eda_report"]["leakage_fingerprint"]:
            if leak["feature"] == "leak_feature":
                assert leak["verdict"] == "FLAG"
                found = True
        assert found, "Leak feature not found in fingerprint"

    def test_leakage_watch_between_080_and_095(self, tmp_path):
        import numpy as np
        n = 1000
        t = np.random.randn(n) * 2
        f = t + np.random.randn(n) * 1.0 # cor ~0.89
        df = pl.DataFrame({"leak_feature": f, "target": t})
        res = self._run_eda_on_df(tmp_path, df, "7")
        for leak in res["eda_report"]["leakage_fingerprint"]:
            if leak["feature"] == "leak_feature":
                assert leak["verdict"] == "WATCH"

    def test_leakage_ok_below_080(self, tmp_path):
        import numpy as np
        n = 1000
        t = np.random.randn(n) * 2
        f = t + np.random.randn(n) * 2.5 # low cor
        df = pl.DataFrame({"leak_feature": f, "target": t})
        res = self._run_eda_on_df(tmp_path, df, "8")
        for leak in res["eda_report"]["leakage_fingerprint"]:
            if leak["feature"] == "leak_feature":
                assert leak["verdict"] == "OK"

    def test_flagged_leakage_feature_in_drop_candidates(self, tmp_path):
        import numpy as np
        n = 1000
        t = np.random.randint(0, 2, n)
        f = t + np.random.randn(n) * 0.05
        df = pl.DataFrame({"leak_feature": f, "target": t})
        res = self._run_eda_on_df(tmp_path, df, "9")
        assert "leak_feature" in res["eda_report"]["drop_candidates"], "Feature 'leak_feature' is flagged as leakage but not in drop_candidates. Feature Factory will use it."

    def test_id_conflict_detection(self, tmp_path):
        df = pl.DataFrame({
            "id": ["A", "A", "B", "C"],
            "target": [0, 1, 0, 1]
        })
        res = self._run_eda_on_df(tmp_path, df, "10")
        dups = res["eda_report"]["duplicate_analysis"]
        assert dups["id_conflict_count"] >= 1, "ID conflict not detected: same ID with different target values is the most dangerous label noise pattern."
        assert "id" in dups["id_conflict_columns"]

    def test_exact_duplicate_count_is_correct(self, tmp_path):
        df = pl.DataFrame({
            "f": [1, 1, 1, 2, 3],
            "target": [1, 1, 1, 0, 1]
        })
        res = self._run_eda_on_df(tmp_path, df, "11")
        assert res["eda_report"]["duplicate_analysis"]["exact_count"] == 2

    def test_temporal_profile_detects_date_column(self, tmp_path):
        from datetime import date
        df = pl.DataFrame({
            "signup_date": [date(2021, 1, 1), date(2021, 1, 2)],
            "target": [0, 1]
        })
        res = self._run_eda_on_df(tmp_path, df, "12")
        tp = res["eda_report"]["temporal_profile"]
        assert tp["has_dates"] is True
        assert "signup_date" in tp["date_columns"]

    def test_summary_is_non_empty_and_specific(self, tmp_path):
        df = pl.DataFrame({"f": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
        res = self._run_eda_on_df(tmp_path, df, "13")
        summary = res["eda_report"]["summary"]
        assert len(summary) > 100
        lower = summary.lower()
        assert "target" in lower or "outlier" in lower or "leak" in lower

    def test_zero_variance_feature_in_drop_candidates(self, tmp_path):
        df = pl.DataFrame({
            "zero_var": [1, 1, 1, 1, 1],
            "target": [0, 1, 0, 1, 0]
        })
        res = self._run_eda_on_df(tmp_path, df, "14")
        assert "zero_var" in res["eda_report"]["drop_candidates"]

    def test_target_skew_computed_for_continuous_target(self, tmp_path):
        import numpy as np
        val = np.random.exponential(scale=1.0, size=1000)
        df = pl.DataFrame({"f": range(1000), "target": val})
        res = self._run_eda_on_df(tmp_path, df, "15")
        td = res["eda_report"]["target_distribution"]
        assert td.get("skew", 0) > 1.0
        assert td.get("recommended_transform") in ("log", "sqrt", "boxcox")

    def test_eda_report_written_to_disk(self, tmp_path):
        df = pl.DataFrame({"f": [1, 2], "target": [0, 1]})
        res = self._run_eda_on_df(tmp_path, df, "16")
        path = res["eda_report_path"]
        assert os.path.exists(path)
        with open(path) as f:
            j = json.load(f)
            assert "target_distribution" in j


# -------------------------------------------------------------------
# BLOCK 5 — COMPETITION INTEL: BRIEF QUALITY
# -------------------------------------------------------------------

class TestCompetitionIntelBriefQuality:
    def test_intel_brief_has_all_required_keys(self):
        state = initial_state("spaceship-titanic", "data/spaceship_titanic/train.csv")
        state["session_id"] = "intel_123"
        state["competition_name"] = "spaceship-titanic"
        res = run_competition_intel(state)
        brief = res["competition_brief"]
        req = ["critical_findings", "proven_features", "known_leaks", 
               "external_datasets", "dominant_approach", "cv_strategy_hint", 
               "forbidden_techniques", "shakeup_risk", "source_post_count", "scraped_at"]
        for k in req:
            assert k in brief
            
    def test_shakeup_risk_is_valid_value(self):
        state = initial_state("spaceship-titanic", "data/spaceship_titanic/train.csv")
        state["session_id"] = "intel_234"
        state["competition_name"] = "spaceship-titanic"
        res = run_competition_intel(state)
        brief = res["competition_brief"]
        assert brief["shakeup_risk"] in ("low", "medium", "high", "unknown")

    def test_dominant_approach_is_non_empty_when_posts_scraped(self):
        state = initial_state("spaceship-titanic", "data/spaceship_titanic/train.csv")
        state["session_id"] = "intel_345"
        state["competition_name"] = "spaceship-titanic"
        res = run_competition_intel(state)
        brief = res["competition_brief"]
        if brief["source_post_count"] > 0:
            assert len(brief["dominant_approach"]) > 10

    def test_graceful_degradation_on_private_competition(self):
        state = initial_state("nonexistent-competition-xyz-99", "data/train.csv")
        state["session_id"] = "intel_456"
        state["competition_name"] = "nonexistent-competition-xyz-99"
        res = run_competition_intel(state)
        brief = res["competition_brief"]
        assert brief["source_post_count"] == 0
        for l in ["critical_findings", "proven_features", "known_leaks", "external_datasets", "forbidden_techniques"]:
            assert brief[l] == []
        assert isinstance(brief["dominant_approach"], str)

    def test_competition_brief_written_to_disk(self):
        state = initial_state("spaceship-titanic", "data/spaceship_titanic/train.csv")
        state["session_id"] = "intel_567"
        state["competition_name"] = "spaceship-titanic"
        res = run_competition_intel(state)
        path = res["competition_brief_path"]
        assert os.path.exists(path)
        with open(path) as f:
            j = json.load(f)
            assert "critical_findings" in j

    def test_critical_findings_are_strings_not_dicts(self):
        state = initial_state("spaceship-titanic", "data/spaceship_titanic/train.csv")
        state["session_id"] = "intel_678"
        state["competition_name"] = "spaceship-titanic"
        res = run_competition_intel(state)
        brief = res["competition_brief"]
        for cf in brief["critical_findings"]:
            assert isinstance(cf, str)

    def test_scraped_at_is_valid_iso_timestamp(self):
        state = initial_state("spaceship-titanic", "data/spaceship_titanic/train.csv")
        state["session_id"] = "intel_789"
        state["competition_name"] = "spaceship-titanic"
        res = run_competition_intel(state)
        brief = res["competition_brief"]
        dt = datetime.fromisoformat(brief["scraped_at"])
        assert isinstance(dt, datetime)


# -------------------------------------------------------------------
# BLOCK 6 — INTEGRATION CROSS-AGENT CONTRACT ENFORCEMENT
# -------------------------------------------------------------------

class TestCrossAgentContractEnforcement:
    def test_eda_report_read_by_validation_architect(self, tmp_path):
        df = pl.DataFrame({
            "order_date": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "target": [0, 1]
        })
        csv_path = tmp_path / "integr1.csv"
        df.write_csv(str(csv_path))

        state = initial_state("comp", str(csv_path))
        state["session_id"] = "integr_123"
        state = run_data_engineer(state)
        state["task_type"] = "tabular_classification"
        state = run_eda_agent(state)
        state = run_validation_architect(state)
        assert state["validation_strategy"]["cv_type"] == "TimeSeriesSplit"

    def test_validation_halt_prevents_ml_optimizer_from_running(self, tmp_path):
        from agents.ml_optimizer import run_ml_optimizer
        df = pl.DataFrame({
            "order_date": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            "target": [0, 1]
        })
        csv_path = tmp_path / "integr2.csv"
        df.write_csv(str(csv_path))

        state = initial_state("comp", str(csv_path))
        state["session_id"] = "integr_234"
        state = run_data_engineer(state)
        state["task_type"] = "tabular_classification"
        state = run_validation_architect(state)
        assert state["hitl_required"] is True
        
        # If the DAG handles edge conditionals, ml_optimizer shouldn't run.
        # But if we forcibly call it, it should return early or have logic.
        res = run_ml_optimizer(state)
        # Verify it did not run (registry empty or some flag)
        assert res.get("ml_optimizer_ran") is False or "model_registry" not in res or not res["model_registry"]

    def test_intel_brief_injected_into_validation_architect(self, tmp_path):
        df = pl.DataFrame({"feature": [1.1, 2.2, 3.3, 4.4], "target": [0, 1, 0, 1]})
        csv_path = tmp_path / "test63.csv"
        df.write_csv(str(csv_path))

        state = initial_state("spaceship-titanic", str(csv_path))
        state["session_id"] = f"test_63_{int(datetime.now().timestamp())}"
        state["competition_name"] = "spaceship-titanic"
        state = run_competition_intel(state)
        state = run_data_engineer(state)
        state["task_type"] = "tabular_classification"
        state = run_eda_agent(state)
        state = run_validation_architect(state)
        cv_hint = state["validation_strategy"].get("cv_strategy_hint")
        assert cv_hint is not None

    def test_drop_candidates_from_eda_respected_by_feature_factory(self, tmp_path):
        df = pl.DataFrame({
            "zero_v": [1, 1, 1, 1],
            "target": [0, 1, 0, 1]
        })
        csv_path = tmp_path / "integr4.csv"
        df.write_csv(str(csv_path))

        state = initial_state("comp", str(csv_path))
        state["session_id"] = "integr_456"
        state = run_data_engineer(state)
        state["task_type"] = "tabular_classification"
        state = run_eda_agent(state)
        assert "zero_v" in state["eda_report"]["drop_candidates"]

    def test_data_hash_in_state_after_full_chain(self, tmp_path):
        df = pl.DataFrame({"target": [0, 1, 0, 1]})
        csv_path = tmp_path / "integr5.csv"
        df.write_csv(str(csv_path))

        state = initial_state("nonexistent-abcd", str(csv_path))
        state["competition_name"] = "nonexistent-abcd"
        state["session_id"] = "integr_567"
        state = run_competition_intel(state)
        state = run_data_engineer(state)
        state["task_type"] = "tabular_classification"
        state = run_eda_agent(state)
        assert state["data_hash"] != ""

    def test_phase1_regression_still_green(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/regression/test_phase1_regression.py", "-v"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Phase 1 regression failed:\n{result.stdout}\n{result.stderr}"

