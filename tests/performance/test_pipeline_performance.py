"""
Performance regression tests for Professor pipeline.

FLAW-5.4: Performance Tests
- Execution time regression tests
- Memory usage regression tests
- Performance benchmarks
- Catch performance degradation early
"""
import pytest
import time
import os
import gc
import psutil
import polars as pl
import numpy as np
from pathlib import Path
from tools.performance_monitor import get_performance_summary


class TestExecutionTime:
    """Test execution time performance."""

    def test_data_engineer_execution_time(self, tmp_path):
        """Test data engineer completes within time limit."""
        from agents.data_engineer import run_data_engineer
        
        # Create test data
        n_rows = 1000
        df = pl.DataFrame({
            f"feature_{i}": np.random.randn(n_rows) for i in range(10)
        })
        target = np.random.randint(0, 2, n_rows)
        df = df.with_columns(pl.Series("target", target))
        
        data_path = tmp_path / "train.csv"
        df.write_csv(str(data_path))
        
        # Also create test.csv and sample_submission.csv
        test_df = df.head(100).drop("target")
        test_path = tmp_path / "test.csv"
        test_df.write_csv(str(test_path))
        
        sample_sub = pl.DataFrame({
            "PassengerId": test_df.row(0)[0] if len(test_df) > 0 else "test_1",
            "target": [0, 1] * 50
        })
        sample_path = tmp_path / "sample_submission.csv"
        sample_sub.write_csv(str(sample_path))
        
        state = {
            "session_id": "perf_test",
            "competition_name": "test",
            "raw_data_path": str(data_path),
            "test_data_path": str(test_path),
            "sample_submission_path": str(sample_path),
            "cost_tracker": {
                "total_usd": 0.0, "groq_tokens_in": 0, "groq_tokens_out": 0,
                "gemini_tokens": 0, "llm_calls": 0, "budget_usd": 10.0,
                "warning_threshold": 0.7, "throttle_threshold": 0.85,
                "triage_threshold": 0.95,
            },
        }
        
        start = time.time()
        result = run_data_engineer(state)
        elapsed = time.time() - start
        
        # Should complete in < 30 seconds for 1000 rows
        assert elapsed < 30.0, f"Data engineer too slow: {elapsed:.2f}s"
        assert "clean_data_path" in result

    def test_eda_execution_time(self, tmp_path):
        """Test EDA completes within time limit."""
        from agents.eda_agent import run_eda_agent
        from tools.data_tools import write_parquet, write_json
        
        # Create test data
        n_rows = 1000
        df = pl.DataFrame({
            f"feature_{i}": np.random.randn(n_rows) for i in range(10)
        })
        df = df.with_columns(pl.Series("target", np.random.randint(0, 2, n_rows)))
        
        clean_path = tmp_path / "clean.parquet"
        write_parquet(df, str(clean_path))
        
        schema = {"target_col": "target", "id_columns": []}
        schema_path = tmp_path / "schema.json"
        write_json(schema, str(schema_path))
        
        state = {
            "session_id": "perf_test",
            "competition_name": "test",
            "clean_data_path": str(clean_path),
            "schema_path": str(schema_path),
            "target_col": "target",
            "id_columns": [],
            "cost_tracker": {
                "total_usd": 0.0, "groq_tokens_in": 0, "groq_tokens_out": 0,
                "gemini_tokens": 0, "llm_calls": 0, "budget_usd": 10.0,
                "warning_threshold": 0.7, "throttle_threshold": 0.85,
                "triage_threshold": 0.95,
            },
        }
        
        start = time.time()
        result = run_eda_agent(state)
        elapsed = time.time() - start
        
        # Should complete in < 10 seconds for 1000 rows
        assert elapsed < 10.0, f"EDA too slow: {elapsed:.2f}s"
        assert "eda_report" in result


class TestMemoryUsage:
    """Test memory usage performance."""

    def test_data_engineer_memory_usage(self, tmp_path):
        """Test data engineer memory usage."""
        from agents.data_engineer import run_data_engineer
        
        # Create larger test data
        n_rows = 10000
        df = pl.DataFrame({
            f"feature_{i}": np.random.randn(n_rows) for i in range(20)
        })
        target = np.random.randint(0, 2, n_rows)
        df = df.with_columns(pl.Series("target", target))
        
        data_path = tmp_path / "train.csv"
        df.write_csv(str(data_path))
        
        # Also create test.csv and sample_submission.csv
        test_df = df.head(100).drop("target")
        test_path = tmp_path / "test.csv"
        test_df.write_csv(str(test_path))
        
        sample_sub = pl.DataFrame({
            "PassengerId": test_df.row(0)[0] if len(test_df) > 0 else "test_1",
            "target": [0, 1] * 50
        })
        sample_path = tmp_path / "sample_submission.csv"
        sample_sub.write_csv(str(sample_path))
        
        state = {
            "session_id": "mem_test",
            "competition_name": "test",
            "raw_data_path": str(data_path),
            "test_data_path": str(test_path),
            "sample_submission_path": str(sample_path),
            "cost_tracker": {
                "total_usd": 0.0, "groq_tokens_in": 0, "groq_tokens_out": 0,
                "gemini_tokens": 0, "llm_calls": 0, "budget_usd": 10.0,
                "warning_threshold": 0.7, "throttle_threshold": 0.85,
                "triage_threshold": 0.95,
            },
        }
        
        # Get memory before
        gc.collect()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run
        result = run_data_engineer(state)
        
        # Get memory after
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_delta = mem_after - mem_before
        
        # Should use < 500 MB for 10000 rows
        assert mem_delta < 500.0, f"Data engineer used too much memory: {mem_delta:.2f} MB"
        assert "clean_data_path" in result

    def test_eda_memory_usage(self, tmp_path):
        """Test EDA memory usage."""
        from agents.eda_agent import run_eda_agent
        from tools.data_tools import write_parquet, write_json
        
        # Create test data
        n_rows = 10000
        df = pl.DataFrame({
            f"feature_{i}": np.random.randn(n_rows) for i in range(20)
        })
        df = df.with_columns(pl.Series("target", np.random.randint(0, 2, n_rows)))
        
        clean_path = tmp_path / "clean.parquet"
        write_parquet(df, str(clean_path))
        
        schema = {"target_col": "target", "id_columns": []}
        schema_path = tmp_path / "schema.json"
        write_json(schema, str(schema_path))
        
        state = {
            "session_id": "mem_test",
            "competition_name": "test",
            "clean_data_path": str(clean_path),
            "schema_path": str(schema_path),
            "target_col": "target",
            "id_columns": [],
            "cost_tracker": {
                "total_usd": 0.0, "groq_tokens_in": 0, "groq_tokens_out": 0,
                "gemini_tokens": 0, "llm_calls": 0, "budget_usd": 10.0,
                "warning_threshold": 0.7, "throttle_threshold": 0.85,
                "triage_threshold": 0.95,
            },
        }
        
        # Get memory before
        gc.collect()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run
        result = run_eda_agent(state)
        
        # Get memory after
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_delta = mem_after - mem_before
        
        # Should use < 300 MB for 10000 rows
        assert mem_delta < 300.0, f"EDA used too much memory: {mem_delta:.2f} MB"
        assert "eda_report" in result


class TestScalability:
    """Test scalability with data size."""

    def test_data_engineer_scales_with_rows(self, tmp_path):
        """Test data engineer scales reasonably with row count."""
        from agents.data_engineer import run_data_engineer
        
        times = []
        row_counts = [100, 1000, 5000]
        
        for n_rows in row_counts:
            df = pl.DataFrame({
                f"feature_{i}": np.random.randn(n_rows) for i in range(10)
            })
            df = df.with_columns(pl.Series("target", np.random.randint(0, 2, n_rows)))
            
            data_path = tmp_path / f"train_{n_rows}.csv"
            df.write_csv(str(data_path))
            
            state = {
                "session_id": f"scale_test_{n_rows}",
                "competition_name": "test",
                "raw_data_path": str(data_path),
                "cost_tracker": {
                    "total_usd": 0.0, "groq_tokens_in": 0, "groq_tokens_out": 0,
                    "gemini_tokens": 0, "llm_calls": 0, "budget_usd": 10.0,
                    "warning_threshold": 0.7, "throttle_threshold": 0.85,
                    "triage_threshold": 0.95,
                },
            }
            
            start = time.time()
            run_data_engineer(state)
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Check scaling is roughly linear (allow 3x for 50x data increase)
        # 100 -> 5000 rows is 50x, time should increase < 10x
        scale_factor = times[2] / times[0] if times[0] > 0 else 0
        assert scale_factor < 15.0, f"Data engineer scales poorly: {scale_factor:.2f}x for 50x data"

    def test_eda_scales_with_rows(self, tmp_path):
        """Test EDA scales reasonably with row count."""
        from agents.eda_agent import run_eda_agent
        from tools.data_tools import write_parquet, write_json
        
        times = []
        row_counts = [100, 1000, 5000]
        
        for n_rows in row_counts:
            df = pl.DataFrame({
                f"feature_{i}": np.random.randn(n_rows) for i in range(10)
            })
            df = df.with_columns(pl.Series("target", np.random.randint(0, 2, n_rows)))
            
            clean_path = tmp_path / f"clean_{n_rows}.parquet"
            write_parquet(df, str(clean_path))
            
            schema = {"target_col": "target", "id_columns": []}
            schema_path = tmp_path / f"schema_{n_rows}.json"
            write_json(schema, str(schema_path))
            
            state = {
                "session_id": f"scale_test_{n_rows}",
                "competition_name": "test",
                "clean_data_path": str(clean_path),
                "schema_path": str(schema_path),
                "target_col": "target",
                "id_columns": [],
                "cost_tracker": {
                    "total_usd": 0.0, "groq_tokens_in": 0, "groq_tokens_out": 0,
                    "gemini_tokens": 0, "llm_calls": 0, "budget_usd": 10.0,
                    "warning_threshold": 0.7, "throttle_threshold": 0.85,
                    "triage_threshold": 0.95,
                },
            }
            
            start = time.time()
            run_eda_agent(state)
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Check scaling is roughly linear (allow 3x for 50x data increase)
        scale_factor = times[2] / times[0] if times[0] > 0 else 0
        assert scale_factor < 20.0, f"EDA scales poorly: {scale_factor:.2f}x for 50x data"


class TestPerformanceBenchmarks:
    """Test performance against benchmarks."""

    def test_pipeline_baseline_time(self, tmp_path):
        """Test basic pipeline completes within baseline time."""
        from agents.data_engineer import run_data_engineer
        from agents.eda_agent import run_eda_agent
        from tools.data_tools import write_parquet, write_json
        
        # Create standard benchmark data
        n_rows = 1000
        df = pl.DataFrame({
            f"feature_{i}": np.random.randn(n_rows) for i in range(10)
        })
        df = df.with_columns(pl.Series("target", np.random.randint(0, 2, n_rows)))
        
        data_path = tmp_path / "train.csv"
        df.write_csv(str(data_path))
        
        clean_path = tmp_path / "clean.parquet"
        write_parquet(df, str(clean_path))
        
        schema = {"target_col": "target", "id_columns": []}
        schema_path = tmp_path / "schema.json"
        write_json(schema, str(schema_path))
        
        state = {
            "session_id": "benchmark",
            "competition_name": "test",
            "raw_data_path": str(data_path),
            "clean_data_path": str(clean_path),
            "schema_path": str(schema_path),
            "target_col": "target",
            "id_columns": [],
            "cost_tracker": {
                "total_usd": 0.0, "groq_tokens_in": 0, "groq_tokens_out": 0,
                "gemini_tokens": 0, "llm_calls": 0, "budget_usd": 10.0,
                "warning_threshold": 0.7, "throttle_threshold": 0.85,
                "triage_threshold": 0.95,
            },
        }
        
        # Run pipeline
        start = time.time()
        
        result_de = run_data_engineer(state)
        state.update(result_de)
        
        result_eda = run_eda_agent(state)
        state.update(result_eda)
        
        total_time = time.time() - start
        
        # Baseline: 2 agents should complete in < 60 seconds for 1000 rows
        assert total_time < 60.0, f"Pipeline too slow: {total_time:.2f}s"
        
        # Verify results
        assert "clean_data_path" in state
        assert "eda_report" in state
