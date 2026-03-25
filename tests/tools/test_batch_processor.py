"""
Tests for batch processing.

FLAW-6.4: Batch Processing for Large Datasets
"""
import pytest
import polars as pl
import numpy as np
from pathlib import Path
from tools.batch_processor import (
    BatchProcessor,
    batch_transform,
    batch_aggregate,
    get_batch_stats,
)


@pytest.fixture
def large_parquet_file(tmp_path):
    """Create large parquet file for testing."""
    n_rows = 50000  # 50K rows for batch testing
    df = pl.DataFrame({
        "id": range(n_rows),
        "value": np.random.randn(n_rows),
        "category": np.random.choice(["A", "B", "C"], n_rows),
        "target": np.random.randint(0, 2, n_rows),
    })
    
    path = tmp_path / "large.parquet"
    df.write_parquet(str(path))
    
    return path


@pytest.fixture
def small_parquet_file(tmp_path):
    """Create small parquet file for testing."""
    n_rows = 100
    df = pl.DataFrame({
        "id": range(n_rows),
        "value": np.random.randn(n_rows),
        "category": np.random.choice(["A", "B", "C"], n_rows),
    })
    
    path = tmp_path / "small.parquet"
    df.write_parquet(str(path))
    
    return path


class TestBatchProcessor:
    """Test BatchProcessor class."""

    def test_batch_processor_initialization(self):
        """Test batch processor initialization."""
        processor = BatchProcessor(batch_size=1000)
        
        assert processor.batch_size == 1000
        assert processor.max_memory_gb == 4.0
        assert processor.progress_interval == 10

    def test_process_in_batches(self, large_parquet_file, tmp_path):
        """Test processing file in batches."""
        processor = BatchProcessor(batch_size=10000)
        output_path = tmp_path / "output.parquet"
        
        def transform(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns((pl.col("value") * 2).alias("value_doubled"))
        
        stats = processor.process_in_batches(
            input_path=str(large_parquet_file),
            output_path=str(output_path),
            process_fn=transform,
        )
        
        assert stats["total_rows"] == 50000
        assert stats["batches_processed"] == 5  # 50K / 10K = 5 batches
        assert stats["rows_processed"] == 50000
        assert len(stats["errors"]) == 0
        
        # Verify output
        df_output = pl.read_parquet(str(output_path))
        assert len(df_output) == 50000
        assert "value_doubled" in df_output.columns

    def test_process_small_file(self, small_parquet_file, tmp_path):
        """Test processing small file (single batch)."""
        processor = BatchProcessor(batch_size=10000)
        output_path = tmp_path / "output.parquet"
        
        def transform(df: pl.DataFrame) -> pl.DataFrame:
            return df
        
        stats = processor.process_in_batches(
            input_path=str(small_parquet_file),
            output_path=str(output_path),
            process_fn=transform,
        )
        
        assert stats["batches_processed"] == 1
        assert stats["rows_processed"] == 100

    def test_transform_column(self, large_parquet_file, tmp_path):
        """Test column transformation in batches."""
        processor = BatchProcessor(batch_size=10000)
        output_path = tmp_path / "output.parquet"
        
        def double_series(s: pl.Series) -> pl.Series:
            return s * 2
        
        stats = processor.transform_column(
            input_path=str(large_parquet_file),
            output_path=str(output_path),
            column="value",
            transform_fn=double_series,
        )
        
        assert stats["rows_processed"] == 50000
        
        # Verify transformation
        df_output = pl.read_parquet(str(output_path))
        df_input = pl.read_parquet(str(large_parquet_file))
        
        assert (df_output["value"] == df_input["value"] * 2).all()

    def test_filter_rows(self, large_parquet_file, tmp_path):
        """Test row filtering in batches."""
        processor = BatchProcessor(batch_size=10000)
        output_path = tmp_path / "filtered.parquet"
        
        def filter_fn(df: pl.DataFrame) -> pl.Series:
            return df["category"] == "A"
        
        stats = processor.filter_rows(
            input_path=str(large_parquet_file),
            output_path=str(output_path),
            filter_fn=filter_fn,
        )
        
        # Should have filtered to ~1/3 of rows
        assert stats["rows_processed"] < 50000
        assert stats["rows_processed"] > 10000  # Should be around 16-17K
        
        # Verify all rows have category A
        df_output = pl.read_parquet(str(output_path))
        assert (df_output["category"] == "A").all()

    def test_process_with_errors(self, tmp_path):
        """Test error handling during batch processing."""
        processor = BatchProcessor(batch_size=1000)
        output_path = tmp_path / "output.parquet"
        
        def failing_transform(df: pl.DataFrame) -> pl.DataFrame:
            raise ValueError("Test error")
        
        # Create small test file
        df = pl.DataFrame({"id": range(100)})
        input_path = tmp_path / "input.parquet"
        df.write_parquet(str(input_path))
        
        stats = processor.process_in_batches(
            input_path=str(input_path),
            output_path=str(output_path),
            process_fn=failing_transform,
        )
        
        assert len(stats["errors"]) > 0
        assert "Test error" in stats["errors"][0]["error"]


class TestBatchTransform:
    """Test batch_transform convenience function."""

    def test_batch_transform_function(self, large_parquet_file, tmp_path):
        """Test batch_transform function."""
        output_path = tmp_path / "output.parquet"
        
        def transform(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns((pl.col("value") + 1).alias("value_plus_one"))
        
        stats = batch_transform(
            input_path=str(large_parquet_file),
            output_path=str(output_path),
            transform_fn=transform,
            batch_size=10000,
        )
        
        assert stats["rows_processed"] == 50000
        
        # Verify transformation
        df_output = pl.read_parquet(str(output_path))
        assert "value_plus_one" in df_output.columns


class TestBatchAggregate:
    """Test batch_aggregate function."""

    def test_batch_aggregate_function(self, large_parquet_file):
        """Test batch_aggregate function."""
        result = batch_aggregate(
            input_path=str(large_parquet_file),
            group_by=["category"],
            aggregations={"value": "mean", "target": "sum"},
            batch_size=10000,
        )
        
        assert len(result) == 3  # Categories A, B, C
        assert "value" in result.columns
        assert "target" in result.columns
        
        # Check aggregation results are reasonable
        assert 20000 <= result["target"].sum() <= 30000  # Sum of ~50% 0/1 values


class TestGetBatchStats:
    """Test get_batch_stats function."""

    def test_get_batch_stats(self, large_parquet_file):
        """Test getting batch statistics."""
        stats = get_batch_stats(str(large_parquet_file), batch_size=10000)
        
        assert stats["total_rows"] == 50000
        assert stats["batch_size"] == 10000
        assert stats["n_batches"] == 5
        assert stats["last_batch_size"] == 10000

    def test_get_batch_stats_uneven(self, small_parquet_file):
        """Test batch stats with uneven division."""
        stats = get_batch_stats(str(small_parquet_file), batch_size=30)
        
        assert stats["total_rows"] == 100
        assert stats["n_batches"] == 4  # ceil(100/30) = 4
        assert stats["last_batch_size"] == 10  # 100 % 30 = 10


class TestBatchProcessorMemory:
    """Test batch processor memory management."""

    def test_batch_processor_gc_after_batch(self, large_parquet_file, tmp_path):
        """Test garbage collection after each batch."""
        import gc
        
        processor = BatchProcessor(batch_size=10000)
        output_path = tmp_path / "output.parquet"
        
        def transform(df: pl.DataFrame) -> pl.DataFrame:
            # Create some temporary data
            temp = df.clone()
            temp = temp.with_columns(pl.col("value") * 2)
            return temp
        
        # Process
        stats = processor.process_in_batches(
            input_path=str(large_parquet_file),
            output_path=str(output_path),
            process_fn=transform,
        )
        
        # Force GC and check no errors
        gc.collect()
        
        assert stats["rows_processed"] == 50000
        assert len(stats["errors"]) == 0
