"""
Tests for lazy loading.

FLAW-6.3: Lazy Loading for Large Data
"""
import pytest
import polars as pl
import numpy as np
from pathlib import Path
from tools.lazy_loader import (
    LazyDataFrame,
    LazyDataset,
    lazy_load_parquet,
    get_file_info,
)


@pytest.fixture
def test_parquet_file(tmp_path):
    """Create test parquet file."""
    n_rows = 1000
    df = pl.DataFrame({
        f"feature_{i}": np.random.randn(n_rows) for i in range(10)
    })
    df = df.with_columns(pl.Series("target", np.random.randint(0, 2, n_rows)))
    
    path = tmp_path / "test.parquet"
    df.write_parquet(str(path))
    
    return path


@pytest.fixture
def test_csv_file(tmp_path):
    """Create test CSV file."""
    n_rows = 1000
    df = pl.DataFrame({
        f"feature_{i}": np.random.randn(n_rows) for i in range(10)
    })
    df = df.with_columns(pl.Series("target", np.random.randint(0, 2, n_rows)))
    
    path = tmp_path / "test.csv"
    df.write_csv(str(path))
    
    return path


class TestLazyDataFrame:
    """Test LazyDataFrame class."""

    def test_lazy_initialization(self, test_parquet_file):
        """Test lazy DataFrame initializes without loading."""
        lazy_df = LazyDataFrame(str(test_parquet_file))
        
        assert lazy_df._df is None
        assert not lazy_df.is_loaded()
        assert lazy_df.path == test_parquet_file

    def test_lazy_loads_on_access(self, test_parquet_file):
        """Test lazy DataFrame loads on first access."""
        lazy_df = LazyDataFrame(str(test_parquet_file))
        
        # Should not be loaded initially
        assert not lazy_df.is_loaded()
        
        # Access DataFrame
        df = lazy_df.df
        
        # Should be loaded now
        assert lazy_df.is_loaded()
        assert df is not None
        assert len(df) == 1000

    def test_lazy_caches_after_load(self, test_parquet_file):
        """Test lazy DataFrame caches after first load."""
        lazy_df = LazyDataFrame(str(test_parquet_file))
        
        # First access
        df1 = lazy_df.df
        assert lazy_df.is_loaded()
        
        # Second access should return same object
        df2 = lazy_df.df
        assert df1 is df2

    def test_lazy_unload(self, test_parquet_file):
        """Test lazy DataFrame unloading."""
        lazy_df = LazyDataFrame(str(test_parquet_file))
        
        # Load
        _ = lazy_df.df
        assert lazy_df.is_loaded()
        
        # Unload
        lazy_df.unload()
        assert not lazy_df.is_loaded()
        assert lazy_df._df is None

    def test_lazy_auto_unload(self, test_parquet_file):
        """Test auto-unload feature."""
        lazy_df = LazyDataFrame(str(test_parquet_file), auto_unload=True)
        
        # Access should trigger auto-unload
        _ = lazy_df.df
        
        # Should be unloaded automatically
        assert not lazy_df.is_loaded()

    def test_lazy_shape_without_load(self, test_parquet_file):
        """Test getting shape without loading full DataFrame."""
        lazy_df = LazyDataFrame(str(test_parquet_file))
        
        shape = lazy_df.shape
        
        assert shape[0] == 1000
        assert shape[1] == 11  # 10 features + target
        assert not lazy_df.is_loaded()

    def test_lazy_columns_without_load(self, test_parquet_file):
        """Test getting columns without loading full DataFrame."""
        lazy_df = LazyDataFrame(str(test_parquet_file))
        
        columns = lazy_df.columns
        
        assert len(columns) == 11
        assert "target" in columns
        assert not lazy_df.is_loaded()

    def test_lazy_head(self, test_parquet_file):
        """Test getting first n rows."""
        lazy_df = LazyDataFrame(str(test_parquet_file))
        
        head_df = lazy_df.head(5)
        
        assert len(head_df) == 5
        assert not lazy_df.is_loaded()  # Should not load full file

    def test_lazy_select_columns(self, test_parquet_file):
        """Test selecting specific columns."""
        lazy_df = LazyDataFrame(str(test_parquet_file))
        
        selected = lazy_df.select_columns(["feature_0", "feature_1"])
        
        assert selected.shape == (1000, 2)
        assert list(selected.columns) == ["feature_0", "feature_1"]

    def test_lazy_repr(self, test_parquet_file):
        """Test string representation."""
        lazy_df = LazyDataFrame(str(test_parquet_file))
        
        repr_str = repr(lazy_df)
        
        assert "test.parquet" in repr_str
        assert "not loaded" in repr_str
        
        # Load and check again
        _ = lazy_df.df
        repr_str = repr(lazy_df)
        assert "loaded" in repr_str

    def test_lazy_csv_support(self, test_csv_file):
        """Test CSV file support."""
        lazy_df = LazyDataFrame(str(test_csv_file))
        
        df = lazy_df.df
        
        assert len(df) == 1000
        assert lazy_df.is_loaded()

    def test_lazy_file_not_found(self, tmp_path):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            LazyDataFrame(str(tmp_path / "nonexistent.parquet"))


class TestLazyDataset:
    """Test LazyDataset class."""

    def test_dataset_initialization(self, test_parquet_file):
        """Test lazy dataset initialization."""
        dataset = LazyDataset(str(test_parquet_file), target_column="target")
        
        assert dataset.n_rows == 1000
        assert len(dataset.columns) == 11
        assert dataset.target_column == "target"

    def test_dataset_get_chunk(self, test_parquet_file):
        """Test getting data chunks."""
        dataset = LazyDataset(
            str(test_parquet_file),
            target_column="target",
            chunk_size=100,
        )
        
        X_chunk, y_chunk = dataset.get_chunk(0, 100)
        
        assert len(X_chunk) == 100
        assert len(y_chunk) == 100
        assert "target" not in X_chunk.columns
        assert len(y_chunk) == len(X_chunk)

    def test_dataset_iterate_chunks(self, test_parquet_file):
        """Test iterating over chunks."""
        dataset = LazyDataset(
            str(test_parquet_file),
            target_column="target",
            chunk_size=200,
        )
        
        chunks = list(dataset.iterate_chunks())
        
        # Should have 5 chunks (1000 / 200)
        assert len(chunks) == 5
        
        # Check first chunk
        X_chunk, y_chunk = chunks[0]
        assert len(X_chunk) == 200
        assert len(y_chunk) == 200

    def test_dataset_feature_stats(self, test_parquet_file):
        """Test getting feature statistics."""
        dataset = LazyDataset(str(test_parquet_file), target_column="target")
        
        stats = dataset.get_feature_stats()
        
        assert len(stats) == 10  # 10 features (excluding target)
        assert "feature_0" in stats
        assert "dtype" in stats["feature_0"]


class TestLazyLoadParquet:
    """Test lazy_load_parquet function."""

    def test_load_full_file(self, test_parquet_file):
        """Test loading full parquet file."""
        df = lazy_load_parquet(str(test_parquet_file))
        
        assert len(df) == 1000
        assert df.shape[1] == 11

    def test_load_columns_only(self, test_parquet_file):
        """Test loading specific columns only."""
        df = lazy_load_parquet(
            str(test_parquet_file),
            columns=["feature_0", "feature_1"],
        )
        
        assert df.shape == (1000, 2)
        assert list(df.columns) == ["feature_0", "feature_1"]

    def test_load_rows_only(self, test_parquet_file):
        """Test loading specific number of rows only."""
        df = lazy_load_parquet(str(test_parquet_file), n_rows=50)
        
        assert len(df) == 50

    def test_load_columns_and_rows(self, test_parquet_file):
        """Test loading specific columns and rows."""
        df = lazy_load_parquet(
            str(test_parquet_file),
            columns=["feature_0"],
            n_rows=10,
        )
        
        assert df.shape == (10, 1)


class TestGetFileInfo:
    """Test get_file_info function."""

    def test_file_info_parquet(self, test_parquet_file):
        """Test getting parquet file info."""
        info = get_file_info(str(test_parquet_file))
        
        assert info["exists"] is True
        assert info["format"] == ".parquet"
        assert info["n_rows"] == 1000
        assert info["n_columns"] == 11
        assert "size_mb" in info

    def test_file_info_not_found(self, tmp_path):
        """Test getting info for non-existent file."""
        info = get_file_info(str(tmp_path / "nonexistent.parquet"))
        
        assert info["exists"] is False

    def test_file_info_size(self, test_parquet_file):
        """Test file size reporting."""
        info = get_file_info(str(test_parquet_file))
        
        assert info["size_mb"] > 0
        assert info["size_mb"] < 100  # Should be small test file
