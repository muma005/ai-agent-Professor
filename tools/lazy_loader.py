# tools/lazy_loader.py

"""
Lazy loading for large datasets.

FLAW-6.3 FIX: Lazy Loading of Large Data
- Load data on-demand, not all at once
- Memory-efficient data access
- Automatic unloading of unused data
- Streaming support for large files
"""

import os
import logging
import gc
from pathlib import Path
from typing import Optional, List, Union, Iterator
import polars as pl

logger = logging.getLogger(__name__)


class LazyDataFrame:
    """
    Lazy-loading Polars DataFrame wrapper.
    
    Loads data only when accessed, not at initialization.
    Automatically unloads after use to free memory.
    
    Usage:
        lazy_df = LazyDataFrame("large_file.parquet")
        # File not loaded yet
        
        # Access data - loads on first access
        print(lazy_df.df.shape)
        
        # Use Polars methods
        result = lazy_df.df.select(["col1", "col2"])
        
        # Unload to free memory
        lazy_df.unload()
    """
    
    def __init__(
        self,
        path: str,
        auto_unload: bool = False,
        unload_after_access: int = 0,  # 0 = never auto-unload
    ):
        """
        Initialize lazy DataFrame.
        
        Args:
            path: Path to data file (parquet, csv)
            auto_unload: Automatically unload after access
            unload_after_access: Unload after N accesses (0 = never)
        """
        self.path = Path(path)
        self.auto_unload = auto_unload
        self.unload_after_access = unload_after_access
        self._df: Optional[pl.DataFrame] = None
        self._access_count = 0
        self._file_size_mb = self._get_file_size_mb()
        
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")
        
        logger.info(
            f"[LazyLoader] Initialized for {self.path.name} "
            f"({self._file_size_mb:.2f} MB)"
        )
    
    def _get_file_size_mb(self) -> float:
        """Get file size in MB."""
        if not self.path.exists():
            return 0.0
        return self.path.stat().st_size / (1024 * 1024)
    
    @property
    def df(self) -> pl.DataFrame:
        """
        Load and return DataFrame.
        
        Loads on first access, returns cached on subsequent accesses.
        """
        if self._df is None:
            logger.info(
                f"[LazyLoader] Loading {self.path.name} "
                f"({self._file_size_mb:.2f} MB)"
            )
            self._df = self._load_dataframe()
            self._access_count += 1
            
            logger.debug(
                f"[LazyLoader] Loaded {self.path.name} "
                f"(access #{self._access_count})"
            )
        else:
            # Increment access count on cached access too
            self._access_count += 1
        
        # Check if should auto-unload (after returning cached data)
        if self.auto_unload and self._df is not None:
            # Auto-unload immediately after first access
            self.unload()
        elif self.unload_after_access > 0 and \
             self._access_count >= self.unload_after_access:
            logger.debug(
                f"[LazyLoader] Auto-unloading after "
                f"{self._access_count} accesses"
            )
            self.unload()
        
        return self._df
    
    def _load_dataframe(self) -> pl.DataFrame:
        """Load DataFrame from file."""
        suffix = self.path.suffix.lower()
        
        if suffix == ".parquet":
            return pl.read_parquet(str(self.path))
        elif suffix == ".csv":
            return pl.read_csv(str(self.path))
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def unload(self) -> None:
        """Unload DataFrame from memory."""
        if self._df is not None:
            logger.debug(
                f"[LazyLoader] Unloading {self.path.name} "
                f"(was {self._file_size_mb:.2f} MB)"
            )
            del self._df
            self._df = None
            gc.collect()
    
    def is_loaded(self) -> bool:
        """Check if DataFrame is currently loaded."""
        return self._df is not None
    
    @property
    def shape(self) -> tuple[int, int]:
        """Get shape without loading full DataFrame."""
        # For parquet, we can read metadata only
        if self.path.suffix.lower() == ".parquet":
            pf = pl.scan_parquet(str(self.path))
            return (pf.collect().height, len(pf.columns))
        else:
            # Must load for CSV
            return self.df.shape
    
    @property
    def columns(self) -> List[str]:
        """Get column names without loading full DataFrame."""
        if self.path.suffix.lower() == ".parquet":
            return pl.scan_parquet(str(self.path)).columns
        else:
            return self.df.columns
    
    def get_schema(self) -> pl.Schema:
        """Get schema without loading full DataFrame."""
        if self.path.suffix.lower() == ".parquet":
            return pl.scan_parquet(str(self.path)).collect_schema()
        else:
            return self.df.schema
    
    def head(self, n: int = 5) -> pl.DataFrame:
        """Get first n rows (loads only those rows)."""
        if self.path.suffix.lower() == ".parquet":
            return pl.scan_parquet(str(self.path)).head(n).collect()
        else:
            return self.df.head(n)
    
    def select_columns(self, columns: List[str]) -> pl.DataFrame:
        """
        Select specific columns (loads only those columns).
        
        More memory-efficient than loading full DataFrame.
        """
        if self.path.suffix.lower() == ".parquet":
            return pl.scan_parquet(str(self.path)).select(columns).collect()
        else:
            return self.df.select(columns)
    
    def __len__(self) -> int:
        """Get length without loading full DataFrame."""
        return self.shape[0]
    
    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self.is_loaded() else "not loaded"
        return (
            f"LazyDataFrame(path={self.path.name}, "
            f"size={self._file_size_mb:.2f} MB, {status})"
        )


class LazyDataset:
    """
    Lazy-loading dataset for machine learning.
    
    Loads data in chunks for training large models.
    """
    
    def __init__(
        self,
        path: str,
        target_column: str,
        chunk_size: int = 10000,
    ):
        """
        Initialize lazy dataset.
        
        Args:
            path: Path to data file
            target_column: Name of target column
            chunk_size: Number of rows per chunk
        """
        self.path = Path(path)
        self.target_column = target_column
        self.chunk_size = chunk_size
        self.lazy_df = LazyDataFrame(str(path))
        
        # Get metadata
        self.n_rows = len(self.lazy_df)
        self.columns = self.lazy_df.columns
        
        logger.info(
            f"[LazyDataset] Initialized: {self.n_rows} rows, "
            f"{len(self.columns)} columns, chunk_size={chunk_size}"
        )
    
    def get_chunk(self, start_idx: int, end_idx: int) -> tuple:
        """
        Get data chunk for training.
        
        Args:
            start_idx: Start row index
            end_idx: End row index
        
        Returns:
            (X_chunk, y_chunk) tuple
        """
        if self.path.suffix.lower() == ".parquet":
            # Use scan for efficient chunk reading
            df_chunk = (
                pl.scan_parquet(str(self.path))
                .slice(start_idx, end_idx - start_idx)
                .collect()
            )
        else:
            # Load full and slice (less efficient)
            df_full = self.lazy_df.df
            df_chunk = df_full.slice(start_idx, end_idx - start_idx)
        
        # Separate features and target
        y_chunk = df_chunk[self.target_column]
        X_chunk = df_chunk.drop(self.target_column)
        
        return X_chunk, y_chunk
    
    def iterate_chunks(self) -> Iterator[tuple]:
        """
        Iterate over all chunks in dataset.
        
        Yields:
            (X_chunk, y_chunk) tuples
        """
        for start_idx in range(0, self.n_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, self.n_rows)
            X_chunk, y_chunk = self.get_chunk(start_idx, end_idx)
            yield X_chunk, y_chunk
    
    def get_feature_stats(self) -> dict:
        """Get feature statistics without loading full data."""
        stats = {}
        
        for col in self.columns:
            if col == self.target_column:
                continue
            
            if self.path.suffix.lower() == ".parquet":
                # Get basic stats from scan
                lf = pl.scan_parquet(str(self.path))
                stats[col] = {
                    "dtype": str(lf.collect_schema()[col]),
                }
        
        return stats


def lazy_load_parquet(
    path: str,
    columns: Optional[List[str]] = None,
    n_rows: Optional[int] = None,
) -> pl.DataFrame:
    """
    Lazily load parquet file with optional column/row selection.
    
    Args:
        path: Path to parquet file
        columns: Optional list of columns to load
        n_rows: Optional number of rows to load
    
    Returns:
        Polars DataFrame
    """
    lazy_df = LazyDataFrame(path)
    
    if columns and n_rows:
        return lazy_df.df.select(columns).head(n_rows)
    elif columns:
        return lazy_df.df.select(columns)
    elif n_rows:
        return lazy_df.df.head(n_rows)
    else:
        return lazy_df.df


def get_file_info(path: str) -> dict:
    """
    Get file information without loading.
    
    Args:
        path: Path to data file
    
    Returns:
        File info dict
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        return {"exists": False}
    
    info = {
        "exists": True,
        "path": str(path_obj),
        "size_mb": path_obj.stat().st_size / (1024 * 1024),
        "format": path_obj.suffix.lower(),
    }
    
    # Get metadata for parquet
    if path_obj.suffix.lower() == ".parquet":
        lf = pl.scan_parquet(str(path_obj))
        info["n_rows"] = lf.collect().height
        info["n_columns"] = len(lf.columns)
        info["columns"] = lf.columns
    
    return info
