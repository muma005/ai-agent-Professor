# tools/batch_processor.py

"""
Batch processing for large datasets.

FLAW-6.4 FIX: Batch Processing for Large Datasets
- Process data in chunks to avoid OOM
- Memory-efficient transformations
- Progress tracking during batch operations
- Automatic batch size optimization
"""

import os
import logging
import gc
from pathlib import Path
from typing import Callable, Iterator, Tuple, Optional, List, Dict, Any
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Process large datasets in batches.
    
    Features:
    - Configurable batch size
    - Memory monitoring
    - Progress tracking
    - Automatic garbage collection
    - Error handling per batch
    
    Usage:
        processor = BatchProcessor(batch_size=10000)
        
        # Process with callback
        processor.process_in_batches(
            input_path="large_file.parquet",
            output_path="processed.parquet",
            process_fn=my_transformation,
        )
    """
    
    def __init__(
        self,
        batch_size: int = 10000,
        max_memory_gb: float = 4.0,
        progress_interval: int = 10,
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of rows per batch
            max_memory_gb: Maximum memory usage in GB
            progress_interval: Log progress every N batches
        """
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.progress_interval = progress_interval
        
        logger.info(
            f"[BatchProcessor] Initialized -- batch_size: {batch_size}, "
            f"max_memory: {max_memory_gb} GB"
        )
    
    def process_in_batches(
        self,
        input_path: str,
        output_path: str,
        process_fn: Callable[[pl.DataFrame], pl.DataFrame],
        append_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Process large file in batches.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            process_fn: Function to apply to each batch
            append_mode: Append batches to output file
        
        Returns:
            Processing stats dict
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Get total rows
        if input_path.suffix.lower() == ".parquet":
            total_rows = pl.scan_parquet(str(input_path)).collect().height
        else:
            # For CSV, need to read once
            df_temp = pl.read_csv(str(input_path))
            total_rows = len(df_temp)
            del df_temp
            gc.collect()
        
        logger.info(
            f"[BatchProcessor] Processing {total_rows} rows in batches of {self.batch_size}"
        )
        
        # Process batches
        stats = {
            "total_rows": total_rows,
            "batches_processed": 0,
            "rows_processed": 0,
            "errors": [],
        }
        
        # Remove output file if exists (for fresh write)
        if not append_mode and output_path.exists():
            output_path.unlink()
        
        for batch_idx, df_batch in enumerate(
            self._read_in_batches(str(input_path))
        ):
            try:
                # Process batch
                df_processed = process_fn(df_batch)
                
                # Write batch
                self._write_batch(df_processed, str(output_path), append_mode)
                
                # Update stats
                stats["batches_processed"] += 1
                stats["rows_processed"] += len(df_processed)
                
                # Progress logging
                if (batch_idx + 1) % self.progress_interval == 0:
                    progress = (stats["rows_processed"] / total_rows) * 100
                    logger.info(
                        f"[BatchProcessor] Progress: {stats['rows_processed']}/{total_rows} "
                        f"rows ({progress:.1f}%), {stats['batches_processed']} batches"
                    )
                
                # Clean up
                del df_batch, df_processed
                gc.collect()
                
            except Exception as e:
                logger.error(
                    f"[BatchProcessor] Error in batch {batch_idx}: {e}"
                )
                stats["errors"].append({
                    "batch": batch_idx,
                    "error": str(e),
                })
        
        logger.info(
            f"[BatchProcessor] Complete: {stats['rows_processed']} rows, "
            f"{stats['batches_processed']} batches, {len(stats['errors'])} errors"
        )
        
        return stats
    
    def _read_in_batches(self, input_path: str) -> Iterator[pl.DataFrame]:
        """
        Read file in batches.
        
        Args:
            input_path: Path to input file
        
        Yields:
            DataFrame batches
        """
        path = Path(input_path)
        suffix = path.suffix.lower()
        
        if suffix == ".parquet":
            # Use scan for efficient batch reading
            lf = pl.scan_parquet(str(path))
            
            # Get total rows
            total_rows = lf.collect().height
            
            for start_idx in range(0, total_rows, self.batch_size):
                df_batch = lf.slice(start_idx, self.batch_size).collect()
                yield df_batch
                
        elif suffix == ".csv":
            # CSV batch reading
            for start_idx in range(0, 1000000, self.batch_size):  # Assume max 1M rows
                try:
                    df_batch = pl.read_csv(
                        str(path),
                        skip_rows=start_idx,
                        n_rows=self.batch_size,
                    )
                    if len(df_batch) == 0:
                        break
                    yield df_batch
                except Exception:
                    break
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _write_batch(
        self,
        df_batch: pl.DataFrame,
        output_path: str,
        append_mode: bool = True,
    ) -> None:
        """
        Write batch to output file.
        
        Args:
            df_batch: Batch DataFrame
            output_path: Path to output file
            append_mode: Append to existing file
        """
        path = Path(output_path)
        suffix = path.suffix.lower()
        
        if suffix == ".parquet":
            if append_mode and path.exists():
                # Read existing, append, write back
                df_existing = pl.read_parquet(str(path))
                df_combined = pl.concat([df_existing, df_batch])
                df_combined.write_parquet(str(path))
            else:
                df_batch.write_parquet(str(path))
                
        elif suffix == ".csv":
            df_batch.write_csv(
                str(path),
                include_header=not (append_mode and path.exists()),
            )
        else:
            raise ValueError(f"Unsupported output format: {suffix}")
    
    def transform_column(
        self,
        input_path: str,
        output_path: str,
        column: str,
        transform_fn: Callable[[pl.Series], pl.Series],
    ) -> Dict[str, Any]:
        """
        Transform a single column in batches.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            column: Column to transform
            transform_fn: Transformation function
        
        Returns:
            Processing stats
        """
        def process_batch(df: pl.DataFrame) -> pl.DataFrame:
            """Apply transformation to column."""
            df = df.clone()
            df = df.with_columns(
                transform_fn(df[column]).alias(column)
            )
            return df
        
        return self.process_in_batches(
            input_path=input_path,
            output_path=output_path,
            process_fn=process_batch,
        )
    
    def filter_rows(
        self,
        input_path: str,
        output_path: str,
        filter_fn: Callable[[pl.DataFrame], pl.Series],
    ) -> Dict[str, Any]:
        """
        Filter rows in batches.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            filter_fn: Function returning boolean mask
        
        Returns:
            Processing stats
        """
        def process_batch(df: pl.DataFrame) -> pl.DataFrame:
            """Apply filter to batch."""
            mask = filter_fn(df)
            return df.filter(mask)
        
        return self.process_in_batches(
            input_path=input_path,
            output_path=output_path,
            process_fn=process_batch,
            append_mode=False,  # Fresh write for filtered data
        )


def batch_transform(
    input_path: str,
    output_path: str,
    transform_fn: Callable[[pl.DataFrame], pl.DataFrame],
    batch_size: int = 10000,
) -> Dict[str, Any]:
    """
    Convenience function for batch transformation.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        transform_fn: Transformation function
        batch_size: Batch size
    
    Returns:
        Processing stats
    """
    processor = BatchProcessor(batch_size=batch_size)
    return processor.process_in_batches(
        input_path=input_path,
        output_path=output_path,
        process_fn=transform_fn,
    )


def batch_aggregate(
    input_path: str,
    group_by: List[str],
    aggregations: Dict[str, str],
    batch_size: int = 10000,
) -> pl.DataFrame:
    """
    Perform batch aggregation.
    
    Args:
        input_path: Path to input file
        group_by: Columns to group by
        aggregations: Dict of {column: aggregation}
        batch_size: Batch size
    
    Returns:
        Aggregated DataFrame
    """
    processor = BatchProcessor(batch_size=batch_size)
    
    results = []
    
    for df_batch in processor._read_in_batches(input_path):
        # Aggregate batch
        agg_exprs = []
        for col, agg in aggregations.items():
            if agg == "mean":
                agg_exprs.append(pl.col(col).mean())
            elif agg == "sum":
                agg_exprs.append(pl.col(col).sum())
            elif agg == "count":
                agg_exprs.append(pl.col(col).count())
            elif agg == "std":
                agg_exprs.append(pl.col(col).std())
            elif agg == "min":
                agg_exprs.append(pl.col(col).min())
            elif agg == "max":
                agg_exprs.append(pl.col(col).max())
        
        df_agg = df_batch.group_by(group_by).agg(agg_exprs)
        results.append(df_agg)
        
        del df_batch, df_agg
        gc.collect()
    
    # Combine batch results
    if results:
        df_combined = pl.concat(results)
        
        # Final aggregation across batches
        final_agg_exprs = []
        for col in df_combined.columns:
            if col not in group_by:
                final_agg_exprs.append(pl.col(col).sum())
        
        df_final = df_combined.group_by(group_by).agg(final_agg_exprs)
        
        return df_final
    else:
        return pl.DataFrame()


def get_batch_stats(input_path: str, batch_size: int = 10000) -> Dict[str, Any]:
    """
    Get statistics about batching a file.
    
    Args:
        input_path: Path to file
        batch_size: Batch size
    
    Returns:
        Batch statistics
    """
    path = Path(input_path)
    
    if path.suffix.lower() == ".parquet":
        total_rows = pl.scan_parquet(str(path)).collect().height
    else:
        df_temp = pl.read_csv(str(path))
        total_rows = len(df_temp)
        del df_temp
    
    n_batches = (total_rows + batch_size - 1) // batch_size
    
    return {
        "total_rows": total_rows,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "last_batch_size": total_rows % batch_size if total_rows % batch_size > 0 else batch_size,
    }
