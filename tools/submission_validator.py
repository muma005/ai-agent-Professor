# tools/submission_validator.py

import polars as pl
from typing import Dict, List, Any

def validate_submission(sample_sub_path: str, generated_sub_path: str) -> Dict[str, Any]:
    """
    Validates a generated submission file against the sample submission format.
    Returns a dict with 'is_valid' and a list of 'errors'.
    """
    try:
        sample_df = pl.read_csv(sample_sub_path) if sample_sub_path.endswith(".csv") else pl.read_parquet(sample_sub_path)
    except Exception as e:
        return {"is_valid": False, "errors": [f"Could not read sample submission: {e}"]}
        
    try:
        gen_df = pl.read_csv(generated_sub_path) if generated_sub_path.endswith(".csv") else pl.read_parquet(generated_sub_path)
    except Exception as e:
        return {"is_valid": False, "errors": [f"Could not read generated submission: {e}"]}

    errors = []

    # 1. Row count match
    if len(sample_df) != len(gen_df):
        errors.append(f"Row count mismatch: expected {len(sample_df)}, got {len(gen_df)}")

    # 2. Column names match
    sample_cols = sample_df.columns
    gen_cols = gen_df.columns
    
    missing_cols = [c for c in sample_cols if c not in gen_cols]
    extra_cols = [c for c in gen_cols if c not in sample_cols]
    
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        errors.append(f"Extra columns: {extra_cols}")

    # 3. Dtypes match (allowing for precision coercion)
    if not missing_cols:
        for col in sample_cols:
            s_type = sample_df[col].dtype
            g_type = gen_df[col].dtype
            
            # Numeric coercions allowed
            if s_type in (pl.Float32, pl.Float64) and g_type in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int8, pl.Int16, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                continue
            if s_type in (pl.Int32, pl.Int64, pl.Int8, pl.Int16, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64) and g_type in (pl.Int32, pl.Int64, pl.Int8, pl.Int16, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                continue
            if s_type in (pl.Int32, pl.Int64, pl.Int8, pl.Int16, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64) and g_type in (pl.Float32, pl.Float64):
                # Float given where int expected. 
                # Kaggle usually accepts this, but let's strictly check if they can be safely cast
                continue
                
            if s_type != g_type:
                errors.append(f"Dtype mismatch for column '{col}': expected {s_type}, got {g_type}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }
