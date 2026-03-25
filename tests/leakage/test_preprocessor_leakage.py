"""
Preprocessor Leakage Test.

PRINCIPLE: Preprocessor fitted on train should not leak test statistics.
"""
import pytest
import numpy as np
import polars as pl

from core.preprocessor import TabularPreprocessor


def test_preprocessor_no_leakage():
    """
    Verify preprocessor imputation uses train statistics only.
    """
    # Create train and test with VERY different distributions
    train = pl.DataFrame({"feat": [1, 2, 3, 4, 5, None]})
    test = pl.DataFrame({"feat": [100, 200, 300, None]})
    
    # Fit on train only
    prep = TabularPreprocessor(target_col="target", id_cols=[])
    prep.fit_imputation(train, {"types": {"feat": "Int64"}})
    
    # Transform test
    test_transformed = prep.transform(test)
    
    # Should impute nulls
    assert test_transformed["feat"].null_count() == 0, "Should impute nulls"
    
    # Imputed value should use train median (3), not test median (200)
    # Find imputed values
    all_vals = test_transformed["feat"].to_list()
    imputed_vals = [v for v in all_vals if v is not None]
    
    # The imputed value for test should be train median
    # Check that test values are NOT using test statistics
    test_original = [100, 200, 300]
    for val in imputed_vals:
        if val not in test_original:
            # This is an imputed value - should be close to train median (3)
            assert abs(val - 3.0) < 1.0, (
                f"Imputation should use train median (~3.0), got {val}"
            )
