"""Inject data corruption and verify shields catch it."""

def test_row_count_change_detected():
    """A join that duplicates rows is caught by Data Integrity."""
    # Create train with 1000 rows
    # Execute code that does a bad join producing 1500 rows
    # Verify: sandbox returns integrity_ok=False or Data Integrity shield fires
    pass

def test_all_null_column_detected():
    """A feature that's all-null is caught by output validation."""
    # Execute code that creates a column of all nulls
    # Verify: output validation flags "constant column" or "all null"
    pass

def test_nan_predictions_caught():
    """Predictions with NaN values are caught by output validation."""
    # Execute code that produces predictions with NaN
    # Verify: output validation flags "predictions contain NaN"
    pass

def test_constant_predictions_caught():
    """All-same predictions are caught by output validation."""
    # Execute code that predicts 0.5 for every row
    # Verify: output validation flags "all predictions identical"
    pass
