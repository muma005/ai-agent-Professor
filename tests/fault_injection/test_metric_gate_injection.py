"""Inject wrong metrics and verify the gate catches them."""

def test_wrong_metric_detected():
    """Configure unknown metric name. Gate must fail."""
    from shields.metric_gate import verify_metric
    
    # Unknown metric should FAIL verification
    success, message = verify_metric("nonexistent_metric", "classification")
    
    assert success == False
    assert "Unknown metric" in message

def test_correct_metric_passes():
    """Configure correct metric name. Gate must pass."""
    from shields.metric_gate import verify_metric
    
    success, message = verify_metric("roc_auc", "classification")
    
    assert success == True
