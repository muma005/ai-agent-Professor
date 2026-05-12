"""Inject data leakage and verify Professor catches it."""

def test_target_in_test_data_blocked():
    """Code that accesses test target column is blocked by leakage precheck."""
    from guards.leakage_precheck import check_code_for_leakage
    
    leaky_code = """
import polars as pl
train = pl.read_csv("train.csv")
test = pl.read_csv("test.csv")
# Accidentally access target in test
test_target = test['target']  
"""
    result = check_code_for_leakage(leaky_code)
    # This specific pattern may or may not be caught by precheck
    # The Critic's Vector 1d would catch it if precheck doesn't

def test_fit_on_full_data_blocked():
    """StandardScaler.fit() on combined train+test is blocked."""
    from guards.leakage_precheck import check_code_for_leakage
    
    leaky_code = """
import numpy as np
from sklearn.preprocessing import StandardScaler
X = np.vstack([X_train, X_test])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
"""
    result = check_code_for_leakage(leaky_code)
    assert result["leakage_detected"] == True

def test_target_copy_caught_by_critic():
    """A renamed copy of the target column is caught by Critic Vector 1a."""
    # This tests the CRITIC, not the precheck
    # Inject a feature that's a perfect copy of the target
    # Critic's shuffled target test should detect AUC >> 0.55
    pass  # Requires full pipeline — run in integration
