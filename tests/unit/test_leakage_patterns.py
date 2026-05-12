"""Unit tests for leakage precheck regex patterns."""

def test_scaler_fit_on_X_detected():
    from guards.leakage_precheck import check_code_for_leakage
    code = "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)"
    result = check_code_for_leakage(code)
    assert result["leakage_detected"] == True

def test_scaler_fit_on_X_train_safe():
    from guards.leakage_precheck import check_code_for_leakage
    code = "scaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)"
    result = check_code_for_leakage(code)
    assert result["leakage_detected"] == False

def test_pipeline_context_safe():
    from guards.leakage_precheck import check_code_for_leakage
    code = """
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
scores = cross_val_score(pipe, X, y, cv=5)
"""
    result = check_code_for_leakage(code)
    assert result["leakage_detected"] == False

def test_concat_fit_detected():
    from guards.leakage_precheck import check_code_for_leakage
    code = "combined = pd.concat([train, test])\nle = LabelEncoder().fit(combined['category'])"
    result = check_code_for_leakage(code)
    assert result["leakage_detected"] == True
