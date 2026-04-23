# tests/contracts/test_leakage_precheck_contract.py

import pytest
import time
from guards.leakage_precheck import check_code_for_leakage
from tools.sandbox import run_in_sandbox

# ── Tests ───────────────────────────────────────────────────────────────────

class TestLeakagePrecheckContract:
    """
    Contract: Pre-Execution Leakage Check (Component 1)
    """

    def test_scaler_on_full_data_blocked(self):
        """Verify StandardScaler().fit_transform(X) triggers leakage detection."""
        code = "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)"
        res = check_code_for_leakage(code)
        assert res["leakage_detected"] is True
        assert "fit_transform on variable named X" in res["description"]

    def test_scaler_on_train_allowed(self):
        """Verify StandardScaler().fit_transform(X_train) is NOT blocked."""
        code = "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X_train)"
        res = check_code_for_leakage(code)
        assert res["leakage_detected"] is False

    def test_pipeline_not_blocked(self):
        """Verify sklearn Pipeline is whitelisted."""
        code = "from sklearn.pipeline import Pipeline\npipe = Pipeline([('scaler', StandardScaler())])\npipe.fit(X, y)"
        res = check_code_for_leakage(code)
        assert res["leakage_detected"] is False

    def test_cross_val_score_not_blocked(self):
        """Verify code inside cross_val_score is safe."""
        code = "scores = cross_val_score(StandardScaler().fit(X), X, y)"
        res = check_code_for_leakage(code)
        assert res["leakage_detected"] is False

    def test_concat_fit_blocked(self):
        """Verify fitting on concatenated data is blocked."""
        code = "le = LabelEncoder()\nle.fit(pd.concat([train, test]))"
        res = check_code_for_leakage(code)
        assert res["leakage_detected"] is True

    def test_blocked_result_has_line_number(self):
        code = "import os\n# some comment\nscaler.fit_transform(X)"
        res = check_code_for_leakage(code)
        assert res["line"] == 3

    def test_blocked_result_has_fix_suggestion(self):
        code = "scaler.fit_transform(X)"
        res = check_code_for_leakage(code)
        assert len(res["fix_suggestion"]) > 20

    def test_non_training_agents_skip_check(self):
        """Verify agents like eda_agent are not blocked by the sandbox."""
        code = "scaler.fit_transform(X)" # This would normally be blocked
        res = run_in_sandbox(code, agent_name="eda_agent")
        # Should NOT have the blocked diagnostic
        assert "leakage_precheck" not in res.get("diagnostics", {})

    def test_precheck_cost_under_1ms(self):
        """Verify performance for a 200-line code block."""
        long_code = "\n".join(["# line %d" % i for i in range(199)] + ["scaler.fit_transform(X_train)"])
        start = time.perf_counter()
        check_code_for_leakage(long_code)
        duration = (time.perf_counter() - start) * 1000
        assert duration < 1.0 # < 1ms
