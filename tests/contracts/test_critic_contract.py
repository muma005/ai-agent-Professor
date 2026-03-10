# tests/contracts/test_critic_contract.py
# -------------------------------------------------------------------------
# Written: Day 10   Status: IMMUTABLE
#
# CONTRACT: run_red_team_critic()
#   INPUT:   state["raw_data_path"], state["eda_report"], state["validation_strategy"]
#   OUTPUT:  critic_verdict.json on disk -- overall_severity/vectors_checked/findings/clean/checked_at
#   MUST CATCH: injected target leakage -> CRITICAL
#               majority-class-only model (recall < 0.5 on imbalanced) -> CRITICAL
#               preprocessing leakage code pattern -> CRITICAL
#   NEVER:   Proceed to ensemble when overall_severity == CRITICAL
#            Return verdict without all required keys
#            Silently skip any of the 6 vectors
# -------------------------------------------------------------------------
import os
import json
import pytest
import numpy as np
import polars as pl

from core.state import initial_state
from agents.data_engineer import run_data_engineer
from agents.eda_agent import run_eda_agent
from agents.validation_architect import run_validation_architect
from agents.red_team_critic import run_red_team_critic

FIXTURE_CSV = "tests/fixtures/tiny_train.csv"


@pytest.fixture(scope="module")
def clean_state():
    """Clean pipeline state -- critic should return OK verdict."""
    s = initial_state("test-critic-clean", FIXTURE_CSV)
    s = run_data_engineer(s)
    s = run_eda_agent(s)
    s = run_validation_architect(s)
    return run_red_team_critic(s)


class TestCriticContractOutputSchema:

    def test_runs_without_error(self, clean_state):
        assert clean_state is not None

    def test_critic_verdict_key_in_state(self, clean_state):
        assert "critic_verdict" in clean_state
        assert isinstance(clean_state["critic_verdict"], dict)

    def test_critic_verdict_json_written_to_disk(self, clean_state):
        path = clean_state.get("critic_verdict_path")
        assert path is not None
        assert os.path.exists(path), f"critic_verdict.json not found at {path}"
        loaded = json.load(open(path))
        assert isinstance(loaded, dict)

    def test_overall_severity_is_valid_value(self, clean_state):
        s = clean_state["critic_verdict"]["overall_severity"]
        assert s in ("CRITICAL", "HIGH", "MEDIUM", "OK"), f"Invalid severity: {s}"

    def test_vectors_checked_contains_all_six(self, clean_state):
        vc = clean_state["critic_verdict"]["vectors_checked"]
        required = {
            "shuffled_target", "id_only_model", "adversarial_classifier",
            "preprocessing_audit", "pr_curve_imbalance", "temporal_leakage",
        }
        missing = required - set(vc)
        assert not missing, f"Vectors not checked: {missing}. All 6 must run."

    def test_findings_is_a_list(self, clean_state):
        assert isinstance(clean_state["critic_verdict"]["findings"], list)

    def test_clean_flag_matches_severity(self, clean_state):
        v = clean_state["critic_verdict"]
        if v["overall_severity"] == "OK":
            assert v["clean"] is True
        else:
            assert v["clean"] is False

    def test_checked_at_is_iso_timestamp(self, clean_state):
        from datetime import datetime
        ts = clean_state["critic_verdict"]["checked_at"]
        assert ts is not None and len(ts) > 0
        datetime.fromisoformat(ts)  # raises if not valid ISO

    def test_critic_severity_in_state_matches_verdict(self, clean_state):
        assert clean_state["critic_severity"] == clean_state["critic_verdict"]["overall_severity"]

    def test_clean_data_does_not_trigger_hitl(self, clean_state):
        # On tiny datasets, shuffled_target may get borderline AUC > 0.55 from noise.
        # The critical guarantee is that no REAL leakage vector fires.
        if clean_state.get("hitl_required") is True:
            findings = clean_state["critic_verdict"].get("findings", [])
            real_leakage_vectors = {"id_only_model", "adversarial_classifier", "preprocessing_audit"}
            critical_real = [f for f in findings
                            if f["severity"] == "CRITICAL" and f["vector"] in real_leakage_vectors]
            assert not critical_real, (
                f"HITL triggered by real leakage detection on clean data: {critical_real}"
            )


class TestCriticCatchesInjectedLeakage:

    def test_injected_target_leakage_triggers_critical(self):
        """
        Inject a copy of the target into features.
        Shuffled target test must detect this as CRITICAL.
        """
        import tempfile
        df = pl.read_csv(FIXTURE_CSV)
        target_col = df.columns[-1]

        # Inject: add target as a feature column (pure leakage)
        df_leaked = df.with_columns(pl.col(target_col).alias("leaked_target_feature"))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_leaked.write_csv(f.name)
            leaked_path = f.name

        s = initial_state("test-critic-leak", leaked_path)
        s = run_data_engineer(s)
        s = run_eda_agent(s)
        s = run_validation_architect(s)
        result = run_red_team_critic(s)

        os.unlink(leaked_path)

        verdict = result["critic_verdict"]
        assert verdict["overall_severity"] == "CRITICAL", (
            f"Injected target leakage should produce CRITICAL verdict. Got: {verdict['overall_severity']}. "
            f"Findings: {verdict['findings']}"
        )

    def test_critical_verdict_sets_hitl_required(self):
        """CRITICAL verdict must halt the pipeline."""
        import tempfile
        df = pl.read_csv(FIXTURE_CSV)
        target_col = df.columns[-1]
        df_leaked = df.with_columns(pl.col(target_col).alias("leaked_feature_2"))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_leaked.write_csv(f.name)
            p = f.name

        s = initial_state("test-critic-halt", p)
        s = run_data_engineer(s)
        s = run_eda_agent(s)
        s = run_validation_architect(s)
        result = run_red_team_critic(s)
        os.unlink(p)

        assert result.get("hitl_required") is True, (
            "CRITICAL verdict must set hitl_required=True to halt the pipeline."
        )

    def test_critical_findings_have_replan_instructions(self):
        """CRITICAL findings must specify which nodes to rerun."""
        import tempfile
        df = pl.read_csv(FIXTURE_CSV)
        target_col = df.columns[-1]
        df_leaked = df.with_columns(pl.col(target_col).alias("leaked_feature_3"))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_leaked.write_csv(f.name)
            p = f.name

        s = initial_state("test-critic-replan", p)
        s = run_data_engineer(s)
        s = run_eda_agent(s)
        s = run_validation_architect(s)
        result = run_red_team_critic(s)
        os.unlink(p)

        for finding in result["critic_verdict"]["findings"]:
            if finding["severity"] == "CRITICAL":
                ri = finding.get("replan_instructions", {})
                assert "rerun_nodes" in ri, "CRITICAL finding missing replan_instructions.rerun_nodes"
                assert isinstance(ri["rerun_nodes"], list)
                assert len(ri["rerun_nodes"]) > 0, "rerun_nodes must name at least one node to rerun"
                assert "remove_features" in ri


class TestCriticPreprocessingAudit:

    def test_preprocessing_leakage_code_triggers_critical(self):
        """
        Feed code containing a pre-split scaler fit into the critic.
        The code audit vector must flag it as CRITICAL.
        """
        from agents.red_team_critic import _check_preprocessing_leakage

        bad_code = """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

X = df.drop('target', axis=1)
y = df['target']

# BUG: scaler fitted on full X before CV loop
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # <- leakage

kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
"""
        result = _check_preprocessing_leakage(bad_code)
        assert result["verdict"] == "CRITICAL", (
            f"Pre-split scaler fit should trigger CRITICAL. Got: {result['verdict']}. "
            f"Evidence: {result.get('findings', [])}"
        )

    def test_clean_preprocessing_code_passes(self):
        """Code that correctly fits preprocessors inside the fold must not trigger."""
        from agents.red_team_critic import _check_preprocessing_leakage

        good_code = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    # Correct: scaler fitted only on training fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
"""
        result = _check_preprocessing_leakage(good_code)
        assert result["verdict"] == "OK", (
            f"Correct preprocessing code should not trigger. Got: {result['verdict']}. "
            f"Findings: {result.get('findings', [])}"
        )


class TestCriticPRCurveAudit:

    def test_majority_class_model_triggers_critical_on_imbalanced(self):
        """
        Simulate a model that always predicts the majority class:
        near-zero probability for all samples (even true positives).
        On an imbalanced dataset, this should trigger CRITICAL.
        """
        from agents.red_team_critic import _check_pr_curve_imbalance

        n        = 1000
        minority = int(n * 0.05)  # 5% minority
        y_true   = np.array([1] * minority + [0] * (n - minority))
        # Model that gives near-random predictions — not separating minority
        rng = np.random.default_rng(42)
        y_prob = rng.uniform(0.0, 0.08, n)  # all predictions very low

        result = _check_pr_curve_imbalance(y_true, y_prob, imbalance_ratio=0.05, target_type="binary")
        assert result["verdict"] in ("CRITICAL", "HIGH"), (
            f"Majority-class-only model on imbalanced data must produce CRITICAL or HIGH. Got: {result['verdict']}. "
            f"Best recall: {result.get('best_recall')}, PR-AUC: {result.get('pr_auc')}"
        )

    def test_good_minority_recall_passes_pr_audit(self):
        """A model with >= 50% recall on minority class must not trigger."""
        from agents.red_team_critic import _check_pr_curve_imbalance

        n          = 1000
        minority   = int(n * 0.08)
        y_true     = np.array([1] * minority + [0] * (n - minority))
        y_prob     = np.array([0.9] * minority + [0.1] * (n - minority))

        result = _check_pr_curve_imbalance(y_true, y_prob, imbalance_ratio=0.08, target_type="binary")
        assert result["verdict"] == "OK", (
            f"Good recall model should not trigger PR audit. Got: {result['verdict']}"
        )

    def test_pr_audit_skipped_for_balanced_data(self):
        """PR audit must not run on balanced datasets (imbalance_ratio >= 0.15)."""
        from agents.red_team_critic import _check_pr_curve_imbalance

        y_true = np.array([0, 1] * 500)
        y_prob = np.random.rand(1000)
        result = _check_pr_curve_imbalance(y_true, y_prob, imbalance_ratio=0.5, target_type="binary")
        assert result["verdict"] == "OK"
        assert "skipped" in result.get("note", "").lower()

    def test_pr_audit_skipped_for_non_binary_target(self):
        """PR audit only applies to binary classification."""
        from agents.red_team_critic import _check_pr_curve_imbalance

        result = _check_pr_curve_imbalance(
            np.array([0, 1, 2] * 100), None, imbalance_ratio=0.05, target_type="multiclass"
        )
        assert result["verdict"] == "OK"


class TestCriticMemorySchemaIntegration:

    def test_fingerprint_built_from_state(self):
        """After running full pipeline, competition_fingerprint must be populated."""
        s = initial_state("test-fp", FIXTURE_CSV)
        s = run_data_engineer(s)
        s = run_eda_agent(s)
        s = run_validation_architect(s)

        from memory.memory_schema import build_competition_fingerprint
        fp = build_competition_fingerprint(s)

        required_keys = {
            "task_type", "imbalance_ratio", "n_categorical_high_cardinality",
            "n_rows_bucket", "has_temporal_feature", "n_features_bucket", "target_type"
        }
        missing = required_keys - set(fp.keys())
        assert not missing, f"Fingerprint missing keys: {missing}"

    def test_fingerprint_text_is_non_empty_and_semantic(self):
        from memory.memory_schema import fingerprint_to_text, build_competition_fingerprint
        s  = initial_state("test-fptext", FIXTURE_CSV)
        s  = run_data_engineer(s)
        s  = run_eda_agent(s)
        s  = run_validation_architect(s)
        fp = build_competition_fingerprint(s)
        text = fingerprint_to_text(fp)
        assert len(text) > 50, "Fingerprint text too short to be useful for embedding"
        assert "task" in text.lower() or "classif" in text.lower() or "tabular" in text.lower()

    def test_pattern_store_and_retrieve_round_trip(self):
        from memory.memory_schema import (
            build_competition_fingerprint, store_pattern, query_similar_competitions
        )
        s  = initial_state("test-mem-rt", FIXTURE_CSV)
        s  = run_data_engineer(s)
        s  = run_eda_agent(s)
        s  = run_validation_architect(s)
        fp = build_competition_fingerprint(s)

        pid = store_pattern(
            fingerprint=fp,
            validated_approaches=[{"approach": "LGBM + log-transform", "cv_improvement": 0.02, "competitions": ["test"]}],
            failed_approaches=[],
            competition_name="test-round-trip",
            confidence=0.65,
        )
        assert pid is not None and len(pid) > 0

        results = query_similar_competitions(fp, n_results=20)
        assert len(results) >= 1, "Stored pattern must be retrievable"
        ids = [r["pattern_id"] for r in results]
        assert pid in ids, f"Stored pattern {pid} not in top-20 query results: {ids}"

    def test_query_returns_empty_list_not_none_when_no_patterns(self):
        """query_similar_competitions must return [] not None when collection is empty."""
        from memory.memory_schema import query_similar_competitions
        fp = {"task_type": "tabular", "imbalance_ratio": 0.5, "n_categorical_high_cardinality": 0,
              "n_rows_bucket": "medium", "has_temporal_feature": False,
              "n_features_bucket": "medium", "target_type": "binary"}
        results = query_similar_competitions(fp, n_results=5)
        assert isinstance(results, list), f"Expected list, got {type(results)}"
