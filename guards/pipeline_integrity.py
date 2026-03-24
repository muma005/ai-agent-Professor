# guards/pipeline_integrity.py
"""
Pipeline Integrity Gate — validates state at critical checkpoints.

Three checkpoints:
  POST_DATA_ENGINEER: Schema authority is established
  POST_EDA:           Analysis complete, drops manifest written
  POST_MODEL:         Model exists, predictions are valid

Each checkpoint runs a set of checks. Any FAIL halts the pipeline
with a clear error message. WARN continues but logs the issue.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class IntegrityCheckResult:
    """Result of a single integrity check."""
    def __init__(self, name: str, passed: bool, severity: str = "FAIL",
                 message: str = "", details: dict = None):
        self.name = name
        self.passed = passed
        self.severity = severity  # "FAIL" or "WARN"
        self.message = message
        self.details = details or {}

    def __repr__(self):
        status = "PASS" if self.passed else self.severity
        return f"[{status}] {self.name}: {self.message}"


class IntegrityGateResult:
    """Aggregated result of all checks at a checkpoint."""
    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        self.checks: list[IntegrityCheckResult] = []

    def add(self, check: IntegrityCheckResult):
        self.checks.append(check)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def has_failures(self) -> bool:
        return any(not c.passed and c.severity == "FAIL" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(not c.passed and c.severity == "WARN" for c in self.checks)

    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        failed = sum(1 for c in self.checks if not c.passed and c.severity == "FAIL")
        warned = sum(1 for c in self.checks if not c.passed and c.severity == "WARN")
        return (
            f"[IntegrityGate:{self.checkpoint}] "
            f"{passed}/{total} passed, {failed} FAIL, {warned} WARN"
        )

    def report(self) -> str:
        lines = [self.summary(), ""]
        for c in self.checks:
            lines.append(f"  {c}")
        return "\n".join(lines)


# ── Checkpoint 1: POST_DATA_ENGINEER ─────────────────────────────

def check_post_data_engineer(state: dict) -> IntegrityGateResult:
    """
    Validates that data_engineer has established schema authority.
    Run AFTER data_engineer, BEFORE eda_agent.
    """
    result = IntegrityGateResult("POST_DATA_ENGINEER")

    # 1. target_col must be set and non-empty
    target = state.get("target_col", "")
    result.add(IntegrityCheckResult(
        name="target_col_set",
        passed=bool(target and target != ""),
        severity="FAIL",
        message=f"target_col='{target}'" if target else "target_col is empty or missing",
    ))

    # 2. task_type must be one of the valid values
    task_type = state.get("task_type", "")
    valid_types = {"binary", "multiclass", "regression"}
    result.add(IntegrityCheckResult(
        name="task_type_valid",
        passed=task_type in valid_types,
        severity="FAIL",
        message=f"task_type='{task_type}'" if task_type in valid_types else
                f"task_type='{task_type}' not in {valid_types}",
    ))

    # 3. clean_data_path must exist on disk
    clean_path = state.get("clean_data_path", "")
    result.add(IntegrityCheckResult(
        name="clean_data_exists",
        passed=bool(clean_path and os.path.exists(clean_path)),
        severity="FAIL",
        message=f"clean_data_path='{clean_path}'" if clean_path else "clean_data_path missing",
    ))

    # 4. schema_path must exist on disk
    schema_path = state.get("schema_path", "")
    result.add(IntegrityCheckResult(
        name="schema_exists",
        passed=bool(schema_path and os.path.exists(schema_path)),
        severity="FAIL",
        message=f"schema_path='{schema_path}'" if schema_path else "schema_path missing",
    ))

    # 5. preprocessor_path must exist on disk
    session_id = state.get("session_id", "")
    preprocessor_path = state.get("preprocessor_path", f"outputs/{session_id}/preprocessor.pkl")
    result.add(IntegrityCheckResult(
        name="preprocessor_exists",
        passed=bool(preprocessor_path and os.path.exists(preprocessor_path)),
        severity="FAIL",
        message=f"preprocessor at '{preprocessor_path}'" if os.path.exists(preprocessor_path or "")
                else "preprocessor.pkl not found",
    ))

    # 6. id_columns must be a list (can be empty)
    id_cols = state.get("id_columns")
    result.add(IntegrityCheckResult(
        name="id_columns_is_list",
        passed=isinstance(id_cols, list),
        severity="FAIL",
        message=f"id_columns={id_cols}" if isinstance(id_cols, list)
                else f"id_columns is {type(id_cols).__name__}, expected list",
    ))

    return result


# ── Checkpoint 2: POST_EDA ───────────────────────────────────────

def check_post_eda(state: dict) -> IntegrityGateResult:
    """
    Validates that EDA has completed and produced a valid report.
    Run AFTER eda_agent, BEFORE feature_factory.
    """
    result = IntegrityGateResult("POST_EDA")

    # 1. eda_report must be a non-empty dict
    report = state.get("eda_report", {})
    result.add(IntegrityCheckResult(
        name="eda_report_exists",
        passed=bool(isinstance(report, dict) and report),
        severity="FAIL",
        message=f"eda_report has {len(report)} keys" if report else "eda_report empty or missing",
    ))

    # 2. dropped_features must be a list (can be empty)
    drops = state.get("dropped_features")
    result.add(IntegrityCheckResult(
        name="dropped_features_is_list",
        passed=isinstance(drops, list),
        severity="FAIL",
        message=f"dropped_features has {len(drops)} items" if isinstance(drops, list)
                else f"dropped_features is {type(drops).__name__}, expected list",
    ))

    # 3. target_col must still be set (not wiped by EDA)
    target = state.get("target_col", "")
    result.add(IntegrityCheckResult(
        name="target_col_preserved",
        passed=bool(target),
        severity="FAIL",
        message=f"target_col='{target}'" if target else "target_col was cleared by EDA!",
    ))

    # 4. EDA report should contain imbalance_ratio
    target_dist = report.get("target_distribution", {})
    has_imbalance = "imbalance_ratio" in target_dist
    result.add(IntegrityCheckResult(
        name="imbalance_ratio_present",
        passed=has_imbalance,
        severity="WARN",
        message=f"imbalance_ratio={target_dist.get('imbalance_ratio')}" if has_imbalance
                else "target_distribution missing imbalance_ratio",
    ))

    return result


# ── Checkpoint 3: POST_MODEL ─────────────────────────────────────

def check_post_model(state: dict) -> IntegrityGateResult:
    """
    Validates that model training completed and produced valid artifacts.
    Run AFTER ml_optimizer, BEFORE submit.
    """
    result = IntegrityGateResult("POST_MODEL")

    # 1. model_registry must have at least one entry
    registry = state.get("model_registry", [])
    result.add(IntegrityCheckResult(
        name="model_registry_populated",
        passed=bool(registry and len(registry) > 0),
        severity="FAIL",
        message=f"model_registry has {len(registry)} entries" if registry
                else "model_registry is empty",
    ))

    # 2. Best model file must exist on disk
    if registry:
        best_model_path = registry[0].get("model_path", "")
        result.add(IntegrityCheckResult(
            name="best_model_file_exists",
            passed=bool(best_model_path and os.path.exists(best_model_path)),
            severity="FAIL",
            message=f"model at '{best_model_path}'" if os.path.exists(best_model_path or "")
                    else f"model file not found: '{best_model_path}'",
        ))

    # 3. best_score must be set
    best_score = state.get("best_score")
    result.add(IntegrityCheckResult(
        name="best_score_set",
        passed=best_score is not None and best_score != 0.0,
        severity="WARN",
        message=f"best_score={best_score}" if best_score else "best_score is 0.0 or None",
    ))

    # 4. pipeline_halted must NOT be True
    halted = state.get("pipeline_halted", False)
    result.add(IntegrityCheckResult(
        name="pipeline_not_halted",
        passed=not halted,
        severity="FAIL",
        message="pipeline running normally" if not halted
                else f"pipeline HALTED: {state.get('pipeline_halt_reason', 'unknown')}",
    ))

    return result


# ── Public API ───────────────────────────────────────────────────

def run_integrity_gate(state: dict, checkpoint: str) -> IntegrityGateResult:
    """
    Run the integrity gate at the specified checkpoint.

    Args:
        state: current ProfessorState dict
        checkpoint: "POST_DATA_ENGINEER", "POST_EDA", or "POST_MODEL"

    Returns:
        IntegrityGateResult with all check results

    Raises:
        ValueError if any FAIL check fails
    """
    if checkpoint == "POST_DATA_ENGINEER":
        result = check_post_data_engineer(state)
    elif checkpoint == "POST_EDA":
        result = check_post_eda(state)
    elif checkpoint == "POST_MODEL":
        result = check_post_model(state)
    else:
        raise ValueError(f"Unknown checkpoint: {checkpoint}")

    # Log the report
    report = result.report()
    if result.has_failures:
        logger.error(report)
        print(f"\n{'='*60}\n{report}\n{'='*60}\n")
        raise ValueError(
            f"Pipeline Integrity Gate FAILED at {checkpoint}.\n{report}"
        )
    elif result.has_warnings:
        logger.warning(report)
        print(f"\n{report}\n")
    else:
        logger.info(report)
        print(f"\n{result.summary()}\n")

    return result
