# core/state_validator.py

"""
State validation for Professor pipeline.

FLAW-3.1 FIX: State Schema Runtime Validation
FLAW-3.2 FIX: State Validation Between Agents

- Validates state schema at runtime
- Checks required keys between agent transitions
- Validates state types and constraints
- Logs state validation errors
"""

import logging
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

# Define state schema with required/optional keys and types
STATE_SCHEMA = {
    # Identity - Required
    "session_id": {"type": str, "required": True},
    "created_at": {"type": str, "required": True},
    
    # Competition - Required
    "competition_name": {"type": str, "required": True},
    "task_type": {"type": str, "required": True},
    "competition_context": {"type": dict, "required": False},
    
    # Intel - Required after competition_intel
    "competition_brief_path": {"type": str, "required": False},
    "competition_brief": {"type": dict, "required": False},
    "intel_brief_path": {"type": str, "required": False},
    
    # Data - Required after data_engineer
    "raw_data_path": {"type": str, "required": False},
    "test_data_path": {"type": str, "required": False},
    "sample_submission_path": {"type": str, "required": False},
    "clean_data_path": {"type": str, "required": False},
    "eda_report_path": {"type": str, "required": False},
    "eda_report": {"type": dict, "required": False},
    "schema_path": {"type": str, "required": False},
    "preprocessor_path": {"type": str, "required": False},
    "preprocessor_config_path": {"type": str, "required": False},
    "data_hash": {"type": str, "required": False},
    
    # Schema Authority - Required after data_engineer
    "target_col": {"type": str, "required": False},
    "id_columns": {"type": list, "required": False},
    "dropped_features": {"type": list, "required": False},
    
    # Feature Engineering - Required after feature_factory
    "feature_manifest": {"type": (dict, type(None)), "required": False},
    "feature_candidates": {"type": list, "required": False},
    "round1_features": {"type": list, "required": False},
    "round2_features": {"type": list, "required": False},
    "feature_factory_checkpoint": {"type": dict, "required": False},
    "feature_order": {"type": list, "required": False},
    "feature_data_path": {"type": str, "required": False},
    "feature_data_path_test": {"type": str, "required": False},
    
    # Validation - Required after validation_architect
    "cv_strategy": {"type": dict, "required": False},
    "metric_contract": {"type": dict, "required": False},
    "cv_scores": {"type": list, "required": False},
    "cv_mean": {"type": float, "required": False},
    
    # Models - Required after ml_optimizer
    "model_registry": {"type": list, "required": False},
    "best_params": {"type": dict, "required": False},
    "optuna_study_path": {"type": str, "required": False},
    "oof_predictions_path": {"type": str, "required": False},
    "test_predictions_path": {"type": str, "required": False},
    
    # Critic - Required after red_team_critic
    "critic_verdict": {"type": dict, "required": False},
    "critic_verdict_path": {"type": str, "required": False},
    "critic_severity": {"type": str, "required": False},
    "replan_requested": {"type": bool, "required": False},
    "replan_remove_features": {"type": list, "required": False},
    "replan_rerun_nodes": {"type": list, "required": False},
    "competition_fingerprint": {"type": dict, "required": False},
    "warm_start_priors": {"type": list, "required": False},
    
    # Supervisor - Required after supervisor_replan
    "features_dropped": {"type": list, "required": False},
    
    # Ensemble - Required after ensemble_architect
    "ensemble_selection": {"type": dict, "required": False},
    "selected_models": {"type": list, "required": False},
    "ensemble_weights": {"type": dict, "required": False},
    "ensemble_oof": {"type": list, "required": False},
    "prize_candidates": {"type": list, "required": False},
    
    # Submission - Required after submit
    "submission_path": {"type": str, "required": False},
    "submission_log": {"type": list, "required": False},
    
    # Routing - Required
    "dag": {"type": list, "required": False},
    "current_node": {"type": str, "required": False},
    "next_node": {"type": str, "required": False},
    "error_count": {"type": int, "required": False},
    "escalation_level": {"type": str, "required": False},
    
    # Circuit Breaker - Required
    "current_node_failure_count": {"type": int, "required": False},
    "error_context": {"type": list, "required": False},
    "dag_version": {"type": int, "required": False},
    "macro_replan_requested": {"type": bool, "required": False},
    "macro_replan_reason": {"type": str, "required": False},
    "pipeline_halted": {"type": bool, "required": False},
    "triage_mode": {"type": bool, "required": False},
    "budget_remaining_usd": {"type": float, "required": False},
    "budget_limit_usd": {"type": float, "required": False},
    
    # Parallel Execution
    "parallel_groups": {"type": dict, "required": False},
    
    # Budget - Required
    "cost_tracker": {"type": dict, "required": True},
    
    # HITL - Required
    "hitl_required": {"type": bool, "required": False},
    "hitl_prompt": {"type": dict, "required": False},
    "hitl_checkpoint_key": {"type": str, "required": False},
    "hitl_intervention_id": {"type": int, "required": False},
    "hitl_intervention_label": {"type": str, "required": False},
    "skip_data_validation": {"type": bool, "required": False},
    "null_threshold": {"type": float, "required": False},
    "impute_strategy": {"type": str, "required": False},
    "lgbm_override": {"type": dict, "required": False},
    "model_fallback": {"type": str, "required": False},
    "data_sample_fraction": {"type": float, "required": False},
    "api_timeout_multiplier": {"type": float, "required": False},
    "api_backoff_enabled": {"type": bool, "required": False},
    "llm_provider": {"type": str, "required": False},
    "debug_logging": {"type": bool, "required": False},
    
    # Memory Monitoring
    "memory_peak_gb": {"type": float, "required": False},
    "memory_oom_risk": {"type": bool, "required": False},
    "optuna_pruned_trials": {"type": int, "required": False},
    
    # Feature Filtering
    "null_importance_result_path": {"type": str, "required": False},
    "features_dropped_stage1": {"type": list, "required": False},
    "features_dropped_stage2": {"type": list, "required": False},
    "features_gate_passed": {"type": list, "required": False},
    "features_gate_dropped": {"type": list, "required": False},
    
    # Feature Factory Rounds 3-5
    "round3_features": {"type": list, "required": False},
    "round4_features": {"type": list, "required": False},
    "round5_features": {"type": list, "required": False},
    
    # Pseudo-Labeling
    "pseudo_label_data_path": {"type": str, "required": False},
    "pseudo_labels_applied": {"type": bool, "required": False},
    "pseudo_label_cv_improvement": {"type": float, "required": False},
    
    # Performance Monitoring (FLAW-6.1)
    "performance_log": {"type": list, "required": False},
    
    # Reproducibility (FLAW-10.2)
    "reproducibility_report_path": {"type": str, "required": False},
}

# Define required keys per pipeline stage
PIPELINE_STAGE_REQUIREMENTS = {
    "initial": {
        "required": ["session_id", "created_at", "competition_name", "task_type", "cost_tracker"],
    },
    "post_competition_intel": {
        "required": ["competition_brief_path", "competition_brief"],
    },
    "post_data_engineer": {
        "required": [
            "clean_data_path", "schema_path", "preprocessor_path",
            "data_hash", "target_col", "id_columns", "task_type",
            "test_data_path", "sample_submission_path",
        ],
    },
    "post_eda_agent": {
        "required": ["eda_report_path", "eda_report"],
    },
    "post_validation_architect": {
        "required": ["cv_strategy", "metric_contract"],
    },
    "post_feature_factory": {
        "required": ["feature_manifest", "feature_data_path"],
    },
    "post_ml_optimizer": {
        "required": [
            "model_registry", "cv_scores", "cv_mean",
            "feature_order", "oof_predictions_path",
        ],
    },
    "post_ensemble_architect": {
        "required": ["ensemble_selection", "selected_models"],
    },
    "post_pseudo_label_agent": {
        "required": ["pseudo_labels_applied", "pseudo_label_cv_improvement"],
    },
    "post_submission_strategist": {
        "required": [
            "submission_a_path", "submission_b_path", "submission_path",
            "submission_a_model", "submission_b_model",
            "submission_freeze_active",
        ],
    },
    "post_publisher": {
        "required": ["report_path", "report_written"],
    },
    "post_qa_gate": {
        "required": ["qa_passed", "qa_failures"],
    },
    "post_submit": {
        "required": ["submission_path"],
    },
}


class StateValidationError(Exception):
    """Raised when state validation fails."""
    def __init__(self, message: str, missing_keys: List[str] = None, type_errors: List[str] = None):
        super().__init__(message)
        self.missing_keys = missing_keys or []
        self.type_errors = type_errors or []


class StateValidator:
    """Validates Professor state at runtime."""
    
    def __init__(self, strict: bool = False):
        """
        Initialize state validator.
        
        Args:
            strict: If True, raise exceptions on validation errors.
                   If False, log warnings only.
        """
        self.strict = strict
        self.validation_history = []
    
    def validate_state(
        self,
        state: Dict[str, Any],
        stage: str = "initial",
        node_name: str = "",
    ) -> bool:
        """
        Validate state schema and stage requirements.
        
        Args:
            state: State dict to validate
            stage: Pipeline stage (e.g., "post_data_engineer")
            node_name: Name of node that produced this state
        
        Returns:
            True if valid, False otherwise
        """
        errors = []
        warnings = []
        
        # Validate schema types
        type_errors = self._validate_types(state)
        errors.extend(type_errors)
        
        # Validate stage requirements
        stage_errors = self._validate_stage_requirements(state, stage)
        errors.extend(stage_errors)
        
        # Record validation
        result = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "node": node_name,
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
        self.validation_history.append(result)
        
        # Handle errors
        if errors:
            error_msg = f"State validation failed at {stage}/{node_name}: {'; '.join(errors)}"
            if self.strict:
                raise StateValidationError(
                    error_msg,
                    missing_keys=[e for e in errors if "Missing" in e],
                    type_errors=[e for e in errors if "Type" in e],
                )
            else:
                logger.error(f"[StateValidator] {error_msg}")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"[StateValidator] {warning}")
        
        return len(errors) == 0
    
    def _validate_types(self, state: Dict[str, Any]) -> List[str]:
        """Validate types of state values."""
        errors = []
        
        for key, value in state.items():
            if key not in STATE_SCHEMA:
                # Unknown key - log but don't error
                logger.debug(f"[StateValidator] Unknown state key: {key}")
                continue
            
            schema = STATE_SCHEMA[key]
            expected_type = schema.get("type")
            
            if expected_type and value is not None:
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Type error for '{key}': expected {expected_type}, "
                        f"got {type(value).__name__}"
                    )
        
        return errors
    
    def _validate_stage_requirements(
        self,
        state: Dict[str, Any],
        stage: str,
    ) -> List[str]:
        """Validate stage-specific required keys."""
        errors = []
        
        if stage not in PIPELINE_STAGE_REQUIREMENTS:
            return errors  # Unknown stage, skip validation
        
        requirements = PIPELINE_STAGE_REQUIREMENTS[stage]
        required_keys = requirements.get("required", [])
        
        for key in required_keys:
            if key not in state or state[key] is None:
                errors.append(f"Missing required key for {stage}: {key}")
        
        return errors
    
    def get_validation_summary(self) -> dict:
        """Get summary of validation history."""
        if not self.validation_history:
            return {
                "total_validations": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
            }
        
        passed = sum(1 for v in self.validation_history if v["valid"])
        failed = len(self.validation_history) - passed
        
        return {
            "total_validations": len(self.validation_history),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / len(self.validation_history) * 100, 2),
            "recent_errors": [
                v for v in self.validation_history[-10:]
                if not v["valid"]
            ],
        }


# Global validator instance
_validator: Optional[StateValidator] = None


def get_validator(strict: bool = False) -> StateValidator:
    """Get or create global state validator."""
    global _validator
    
    if _validator is None:
        _validator = StateValidator(strict=strict)
    elif strict != _validator.strict:
        _validator.strict = strict
    
    return _validator


def validate_state(
    state: Dict[str, Any],
    stage: str = "initial",
    node_name: str = "",
    strict: bool = False,
) -> bool:
    """
    Convenience function to validate state.
    
    Args:
        state: State dict to validate
        stage: Pipeline stage
        node_name: Node name
        strict: Raise exception on failure
    
    Returns:
        True if valid
    """
    validator = get_validator(strict=strict)
    return validator.validate_state(state, stage, node_name)


def log_validation_summary() -> None:
    """Log validation summary."""
    validator = get_validator()
    summary = validator.get_validation_summary()
    
    logger.info("=" * 70)
    logger.info("STATE VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total validations: {summary['total_validations']}")
    logger.info(f"Passed: {summary['passed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Pass rate: {summary['pass_rate']}%")
    
    if summary['recent_errors']:
        logger.warning(f"Recent errors: {len(summary['recent_errors'])}")
        for error in summary['recent_errors'][-3:]:
            logger.warning(f"  - {error['stage']}/{error['node']}: {error['errors']}")
    
    logger.info("=" * 70)
