# graph/state.py

import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List, Union
from pydantic import BaseModel, Field, ValidationError, ConfigDict

# ── Custom Exceptions ────────────────────────────────────────────────────────
class OwnershipError(Exception):
    """Raised when an agent attempts to write to a field it does not own."""
    pass

class ImmutableFieldError(Exception):
    """Raised when an [IMMUTABLE] field is overwritten after initial set."""
    pass

class SchemaVersionError(Exception):
    """Raised when a checkpoint schema version is unrecognized or incompatible."""
    pass

# ── Field Ownership Map ──────────────────────────────────────────────────────
# Every field from STATE.md must have exactly one owner.
_FIELD_OWNERS = {
    # Core Pipeline
    "session_id": "supervisor",
    "competition_name": "supervisor",
    "competition_url": "supervisor",
    "pipeline_depth": "supervisor",
    "pipeline_depth_auto_detected": "supervisor",
    "pipeline_depth_reason": "supervisor",
    "dag_version": "supervisor",
    "replan_target": "supervisor",
    "state_schema_version": "supervisor",
    "state_size_bytes": "supervisor",

    # Competition Intel
    "competition_description": "competition_intel",
    "competition_type": "competition_intel",
    "target_column": "competition_intel",
    "metric_name": "competition_intel",
    "metric_config": "competition_intel",
    "id_column": "competition_intel",
    "submission_format": "competition_intel",
    "intel_brief": "competition_intel",
    "forbidden_techniques": "competition_intel",

    # Metric Verification
    "metric_verified": "metric_gate",
    "metric_verification_test": "metric_gate",
    "metric_verification_method": "metric_gate",

    # Pre-flight
    "preflight_passed": "preflight",
    "preflight_data_files": "preflight",
    "preflight_warnings": "preflight",
    "preflight_data_size_mb": "preflight",
    "preflight_submission_format": "preflight",
    "preflight_target_type": "preflight",
    "preflight_unsupported_modalities": "preflight",

    # Data Engineer & Integrity
    "raw_data_path": "data_engineer",
    "clean_data_path": "data_engineer",
    "clean_test_path": "data_engineer",
    "data_schema": "data_engineer",
    "drop_candidates": "data_engineer",
    "canonical_train_rows": "data_engineer",
    "canonical_test_rows": "data_engineer",
    "canonical_schema": "data_engineer",
    "canonical_target_stats": "data_engineer",
    "test_data_checksum": "data_engineer",
    "data_integrity_violations": "data_engineer",

    # EDA
    "eda_report": "eda_agent",
    "eda_insights_summary": "eda_agent",
    "eda_mutual_info": "eda_agent",
    "eda_vif_report": "eda_agent",
    "eda_modality_flags": "eda_agent",
    "eda_plots_paths": "eda_agent",
    "eda_plots_delivered": "eda_agent",
    "eda_quick_baseline_importance": "eda_agent",

    # Domain
    "domain_brief": "domain_research",
    "domain_classification": "domain_research",
    "domain_templates_applied": "domain_research",

    # Shift
    "shift_report": "shift_detector",
    "shift_severity": "shift_detector",
    "shift_sample_weights": "shift_detector",

    # Validation
    "validation_strategy": "validation_architect",
    "scorer_name": "validation_architect",
    "scorer_func_path": "validation_architect",

    # Reframer
    "reframe_applied": "problem_reframer",
    "reframe_details": "problem_reframer",

    # Features
    "feature_manifest": "feature_factory",
    "feature_factory_rounds_completed": "feature_factory",
    "features_train_path": "feature_factory",
    "features_test_path": "feature_factory",

    # Creative
    "creative_features_generated": "creative_hypothesis",
    "creative_features_accepted": "creative_hypothesis",

    # Model
    "model_configs": "ml_optimizer",
    "best_model_type": "ml_optimizer",
    "best_model_params": "ml_optimizer",
    "cv_scores": "ml_optimizer",
    "cv_mean": "ml_optimizer",
    "cv_std": "ml_optimizer",
    "oof_predictions_path": "ml_optimizer",
    "test_predictions_path": "ml_optimizer",
    "optuna_trials_completed": "ml_optimizer",

    # Critic
    "critic_verdict": "red_team_critic",
    "critic_calibration_log": "red_team_critic",

    # Reflection
    "reflection_notes": "self_reflection",
    "dynamic_rules_active": "self_reflection",
    "dynamic_rules_pending": "self_reflection",

    # Pseudo-labels
    "pseudo_label_activated": "pseudo_label",
    "pseudo_label_fraction": "pseudo_label",
    "pseudo_label_cv_delta": "pseudo_label",

    # Ensemble
    "ensemble_method": "ensemble_architect",
    "ensemble_weights": "ensemble_architect",
    "ensemble_cv": "ensemble_architect",
    "ensemble_diversity_report": "ensemble_architect",

    # Post-processing
    "postprocess_config": "post_processor",
    "postprocess_cv_delta": "post_processor",

    # Submission
    "submission_path": "submission_strategist",
    "public_lb_score": "submission_strategist",
    "ewma_score": "submission_strategist",
    "cv_lb_gap": "submission_strategist",
    "submissions_history": "submission_strategist",
    "submission_frozen": "submission_strategist",

    # Submission Safety
    "submission_verified": "submission_safety",
    "final_submission_correlation": "submission_safety",
    "final_submission_diversity_rating": "submission_safety",
    "ewma_freeze_operator_approved": "submission_safety",
    "lb_noise_estimate": "submission_safety",
    "submission_backups": "submission_safety",

    # Provenance
    "code_ledger_path": "publisher",
    "solution_notebook_path": "publisher",
    "solution_writeup_path": "publisher",
    "notebook_reproduction_validated": "publisher",
    "notebook_reproduction_diff": "publisher",

    # HITL
    "hitl_mode": "hitl_listener",
    "hitl_channels": "hitl_listener",
    "hitl_injections": "hitl_listener",
    "hitl_overrides": "hitl_listener",
    "hitl_feature_hints": "hitl_listener",
    "hitl_skip_agents": "hitl_listener",
    "hitl_messages_sent": "hitl_listener",
    "hitl_checkpoint_responses": "hitl_listener",
    "pipeline_paused": "hitl_listener",
    "pipeline_aborted": "hitl_listener",
    "hitl_checkpoint_timeout": "hitl_listener",
    "hitl_gate_timeout": "hitl_listener",

    # Debugging
    "debug_diagnostics": "sandbox",
    "debug_error_class": "sandbox",
    "debug_retry_layer": "sandbox",
    "debug_checkpoints": "sandbox",
    "debug_decomposition": "sandbox",
    "debug_silent_failures": "sandbox",
    "debug_fix_rate_by_class": "sandbox",

    # Cost
    "llm_call_count": "cost_governor",
    "llm_cost_estimate_usd": "cost_governor",
    "llm_budget_calls_max": "cost_governor",
    "llm_budget_usd_max": "cost_governor",
    "llm_calls_per_agent": "cost_governor",
    "llm_budget_exhausted": "cost_governor",

    # Memory
    "memory_retrieval_validated": "chromadb_memory",
    "memory_contradictions_found": "chromadb_memory",
    "memory_quarantined_this_run": "chromadb_memory",

    # Freeform
    "freeform_runs": "freeform handler",
    "freeform_active": "freeform handler",

    # Lineage
    "lineage_log": "graph builder",
    "state_mutations_log": "graph builder",
}

_IMMUTABLE_FIELDS = {
    "canonical_train_rows", "canonical_test_rows", "canonical_schema",
    "canonical_target_stats", "test_data_checksum"
}

_DEFAULT_VALUES = {
    "canonical_train_rows": 0,
    "canonical_test_rows": 0,
    "canonical_schema": {},
    "canonical_target_stats": {},
    "test_data_checksum": ""
}

STATE_SIZE_BUDGET_MB = 20
OWNERSHIP_STRICT = True

# ── Professor State Model ────────────────────────────────────────────────────

class ProfessorState(BaseModel):
    """
    The single state object that flows through the entire LangGraph pipeline.
    Enforces strict typing, ownership, and immutability.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core Pipeline
    session_id: str = ""
    competition_name: str = ""
    competition_url: str = ""
    pipeline_depth: str = "standard"
    pipeline_depth_auto_detected: bool = True
    pipeline_depth_reason: str = ""
    dag_version: int = 0
    replan_target: str = ""
    state_schema_version: str = "v2.0"
    state_size_bytes: int = 0

    # Competition Intel
    competition_description: str = ""
    competition_type: str = ""
    target_column: str = ""
    metric_name: str = ""
    metric_config: Dict = Field(default_factory=dict)
    id_column: str = ""
    submission_format: Dict = Field(default_factory=dict)
    intel_brief: Dict = Field(default_factory=dict)
    forbidden_techniques: List = Field(default_factory=list)

    # Metric Verification
    metric_verified: bool = False
    metric_verification_test: Dict = Field(default_factory=dict)
    metric_verification_method: str = "unverified"

    # Pre-flight
    preflight_passed: bool = False
    preflight_data_files: List = Field(default_factory=list)
    preflight_warnings: List = Field(default_factory=list)
    preflight_data_size_mb: float = 0.0
    preflight_submission_format: Dict = Field(default_factory=dict)
    preflight_target_type: str = ""
    preflight_unsupported_modalities: List = Field(default_factory=list)

    # Data Engineer
    raw_data_path: str = ""
    clean_data_path: str = ""
    clean_test_path: str = ""
    data_schema: Dict = Field(default_factory=dict)
    drop_candidates: List = Field(default_factory=list)

    # Data Integrity [IMMUTABLE]
    canonical_train_rows: int = 0
    canonical_test_rows: int = 0
    canonical_schema: Dict = Field(default_factory=dict)
    canonical_target_stats: Dict = Field(default_factory=dict)
    test_data_checksum: str = ""
    data_integrity_violations: List = Field(default_factory=list)

    # EDA
    eda_report: Dict = Field(default_factory=dict)
    eda_insights_summary: str = ""
    eda_mutual_info: Dict = Field(default_factory=dict)
    eda_vif_report: Dict = Field(default_factory=dict)
    eda_modality_flags: List = Field(default_factory=list)
    eda_plots_paths: List = Field(default_factory=list)
    eda_plots_delivered: bool = False
    eda_quick_baseline_importance: Dict = Field(default_factory=dict)

    # Domain
    domain_brief: Dict = Field(default_factory=dict)
    domain_classification: str = ""
    domain_templates_applied: List = Field(default_factory=list)

    # Shift
    shift_report: Dict = Field(default_factory=dict)
    shift_severity: str = "none"
    shift_sample_weights: List = Field(default_factory=list)

    # Validation
    validation_strategy: Dict = Field(default_factory=dict)
    scorer_name: str = ""
    scorer_func_path: str = ""

    # Reframer
    reframe_applied: str = "none"
    reframe_details: Dict = Field(default_factory=dict)

    # Features
    feature_manifest: List = Field(default_factory=list)
    feature_factory_rounds_completed: int = 0
    features_train_path: str = ""
    features_test_path: str = ""

    # Creative
    creative_features_generated: List = Field(default_factory=list)
    creative_features_accepted: List = Field(default_factory=list)

    # Model
    model_configs: List = Field(default_factory=list)
    best_model_type: str = ""
    best_model_params: Dict = Field(default_factory=dict)
    cv_scores: List = Field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    oof_predictions_path: str = ""
    test_predictions_path: str = ""
    optuna_trials_completed: int = 0

    # Critic
    critic_verdict: Dict = Field(default_factory=dict)
    critic_calibration_log: List = Field(default_factory=list)

    # Reflection
    reflection_notes: List = Field(default_factory=list)
    dynamic_rules_active: List = Field(default_factory=list)
    dynamic_rules_pending: List = Field(default_factory=list)

    # Pseudo-labels
    pseudo_label_activated: bool = False
    pseudo_label_fraction: float = 0.0
    pseudo_label_cv_delta: float = 0.0

    # Ensemble
    ensemble_method: str = ""
    ensemble_weights: List = Field(default_factory=list)
    ensemble_cv: float = 0.0
    ensemble_diversity_report: Dict = Field(default_factory=dict)

    # Post-processing
    postprocess_config: Dict = Field(default_factory=dict)
    postprocess_cv_delta: float = 0.0

    # Submission
    submission_path: str = ""
    public_lb_score: Optional[float] = None
    ewma_score: Optional[float] = None
    cv_lb_gap: Optional[float] = None
    submissions_history: List = Field(default_factory=list)
    submission_frozen: bool = False

    # Submission Safety
    submission_verified: Dict = Field(default_factory=dict)
    final_submission_correlation: float = 0.0
    final_submission_diversity_rating: str = ""
    ewma_freeze_operator_approved: bool = False
    lb_noise_estimate: float = 0.0
    submission_backups: List = Field(default_factory=list)

    # Provenance
    code_ledger_path: str = ""
    solution_notebook_path: str = ""
    solution_writeup_path: str = ""
    notebook_reproduction_validated: bool = False
    notebook_reproduction_diff: int = 0

    # HITL
    hitl_mode: str = "supervised"
    hitl_channels: List = Field(default_factory=lambda: ["cli"])
    hitl_injections: List = Field(default_factory=list)
    hitl_overrides: Dict = Field(default_factory=dict)
    hitl_feature_hints: List = Field(default_factory=list)
    hitl_skip_agents: List = Field(default_factory=list)
    hitl_messages_sent: List = Field(default_factory=list)
    hitl_checkpoint_responses: List = Field(default_factory=list)
    pipeline_paused: bool = False
    pipeline_aborted: bool = False
    hitl_checkpoint_timeout: int = 180
    hitl_gate_timeout: int = 900

    # Debugging
    debug_diagnostics: Dict = Field(default_factory=dict)
    debug_error_class: str = ""
    debug_retry_layer: int = 0
    debug_checkpoints: List = Field(default_factory=list)
    debug_decomposition: List = Field(default_factory=list)
    debug_silent_failures: List = Field(default_factory=list)
    debug_fix_rate_by_class: Dict = Field(default_factory=dict)

    # Cost Governor
    llm_call_count: int = 0
    llm_cost_estimate_usd: float = 0.0
    llm_budget_calls_max: int = 150
    llm_budget_usd_max: float = 5.0
    llm_calls_per_agent: Dict = Field(default_factory=dict)
    llm_budget_exhausted: bool = False

    # Memory
    memory_retrieval_validated: Dict = Field(default_factory=dict)
    memory_contradictions_found: List = Field(default_factory=list)
    memory_quarantined_this_run: List = Field(default_factory=list)

    # Freeform Sandbox
    freeform_runs: List = Field(default_factory=list)
    freeform_active: bool = False

    # Lineage
    lineage_log: List = Field(default_factory=list)
    state_mutations_log: List = Field(default_factory=list)

    # ── Enforcement Logic ─────────────────────────────────────────────────────

    @classmethod
    def validated_update(cls, state: "ProfessorState", agent_name: str, updates: Dict[str, Any]) -> "ProfessorState":
        """
        Agent-safe state update method. Enforces ownership, immutability, and type safety.
        """
        for field, new_value in updates.items():
            # 1. Ownership Check
            owner = _FIELD_OWNERS.get(field)
            if owner and owner != agent_name:
                msg = f"Agent '{agent_name}' cannot write to '{field}' — owned by '{owner}'"
                if OWNERSHIP_STRICT:
                    raise OwnershipError(msg)
                else:
                    print(f"WARNING: {msg}")

            # 2. Immutability Check
            if field in _IMMUTABLE_FIELDS:
                current_val = getattr(state, field)
                default_val = _DEFAULT_VALUES.get(field)
                if current_val != default_val:
                    raise ImmutableFieldError(f"Cannot overwrite [IMMUTABLE] field '{field}' after initial set.")

            # 3. Log Mutation
            old_value = getattr(state, field)
            state._log_mutation(agent_name, field, old_value, new_value)

            # 4. Set Value (Pydantic will validate types on next instantiation if we use model_copy)
            setattr(state, field, new_value)

        # 5. Type Validation (Re-validate the entire model)
        # We convert to dict and back to trigger Pydantic's internal validation
        try:
            validated_state = cls(**state.model_dump())
        except ValidationError as e:
            raise e

        # 6. Size Management
        validated_state._check_size()
        
        return validated_state

    def _log_mutation(self, agent_name: str, field: str, old_value: Any, new_value: Any):
        """Appends a hashed audit trail of the mutation to state_mutations_log."""
        def get_hash(val):
            return hashlib.sha256(str(val).encode()).hexdigest()

        entry = {
            "agent": agent_name,
            "field": field,
            "old_hash": get_hash(old_value),
            "new_hash": get_hash(new_value),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.state_mutations_log.append(entry)

    def _check_size(self):
        """Serializes to JSON and truncates non-core logs if exceeding 20MB."""
        state_json = self.model_dump_json()
        self.state_size_bytes = len(state_json.encode('utf-8'))
        
        budget_bytes = STATE_SIZE_BUDGET_MB * 1024 * 1024
        if self.state_size_bytes > budget_bytes:
            # Truncate oldest entries from specific fields
            self.hitl_messages_sent = self.hitl_messages_sent[-50:]
            self.state_mutations_log = self.state_mutations_log[-100:]
            self.debug_diagnostics = {}
            self.debug_checkpoints = []
            self.lineage_log = self.lineage_log[-200:]
            
            # Update size after truncation
            self.state_size_bytes = len(self.model_dump_json().encode('utf-8'))

    @classmethod
    def validate_checkpoint_version(cls, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Checks schema version and runs migrations if necessary."""
        ver = checkpoint_data.get("state_schema_version", "v1.0")
        
        if ver == "v2.0":
            return checkpoint_data
        
        if ver == "v1.0":
            # Migration: v1.0 -> v2.0
            # v1.0 was likely a loose dict. We'll just let Pydantic handle 
            # missing fields by using defaults from the model.
            checkpoint_data["state_schema_version"] = "v2.0"
            return checkpoint_data
            
        raise SchemaVersionError(f"Unrecognized or incompatible schema version: {ver}")
