# core/state.py

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

# ── Config Flag ──────────────────────────────────────────────────────────────
OWNERSHIP_STRICT = True

# ── Field Ownership Map ──────────────────────────────────────────────────────
# Every field from the existing state.py mapped to its owning agent.
_FIELD_OWNERS = {
    # Core Pipeline & Supervisor
    "session_id": "supervisor",
    "created_at": "supervisor",
    "competition_name": "supervisor",
    "dag": "supervisor",
    "current_node": "supervisor",
    "next_node": "supervisor",
    "error_count": "supervisor",
    "escalation_level": "supervisor",
    "dag_version": "supervisor",
    "pipeline_halted": "cost_governor",
    "pipeline_halt_reason": "cost_governor",
    "state_schema_version": "supervisor",
    "state_size_bytes": "supervisor",

    # Competition Intel
    "competition_brief": "competition_intel",
    "competition_brief_path": "competition_intel",
    "intel_brief_path": "competition_intel",
    "competition_context": "competition_intel",
    "task_type": "competition_intel",

    # Pre-flight (Shield 6)
    "preflight_passed": "preflight",
    "preflight_warnings": "preflight",

    # Data Engineer
    "raw_data_path": "data_engineer",
    "test_data_path": "data_engineer",
    "sample_submission_path": "data_engineer",
    "clean_data_path": "data_engineer",
    "data_hash": "data_engineer",
    "target_col": "data_engineer",
    "id_columns": "data_engineer",
    "schema_path": "data_engineer",
    "preprocessor_path": "data_engineer",
    "preprocessor_config_path": "data_engineer",
    "canonical_train_rows": "data_engineer",
    "canonical_test_rows": "data_engineer",
    "canonical_schema": "data_engineer",
    "canonical_target_stats": "data_engineer",
    "test_data_checksum": "data_engineer",

    # EDA Agent
    "eda_report": "eda_agent",
    "eda_report_path": "eda_agent",
    "dropped_features": "eda_agent",

    # Feature Factory
    "feature_manifest": "feature_factory",
    "feature_candidates": "feature_factory",
    "round1_features": "feature_factory",
    "round2_features": "feature_factory",
    "round3_features": "feature_factory",
    "round4_features": "feature_factory",
    "round5_features": "feature_factory",
    "feature_factory_checkpoint": "feature_factory",
    "feature_order": "feature_factory",
    "feature_data_path": "feature_factory",
    "features_dropped_stage1": "feature_factory",
    "features_dropped_stage2": "feature_factory",
    "features_gate_passed": "feature_factory",
    "features_gate_dropped": "feature_factory",

    # Validation Architect
    "cv_strategy": "validation_architect",
    "metric_contract": "validation_architect",

    # ML Optimizer
    "cv_scores": "ml_optimizer",
    "cv_mean": "ml_optimizer",
    "model_registry": "ml_optimizer",
    "best_params": "ml_optimizer",
    "optuna_study_path": "ml_optimizer",
    "optuna_pruned_trials": "ml_optimizer",
    "oof_predictions_path": "ml_optimizer",
    "test_predictions_path": "ml_optimizer",
    "feature_data_path_test": "ml_optimizer",
    "memory_peak_gb": "ml_optimizer",
    "memory_oom_risk": "ml_optimizer",

    # Red Team Critic
    "critic_verdict": "red_team_critic",
    "critic_verdict_path": "red_team_critic",
    "critic_severity": "red_team_critic",
    "replan_requested": "red_team_critic",
    "replan_remove_features": "red_team_critic",
    "replan_rerun_nodes": "red_team_critic",
    "competition_fingerprint": "red_team_critic",
    "warm_start_priors": "red_team_critic",

    # Problem Reframer
    "reframe_details": "problem_reframer",

    # Supervisor Replan
    "features_dropped": "supervisor",

    # Post-Mortem Agent
    "post_mortem_completed": "post_mortem_agent",
    "post_mortem_report_path": "post_mortem_agent",
    "lb_score": "post_mortem_agent",
    "lb_rank": "post_mortem_agent",
    "cv_lb_gap": "post_mortem_agent",
    "gap_root_cause": "post_mortem_agent",

    # Ensemble Architect
    "ensemble_selection": "ensemble_architect",
    "selected_models": "ensemble_architect",
    "ensemble_weights": "ensemble_architect",
    "ensemble_oof": "ensemble_architect",
    "prize_candidates": "ensemble_architect",

    # Submission Strategist
    "submission_path": "submission_strategist",
    "submission_log": "submission_strategist",

    # Circuit Breaker & Cost Governor
    "current_node_failure_count": "cost_governor",
    "error_context": "cost_governor",
    "macro_replan_requested": "cost_governor",
    "macro_replan_reason": "cost_governor",
    "triage_mode": "cost_governor",
    "budget_remaining_usd": "cost_governor",
    "budget_limit_usd": "cost_governor",
    "cost_tracker": "cost_governor",

    # HITL Listener
    "hitl_required": "hitl_listener",
    "hitl_prompt": "hitl_listener",
    "hitl_checkpoint_key": "hitl_listener",
    "hitl_intervention_id": "hitl_listener",
    "hitl_intervention_label": "hitl_listener",
    "skip_data_validation": "hitl_listener",
    "null_threshold": "hitl_listener",
    "impute_strategy": "hitl_listener",
    "lgbm_override": "hitl_listener",
    "model_fallback": "hitl_listener",
    "data_sample_fraction": "hitl_listener",
    "api_timeout_multiplier": "hitl_listener",
    "api_backoff_enabled": "hitl_listener",
    "llm_provider": "hitl_listener",
    "debug_logging": "hitl_listener",
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

    # Sandbox
    "debug_diagnostics": "sandbox",
    "debug_error_class": "sandbox",
    "debug_retry_layer": "sandbox",
    "debug_checkpoints": "sandbox",
    "debug_decomposition": "sandbox",
    "debug_silent_failures": "sandbox",
    "debug_fix_rate_by_class": "sandbox",

    # External Data Scout
    "external_data_allowed": "data_scout",
    "external_data_manifest": "data_scout",

    # Lineage & Output
    "performance_log": "graph builder",
    "lineage_log": "graph builder",
    "state_mutations_log": "graph builder",
    "report_path": "publisher",
    "lineage_log_path": "publisher",

    # Config
    "config": "supervisor",
}

_IMMUTABLE_FIELDS = {
    "canonical_train_rows", "canonical_test_rows", "canonical_schema",
    "canonical_target_stats", "test_data_checksum"
}

STATE_SIZE_BUDGET_MB = 20

# ── Professor State Model ────────────────────────────────────────────────────

class ProfessorState(BaseModel):
    """
    The strictly-typed, validated state for the Professor pipeline.
    Replaces loose dictionaries and enforces ownership/immutability.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core Pipeline
    session_id: str = ""
    created_at: str = ""
    competition_name: str = ""
    dag: Optional[List] = Field(default_factory=list)
    current_node: Optional[str] = None
    next_node: Optional[str] = None
    error_count: int = 0
    escalation_level: str = "micro"
    dag_version: int = 1
    pipeline_halted: bool = False
    pipeline_halt_reason: str = ""
    state_schema_version: str = "v2.0"
    state_size_bytes: int = 0

    # Competition Intel
    competition_brief: Dict = Field(default_factory=dict)
    competition_brief_path: str = ""
    intel_brief_path: str = ""
    competition_context: Dict = Field(default_factory=dict)
    task_type: str = "unknown"

    # Pre-flight
    preflight_passed: bool = False
    preflight_warnings: List = Field(default_factory=list)

    # Data Engineer
    raw_data_path: str = ""
    test_data_path: str = ""
    sample_submission_path: str = ""
    clean_data_path: str = ""
    data_hash: str = ""
    target_col: str = ""
    id_columns: List = Field(default_factory=list)
    schema_path: Optional[str] = None
    preprocessor_path: Optional[str] = None
    preprocessor_config_path: Optional[str] = None

    # [IMMUTABLE] Data Integrity
    canonical_train_rows: int = 0
    canonical_test_rows: int = 0
    canonical_schema: Dict = Field(default_factory=dict)
    canonical_target_stats: Dict = Field(default_factory=dict)
    test_data_checksum: str = ""

    # EDA
    eda_report: Dict = Field(default_factory=dict)
    eda_report_path: str = ""
    eda_insights_summary: str = ""
    dropped_features: List = Field(default_factory=list)

    # Feature Factory
    feature_manifest: Optional[Dict] = None
    feature_candidates: Optional[List] = None
    round1_features: Optional[List] = None
    round2_features: Optional[List] = None
    round3_features: Optional[List] = None
    round4_features: Optional[List] = None
    round5_features: Optional[List] = None
    feature_factory_checkpoint: Optional[Dict] = None
    feature_order: List = Field(default_factory=list)
    feature_data_path: Optional[str] = None
    features_dropped_stage1: Optional[List] = None
    features_dropped_stage2: Optional[List] = None
    features_gate_passed: Optional[List] = None
    features_gate_dropped: Optional[List] = None

    # Validation
    cv_strategy: Optional[Dict] = None
    metric_contract: Optional[Dict] = None

    # Model
    cv_scores: Optional[List] = None
    cv_mean: Optional[float] = None
    model_registry: List = Field(default_factory=list)
    best_params: Optional[Dict] = None
    optuna_study_path: Optional[str] = None
    optuna_pruned_trials: int = 0
    oof_predictions_path: Optional[str] = None
    test_predictions_path: Optional[str] = None
    feature_data_path_test: Optional[str] = None
    memory_peak_gb: float = 0.0
    memory_oom_risk: bool = False

    # Critic
    critic_verdict: Optional[Dict] = None
    critic_verdict_path: str = ""
    critic_severity: str = "unchecked"
    replan_requested: bool = False
    replan_remove_features: List = Field(default_factory=list)
    replan_rerun_nodes: List = Field(default_factory=list)
    competition_fingerprint: Dict = Field(default_factory=dict)
    warm_start_priors: List = Field(default_factory=list)

    # Reframer
    reframe_details: Dict = Field(default_factory=dict)

    # Supervisor
    features_dropped: List = Field(default_factory=list)

    # Post-Mortem
    post_mortem_completed: bool = False
    post_mortem_report_path: str = ""
    lb_score: Optional[float] = None
    lb_rank: Optional[int] = None
    cv_lb_gap: Optional[float] = None
    gap_root_cause: str = ""

    # Ensemble
    ensemble_selection: Optional[Dict] = None
    selected_models: Optional[List] = None
    ensemble_weights: Optional[Dict] = None
    ensemble_oof: Optional[List] = None
    prize_candidates: Optional[List] = None

    # Submission
    submission_path: Optional[str] = None
    submission_log: List = Field(default_factory=list)

    # Cost & Circuit Breaker
    current_node_failure_count: int = 0
    error_context: List = Field(default_factory=list)
    macro_replan_requested: bool = False
    macro_replan_reason: str = ""
    triage_mode: bool = False
    budget_remaining_usd: float = 2.0
    budget_limit_usd: float = 2.0
    cost_tracker: Dict = Field(default_factory=dict)

    # HITL
    hitl_required: bool = False
    hitl_prompt: Dict = Field(default_factory=dict)
    hitl_checkpoint_key: str = ""
    hitl_intervention_id: int = 0
    hitl_intervention_label: str = ""
    skip_data_validation: bool = False
    null_threshold: float = 1.0
    impute_strategy: str = "default"
    lgbm_override: Dict = Field(default_factory=dict)
    model_fallback: str = ""
    data_sample_fraction: float = 1.0
    api_timeout_multiplier: float = 1.0
    api_backoff_enabled: bool = False
    llm_provider: str = "groq"
    debug_logging: bool = False
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

    # Sandbox
    debug_diagnostics: Dict = Field(default_factory=dict)
    debug_error_class: str = ""
    debug_retry_layer: int = 0
    debug_checkpoints: List = Field(default_factory=list)
    debug_decomposition: List = Field(default_factory=list)
    debug_silent_failures: List = Field(default_factory=list)
    debug_fix_rate_by_class: Dict = Field(default_factory=dict)

    # External Data
    external_data_allowed: bool = False
    external_data_manifest: Dict = Field(default_factory=dict)

    # Lineage
    performance_log: List = Field(default_factory=list)
    lineage_log: List = Field(default_factory=list)
    state_mutations_log: List = Field(default_factory=list)
    report_path: Optional[str] = None
    lineage_log_path: Optional[str] = None

    # Config
    config: Any = None

    # ── Legacy Mapping Protocol ──────────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any):
        # Direct assignment for legacy compatibility
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.model_dump().keys()

    def items(self):
        return self.model_dump().items()

    def __iter__(self):
        return iter(self.keys())

    # ── Validation & Enforcement ─────────────────────────────────────────────

    @classmethod
    def validated_update(cls, state: Union["ProfessorState", Dict[str, Any]], agent_name: str, updates: Dict[str, Any]) -> "ProfessorState":
        """
        Agent-safe state update. Enforces ownership, immutability, and types.
        Handles both ProfessorState objects and legacy dictionaries.
        """
        # 1. Normalize to dictionary for processing
        if isinstance(state, cls):
            new_data = state.model_dump()
            current_state_obj = state
        else:
            new_data = dict(state)
            current_state_obj = cls(**new_data)
        
        for field, new_value in updates.items():
            # 2. Ownership
            owner = _FIELD_OWNERS.get(field)
            if owner and owner != agent_name:
                msg = f"Agent '{agent_name}' cannot write to '{field}' — owned by '{owner}'"
                if OWNERSHIP_STRICT:
                    raise OwnershipError(msg)
                else:
                    print(f"WARNING: {msg}")

            # 3. Immutability
            if field in _IMMUTABLE_FIELDS:
                current_val = getattr(current_state_obj, field)
                # Defaults: 0 for int, {} for dict, "" for str
                if (isinstance(current_val, int) and current_val != 0) or \
                   (isinstance(current_val, dict) and current_val) or \
                   (isinstance(current_val, str) and current_val):
                    raise ImmutableFieldError(f"Cannot overwrite [IMMUTABLE] field '{field}' after initial set.")

            # 4. Audit Log
            current_state_obj._log_mutation(agent_name, field, getattr(current_state_obj, field), new_value)
            new_data[field] = new_value

        # 5. Ensure mutations log is carried over
        new_data["state_mutations_log"] = current_state_obj.state_mutations_log

        # 6. Re-validate & Return as Object
        new_state = cls(**new_data)
        new_state._check_size()
        return new_state

    def _log_mutation(self, agent_name: str, field: str, old_value: Any, new_value: Any):
        def get_hash(v):
            return hashlib.sha256(str(v).encode()).hexdigest()
        
        self.state_mutations_log.append({
            "agent": agent_name,
            "field": field,
            "old_hash": get_hash(old_value),
            "new_hash": get_hash(new_value),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def _check_size(self):
        serialized = self.model_dump_json()
        self.state_size_bytes = len(serialized.encode("utf-8"))
        
        budget = STATE_SIZE_BUDGET_MB * 1024 * 1024
        if self.state_size_bytes > budget:
            self.hitl_messages_sent = self.hitl_messages_sent[-50:]
            self.state_mutations_log = self.state_mutations_log[-100:]
            self.debug_diagnostics = {}
            self.debug_checkpoints = []
            self.lineage_log = self.lineage_log[-200:]
            
            # Recalculate size
            self.state_size_bytes = len(self.model_dump_json().encode("utf-8"))

    @classmethod
    def validate_checkpoint_version(cls, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        ver = checkpoint_data.get("state_schema_version", "v1.0")
        if ver == "v2.0":
            return checkpoint_data
        if ver == "v1.0":
            # Migration logic
            checkpoint_data["state_schema_version"] = "v2.0"
            return checkpoint_data
        raise SchemaVersionError(f"Unrecognized schema version: {ver}")

# ── Initial State (Legacy Compatibility) ───────────────────────────────────
def initial_state(session_id: str = "", raw_data_path: str = "", competition_name: str = "") -> Dict[str, Any]:
    """Callable initializer for legacy v1 compatibility."""
    state = ProfessorState(
        session_id=session_id,
        raw_data_path=raw_data_path,
        competition_name=competition_name
    )
    return state.model_dump()
