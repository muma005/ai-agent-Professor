# core/state.py

from typing import TypedDict, Optional, Any, Annotated, Literal
import operator
import uuid
import hashlib
from datetime import datetime, timezone


# ── Custom reducers for LangGraph state channels ─────────────────
# LangGraph uses reducers to merge state between nodes.
# Default for list = operator.add (APPEND). This corrupts fields
# that should be REPLACED each run (dag, cv_scores).
# We define explicit reducers for every list field.

def _replace(existing, new):
    """Replace reducer: last writer wins. Used for fields that are
    set fresh each run (dag, cv_scores, feature_manifest)."""
    return new


class CostTracker(TypedDict):
    total_usd: float
    groq_tokens_in: int
    groq_tokens_out: int
    gemini_tokens: int
    llm_calls: int
    budget_usd: float
    warning_threshold: float   # 0.70 -- warn at 70% budget
    throttle_threshold: float  # 0.85 -- throttle at 85%
    triage_threshold: float    # 0.95 -- HITL at 95%


class CompetitionContext(TypedDict):
    competition_name: str
    days_remaining: Optional[int]
    current_rank: Optional[int]
    total_teams: Optional[int]
    submission_count: int
    submission_limit: int
    public_lb_score: Optional[float]
    best_cv_score: Optional[float]
    lb_cv_gap: Optional[float]
    shakeup_risk: Optional[str]   # "low" | "medium" | "high"


class ProfessorState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────
    session_id: str              # namespaces ALL resources for this run
    created_at: str

    # ── Competition ───────────────────────────────────────────────
    competition_name: str
    task_type: Literal["tabular", "timeseries", "nlp", "image", "unknown"]
    competition_context: dict

    # ── Intel ─────────────────────────────────────────────────────
    competition_brief_path: str
    competition_brief: dict
    intel_brief_path: str

    # ── Data (pointers only -- never raw DataFrames in state) ─────
    raw_data_path: str
    test_data_path: str           # path to test.csv (set by data_engineer)
    sample_submission_path: str   # path to sample_submission.csv
    clean_data_path: str
    eda_report_path: str
    eda_report: dict
    schema_path: Optional[str]
    preprocessor_path: Optional[str]
    preprocessor_config_path: Optional[str]  # LEAKAGE FIX: For CV-safe preprocessor reconstruction
    data_hash: str     # SHA-256 of source file, first 16 chars

    # ── Schema Authority (Phase 0) — single source of truth ───────
    target_col: str               # authoritative target column name
    id_columns: list              # authoritative ID column names
    dropped_features: list        # features to exclude (set by EDA, consumed by all)

    # ── Feature Engineering (Day 16: feature factory) ─────────────
    # REPLACE: feature factory sets the full manifest each run
    feature_manifest: Annotated[Optional[dict], _replace]
    feature_candidates: Annotated[Optional[list], _replace]  # names of kept features
    round1_features: Annotated[Optional[list], _replace]     # Round 1 generic feature names
    round2_features: Annotated[Optional[list], _replace]     # Round 2 domain feature names
    feature_factory_checkpoint: Optional[dict]
    feature_order: Annotated[Optional[list], _replace]   # exact column order at training time
    feature_data_path: Optional[str]  # path to feature matrix parquet (set by feature_factory)
    feature_data_path_test: Optional[str]  # path to test feature matrix (set by ml_optimizer)

    # ── Validation ────────────────────────────────────────────────
    cv_strategy: Optional[dict]
    metric_contract: Optional[dict]
    # REPLACE: optimizer sets current run's fold scores
    cv_scores: Annotated[Optional[list], _replace]
    cv_mean: Optional[float]

    # ── Models ────────────────────────────────────────────────────
    # REPLACE: managed manually by the node before returning
    model_registry: Annotated[Optional[list], _replace]
    best_params: Optional[dict]
    optuna_study_path: Optional[str]

    # -- Critic (Day 10) -------------------------------------------
    critic_verdict: Optional[dict]
    critic_verdict_path: str
    critic_severity: str              # CRITICAL | HIGH | MEDIUM | OK | unchecked
    replan_requested: bool
    replan_remove_features: list      # features the critic wants dropped
    replan_rerun_nodes: list          # nodes to re-run after replan
    competition_fingerprint: dict     # built from EDA + schema
    warm_start_priors: list           # retrieved from memory

    # -- Supervisor Replan (Day 11) --------------------------------
    features_dropped: list            # accumulated across all replan cycles

    # -- Post-Mortem (Day 11) --------------------------------------
    post_mortem_completed: bool
    post_mortem_report_path: str
    lb_score: Optional[float]
    lb_rank: Optional[int]
    cv_lb_gap: Optional[float]
    gap_root_cause: str

    # ── Ensemble (Day 16: diversity selection) ────────────────────
    ensemble_selection: Optional[dict]         # full result from select_diverse_ensemble()
    selected_models: Annotated[Optional[list], _replace]  # names of selected models
    ensemble_weights: Optional[dict]
    ensemble_oof: Annotated[Optional[list], _replace]     # blended OOF predictions
    prize_candidates: Annotated[Optional[list], _replace] # low-corr + competitive CV
    oof_predictions_path: Optional[str]
    test_predictions_path: Optional[str]

    # ── Submission ────────────────────────────────────────────────
    submission_path: Optional[str]
    # REPLACE: managed manually by the node before returning
    submission_log: Annotated[Optional[list], _replace]

    # ── Routing ───────────────────────────────────────────────────
    # REPLACE: router sets the full route, optimizer never appends to it
    dag: Annotated[Optional[list], _replace]
    current_node: Optional[str]
    next_node: Optional[str]
    error_count: int
    escalation_level: str        # "micro" | "macro" | "hitl" | "triage"

    # -- Circuit Breaker (Day 9) -----------------------------------
    current_node_failure_count: int
    error_context: list           # [{agent, attempt, error, traceback}]
    dag_version: int
    macro_replan_requested: bool
    macro_replan_reason: str
    pipeline_halted: bool
    triage_mode: bool
    budget_remaining_usd: float
    budget_limit_usd: float

    # -- Parallel Execution (Day 9) --------------------------------
    parallel_groups: dict   # {group_name: {status, members}}

    # ── Budget ────────────────────────────────────────────────────
    cost_tracker: CostTracker

    # -- HITL Human Layer & Interventions (Day 12) -----------------
    hitl_required: bool
    hitl_prompt: dict
    hitl_checkpoint_key: str
    hitl_intervention_id: int
    hitl_intervention_label: str
    skip_data_validation: bool
    null_threshold: float
    impute_strategy: str
    lgbm_override: dict
    model_fallback: str
    data_sample_fraction: float
    api_timeout_multiplier: float
    api_backoff_enabled: bool
    llm_provider: str
    debug_logging: bool

    # -- Memory Monitoring (Day 12) --------------------------------
    memory_peak_gb: float
    memory_oom_risk: bool
    optuna_pruned_trials: int

    # -- Feature Filtering (Day 17) ---------------------------------
    null_importance_result_path: Optional[str]     # path to saved NullImportanceResult (disk, not state)
    features_dropped_stage1: Annotated[Optional[list], _replace]
    features_dropped_stage2: Annotated[Optional[list], _replace]
    features_gate_passed: Annotated[Optional[list], _replace]
    features_gate_dropped: Annotated[Optional[list], _replace]

    # -- Feature Factory Rounds 3-5 (Day 18) -----------------------
    round3_features: Annotated[Optional[list], _replace]
    round4_features: Annotated[Optional[list], _replace]
    round5_features: Annotated[Optional[list], _replace]

    # -- Pseudo-Labeling (Day 18) ----------------------------------
    pseudo_label_data_path: Optional[str]          # path to pseudo-labeled data (disk, not state)
    pseudo_labels_applied: bool
    pseudo_label_cv_improvement: float

    # -- Performance Monitoring (FLAW-6.1) -------------------------
    performance_log: Annotated[Optional[list], _replace]  # timing info per node

    # -- External Data Scout (Day 15) ------------------------------
    external_data_allowed: bool
    external_data_manifest: dict

    # -- Configuration (Phase 2) -----------------------------------
    config: Optional["ProfessorConfig"]

    # ── Output ────────────────────────────────────────────────────
    report_path: Optional[str]
    lineage_log_path: Optional[str]


# =========================================================================
# GAP 8: Session ID namespace isolation
# =========================================================================

def generate_session_id(competition_name: str) -> str:
    """
    Generates a unique, namespaced session ID.
    Format: professor_{competition_slug}_{timestamp}_{short_hash}
    Example: professor_spaceship-titanic_20260301_142200_a3f9c2

    The short hash includes microseconds and uuid4 to ensure uniqueness
    even if two runs start in the same second.
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    slug = competition_name.lower().replace(" ", "-").replace("_", "-")[:30]
    unique_seed = f"{slug}{timestamp}{now.microsecond}{uuid.uuid4().hex}"
    short_hash = hashlib.md5(unique_seed.encode()).hexdigest()[:6]
    return f"professor_{slug}_{timestamp}_{short_hash}"


def build_initial_state(competition_name: str, **kwargs) -> dict:
    """
    Factory function for building a valid initial ProfessorState.
    Always use this instead of building state dicts manually.
    """
    session_id = generate_session_id(competition_name)
    return {
        "competition_name": competition_name,
        "session_id": session_id,
        "output_dir": f"outputs/{session_id}",
        "budget_session_id": session_id,
        "budget_limit_usd": 10.0,
        "budget_spent_usd": 0.0,
        "dag_version": 1,
        "current_node_failure_count": 0,
        "hitl_required": False,
        "replan_requested": False,
        "critic_severity": "unchecked",
        "model_registry": {},
        "features_dropped": [],
        "feature_order": [],
        "external_data_allowed": False,
        **kwargs,
    }


def initial_state(
    competition: str,
    data_path: str,
    budget_usd: float = 2.00,
    task_type: str = "unknown",
    config: Optional["ProfessorConfig"] = None,
) -> ProfessorState:
    """
    Create a fresh state for a new competition run.
    
    Args:
        competition: Competition name (e.g., "spaceship-titanic")
        data_path: Path to training data
        budget_usd: API budget in USD
        task_type: Task type ("binary", "multiclass", "regression", etc.)
        config: ProfessorConfig instance. If None, loads from environment.
    
    Returns:
        ProfessorState initialized with default values
    """
    # Import here to avoid circular dependency
    from core.config import ProfessorConfig
    
    # Load config from parameter or environment
    if config is None:
        config = ProfessorConfig.from_env()
    
    # Apply config to environment (ensures all components see it)
    config.apply_env()
    
    session_id = f"{competition[:8].replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

    return ProfessorState(
        session_id=session_id,
        created_at=datetime.utcnow().isoformat(),
        competition_name=competition,
        task_type=task_type,
        competition_context={
            "days_remaining":        None,
            "hours_remaining":       None,
            "submissions_used":      0,
            "submissions_remaining": None,
            "current_public_rank":   None,
            "total_competitors":     None,
            "current_percentile":    None,
            "shakeup_risk":          "unknown",
            "strategy":              "balanced",
            "last_updated":          None,
        },
        raw_data_path=data_path,
        test_data_path="",
        sample_submission_path="",
        clean_data_path="",
        eda_report_path="",
        eda_report={},
        schema_path=None,
        preprocessor_path=None,
        preprocessor_config_path=None,
        data_hash="",
        # -- Schema Authority (Phase 0) --
        target_col="",
        id_columns=[],
        dropped_features=[],
        feature_manifest=None,
        feature_candidates=None,
        round1_features=None,
        round2_features=None,
        feature_factory_checkpoint=None,
        feature_order=[],
        feature_data_path=None,
        feature_data_path_test=None,
        cv_strategy=None,
        metric_contract=None,
        cv_scores=None,
        cv_mean=None,
        model_registry=[],
        best_params=None,
        optuna_study_path=None,
        critic_verdict=None,
        critic_verdict_path="",
        critic_severity="unchecked",
        replan_requested=False,
        replan_remove_features=[],
        replan_rerun_nodes=[],
        competition_fingerprint={},
        warm_start_priors=[],
        # -- Supervisor Replan (Day 11) --
        features_dropped=[],
        # -- Post-Mortem (Day 11) --
        post_mortem_completed=False,
        post_mortem_report_path="",
        lb_score=None,
        lb_rank=None,
        cv_lb_gap=None,
        gap_root_cause="",
        ensemble_selection=None,
        selected_models=None,
        ensemble_weights=None,
        ensemble_oof=None,
        prize_candidates=None,
        oof_predictions_path=None,
        test_predictions_path=None,
        submission_path=None,
        submission_log=[],
        dag=None,
        current_node=None,
        next_node=None,
        error_count=0,
        escalation_level="micro",
        cost_tracker=CostTracker(
            total_usd=0.0,
            groq_tokens_in=0,
            groq_tokens_out=0,
            gemini_tokens=0,
            llm_calls=0,
            budget_usd=budget_usd,
            warning_threshold=0.70,
            throttle_threshold=0.85,
            triage_threshold=0.95
        ),
        report_path=None,
        lineage_log_path=f"outputs/logs/{session_id}.jsonl",
        # -- Circuit Breaker (Day 9) --
        current_node_failure_count=0,
        error_context=[],
        dag_version=1,
        macro_replan_requested=False,
        macro_replan_reason="",
        pipeline_halted=False,
        triage_mode=False,
        budget_remaining_usd=budget_usd,
        budget_limit_usd=budget_usd,
        # -- Parallel Execution (Day 9) --
        parallel_groups={
            "intelligence": {"status": "pending", "members": ["competition_intel", "data_engineer"]},
            "model_trials": {"status": "pending", "members": ["lgbm", "xgb", "catboost"]},
            "critic":       {"status": "pending", "members": ["vector_1", "vector_2", "vector_3", "vector_4"]},
        },
        # -- HITL Human Layer & Interventions (Day 12) --
        hitl_required=False,
        hitl_prompt={},
        hitl_checkpoint_key="",
        hitl_intervention_id=0,
        hitl_intervention_label="",
        skip_data_validation=False,
        null_threshold=1.0,
        impute_strategy="default",
        lgbm_override={},
        model_fallback="",
        data_sample_fraction=1.0,
        api_timeout_multiplier=1.0,
        api_backoff_enabled=False,
        llm_provider="groq",
        debug_logging=False,
        # -- Memory Monitoring (Day 12) --
        memory_peak_gb=0.0,
        memory_oom_risk=False,
        optuna_pruned_trials=0,
        # -- Feature Filtering (Day 17) --
        null_importance_result_path=None,
        features_dropped_stage1=None,
        features_dropped_stage2=None,
        features_gate_passed=None,
        features_gate_dropped=None,
        # -- Feature Factory Rounds 3-5 (Day 18) --
        round3_features=None,
        round4_features=None,
        round5_features=None,
        # -- Pseudo-Labeling (Day 18) --
        pseudo_label_data_path=None,
        pseudo_labels_applied=False,
        pseudo_label_cv_improvement=0.0,
        # -- External Data Scout (Day 15) --
        external_data_allowed=False,
        external_data_manifest={},
        # -- Configuration (Phase 2) --
        config=config,
    )
