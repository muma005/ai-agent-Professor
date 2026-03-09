# core/state.py

from typing import TypedDict, Optional, Any, Annotated
import operator
import uuid
from datetime import datetime


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
    task_type: str               # "tabular_classification" | "tabular_regression" | "timeseries" | "auto"
    competition_context: Optional[CompetitionContext]

    # ── Data (pointers only -- never raw DataFrames in state) ─────
    raw_data_path: str
    clean_data_path: Optional[str]
    schema_path: Optional[str]
    eda_report_path: Optional[str]
    data_hash: Optional[str]     # SHA-256 of source file, first 16 chars

    # ── Feature Engineering ───────────────────────────────────────
    # REPLACE: feature factory sets the full list each run
    feature_manifest: Annotated[Optional[list], _replace]
    feature_factory_checkpoint: Optional[dict]

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

    # ── Critic ────────────────────────────────────────────────────
    critic_verdict: Optional[dict]

    # ── Ensemble ──────────────────────────────────────────────────
    ensemble_weights: Optional[dict]
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

    # ── Budget ────────────────────────────────────────────────────
    cost_tracker: CostTracker

    # ── Output ────────────────────────────────────────────────────
    report_path: Optional[str]
    lineage_log_path: Optional[str]


def initial_state(
    competition: str,
    data_path: str,
    budget_usd: float = 2.00,
    task_type: str = "auto"
) -> ProfessorState:
    """Create a fresh state for a new competition run."""

    session_id = f"{competition[:8].replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

    return ProfessorState(
        session_id=session_id,
        created_at=datetime.utcnow().isoformat(),
        competition_name=competition,
        task_type=task_type,
        competition_context=None,
        raw_data_path=data_path,
        clean_data_path=None,
        schema_path=None,
        eda_report_path=None,
        data_hash=None,
        feature_manifest=None,
        feature_factory_checkpoint=None,
        cv_strategy=None,
        metric_contract=None,
        cv_scores=None,
        cv_mean=None,
        model_registry=[],
        best_params=None,
        optuna_study_path=None,
        critic_verdict=None,
        ensemble_weights=None,
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
        lineage_log_path=f"outputs/logs/{session_id}.jsonl"
    )
