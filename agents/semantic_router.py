# agents/semantic_router.py

from core.state import ProfessorState


# ── v0 linear route ───────────────────────────────────────────────
# Phase 1: fixed sequence, no branching, no LLM involvement.
# Phase 2: replaced with LLM-driven DAG construction.

LINEAR_ROUTE_V0 = [
    "competition_intel",
    "data_engineer",
    "eda_agent",
    "validation_architect",
    "ml_optimizer",
    "submit",
]


def run_semantic_router(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Semantic Router v0.

    Phase 1 behaviour:
      - Sets task_type to "tabular_classification" (hardcoded)
      - Populates state["dag"] with the linear route
      - Sets state["next_node"] to first node in route
      - Never writes code, never touches data files

    Reads:   state["competition_name"], state["task_type"]
    Writes:  state["dag"], state["task_type"], state["next_node"],
             state["current_node"]
    NEVER:   writes code, reads/writes data files, calls external APIs
    """
    competition = state["competition_name"]
    task_type   = state.get("task_type", "auto")

    print(f"[SemanticRouter] Competition: {competition}")

    # ── Task type detection — v0: rule-based, v1: LLM-driven ─────
    if task_type in ("auto", "unknown"):
        task_type = _detect_task_type(competition)

    print(f"[SemanticRouter] Task type: {task_type}")
    print(f"[SemanticRouter] Route: {' -> '.join(LINEAR_ROUTE_V0)}")

    # ── Update competition strategy ───────────────────────────────
    comp_context = dict(state.get("competition_context", {}))
    if comp_context:
        comp_context["strategy"] = _determine_strategy(
            comp_context.get("current_percentile"),
            comp_context.get("days_remaining")
        )
        print(f"[SemanticRouter] Strategy: {comp_context['strategy']}")

    return {
        **state,
        "task_type":           task_type,
        "competition_context": comp_context,
        "dag":                 LINEAR_ROUTE_V0.copy(),
        "current_node":        "semantic_router",
        "next_node":           LINEAR_ROUTE_V0[0],
    }


def _detect_task_type(competition_name: str) -> str:
    """
    Rule-based task type detection for v0.
    Phase 2: replaced with LLM parsing of competition description.
    """
    name = competition_name.lower()

    # Known time-series patterns
    if any(kw in name for kw in ["forecast", "time-series", "timeseries",
                                  "temporal", "sales", "demand", "predict-future"]):
        return "timeseries"

    # Known regression patterns
    if any(kw in name for kw in ["price", "cost", "revenue", "salary",
                                  "amount", "value", "regression", "house"]):
        return "tabular_regression"

    # Default: classification
    return "tabular_classification"


def _determine_strategy(percentile: float, days_remaining: float) -> str:
    """
    Called by router after every submission or at pipeline start.
    Returns the strategy the Supervisor should use for routing.
    """

    if percentile is None or days_remaining is None:
        return "balanced"  # not enough data yet

    if days_remaining <= 2 and percentile <= 0.10:
        return "conservative"   # top 10%, almost done — lock in the rank
    if days_remaining <= 2 and percentile > 0.10:
        return "aggressive"     # running out of time, need big moves
    if days_remaining > 7 and percentile > 0.40:
        return "aggressive"     # plenty of time, far from goal
    return "balanced"
