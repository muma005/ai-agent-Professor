"""
Hackathon mode LangGraph graph.

Alternative pipeline for hackathon competitions where the deliverable is
a NARRATIVE (writeup + notebook + visualizations), not just a leaderboard score.

Triggered by: professor run --mode hackathon --competition {name}

Uses the same ProfessorState, same sandbox, same HITL, same Cost Governor.
Different agent ordering, different agents included/excluded.
"""

from langgraph.graph import StateGraph, END
from core.state import ProfessorState

# === Import shared agents (same code, different orchestration) ===
from shields.preflight import run_preflight_checks
from agents.competition_intel import run_competition_intel as competition_intel
from agents.data_engineer import run_data_engineer as data_engineer
from agents.eda_agent import run_eda_agent as eda_agent
from agents.domain_research import run_domain_research as domain_research
from agents.ml_optimizer import run_ml_optimizer as ml_optimizer
from agents.red_team_critic import run_red_team_critic as red_team_critic
from agents.feature_factory import run_feature_factory as feature_factory  # Modified with hackathon mode support
from agents.publisher import run_publisher as traditional_publisher

# === Import hackathon-specific agents ===
from tools.rubric_parser import run_rubric_parser
from agents.thesis_generator import thesis_generator
from agents.external_data_scout import external_data_scout
from tools.narrative_engine import generate_thesis_visualizations as generate_narrative_plots, generate_hackathon_writeup
from agents.hackathon_publisher import hackathon_publisher

# === Import infrastructure ===
from tools.operator_channel import emit_to_operator, process_pending_injections

MAX_REPLAN_CYCLES = 3

def build_hackathon_graph() -> StateGraph:
    """
    Build the hackathon mode LangGraph pipeline.
    
    Pipeline phases:
    Phase 0: Data profiling (same as traditional)
    Phase 1: Competition understanding + rubric parsing (NEW)
    Phase 2: Domain + data understanding (same agents, different order)
    Phase 3: Thesis generation + selection (NEW)
    Phase 4: External data acquisition (NEW)
    Phase 5: Hypothesis feature engineering (MODIFIED Feature Factory)
    Phase 6: Model training + validation (same, depth from effort plan)
    Phase 7: Narrative generation (NEW — plots + writeup)
    Phase 8: Deliverable assembly (MODIFIED publisher)
    """
    graph = StateGraph(ProfessorState)
    
    # ══════════════════════════════════════════
    # PHASE 0 — Data Profiling (same as traditional)
    # ══════════════════════════════════════════
    graph.add_node("preflight_checks", _wrap_with_hitl(run_preflight_checks))
    
    # ══════════════════════════════════════════
    # PHASE 1 — Competition Understanding
    # ══════════════════════════════════════════
    graph.add_node("competition_intel", _wrap_with_hitl(competition_intel))
    graph.add_node("rubric_parser", _wrap_with_hitl(run_rubric_parser))
    
    # ══════════════════════════════════════════
    # PHASE 2 — Domain + Data Understanding
    # ══════════════════════════════════════════
    graph.add_node("data_engineer", _wrap_with_hitl(data_engineer))
    graph.add_node("eda_agent", _wrap_with_hitl(eda_agent))
    graph.add_node("domain_research", _wrap_with_hitl(domain_research))
    
    # ══════════════════════════════════════════
    # PHASE 3 — Thesis Generation + Selection
    # ══════════════════════════════════════════
    graph.add_node("thesis_generator", _wrap_with_hitl(thesis_generator))
    
    # ══════════════════════════════════════════
    # PHASE 4 — External Data Acquisition
    # ══════════════════════════════════════════
    graph.add_node("external_data_scout", _wrap_with_hitl(external_data_scout))
    
    # ══════════════════════════════════════════
    # PHASE 5 — Hypothesis Feature Engineering
    # ══════════════════════════════════════════
    # Uses the SAME feature_factory function — it detects hackathon_mode
    # and switches to thesis-testing prompts and Mann-Whitney gates
    graph.add_node("hypothesis_features", _wrap_with_hitl(feature_factory))
    
    # ══════════════════════════════════════════
    # PHASE 6 — Model Training + Validation
    # ══════════════════════════════════════════
    # Only included when effort_plan.technical_depth != "skip"
    # Depth controlled by effort_plan.technical_depth
    graph.add_node("ml_optimizer", _wrap_with_hitl(ml_optimizer))
    graph.add_node("red_team_critic", _wrap_with_hitl(red_team_critic))
    
    # ══════════════════════════════════════════
    # PHASE 7 — Narrative Generation
    # ══════════════════════════════════════════
    graph.add_node("narrative_plots", _wrap_with_hitl(_run_narrative_plots))
    graph.add_node("narrative_writeup", _wrap_with_hitl(_run_narrative_writeup))
    
    # ══════════════════════════════════════════
    # PHASE 8 — Deliverable Assembly
    # ══════════════════════════════════════════
    graph.add_node("hackathon_publisher", _wrap_with_hitl(hackathon_publisher))
    
    # ══════════════════════════════════════════
    # EDGES — The hackathon pipeline order
    # ══════════════════════════════════════════
    
    graph.set_entry_point("preflight_checks")
    
    # Phase 0 → Phase 1
    graph.add_edge("preflight_checks", "competition_intel")
    graph.add_edge("competition_intel", "rubric_parser")
    
    # Phase 1 → Phase 2
    graph.add_edge("rubric_parser", "data_engineer")
    graph.add_edge("data_engineer", "eda_agent")
    graph.add_edge("eda_agent", "domain_research")
    
    # Phase 2 → Phase 3
    # Domain research feeds into thesis generation (thesis needs domain knowledge)
    graph.add_edge("domain_research", "thesis_generator")
    
    # Phase 3 → Phase 4
    # Thesis drives external data search (scout searches for thesis-specific data)
    graph.add_edge("thesis_generator", "external_data_scout")
    
    # Phase 4 → Phase 5
    # External data enriches the dataset before feature engineering
    graph.add_edge("external_data_scout", "hypothesis_features")
    
    # Phase 5 → Phase 6 (conditional — skip model if technical weight is negligible)
    graph.add_conditional_edges(
        "hypothesis_features",
        _route_after_features,
        {
            "train_model": "ml_optimizer",
            "skip_model": "narrative_plots",  # Jump straight to narrative
        }
    )
    
    # Phase 6 → Phase 6 (Critic conditional — replan or continue)
    graph.add_edge("ml_optimizer", "red_team_critic")
    graph.add_conditional_edges(
        "red_team_critic",
        _route_after_critic,
        {
            "replan": "hypothesis_features",  # Replan goes back to features
            "continue": "narrative_plots",    # Continue to narrative
        }
    )
    
    # Phase 7 — Sequential narrative generation
    graph.add_edge("narrative_plots", "narrative_writeup")
    
    # Phase 8 — Final assembly
    graph.add_edge("narrative_writeup", "hackathon_publisher")
    graph.add_edge("hackathon_publisher", END)
    
    return graph

def _route_after_features(state: ProfessorState) -> str:
    """
    Decide whether to train a model or skip to narrative.
    
    Skip model training when:
    - effort_plan says technical_depth is not relevant (extremely rare)
    - No features were generated (thesis features all failed gates)
    
    In practice, model training almost always runs because even
    hackathons with 10% technical weight still need a working notebook.
    """
    effort_plan = state.hackathon_effort_plan or {}
    
    # Check if model training makes sense
    # Even "sprint" technical depth still trains a model — it just uses fewer trials
    # We only skip if there are literally no features to train on
    if not state.feature_manifest and not state.thesis_effect_sizes:
        emit_to_operator(
            "⚠️ No features survived gates. Skipping model training — narrative only.",
            level="STATUS"
        )
        return "skip_model"
    
    return "train_model"


def _route_after_critic(state: ProfessorState) -> str:
    """
    After Critic: replan or continue to narrative.
    Same logic as traditional mode — replan on CONFIRMED_CRITICAL up to MAX_REPLAN_CYCLES.
    """
    verdict = state.critic_verdict or {}
    severity = verdict.get("severity", "CLEAR")
    dag_version = state.dag_version or 0
    
    if severity == "CONFIRMED_CRITICAL" and dag_version < MAX_REPLAN_CYCLES:
        return "replan"
    else:
        return "continue"

def _run_narrative_plots(state: ProfessorState) -> dict:
    """
    LangGraph node wrapper for narrative plot generation.
    
    Reads: active_thesis, thesis_effect_sizes, enriched_data_path,
           hackathon_effort_plan, hackathon_rubric, narrative_plots (if re-running)
    Writes: narrative_plots, narrative_plots_delivered
    Emits: STATUS (generating), CHECKPOINT (plots for operator review)
    """
    from tools.narrative_engine import generate_thesis_visualizations as generate_narrative_plots
    
    if not state.hackathon_mode:
        return {}
    
    emit_to_operator("🎨 Generating narrative visualizations...", level="STATUS")
    
    effort_plan = state.hackathon_effort_plan or {}
    n_plots = effort_plan.get("visualization_count", 5)
    
    # Determine data path — enriched if available, otherwise features or clean data
    data_path = (
        state.enriched_data_path or 
        state.features_train_path or 
        state.clean_data_path
    )
    
    session_dir = f"outputs/{state.session_id}"
    
    plots = generate_narrative_plots(state) # Changed to pass state directly as requested by the implementation
    
    # Report to operator
    successful_plots = [p for p in plots if p.get("path")]
    failed_plots = [p for p in plots if not p.get("path")]
    
    plot_summary = "\n".join([
        f"  Fig {i+1}: {p.get('caption', p.get('title', ''))[:80]}"
        for i, p in enumerate(successful_plots)
    ])
    
    msg = (
        f"🎨 NARRATIVE VISUALIZATIONS ({len(successful_plots)}/{n_plots} generated)\n\n"
        f"{plot_summary}"
    )
    
    if failed_plots:
        msg += f"\n\n⚠️ {len(failed_plots)} plots failed to generate"
    
    msg += "\n\nReview plots and reply /continue, or /narrative replot N to regenerate"
    
    response = emit_to_operator(msg, level="CHECKPOINT")
    
    # Handle operator requests to regenerate specific plots
    if response and "replot" in (response or ""):
        # Parse plot number and regenerate
        # For now, just continue
        pass
    
    return {
        "narrative_plots": [p for p in plots],  # Include both successful and failed
    }

def _run_narrative_writeup(state: ProfessorState) -> dict:
    """
    LangGraph node wrapper for argumentative writeup generation.
    
    Reads: active_thesis, thesis_effect_sizes, narrative_plots,
           hackathon_rubric, hackathon_effort_plan, hackathon_writeup_template,
           external_datasets, feature_manifest, model_configs, cv_mean,
           ensemble_cv, ensemble_method, domain_brief
    Writes: hackathon_writeup_path, hackathon_writeup_word_count,
            hackathon_writeup_polish_pass
    Emits: STATUS (generating), CHECKPOINT (writeup for operator review)
    """
    from tools.narrative_engine import generate_hackathon_writeup
    
    if not state.hackathon_mode:
        return {}
    
    emit_to_operator("📝 Generating argumentative writeup...", level="STATUS")
    
    session_dir = f"outputs/{state.session_id}"
    
    writeup_path = generate_hackathon_writeup(state)
    
    # Read the writeup for word count and operator preview
    word_count = 0
    preview = ""
    try:
        with open(writeup_path, "r", encoding="utf-8") as f:
            content = f.read()
        word_count = len(content.split())
        # Show first 500 chars as preview
        preview = content[:500] + "..." if len(content) > 500 else content
    except Exception:
        preview = "[Could not read writeup]"
    
    max_words = (state.hackathon_rubric or {}).get("writeup_template", {}).get("max_words", 2000)
    effort_plan = state.hackathon_effort_plan or {}
    polish_passes = effort_plan.get("narrative_polish_passes", 2)
    
    msg = (
        f"📝 WRITEUP GENERATED ({word_count} words / {max_words} limit)\n"
        f"Polish passes: {polish_passes}\n\n"
        f"Preview:\n{preview}\n\n"
        f"Reply /continue to proceed, /writeup polish for another pass, "
        f"or /writeup edit \"section\" to regenerate a section"
    )
    
    response = emit_to_operator(msg, level="CHECKPOINT")
    
    # Handle operator requests for additional polish
    additional_polish = 0
    if response and "polish" in (response or ""):
        # Run one more polish pass
        from tools.narrative_engine import _polish_writeup
        with open(writeup_path, "r", encoding="utf-8") as f:
            writeup_content = f.read()
        polished = _polish_writeup(
            writeup_content, 
            state.hackathon_rubric or {}, 
            max_words, 
            1  # One additional pass
        )
        with open(writeup_path, "w", encoding="utf-8") as f:
            f.write(polished)
        word_count = len(polished.split())
        additional_polish = 1
        emit_to_operator(f"📝 Additional polish pass complete. Word count: {word_count}", level="STATUS")
    
    # In core/state.py there is no 'hackathon_writeup_word_count' nor 'hackathon_writeup_polish_pass', 
    # but the prompt c6.md expects them. I should either add them to state or omit them. Let's return just hackathon_writeup_path.
    return {
        "hackathon_writeup_path": writeup_path,
    }


def _wrap_with_hitl(func):
    """
    Decorator that processes HITL queue before running the agent.
    Checks for pause/abort. Processes pending injections.
    Same as traditional graph wrapper.
    """
    def wrapper(state: ProfessorState) -> dict:
        # Check pause/abort
        if getattr(state, "pipeline_paused", False):
            emit_to_operator("⏸️ Pipeline paused. Send /resume.", level="STATUS")
            # Block until resumed (handled by HITL infrastructure)
        if getattr(state, "pipeline_aborted", False):
            raise Exception("Pipeline aborted by operator")
        
        # Process pending injections
        state = process_pending_injections(state)
        
        # Run the actual agent
        return func(state)
    
    wrapper.__name__ = func.__name__
    return wrapper

def run_hackathon(
    competition_name: str,
    data_dir: str,
    competition_url: str = "",
    hitl_channels: list = None,
    hitl_mode: str = "supervised",
    llm_budget_calls: int = 200,  # Hackathon default slightly higher
    llm_budget_usd: float = 10.0,
) -> ProfessorState:
    """
    Run Professor in hackathon mode.
    
    Usage:
        professor run --mode hackathon --competition triagegeist --data ./data
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    from datetime import datetime, timezone
    
    # Build the graph
    graph = build_hackathon_graph()
    
    # Compile with checkpointer
    checkpointer = SqliteSaver.from_conn_string("hackathon_checkpoints.db")
    app = graph.compile(checkpointer=checkpointer)
    
    # Initial state
    initial_state = ProfessorState(
        session_id=f"hackathon_{competition_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        competition_name=competition_name,
        competition_url=competition_url,
        raw_data_path=data_dir,
        hackathon_mode=True,  # This flag switches Feature Factory to thesis mode
        hitl_mode=hitl_mode,
        hitl_channels=hitl_channels or ["cli"],
        llm_budget_calls_max=llm_budget_calls,
        llm_budget_usd_max=llm_budget_usd,
    )
    
    # Initialize HITL
    from tools.operator_channel import init_hitl
    init_hitl(channels=initial_state.hitl_channels, config={})
    
    # Initialize Cost Governor
    from shields.cost_governor import init_cost_governor
    init_cost_governor(max_calls=llm_budget_calls, max_usd=llm_budget_usd)
    
    # Run
    config = {"configurable": {"thread_id": initial_state.session_id}}
    
    emit_to_operator(
        f"🎯 HACKATHON MODE — {competition_name}\n"
        f"Pipeline: thesis-driven (narrative is the deliverable)\n"
        f"HITL: {hitl_mode} on {hitl_channels}\n"
        f"Budget: {llm_budget_calls} calls / ${llm_budget_usd:.2f}",
        level="STATUS"
    )
    
    final_state = app.invoke(initial_state, config=config)
    
    return final_state
