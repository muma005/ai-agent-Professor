"""
Professor CLI — the actual command that starts everything.

Usage:
    professor run --competition spaceship-titanic --data ./data
    professor run --competition triagegeist --mode hackathon --data ./data
    professor run --competition titanic --depth sprint --hitl cli
    professor run --competition titanic --hitl telegram --budget-calls 300
    professor resume --session session_20260501_083000
    professor status --session session_20260501_083000
"""

import argparse
import sys
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Professor — Autonomous Kaggle Competition System")
    subparsers = parser.add_subparsers(dest="command")
    
    # === RUN command ===
    run_parser = subparsers.add_parser("run", help="Start a new competition run")
    run_parser.add_argument("--competition", required=True, help="Competition name or URL")
    run_parser.add_argument("--data", required=True, help="Path to competition data directory")
    run_parser.add_argument("--mode", choices=["traditional", "hackathon"], default="traditional")
    run_parser.add_argument("--depth", choices=["sprint", "standard", "marathon"], default=None,
                           help="Override auto-detected pipeline depth")
    run_parser.add_argument("--hitl", default="cli", help="HITL channels: cli, telegram, cli,telegram, none")
    run_parser.add_argument("--hitl-mode", choices=["autonomous", "supervised", "guided"], default="supervised")
    run_parser.add_argument("--budget-calls", type=int, default=150, help="Max LLM calls")
    run_parser.add_argument("--budget-usd", type=float, default=5.0, help="Max LLM spend in USD")
    run_parser.add_argument("--provider", default=None, help="Override LLM provider: groq, google, anthropic")
    run_parser.add_argument("--session-dir", default="outputs", help="Session output directory")
    
    # === RESUME command ===
    resume_parser = subparsers.add_parser("resume", help="Resume a checkpointed run")
    resume_parser.add_argument("--session", required=True, help="Session ID to resume")
    
    # === STATUS command ===
    status_parser = subparsers.add_parser("status", help="Check run status")
    status_parser.add_argument("--session", required=True, help="Session ID to check")
    
    # === DOWNLOAD command ===
    dl_parser = subparsers.add_parser("download", help="Download competition data from Kaggle")
    dl_parser.add_argument("--competition", required=True, help="Competition name")
    dl_parser.add_argument("--output", default="./data", help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "run":
        _handle_run(args)
    elif args.command == "resume":
        _handle_resume(args)
    elif args.command == "status":
        _handle_status(args)
    elif args.command == "download":
        _handle_download(args)
    else:
        parser.print_help()


def _handle_run(args):
    """Start a new competition run."""
    from core.state import ProfessorState
    from tools.operator_channel import init_hitl, emit_to_operator
    from shields.cost_governor import init_cost_governor
    
    # Create session
    session_id = f"{args.competition}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    session_dir = os.path.join(args.session_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Parse HITL channels
    channels = [c.strip() for c in args.hitl.split(",") if c.strip() != "none"]
    
    # Initialize infrastructure
    init_hitl(channels=channels, config={})
    init_cost_governor(max_calls=args.budget_calls, max_usd=args.budget_usd)
    
    # Initialize logging
    _setup_logging(session_dir)
    
    # Build initial state
    initial_state = ProfessorState(
        session_id=session_id,
        competition_name=args.competition,
        competition_url=args.competition if "kaggle.com" in args.competition else "",
        raw_data_path=os.path.abspath(args.data),
        pipeline_depth=args.depth or "standard",
        pipeline_depth_auto_detected=args.depth is None,
        hitl_mode=args.hitl_mode,
        hitl_channels=channels,
        hackathon_mode=args.mode == "hackathon",
        llm_budget_calls_max=args.budget_calls,
        llm_budget_usd_max=args.budget_usd,
    )
    
    # Select and build graph
    if args.mode == "hackathon":
        from graph.hackathon_builder import build_hackathon_graph
        graph = build_hackathon_graph()
    else:
        from core.professor import build_professor_graph
        graph = build_professor_graph()
    
    # Compile with checkpointer
    from langgraph.checkpoint.sqlite import SqliteSaver
    checkpointer = SqliteSaver.from_conn_string(os.path.join(session_dir, "checkpoints.db"))
    app = graph.compile(checkpointer=checkpointer)
    
    # Run
    config = {"configurable": {"thread_id": session_id}}
    
    emit_to_operator(
        f"🚀 Professor v2 starting\n"
        f"Competition: {args.competition}\n"
        f"Mode: {args.mode}\n"
        f"Depth: {args.depth or 'auto-detect'}\n"
        f"HITL: {args.hitl_mode} on {channels}\n"
        f"Budget: {args.budget_calls} calls / ${args.budget_usd:.2f}\n"
        f"Session: {session_id}",
        level="STATUS"
    )
    
    try:
        final_state = app.invoke(initial_state, config=config)
        _save_final_state(final_state, session_dir)
    except KeyboardInterrupt:
        emit_to_operator(f"⏸️ Run interrupted. State checkpointed. Resume with: professor resume --session {session_id}", level="STATUS")
    except Exception as e:
        emit_to_operator(f"🚨 Fatal error: {str(e)[:200]}", level="ESCALATION")
        import traceback
        traceback.print_exc()
        raise


def _find_session_dir(session_id: str, base_dir: str = "outputs") -> str:
    path = os.path.join(base_dir, session_id)
    if os.path.exists(path):
        return path
    raise ValueError(f"Session not found: {session_id}")


def _handle_resume(args):
    """Resume a checkpointed run."""
    session_dir = _find_session_dir(args.session)
    
    from langgraph.checkpoint.sqlite import SqliteSaver
    checkpointer = SqliteSaver.from_conn_string(os.path.join(session_dir, "checkpoints.db"))
    
    # Detect mode from checkpointed state
    # (Simplification for mock / implementation, normally we read state or just invoke)
    print(f"Resuming session: {args.session} from {session_dir}")
    

def _handle_status(args):
    """Check run status."""
    try:
        session_dir = _find_session_dir(args.session)
        print(f"Session {args.session} exists at {session_dir}.")
    except Exception as e:
        print(e)


def _handle_download(args):
    """Download competition data from Kaggle API."""
    import subprocess
    cmd = ["kaggle", "competitions", "download", "-c", args.competition, "-p", args.output]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Downloaded to {args.output}")
        # Unzip if needed
        import glob
        for zipfile in glob.glob(os.path.join(args.output, "*.zip")):
            subprocess.run(["unzip", "-o", zipfile, "-d", args.output])
            os.remove(zipfile)
    else:
        print(f"❌ Download failed: {result.stderr}")


def _setup_logging(session_dir):
    """Configure file + console logging."""
    import logging
    logger = logging.getLogger("professor")
    logger.setLevel(logging.INFO)
    
    # File handler — everything goes to session log
    fh = logging.FileHandler(os.path.join(session_dir, "professor.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)
    
    # Console handler — warnings and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)


def _save_final_state(state, session_dir):
    """Save final state as JSON for post-run analysis."""
    import json
    from dataclasses import asdict
    state_path = os.path.join(session_dir, "final_state.json")
    with open(state_path, "w") as f:
        json.dump(asdict(state) if hasattr(state, '__dataclass_fields__') else dict(state), f, indent=2, default=str)


if __name__ == "__main__":
    main()
