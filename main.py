# main.py

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="Professor — Autonomous Kaggle Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run --competition spaceship-titanic --data ./data/spaceship_titanic/
  python main.py run --competition spaceship-titanic --data ./data/spaceship_titanic/ --budget 2.00
  python main.py status --session spaceship_abc123
  python main.py history
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── run command ───────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Start a competition run")
    run_parser.add_argument("--competition", required=True, help="Competition name")
    run_parser.add_argument("--data", required=True, help="Path to data folder")
    run_parser.add_argument("--budget", type=float, default=2.00, help="Budget in USD (default: 2.00)")
    run_parser.add_argument("--task-type", default="auto",
                            choices=["auto", "tabular_classification",
                                     "tabular_regression", "timeseries"],
                            help="Task type (default: auto-detect)")

    # ── status command ────────────────────────────────────────────
    status_parser = subparsers.add_parser("status", help="Check status of a running session")
    status_parser.add_argument("--session", required=True, help="Session ID")

    # ── history command ───────────────────────────────────────────
    subparsers.add_parser("history", help="List all past runs")

    # ── check command (verify environment) ───────────────────────
    subparsers.add_parser("check", help="Verify environment — run this on Day 1")

    args = parser.parse_args()

    if args.command == "run":
        _run(args)

    elif args.command == "status":
        _status(args.session)

    elif args.command == "history":
        _history()

    elif args.command == "check":
        _check_environment()

    else:
        parser.print_help()


def _run(args):
    from core.state import initial_state
    from core.professor import run_professor

    # Validate data path
    if not os.path.exists(args.data):
        print(f"[ERROR] Data path does not exist: {args.data}")
        sys.exit(1)

    state = initial_state(
        competition=args.competition,
        data_path=args.data,
        budget_usd=args.budget,
        task_type=args.task_type
    )

    print(f"[Professor] Session:     {state['session_id']}")
    print(f"[Professor] Competition: {state['competition_name']}")
    print(f"[Professor] Data:        {state['raw_data_path']}")
    print(f"[Professor] Budget:      ${state['cost_tracker']['budget_usd']:.2f}")
    print()

    result = run_professor(state)

    print()
    print(f"[Professor] COMPLETE")
    print(f"[Professor] CV score:    {result.get('cv_mean', 'N/A')}")
    print(f"[Professor] Submission:  {result.get('submission_path', 'N/A')}")
    print(f"[Professor] LLM calls:   {result['cost_tracker']['llm_calls']}")


def _status(session_id: str):
    log_path = f"outputs/logs/{session_id}.jsonl"
    if not os.path.exists(log_path):
        print(f"[ERROR] No session found: {session_id}")
        return

    with open(log_path) as f:
        lines = f.readlines()
    print(f"[Status] Session {session_id}: {len(lines)} log entries")


def _history():
    log_dir = "outputs/logs"
    if not os.path.exists(log_dir):
        print("[History] No runs yet.")
        return
    sessions = [f.replace(".jsonl", "") for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    if not sessions:
        print("[History] No completed runs.")
    for s in sessions:
        print(f"  {s}")


def _check_environment():
    """Verify all Day 1 setup is correct. Run: python main.py check"""
    import importlib
    print("[Check] Verifying Professor environment...\n")
    ok = True

    # Check API keys
    for key in ["GROQ_API_KEY", "GOOGLE_API_KEY"]:
        val = os.getenv(key)
        if val:
            print(f"  [OK] {key} present")
        else:
            print(f"  [FAIL] {key} MISSING - add to .env")
            ok = False

    # Check critical libraries
    libs = ["langgraph", "groq", "google.generativeai", "polars",
            "lightgbm", "optuna", "chromadb", "fakeredis", "mlflow",
            "RestrictedPython"]
    for lib in libs:
        try:
            importlib.import_module(lib.replace("-", "_"))
            print(f"  [OK] {lib}")
        except ImportError:
            print(f"  [FAIL] {lib} NOT INSTALLED")
            ok = False

    # Check folder structure
    required_dirs = ["core", "agents", "tools", "guards", "memory",
                     "tests/contracts", "outputs/logs", "data"]
    for d in required_dirs:
        if os.path.exists(d):
            print(f"  [OK] {d}/")
        else:
            print(f"  [FAIL] {d}/ MISSING")
            ok = False

    print()
    if ok:
        print("[Check] Environment ready. Start building.")
    else:
        print("[Check] Fix the issues above before proceeding.")


if __name__ == "__main__":
    main()
