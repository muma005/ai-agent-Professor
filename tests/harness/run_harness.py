"""
CLI entry point for the historical competition harness.

Usage:
  python tests/harness/run_harness.py --list
  python tests/harness/run_harness.py -c spaceship-titanic --fast
  python tests/harness/run_harness.py -c spaceship-titanic -s my_run_01
  python tests/harness/run_harness.py -c titanic
  python tests/harness/run_harness.py -c house-prices-advanced-regression-techniques
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.harness.harness_runner import run_harness
from tools.harness.competition_registry import COMPETITION_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Professor Agent — Historical Competition Harness")
    parser.add_argument("--competition", "-c", type=str)
    parser.add_argument("--session",     "-s", type=str)
    parser.add_argument("--fast",        action="store_true", help="Enable FAST_MODE (skip null importance Stage 2)")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-split",    action="store_true")
    parser.add_argument("--list", "-l",  action="store_true", help="List available competitions")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable competitions:")
        for cid, spec in COMPETITION_REGISTRY.items():
            print(f"  {cid:<55} metric={spec.evaluation_metric}")
        return

    if not args.competition:
        parser.print_help()
        sys.exit(1)

    report = run_harness(
        competition_id=args.competition,
        session_id=args.session,
        fast_mode=args.fast,
        force_download=args.force_download,
        force_split=args.force_split,
    )

    sys.exit(0 if report.get("simulated_medal", "None") != "None" else 1)


if __name__ == "__main__":
    main()
