# tests/phase1_gate.py
#
# Phase 1 Gate Script — standalone, not pytest.
# Run: python tests/phase1_gate.py
#
# Defines explicit pass/fail conditions for the Phase 1 gate.
# Must print PASSED or FAILED at the end.

import os
import sys
import time
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import initial_state
from core.professor import run_professor
from tools.submit_tools import validate_existing_submission


# ── Gate Constants ────────────────────────────────────────────────
# Submission 0 CV: our Day 4 manual baseline CV AUC on Spaceship Titanic.
# If you didn't record it, 0.775 is a conservative floor.
SUBMISSION_0_CV = 0.775

# Buffer for random seed variance between runs
CV_BUFFER = 0.005

# Absolute floor -- below this means the pipeline is broken
CV_ABSOLUTE_FLOOR = 0.70

# Wall clock limit in seconds (30 minutes)
WALL_CLOCK_LIMIT_SECONDS = 1800

# Competition data paths
COMPETITION = "spaceship-titanic"
DATA_PATH = "data/spaceship_titanic/train.csv"
SAMPLE_SUBMISSION_PATH = "data/spaceship_titanic/sample_submission.csv"


def main():
    results = []
    gate_passed = True

    print("=" * 60)
    print("  PHASE 1 GATE -- Professor Agent")
    print("=" * 60)
    print()

    # ── Hard fail: verify data exists ─────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"HARD FAIL: train.csv not found at {DATA_PATH}")
        print("=== PHASE 1 GATE: FAILED ===")
        sys.exit(1)

    if not os.path.exists(SAMPLE_SUBMISSION_PATH):
        print(f"HARD FAIL: sample_submission.csv not found at {SAMPLE_SUBMISSION_PATH}")
        print("=== PHASE 1 GATE: FAILED ===")
        sys.exit(1)

    # ── Run the full pipeline ─────────────────────────────────────
    print("[Gate] Running full pipeline...")
    print()

    state = initial_state(
        competition=COMPETITION,
        data_path=DATA_PATH,
        budget_usd=2.0,
        task_type="auto"
    )
    session_id = state["session_id"]

    start_time = time.time()
    try:
        result = run_professor(state)
    except Exception as e:
        print(f"\nHARD FAIL: Pipeline crashed with exception:\n  {type(e).__name__}: {e}")
        print("=== PHASE 1 GATE: FAILED ===")
        sys.exit(1)
    elapsed = time.time() - start_time

    print()
    print("-" * 60)
    print("  GATE RESULTS")
    print("-" * 60)
    print()

    # ── Hard fail: model_registry must not be empty ───────────────
    if not result.get("model_registry"):
        print("HARD FAIL: model_registry is empty after pipeline run")
        print("=== PHASE 1 GATE: FAILED ===")
        sys.exit(1)

    # ── Hard fail: submission_path must exist ─────────────────────
    submission_path = result.get("submission_path")
    if not submission_path or not os.path.exists(submission_path):
        print(f"HARD FAIL: submission.csv not found at {submission_path}")
        print("=== PHASE 1 GATE: FAILED ===")
        sys.exit(1)

    cv_mean = result.get("cv_mean", 0.0)
    cv_floor = SUBMISSION_0_CV - CV_BUFFER

    # ── Condition 1: CV >= Submission 0 - buffer ──────────────────
    if cv_mean >= cv_floor:
        msg = f"  PASS  CV {cv_mean:.4f} >= floor {cv_floor:.4f} (Sub0: {SUBMISSION_0_CV} - {CV_BUFFER})"
        results.append(("PASS", msg))
    else:
        msg = f"  FAIL  CV {cv_mean:.4f} < floor {cv_floor:.4f} (Sub0: {SUBMISSION_0_CV} - {CV_BUFFER})"
        results.append(("FAIL", msg))
        gate_passed = False

    # ── Condition 2: submission.csv validates ─────────────────────
    v = validate_existing_submission(submission_path, SAMPLE_SUBMISSION_PATH)
    if v["valid"]:
        msg = "  PASS  submission.csv valid: correct columns, row count, IDs, zero nulls"
        results.append(("PASS", msg))
    else:
        msg = f"  FAIL  submission.csv invalid: {v['errors']}"
        results.append(("FAIL", msg))
        gate_passed = False

    # ── Condition 3: All contract tests green ─────────────────────
    contract_result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/contracts/", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    if contract_result.returncode == 0:
        msg = f"  PASS  All contract tests green (pytest exit code 0)"
        results.append(("PASS", msg))
    else:
        msg = f"  FAIL  Contract tests failed (pytest exit code {contract_result.returncode})"
        results.append(("FAIL", msg))
        gate_passed = False

    # ── Condition 4: Wall clock < 30 minutes ──────────────────────
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    if elapsed < WALL_CLOCK_LIMIT_SECONDS:
        msg = f"  PASS  Wall clock: {minutes}m {seconds}s < 30m limit"
        results.append(("PASS", msg))
    else:
        msg = f"  FAIL  Wall clock: {minutes}m {seconds}s > 30m limit"
        results.append(("FAIL", msg))
        gate_passed = False

    # ── Condition 5: CV > 0.70 absolute floor ─────────────────────
    if cv_mean > CV_ABSOLUTE_FLOOR:
        msg = f"  PASS  CV {cv_mean:.4f} > {CV_ABSOLUTE_FLOOR} absolute floor"
        results.append(("PASS", msg))
    else:
        msg = f"  FAIL  CV {cv_mean:.4f} <= {CV_ABSOLUTE_FLOOR} absolute floor (pipeline broken)"
        results.append(("FAIL", msg))
        gate_passed = False

    # ── Print report ──────────────────────────────────────────────
    for status, msg in results:
        print(msg)

    print()
    print(f"  Session:    {session_id}")
    print(f"  CV AUC:     {cv_mean:.4f}")
    print(f"  Sub 0 CV:   {SUBMISSION_0_CV}")
    print(f"  Wall clock: {minutes}m {seconds}s")
    print(f"  Submission: {submission_path}")
    print()

    if gate_passed:
        print("=" * 60)
        print("  === PHASE 1 GATE: PASSED ===")
        print("=" * 60)
        print()
        print(f"  Upload to Kaggle:")
        print(f"    kaggle competitions submit -c {COMPETITION} "
              f"-f {submission_path} -m 'Professor Phase 1 baseline'")
    else:
        print("=" * 60)
        print("  === PHASE 1 GATE: FAILED ===")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
