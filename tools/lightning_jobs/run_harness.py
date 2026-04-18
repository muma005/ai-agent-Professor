import os
import json
import argparse
import time
import sys
from pathlib import Path

# Add project root to sys path as Job.run uploads the whole directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import simulator runner 
try:
    from simulator.cli import _run_professor_pipeline
except ImportError:
    # Maybe older hierarchy where harness runner is used
    try:
        from tools.harness.harness_runner import run_harness as _run_professor_pipeline
    except ImportError:
        _run_professor_pipeline = None

def main():
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition_id", required=True)
    parser.add_argument("--session_id", required=True)
    args = parser.parse_args()

    # Simulator runs might just create their own paths or we use standard 
    result_path = f"/home/{args.session_id}/harness_result.json"

    result = {
        "success": False,
        "runtime_seconds": 0.0,
        "report": {}
    }

    try:
        if not _run_professor_pipeline:
            raise ImportError("Could not import harness runner.")
            
        # The user's harness runner usually returns a dict report or we just mark success
        # The prompt says: "What it sends back: the full harness report JSON"
        
        # We assume _run_professor_pipeline can be called. But simulator uses CLI.
        # Actually tests/harness/run_harness.py does `run_harness(competition_id=..., session_id=...)`
        from tools.harness.harness_runner import run_harness
        
        report = run_harness(
            competition_id=args.competition_id,
            session_id=args.session_id,
            fast_mode=False
        )
        
        result["report"] = report
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        
    finally:
        result["runtime_seconds"] = round(time.time() - t_start, 1)
        os.makedirs(f"/home/{args.session_id}", exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
