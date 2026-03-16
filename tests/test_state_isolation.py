import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import initial_state
from core.professor import run_professor


@pytest.mark.slow
def test_state_isolation():
    print("=== First run ===")
    state1 = initial_state(
        competition="spaceship-titanic",
        data_path="data/spaceship_titanic/train.csv",
        budget_usd=2.0
    )
    result1 = run_professor(state1)

    print("\n=== Second run (fresh state) ===")
    state2 = initial_state(
        competition="spaceship-titanic",
        data_path="data/spaceship_titanic/train.csv",
        budget_usd=2.0
    )
    result2 = run_professor(state2)

    # Critical checks
    print("\n=== Isolation Checks ===")

    # 1. model_registry must have exactly 1 entry after each run
    assert len(result1["model_registry"]) == 1, \
        f"Run 1 model_registry has {len(result1['model_registry'])} entries — expected 1"
    assert len(result2["model_registry"]) == 1, \
        f"Run 2 model_registry has {len(result2['model_registry'])} entries — expected 1"
    print("PASS: model_registry has exactly 1 entry per run")

    # 2. session_ids must be different
    assert result1["session_id"] != result2["session_id"], \
        "Both runs have the same session_id — namespace collision"
    print(f"PASS: Sessions isolated: {result1['session_id']} vs {result2['session_id']}")

    # 3. CV scores must be close (same data, same default seed)
    delta = abs(result1["cv_mean"] - result2["cv_mean"])
    assert delta < 0.005, f"CV unstable between identical runs: delta={delta:.4f}"
    print(f"PASS: CV stable: {result1['cv_mean']:.4f} vs {result2['cv_mean']:.4f} (delta={delta:.4f})")

    # 4. Both submission.csvs must be valid
    import polars as pl
    for run_num, result in [(1, result1), (2, result2)]:
        sub = pl.read_csv(result["submission_path"])
        assert sub.shape == (4277, 2), f"Run {run_num} submission wrong shape: {sub.shape}"
        assert sub.null_count().sum_horizontal().item() == 0, \
            f"Run {run_num} submission has nulls"
        print(f"PASS: Run {run_num} submission valid: {sub.shape}")

    # 5. No raw DataFrames leaked into state
    for run_num, result in [(1, result1), (2, result2)]:
        for key, value in result.items():
            assert not hasattr(value, 'columns'), \
                f"Run {run_num}: Raw DataFrame in state['{key}'] — pointer contract violated"
    print("PASS: No raw DataFrames in either state")

print("\n=== ALL ISOLATION CHECKS PASSED ===")
