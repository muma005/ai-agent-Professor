import time
import os
import psutil
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import initial_state
from core.professor import run_professor


@pytest.mark.slow
def test_performance_profile():
    process = psutil.Process()

    # Baseline memory before run
    mem_before = process.memory_info().rss / 1e6  # MB
    time_start = time.time()

    state = initial_state(
        competition="spaceship-titanic",
        data_path="data/spaceship_titanic/train.csv",
        budget_usd=2.0
    )
    result = run_professor(state)

    time_end = time.time()
    mem_after = process.memory_info().rss / 1e6  # MB

    wall_clock = time_end - time_start
    mem_delta  = mem_after - mem_before

    print(f"\n=== Performance Profile ===")
    print(f"Wall clock:     {wall_clock:.1f}s ({wall_clock/60:.1f} min)")
    print(f"Memory before:  {mem_before:.0f} MB")
    print(f"Memory after:   {mem_after:.0f} MB")
    print(f"Memory delta:   {mem_delta:.0f} MB")
    print(f"CV score:       {result['cv_mean']:.4f}")

    # Thresholds
    assert wall_clock < 1800, \
        f"Pipeline too slow: {wall_clock/60:.1f} min > 30 min limit. Phase 3 Optuna will take 10x this."

    assert mem_after < 4000, \
        f"Memory usage too high: {mem_after:.0f} MB. Will OOM during Optuna with multiple trials."

    assert mem_delta < 1000, \
        f"Memory leak suspected: {mem_delta:.0f} MB retained after run. GC not cleaning up properly."

    print(f"\n✓ Wall clock {wall_clock/60:.1f} min < 30 min limit")
    print(f"✓ Peak memory {mem_after:.0f} MB < 4000 MB limit")
    print(f"✓ Memory delta {mem_delta:.0f} MB < 1000 MB (no leak)")
    print(f"\n=== PERFORMANCE PROFILE: PASSED ===")
