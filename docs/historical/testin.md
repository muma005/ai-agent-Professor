Good instinct. Before Phase 2 builds on top of Phase 1, you need to know the base is actually solid — not just "it ran once." Let me check if there's anything in Notion about validation protocol first, then give you the full guide.Good. The multi-seed stability validator is formally a Day 19 task — but doing a manual version right now before Phase 2 is the right call. Here is the exact protocol.

---

## Why You're Doing This

One passing gate run proves the pipeline can work. Five runs across different conditions proves the pipeline is **stable**. You need to know three things before Phase 2 builds on top of this:

1. **CV variance is acceptable** — random seed shouldn't swing results by more than ±0.005
2. **State doesn't corrupt across runs** — model_registry, lineage, outputs are clean on every fresh run
3. **No silent failures** — same inputs always produce the same format outputs

---

## The 5-Run Protocol

Run these in order. Each run tests something specific. Don't skip ahead.

---

### Run 1 — Baseline Confirmation (Seed 42)

This is your reference run. Every other run is compared against this.

```bash
python main.py run \
  --competition spaceship-titanic \
  --data ./data/spaceship_titanic/train.csv \
  --budget 2.0

# After the run, record these exact values:
python -c "
import json, polars as pl, os, glob

# Find the most recent session output directory
sessions = sorted(glob.glob('outputs/spaceship*'), key=os.path.getmtime)
session_dir = sessions[-1]
print('Session dir:', session_dir)

# CV score
metrics = json.load(open(f'{session_dir}/metrics.json'))
print('CV mean:   ', metrics['cv_mean'])
print('CV std:    ', metrics['cv_std'])
print('CV scores: ', metrics['fold_scores'])
print('N features:', metrics['n_features'])

# Submission
sub = pl.read_csv(f'{session_dir}/submission.csv')
print('Sub shape: ', sub.shape)
print('Sub cols:  ', sub.columns)
print('Nulls:     ', sub.null_count().sum_horizontal().item())
print('True pct:  ', sub['Transported'].mean())

# Lineage
from core.lineage import read_lineage
session_id = session_dir.split('/')[-1]
events = read_lineage(session_id)
print('Lineage events:', len(events))
for e in events:
    print(f'  {e[\"agent\"]} → {e[\"action\"]}')
"
```

**Record this in a comparison table:**
```
Run 1 | Seed: default | CV: X.XXXX | std: X.XXXX | Sub rows: 4277 | True%: XX.X% | Events: 3 | Time: Xm
```

---

### Run 2 — Different Random Seed (Seed 7)

Tests whether CV results are seed-stable. LightGBM has internal randomness. A well-built pipeline should produce CV scores within ±0.005 of Run 1.

First, add seed control to `main.py` if it doesn't already exist:

```bash
python main.py run \
  --competition spaceship-titanic \
  --data ./data/spaceship_titanic/train.csv \
  --budget 2.0 \
  --seed 7
```

If `--seed` flag doesn't exist yet, pass it via a temporary env var or directly patch the LightGBM call with `random_state=7`. The point is to force different internal randomness.

**What to check:**
```python
# CV delta from Run 1 must be < 0.005
# If delta > 0.005 you have a seed stability problem
delta = abs(run2_cv - run1_cv)
assert delta < 0.005, f"CV too sensitive to seed: delta={delta:.4f}"

# submission.csv row count must still be 4277
# True% prediction rate should be within 2% of Run 1
# (large swings = instability in predictions, not just scores)
```

---

### Run 3 — State Isolation Test (Run Twice in Same Process)

Tests whether running the pipeline twice in the same session corrupts state. This is specifically testing the LangGraph state merge fix from Task 1.

```python
# Run this as a script: tests/test_state_isolation.py

from core.state import initial_state
from core.professor import run_professor

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
print("✓ model_registry has exactly 1 entry per run")

# 2. session_ids must be different
assert result1["session_id"] != result2["session_id"], \
    "Both runs have the same session_id — namespace collision"
print(f"✓ Sessions isolated: {result1['session_id']} vs {result2['session_id']}")

# 3. CV scores must be close (same data, same default seed)
delta = abs(result1["cv_mean"] - result2["cv_mean"])
assert delta < 0.005, f"CV unstable between identical runs: delta={delta:.4f}"
print(f"✓ CV stable: {result1['cv_mean']:.4f} vs {result2['cv_mean']:.4f} (delta={delta:.4f})")

# 4. Both submission.csvs must be valid
import polars as pl
for run_num, result in [(1, result1), (2, result2)]:
    sub = pl.read_csv(result["submission_path"])
    assert sub.shape == (4277, 2), f"Run {run_num} submission wrong shape: {sub.shape}"
    assert sub.null_count().sum_horizontal().item() == 0, \
        f"Run {run_num} submission has nulls"
    print(f"✓ Run {run_num} submission valid: {sub.shape}")

# 5. No raw DataFrames leaked into state
for run_num, result in [(1, result1), (2, result2)]:
    for key, value in result.items():
        assert not hasattr(value, 'columns'), \
            f"Run {run_num}: Raw DataFrame in state['{key}'] — pointer contract violated"
print("✓ No raw DataFrames in either state")

print("\n=== ALL ISOLATION CHECKS PASSED ===")
```

**This is the most important run.** If this fails you have a state corruption bug that will destroy Phase 2.

---

### Run 4 — Regression Test Suite

Runs the frozen regression test to confirm Phase 1 is still solid. This should already be passing from Day 7 — confirm it stays green.

```bash
# Full contract suite
pytest tests/contracts/ -v --tb=short

# Phase 1 regression test
pytest tests/regression/test_phase1_regression.py -v --tb=short

# Gate script
python tests/phase1_gate.py

# Expected final line:
# === PHASE 1 GATE: PASSED ===
```

Record the exact pytest output:
```
X passed, 0 failed, 0 errors in X.XXs
```

If anything fails here that passed on Day 7, something was changed after the freeze. Find the change with `git diff` before proceeding.

---

### Run 5 — Wall Clock & Memory Profiling

Tests whether the pipeline has memory or performance issues that will compound in Phase 2 when Optuna adds 10x the runtime.

```python
# tests/test_performance_profile.py

import time
import psutil
import os
from core.state import initial_state
from core.professor import run_professor

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
```

---

## Comparison Table — Fill This In

After all 5 runs, fill this table. If any column shows unexpected variance, investigate before Phase 2.

```
┌──────┬──────────┬────────┬────────┬──────────┬────────┬────────┬──────────┐
│ Run  │ Seed     │ CV     │ CV std │ Sub rows │ True%  │ Time   │ Mem peak │
├──────┼──────────┼────────┼────────┼──────────┼────────┼────────┼──────────┤
│  1   │ default  │ X.XXXX │ X.XXXX │ 4277     │ XX.X%  │ Xm XXs │ XXXX MB  │
│  2   │ 7        │ X.XXXX │ X.XXXX │ 4277     │ XX.X%  │ Xm XXs │ XXXX MB  │
│  3a  │ default  │ X.XXXX │ X.XXXX │ 4277     │ XX.X%  │ Xm XXs │ XXXX MB  │
│  3b  │ default  │ X.XXXX │ X.XXXX │ 4277     │ XX.X%  │ Xm XXs │ XXXX MB  │
│  4   │ (tests)  │  N/A   │  N/A   │  N/A     │  N/A   │  N/A   │  N/A     │
│  5   │ default  │ X.XXXX │ X.XXXX │ 4277     │ XX.X%  │ Xm XXs │ XXXX MB  │
└──────┴──────────┴────────┴────────┴──────────┴────────┴────────┴──────────┘
```

---

## Pass Criteria — Phase 2 Green Light

**All of these must be true before Phase 2 starts:**

```python
# 1. CV variance across runs
max_cv = max(run1_cv, run2_cv, run3a_cv, run3b_cv, run5_cv)
min_cv = min(run1_cv, run2_cv, run3a_cv, run3b_cv, run5_cv)
assert (max_cv - min_cv) < 0.008, "CV too unstable across runs"

# 2. All submissions have identical format
# Every run: 4277 rows, 2 columns, 0 nulls — no exceptions

# 3. State isolation test passed (Run 3)
# model_registry exactly 1 entry per run — no duplicates

# 4. All contract + regression tests green (Run 4)
# pytest exit code 0 — no exceptions

# 5. Wall clock under 30 min, memory under 4GB (Run 5)
# Phase 2 Optuna will multiply both — you need headroom
```

If all five pass, you have a verified, stable Phase 1 base. Phase 2 is greenlit.

If any fail, you have found a bug that would have silently corrupted Phase 2 work. Fix it now — it costs 30 minutes today versus 3 days of debugging in Phase 2.