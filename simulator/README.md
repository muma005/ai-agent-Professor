# Professor Simulator — Private Leaderboard Benchmark Infrastructure

## Overview

The Simulator is a complete competition simulation environment that mirrors real Kaggle competition dynamics. It provides:

- **Public/Private leaderboard split** (30/70) — Tests shakeup resilience
- **Daily submission limits** — Forces strategic submission behavior  
- **Percentile-calibrated medal awards** — Meaningful performance metrics
- **Deterministic, reproducible splits** — Direct comparison across code changes
- **10 initial competitions** — Binary, multiclass, regression, temporal, group splits

## Quick Start

```bash
# List available competitions
professor simulator list

# Run benchmark on all competitions (fast mode, ~2 hours)
professor simulator benchmark --all

# Run specific competition
professor simulator benchmark --competition spaceship-titanic

# Deep mode (full fidelity, ~30-60 min per competition)
professor simulator benchmark --competition spaceship-titanic --deep

# Compare with previous benchmark
professor simulator benchmark --all --compare results/benchmark_v2.2.json
```

## Architecture

```
simulator/
├── competition_registry.py   # 10 competition entries with metadata
├── data_splitter.py          # 60/12/28 split (train/public/private)
├── data_downloader.py        # Kaggle API download utility
├── leaderboard.py            # Simulated LB (public score only)
├── scorers.py                # Metric implementations (sklearn-matched)
├── report_generator.py       # Aggregate benchmark reports
├── cli.py                    # CLI: professor simulator benchmark
├── tests/
│   └── test_simulator_contract.py  # Contract tests (IMMUTABLE)
├── data/                     # Downloaded + split competition data
└── results/                  # Benchmark reports (JSON)
```

## Key Design Decisions

### Why 40% holdout (not 20%)?

Real Kaggle competitions typically have test sets equal to or larger than training sets. A 20% holdout means Professor trains on 80% of the original train — MORE data than real competitors had. That inflates simulated scores and gives false confidence.

**40% holdout** means Professor trains on 60% of the available data, which is closer to the real constraint. The 40% is then split 30/70 into public/private, matching Kaggle's typical ratio.

### Why public/private split matters

An 80/20 split with one score tells you nothing about shakeup resilience. The shakeup between public and private LB is WHERE the real test happens. Professor that overfits to public will have a large positive shakeup (dropped rank).

### Determinism is non-negotiable

Same data + same seed = exact same split every time. This is essential for regression testing — if Professor's score changes between runs on the same split, the change is real, not split variance.

## Competition Registry

Initial 10 competitions covering major types:

| Slug | Task | Metric | Domain |
|------|------|--------|--------|
| spaceship-titanic | Binary | Accuracy | Transport |
| titanic | Binary | Accuracy | Transport |
| playground-series-s4e8 | Binary | AUC | General |
| icr-identify-age-related-conditions | Binary (Imbalanced) | Balanced Log Loss | Healthcare |
| house-prices-advanced-regression-techniques | Regression | RMSLE | Real Estate |
| playground-series-s4e7 | Regression | RMSE | Materials |
| playground-series-s4e9 | Multiclass | Macro F1 | Healthcare |
| tabular-playground-series-dec-2022 | Multiclass | Macro F1 | General |
| store-sales-time-series-forecasting | Regression (Temporal) | RMSLE | Retail |
| livestock-disease-prediction | Binary (Group) | AUC | Agriculture |

## Data Isolation Rules (NON-NEGOTIABLE)

1. **Professor NEVER sees private labels.** `.private_labels.csv` is a dotfile (hidden). Only the Simulated Leaderboard component reads it.

2. **Professor NEVER sees public labels directly.** Public labels are used only by the Simulated Leaderboard to return a "public LB score" after each submission. Professor receives only the score number, never the actual labels.

3. **The test.csv Professor receives contains BOTH public and private rows, shuffled together.** Professor cannot distinguish which rows are public vs private. Same as real Kaggle.

4. **The split is deterministic.** Same seed = same split = same rows. This means Professor's score on the same competition is directly comparable across code changes.

## Benchmark Modes

| Mode | Time/Comp | Purpose |
|------|-----------|---------|
| Fast (default) | 10 min | Daily regression checks |
| Deep | 30-60 min | Performance testing, phase gates |
| A/B Test | 10 min | Component validation |

## Report Schema

Benchmark reports (`benchmark_report.json`) include:

```json
{
  "run_id": "benchmark_2026_03_27_v2.0.0",
  "professor_version": "2.0.0",
  "n_competitions": 10,
  "aggregate_metrics": {
    "median_percentile": 12.5,
    "gold_rate": 0.30,
    "silver_rate": 0.10,
    "bronze_rate": 0.40,
    "medal_rate": 0.80,
    "mean_shakeup": -2.1
  },
  "per_competition": [...],
  "version_comparison": {...}
}
```

## Running Contract Tests

```bash
# Run all contract tests
pytest simulator/tests/test_simulator_contract.py -v

# These tests verify:
# - Split determinism
# - Data isolation (no leakage)
# - Scorer accuracy (matches sklearn)
# - Leaderboard behavior (public/private separation)
```

## Integration with Professor

The simulator wraps Professor — it does NOT modify Professor's code. Professor receives:
- `train.csv` (with target)
- `test.csv` (features only)
- `sample_submission.csv`

Professor runs its full pipeline and produces `submission.csv`. The simulator scores it against hidden labels and returns public scores.

```python
from simulator import CompetitionEntry, split_competition_data, SimulatedLeaderboard

# Get competition
entry = get_competition("spaceship-titanic")

# Split data
split = split_competition_data("path/to/data.csv", entry)

# Create leaderboard
lb = SimulatedLeaderboard(entry, split)

# Run Professor (produces submission.csv)
run_professor(entry.get_train_path(), entry.get_test_path())

# Submit
result = lb.submit("outputs/submission.csv")
print(f"Public score: {result.public_score}")

# End competition (reveals private score)
final = lb.competition_end()
print(f"Private score: {final.best_private_score}")
print(f"Medal: {final.medal}")
```

## Extending the Registry

Add new competitions to `competition_registry.py`:

```python
REGISTRY.append(CompetitionEntry(
    slug="my-competition",
    title="My Competition",
    kaggle_id="my-competition",
    task_type="binary",
    target_column="target",
    id_column="id",
    metric="auc",
    metric_direction="maximize",
    lb_percentiles={10: 0.90, 25: 0.88, 50: 0.85, 75: 0.82},
    gold_threshold=0.90,
    silver_threshold=0.88,
    bronze_threshold=0.85,
    total_teams=5000,
    split_strategy="stratified",
    primary_domain="finance",
))
```

## Troubleshooting

### "Data not found" error

Download competition data:
```bash
# Using Kaggle CLI
kaggle competitions download -c spaceship-titanic -p simulator/data/spaceship-titanic/raw

# Or manually from kaggle.com and extract to simulator/data/<slug>/raw/
```

### "Kaggle CLI not found"

Install:
```bash
pip install kaggle
```

Configure (`~/.kaggle/kaggle.json`):
```json
{"username":"your-username","key":"your-api-key"}
```

### Split validation failed

If target distribution is not preserved, the split is re-done with seed + 100. Maximum 5 re-seeds before raising an error. Check your data for extreme imbalance or rare classes.

## References

- Full specification: `harness.md`
- Contract tests: `simulator/tests/test_simulator_contract.py`
- Original v1 harness: `tools/harness/` (deprecated, kept for reference)
