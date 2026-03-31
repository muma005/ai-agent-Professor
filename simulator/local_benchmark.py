"""
Local Benchmark Runner — Run Professor on 20 trials locally with fast mode.

No Modal, no cloud. Just run all competitions sequentially on your local machine.

Usage:
    python simulator/local_benchmark.py --mode fast --trials 20
    python simulator/local_benchmark.py --competition spaceship-titanic --trials 20

Fast mode:
- Uses ProfessorConfig(fast_mode=True)
- Skips CompetitionIntel, EDA, RedTeamCritic
- 1 Optuna trial (default parameters)
- Skips LLM feature rounds
- Completes in ~5 minutes per trial
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.competition_registry import REGISTRY, get_competition, CompetitionEntry
from simulator.data_splitter import split_competition_data, ensure_data_cached
from simulator.leaderboard import SimulatedLeaderboard


def run_single_trial(
    entry: CompetitionEntry,
    trial_num: int,
    mode: str = "fast",
    base_output_dir: str = "simulator/results/local_benchmark"
) -> dict:
    """Run Professor against one competition for one trial. Returns result dict."""
    
    # P5.1 FIX: Import ProfessorConfig
    from core.config import ProfessorConfig
    from core.state import initial_state
    from core.professor import run_professor

    print(f"\n{'='*70}")
    print(f"  TRIAL {trial_num}: {entry.slug}")
    print(f"{'='*70}")

    # Create config based on mode
    if mode == "fast":
        config = ProfessorConfig(fast_mode=True)
    else:
        config = ProfessorConfig(production_mode=True)
    
    print(f"[Benchmark] Using config: fast_mode={config.fast_mode}")
    print(f"[Benchmark] Optuna trials: {config.ml_optimizer.optuna_trials}")
    print(f"[Benchmark] Models: {config.ml_optimizer.models_to_try}")

    # Cache data
    cache_dir = "simulator/data/competitions"
    try:
        ensure_data_cached(entry, cache_dir)
    except Exception as e:
        print(f"[WARNING] Could not cache data: {e}")
        # Continue anyway - data might already be there

    # Split (deterministic - same seed every time for reproducibility)
    split_dir = f"simulator/data/splits/{entry.slug}"
    data_path = f"{cache_dir}/{entry.slug}/full_data.csv"

    if not Path(data_path).exists():
        # Try to find any CSV in the competition folder
        comp_folder = Path(cache_dir) / entry.slug
        if comp_folder.exists():
            csv_files = list(comp_folder.glob("*.csv"))
            if csv_files:
                data_path = str(csv_files[0])
                print(f"[INFO] Using alternative data file: {data_path}")

    if not Path(data_path).exists():
        return {
            "trial": trial_num,
            "slug": entry.slug,
            "error": f"Data file not found: {data_path}",
            "runtime_seconds": 0,
        }

    try:
        split = split_competition_data(
            data_path=data_path,
            entry=entry,
            output_dir=split_dir,
        )
    except Exception as e:
        return {
            "trial": trial_num,
            "slug": entry.slug,
            "error": f"Split failed: {e}",
            "runtime_seconds": 0,
        }

    # Create leaderboard
    lb = SimulatedLeaderboard(entry, split)

    # Run Professor
    start = time.time()

    try:
        # Import Professor's pipeline
        from core.professor import run_professor
        from core.state import ProfessorState

        # Create unique session for this trial
        session_id = f"trial_{entry.slug}_{trial_num}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        output_dir = f"{base_output_dir}/{session_id}"
        os.makedirs(output_dir, exist_ok=True)

        # FORCE environment variables for fast execution
        os.environ["PROFESSOR_FAST_MODE"] = "1"
        os.environ["PROFESSOR_OPTUNA_TRIALS"] = str(config["optuna_trials"])
        os.environ["PROFESSOR_NULL_IMPORTANCE_SHUFFLES"] = str(config["null_importance_shuffles"])
        os.environ["PROFESSOR_SKIP_FEATURE_FACTORY"] = str(config["skip_feature_factory"])
        os.environ["PROFESSOR_SKIP_EDA"] = str(config["skip_eda"])
        os.environ["PROFESSOR_SKIP_COMPETITION_INTEL"] = str(config["skip_competition_intel"])

        # Create schema.json
        import polars as pl
        import pickle

        train_df = pl.read_csv(split.train_path)

        schema = {
            "columns": [],
            "target": entry.target_column,
            "id_columns": [entry.id_column],
        }

        for col in train_df.columns:
            dtype = str(train_df[col].dtype)
            col_type = "categorical" if dtype in ["Utf8", "Categorical"] else "numerical"
            if col == entry.target_column:
                col_role = "target"
            elif col == entry.id_column:
                col_role = "id"
            else:
                col_role = "feature"

            schema["columns"].append({
                "name": col,
                "dtype": dtype,
                "type": col_type,
                "role": col_role,
                "missing_ratio": float(train_df[col].null_count()) / len(train_df),
            })

        schema_path = f"{output_dir}/schema.json"
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)

        # Create preprocessor.pkl
        preprocessor = {
            "columns": train_df.columns,
            "target": entry.target_column,
            "id_column": entry.id_column,
            "dtypes": {col: str(train_df[col].dtype) for col in train_df.columns},
        }
        preprocessor_path = f"{output_dir}/preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)

        # Create clean_data.csv
        clean_data_path = f"{output_dir}/clean_data.csv"
        train_df.write_csv(clean_data_path)

        # Build initial state
        state = {
            "session_id": session_id,
            "competition_name": entry.slug,
            "raw_data_path": split.train_path,
            "clean_data_path": clean_data_path,
            "schema_path": schema_path,
            "preprocessor_path": preprocessor_path,
            "train_path": split.train_path,
            "test_path": split.test_path,
            "sample_submission_path": split.sample_submission_path,
            "target_col": entry.target_column,
            "target_column": entry.target_column,
            "id_column": entry.id_column,
            "id_columns": [entry.id_column],
            "metric": entry.metric,
            "task_type": entry.task_type,
            "config": config,
            "output_dir": output_dir,
        }

        # Run pipeline with config
        result = run_professor(state, timeout_seconds=3000, config=config)

        # Submit
        submission_path = result.get("submission_path")
        if submission_path:
            submit_result = lb.submit(submission_path)
        else:
            submit_result = None

        # Reveal scores
        final = lb.competition_end()

        runtime = time.time() - start

        # Build result
        result_dict = {
            "trial": trial_num,
            "slug": entry.slug,
            "task_type": entry.task_type,
            "domain": entry.primary_domain,
            "metric": entry.metric,
            "cv_score": result.get("cv_mean"),
            "public_score": final.best_public_score,
            "private_score": final.best_private_score,
            "public_percentile": final.public_rank_pct,
            "private_percentile": final.private_rank_pct,
            "shakeup": final.shakeup_positions,
            "medal": final.medal,
            "total_submissions": final.total_submissions,
            "runtime_seconds": round(runtime, 1),
            "mode": mode,
            "error": None,
        }

        # PRINT RESULTS
        print(f"\n{'='*70}")
        print(f"  TRIAL {trial_num} COMPLETE: {entry.slug}")
        print(f"{'='*70}")
        print(f"  CV Score:           {result_dict['cv_score']}")
        print(f"  Public Score:       {result_dict['public_score']}")
        print(f"  Private Score:      {result_dict['private_score']}")
        print(f"  Public Percentile:  {result_dict['public_percentile']}%")
        print(f"  Private Percentile: {result_dict['private_percentile']}%")
        print(f"  Shakeup:            {result_dict['shakeup']:+.1f} positions")
        print(f"  Medal:              {result_dict['medal'].upper()}")
        print(f"  Runtime:            {result_dict['runtime_seconds']:.0f}s ({result_dict['runtime_seconds']/60:.1f} min)")
        print(f"{'='*70}\n")

        # Save individual result
        result_path = f"{output_dir}/trial_result.json"
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        return result_dict

    except Exception as e:
        runtime = time.time() - start
        error_msg = f"{str(e)}\n{traceback.format_exc()}"

        print(f"\n{'='*70}")
        print(f"  TRIAL {trial_num} FAILED: {entry.slug}")
        print(f"  Error: {str(e)[:200]}")
        print(f"{'='*70}\n")

        return {
            "trial": trial_num,
            "slug": entry.slug,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "runtime_seconds": round(runtime, 1),
            "mode": mode,
        }


def run_full_benchmark(
    mode: str = "fast",
    trials: int = 20,
    competition_slug: Optional[str] = None,
) -> dict:
    """Run full benchmark across all competitions and trials sequentially."""

    start_time = time.time()

    if competition_slug:
        slugs = [competition_slug]
    else:
        slugs = [e.slug for e in REGISTRY]

    all_results = []

    for slug in slugs:
        entry = get_competition(slug)
        print(f"\n{'='*70}")
        print(f"  COMPETITION: {entry.slug}")
        print(f"  Trials: {trials}")
        print(f"{'='*70}")

        for trial_num in range(1, trials + 1):
            result = run_single_trial(entry, trial_num, mode)
            all_results.append(result)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / len(all_results)
            remaining = (trials * len(slugs)) - len(all_results)
            eta_min = (remaining * avg_time) / 60
            print(f"\n[PROGRESS] {len(all_results)}/{trials * len(slugs)} trials complete | "
                  f"Avg: {avg_time:.1f}s/trial | ETA: {eta_min:.1f} min")

    # Aggregate
    successful = [r for r in all_results if r.get("error") is None]
    failed = [r for r in all_results if r.get("error") is not None]

    private_scores = [r["private_percentile"] for r in successful if r.get("private_percentile") is not None]
    medals = [r["medal"] for r in successful]

    import numpy as np

    total_runtime = time.time() - start_time

    report = {
        "run_id": f"local_benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "professor_version": "v2.0",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "trials_per_competition": trials,
        "n_competitions": len(slugs),
        "n_successful": len(successful),
        "n_failed": len(failed),
        "total_runtime_seconds": round(total_runtime, 1),
        "aggregate_metrics": {
            "median_percentile": float(np.median(private_scores)) if private_scores else None,
            "mean_percentile": float(np.mean(private_scores)) if private_scores else None,
            "std_percentile": float(np.std(private_scores)) if private_scores else None,
            "min_percentile": float(np.min(private_scores)) if private_scores else None,
            "max_percentile": float(np.max(private_scores)) if private_scores else None,
            "gold_rate": medals.count("gold") / len(medals) if medals else 0,
            "silver_rate": medals.count("silver") / len(medals) if medals else 0,
            "bronze_rate": medals.count("bronze") / len(medals) if medals else 0,
            "medal_rate": sum(1 for m in medals if m != "none") / len(medals) if medals else 0,
        },
        "per_competition_summary": {},
        "all_trials": all_results,
    }

    # Per-competition summary
    for slug in slugs:
        comp_results = [r for r in successful if r["slug"] == slug]
        if comp_results:
            comp_private = [r["private_percentile"] for r in comp_results if r.get("private_percentile")]
            comp_medals = [r["medal"] for r in comp_results]
            report["per_competition_summary"][slug] = {
                "n_trials": len(comp_results),
                "median_percentile": float(np.median(comp_private)) if comp_private else None,
                "mean_percentile": float(np.mean(comp_private)) if comp_private else None,
                "std_percentile": float(np.std(comp_private)) if comp_private else None,
                "gold": comp_medals.count("gold"),
                "silver": comp_medals.count("silver"),
                "bronze": comp_medals.count("bronze"),
                "none": comp_medals.count("none"),
            }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Local Benchmark Runner — Run Professor on 20 trials locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulator/local_benchmark.py --mode fast --trials 20
  python simulator/local_benchmark.py --mode deep --trials 5
  python simulator/local_benchmark.py --competition spaceship-titanic --trials 20
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "deep"],
        help="Run mode: fast (30 trials) or deep (200 trials)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of trials per competition (default: 20)"
    )
    parser.add_argument(
        "--competition",
        type=str,
        default="",
        help="Run single competition (default: all competitions)"
    )

    args = parser.parse_args()

    # Run benchmark
    report = run_full_benchmark(
        mode=args.mode,
        trials=args.trials,
        competition_slug=args.competition if args.competition else None,
    )

    # Print summary
    agg = report["aggregate_metrics"]
    print(f"\n{'='*70}")
    print(f"  PROFESSOR LOCAL BENCHMARK — {report['professor_version']}")
    print(f"{'='*70}")
    print(f"  Mode:                 {args.mode}")
    print(f"  Competitions:         {report['n_competitions']}")
    print(f"  Trials per comp:      {args.trials}")
    print(f"  Total runs:           {report['n_successful'] + report['n_failed']}")
    print(f"  Successful:           {report['n_successful']}")
    print(f"  Failed:               {report['n_failed']}")
    print(f"  Total runtime:        {report['total_runtime_seconds']:.0f}s ({report['total_runtime_seconds']/3600:.2f} hours)")
    print(f"{'='*70}")

    if agg["median_percentile"]:
        print(f"\n  AGGREGATE METRICS:")
        print(f"  Median percentile:  {agg['median_percentile']:.1f}%")
        print(f"  Mean percentile:    {agg['mean_percentile']:.1f}%")
        print(f"  Std deviation:      {agg['std_percentile']:.1f}%")
        print(f"  Range:              [{agg['min_percentile']:.1f}%, {agg['max_percentile']:.1f}%]")
        print(f"  Gold rate:          {agg['gold_rate']:.0%}")
        print(f"  Silver rate:        {agg['silver_rate']:.0%}")
        print(f"  Bronze rate:        {agg['bronze_rate']:.0%}")
        print(f"  Medal rate:         {agg['medal_rate']:.0%}")

    if report["per_competition_summary"]:
        print(f"\n  PER-COMPETITION SUMMARY:")
        for slug, stats in report["per_competition_summary"].items():
            print(f"\n  {slug}:")
            print(f"    Trials:     {stats['n_trials']}")
            print(f"    Median pct: {stats['median_percentile']:.1f}% (±{stats['std_percentile']:.1f}%)")
            print(f"    Medals:     🥇{stats['gold']} 🥈{stats['silver']} 🥉{stats['bronze']} ({stats['none']} no medal)")

    # Save report
    Path("simulator/results/local_benchmark").mkdir(parents=True, exist_ok=True)
    out = f"simulator/results/local_benchmark/{report['run_id']}.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {out}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
