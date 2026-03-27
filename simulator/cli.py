"""
CLI for Professor Benchmark Simulator.

Usage:
    professor benchmark --all                    # Run all competitions (fast mode)
    professor benchmark --competition <slug>     # Run specific competition
    professor benchmark --deep                   # Full-fidelity simulation
    professor benchmark --ab-test <component>    # A/B test a component
    professor benchmark --compare <prev.json>    # Compare with previous run

Modes:
- Fast mode (default): 10 min/competition, capped Optuna trials
- Deep mode: Full fidelity, multi-day simulation
- A/B test: Run with/without a component, compare results
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from simulator.competition_registry import (
    REGISTRY,
    REGISTRY_BY_SLUG,
    CompetitionEntry,
    get_competition,
)
from simulator.data_splitter import split_competition_data
from simulator.leaderboard import SimulatedLeaderboard
from simulator.report_generator import (
    generate_benchmark_report,
    print_benchmark_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Professor Benchmark Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  professor benchmark --all
  professor benchmark --competition spaceship-titanic
  professor benchmark --all --deep
  professor benchmark --all --ab-test creative_hypothesis_engine
  professor benchmark --all --compare results/benchmark_v2.2.json
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument(
        "--all",
        action="store_true",
        help="Run all competitions in registry",
    )
    bench_parser.add_argument(
        "--competition",
        type=str,
        help="Run a specific competition (slug)",
    )
    bench_parser.add_argument(
        "--deep",
        action="store_true",
        help="Deep mode: full fidelity, multi-day simulation",
    )
    bench_parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: capped trials, single submission (default)",
    )
    bench_parser.add_argument(
        "--ab-test",
        type=str,
        metavar="COMPONENT",
        help="A/B test a specific component",
    )
    bench_parser.add_argument(
        "--compare",
        type=str,
        metavar="PREV_REPORT",
        help="Compare with previous benchmark report",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default="simulator/results",
        help="Output directory for reports",
    )
    bench_parser.add_argument(
        "--force-split",
        action="store_true",
        help="Force re-split of data",
    )
    bench_parser.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Time limit per competition in seconds (default: 600)",
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available competitions")
    
    args = parser.parse_args()
    
    if args.command == "list":
        _list_competitions()
        return
    
    if args.command == "benchmark":
        _run_benchmark(args)
        return
    
    parser.print_help()


def _list_competitions():
    """List all available competitions in the registry."""
    print("\n" + "=" * 70)
    print("AVAILABLE COMPETITIONS")
    print("=" * 70)
    print(f"{'Slug':<45} {'Task':<15} {'Metric':<15}")
    print("-" * 70)
    
    for entry in REGISTRY:
        print(f"{entry.slug:<45} {entry.task_type:<15} {entry.metric:<15}")
    
    print("=" * 70)
    print(f"Total: {len(REGISTRY)} competitions")


def _run_benchmark(args):
    """Run benchmark based on arguments."""
    # Determine which competitions to run
    if args.all:
        competitions = REGISTRY
    elif args.competition:
        try:
            competitions = [get_competition(args.competition)]
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Error: Specify --all or --competition <slug>")
        sys.exit(1)
    
    # Determine mode
    fast_mode = args.fast or (not args.deep)
    deep_mode = args.deep
    
    print("\n" + "=" * 70)
    print("PROFESSOR BENCHMARK SIMULATOR")
    print("=" * 70)
    print(f"Mode: {'DEEP' if deep_mode else 'FAST'}")
    print(f"Competitions: {len(competitions)}")
    print(f"Time limit: {args.time_limit}s per competition")
    if args.ab_test:
        print(f"A/B Test: {args.ab_test}")
    if args.compare:
        print(f"Compare with: {args.compare}")
    print("=" * 70 + "\n")
    
    # Set environment for fast mode
    if fast_mode:
        os.environ["PROFESSOR_FAST_MODE"] = "1"
        os.environ["PROFESSOR_OPTUNA_TRIALS"] = "30"
        os.environ["PROFESSOR_NULL_IMPORTANCE_SHUFFLES"] = "5"
    else:
        os.environ["PROFESSOR_FAST_MODE"] = "0"
        os.environ["PROFESSOR_OPTUNA_TRIALS"] = "200"
        os.environ["PROFESSOR_NULL_IMPORTANCE_SHUFFLES"] = "50"
    
    # Run competitions
    results = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, entry in enumerate(competitions, 1):
        print(f"\n[{i}/{len(competitions)}] {entry.slug}")
        print("-" * 50)
        
        try:
            result = _run_single_competition(
                entry,
                fast_mode=fast_mode,
                deep_mode=deep_mode,
                force_split=args.force_split,
                time_limit=args.time_limit,
                ab_test_component=args.ab_test,
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "slug": entry.slug,
                "error": str(e),
                "private_percentile": 100.0,  # Worst case for failed
                "medal": "none",
                "shakeup": 0.0,
            })
    
    # Generate report
    run_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report = generate_benchmark_report(
        results=results,
        professor_version="2.0.0",
        run_id=run_id,
        previous_report_path=args.compare,
    )
    
    # Save report
    report_path = output_dir / f"{run_id}.json"
    report.save(str(report_path))
    print(f"\nReport saved to: {report_path}")
    
    # Print summary
    print_benchmark_summary(report)
    
    return report


def _run_single_competition(
    entry: CompetitionEntry,
    fast_mode: bool = True,
    deep_mode: bool = False,
    force_split: bool = False,
    time_limit: int = 600,
    ab_test_component: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single competition through the full simulation.
    
    This function:
    1. Splits data into train/public/private
    2. Runs Professor pipeline
    3. Submits to simulated leaderboard
    4. Returns result dict
    """
    import polars as pl
    import numpy as np
    
    start_time = time.time()
    
    # Step 1: Prepare data (split if needed)
    print(f"  [1/4] Preparing data...")
    
    # For simulation, we need source data
    # Check if cached split exists
    train_path = entry.get_train_path()
    test_path = entry.get_test_path()
    
    if not train_path.exists() or not test_path.exists() or force_split:
        # Need to download and split
        # For now, use placeholder - in real usage, download from Kaggle
        print(f"  WARNING: Data not found. Please download competition data.")
        print(f"  Expected source: {entry.cached_path or 'Kaggle API'}")
        
        # Return placeholder result
        return {
            "slug": entry.slug,
            "task_type": entry.task_type,
            "domain": entry.primary_domain,
            "metric": entry.metric,
            "cv_score": None,
            "public_score": None,
            "private_score": None,
            "cv_public_gap": None,
            "cv_private_gap": None,
            "public_percentile": 50.0,
            "private_percentile": 50.0,
            "shakeup": 0.0,
            "medal": "none",
            "total_submissions": 0,
            "runtime_seconds": 0.0,
            "winning_model": None,
            "n_features_final": None,
            "domain_features_generated": None,
            "domain_features_kept": None,
            "error": "Data not found",
        }
    
    # Step 2: Create leaderboard
    print(f"  [2/4] Creating simulated leaderboard...")
    
    split_result = split_competition_data(
        str(entry.get_train_path()),  # Use train as source (already split)
        entry,
        force=force_split,
    )
    
    leaderboard = SimulatedLeaderboard(entry, split_result)
    
    # Step 3: Run Professor pipeline
    print(f"  [3/4] Running Professor pipeline...")
    
    # Set up paths for Professor
    # Professor receives: train.csv (with target), test.csv (features only)
    professor_train_path = entry.get_train_path()
    professor_test_path = entry.get_test_path()
    sample_submission_path = entry.get_sample_submission_path()
    
    # Run Professor (placeholder - integrate with actual Professor pipeline)
    professor_result = _run_professor_pipeline(
        train_path=str(professor_train_path),
        test_path=str(professor_test_path),
        sample_submission_path=str(sample_submission_path),
        target_column=entry.target_column,
        id_column=entry.id_column,
        metric=entry.metric,
        task_type=entry.task_type,
        fast_mode=fast_mode,
        ab_test_component=ab_test_component,
        time_limit=time_limit,
    )
    
    # Step 4: Submit to leaderboard
    print(f"  [4/4] Submitting to leaderboard...")
    
    if professor_result.get("submission_path"):
        # Single submission for fast mode
        submit_result = leaderboard.submit(professor_result["submission_path"])
        
        if deep_mode:
            # Multi-day simulation
            for day in range(2, 4):  # Simulate 3 days
                leaderboard.advance_day()
                # In deep mode, Professor could iterate and improve
                # For now, just submit the same result
                # In real implementation, Professor would refine
        
        # Get final results
        final_result = leaderboard.competition_end()
        
        runtime = time.time() - start_time
        
        return {
            "slug": entry.slug,
            "task_type": entry.task_type,
            "domain": entry.primary_domain,
            "metric": entry.metric,
            "cv_score": professor_result.get("cv_score"),
            "public_score": round(final_result.best_public_score, 6),
            "private_score": round(final_result.best_private_score, 6),
            "cv_public_gap": (
                round(abs(professor_result.get("cv_score", 0) - final_result.best_public_score), 6)
                if professor_result.get("cv_score") else None
            ),
            "cv_private_gap": (
                round(abs(professor_result.get("cv_score", 0) - final_result.best_private_score), 6)
                if professor_result.get("cv_score") else None
            ),
            "public_percentile": round(final_result.public_rank_pct, 1),
            "private_percentile": round(final_result.private_rank_pct, 1),
            "shakeup": round(final_result.shakeup_positions, 1),
            "medal": final_result.medal,
            "total_submissions": final_result.total_submissions,
            "runtime_seconds": round(runtime, 1),
            "winning_model": professor_result.get("winning_model"),
            "n_features_final": professor_result.get("n_features"),
            "domain_features_generated": professor_result.get("domain_features_generated"),
            "domain_features_kept": professor_result.get("domain_features_kept"),
        }
    else:
        # Professor failed to produce submission
        runtime = time.time() - start_time
        return {
            "slug": entry.slug,
            "task_type": entry.task_type,
            "domain": entry.primary_domain,
            "metric": entry.metric,
            "cv_score": None,
            "public_score": None,
            "private_score": None,
            "cv_public_gap": None,
            "cv_private_gap": None,
            "public_percentile": 100.0,
            "private_percentile": 100.0,
            "shakeup": 0.0,
            "medal": "none",
            "total_submissions": 0,
            "runtime_seconds": round(runtime, 1),
            "winning_model": None,
            "n_features_final": None,
            "domain_features_generated": None,
            "domain_features_kept": None,
            "error": professor_result.get("error", "Unknown error"),
        }


def _run_professor_pipeline(
    train_path: str,
    test_path: str,
    sample_submission_path: str,
    target_column: str,
    id_column: str,
    metric: str,
    task_type: str,
    fast_mode: bool = True,
    ab_test_component: Optional[str] = None,
    time_limit: int = 600,
) -> Dict[str, Any]:
    """
    Run Professor pipeline on prepared data.
    
    This is a placeholder that should be replaced with actual Professor integration.
    For now, returns a dummy submission for testing the simulator infrastructure.
    """
    import polars as pl
    import numpy as np
    
    print(f"    Running Professor (fast={fast_mode})...")
    
    # Load data
    train_df = pl.read_csv(train_path)
    test_df = pl.read_csv(test_path)
    
    # Placeholder: Generate baseline predictions
    # In real implementation, this calls the full Professor pipeline
    
    if task_type == "binary":
        # Simple baseline: predict based on target mean
        target_mean = train_df[target_column].mean()
        predictions = np.full(len(test_df), target_mean)
    elif task_type == "multiclass":
        # Predict most common class
        most_common = train_df[target_column].mode()[0]
        predictions = np.full(len(test_df), most_common)
    else:
        # Regression: predict mean
        target_mean = train_df[target_column].mean()
        predictions = np.full(len(test_df), target_mean)
    
    # Create submission
    submission_df = test_df.select([id_column]).with_columns(
        pl.Series(predictions).alias(target_column)
    )
    
    # Save submission
    submission_path = Path(test_path).parent / "submission.csv"
    submission_df.write_csv(str(submission_path))
    
    # Compute CV score (placeholder)
    if task_type == "binary":
        from sklearn.metrics import accuracy_score
        cv_score = accuracy_score(
            train_df[target_column],
            np.full(len(train_df), target_mean)
        )
    else:
        cv_score = None
    
    return {
        "submission_path": str(submission_path),
        "cv_score": cv_score,
        "winning_model": "baseline",
        "n_features": len([c for c in train_df.columns if c not in (id_column, target_column)]),
        "domain_features_generated": 0,
        "domain_features_kept": 0,
        "error": None,
    }


if __name__ == "__main__":
    main()
