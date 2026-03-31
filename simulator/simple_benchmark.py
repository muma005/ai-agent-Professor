"""
Simple Benchmark — Run 20 trials with basic LightGBM (no agents).

This bypasses the full Professor pipeline and just:
1. Loads data
2. Trains LightGBM with default params
3. Evaluates on test set
4. Submits to simulated leaderboard

Usage:
    python simulator/simple_benchmark.py --competition spaceship-titanic --trials 20
"""

import argparse
import json
import os
import sys
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.competition_registry import REGISTRY, get_competition, CompetitionEntry
from simulator.data_splitter import split_competition_data, ensure_data_cached
from simulator.leaderboard import SimulatedLeaderboard


def run_single_trial(
    entry: CompetitionEntry,
    trial_num: int,
    base_output_dir: str = "simulator/results/simple_benchmark"
) -> dict:
    """Run simple LightGBM benchmark for one trial."""

    print(f"\n{'='*70}")
    print(f"  TRIAL {trial_num}: {entry.slug}")
    print(f"{'='*70}")

    start = time.time()

    # Cache data
    cache_dir = "simulator/data/competitions"
    try:
        ensure_data_cached(entry, cache_dir)
    except Exception as e:
        print(f"[WARNING] Could not cache data: {e}")

    # Split data
    split_dir = f"simulator/data/splits/{entry.slug}"
    data_path = f"{cache_dir}/{entry.slug}/full_data.csv"

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
            "runtime_seconds": time.time() - start,
        }

    # Create leaderboard
    lb = SimulatedLeaderboard(entry, split)

    try:
        import polars as pl
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        import lightgbm as lgb

        # Load train data
        train_df = pl.read_csv(split.train_path)
        test_df = pl.read_csv(split.test_path)

        target = entry.target_column
        id_col = entry.id_column

        # Get features (exclude target and ID)
        feature_cols = [c for c in train_df.columns if c not in [target, id_col]]

        # Encode categoricals
        for col in feature_cols:
            if train_df[col].dtype in [pl.Utf8, pl.Categorical, pl.Boolean]:
                # Label encode
                unique_vals = train_df[col].unique().to_list()
                val_to_idx = {v: i for i, v in enumerate(unique_vals)}
                train_df = train_df.with_columns(
                    pl.col(col).replace_strict(val_to_idx, default=None).alias(f"{col}_enc")
                )
                test_df = test_df.with_columns(
                    pl.col(col).replace_strict(val_to_idx, default=-1).alias(f"{col}_enc")
                )
                feature_cols[feature_cols.index(col)] = f"{col}_enc"

        # Prepare data
        X = train_df[feature_cols].to_numpy()
        y = train_df[target].to_numpy()
        X_test = test_df[feature_cols].to_numpy()

        # Handle NaN
        X = np.nan_to_num(X, nan=-999)
        X_test = np.nan_to_num(X_test, nan=-999)

        # Simple CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_seed=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=-1
            )
            model.fit(X_tr, y_tr)

            y_pred = model.predict_proba(X_val)[:, 1]
            fold_score = roc_auc_score(y_val, y_pred)
            cv_scores.append(fold_score)

        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Train final model
        final_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        final_model.fit(X, y)

        # Predict on test
        test_pred = final_model.predict_proba(X_test)[:, 1]

        # Create submission
        submission_df = test_df.select(id_col).with_columns(
            pl.lit(test_pred).alias(target)
        )

        # Save submission
        output_dir = f"{base_output_dir}/trial_{entry.slug}_{trial_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        submission_path = f"{output_dir}/submission.csv"
        submission_df.write_csv(submission_path)

        # Save model
        model_path = f"{output_dir}/model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(final_model, f)

        # Submit to leaderboard
        submit_result = lb.submit(submission_path)
        final = lb.competition_end()

        runtime = time.time() - start

        result_dict = {
            "trial": trial_num,
            "slug": entry.slug,
            "task_type": entry.task_type,
            "metric": entry.metric,
            "cv_score": cv_mean,
            "cv_std": cv_std,
            "public_score": final.best_public_score,
            "private_score": final.best_private_score,
            "public_percentile": final.public_rank_pct,
            "private_percentile": final.private_rank_pct,
            "shakeup": final.shakeup_positions,
            "medal": final.medal,
            "runtime_seconds": round(runtime, 1),
            "error": None,
        }

        print(f"\n{'='*70}")
        print(f"  TRIAL {trial_num} COMPLETE: {entry.slug}")
        print(f"{'='*70}")
        print(f"  CV Score:           {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"  Public Score:       {result_dict['public_score']}")
        print(f"  Private Score:      {result_dict['private_score']}")
        print(f"  Public Percentile:  {result_dict['public_percentile']}%")
        print(f"  Private Percentile: {result_dict['private_percentile']}%")
        print(f"  Shakeup:            {result_dict['shakeup']:+.1f} positions")
        print(f"  Medal:              {result_dict['medal'].upper()}")
        print(f"  Runtime:            {result_dict['runtime_seconds']:.1f}s")
        print(f"{'='*70}\n")

        # Save result
        result_path = f"{output_dir}/trial_result.json"
        with open(result_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

        return result_dict

    except Exception as e:
        import traceback
        runtime = time.time() - start
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
        }


def run_full_benchmark(
    trials: int = 20,
    competition_slug: Optional[str] = None,
) -> dict:
    """Run full benchmark."""

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
            result = run_single_trial(entry, trial_num)
            all_results.append(result)

            # Progress
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
        "run_id": f"simple_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "trials_per_competition": trials,
        "n_competitions": len(slugs),
        "n_successful": len(successful),
        "n_failed": len(failed),
        "total_runtime_seconds": round(total_runtime, 1),
        "aggregate_metrics": {
            "median_percentile": float(np.median(private_scores)) if private_scores else None,
            "mean_percentile": float(np.mean(private_scores)) if private_scores else None,
            "std_percentile": float(np.std(private_scores)) if private_scores else None,
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
        description="Simple Benchmark — LightGBM baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python simulator/simple_benchmark.py --competition spaceship-titanic --trials 20
        """
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
        trials=args.trials,
        competition_slug=args.competition if args.competition else None,
    )

    # Print summary
    agg = report["aggregate_metrics"]
    print(f"\n{'='*70}")
    print(f"  SIMPLE BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"  Competitions:         {report['n_competitions']}")
    print(f"  Trials per comp:      {args.trials}")
    print(f"  Total runs:           {report['n_successful'] + report['n_failed']}")
    print(f"  Successful:           {report['n_successful']}")
    print(f"  Failed:               {report['n_failed']}")
    print(f"  Total runtime:        {report['total_runtime_seconds']:.1f}s ({report['total_runtime_seconds']/60:.1f} min)")
    print(f"{'='*70}")

    if agg["median_percentile"]:
        print(f"\n  AGGREGATE METRICS:")
        print(f"  Median percentile:  {agg['median_percentile']:.1f}%")
        print(f"  Mean percentile:    {agg['mean_percentile']:.1f}%")
        print(f"  Std deviation:      {agg['std_percentile']:.1f}%")
        print(f"  Gold rate:          {agg['gold_rate']:.0%}")
        print(f"  Medal rate:         {agg['medal_rate']:.0%}")

    if report["per_competition_summary"]:
        print(f"\n  PER-COMPETITION SUMMARY:")
        for slug, stats in report["per_competition_summary"].items():
            print(f"\n  {slug}:")
            print(f"    Trials:     {stats['n_trials']}")
            print(f"    Median pct: {stats['median_percentile']:.1f}% (±{stats['std_percentile']:.1f}%)")
            print(f"    Medals:     🥇{stats['gold']} 🥈{stats['silver']} 🥉{stats['bronze']} ({stats['none']} no medal)")

    # Save report
    Path("simulator/results/simple_benchmark").mkdir(parents=True, exist_ok=True)
    out = f"simulator/results/simple_benchmark/{report['run_id']}.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {out}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
