"""
Professor Cloud Benchmark — Modal App

Run all competitions in parallel on Modal's serverless infrastructure.
Each competition runs in its own container. Wall clock = slowest single competition.

Usage:
    modal run simulator/cloud_benchmark.py                              # fast, all comps
    modal run simulator/cloud_benchmark.py --mode deep                  # deep, all comps
    modal run simulator/cloud_benchmark.py --competition spaceship-titanic  # single comp
"""

import modal
import json
import time
from pathlib import Path
from datetime import datetime

# ── Modal Image: built once, cached ──
# Only upload simulator/ and core/ to minimize upload time
professor_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "lightgbm>=4.0", "xgboost>=2.0", "catboost>=1.2",
        "scikit-learn>=1.3", "optuna>=3.0", "polars>=1.0",
        "pyarrow>=14.0", "scipy>=1.11", "numpy>=1.24",
        "chromadb>=0.5", "sentence-transformers>=2.0",
        "openai>=1.0", "google-generativeai>=0.5",
        "kaggle>=1.6", "psutil>=5.9", "mlflow>=2.0",
        "langgraph>=0.2", "python-dotenv>=1.0",
    )
    .add_local_dir("simulator", remote_path="/root/professor-agent/simulator")
    .add_local_dir("core", remote_path="/root/professor-agent/core")
    .add_local_dir("agents", remote_path="/root/professor-agent/agents")
    .add_local_dir("adapters", remote_path="/root/professor-agent/adapters")
    .add_local_dir("guards", remote_path="/root/professor-agent/guards")
    .add_local_dir("memory", remote_path="/root/professor-agent/memory")
    .add_local_dir("tools", remote_path="/root/professor-agent/tools")
)

# ── Persistent Volume: competition data cached across runs ──
data_volume = modal.Volume.from_name("professor-benchmark-data", create_if_missing=True)

# ── Secrets ──
secrets = modal.Secret.from_name("professor-keys")

# ── App ──
app = modal.App("professor-benchmark")


@app.function(
    image=professor_image,
    volumes={"/data": data_volume},
    secrets=[secrets],
    cpu=4.0,
    memory=16384,
    timeout=3600,
)
def run_single_competition(competition_slug: str, mode: str = "fast") -> dict:
    """Run Professor against one competition. Returns result dict."""
    import sys
    import os
    import pickle
    import json
    sys.path.insert(0, "/root/professor-agent")

    # Set up environment from secrets
    os.environ.setdefault("FIREWORKS_API_KEY", "")
    os.environ.setdefault("FIREWORKS_GLM_API_KEY", "")
    os.environ.setdefault("GOOGLE_API_KEY", "")
    os.environ.setdefault("KAGGLE_USERNAME", "")
    os.environ.setdefault("KAGGLE_KEY", "")

    from simulator.competition_registry import get_competition
    from simulator.data_splitter import split_competition_data, ensure_data_cached
    from simulator.leaderboard import SimulatedLeaderboard

    entry = get_competition(competition_slug)

    # Cache data on persistent volume
    cache_dir = "/data/competitions"
    ensure_data_cached(entry, cache_dir)

    # Split (deterministic)
    split_dir = f"/data/splits/{entry.slug}"
    split = split_competition_data(
        data_path=f"{cache_dir}/{entry.slug}/full_data.csv",
        entry=entry,
        output_dir=split_dir,
    )

    # Create leaderboard
    lb = SimulatedLeaderboard(entry, split)

    # Configure for benchmark mode
    fast_config = {
        "optuna_trials": 30,
        "null_importance_shuffles": 5,
        "max_submissions": 1,
        "skip_forum_scrape": True,
    }
    deep_config = {
        "optuna_trials": 200,
        "null_importance_shuffles": 50,
        "max_submissions": 3,
        "skip_forum_scrape": False,
    }
    config = fast_config if mode == "fast" else deep_config

    # Run Professor
    start = time.time()

    try:
        # Import Professor's pipeline
        from core.professor import run_professor
        from core.state import ProfessorState

        session_id = f"benchmark_{entry.slug}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        output_dir = f"/data/outputs/{session_id}"
        os.makedirs(output_dir, exist_ok=True)

        # Create schema.json (what DataEngineer would produce)
        import polars as pl
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

        # Create preprocessor.pkl (minimal - just stores column info)
        preprocessor = {
            "columns": train_df.columns,
            "target": entry.target_column,
            "id_column": entry.id_column,
            "dtypes": {col: str(train_df[col].dtype) for col in train_df.columns},
        }
        preprocessor_path = f"{output_dir}/preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)

        # Create clean_data.csv (just use train.csv as-is for now)
        clean_data_path = f"{output_dir}/clean_data.csv"
        train_df.write_csv(clean_data_path)

        # Build initial state with ALL required fields
        state = {
            "session_id": session_id,
            "competition_name": entry.slug,
            "raw_data_path": str(split.train_path),  # DataEngineer reads this
            "clean_data_path": clean_data_path,
            "schema_path": schema_path,
            "preprocessor_path": preprocessor_path,
            "train_path": split.train_path,
            "test_path": split.test_path,
            "sample_submission_path": split.sample_submission_path,
            "target_col": entry.target_column,  # Integrity gate checks this
            "target_column": entry.target_column,
            "id_column": entry.id_column,
            "id_columns": [entry.id_column],  # Must be a list!
            "metric": entry.metric,
            "task_type": entry.task_type,
            "config": config,
            "output_dir": output_dir,
        }

        # Run pipeline
        result = run_professor(state, timeout_seconds=3000)

        runtime = time.time() - start

        # Submit
        submission_path = result.get("submission_path")
        if submission_path:
            submit_result = lb.submit(submission_path)
        else:
            submit_result = None

        # Reveal scores
        final = lb.competition_end()

        return {
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

    except Exception as e:
        import traceback
        return {
            "slug": entry.slug,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "runtime_seconds": round(time.time() - start, 1),
            "mode": mode,
        }

    finally:
        data_volume.commit()


@app.function(
    image=professor_image,
    timeout=7200,
)
def run_full_benchmark(mode: str = "fast", slugs: list = None) -> dict:
    """Orchestrate all competitions in parallel. Return aggregate report."""
    import sys
    sys.path.insert(0, "/root/professor-agent")
    from simulator.competition_registry import REGISTRY

    if not slugs:
        slugs = [e.slug for e in REGISTRY]

    # Launch all in parallel
    results = list(run_single_competition.map(
        slugs, [mode] * len(slugs)
    ))

    # Aggregate
    successful = [r for r in results if r.get("error") is None]
    failed = [r for r in results if r.get("error") is not None]

    private_scores = [r["private_percentile"] for r in successful if r.get("private_percentile")]
    medals = [r["medal"] for r in successful]

    import numpy as np

    report = {
        "run_id": f"benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "professor_version": "v2.0",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": mode,
        "n_competitions": len(slugs),
        "n_successful": len(successful),
        "n_failed": len(failed),
        "aggregate_metrics": {
            "median_percentile": float(np.median(private_scores)) if private_scores else None,
            "mean_percentile": float(np.mean(private_scores)) if private_scores else None,
            "gold_rate": medals.count("gold") / len(medals) if medals else 0,
            "silver_rate": medals.count("silver") / len(medals) if medals else 0,
            "bronze_rate": medals.count("bronze") / len(medals) if medals else 0,
            "medal_rate": sum(1 for m in medals if m != "none") / len(medals) if medals else 0,
        },
        "per_competition": results,
    }

    return report


@app.local_entrypoint()
def main(
    mode: str = "fast",
    competition: str = "",
):
    """
    CLI entry point.

    Usage:
        modal run simulator/cloud_benchmark.py
        modal run simulator/cloud_benchmark.py --mode deep
        modal run simulator/cloud_benchmark.py --competition spaceship-titanic
    """
    if competition:
        result = run_single_competition.remote(competition, mode)
        print(json.dumps(result, indent=2))
    else:
        report = run_full_benchmark.remote(mode)

        agg = report["aggregate_metrics"]
        print(f"\n{'='*60}")
        print(f"  PROFESSOR BENCHMARK — {report['professor_version']}")
        print(f"  Mode: {mode} | Competitions: {report['n_competitions']}")
        print(f"{'='*60}")

        if agg["median_percentile"]:
            print(f"  Median percentile:  {agg['median_percentile']:.1f}%")
            print(f"  Gold rate:          {agg['gold_rate']:.0%}")
            print(f"  Medal rate:         {agg['medal_rate']:.0%}")

        print(f"\n  Per competition:")
        emojis = {"gold": "🥇", "silver": "🥈", "bronze": "🥉", "none": "  "}
        for c in report["per_competition"]:
            if c.get("error"):
                print(f"  ❌ {c['slug']}: {c['error'][:80]}")
            else:
                m = emojis.get(c.get("medal", "none"), "  ")
                print(f"  {m} {c['slug']:<40} "
                      f"private={c.get('private_score', 'N/A')}  "
                      f"pct={c.get('private_percentile', 'N/A')}%  "
                      f"{c.get('runtime_seconds', 0):.0f}s")

        # Save locally
        Path("simulator/results").mkdir(parents=True, exist_ok=True)
        out = f"simulator/results/{report['run_id']}.json"
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"\n  Report saved: {out}")
