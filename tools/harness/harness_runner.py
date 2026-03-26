"""
Main harness orchestrator. Runs all 5 steps and writes benchmark_report.json.
"""

import os, json, time, traceback
from pathlib import Path
from datetime import datetime

from tools.harness.competition_registry import COMPETITION_REGISTRY
from tools.harness.data_downloader import download_competition, make_split
from tools.harness.scorer import score_predictions
from tools.harness.leaderboard_comparator import compare_to_leaderboard

RESULTS_DIR = Path("tests/harness/results")


def run_harness(
    competition_id: str,
    session_id: str = None,
    fast_mode: bool = False,
    force_download: bool = False,
    force_split: bool = False,
) -> dict:
    if competition_id not in COMPETITION_REGISTRY:
        raise ValueError(
            f"Unknown competition: '{competition_id}'. "
            f"Available: {list(COMPETITION_REGISTRY.keys())}"
        )

    spec       = COMPETITION_REGISTRY[competition_id]
    session_id = session_id or f"harness_{competition_id}_{int(time.time())}"
    result_dir = RESULTS_DIR / session_id
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PROFESSOR HARNESS: {spec.display_name}")
    print(f"Session: {session_id} | Fast: {fast_mode}")
    print(f"{'='*60}\n")

    t_total = time.time()

    # Step 1: Download
    print("[1/5] Downloading competition data...")
    download_competition(competition_id, force=force_download)

    # Step 2: Split
    print("[2/5] Creating 80/20 split...")
    professor_train, private_test, original_test = make_split(
        competition_id, spec, force=force_split
    )

    # Step 3: Run Professor
    print("[3/5] Running Professor pipeline...")
    if fast_mode:
        os.environ["PROFESSOR_FAST_MODE"] = "1"

    import polars as pl
    train_path = result_dir / "professor_train.csv"
    test_path  = result_dir / "professor_test.csv"
    professor_train.write_csv(train_path)
    private_test.drop(spec.target_column).write_csv(test_path)

    from core.state import initial_state
    state = initial_state(
        competition=spec.competition_id,
        data_path=str(train_path),
        budget_usd=5.0,
        task_type=spec.task_type
    )
    # Override harness-specific paths and session
    state.update({
        "session_id":                session_id,
        "train_path":                str(train_path),
        "test_path":                 str(test_path),
        "target_column":             spec.target_column,
        "id_column":                 spec.id_column,
        "evaluation_metric":         spec.evaluation_metric,
    })

    t_pipeline    = time.time()
    final_state   = None
    pipeline_error = None

    try:
        from agents import ml_optimizer
        ml_optimizer.N_OPTUNA_TRIALS = 20
        
        from core.professor import run_professor
        final_state = run_professor(state)
    except Exception as e:
        pipeline_error = f"{type(e).__name__}: {str(e)[:500]}"
        print(f"[harness] Pipeline error: {pipeline_error}")
        traceback.print_exc()

    pipeline_seconds = time.time() - t_pipeline

    # Step 4: Score
    print("[4/5] Scoring against private held-out set...")
    score_result = {"private_score": None, "error": "pipeline_failed"}

    if final_state is not None:
        # Check state for submission_df first
        if "submission_df" in final_state:
            try:
                score_result = score_predictions(private_test, final_state["submission_df"], spec)
            except Exception as e:
                score_result = {"private_score": None, "error": str(e)[:200]}
        else:
            # Fall back to submission CSV on disk
            sub_path = Path(f"outputs/{session_id}/submission.csv")
            if sub_path.exists():
                try:
                    preds = pl.read_csv(sub_path)
                    score_result = score_predictions(private_test, preds, spec)
                except Exception as e:
                    score_result = {"private_score": None, "error": str(e)[:200]}

    # Step 5: Compare to LB
    print("[5/5] Comparing to leaderboard...")
    lb_result = {}
    if score_result.get("private_score") is not None:
        lb_result = compare_to_leaderboard(score_result["private_score"], spec)

    # Build report
    cv_score = final_state.get("cv_mean") if final_state else None
    private  = score_result.get("private_score")
    cv_gap   = round(abs(cv_score - private), 6) if (cv_score and private) else None

    report = {
        "session_id":             session_id,
        "competition_id":         competition_id,
        "competition_name":       spec.display_name,
        "run_timestamp":          datetime.utcnow().isoformat(),
        "private_score":          private,
        "cv_score":               cv_score,
        "cv_lb_gap":              cv_gap,
        "metric":                 spec.evaluation_metric,
        **lb_result,
        "total_runtime_seconds":  round(time.time() - t_total, 1),
        "pipeline_seconds":       round(pipeline_seconds, 1),
        "pipeline_error":         pipeline_error,
        "winning_model_type":     _get(final_state, "winning_model_type"),
        "n_features_final":       _get(final_state, "n_features_final"),
        "n_features_dropped":     (_get(final_state, "stage1_drop_count", 0)
                                   + _get(final_state, "stage2_drop_count", 0)),
        "n_pseudo_label_iterations": _get(final_state, "pseudo_label_iterations", 0),
        "estimated_llm_cost_usd": _estimate_cost(_get(final_state, "total_tokens_used", 0)),
        "optimisation_flags": {
            "PROFESSOR_FAST_MODE":    os.getenv("PROFESSOR_FAST_MODE", "0"),
            "PROFESSOR_DEVICE":       os.getenv("PROFESSOR_DEVICE", "cpu"),
        },
        "n_professor_train_rows": len(professor_train),
        "n_private_test_rows":    len(private_test),
    }

    report_path = result_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    return report


def _get(state, key, default=None):
    return state.get(key, default) if state else default

def _estimate_cost(tokens: int) -> float:
    return round((tokens * 0.80 * 0.05 + tokens * 0.20 * 0.59) / 1_000_000, 4)

def _print_summary(r: dict):
    print(f"\n{'='*60}")
    print(f"HARNESS COMPLETE: {r['competition_name']}")
    print(f"{'='*60}")
    print(f"  Private score:        {r.get('private_score', 'N/A')}")
    print(f"  CV score:             {r.get('cv_score', 'N/A')}")
    print(f"  CV/Private gap:       {r.get('cv_lb_gap', 'N/A')}")
    print(f"  Simulated percentile: {r.get('simulated_percentile', 'N/A')}%")
    print(f"  Simulated medal:      {r.get('simulated_medal', 'N/A')}")
    print(f"  Gap to gold:          {r.get('gap_to_gold', 'N/A')}")
    print(f"  Runtime:              {r.get('total_runtime_seconds')}s")
    print(f"  Winning model:        {r.get('winning_model_type', 'N/A')}")
    print(f"  Features (final):     {r.get('n_features_final', 'N/A')}")
    print(f"  Est. LLM cost:        ${r.get('estimated_llm_cost_usd', 0)}")
    if r.get("pipeline_error"):
        print(f"\n  ERROR: {r['pipeline_error']}")
    print(f"{'='*60}\n")
