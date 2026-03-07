# tools/mlflow_tracker.py
#
# Thin MLflow wrapper for experiment tracking.
# Graceful fallback: if MLflow is not installed, prints summary to stdout.
# MLflow failure must NEVER crash the pipeline.
#
# Setup:
#   pip install mlflow
#   mlflow ui --port 5000
#   View at: http://localhost:5000
#   Set MLFLOW_TRACKING_URI in .env to persist across sessions

import os

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def log_run(
    session_id: str,
    competition: str,
    model_type: str,
    params: dict,
    cv_mean: float,
    cv_std: float,
    n_features: int,
    data_hash: str,
) -> None:
    """
    Log a training run to MLflow.

    If MLflow is not available or fails, prints a summary and returns.
    Never raises -- MLflow failure must never crash the pipeline.
    """
    if not MLFLOW_AVAILABLE:
        print(f"[MLflow] (not installed) Run: {model_type} | "
              f"CV={cv_mean:.4f} +/- {cv_std:.4f} | {n_features} features")
        return

    try:
        mlflow.set_experiment(competition)
        with mlflow.start_run(run_name=session_id):
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("competition", competition)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("data_hash", data_hash)
            mlflow.log_param("n_features", n_features)

            for k, v in params.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass  # skip un-loggable params

            mlflow.log_metric("cv_mean", cv_mean)
            mlflow.log_metric("cv_std", cv_std)

        print(f"[MLflow] Logged run: {model_type} | "
              f"CV={cv_mean:.4f} +/- {cv_std:.4f}")
    except Exception as e:
        print(f"[MLflow] WARNING: failed to log run ({e}). Continuing.")


def log_submission(
    session_id: str,
    submission_path: str,
    cv_mean: float,
    lb_score: float = None,
) -> None:
    """
    Log a submission event to MLflow.

    If MLflow is not available or fails, prints a summary and returns.
    Never raises.
    """
    if not MLFLOW_AVAILABLE:
        print(f"[MLflow] (not installed) Submission: CV={cv_mean:.4f} "
              f"LB={lb_score if lb_score else 'pending'}")
        return

    try:
        with mlflow.start_run(run_name=f"{session_id}_submission"):
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("submission_path", submission_path)
            mlflow.log_metric("cv_mean", cv_mean)
            if lb_score is not None:
                mlflow.log_metric("lb_score", lb_score)

        print(f"[MLflow] Logged submission: CV={cv_mean:.4f} "
              f"LB={lb_score if lb_score else 'pending'}")
    except Exception as e:
        print(f"[MLflow] WARNING: failed to log submission ({e}). Continuing.")
