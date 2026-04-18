# tools/lightning_runner.py
"""
Lightning AI compute offload for Professor Agent.

Core pattern:
  - Local machine sends: data + Python script
  - Lightning runs: heavy compute on cloud hardware
  - Lightning returns: small JSON with results
  - Local machine: reads JSON, continues pipeline

Falls back to local execution silently if Lightning is not configured
or if any error occurs. Lightning is always optional — never required.
"""

import os
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 15
JOB_TIMEOUT_SECONDS   = 7200    # 2 hours default


def is_lightning_configured() -> bool:
    """
    Returns True only if all required credentials are set AND
    lightning-sdk is installed.
    Returns False silently otherwise — never raises.
    """
    required = [
        "LIGHTNING_API_KEY",
        "LIGHTNING_USER_ID",
        "LIGHTNING_USERNAME",
        "LIGHTNING_STUDIO_NAME",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        return False
    try:
        import lightning_sdk
        return True
    except ImportError:
        logger.warning(
            "[lightning] lightning-sdk not installed. "
            "Run: pip install lightning-sdk"
        )
        return False


def run_on_lightning(
    script: str,
    args: dict,
    job_name: str,
    machine: str = "CPU",
    interruptible: bool = True,
    result_path: str = None,
    timeout: int = JOB_TIMEOUT_SECONDS,
) -> dict:
    """
    Submits a job to Lightning AI and waits for the result JSON.

    Args:
        script:        Path to Python script relative to project root.
                       This script runs ON Lightning's machine.
        args:          Dict passed to script as CLI args: --key value
        job_name:      Unique name for tracking (shown in Lightning dashboard)
        machine:       "CPU" | "L4" | "L40S" | "A10G"
                       CPU is cheapest. L4 is best for LightGBM.
        interruptible: True = spot instance (80% cheaper).
                       Use True for Optuna, null importance, stability.
                       Use False for harness (full pipeline runs).
        result_path:   Local path where the script writes its JSON result.
                       Professor reads this after the job completes.
        timeout:       Max seconds to wait. Default 2 hours.

    Returns dict with:
        success:    bool
        result:     dict loaded from result_path, or {} on failure
        job_link:   str — Lightning dashboard URL for this job
        runtime_s:  float — wall clock seconds
        error:      str — empty if success

    NEVER raises. Returns success=False with error message on any failure.
    The caller always checks success and falls back to local execution.
    """
    from lightning_sdk import Studio, Machine

    MACHINE_MAP = {
        "CPU":  Machine.CPU_SMALL,
        "L4":   Machine.L4,
        "L40S": Machine.L40S,
        "A10G": Machine.A10G,
    }

    t_start = time.time()

    try:
        machine_obj  = MACHINE_MAP.get(machine, Machine.CPU_SMALL)
        studio_name  = os.environ["LIGHTNING_STUDIO_NAME"]
        teamspace    = os.getenv("LIGHTNING_TEAMSPACE") or None

        logger.info(f"[lightning] Connecting to Studio: {studio_name}")
        studio = Studio(name=studio_name, teamspace=teamspace)
        studio.start()

        # Build command string
        args_str = " ".join(f"--{k} {v}" for k, v in args.items())
        command  = f"python {script} {args_str}"

        logger.info(
            f"[lightning] Submitting: {job_name}\n"
            f"  Machine:       {machine} (spot={interruptible})\n"
            f"  Command:       {command}"
        )

        from lightning_sdk import Job
        job = Job.run(
            command=command,
            name=job_name,
            machine=machine_obj,
            studio=studio,
            interruptible=interruptible,
        )

        logger.info(f"[lightning] Job started. Track: {job.link}")

        # Poll until complete or timeout
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = job.status

            if status == "completed":
                break
            elif status in ("failed", "stopped", "cancelled"):
                return {
                    "success":   False,
                    "result":    {},
                    "job_link":  job.link,
                    "runtime_s": round(time.time() - t_start, 1),
                    "error":     f"Job {status}: {job_name}",
                }

            remaining = int(deadline - time.time())
            logger.info(
                f"[lightning] Waiting... ({remaining}s remaining, status={status})"
            )
            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            return {
                "success":   False,
                "result":    {},
                "job_link":  getattr(job, "link", ""),
                "runtime_s": round(time.time() - t_start, 1),
                "error":     f"Timed out after {timeout}s: {job_name}",
            }

        # Download and read result JSON
        result = {}
        if result_path:
            local_path = Path(result_path)
            # Download from Lightning Studio to local path
            try:
                studio.download_file(
                    remote_path=f"/home/{args.get('session_id', 'unknown')}/{Path(result_path).name}",
                    local_path=str(local_path),
                )
            except Exception:
                pass  # File may already be local if Studio is mounted

            if local_path.exists():
                result = json.loads(local_path.read_text())
                logger.info(f"[lightning] Result loaded: {result_path}")
            else:
                logger.warning(f"[lightning] Result file not found: {result_path}")

        return {
            "success":   True,
            "result":    result,
            "job_link":  job.link,
            "runtime_s": round(time.time() - t_start, 1),
            "error":     "",
        }

    except Exception as e:
        logger.warning(
            f"[lightning] Job failed: {e}. "
            "Falling back to local execution."
        )
        return {
            "success":   False,
            "result":    {},
            "job_link":  "",
            "runtime_s": round(time.time() - t_start, 1),
            "error":     str(e),
        }


def sync_files_to_lightning(
    session_id: str,
    files: dict,
) -> bool:
    """
    Uploads files to Lightning Studio persistent storage.

    Args:
        session_id: Professor session ID — used as the remote directory name
        files: dict of {local_path: remote_filename}
               Remote files land at /home/{session_id}/{remote_filename}

    Returns True on success, False on failure. Never raises.

    Lightning's /home directory is persistent — files survive job restarts.
    This is why spot instances are safe: data is already there when job restarts.
    """
    try:
        from lightning_sdk import Studio
        studio = Studio(
            name=os.environ["LIGHTNING_STUDIO_NAME"],
            teamspace=os.getenv("LIGHTNING_TEAMSPACE") or None,
        )
        studio.start()

        for local_path, remote_name in files.items():
            remote_path = f"/home/{session_id}/{remote_name}"
            studio.upload_file(
                local_path=str(local_path),
                remote_path=remote_path,
            )
            logger.info(f"[lightning] Uploaded: {local_path} → {remote_path}")

        return True

    except Exception as e:
        logger.warning(f"[lightning] File sync failed: {e}")
        return False
