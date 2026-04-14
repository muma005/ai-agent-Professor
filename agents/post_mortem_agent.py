# agents/post_mortem_agent.py
# -------------------------------------------------------------------------
# Day 27: Post-Mortem Agent
# Runs manually after a competition closes and private LB score is revealed.
# Never called automatically by the pipeline.
#
# Usage:
#   python -m professor post-mortem --session abc123 --lb-score 0.8073 --lb-rank 150 --total-teams 1000
#
# Outputs:
#   outputs/{session_id}/competition_memory.json
#   ChromaDB pattern updates
#   critic_failure_patterns (if critic missed a leak)
# -------------------------------------------------------------------------

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Helper ─────────────────────────────────────────────────────────

def _load_json(path: Path, default=None):
    """Load JSON from path, returning default on any error."""
    try:
        return json.loads(path.read_text()) if path.exists() else (default or {})
    except Exception:
        return default or {}


def _compute_confidence(strategy_eval: dict) -> float:
    """
    Confidence based on:
    - CV/LB gap (small gap = high confidence)
    - LB percentile (top 10% = high confidence)
    """
    percentile = float(strategy_eval.get("percentile") or 50.0)
    cv_lb_gap = float(strategy_eval.get("cv_lb_gap") or 0.010)

    gap_factor = max(0.0, 1.0 - cv_lb_gap / 0.030)
    percentile_factor = min(1.0, percentile / 100.0)

    return round(min(0.90, 0.40 + 0.30 * gap_factor + 0.20 * percentile_factor), 3)


# ── Section 1: Solution Autopsy ───────────────────────────────────

def _build_solution_autopsy(session_dir: Path, lb_score: float) -> dict:
    """
    Analyses which features contributed to the result.

    For each feature that survived null importance filtering:
      - Was it validated by the critic? (not in critic_verdict findings)
      - Is it stable? (fold importance variance < 0.02)
      - Classification: "contributed" | "noise" | "unknown"

    For each feature that was dropped:
      - What stage dropped it? Stage 1 or Stage 2 null importance?
      - Was the cv_delta positive (we lost something) or negative (right call)?
    """
    metrics = _load_json(session_dir / "metrics.json")
    null_result = _load_json(session_dir / "null_importance_result.json", default={})
    critic = _load_json(session_dir / "critic_verdict.json", default={})
    feat_imp = _load_json(session_dir / "feature_importance.json", default={})

    # Features that survived
    survived = metrics.get("feature_order", [])
    dropped_s1 = null_result.get("dropped_stage1", [])
    dropped_s2 = null_result.get("dropped_stage2", [])

    # Critic findings — which features were flagged
    flagged_by_critic = set()
    for finding in critic.get("findings", []):
        feat = finding.get("feature_flagged", "")
        if feat:
            flagged_by_critic.add(feat)

    feature_audit = []
    for feat in survived:
        importance = float(feat_imp.get(feat, 0.0))
        flagged = feat in flagged_by_critic
        feature_audit.append({
            "feature": feat,
            "status": "survived_null_importance",
            "importance": round(importance, 6),
            "flagged_by_critic": flagged,
            "classification": "noise" if flagged else ("contributed" if importance > 0 else "unknown"),
        })

    for feat in dropped_s1:
        feature_audit.append({
            "feature": feat,
            "status": "dropped_stage1",
            "importance": 0.0,
            "flagged_by_critic": False,
            "classification": "correctly_pruned",
        })

    for feat in dropped_s2:
        feature_audit.append({
            "feature": feat,
            "status": "dropped_stage2",
            "importance": 0.0,
            "flagged_by_critic": feat in flagged_by_critic,
            "classification": "correctly_pruned",
        })

    return {
        "total_features_trained": len(survived),
        "total_dropped_stage1": len(dropped_s1),
        "total_dropped_stage2": len(dropped_s2),
        "features_flagged_by_critic": len(flagged_by_critic),
        "feature_audit": feature_audit,
    }


# ── Section 2: Strategy Evaluation ────────────────────────────────

def _build_strategy_evaluation(
    session_dir: Path,
    lb_score: float,
    lb_rank=None,
    total_teams=None,
) -> dict:
    """
    Evaluates what strategy choices worked and what did not.
    """
    metrics = _load_json(session_dir / "metrics.json")
    ensemble = _load_json(session_dir / "ensemble_selection.json", default={})

    cv_mean = float(metrics.get("cv_mean", 0.0))
    cv_std = float(metrics.get("cv_std", 0.0))
    cv_lb_gap = round(abs(cv_mean - lb_score), 6)

    percentile = None
    if lb_rank and total_teams:
        percentile = round(100 * (1 - lb_rank / total_teams), 2)

    # Was ensemble better than single model?
    ensemble_accepted = ensemble.get("ensemble_accepted", False)
    ensemble_holdout_score = float(ensemble.get("ensemble_holdout_score") or 0.0)
    n_models_in_ensemble = len(ensemble.get("selected_models", []))

    # CV/LB gap root cause classification
    critic_verdict = _load_json(session_dir / "critic_verdict.json", default={})
    critic_ok = critic_verdict.get("overall_severity") == "OK"

    if cv_lb_gap > 0.020 and critic_ok:
        gap_root_cause = "critic_missed"
    elif cv_lb_gap > 0.020 and not critic_ok:
        gap_root_cause = "known_risk"
    elif cv_std > 0.020:
        gap_root_cause = "high_variance"
    else:
        gap_root_cause = "acceptable"

    return {
        "cv_mean": round(cv_mean, 6),
        "cv_std": round(cv_std, 6),
        "lb_score": round(lb_score, 6),
        "cv_lb_gap": cv_lb_gap,
        "gap_root_cause": gap_root_cause,
        "lb_rank": lb_rank,
        "total_teams": total_teams,
        "percentile": percentile,
        "ensemble_accepted": ensemble_accepted,
        "ensemble_holdout_score": round(ensemble_holdout_score, 6),
        "n_models_in_ensemble": n_models_in_ensemble,
        "winning_model_type": metrics.get("winning_model_type", "unknown"),
    }


# ── Section 3: Structured Memory Writes ───────────────────────────

def _build_memory_writes(
    session_dir: Path,
    strategy_eval: dict,
    solution_autopsy: dict,
    state: dict,
) -> list[dict]:
    """
    Generates structured memory entries to write to ChromaDB.
    """
    brief = _load_json(session_dir / "competition_brief.json", default={})
    domain = brief.get("domain", "tabular")
    lb_gap = strategy_eval["cv_lb_gap"]
    cv = strategy_eval["cv_mean"]
    lb = strategy_eval["lb_score"]

    writes = []

    # Feature-level findings
    for item in solution_autopsy.get("feature_audit", []):
        if item["classification"] == "contributed" and item["importance"] > 0.01:
            writes.append({
                "domain": domain,
                "feature": item["feature"],
                "cv_delta": item["importance"],
                "private_lb_delta": item["importance"] * (lb / cv) if cv > 0 else 0.0,
                "validated": not item["flagged_by_critic"] and lb_gap < 0.010,
                "reusable": True,
                "confidence": _compute_confidence(strategy_eval),
                "finding_type": "feature",
            })
        elif item["classification"] == "noise" and item["flagged_by_critic"]:
            writes.append({
                "domain": domain,
                "feature": item["feature"],
                "cv_delta": 0.0,
                "private_lb_delta": 0.0,
                "validated": True,
                "reusable": True,
                "confidence": 0.80,
                "finding_type": "pitfall",
                "note": "Flagged by critic as noise or leakage. Avoid in future.",
            })

    # Strategy findings
    if strategy_eval["ensemble_accepted"] and strategy_eval["ensemble_holdout_score"] > 0:
        writes.append({
            "domain": domain,
            "feature": "diversity_ensemble",
            "cv_delta": 0.0,
            "private_lb_delta": 0.0,
            "validated": lb_gap < 0.010,
            "reusable": True,
            "confidence": _compute_confidence(strategy_eval),
            "finding_type": "strategy",
            "note": f"Diversity ensemble accepted, holdout={strategy_eval['ensemble_holdout_score']:.5f}",
        })

    # Model choice finding
    winning_model = strategy_eval.get("winning_model_type", "unknown")
    if winning_model != "unknown":
        writes.append({
            "domain": domain,
            "feature": f"model_type_{winning_model}",
            "cv_delta": 0.0,
            "private_lb_delta": 0.0,
            "validated": lb_gap < 0.010,
            "reusable": True,
            "confidence": _compute_confidence(strategy_eval),
            "finding_type": "model_choice",
            "note": f"{winning_model} won with cv={cv:.5f}, lb={lb:.5f}",
        })

    return writes


# ── Section 4: Critic Calibration ─────────────────────────────────

def _build_critic_calibration(session_dir: Path, lb_score: float, cv_mean: float) -> dict:
    """
    Evaluates critic accuracy: how many CRITICAL verdicts were correct vs false positives.
    Adjusts recommended sensitivity thresholds for future runs.

    A CRITICAL verdict is "correct" if the CV/LB gap is large (gap > 0.010).
    A CRITICAL verdict is a "false positive" if CV and LB are well-aligned (gap <= 0.005).
    """
    critic = _load_json(session_dir / "critic_verdict.json", default={})
    gap = abs(cv_mean - lb_score)

    overall_severity = critic.get("overall_severity", "unchecked")
    findings = critic.get("findings", [])

    n_critical = sum(1 for f in findings if f.get("severity") == "CRITICAL")
    n_high = sum(1 for f in findings if f.get("severity") == "HIGH")

    # Was the critic right?
    critic_fired = overall_severity in ("CRITICAL", "HIGH")
    gap_was_large = gap > 0.010

    if critic_fired and gap_was_large:
        calibration_verdict = "true_positive"
    elif critic_fired and not gap_was_large:
        calibration_verdict = "false_positive"
    elif not critic_fired and gap_was_large:
        calibration_verdict = "false_negative"
    else:
        calibration_verdict = "true_negative"

    # Threshold adjustment recommendation
    if calibration_verdict == "false_positive":
        threshold_recommendation = "consider_raising_thresholds"
        threshold_note = (
            "Critic fired but CV/LB gap was small. "
            "ECE threshold or adversarial AUC threshold may be too sensitive."
        )
    elif calibration_verdict == "false_negative":
        threshold_recommendation = "consider_lowering_thresholds"
        threshold_note = (
            "Critic did not fire but CV/LB gap was large. "
            "A leakage pattern was missed. "
            "This finding is written to critic_failure_patterns."
        )
    else:
        threshold_recommendation = "thresholds_appropriate"
        threshold_note = "Critic verdict consistent with private LB outcome."

    return {
        "overall_severity": overall_severity,
        "n_critical_findings": n_critical,
        "n_high_findings": n_high,
        "cv_lb_gap": round(gap, 6),
        "calibration_verdict": calibration_verdict,
        "threshold_recommendation": threshold_recommendation,
        "threshold_note": threshold_note,
        "vectors_checked": critic.get("vectors_checked", []),
    }


# ── ChromaDB / Critic Failure writes ──────────────────────────────

def _write_to_chromadb(report: dict, session_id: str) -> None:
    """Writes memory_writes to ChromaDB professor_patterns_v2 collection."""
    try:
        from memory.memory_schema import store_pattern, fingerprint_to_text

        brief_path = Path(f"outputs/{session_id}/competition_brief.json")
        if not brief_path.exists():
            logger.warning("[post_mortem] competition_brief.json missing — ChromaDB write skipped.")
            return

        brief = json.loads(brief_path.read_text())

        # Build a simple fingerprint from the brief
        fingerprint = {
            "task_type": brief.get("task_type", "binary_classification"),
            "n_rows_bucket": brief.get("n_rows_bucket", "medium"),
            "imbalance_ratio": brief.get("imbalance_ratio", 0.5),
        }

        validated = [
            w for w in report["memory_writes"]
            if w.get("validated") and w.get("finding_type") == "feature"
        ]
        failed = [
            w["feature"] for w in report["memory_writes"]
            if w.get("finding_type") == "pitfall"
        ]

        if validated or failed:
            store_pattern(
                fingerprint=fingerprint,
                validated_approaches=[{"approach": w["feature"], "cv_improvement": w.get("cv_delta", 0)} for w in validated],
                failed_approaches=failed,
                competition_name=brief.get("competition_name", session_id),
                confidence=report["strategy_evaluation"].get("cv_mean", 0.70),
            )
            logger.info(
                f"[post_mortem] ChromaDB: {len(validated)} validated approaches, "
                f"{len(failed)} failed approaches written."
            )

    except Exception as e:
        logger.warning(f"[post_mortem] ChromaDB write failed: {e}")


def _write_critic_failure_pattern(
    session_dir: Path,
    strategy_eval: dict,
    metrics: dict,
) -> None:
    """Writes to critic_failure_patterns when critic missed a large CV/LB gap."""
    try:
        from memory.memory_schema import store_critic_failure_pattern
        brief_path = session_dir / "competition_brief.json"
        if not brief_path.exists():
            return
        brief = json.loads(brief_path.read_text())
        store_critic_failure_pattern(
            fingerprint=brief.get("fingerprint", {}),
            missed_issue="critic_missed_cv_lb_gap",
            competition_name=brief.get("competition_name", "unknown"),
            feature_flagged="unknown_leakage",
            failure_mode="critic_missed_cv_lb_gap",
            cv_lb_gap=strategy_eval["cv_lb_gap"],
            confidence=0.70,
        )
    except Exception as e:
        logger.warning(f"[post_mortem] critic_failure_pattern write failed: {e}")


# ── Main entry point ──────────────────────────────────────────────

def run_post_mortem_agent(
    session_id: str,
    lb_score: float,
    lb_rank=None,
    total_teams=None,
) -> dict:
    """
    Main entry point for post-mortem analysis.
    Reads all data from outputs/{session_id}/.
    Writes competition_memory.json and updates ChromaDB.
    Never raises — returns error dict on failure.
    """
    session_dir = Path(f"outputs/{session_id}")
    if not session_dir.exists():
        return {"error": f"Session directory not found: {session_dir}"}

    metrics_path = session_dir / "metrics.json"
    if not metrics_path.exists():
        return {"error": f"metrics.json not found in {session_dir}. Did the pipeline complete?"}

    metrics = _load_json(metrics_path)
    cv_mean = float(metrics.get("cv_mean", 0.0))

    # Build the four sections
    solution_autopsy = _build_solution_autopsy(session_dir, lb_score)
    strategy_evaluation = _build_strategy_evaluation(session_dir, lb_score, lb_rank, total_teams)
    memory_writes = _build_memory_writes(session_dir, strategy_evaluation, solution_autopsy, {})
    critic_calibration = _build_critic_calibration(session_dir, lb_score, cv_mean)

    # Compose the full report
    report = {
        "session_id": session_id,
        "lb_score": round(lb_score, 6),
        "lb_rank": lb_rank,
        "total_teams": total_teams,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "solution_autopsy": solution_autopsy,
        "strategy_evaluation": strategy_evaluation,
        "memory_writes": memory_writes,
        "critic_calibration": critic_calibration,
        "n_memory_writes": len(memory_writes),
    }

    # Write competition_memory.json
    output_path = session_dir / "competition_memory.json"
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(f"[post_mortem] competition_memory.json written to {output_path}")

    # Write to ChromaDB
    _write_to_chromadb(report, session_id)

    # Write to critic_failure_patterns if critic missed a leak
    if critic_calibration["calibration_verdict"] == "false_negative":
        _write_critic_failure_pattern(session_dir, strategy_evaluation, metrics)

    logger.info(
        f"[post_mortem] Complete: {len(memory_writes)} memory writes, "
        f"critic={critic_calibration['calibration_verdict']}, "
        f"gap_root_cause={strategy_evaluation['gap_root_cause']}"
    )

    return report
