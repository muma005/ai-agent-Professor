# agents/post_mortem_agent.py
# -------------------------------------------------------------------------
# Day 11 — Post-Mortem Agent
# Runs after a competition closes and private LB score is known.
# Manual trigger — not part of the main pipeline.
# Three analyses: CV/LB gap root cause, feature retrospective, memory write.
# -------------------------------------------------------------------------

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def run_post_mortem(
    session_id: str,
    lb_score: float,
    lb_rank: Optional[int] = None,
    total_competitors: Optional[int] = None,
) -> dict:
    """
    Run post-mortem analysis on a completed competition session.

    Loads session artifacts from outputs/{session_id}/, performs 3 analyses,
    writes patterns to ChromaDB, and produces post_mortem_report.json.

    Returns the report dict.
    """
    if lb_score is None:
        raise ValueError(
            "lb_score is required for post-mortem analysis. "
            "Pass the private LB score from the competition results page."
        )

    output_dir = f"outputs/{session_id}"
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Session directory not found: {output_dir}. "
            f"Run the competition pipeline first to generate outputs."
        )

    # -- Load session artifacts --------------------------------------------------
    def _load_json(filename: str, default=None):
        path = f"{output_dir}/{filename}"
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default

    validation_strategy = _load_json("validation_strategy.json", {})
    critic_verdict      = _load_json("critic_verdict.json", {})
    feature_importance  = _load_json("feature_importance.json", {})
    competition_fp      = _load_json("competition_fingerprint.json", {})

    # Derive competition name from session_id
    competition_name = session_id.rsplit("_", 1)[0] if "_" in session_id else session_id

    # -- Analysis 1: CV/LB Gap Root Cause ----------------------------------------
    cv_mean = validation_strategy.get("cv_mean", 0.0)
    cv_std  = validation_strategy.get("cv_std", 0.0)
    gap     = abs(cv_mean - lb_score)

    critic_severity = critic_verdict.get("overall_severity", "unchecked")

    gap_root_cause, gap_explanation = _classify_gap(
        gap, cv_std, critic_severity, cv_mean, lb_score,
    )

    # -- Analysis 2: Feature Retrospective ---------------------------------------
    feature_retrospective = _build_feature_retrospective(
        feature_importance=feature_importance,
        critic_verdict=critic_verdict,
        features_dropped=[],  # loaded from state if available
        gap=gap,
    )

    # -- Analysis 3: Pattern Extraction + Memory Write ---------------------------
    validated = [
        {"approach": f["feature"], "cv_improvement": f.get("importance", 0.0),
         "competitions": [competition_name]}
        for f in feature_retrospective if f["verdict"] == "helped"
    ]
    failed = [
        {"approach": f["feature"], "cv_degradation": abs(f.get("importance", 0.0)),
         "competitions": [competition_name]}
        for f in feature_retrospective if f["verdict"] == "hurt"
    ]

    # Confidence grows with LB performance
    if lb_rank and total_competitors and total_competitors > 0:
        percentile = 1.0 - (lb_rank / total_competitors)
    else:
        percentile = 0.5
    confidence = min(0.9, 0.4 + percentile * 0.5)

    pattern_id = None
    critic_failures_written = 0

    try:
        from memory.memory_schema import store_pattern, store_critic_failure_pattern

        if competition_fp:
            pattern_id = store_pattern(
                fingerprint=competition_fp,
                validated_approaches=validated,
                failed_approaches=failed,
                competition_name=competition_name,
                confidence=confidence,
                cv_lb_gap=gap,
            )

        # If critic missed leakage, write to critic_failure_patterns
        if gap_root_cause == "critic_missed" and competition_fp:
            suspected = _find_suspected_feature(feature_retrospective)
            store_critic_failure_pattern(
                fingerprint=competition_fp,
                missed_issue=(
                    f"CV/LB gap {gap:.3f} not caught by critic. "
                    f"Suspected: {suspected}."
                ),
                competition_name=competition_name,
            )
            critic_failures_written = 1

    except Exception as e:
        logger.warning(f"[PostMortem] Memory write failed: {e}")

    # -- Build report ------------------------------------------------------------
    report = {
        "session_id":              session_id,
        "competition_name":        competition_name,
        "cv_mean":                 round(cv_mean, 4),
        "lb_score":                round(lb_score, 4),
        "cv_lb_gap":               round(gap, 4),
        "gap_root_cause":          gap_root_cause,
        "gap_explanation":         gap_explanation,
        "feature_retrospective":   feature_retrospective,
        "patterns_written":        len(validated) + len(failed),
        "critic_failures_written": critic_failures_written,
        "pattern_id":              pattern_id,
        "confidence":              round(confidence, 4),
        "generated_at":            datetime.now(timezone.utc).isoformat(),
    }

    report_path = f"{output_dir}/post_mortem_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[PostMortem] Report written: {report_path}")
    return report


def _classify_gap(
    gap: float,
    cv_std: float,
    critic_severity: str,
    cv_mean: float,
    lb_score: float,
) -> tuple:
    """Classify the CV/LB gap root cause. Returns (cause, explanation)."""

    if gap <= 0.02:
        if critic_severity in ("OK", "unchecked"):
            return "acceptable", (
                f"CV/LB gap {gap:.3f} is within acceptable range. "
                f"CV: {cv_mean:.4f}, LB: {lb_score:.4f}."
            )
        else:
            return "acceptable", (
                f"CV/LB gap {gap:.3f} is small despite critic severity {critic_severity}."
            )

    # gap > 0.02
    if cv_std > 0.02:
        return "high_variance_cv", (
            f"CV std {cv_std:.3f} > 0.02 — model is unstable across folds. "
            f"Gap {gap:.3f} likely due to fold variance, not leakage."
        )

    if critic_severity in ("OK", "unchecked"):
        return "critic_missed", (
            f"Critic approved all features (severity={critic_severity}) but "
            f"CV/LB gap is {gap:.3f}. Probable undetected leakage or distribution shift."
        )

    if critic_severity == "CRITICAL":
        return "known_risk", (
            f"Critic flagged CRITICAL issues. Gap {gap:.3f} confirms the finding. "
            f"The risk was known but engineer continued."
        )

    # HIGH or MEDIUM
    return "known_risk", (
        f"Critic flagged {critic_severity} issues. Gap {gap:.3f} "
        f"confirms partial risk materialised."
    )


def _build_feature_retrospective(
    feature_importance: dict,
    critic_verdict: dict,
    features_dropped: list,
    gap: float,
) -> list:
    """
    For top-20 features by importance: classify as helped/hurt/noisy.
    """
    # Get feature importance list (handle both dict and list formats)
    if isinstance(feature_importance, dict):
        features = feature_importance.get("features", [])
    elif isinstance(feature_importance, list):
        features = feature_importance
    else:
        features = []

    # Get critic findings
    critic_findings = critic_verdict.get("findings", [])
    flagged_features = set()
    for finding in critic_findings:
        if finding.get("severity") in ("HIGH", "CRITICAL"):
            # Collect features mentioned in replan_instructions
            ri = finding.get("replan_instructions", {})
            for feat in ri.get("remove_features", []):
                flagged_features.add(feat)
            # Also check top_drift_features
            for feat in finding.get("top_drift_features", []):
                flagged_features.add(feat)

    retrospective = []
    for i, feat_entry in enumerate(features[:20]):
        if isinstance(feat_entry, dict):
            name = feat_entry.get("feature", feat_entry.get("name", f"feature_{i}"))
            importance = feat_entry.get("importance", 0.0)
            variance = feat_entry.get("fold_variance", 0.0)
        elif isinstance(feat_entry, str):
            name = feat_entry
            importance = 0.0
            variance = 0.0
        else:
            continue

        is_flagged = name in flagged_features
        is_dropped = name in features_dropped

        if is_flagged and gap > 0.02:
            verdict = "hurt"
        elif variance > 0.1:
            verdict = "noisy"
        elif not is_flagged and not is_dropped:
            verdict = "helped"
        else:
            verdict = "helped"

        retrospective.append({
            "feature":       name,
            "importance":    round(importance, 4),
            "fold_variance": round(variance, 4),
            "critic_flagged": is_flagged,
            "dropped":       is_dropped,
            "verdict":       verdict,
        })

    return retrospective


def _find_suspected_feature(retrospective: list) -> str:
    """Find the most likely feature that caused the CV/LB gap."""
    # Look for hurt features first
    hurt = [f for f in retrospective if f["verdict"] == "hurt"]
    if hurt:
        return hurt[0]["feature"]
    # Then noisy features
    noisy = [f for f in retrospective if f["verdict"] == "noisy"]
    if noisy:
        return noisy[0]["feature"]
    # Fallback
    return "unknown — manual inspection needed"
