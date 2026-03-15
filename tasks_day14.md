# Professor Agent — Day 14 Implementation
**Theme: Compounding advantage + Phase 2 gate**

Build order: Task 1 → Task 2 → Task 3
```
Task 1  →  GM-CAP 4: critic reads critic_failure_patterns from ChromaDB
           agents/red_team_critic.py
Task 2  →  Phase 2 gate: three mandatory conditions
           tests/phase2_gate.py
Task 3  →  Freeze Phase 2 regression test
           tests/regression/test_phase2_regression.py
           commit: "Day 14: GM-CAP 4 critic memory, Phase 2 gate passed, regression frozen"
```

**Prerequisite check before starting:**
- `post_mortem_agent.py` (Day 11) must be complete — it writes to `critic_failure_patterns`
- `critic_failure_patterns` ChromaDB collection must exist (created by `store_critic_failure_pattern()`)
- If collection is empty (no competitions completed yet), Day 14 gracefully degrades — static 7 vectors run, historical vector returns `OK` with note "no historical patterns yet"

---

## TASK 1 — GM-CAP 4: `agents/red_team_critic.py`

**What changes:** The critic gains an 8th vector — `historical_failures`. Unlike the 7 static vectors (fixed rules), this vector is dynamic: it queries `critic_failure_patterns` ChromaDB collection for patterns similar to the current competition fingerprint, then checks whether any high-confidence historical failure modes are present in the current feature set.

**The compounding advantage:** After competition 1, the collection has 0 patterns. After competition 5, it has 5+ patterns. After competition 15, it has 15+ patterns — each one a specific lesson from a specific past mistake. The critic grows smarter with every competition run. Static rules stay at 7. Battle-tested rules grow without bound.

### New function: `_check_historical_failures(state) -> dict`
```python
def _check_historical_failures(state: ProfessorState) -> dict:
    """
    Vector 8: Retrieves top-5 historical failure patterns similar to this competition
    from the critic_failure_patterns ChromaDB collection. Flags any patterns where
    the flagged feature or failure mode is present in the current feature set.

    Severity logic:
        confidence >= 0.85 AND feature present  →  CRITICAL
        confidence >= 0.70 AND feature present  →  HIGH
        confidence >= 0.50 AND feature present  →  MEDIUM
        no matches or collection empty          →  OK

    Never raises — returns OK with diagnostic note on any failure.
    """
    fingerprint = state.get("competition_fingerprint", {})
    feature_names = state.get("feature_names", [])

    # 1. Query ChromaDB for similar failure patterns
    try:
        patterns = query_critic_failure_patterns(
            fingerprint=fingerprint,
            n_results=5,
            max_distance=0.75,   # only retrieve genuinely similar competitions
        )
    except Exception as e:
        return {
            "verdict": "OK",
            "note": f"ChromaDB query failed ({e}). Historical check skipped.",
            "patterns_retrieved": 0,
            "findings": [],
        }

    if not patterns:
        return {
            "verdict": "OK",
            "note": "No similar historical failure patterns found in memory.",
            "patterns_retrieved": 0,
            "findings": [],
        }

    # 2. For each retrieved pattern, check if the failure mode is present now
    findings = []
    for pattern in patterns:
        feature_flagged    = pattern.get("feature_flagged", "")
        failure_mode       = pattern.get("failure_mode", "")
        confidence         = float(pattern.get("confidence", 0.0))
        cv_lb_gap          = float(pattern.get("cv_lb_gap", 0.0))
        competition_source = pattern.get("competition_name", "unknown")
        distance           = float(pattern.get("distance", 1.0))

        # Match: exact name OR substring (catches "target_enc_cabin" matching "target_enc")
        feature_present = (
            feature_flagged in feature_names
            or any(feature_flagged in f for f in feature_names)
            or any(f in feature_flagged for f in feature_names)
        )

        if not feature_present:
            continue   # pattern doesn't apply to current feature set

        # Determine severity
        if confidence >= 0.85:
            severity = "CRITICAL"
        elif confidence >= 0.70:
            severity = "HIGH"
        elif confidence >= 0.50:
            severity = "MEDIUM"
        else:
            continue   # too low confidence to flag

        findings.append({
            "severity":          severity,
            "vector":            "historical_failures",
            "feature_flagged":   feature_flagged,
            "failure_mode":      failure_mode,
            "confidence":        round(confidence, 3),
            "cv_lb_gap_history": round(cv_lb_gap, 4),
            "competition_source": competition_source,
            "similarity_distance": round(distance, 3),
            "evidence": (
                f"In {competition_source} (similar competition profile, "
                f"distance={distance:.2f}), {failure_mode} caused "
                f"CV/LB gap={cv_lb_gap:.3f}. Confidence: {confidence:.2f}. "
                f"Feature '{feature_flagged}' is present in current feature set."
            ),
            "action": f"Investigate '{feature_flagged}' for {failure_mode}.",
            "replan_instructions": {
                "remove_features":  [feature_flagged] if severity == "CRITICAL" else [],
                "rerun_nodes":      ["feature_factory"] if severity == "CRITICAL" else [],
            },
        })

    if not findings:
        return {
            "verdict": "OK",
            "note": (
                f"Retrieved {len(patterns)} historical patterns. "
                f"None matched current feature set."
            ),
            "patterns_retrieved": len(patterns),
            "findings": [],
        }

    overall_severity = max(
        findings,
        key=lambda f: {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "OK": 0}[f["severity"]]
    )["severity"]

    return {
        "verdict":           overall_severity,
        "patterns_retrieved": len(patterns),
        "patterns_matched":  len(findings),
        "findings":          findings,
    }
```

### `query_critic_failure_patterns(fingerprint, n_results, max_distance) -> list[dict]`

Placed in `memory/memory_schema.py` alongside existing pattern functions.
```python
def query_critic_failure_patterns(
    fingerprint: dict,
    n_results: int = 5,
    max_distance: float = 0.75,
) -> list[dict]:
    """
    Queries the critic_failure_patterns ChromaDB collection.
    Returns [] if collection doesn't exist or is empty.
    Never raises.
    """
    try:
        client = build_chroma_client()
        try:
            collection = client.get_collection("critic_failure_patterns")
        except Exception:
            return []   # collection doesn't exist yet — first competition

        if collection.count() == 0:
            return []

        query_text = fingerprint_to_text(fingerprint)
        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        patterns = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist > max_distance:
                continue   # too dissimilar to be useful
            patterns.append({**meta, "distance": dist, "document": doc})

        return patterns

    except Exception as e:
        logger.warning(f"[query_critic_failure_patterns] Failed: {e}")
        return []
```

### Wire into critic orchestrator

In `_run_core_logic()` (the main critic dispatch function), add vector 8:
```python
VECTOR_FUNCTIONS = {
    "shuffled_target":     _check_shuffled_target,
    "id_only_model":       _check_id_only_model,
    "adversarial_classifier": _check_adversarial_classifier,
    "preprocessing_audit": _check_preprocessing_audit,
    "pr_curve_imbalance":  _check_pr_curve_imbalance,
    "temporal_leakage":    _check_temporal_leakage,
    "robustness":          _check_robustness,       # Day 11
    "historical_failures": _check_historical_failures,  # Day 14 ← NEW
}
```

`vectors_checked` in `critic_verdict.json` must now be a list of 8 names.

The `overall_severity` computation already takes the max across all vectors — no change needed there.

### `store_critic_failure_pattern()` — verify correct fields written (Day 11)

The `post_mortem_agent` must write these exact fields to `critic_failure_patterns` metadata for the query to work:
```python
# In post_mortem_agent.py — verify these keys are present:
{
    "competition_name":  str,   # e.g. "spaceship-titanic-2024"
    "feature_flagged":   str,   # e.g. "target_enc_cabin"
    "failure_mode":      str,   # e.g. "target encoding without fold isolation"
    "cv_lb_gap":         float, # e.g. 0.026
    "confidence":        float, # e.g. 0.85
    "fingerprint_text":  str,   # from fingerprint_to_text() — for ChromaDB embedding
    "stored_at":         str,   # ISO timestamp
}
# The document text (embedded by ChromaDB) = fingerprint_to_text(fingerprint)
```

If any of these keys is missing from the stored metadata, the `_check_historical_failures` function degrades gracefully (`.get("key", default)` on all fields) — but the pattern will have less useful output.

### New `ProfessorState` fields (none)

No new state fields. Uses existing `competition_fingerprint` and `feature_names`.

---

## TASK 2 — Phase 2 Gate: `tests/phase2_gate.py`

**What this is:** A one-time gate test. Run it after Day 14 is complete. All three conditions must pass before proceeding to Phase 3 or freezing the regression test. If any condition fails, Phase 2 is not complete — fix the failure and re-run.

**Structure:** Three independent scenarios run sequentially. Each must pass for the gate to pass. Written as a pytest file but annotated clearly so it reads as a gate checklist.
```python
# tests/phase2_gate.py
#
# PHASE 2 GATE — Run once after Day 14. All 3 conditions must pass.
# If any fail, Phase 2 is NOT complete. Fix the failure, re-run.
#
# Record passing commit hash in tests/regression/test_phase2_regression.py header.
#
# DO NOT run this file as part of regular pytest suite:
#   pytest tests/phase2_gate.py   ← run explicitly only

import pytest

# ─── CONDITION 1: Critic catches injected leakage ─────────────────────────────

class TestPhase2Condition1_CriticCatchesLeakage:
    """
    Inject a feature that is directly derived from the target column.
    Critic must return CRITICAL via the shuffled_target or id_only_model vector.
    Gate fails if critic returns OK or HIGH on this input.
    """

    def test_critic_catches_target_derived_feature(self, full_pipeline_state):
        """
        Inject leaky_feature = target_column.shift(1) into training data.
        Run red_team_critic. Assert overall_severity == "CRITICAL".
        """
        state = inject_leaky_feature(full_pipeline_state, feature_type="target_derived")
        result = run_red_team_critic(state)

        assert result["critic_verdict"]["overall_severity"] == "CRITICAL", (
            f"GATE FAIL: Critic returned {result['critic_verdict']['overall_severity']} "
            f"on a target-derived feature. Expected CRITICAL. "
            f"Findings: {result['critic_verdict']['findings']}"
        )

    def test_critic_catches_id_as_feature(self, full_pipeline_state):
        """
        Inject row_id as a training feature (perfect predictor on train, noise on test).
        Run critic. Assert CRITICAL from id_only_model vector.
        """
        state = inject_leaky_feature(full_pipeline_state, feature_type="row_id")
        result = run_red_team_critic(state)

        assert result["critic_verdict"]["overall_severity"] == "CRITICAL", (
            f"GATE FAIL: Critic returned {result['critic_verdict']['overall_severity']} "
            f"on row_id-as-feature. Expected CRITICAL."
        )

    def test_critic_clean_on_legitimate_features(self, full_pipeline_state):
        """
        Sanity check: critic must NOT flag CRITICAL on clean, legitimate features.
        Gate fails if critic over-fires on clean data.
        """
        state = inject_legitimate_features(full_pipeline_state)
        result = run_red_team_critic(state)

        assert result["critic_verdict"]["overall_severity"] != "CRITICAL", (
            f"GATE FAIL: Critic returned CRITICAL on clean features (false positive). "
            f"Findings: {result['critic_verdict']['findings']}"
        )


# ─── CONDITION 2: Validation Architect blocks wrong metric ────────────────────

class TestPhase2Condition2_ValidationArchitectBlocksWrongMetric:
    """
    Set competition metric to AUC but inject a regression dataset (continuous target).
    Validation Architect must block this with an error before any training occurs.
    Gate fails if training proceeds with the wrong metric.
    """

    def test_validation_architect_blocks_auc_on_regression(self, full_pipeline_state):
        """
        Continuous target + metric="auc". Validation Architect must raise or set
        validation_error in state before ml_optimizer runs.
        """
        state = {
            **full_pipeline_state,
            "metric": "auc",
            "target_type": "continuous",
        }
        result = run_validation_architect(state)

        assert result.get("validation_error") is not None, (
            "GATE FAIL: Validation Architect did not set validation_error "
            "for AUC metric on continuous target."
        )
        assert "ml_optimizer" not in get_nodes_executed(result), (
            "GATE FAIL: ml_optimizer executed despite Validation Architect block."
        )

    def test_validation_architect_blocks_rmse_on_binary(self, full_pipeline_state):
        """
        Binary target + metric="rmse". Must be blocked.
        """
        state = {
            **full_pipeline_state,
            "metric": "rmse",
            "target_type": "binary",
        }
        result = run_validation_architect(state)

        assert result.get("validation_error") is not None, (
            "GATE FAIL: Validation Architect did not block RMSE on binary target."
        )

    def test_validation_architect_passes_correct_metric(self, full_pipeline_state):
        """
        Sanity check: binary target + metric="auc" must not be blocked.
        """
        state = {
            **full_pipeline_state,
            "metric": "auc",
            "target_type": "binary",
        }
        result = run_validation_architect(state)

        assert result.get("validation_error") is None, (
            "GATE FAIL: Validation Architect blocked a correct metric/target combination."
        )


# ─── CONDITION 3: End-to-end CV better than Phase 1 baseline ─────────────────

class TestPhase2Condition3_CVBetterThanPhase1Baseline:
    """
    Run the full pipeline end-to-end on the Phase 1 benchmark dataset
    (Spaceship Titanic or equivalent). Final CV mean must exceed Phase 1 baseline.

    Phase 1 baseline: stored in tests/regression/phase1_baseline.json
    Minimum improvement required: 0.005 (0.5pp — not noise, genuine gain)
    """

    MINIMUM_IMPROVEMENT = 0.005

    def test_phase2_cv_beats_phase1_baseline(self, benchmark_dataset):
        import json
        from pathlib import Path

        baseline_path = Path("tests/regression/phase1_baseline.json")
        assert baseline_path.exists(), (
            "GATE FAIL: tests/regression/phase1_baseline.json not found. "
            "Run Phase 1 gate first and record the baseline CV score."
        )

        baseline = json.loads(baseline_path.read_text())
        phase1_cv = float(baseline["cv_mean"])

        result = run_full_pipeline(benchmark_dataset)
        phase2_cv = float(result["state"]["cv_mean"])

        improvement = phase2_cv - phase1_cv

        assert improvement >= self.MINIMUM_IMPROVEMENT, (
            f"GATE FAIL: Phase 2 CV ({phase2_cv:.5f}) does not beat "
            f"Phase 1 baseline ({phase1_cv:.5f}) by the required {self.MINIMUM_IMPROVEMENT}. "
            f"Improvement: {improvement:+.5f}. "
            f"Phase 2 additions (critic, Wilcoxon gate, memory) must improve CV."
        )

        # Record Phase 2 result for use in regression test
        phase2_record = {
            "cv_mean": phase2_cv,
            "phase1_cv_mean": phase1_cv,
            "improvement": round(improvement, 6),
            "recorded_at": __import__("datetime").datetime.utcnow().isoformat(),
        }
        Path("tests/regression/phase2_baseline.json").write_text(
            json.dumps(phase2_record, indent=2)
        )


# ─── GATE SUMMARY ─────────────────────────────────────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    """Print gate summary on completion."""
    if exitstatus == 0:
        print("\n" + "="*60)
        print("PHASE 2 GATE: ALL CONDITIONS PASSED")
        print("Next step: freeze tests/regression/test_phase2_regression.py")
        print("Record this commit hash in the regression file header.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("PHASE 2 GATE: FAILED — do not freeze regression test yet")
        print("Fix failures above and re-run: pytest tests/phase2_gate.py")
        print("="*60)
```

---

## TASK 3 — Freeze Phase 2 Regression: `tests/regression/test_phase2_regression.py`

**Written only after Phase 2 gate passes. NEVER edited thereafter.**
```python
# tests/regression/test_phase2_regression.py
#
# PHASE 2 REGRESSION FREEZE
# Written: [date of Phase 2 gate pass]
# Commit hash at freeze: [git rev-parse HEAD]
#
# These tests are IMMUTABLE. They encode the minimum quality floor
# established when Phase 2 gate passed. Any regression in this file
# means Phase 2 capabilities have degraded.
#
# DO NOT EDIT. DO NOT PARAMETRIZE. DO NOT ADD CONDITIONS.
# If a test fails, fix the underlying capability — never fix the test.
#
# Tests run as part of the standard regression suite:
#   pytest tests/regression/ -v

import json
import pytest
from pathlib import Path


# ─── FREEZE 1: Phase 1 CV floor still holds ───────────────────────────────────

class TestPhase1FloorStillHolds:
    """Phase 2 additions must not regress Phase 1 CV."""

    def test_cv_above_phase1_baseline(self, benchmark_dataset):
        baseline = json.loads(
            Path("tests/regression/phase1_baseline.json").read_text()
        )
        result = run_full_pipeline(benchmark_dataset)
        cv_mean = float(result["state"]["cv_mean"])
        phase1_floor = float(baseline["cv_mean"])

        assert cv_mean >= phase1_floor - 0.002, (
            f"REGRESSION: CV ({cv_mean:.5f}) dropped below Phase 1 floor "
            f"({phase1_floor:.5f}). Phase 2 additions regressed baseline performance."
        )


# ─── FREEZE 2: Critic catches injected leakage ────────────────────────────────

class TestCriticCatchesLeakageAlways:
    """Critic must catch target-derived leakage on every run. No exceptions."""

    def test_critic_critical_on_target_derived_feature(self, full_pipeline_state):
        state = inject_leaky_feature(full_pipeline_state, feature_type="target_derived")
        result = run_red_team_critic(state)

        assert result["critic_verdict"]["overall_severity"] == "CRITICAL", (
            f"REGRESSION: Critic returned "
            f"{result['critic_verdict']['overall_severity']} "
            f"on target-derived leakage. Must always be CRITICAL."
        )


# ─── FREEZE 3: Validation Architect blocks wrong metric ───────────────────────

class TestValidationArchitectBlocksWrongMetricAlways:
    """Metric/target mismatch must always be blocked. No regressions."""

    def test_auc_blocked_on_regression_target(self, full_pipeline_state):
        state = {**full_pipeline_state, "metric": "auc", "target_type": "continuous"}
        result = run_validation_architect(state)
        assert result.get("validation_error") is not None, (
            "REGRESSION: Validation Architect no longer blocks AUC on regression target."
        )


# ─── FREEZE 4: HITL fires on 3x consecutive failure ──────────────────────────

class TestHITLFiresOn3xFailure:
    """Circuit breaker must escalate to HITL after exactly 3 failures."""

    def test_hitl_triggered_after_3_consecutive_failures(self, full_pipeline_state):
        state = inject_permanent_failure(full_pipeline_state, agent="data_engineer")
        result = run_pipeline_until_halt(state)

        assert result["state"].get("hitl_required") is True, (
            "REGRESSION: HITL did not fire after 3 consecutive data_engineer failures."
        )
        assert result["state"].get("current_node_failure_count") >= 3, (
            "REGRESSION: Failure count below 3 when HITL fired."
        )


# ─── FREEZE 5: Wilcoxon gate rejects non-significant improvements ────────────

class TestWilcoxonGateRejectsNoise:
    """
    Non-significant fold score differences must be rejected by the gate.
    Complex models must not replace simple models on noise.
    """

    def test_gate_returns_false_for_noise_level_difference(self):
        from tools.wilcoxon_gate import is_significantly_better

        # Near-identical fold scores — within rounding noise
        scores_a = [0.8012, 0.8009, 0.8015, 0.8011, 0.8013]
        scores_b = [0.8010, 0.8012, 0.8013, 0.8009, 0.8014]

        result = is_significantly_better(scores_a, scores_b)
        assert result is False, (
            "REGRESSION: Wilcoxon gate approved a non-significant improvement. "
            "Complex models are being selected on noise."
        )
```

---

## INTEGRATION CHECKLIST

- [ ] `_check_historical_failures()` added to `VECTOR_FUNCTIONS` dict in critic
- [ ] `vectors_checked` in `critic_verdict.json` now has 8 entries (not 7)
- [ ] `query_critic_failure_patterns()` in `memory/memory_schema.py` — returns `[]` when collection empty, never raises
- [ ] `_check_historical_failures()` uses `.get()` with defaults on all pattern fields — never raises on malformed pattern
- [ ] `max_distance=0.75` filter prevents distant/irrelevant competitions from firing
- [ ] Phase 2 gate file: `tests/phase2_gate.py` — NOT in standard pytest discovery
- [ ] Phase 2 gate run: `pytest tests/phase2_gate.py -v` — all 7 conditions pass
- [ ] `tests/regression/phase2_baseline.json` written by gate (auto-written on pass)
- [ ] Regression file: `tests/regression/test_phase2_regression.py` — commit hash recorded in header
- [ ] `pytest tests/regression/` — all 5 frozen tests pass
- [ ] Day 11 `post_mortem_agent.py` verified: writes `feature_flagged`, `failure_mode`, `cv_lb_gap`, `confidence` to `critic_failure_patterns` metadata

## GIT COMMIT MESSAGE
```
Day 14: GM-CAP 4 critic memory + Phase 2 gate passed + regression frozen

- red_team_critic: vector 8 (historical_failures) — queries critic_failure_patterns
- memory_schema: query_critic_failure_patterns() — max_distance=0.75 filter
- tests/phase2_gate.py: 7 gate conditions — all pass
- tests/regression/phase2_baseline.json: phase2 CV recorded
- tests/regression/test_phase2_regression.py: 5 frozen conditions — immutable
- Critic now has 8 vectors: 7 static + 1 compounding from competition memory
```