# Day 27 — Post-Mortem Agent + Full Stress Test
## Prompt for Qwen Code

---

## BEFORE YOU WRITE A SINGLE LINE

Read these files completely first:

```
CLAUDE.md
AGENTS.md
core/state.py
agents/red_team_critic.py
agents/ensemble_architect.py
memory/memory_schema.py
memory/memory_quality.py
tools/wilcoxon_gate.py
guards/circuit_breaker.py
agents/submission_strategist.py
tests/regression/test_phase3_regression.py
```

After reading, answer before writing:
1. What fields does `model_registry` contain per entry? List every key.
2. What does `critic_verdict` look like? List every key including `findings`.
3. What is the format of `lineage.jsonl` events? What keys does each event have?
4. What does `store_pattern()` in `memory/memory_schema.py` expect as input?
5. What does `circuit_breaker.py` set in state when `failure_count == 3`?

Do not proceed until you have answered all five from the actual code.

---

## TASK 1 — Build `agents/post_mortem_agent.py`

This agent runs manually after a competition closes and the private LB score is revealed. It is never called automatically by the pipeline — it is triggered by the engineer with:

```bash
python -m professor post-mortem --session abc123 --lb-score 0.8073 --lb-rank 150 --total-teams 1000
```

### What it reads from disk

All inputs come from disk — the session outputs directory. Never from live state.

```python
REQUIRED_FILES = {
    "metrics.json":          "cv scores, feature_order, model params",
    "lineage.jsonl":         "what happened during the run",
    "submission_log.json":   "submission history with lb_score",
    "critic_verdict.json":   "what the critic found",
}

OPTIONAL_FILES = {
    "feature_importance.json":   "model feature importances",
    "competition_brief.json":    "domain info from competition_intel",
    "ensemble_selection.json":   "diversity selection result",
    "null_importance_result.json": "which features were dropped",
}
```

### The four structured sections of `competition_memory.json`

#### Section 1 — Solution Autopsy

```python
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
    metrics     = _load_json(session_dir / "metrics.json")
    null_result = _load_json(session_dir / "null_importance_result.json", default={})
    critic      = _load_json(session_dir / "critic_verdict.json", default={})
    feat_imp    = _load_json(session_dir / "feature_importance.json", default={})

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
        flagged    = feat in flagged_by_critic
        feature_audit.append({
            "feature":          feat,
            "status":           "survived_null_importance",
            "importance":       round(importance, 6),
            "flagged_by_critic": flagged,
            "classification":   "noise" if flagged else ("contributed" if importance > 0 else "unknown"),
        })

    for feat in dropped_s1:
        feature_audit.append({
            "feature":          feat,
            "status":           "dropped_stage1",
            "importance":       0.0,
            "flagged_by_critic": False,
            "classification":   "correctly_pruned",
        })

    for feat in dropped_s2:
        feature_audit.append({
            "feature":          feat,
            "status":           "dropped_stage2",
            "importance":       0.0,
            "flagged_by_critic": feat in flagged_by_critic,
            "classification":   "correctly_pruned",
        })

    return {
        "total_features_trained":  len(survived),
        "total_dropped_stage1":    len(dropped_s1),
        "total_dropped_stage2":    len(dropped_s2),
        "features_flagged_by_critic": len(flagged_by_critic),
        "feature_audit":           feature_audit,
    }
```

#### Section 2 — Strategy Evaluation

```python
def _build_strategy_evaluation(
    session_dir: Path,
    lb_score: float,
    lb_rank: int | None,
    total_teams: int | None,
) -> dict:
    """
    Evaluates what strategy choices worked and what did not.
    """
    metrics  = _load_json(session_dir / "metrics.json")
    ensemble = _load_json(session_dir / "ensemble_selection.json", default={})
    sub_log  = _load_json(session_dir / "submission_log.json", default=[])

    cv_mean  = float(metrics.get("cv_mean", 0.0))
    cv_std   = float(metrics.get("cv_std", 0.0))
    cv_lb_gap = round(abs(cv_mean - lb_score), 6)

    percentile = None
    if lb_rank and total_teams:
        percentile = round(100 * (1 - lb_rank / total_teams), 2)

    # Was ensemble better than single model?
    ensemble_accepted       = ensemble.get("ensemble_accepted", False)
    ensemble_holdout_score  = float(ensemble.get("ensemble_holdout_score") or 0.0)
    n_models_in_ensemble    = len(ensemble.get("selected_models", []))

    # CV/LB gap root cause classification — same logic as Day 11 post_mortem_agent
    critic_verdict = _load_json(session_dir / "critic_verdict.json", default={})
    critic_ok      = critic_verdict.get("overall_severity") == "OK"

    if cv_lb_gap > 0.020 and critic_ok:
        gap_root_cause = "critic_missed"
    elif cv_lb_gap > 0.020 and not critic_ok:
        gap_root_cause = "known_risk"
    elif cv_std > 0.020:
        gap_root_cause = "high_variance"
    else:
        gap_root_cause = "acceptable"

    return {
        "cv_mean":              round(cv_mean, 6),
        "cv_std":               round(cv_std, 6),
        "lb_score":             round(lb_score, 6),
        "cv_lb_gap":            cv_lb_gap,
        "gap_root_cause":       gap_root_cause,
        "lb_rank":              lb_rank,
        "total_teams":          total_teams,
        "percentile":           percentile,
        "ensemble_accepted":    ensemble_accepted,
        "ensemble_holdout_score": round(ensemble_holdout_score, 6),
        "n_models_in_ensemble": n_models_in_ensemble,
        "winning_model_type":   metrics.get("winning_model_type", "unknown"),
    }
```

#### Section 3 — Structured Memory Writes

```python
def _build_memory_writes(
    session_dir: Path,
    strategy_eval: dict,
    solution_autopsy: dict,
    state: dict,
) -> list[dict]:
    """
    Generates structured memory entries to write to ChromaDB.

    Each entry has:
      domain:           str   — competition domain
      feature:          str   — feature name (or "ensemble_strategy", "model_type")
      cv_delta:         float — CV improvement this finding produced
      private_lb_delta: float — LB improvement (positive = helped)
      validated:        bool  — proven by both CV and LB
      reusable:         bool  — applicable to future competitions
      confidence:       float — 0.0 to 1.0
      finding_type:     str   — "feature" | "strategy" | "model_choice" | "pitfall"
    """
    brief  = _load_json(session_dir / "competition_brief.json", default={})
    domain = brief.get("domain", "tabular")
    lb_gap = strategy_eval["cv_lb_gap"]
    cv     = strategy_eval["cv_mean"]
    lb     = strategy_eval["lb_score"]

    writes = []

    # Feature-level findings
    for item in solution_autopsy.get("feature_audit", []):
        if item["classification"] == "contributed" and item["importance"] > 0.01:
            writes.append({
                "domain":            domain,
                "feature":           item["feature"],
                "cv_delta":          item["importance"],
                "private_lb_delta":  item["importance"] * (lb / cv) if cv > 0 else 0.0,
                "validated":         not item["flagged_by_critic"],
                "reusable":          True,
                "confidence":        _compute_confidence(strategy_eval),
                "finding_type":      "feature",
            })
        elif item["classification"] == "noise" and item["flagged_by_critic"]:
            writes.append({
                "domain":           domain,
                "feature":          item["feature"],
                "cv_delta":         0.0,
                "private_lb_delta": 0.0,
                "validated":        True,
                "reusable":         True,
                "confidence":       0.80,
                "finding_type":     "pitfall",
                "note":             "Flagged by critic as noise or leakage. Avoid in future.",
            })

    # Strategy findings
    if strategy_eval["ensemble_accepted"] and strategy_eval["ensemble_holdout_score"] > 0:
        writes.append({
            "domain":            domain,
            "feature":           "diversity_ensemble",
            "cv_delta":          0.0,
            "private_lb_delta":  0.0,
            "validated":         lb_gap < 0.010,
            "reusable":          True,
            "confidence":        _compute_confidence(strategy_eval),
            "finding_type":      "strategy",
            "note":              f"Diversity ensemble accepted, holdout={strategy_eval['ensemble_holdout_score']:.5f}",
        })

    # Model choice finding
    winning_model = strategy_eval.get("winning_model_type", "unknown")
    if winning_model != "unknown":
        writes.append({
            "domain":           domain,
            "feature":          f"model_type_{winning_model}",
            "cv_delta":         0.0,
            "private_lb_delta": 0.0,
            "validated":        lb_gap < 0.010,
            "reusable":         True,
            "confidence":       _compute_confidence(strategy_eval),
            "finding_type":     "model_choice",
            "note":             f"{winning_model} won with cv={cv:.5f}, lb={lb:.5f}",
        })

    return writes


def _compute_confidence(strategy_eval: dict) -> float:
    """
    Confidence based on:
    - CV/LB gap (small gap = high confidence)
    - LB percentile (top 10% = high confidence)
    """
    percentile = float(strategy_eval.get("percentile") or 50.0)
    cv_lb_gap  = float(strategy_eval.get("cv_lb_gap") or 0.010)

    gap_factor        = max(0.0, 1.0 - cv_lb_gap / 0.030)
    percentile_factor = min(1.0, percentile / 100.0)

    return round(min(0.90, 0.40 + 0.30 * gap_factor + 0.20 * percentile_factor), 3)
```

#### Section 4 — Critic Calibration

```python
def _build_critic_calibration(session_dir: Path, lb_score: float, cv_mean: float) -> dict:
    """
    Evaluates critic accuracy: how many CRITICAL verdicts were correct vs false positives.
    Adjusts recommended sensitivity thresholds for future runs.

    A CRITICAL verdict is "correct" if the CV/LB gap is large (gap > 0.010).
    A CRITICAL verdict is a "false positive" if CV and LB are well-aligned (gap <= 0.005).
    """
    critic = _load_json(session_dir / "critic_verdict.json", default={})
    gap    = abs(cv_mean - lb_score)

    overall_severity = critic.get("overall_severity", "unchecked")
    findings         = critic.get("findings", [])

    n_critical = sum(1 for f in findings if f.get("severity") == "CRITICAL")
    n_high     = sum(1 for f in findings if f.get("severity") == "HIGH")

    # Was the critic right?
    critic_fired    = overall_severity in ("CRITICAL", "HIGH")
    gap_was_large   = gap > 0.010

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
        "overall_severity":       overall_severity,
        "n_critical_findings":    n_critical,
        "n_high_findings":        n_high,
        "cv_lb_gap":              round(gap, 6),
        "calibration_verdict":    calibration_verdict,
        "threshold_recommendation": threshold_recommendation,
        "threshold_note":         threshold_note,
        "vectors_checked":        critic.get("vectors_checked", []),
    }
```

### Main entry point

```python
def run_post_mortem_agent(
    session_id: str,
    lb_score: float,
    lb_rank: int | None = None,
    total_teams: int | None = None,
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
    solution_autopsy    = _build_solution_autopsy(session_dir, lb_score)
    strategy_evaluation = _build_strategy_evaluation(session_dir, lb_score, lb_rank, total_teams)
    memory_writes       = _build_memory_writes(session_dir, strategy_evaluation, solution_autopsy, {})
    critic_calibration  = _build_critic_calibration(session_dir, lb_score, cv_mean)

    # Compose the full report
    report = {
        "session_id":           session_id,
        "lb_score":             round(lb_score, 6),
        "lb_rank":              lb_rank,
        "total_teams":          total_teams,
        "generated_at":         datetime.utcnow().isoformat(),
        "solution_autopsy":     solution_autopsy,
        "strategy_evaluation":  strategy_evaluation,
        "memory_writes":        memory_writes,
        "critic_calibration":   critic_calibration,
        "n_memory_writes":      len(memory_writes),
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


def _write_to_chromadb(report: dict, session_id: str) -> None:
    """Writes memory_writes to ChromaDB professor_patterns_v2 collection."""
    try:
        from memory.memory_schema import store_pattern, build_competition_fingerprint_from_brief
        from memory.memory_schema import build_chroma_client

        brief_path = Path(f"outputs/{session_id}/competition_brief.json")
        if not brief_path.exists():
            logger.warning("[post_mortem] competition_brief.json missing — ChromaDB write skipped.")
            return

        brief       = json.loads(brief_path.read_text())
        fingerprint = build_competition_fingerprint_from_brief(brief)

        validated   = [
            w for w in report["memory_writes"]
            if w.get("validated") and w.get("finding_type") == "feature"
        ]
        failed      = [
            w["feature"] for w in report["memory_writes"]
            if w.get("finding_type") == "pitfall"
        ]

        store_pattern(
            competition_fingerprint=fingerprint,
            competition_name=brief.get("competition_name", session_id),
            validated_approaches=[w["feature"] for w in validated],
            failed_approaches=failed,
            confidence=report["strategy_evaluation"].get("confidence", 0.70),
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
            competition_fingerprint=brief.get("fingerprint", {}),
            competition_name=brief.get("competition_name", "unknown"),
            cv_lb_gap=strategy_eval["cv_lb_gap"],
            feature_flagged="unknown_leakage",
            failure_mode="critic_missed_cv_lb_gap",
            confidence=0.70,
        )
    except Exception as e:
        logger.warning(f"[post_mortem] critic_failure_pattern write failed: {e}")


def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text()) if path.exists() else (default or {})
    except Exception:
        return default or {}
```

---

## TASK 2 — Full stress test (`tests/stress_test.py`)

This file tests that every failure mode in the system is correctly handled. Each test injects a known failure and asserts the correct response. These are integration tests — they run the real pipeline components, not mocks.

```python
# tests/stress_test.py
#
# FULL PIPELINE STRESS TEST
# Tests every major failure mode by injection.
# All five must pass before the Day 30 gate.
#
# Run individually (each test is independent and slow):
#   pytest tests/stress_test.py::TestStress1LeakageCaught -v
#   pytest tests/stress_test.py -v --tb=short
#
# Run time: ~10-20 min for all five. Do not run in CI on every commit.
# Run manually before gate submissions.
```

### Stress Test 1 — Injected data leakage caught by Critic

```python
class TestStress1LeakageCaught:
    """
    Inject a feature that is a direct copy of the target column.
    Critic MUST return CRITICAL via the shuffled_target vector.
    Pipeline MUST NOT submit a prediction using this feature.
    """

    def test_critic_catches_target_derived_feature(self, pipeline_state_with_leakage):
        """
        Build state with a target-derived feature in X_train.
        Run red_team_critic.
        Assert: overall_severity == CRITICAL.
        Assert: the leaking feature is named in findings.
        """
        from agents.red_team_critic import run_red_team_critic

        result = run_red_team_critic(pipeline_state_with_leakage)
        verdict = result["critic_verdict"]

        assert verdict["overall_severity"] == "CRITICAL", (
            f"STRESS TEST FAILED: Critic returned '{verdict['overall_severity']}' "
            "on target-derived leakage. Must return CRITICAL."
        )

        flagged_features = [
            f.get("feature_flagged", "") or f.get("evidence", "")
            for f in verdict.get("findings", [])
        ]
        assert any("leaked" in s.lower() or "target" in s.lower()
                   for s in flagged_features), (
            "STRESS TEST FAILED: Critic returned CRITICAL but did not "
            "name the leaking feature in findings."
        )

    def test_pipeline_does_not_submit_with_critical_verdict(
        self, pipeline_state_with_leakage, monkeypatch
    ):
        """
        After Critic returns CRITICAL, pipeline must replan or halt.
        submission.csv must NOT be written in the same run.
        """
        import agents.ensemble_architect as ea
        ensemble_calls = []
        original = ea.run_ensemble_architect
        def tracking(*a, **k):
            ensemble_calls.append(1)
            return original(*a, **k)
        monkeypatch.setattr(ea, "run_ensemble_architect", tracking)

        from agents.red_team_critic import run_red_team_critic
        from core.professor import get_graph
        state = run_red_team_critic(pipeline_state_with_leakage)

        # If CRITICAL — replan must be requested, ensemble must not run
        if state["critic_verdict"]["overall_severity"] == "CRITICAL":
            assert state.get("replan_requested") is True or \
                   state.get("hitl_required") is True, (
                "STRESS TEST FAILED: CRITICAL verdict did not trigger replan or HITL."
            )
```

### Stress Test 2 — Budget overrun triggers TRIAGE

```python
class TestStress2BudgetOverrunTriage:
    """
    Set budget_remaining_usd to near-zero before the pipeline runs.
    Circuit breaker MUST trigger TRIAGE mode.
    Non-essential agents must be skipped.
    """

    def test_triage_fires_when_budget_exhausted(self):
        from guards.circuit_breaker import get_escalation_level, EscalationLevel

        state = {
            "budget_remaining_usd": 0.001,
            "budget_limit_usd":     5.0,
            "current_node_failure_count": 0,
        }
        level = get_escalation_level(state)
        assert level == EscalationLevel.TRIAGE, (
            f"STRESS TEST FAILED: get_escalation_level returned {level} "
            "with budget at 0.02% remaining. Expected TRIAGE."
        )

    def test_triage_fires_below_5_percent_budget(self):
        from guards.circuit_breaker import get_escalation_level, EscalationLevel

        state = {
            "budget_remaining_usd": 0.24,   # exactly 4.8% of 5.0 → below 5%
            "budget_limit_usd":     5.0,
            "current_node_failure_count": 0,
        }
        level = get_escalation_level(state)
        assert level == EscalationLevel.TRIAGE

    def test_triage_does_not_fire_at_10_percent(self):
        from guards.circuit_breaker import get_escalation_level, EscalationLevel

        state = {
            "budget_remaining_usd": 0.50,   # 10% of 5.0 → above threshold
            "budget_limit_usd":     5.0,
            "current_node_failure_count": 0,
        }
        level = get_escalation_level(state)
        assert level != EscalationLevel.TRIAGE, (
            "STRESS TEST FAILED: TRIAGE fired at 10% budget remaining. "
            "Threshold should be 5%."
        )
```

### Stress Test 3 — Wrong metric blocked by Validation Architect

```python
class TestStress3WrongMetricBlocked:
    """
    Set metric=AUC with a continuous regression target.
    Validation Architect MUST block this before any training occurs.
    ml_optimizer MUST NOT run.
    """

    def test_validation_architect_blocks_auc_on_regression(self):
        from agents.validation_architect import run_validation_architect

        state = {
            "evaluation_metric": "auc",
            "task_type":         "regression",
            "target_column":     "SalePrice",
            "session_id":        "stress_test_3",
        }
        result = run_validation_architect(state)
        assert result.get("validation_error") is not None, (
            "STRESS TEST FAILED: Validation Architect did not set validation_error "
            "for AUC on a regression target."
        )

    def test_validation_architect_blocks_rmse_on_binary(self):
        from agents.validation_architect import run_validation_architect

        state = {
            "evaluation_metric": "rmse",
            "task_type":         "binary_classification",
            "target_column":     "Survived",
            "session_id":        "stress_test_3b",
        }
        result = run_validation_architect(state)
        assert result.get("validation_error") is not None, (
            "STRESS TEST FAILED: Validation Architect did not block RMSE on binary target."
        )

    def test_validation_architect_passes_correct_combination(self):
        from agents.validation_architect import run_validation_architect

        state = {
            "evaluation_metric": "accuracy",
            "task_type":         "binary_classification",
            "target_column":     "Transported",
            "session_id":        "stress_test_3c",
        }
        result = run_validation_architect(state)
        assert result.get("validation_error") is None, (
            f"STRESS TEST FAILED: Validation Architect blocked a correct combination. "
            f"Error: {result.get('validation_error')}"
        )
```

### Stress Test 4 — Three consecutive failures trigger HITL

```python
class TestStress4ThreeFailuresHITL:
    """
    Inject a permanently failing data_engineer.
    After exactly 3 failures, circuit breaker MUST:
      - Set hitl_required=True
      - Set pipeline_halted=True
      - Write state to Redis checkpoint
      - Generate hitl_prompt.json with 3 interventions
    """

    def test_hitl_fires_after_3_failures(self):
        from guards.circuit_breaker import get_escalation_level, handle_escalation, EscalationLevel

        state = {
            "session_id":                "stress_test_4",
            "current_node_failure_count": 3,
            "budget_remaining_usd":       5.0,
            "budget_limit_usd":           5.0,
        }
        level = get_escalation_level(state)
        assert level == EscalationLevel.HITL, (
            f"STRESS TEST FAILED: Expected HITL at failure_count=3, got {level}."
        )

    def test_hitl_not_triggered_at_2_failures(self):
        from guards.circuit_breaker import get_escalation_level, EscalationLevel

        state = {
            "session_id":                "stress_test_4b",
            "current_node_failure_count": 2,
            "budget_remaining_usd":       5.0,
            "budget_limit_usd":           5.0,
        }
        level = get_escalation_level(state)
        assert level == EscalationLevel.MACRO, (
            f"STRESS TEST FAILED: Expected MACRO at failure_count=2, got {level}."
        )

    def test_hitl_sets_required_state_keys(self):
        from guards.circuit_breaker import handle_escalation, EscalationLevel

        state = {
            "session_id":                "stress_test_4c",
            "current_node_failure_count": 3,
            "budget_remaining_usd":       5.0,
            "budget_limit_usd":           5.0,
        }
        result = handle_escalation(
            state=state,
            level=EscalationLevel.HITL,
            agent_name="data_engineer",
            error=RuntimeError("Permanent failure injected by stress test"),
            traceback_str="stress test traceback",
        )
        assert result.get("hitl_required") is True, (
            "STRESS TEST FAILED: hitl_required not set after HITL escalation."
        )
        assert result.get("pipeline_halted") is True, (
            "STRESS TEST FAILED: pipeline_halted not set after HITL escalation."
        )

    def test_hitl_generates_prompt_with_3_interventions(self, tmp_path, monkeypatch):
        from guards.circuit_breaker import handle_escalation, EscalationLevel
        import json

        state = {
            "session_id":                "stress_test_4d",
            "current_node_failure_count": 3,
            "budget_remaining_usd":       5.0,
            "budget_limit_usd":           5.0,
            "output_dir":                 str(tmp_path),
        }
        result = handle_escalation(
            state=state,
            level=EscalationLevel.HITL,
            agent_name="data_engineer",
            error=KeyError("target_column"),
            traceback_str="KeyError traceback",
        )
        prompt = result.get("hitl_prompt", {})
        assert len(prompt.get("interventions", [])) == 3, (
            f"STRESS TEST FAILED: Expected 3 interventions in HITL prompt, "
            f"got {len(prompt.get('interventions', []))}."
        )
```

### Stress Test 5 — EWMA freeze fires on simulated LB drift

```python
class TestStress5EWMAFreezeOnDrift:
    """
    Simulate 10 submissions with worsening CV/LB gap.
    Submission Strategist MUST:
      - Activate monitor after 5 submissions with lb_score
      - Fire freeze when current_ewma > 2x initial_ewma
    """

    def test_ewma_freeze_fires_on_drift(self, tmp_path):
        from agents.submission_strategist import run_submission_strategist
        import json, polars as pl, numpy as np

        early_gaps = [0.005, 0.005, 0.006, 0.005, 0.005]
        later_gaps = [0.018, 0.021, 0.022, 0.020, 0.025]
        all_gaps   = early_gaps + later_gaps

        log = [
            {
                "submission_number": i + 1,
                "session_id":        "stress_test_5",
                "competition_id":    "spaceship-titanic",
                "timestamp":         f"2026-03-{i+1:02d}T12:00:00Z",
                "cv_score":          0.820,
                "lb_score":          0.820 - all_gaps[i],
                "cv_lb_gap":         all_gaps[i],
                "model_used":        "model_best",
                "ensemble_accepted": True,
                "submission_path":   f"outputs/test/sub_{i+1}.csv",
                "is_final_pair_submission": False,
            }
            for i in range(10)
        ]

        out_dir = tmp_path / "outputs" / "stress_test_5"
        out_dir.mkdir(parents=True)
        (out_dir / "submission_log.json").write_text(json.dumps(log))

        sub_path    = tmp_path / "sub.csv"
        sample_path = tmp_path / "sample.csv"
        rng = np.random.default_rng(42)
        df  = pl.DataFrame({
            "PassengerId": [f"0001_{i:03d}" for i in range(100)],
            "Transported": [True] * 100,
        })
        df.write_csv(sub_path); df.write_csv(sample_path)

        state = {
            "competition_name":        "spaceship-titanic",
            "session_id":              "stress_test_5",
            "model_registry": {
                "model_best": {
                    "cv_mean": 0.820, "cv_std": 0.010,
                    "stability_score": 0.805,
                    "fold_scores": [0.820] * 5,
                    "oof_predictions": rng.uniform(0, 1, 100).tolist(),
                    "data_hash": "abc123",
                }
            },
            "y_train":                 np.ones(100),
            "evaluation_metric":       "accuracy",
            "task_type":               "binary_classification",
            "target_column":           "Transported",
            "id_column":               "PassengerId",
            "ensemble_accepted":       False,
            "ensemble_oof":            [0.8] * 100,
            "cv_mean":                 0.820,
            "data_hash":               "abc123",
            "sample_submission_path":  str(sample_path),
            "output_dir":              str(out_dir),
        }

        result = run_submission_strategist(state)

        assert result["submission_freeze_active"] is True, (
            f"STRESS TEST FAILED: EWMA freeze did not fire on simulated LB drift. "
            f"ewma_current={result.get('ewma_current')}, "
            f"ewma_initial={result.get('ewma_initial')}"
        )
        assert result["submission_freeze_reason"] in (
            "ewma_exceeded_2x_initial", "gap_increasing_5_of_7"
        ), (
            f"STRESS TEST FAILED: Unexpected freeze reason: "
            f"{result.get('submission_freeze_reason')}"
        )
```

---

## COMMIT SEQUENCE

```
git commit -m "Day 27: agents/post_mortem_agent.py — solution autopsy, strategy evaluation, memory writes, critic calibration"
git commit -m "Day 27: tests/stress_test.py — 5 failure mode injection tests"
```

---

## VERIFICATION

```bash
python -c "from agents.post_mortem_agent import run_post_mortem_agent; print('OK')"

pytest tests/stress_test.py -v --tb=short

pytest tests/contracts/ -v --tb=short
pytest tests/regression/ -v --tb=short
```

All four commands must show zero failures before committing.