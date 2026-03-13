# Professor Agent — Day 11 High-Level Plan
**For: Claude Code**
**Status: Day 10 COMPLETE — memory schema v2, critic all 6 vectors, contracts green.**
**Mission: Close the learning loop. Day 11 is when Professor starts getting smarter with every competition.**

---

## ARCHITECTURAL OVERVIEW

Day 11 has three tasks with a hidden dependency that will break the build if ignored.

```
Task 2 (Vector 4)  →  Task 3 (LangGraph wiring)  →  Task 1 (post_mortem_agent)
```

**Why this order:**
- Vector 4 must be complete before LangGraph wiring — the routing logic reads `critic_severity`, which only has full meaning once all vectors run
- LangGraph wiring introduces `supervisor_replan` — a new node that `post_mortem_agent` reads from (it checks whether a replan occurred during the run)
- `post_mortem_agent` depends on `memory_schema.py` (Day 10) and the complete critic output

---

## THE KEY ARCHITECTURAL DECISION: CRITIC → SUPERVISOR, NOT HITL

Day 10 built: `CRITICAL verdict → hitl_required=True → human intervention`.

Day 11 changes this to a two-stage response:

```
CRITICAL verdict
    ↓
Stage 1: supervisor_replan node
  - Reads replan_rerun_nodes + replan_remove_features from critic verdict
  - Drops bad features from state
  - Increments dag_version
  - Re-enters pipeline at earliest affected node
    ↓
If CRITICAL verdict again (same session, dag_version ≥ 3):
Stage 2: hitl_handler (human-in-the-loop)
```

**Why this matters:** An automatic replan handles the 80% case where the fix is mechanical — remove the leaking feature and retrain. HITL is reserved for the 20% case where the fix requires human judgment. Without Stage 1, every leakage detection requires human intervention. With Stage 1, most are self-healing.

**The gate that prevents infinite loops:** `dag_version` is the counter. If `dag_version >= MAX_REPLAN_ATTEMPTS` (default: 3), the Supervisor stops replanning and escalates to HITL. Without this gate, Professor loops: critic flags leakage → supervisor removes feature → feature_factory reconstructs it in different form → critic flags again.

**Severity routing:**
```
CRITICAL  →  supervisor_replan  (dag_version++)
HIGH      →  log finding, continue to ensemble_architect
MEDIUM    →  log finding, continue to ensemble_architect  
OK        →  continue to ensemble_architect
```

HIGH is not HITL. HIGH is a warning that the engineer should know about. Only CRITICAL triggers automated remediation.

---

## TASK 2 — Critic Vector 4: Robustness

**File:** `agents/red_team_critic.py` (extends existing)
**New vector name in `vectors_checked`:** `"robustness"`

This vector runs three sub-checks. All three must run. The overall `robustness` verdict is the max severity across the three.

### Sub-check A: Gaussian Noise Injection

**What it catches:** A model that is overfit to noise in the training data. If adding small Gaussian noise to the top features causes the score to collapse (not degrade gracefully), the model has no genuine signal — it memorised specific values.

**Method:**
1. Get top-k features by importance from OOF model (k=5 or all if < 5)
2. Add Gaussian noise: σ = 10% of each feature's standard deviation
3. Re-score model on noisy features using OOF predictions
4. Compare to clean score: degradation > 20% → CRITICAL, 10-20% → HIGH, < 10% → OK

**Threshold rationale:** A robust model should lose at most 10-20% of its score when features are perturbed within the natural variance of the data. If it loses more, it learned to discriminate on specific numerical values rather than patterns.

### Sub-check B: Slice Performance Audit

**What it catches:** A model that performs well on average but is systematically broken for a specific subgroup. This is catastrophic in competitions where the private LB has a different segment distribution than public.

**Method:**
1. For each categorical feature with 2-10 unique values: compute per-slice AUC
2. For numeric features: split into top/bottom quartile, compute AUC per half
3. If max_slice_AUC - min_slice_AUC > 0.15 → HIGH verdict with slice breakdown

**Output format:**
```json
{
  "verdict": "HIGH",
  "worst_slice": {"feature": "region", "value": "Northeast", "auc": 0.51},
  "best_slice": {"feature": "region", "value": "Southwest", "auc": 0.78},
  "spread": 0.27
}
```

### Sub-check C: OOF Calibration Check

**What it catches:** A model that ranks correctly (good AUC) but whose predicted probabilities are wrong (bad Brier Score). This is the core failure mode for WiDS 2026 — evaluated on Brier Score, not AUC.

**Method:**
1. Requires `y_prob` (OOF probabilities) — if not available, return OK with note
2. Compute Expected Calibration Error (ECE) using 10 equal-width bins
3. ECE > 0.10 → HIGH, include calibration curve data in evidence
4. Brier Score > 2× random baseline → HIGH

**Why ECE, not just Brier:** Brier Score conflates calibration with discrimination. ECE isolates the calibration component. A model can have good Brier and poor ECE (well-calibrated but not discriminating) or poor Brier and good ECE (discriminating but miscalibrated). Both are informative. Report both.

**State additions for robustness vector:**
```python
# No new state keys — findings appended to existing critic_verdict["findings"]
# vectors_checked gains "robustness"
```

### Wiring robustness into the orchestrator

Add to `_run_core_logic()` in `red_team_critic.py`:
```python
_run_vector("robustness", _check_robustness(
    X_train=X_train,
    y_true=y_true,
    y_prob=y_prob,
    eda_report=eda_report,
    model_registry=state.get("model_registry", {}),
))
```

`_check_robustness()` internally calls all three sub-checks and returns the max-severity result.

---

## TASK 3 — Wire Critic into LangGraph: FAIL Path to Supervisor

**Files:** `core/professor.py`, `agents/supervisor.py` (new or extends existing)

### New Node: `supervisor_replan`

This is not a retry. It is a targeted DAG rewrite based on the critic's specific findings.

```python
# agents/supervisor.py (or core/professor.py)

MAX_REPLAN_ATTEMPTS = 3

def run_supervisor_replan(state: ProfessorState) -> ProfessorState:
    """
    Called when critic returns CRITICAL verdict.
    Reads replan instructions and constructs a new execution path.
    Increments dag_version. If max attempts reached, escalates to HITL.
    Never a retry — this is a new DAG, not a repeat of the old one.
    """
    dag_version = state.get("dag_version", 1)

    if dag_version >= MAX_REPLAN_ATTEMPTS:
        # Self-healing exhausted — escalate to human
        return {
            **state,
            "hitl_required": True,
            "hitl_reason": (
                f"Supervisor exhausted {MAX_REPLAN_ATTEMPTS} replan attempts. "
                f"Critic still finding CRITICAL issues. Manual review required. "
                f"Last verdict: {state.get('hitl_reason', 'unknown')}"
            ),
            "pipeline_halted": True,
        }

    # Read replan instructions from critic verdict
    remove_features = state.get("replan_remove_features", [])
    rerun_nodes     = state.get("replan_rerun_nodes", [])

    # Build the new execution context
    features_dropped = list(set(state.get("features_dropped", []) + remove_features))

    new_dag_version = dag_version + 1
    print(
        f"[Supervisor] Replan v{new_dag_version}. "
        f"Dropping features: {remove_features}. "
        f"Rerunning nodes: {rerun_nodes}."
    )

    log_event(
        session_id=state["session_id"],
        agent="supervisor",
        action="dag_replan",
        keys_read=["replan_remove_features", "replan_rerun_nodes"],
        keys_written=["dag_version", "features_dropped"],
        values_changed={
            "dag_version_before": dag_version,
            "dag_version_after":  new_dag_version,
            "features_dropped":   remove_features,
            "nodes_to_rerun":     rerun_nodes,
        },
    )

    return {
        **state,
        "dag_version":           new_dag_version,
        "features_dropped":      features_dropped,
        "replan_requested":      False,       # consumed
        "replan_remove_features": [],          # cleared
        "replan_rerun_nodes":    rerun_nodes,  # kept for routing
        "hitl_required":         False,        # critic set this — clear it for replan pass
        "critic_severity":       "unchecked",  # critic will re-run after replan
        "critic_verdict":        {},
    }
```

### Routing Logic

```python
# core/professor.py

def _route_from_critic(state: ProfessorState) -> str:
    severity = state.get("critic_severity", "unchecked")
    if severity == "CRITICAL":
        dag_version = state.get("dag_version", 1)
        if dag_version >= MAX_REPLAN_ATTEMPTS:
            return "hitl_handler"
        return "supervisor_replan"
    # HIGH and MEDIUM: log and continue
    return "ensemble_architect"


graph.add_node("supervisor_replan", run_supervisor_replan)
graph.add_conditional_edges(
    "red_team_critic",
    _route_from_critic,
    {
        "supervisor_replan":  "supervisor_replan",
        "ensemble_architect": "ensemble_architect",
        "hitl_handler":       "hitl_handler",
    },
)
```

### Re-entry Routing After Supervisor Replan

After `supervisor_replan`, the graph needs to re-enter at the earliest node in `replan_rerun_nodes`.

```python
# Node priority order (earlier = lower number):
NODE_PRIORITY = {
    "data_engineer":        1,
    "eda_agent":            2,
    "validation_architect": 3,
    "feature_factory":      4,
    "ml_optimizer":         5,
    "red_team_critic":      6,
}

def _route_from_supervisor_replan(state: ProfessorState) -> str:
    rerun_nodes = state.get("replan_rerun_nodes", [])
    if not rerun_nodes:
        return "feature_factory"  # default: re-run from feature factory
    # Re-enter at the earliest affected node
    earliest = min(rerun_nodes, key=lambda n: NODE_PRIORITY.get(n, 99))
    return earliest

graph.add_conditional_edges(
    "supervisor_replan",
    _route_from_supervisor_replan,
    {node: node for node in NODE_PRIORITY},
)
```

### State Additions

```python
# ProfessorState:
features_dropped:     list   # accumulated across all replan cycles
replan_rerun_nodes:   list   # set by critic, consumed by supervisor

# initial_state():
"features_dropped":   [],
"replan_rerun_nodes": [],
```

---

## TASK 1 — GM-CAP 3: Post-Mortem Agent

**File:** `agents/post_mortem_agent.py`
**When it runs:** After competition closes and private LB is revealed. This is a manual trigger — not part of the main pipeline. Called by the engineer: `professor post-mortem --session <id> --lb-score <score>`.

### Inputs

All inputs must exist before the agent runs:

```python
REQUIRED_INPUTS = {
    "session_id":       str,    # from the completed competition run
    "lb_score":         float,  # private LB score — entered manually by engineer
    "lb_rank":          int,    # optional, entered manually
    "total_competitors": int,   # optional, from competition page
}

# Loaded from session outputs/:
LOADED_FROM_DISK = {
    "lineage_path":              f"outputs/{session_id}/lineage.jsonl",
    "cv_scores_path":            f"outputs/{session_id}/validation_strategy.json",
    "submission_log_path":       f"outputs/{session_id}/submission_log.json",
    "critic_verdict_path":       f"outputs/{session_id}/critic_verdict.json",
    "competition_fingerprint":   f"outputs/{session_id}/competition_fingerprint.json",
    "feature_importance_path":   f"outputs/{session_id}/feature_importance.json",
    "eda_report_path":           f"outputs/{session_id}/eda_report.json",
}
```

### Three Analyses

#### Analysis 1 — CV/LB Gap Root Cause

```python
cv_mean  = loaded from validation_strategy.json (mean CV score across all folds)
cv_std   = loaded from validation_strategy.json
lb_score = provided by engineer
gap      = abs(cv_mean - lb_score)
```

If `gap > 0.02`, classify the root cause:

| Condition | Root cause | ChromaDB pattern |
|---|---|---|
| Critic said OK + gap > 0.02 | **Critic missed leakage** → write to `critic_failure_patterns` | What feature was likely responsible |
| Critic said CRITICAL + engineer continued + gap > 0.02 | **Known risk materialised** → expected | Confirms the critic was right |
| Critic said HIGH (drift) + gap > 0.02 | **Distribution shift as predicted** | Confirms adversarial classifier was right |
| Critic said OK + gap ≤ 0.02 | **LB shakeup** (private ≠ public) | Note shakeup risk for this fingerprint type |
| CV std > 0.02 | **High variance CV** → model instability | Recommend more folds or ensemble |

#### Analysis 2 — Feature Retrospective

For each feature in `feature_importance.json` (top 20 by importance):
- Was it flagged by any critic vector? (HIGH or CRITICAL)
- Was it in `features_dropped` by the supervisor replan?
- What was its fold-level importance variance? (high variance = unstable feature)

Output: `feature_retrospective.json` with verdict per feature:
- `"helped"` — high importance, critic-clean, low variance → write as validated approach
- `"hurt"` — high importance, critic-flagged, gap large → write as failed approach
- `"noisy"` — high importance but high fold variance → note as unstable

#### Analysis 3 — Pattern Extraction and Memory Write

This is the write step. The post-mortem agent constructs the final `validated_approaches` and `failed_approaches` lists from Analyses 1 and 2, then calls `store_pattern()`.

```python
from memory.memory_schema import store_pattern, build_competition_fingerprint

validated = [
    {"approach": a["approach"], "cv_improvement": a["delta"], "competitions": [competition_name]}
    for a in feature_retrospective if a["verdict"] == "helped"
]
failed = [
    {"approach": a["approach"], "cv_degradation": abs(a["delta"]), "competitions": [competition_name]}
    for a in feature_retrospective if a["verdict"] == "hurt"
]

# Confidence grows with LB performance: top 10% = 0.9, top 25% = 0.75, etc.
percentile  = 1.0 - (lb_rank / total_competitors) if lb_rank else 0.5
confidence  = min(0.9, 0.4 + percentile * 0.5)

pattern_id = store_pattern(
    fingerprint=competition_fingerprint,
    validated_approaches=validated,
    failed_approaches=failed,
    competition_name=competition_name,
    confidence=confidence,
    cv_lb_gap=gap,
)
```

If critic missed leakage (Analysis 1 root cause = "critic missed"), also write to `critic_failure_patterns`:

```python
from memory.memory_schema import store_critic_failure_pattern

store_critic_failure_pattern(
    fingerprint=competition_fingerprint,
    missed_issue=f"CV/LB gap {gap:.3f} not caught by critic. Suspected: {suspected_feature}.",
    competition_name=competition_name,
)
```

### Two ChromaDB Collections After Day 11

| Collection | Written by | Read by | Purpose |
|---|---|---|---|
| `professor_patterns_v2` | `post_mortem_agent` | `ml_optimizer` (warm-start) | What approaches work |
| `critic_failure_patterns` | `post_mortem_agent` | `red_team_critic` (future) | What the critic missed |

The `critic_failure_patterns` collection is Day 11's most forward-looking feature. After 5 competitions, the critic can query this collection before running its own analysis and ask: "have I missed this type of issue before on similar competitions?" This closes the self-improvement loop.

### `post_mortem_report.json` Output

```json
{
  "session_id":          str,
  "competition_name":    str,
  "cv_mean":             float,
  "lb_score":            float,
  "cv_lb_gap":           float,
  "gap_root_cause":      "critic_missed|known_risk|shakeup|acceptable",
  "gap_explanation":     str,
  "feature_retrospective": [...],
  "patterns_written":    int,
  "critic_failures_written": int,
  "pattern_id":          str,
  "confidence":          float,
  "generated_at":        str
}
```

### State Additions

```python
# ProfessorState (post-competition fields):
post_mortem_completed:     bool
post_mortem_report_path:   str
lb_score:                  float
lb_rank:                   int
cv_lb_gap:                 float
gap_root_cause:            str
```

---

## BUILD ORDER (Detailed)

```
Step 1:  Add _check_robustness() to agents/red_team_critic.py
         Add "robustness" to vectors_checked list
         Update vectors_checked contract test to expect 7 vectors
         ── commit: "Day 11: critic vector 4 robustness (noise, slice, calibration)" ──

Step 2:  Add features_dropped and replan_rerun_nodes to ProfessorState + initial_state()
         Build run_supervisor_replan() in agents/supervisor.py
         ── commit: "Day 11: supervisor_replan node" ──

Step 3:  Add _route_from_critic() and _route_from_supervisor_replan() to core/professor.py
         Wire conditional edges in the LangGraph graph
         Update contract test to verify CRITICAL → supervisor_replan, not hitl_handler
         ── commit: "Day 11: critic → supervisor replan wiring in LangGraph" ──

Step 4:  Build agents/post_mortem_agent.py
         Analysis 1: CV/LB gap root cause
         Analysis 2: Feature retrospective
         Analysis 3: Pattern extraction + memory write
         ── commit: "Day 11: post_mortem_agent — closes learning loop" ──

Step 5:  Write tests/test_day11_quality.py
         ── commit: "Day 11: adversarial test suite — all green" ──
```

---

## CONTRACTS THAT CHANGE ON DAY 11

### `tests/contracts/test_critic_contract.py` — ONE TARGETED CHANGE

The Day 10 contract asserts: `CRITICAL verdict → hitl_required=True`.
Day 11 changes this: `CRITICAL verdict → supervisor_replan, NOT hitl_handler directly`.

Update the routing test in the contract:
```python
# OLD (Day 10):
assert result.get("hitl_required") is True

# NEW (Day 11):
# First CRITICAL → supervisor_replan (hitl_required=False, replan_requested=True)
# Only after MAX_REPLAN_ATTEMPTS → hitl_required=True
assert result.get("replan_requested") is True
assert result.get("hitl_required") is not True  # not yet — supervisor gets first attempt
```

This is the only change to an existing immutable contract. It is necessary because the routing logic changed fundamentally. Document it in the contract file header.

### `tests/contracts/test_supervisor_replan_contract.py` — NEW (IMMUTABLE)

New contract for the supervisor_replan node. See test section below.

---

## TEST BLOCKS (53 tests across 5 blocks)

### Block 1 — Critic Vector 4: Robustness (14 tests)

**Class:** `TestCriticVector4Robustness`

Key tests:
- `test_robustness_vector_appears_in_vectors_checked` — 7 vectors now, not 6
- `test_noise_injection_degrades_score_gracefully_for_robust_model` — < 20% degradation on clean model
- `test_noise_injection_triggers_critical_for_overfit_model` — > 20% degradation on noise-memorising model
- `test_slice_audit_finds_performance_gap_across_categorical_slices` — inject category-specific signal
- `test_slice_audit_ok_when_performance_uniform` — uniform performance must not trigger
- `test_ece_over_threshold_triggers_high` — inject badly calibrated probabilities (all 0.9)
- `test_ece_under_threshold_passes` — well-calibrated probabilities pass
- `test_brier_score_versus_random_baseline_computed_correctly` — random Brier = `p*(1-p)` not 0.5
- `test_robustness_verdict_is_max_of_three_subchecks` — one CRITICAL sub-check → CRITICAL robustness verdict
- `test_robustness_skipped_gracefully_when_no_model_available` — no model_registry → OK with note
- `test_noise_injection_uses_feature_stddev_not_absolute_noise` — σ = 10% of feature stddev
- `test_slice_audit_skips_high_cardinality_features` — 200-unique categorical must not be sliced
- `test_calibration_check_skipped_when_no_oof_probs` — y_prob=None → OK
- `test_all_robustness_subchecks_run_even_if_first_fails` — sub-check isolation

---

### Block 2 — Supervisor Replan: Correctness (12 tests)

**Class:** `TestSupervisorReplanCorrectness`

Key tests:
- `test_replan_increments_dag_version_from_current_value` — v2 → v3, not reset to 2
- `test_replan_adds_to_features_dropped_accumulation` — union across replan cycles
- `test_replan_clears_replan_requested_flag` — consumed after use
- `test_replan_clears_hitl_required` — critic set it, supervisor clears for replan pass
- `test_replan_resets_critic_severity_to_unchecked` — critic will re-run fresh
- `test_replan_routes_to_earliest_affected_node` — `{feature_factory, ml_optimizer}` → re-enter at `feature_factory`
- `test_replan_routes_to_data_engineer_when_in_rerun_nodes` — data_engineer is earliest possible
- `test_max_replan_attempts_triggers_hitl` — dag_version >= 3 → hitl_required=True
- `test_max_replan_hitl_reason_mentions_attempt_count` — reason must say "exhausted 3 attempts"
- `test_replan_accumulates_dropped_features_across_cycles` — cycle 1 drops A, cycle 2 drops B → both in state
- `test_replan_does_not_drop_features_not_in_remove_list` — surgical, not wholesale
- `test_second_critic_pass_sees_features_dropped` — after replan, feature_factory should not use dropped features

---

### Block 3 — LangGraph Routing (8 tests)

**Class:** `TestCriticRoutingInLangGraph`

Key tests:
- `test_critical_routes_to_supervisor_replan_not_hitl` — first CRITICAL → supervisor
- `test_high_routes_to_ensemble_not_supervisor` — HIGH → continue
- `test_ok_routes_to_ensemble` — OK → continue
- `test_critical_at_max_attempts_routes_to_hitl` — exhausted → human
- `test_supervisor_replan_routes_to_feature_factory_by_default` — default re-entry
- `test_supervisor_replan_routes_to_data_engineer_when_specified` — correct node priority
- `test_ensemble_runs_when_high_severity_only` — HIGH should not block ensemble
- `test_dag_version_in_lineage_after_replan` — replan event logged to lineage.jsonl

---

### Block 4 — Post-Mortem: Gap Root Cause (11 tests)

**Class:** `TestPostMortemGapRootCause`

Key tests:
- `test_gap_over_threshold_flagged` — gap=0.025 → flagged, gap=0.015 → not flagged
- `test_root_cause_critic_missed_when_critic_ok_and_large_gap` — critic OK + gap 0.03 → "critic_missed"
- `test_root_cause_known_risk_when_critic_high_and_large_gap` — critic warned us → "known_risk"
- `test_root_cause_shakeup_when_critic_ok_and_small_gap` — gap=0.005 → "acceptable"
- `test_critic_failure_pattern_written_when_critic_missed` — ChromaDB write triggered
- `test_no_critic_failure_pattern_when_gap_acceptable` — no false writes
- `test_feature_retrospective_classifies_flagged_high_importance_as_hurt` 
- `test_feature_retrospective_classifies_clean_stable_feature_as_helped`
- `test_confidence_increases_with_lb_percentile` — top 10% → confidence ≥ 0.8
- `test_pattern_id_returned_after_store` — store_pattern called, ID returned
- `test_post_mortem_report_json_written_to_disk` — file exists, all keys present

---

### Block 5 — Integration: Full Loop (8 tests)

**Class:** `TestFullLearningLoop`

Key tests:
- `test_second_competition_has_warm_start_priors` — after post-mortem of comp 1, comp 2 starts with priors
- `test_warm_start_priors_from_structurally_similar_competition` — NLP comp pattern not applied to tabular
- `test_critic_failure_patterns_queryable_from_chromadb` — written by post-mortem, readable
- `test_replan_drops_correct_features_in_feature_factory` — features_dropped actually excluded
- `test_post_mortem_runs_without_competition_rank` — lb_rank=None is graceful
- `test_post_mortem_requires_lb_score` — missing lb_score → ValueError with clear message
- `test_full_pipeline_with_replan_cycle_completes` — CRITICAL → replan → re-run → OK → ensemble
- `test_lineage_contains_replan_event` — lineage.jsonl has dag_replan entry

---

## DEFINITION OF DONE

- [ ] Critic vector 4: noise injection + slice audit + calibration — all three sub-checks run
- [ ] `supervisor_replan` node: reads replan instructions, drops features, increments dag_version, routes correctly
- [ ] LangGraph routing: CRITICAL → supervisor_replan (not hitl), HIGH → ensemble, dag_version ≥ 3 → hitl
- [ ] `post_mortem_agent.py`: 3 analyses, writes to both ChromaDB collections, produces `post_mortem_report.json`
- [ ] `critic_failure_patterns` ChromaDB collection exists and is writable
- [ ] Critic contract test updated: CRITICAL → replan_requested (not hitl_required on first pass)
- [ ] Supervisor contract test: new file, immutable
- [ ] `pytest tests/regression/` — green (Phase 1 + Days 8-10 unchanged)
- [ ] `pytest tests/contracts/` — green (updated critic contract + new supervisor contract)
- [ ] All 53 Day 11 quality tests green

---

## WHAT PODIUM WORK LOOKS LIKE ON THIS DAY

After Day 11, Professor can say the following things it could not say before:

**During a competition (Supervisor replan in action):**
*"Critic found CRITICAL leakage in 'customer_age_encoded'. Dropping feature. Rerunning from feature_factory. DAG version 2."*
*"Critic still finding issues after replan. DAG version 3 — maximum replan attempts reached. Pausing for human review."*

**After a competition closes (post-mortem):**
*"CV/LB gap was 0.031. Critic approved all features. Root cause: critic missed target-encoding leakage on 'product_category'. Writing to critic_failure_patterns."*
*"3 validated approaches written to memory: LGBM + log-transform, temporal aggregation within fold, ordinal encoding for high-cardinality. Confidence 0.82 (top 12% finish)."*

**At start of next structurally similar competition:**
*"Found 2 similar competitions in memory. Warm-start priors: log-transform on skewed target (confidence 0.71), avoid target-encoding before CV split (flagged as critic-missed in Home Credit run). Starting Optuna from these priors."*

The loop is closed. Professor is learning.