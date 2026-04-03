# Day 24 — Complexity Neutralisation Plan

## Problem Statement

19 agents exist. 11 are wired into the graph. 4 are dead code. 3 are Day 23 additions not yet wired. State coupling between agents has no runtime enforcement — the `StateValidator` exists in `core/state_validator.py` but is **never called** from `core/professor.py`.

---

## Phase A — Remove Dead Agents (Zero Risk)

These 4 agents are imported **only by their own test files**. No production code references them.

| File | Tests | Superseded By |
|------|-------|---------------|
| `agents/hpo_agent.py` | `tests/agents/test_hpo_agent.py` | `ml_optimizer.py` (Optuna built-in) |
| `agents/ensemble_optimizer.py` | `tests/agents/test_ensemble_optimizer.py` | `ensemble_architect.py` (Day 22) |
| `agents/stacking_agent.py` | `tests/agents/test_stacking_agent.py` | `ensemble_architect.py` (Day 22 meta-learner) |
| `agents/feature_selector.py` | `tests/agents/test_feature_selector.py` | `feature_factory.py` (null importance moved in) |

### Execution (1 commit each, 8 files total):

```
# Commit 1: Remove hpo_agent + its test
git rm agents/hpo_agent.py tests/agents/test_hpo_agent.py

# Commit 2: Remove ensemble_optimizer + its test
git rm agents/ensemble_optimizer.py tests/agents/test_ensemble_optimizer.py

# Commit 3: Remove stacking_agent + its test
git rm agents/stacking_agent.py tests/agents/test_stacking_agent.py

# Commit 4: Remove feature_selector + its test
git rm agents/feature_selector.py tests/agents/test_feature_selector.py
```

**Risk: ZERO.** These files have zero importers outside their own tests. Removing them cannot break any production path.

**Verification after each commit:**
```bash
pytest tests/contracts/ -v --tb=short    # all contracts must pass
pytest tests/regression/ -v --tb=short   # all regression must pass
```

---

## Phase B — Wire State Validator Into Pipeline (Medium Risk → Eliminated)

The `StateValidator` in `core/state_validator.py` already has:
- Full schema with types for every state key
- Per-stage required keys (`PIPELINE_STAGE_REQUIREMENTS`)
- Type checking + missing key detection

**It just isn't called.** The fix is to wire it into the `_advance_dag()` function — the single choke point every node passes through.

### Changes (1 commit, 1 file):

**File: `core/professor.py`**

In `_advance_dag()`, add validation between nodes:

```python
def _advance_dag(state: ProfessorState, current: str) -> str:
    """Find current node in DAG, validate its outputs, return next node."""
    if state.get("pipeline_halted") or state.get("triage_mode"):
        return END

    dag = state.get("dag", [])
    if current not in dag:
        return END

    idx = dag.index(current)

    # ── NEW: Validate state after each node ──────────────────────
    stage_map = {
        "competition_intel":    "post_competition_intel",
        "data_engineer":        "post_data_engineer",
        "eda_agent":            "post_eda_agent",
        "validation_architect": "post_validation_architect",
        "feature_factory":      "post_feature_factory",
        "ml_optimizer":         "post_ml_optimizer",
        "ensemble_architect":   "post_ensemble_architect",
        "submit":               "post_submit",
    }
    stage = stage_map.get(current)
    if stage:
        from core.state_validator import validate_state
        validate_state(state, stage=stage, node_name=current, strict=False)
        # strict=False → log errors but don't crash the pipeline

    if idx + 1 >= len(dag):
        return END

    return dag[idx + 1]
```

**Why this is safe:**
- `strict=False` means validation failures log warnings but don't halt the pipeline
- The validator only checks keys that should exist — it never modifies state
- If a key is missing, the downstream agent will fail anyway (same failure, but now we know *why* earlier)
- Zero changes to any agent code — purely additive

**Verification:**
```bash
python -c "from core.professor import build_graph; print('Graph builds OK')"
```

---

## Phase C — Wire Day 23 Agents Into Graph (Medium Risk → Eliminated)

Replace the current `submit` node with the Day 23 pipeline:

```
Before:  ensemble_architect → red_team_critic → submit → END
After:   ensemble_architect → red_team_critic → submission_strategist → publisher → qa_gate → END
```

### Changes (1 commit, 1 file):

**File: `core/professor.py`**

1. Add imports:
```python
from agents.submission_strategist import run_submission_strategist
from agents.publisher import run_publisher
from agents.qa_gate import run_qa_gate
```

2. Add nodes:
```python
graph.add_node("submission_strategist", run_submission_strategist)
graph.add_node("publisher", run_publisher)
graph.add_node("qa_gate", run_qa_gate)
```

3. Replace `submit` routing:
```python
# Critic → submission_strategist (instead of submit)
# route_after_critic: change "submit" → "submission_strategist"

# New edges:
graph.add_conditional_edges("submission_strategist", route_after_strategist, _all_nodes)
graph.add_conditional_edges("publisher", route_after_publisher, _all_nodes)
graph.add_conditional_edges("qa_gate", route_after_qa_gate, _all_nodes)
graph.add_edge("qa_gate", END)
```

4. Add routing functions:
```python
def route_after_strategist(state): return _advance_dag(state, "submission_strategist")
def route_after_publisher(state):  return _advance_dag(state, "publisher")
def route_after_qa_gate(state):    return _advance_dag(state, "qa_gate")
```

**Why this is safe:**
- The old `submit` node is not deleted — it stays as a fallback
- Day 23 agents already have 46 passing tests
- The `submission_strategist` writes `submission_path` (same key as `submit`), so downstream consumers don't break

---

## Phase D — Update State Validator for Day 23 (Low Risk)

Add Day 23 stage requirements to `core/state_validator.py`:

```python
PIPELINE_STAGE_REQUIREMENTS = {
    # ... existing stages ...
    "post_submission_strategist": {
        "required": [
            "submission_a_path", "submission_b_path", "submission_path",
            "submission_a_model", "submission_b_model",
            "submission_freeze_active",
        ],
    },
    "post_publisher": {
        "required": ["report_path", "report_written"],
    },
    "post_qa_gate": {
        "required": ["qa_passed", "qa_failures"],
    },
}
```

---

## Execution Order

```
Phase A1: Remove hpo_agent          → commit → verify tests
Phase A2: Remove ensemble_optimizer → commit → verify tests
Phase A3: Remove stacking_agent     → commit → verify tests
Phase A4: Remove feature_selector   → commit → verify tests
Phase B:  Wire state validator      → commit → verify graph builds
Phase C:  Wire Day 23 agents        → commit → verify all tests
Phase D:  Update validator schema   → commit → verify all tests
```

**Total: 7 commits. 0 production code changes (only additions). 4 files deleted. 1 file modified (professor.py).**

---

## Risk Matrix

| Phase | Risk | Mitigation | Rollback |
|-------|------|------------|----------|
| A (dead code removal) | None | Only files with zero production importers | `git revert` restores files |
| B (state validator) | Low | `strict=False` — logs only, never crashes | Remove 6 lines from `_advance_dag` |
| C (wire Day 23) | Low | Old `submit` node preserved; new agents have 46 passing tests | Revert routing changes |
| D (validator schema) | None | Purely additive — adds keys, removes nothing | Revert schema additions |

---

## What This Does NOT Touch

- No agent logic changes
- No state structure changes
- No test changes (except deleting dead agent tests)
- No `requirements.txt` changes
- No `core/state.py` changes

The plan is purely: **delete dead code, wire existing infrastructure, wire existing agents.**
