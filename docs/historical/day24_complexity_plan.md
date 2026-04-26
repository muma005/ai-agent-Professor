# Day 24 — Complexity Neutralisation Plan

## Problem Statement

19 agents exist. 12 are in the graph. 4 are dead code. 3 are Day 23 additions not wired. 1 node (`pseudo_label_agent`) has **no outgoing edge** — if reached, the graph hangs. State coupling has zero runtime enforcement.

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
git rm agents/hpo_agent.py tests/agents/test_hpo_agent.py
git rm agents/ensemble_optimizer.py tests/agents/test_ensemble_optimizer.py
git rm agents/stacking_agent.py tests/agents/test_stacking_agent.py
git rm agents/feature_selector.py tests/agents/test_feature_selector.py
```

**Risk: ZERO.** Zero production importers. Removing cannot break any path.

---

## Phase B — State Validation Between Agents (Medium Risk → Eliminated)

### The Problem

11 agents pass state through a shared dict. Zero runtime enforcement. If `ml_optimizer` forgets `feature_order`, `submit` crashes with `KeyError` three nodes later. `StateValidator` exists in `core/state_validator.py` but is **never called** from `core/professor.py`.

### What Already Exists

`core/state_validator.py` has:
- `STATE_SCHEMA` — type definitions for 70+ state keys
- `PIPELINE_STAGE_REQUIREMENTS` — required keys per stage (9 stages defined)
- `validate_state()` — convenience function, never called

### The Plan

Wire `validate_state()` into `_advance_dag()` — the single choke point every node passes through.

#### Step 1: Add `_validate_node_output()` to `core/professor.py`

```python
_NODE_TO_STAGE = {
    "competition_intel":    "post_competition_intel",
    "data_engineer":        "post_data_engineer",
    "eda_agent":            "post_eda_agent",
    "validation_architect": "post_validation_architect",
    "feature_factory":      "post_feature_factory",
    "ml_optimizer":         "post_ml_optimizer",
    "ensemble_architect":   "post_ensemble_architect",
    "submit":               "post_submit",
}

def _validate_node_output(state: ProfessorState, node_name: str) -> None:
    stage = _NODE_TO_STAGE.get(node_name)
    if not stage:
        return
    try:
        from core.state_validator import validate_state
        validate_state(state, stage=stage, node_name=node_name, strict=False)
    except Exception as e:
        logger.warning(f"[Professor] State validation failed after '{node_name}': {e}")
```

#### Step 2: Call it in `_advance_dag()`

```python
def _advance_dag(state: ProfessorState, current: str) -> str:
    # ... existing halt/triage checks ...
    idx = dag.index(current)

    _validate_node_output(state, current)  # ← NEW

    if idx + 1 >= len(dag):
        return END
    return dag[idx + 1]
```

#### Step 3: Add Day 23 stages to `core/state_validator.py`

```python
"post_submission_strategist": {
    "required": ["submission_a_path", "submission_b_path", "submission_path",
                 "submission_a_model", "submission_b_model", "submission_freeze_active"],
},
"post_publisher": {
    "required": ["report_path", "report_written"],
},
"post_qa_gate": {
    "required": ["qa_passed", "qa_failures"],
},
```

**Why safe:** `strict=False` → logs only, never crashes. Wrapped in `try/except`. Zero agent changes.

**Rollback:** Delete 1 line from `_advance_dag()`.

---

## Phase C — Fix Broken Graph Connections (Medium Risk → Eliminated)

### Current Graph — What's Actually Wired

```
semantic_router ──→ competition_intel ─┐
                  ──→ data_engineer ───┘ (fan-join)
                        │
                        ▼
                   eda_agent
                        │
                        ▼
              validation_architect ──→ [HITL?] ──→ END
                        │
                        ▼
                feature_factory
                        │
                        ▼
                 ml_optimizer
                        │
                        ▼
              ensemble_architect
                        │
                        ▼
               red_team_critic ──→ [severity?]
                    │                    │
                    ├─ CRITICAL + replan < 3 → supervisor_replan → re-enter DAG
                    ├─ CRITICAL + replan ≥ 3 → END (HITL)
                    └─ HIGH/MEDIUM/OK ──→ submit ──→ END
```

### Broken / Missing Connections

| Issue | Severity | Detail |
|-------|----------|--------|
| `pseudo_label_agent` has NO outgoing edge | 🔴 **Graph hang** | Registered as node, no `add_edge` or `add_conditional_edges`. If DAG ever routes here, execution stops silently. |
| `submit` node is inline in `professor.py` | 🟡 Maintenance | Works fine, but should be in `agents/` for consistency. |
| Day 23 agents not wired | 🟡 Incomplete | `submission_strategist`, `publisher`, `qa_gate` all have proper `run_X(state)` signatures but zero graph presence. |
| `ensemble_architect` imported as `blend_models` | 🟡 Confusing | `run_ensemble_architect` exists but graph uses old `blend_models` function. |

### The Fix — Complete Rewired Graph

```
semantic_router ──→ competition_intel ─┐
                  ──→ data_engineer ───┘ (fan-join)
                        │
                        ▼
                   eda_agent
                        │
                        ▼
              validation_architect ──→ [HITL?] ──→ END
                        │
                        ▼
                feature_factory
                        │
                        ▼
                 ml_optimizer
                        │
                        ▼
              ensemble_architect
                        │
                        ▼
               red_team_critic ──→ [severity?]
                    │                    │
                    ├─ CRITICAL + replan < 3 → supervisor_replan → re-enter DAG
                    ├─ CRITICAL + replan ≥ 3 → END (HITL)
                    └─ HIGH/MEDIUM/OK ──→ submission_strategist
                                               │
                                               ▼
                                          publisher
                                               │
                                               ▼
                                          qa_gate ──→ END
```

### Changes to `core/professor.py` (1 commit)

#### 1. Add imports

```python
from agents.submission_strategist import run_submission_strategist
from agents.publisher import run_publisher
from agents.qa_gate import run_qa_gate
```

#### 2. Add nodes

```python
graph.add_node("submission_strategist", run_submission_strategist)
graph.add_node("publisher", run_publisher)
graph.add_node("qa_gate", run_qa_gate)
```

#### 3. Add to `_all_nodes`

```python
_all_nodes = {
    # ... existing 12 entries ...
    "submission_strategist": "submission_strategist",
    "publisher":             "publisher",
    "qa_gate":               "qa_gate",
}
```

#### 4. Fix `route_after_critic` — route to `submission_strategist` instead of `submit`

```python
def route_after_critic(state: ProfessorState) -> str:
    # ... existing severity checks ...
    # HIGH, MEDIUM, OK: continue to submission_strategist
    return "submission_strategist"  # was "submit"
```

#### 5. Add routing functions

```python
def route_after_strategist(state: ProfessorState) -> str:
    return _advance_dag(state, current="submission_strategist")

def route_after_publisher(state: ProfessorState) -> str:
    return _advance_dag(state, current="publisher")

def route_after_qa_gate(state: ProfessorState) -> str:
    return _advance_dag(state, current="qa_gate")
```

#### 6. Add conditional edges

```python
graph.add_conditional_edges(
    "submission_strategist", route_after_strategist, _all_nodes,
)
graph.add_conditional_edges(
    "publisher", route_after_publisher, _all_nodes,
)
graph.add_conditional_edges(
    "qa_gate", route_after_qa_gate, _all_nodes,
)
graph.add_edge("qa_gate", END)
```

#### 7. Fix `pseudo_label_agent` — add outgoing edge

```python
def route_after_pseudo_label(state: ProfessorState) -> str:
    return _advance_dag(state, current="pseudo_label_agent")

graph.add_conditional_edges(
    "pseudo_label_agent", route_after_pseudo_label, _all_nodes,
)
```

#### 8. Fix `ensemble_architect` import — use `run_ensemble_architect`

```python
# Before:
from agents.ensemble_architect import blend_models
graph.add_node("ensemble_architect", blend_models)

# After:
from agents.ensemble_architect import run_ensemble_architect
graph.add_node("ensemble_architect", run_ensemble_architect)
```

### Why This Is Safe

| Concern | Mitigation |
|---------|-----------|
| Old `submit` node breaks? | Not deleted — stays in graph as fallback. Just not the default route anymore. |
| Day 23 agents untested in graph? | 46 passing unit tests already. Graph wiring is just `add_node` + `add_conditional_edges`. |
| `pseudo_label_agent` fix could break existing runs? | It had no edge before — it was already broken. Adding an edge fixes it. |
| `ensemble_architect` function signature change? | `run_ensemble_architect` returns `dict` (same as `blend_models`). LangGraph merges dict into state. |

---

## Phase D — Update State Validator for Day 23 (Low Risk)

Add Day 23 stage requirements to `core/state_validator.py` (covered in Phase B Step 3).

Also add `pseudo_label_agent` stage:

```python
"post_pseudo_label_agent": {
    "required": ["pseudo_labels_applied", "pseudo_label_cv_improvement"],
},
```

---

## Execution Order

```
Phase A1: Remove hpo_agent          → commit → verify tests
Phase A2: Remove ensemble_optimizer → commit → verify tests
Phase A3: Remove stacking_agent     → commit → verify tests
Phase A4: Remove feature_selector   → commit → verify tests
Phase B:  Wire state validator      → commit → verify graph builds
Phase C:  Fix all graph connections → commit → verify all tests
Phase D:  Update validator schema   → commit → verify all tests
```

**Total: 7 commits. 0 agent logic changes. 8 files deleted. 2 files modified (`professor.py`, `state_validator.py`).**

---

## Complete Agent Inventory (After Cleanup)

| # | Agent | In Graph? | Has Outgoing Edge? | Status |
|---|-------|-----------|-------------------|--------|
| 1 | `semantic_router` | ✅ | ✅ `route_after_router` | OK |
| 2 | `competition_intel` | ✅ | ✅ `route_after_intel` | OK |
| 3 | `data_engineer` | ✅ | ✅ `route_after_data_engineer` | OK |
| 4 | `eda_agent` | ✅ | ✅ `route_after_eda` | OK |
| 5 | `validation_architect` | ✅ | ✅ `route_after_validation` | OK |
| 6 | `feature_factory` | ✅ | ✅ `route_after_feature_factory` | OK |
| 7 | `ml_optimizer` | ✅ | ✅ `route_after_optimizer` | OK |
| 8 | `ensemble_architect` | ✅ | ✅ `route_after_ensemble` | OK (fix import) |
| 9 | `red_team_critic` | ✅ | ✅ `route_after_critic` | OK |
| 10 | `supervisor_replan` | ✅ | ✅ `route_after_supervisor_replan` | OK |
| 11 | `pseudo_label_agent` | ✅ | ❌ → ✅ `route_after_pseudo_label` | **FIX: add edge** |
| 12 | `submit` | ✅ | ✅ `add_edge("submit", END)` | OK (legacy fallback) |
| 13 | `submission_strategist` | ❌ → ✅ | ❌ → ✅ `route_after_strategist` | **WIRE: replace submit** |
| 14 | `publisher` | ❌ → ✅ | ❌ → ✅ `route_after_publisher` | **WIRE: after strategist** |
| 15 | `qa_gate` | ❌ → ✅ | ❌ → ✅ `route_after_qa_gate` | **WIRE: terminal node** |
| ~~16~~ | ~~`hpo_agent`~~ | — | — | **DELETE** |
| ~~17~~ | ~~`ensemble_optimizer`~~ | — | — | **DELETE** |
| ~~18~~ | ~~`stacking_agent`~~ | — | — | **DELETE** |
| ~~19~~ | ~~`feature_selector`~~ | — | — | **DELETE** |
| 20 | `post_mortem_agent` | No | N/A | Intentionally manual |

**After cleanup: 15 agents. 15 in graph. 15 with outgoing edges. Zero hanging nodes.**

---

## Risk Matrix

| Phase | Risk | Mitigation | Rollback |
|-------|------|------------|----------|
| A (dead code removal) | None | Zero production importers | `git revert` restores files |
| B (state validator) | Low | `strict=False` — logs only | Remove 1 line from `_advance_dag` |
| C (graph connections) | Low | Old `submit` preserved; Day 23 agents have 46 passing tests | Revert routing changes |
| D (validator schema) | None | Purely additive | Revert schema additions |

---

## What This Does NOT Touch

- No agent logic changes
- No state structure changes (`core/state.py` untouched)
- No test changes (except deleting dead agent tests)
- No `requirements.txt` changes

The plan is purely: **delete dead code, wire existing infrastructure, fix broken edges.**
