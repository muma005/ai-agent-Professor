# Professor — Execution Graph

## Current Graph (as of Day 23)

This is the **actual** execution graph as wired in `core/professor.py`. Not what the README says — what the code does.

```
semantic_router
    │
    ├─→ competition_intel ─┐
    │                       │ (fan-join: both must complete)
    ├─→ data_engineer ─────┘
    │
    ├─→ eda_agent
    │
    ├─→ validation_architect ──→ [HITL?] ──→ END (if hitl_required)
    │
    ├─→ feature_factory
    │
    ├─→ ml_optimizer
    │
    ├─→ ensemble_architect  (calls blend_models)
    │
    ├─→ red_team_critic ──→ [severity?]
    │        │
    │        ├─ CRITICAL + replan_attempts < 3 → supervisor_replan → re-enter at earliest affected node
    │        └─ HIGH / MEDIUM / OK → submit
    │
    └─→ submit ──→ END
```

## Node-by-Node Detail

### 1. `semantic_router` (supervisor)
- **Role:** Plans the DAG, sets `state["dag"]`, routes to first node
- **Reads:** `competition_name`, `raw_data_path`
- **Writes:** `dag`, `current_node`, `competition_context`
- **Depends on:** Nothing (entry point)

### 2. `competition_intel` (parallel branch A)
- **Role:** Scrapes competition forums, writes `intel_brief.json`
- **Reads:** `competition_name`
- **Writes:** `competition_brief_path`, `competition_brief`
- **Depends on:** `semantic_router`

### 3. `data_engineer` (parallel branch B)
- **Role:** CSV → Parquet, schema inference, data hash, preprocessor fit
- **Reads:** `raw_data_path`, `test_data_path`, `sample_submission_path`
- **Writes:** `clean_data_path`, `schema_path`, `preprocessor_path`, `data_hash`, `target_col`, `id_columns`
- **Depends on:** `semantic_router`
- **Fan-join gate:** Both `competition_intel` AND `data_engineer` must complete before `eda_agent` runs

### 4. `eda_agent`
- **Role:** Target distribution, correlations, outliers, leakage fingerprint
- **Reads:** `clean_data_path`, `schema_path`
- **Writes:** `eda_report_path`, `eda_report`, `dropped_features`
- **Depends on:** `data_engineer` (via fan-join)

### 5. `validation_architect`
- **Role:** CV strategy selection, metric contract
- **Reads:** `clean_data_path`, `schema_path`, `eda_report`
- **Writes:** `cv_strategy`, `metric_contract`
- **Depends on:** `eda_agent`
- **HITL gate:** If `hitl_required=True`, pipeline halts → END

### 6. `feature_factory`
- **Role:** Hypothesis-driven feature generation, interaction cap
- **Reads:** `clean_data_path`, `schema_path`, `feature_candidates`
- **Writes:** `feature_manifest`, `feature_candidates`, `feature_data_path`
- **Depends on:** `validation_architect`

### 7. `ml_optimizer`
- **Role:** Optuna HPO, multi-seed stability, calibration
- **Reads:** `feature_data_path`, `schema_path`, `metric_contract`
- **Writes:** `model_registry`, `cv_mean`, `cv_std`, `oof_predictions_path`, `feature_order`
- **Depends on:** `feature_factory`

### 8. `ensemble_architect` (calls `blend_models`)
- **Role:** Diversity-first ensemble selection, OOF blending
- **Reads:** `model_registry`, `data_hash`, `y_train`
- **Writes:** `selected_models`, `ensemble_weights`, `ensemble_oof`, `ensemble_selection`
- **Depends on:** `ml_optimizer`

### 9. `red_team_critic`
- **Role:** 4-vector adversarial audit (shuffled_target, preprocessing_audit, pr_curve_imbalance, temporal_leakage)
- **Reads:** `model_registry`, `X_train`, `y_train`, `feature_names`
- **Writes:** `critic_verdict`, `critic_severity`, `replan_requested`, `replan_remove_features`
- **Depends on:** `ensemble_architect`
- **Conditional routing:**
  - `CRITICAL` + `dag_version < 3` → `supervisor_replan` → re-enter at earliest affected node
  - `CRITICAL` + `dag_version >= 3` → END (HITL)
  - `HIGH` / `MEDIUM` / `OK` → `submit`

### 10. `supervisor_replan`
- **Role:** Decides which nodes to re-run based on critic verdict
- **Reads:** `critic_verdict`, `replan_remove_features`, `replan_rerun_nodes`
- **Writes:** `features_dropped`, `dag` (modified), `dag_version`
- **Depends on:** `red_team_critic` (only on CRITICAL)

### 11. `submit`
- **Role:** Generates validated `submission.csv` from best model
- **Reads:** `model_registry`, `test_data_path`, `sample_submission_path`, `preprocessor_path`
- **Writes:** `submission_path`, `submission_log`
- **Depends on:** `red_team_critic` (non-CRITICAL) or `supervisor_replan` loop exit

---

## Agents NOT in the Graph (dead or superseded code)

These files exist in `agents/` but are **not wired into `build_graph()`**:

| File | Status | Notes |
|------|--------|-------|
| `ensemble_optimizer.py` | 🔴 Dead | Superseded by `ensemble_architect.py` (Day 22) |
| `stacking_agent.py` | 🔴 Dead | Superseded by `ensemble_architect.py` (Day 22 meta-learner) |
| `hpo_agent.py` | 🔴 Dead | Superseded by `ml_optimizer.py` (Optuna built-in) |
| `feature_selector.py` | 🟡 Unused | Null importance logic moved into `feature_factory.py` |
| `publisher.py` | 🟢 New (Day 23) | Not yet wired into graph — called manually |
| `qa_gate.py` | 🟢 New (Day 23) | Not yet wired into graph — called manually |
| `submission_strategist.py` | 🟢 New (Day 23) | Not yet wired into graph — replaces `submit` node |

---

## State Dependencies (who reads/writes what)

The most contested state keys:

| State Key | Written By | Read By | Risk |
|-----------|-----------|---------|------|
| `model_registry` | `ml_optimizer` | `ensemble_architect`, `submit`, `red_team_critic` | 🟡 High |
| `data_hash` | `data_engineer` | `ensemble_architect` | 🟢 Low |
| `feature_order` | `ml_optimizer` | `submit` | 🟢 Low |
| `critic_severity` | `red_team_critic` | routing logic | 🟡 Medium |
| `ensemble_oof` | `ensemble_architect` | `submission_strategist` (Day 23) | 🟢 Low |
| `cv_mean` | `ml_optimizer` | `submit`, `submission_strategist` | 🟢 Low |
| `dag` | `semantic_router`, `supervisor_replan` | all routing functions | 🟡 Medium |
| `feature_manifest` | `feature_factory` | `ml_optimizer` | 🟢 Low |

---

## Day 23 Agents — Not Yet Wired

`submission_strategist`, `publisher`, and `qa_gate` are implemented but not yet integrated into the LangGraph DAG. They are designed to replace/extend the current `submit` node:

```
Current:  ensemble_architect → red_team_critic → submit → END

Planned:  ensemble_architect → red_team_critic → submission_strategist → publisher → qa_gate → END
```
