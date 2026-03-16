# Phase 2 Implementation Summary — "Make It Smart"
**Days 8–15 | March 10–15, 2026**
**Branch:** `phase-2`
**Final commit:** `688c6d8`

---

## Phase 2 Objective

Transform Professor from "runs end-to-end" (Phase 1) to "decides intelligently" — with intelligence agents, quality auditing, resilience, self-healing, observability, and competition memory.

---

## Day-by-Day Implementation

### Day 8 — Phase 2 Kickoff (commit `c95ecc5`)
**Theme:** Build intelligence and quality agents

**Tasks completed:**
1. **ChromaDB embedding fix** (`memory/chroma_client.py`) — Validated `bge-small-en-v1.5` (384-dim) on init. Prevents silent fallback to random embeddings that corrupts all memory queries.
2. **State extensions** (`core/state.py`) — Added `task_type` (Literal), `data_hash` (SHA-256[:16]), `competition_context` fields.
3. **Validation Architect** (`agents/validation_architect.py`) — Deterministic CV strategy selection (StratifiedKFold / GroupKFold / TimeSeriesSplit / KFold). CV/LB mismatch detection with HITL escalation.
4. **EDA Agent** (`agents/eda_agent.py`) — Target distribution, correlation analysis, outlier profiling, leakage fingerprinting, duplicate/ID-conflict detection.
5. **Competition Intel upgrade** (`agents/competition_intel.py`) — GM-CAP 1: forum scraper for top Kaggle notebooks with LLM brief synthesis.
6. **Contract test:** `test_validation_architect_contract.py` (IMMUTABLE)

**Tests:** 57 new, 66 total after merge.

---

### Day 9 — Resilience Layer (commit `010371d`)
**Theme:** Armour the pipeline against failures

**Tasks completed:**
1. **Circuit Breaker** (`guards/circuit_breaker.py`) — 4-level escalation: MICRO (patch node) → MACRO (rewrite DAG) → HITL (checkpoint to Redis, halt) → TRIAGE (protect budget/submission).
2. **Subprocess Sandbox** (`tools/e2b_sandbox.py`) — Replaced RestrictedPython (blocks C-extensions) with subprocess sandbox supporting numpy, polars, LightGBM.
3. **Service Health** (`guards/service_health.py`) — Provider fallback chain (Groq→Gemini), exponential backoff, ChromaDB/Redis graceful degradation.
4. **Agent Retry** (`guards/agent_retry.py`) — Inner retry loop (3 attempts) with LLM error context feedback for all 8 LLM-calling agents.
5. **Parallel Execution Groups** (`core/professor.py`) — Fan-out/join design for intelligence, model trials, and critic vectors.

**Tests:** 54 resilience tests (52 pass, 2 skip — Docker Redis not running).

---

### Day 10 — Quality Conscience (commit `610b7af`)
**Theme:** Professor learns to catch its own mistakes

**Tasks completed:**
1. **Memory Schema v2** (`memory/memory_schema.py`) — Competition fingerprints (task_type, imbalance_ratio, n_rows_bucket, etc.) with NL embeddings for semantic retrieval. Patterns replace raw hyperparams (hyperparams don't transfer between competitions, patterns do).
2. **Red Team Critic — 6 vectors** (`agents/red_team_critic.py`):
   - Vector 1: Shuffled target detection (is model better than random?)
   - Vector 2: ID-only model detection (model memorising row IDs)
   - Vector 3: Adversarial classifier (train vs test distribution shift)
   - Vector 4: Preprocessing code audit (regex-based leakage detection)
   - Vector 5: PR curve audit for imbalanced datasets
   - Vector 6: Temporal leakage check
3. **Severity routing:** CRITICAL → hitl_required + replan_requested; HIGH/MEDIUM → log + continue.
4. **Contract test:** `test_critic_contract.py` (IMMUTABLE)

**Tests:** 53 adversarial quality tests, all green.

---

### Day 11 — Learning Loop (commit `3c267c6`)
**Theme:** Close the feedback loop — Professor gets smarter with every competition

**Tasks completed:**
1. **Critic Vector 7: Robustness** (`agents/red_team_critic.py`) — Three sub-checks:
   - Gaussian noise injection (overfit detection — 20% degradation → CRITICAL)
   - Slice performance audit (subgroup AUC spread > 0.15 → HIGH)
   - OOF calibration (ECE + Brier Score for probability-evaluated competitions)
2. **Supervisor Replan** (`core/professor.py` + `agents/supervisor.py`) — CRITICAL verdict → automatic DAG rewrite (drop bad features, re-enter at affected node, increment `dag_version`). Max 3 replans before HITL. Handles the 80% case where fixes are mechanical.
3. **Post-Mortem Agent** (`agents/post_mortem_agent.py`) — CV/LB gap root cause analysis, feature retrospective, pattern extraction → `professor_patterns_v2` + `critic_failure_patterns` ChromaDB collections. Feeds lessons back into the critic for future competitions.

**Tests:** 66 new tests, all green.

---

### Day 12 — Podium-Level Hardening (commit `b9e13c8`)
**Theme:** Survive overnight unsupervised runs

**Tasks completed:**
1. **HITL Human Layer** (`guards/circuit_breaker.py`):
   - `generate_hitl_prompt()` — 5 error classes (data_quality, model_failure, memory, api_timeout, unknown), 3 structured interventions per class, terminal banner
   - `resume_from_checkpoint()` — AUTO/MANUAL intervention application from Redis checkpoint
2. **OOM Prevention** (`agents/ml_optimizer.py`):
   - Per-fold memory check via psutil (TrialPruned on threshold)
   - `del models` in finally block (not try — never runs on exception)
   - `gc_after_trial`, `n_jobs=1` (prevents 8x memory multiplier)
3. **LangSmith Cost Control** (`core/professor.py`):
   - Tracing disabled during Optuna loop (prevents 200+ trace events per run)
   - Sampling rate from env var (default 0.10 — 10% of runs traced)
4. **Contract tests:** test_hitl_prompt_contract.py, test_resume_checkpoint_contract.py (IMMUTABLE)

**Bugs fixed:** `del models` in try → finally, env var restore "false" for absent key, 50KB HITL event truncation, n_jobs=-1 memory explosion.

---

### Day 13 — Submission Integrity (commits `23d4b17`, `a42a938`)
**Theme:** Silent bugs that kill LB score after correct CV

**Tasks completed:**
1. **Feature Order Enforcement** (`agents/ml_optimizer.py` + `tools/submit_tools.py`):
   - Save `feature_order` (exact ordered column list) to `metrics.json` after training
   - Enforce at submit time — raises ValueError on mismatch instead of silent wrong predictions
   - Catches Polars CSV column reordering bug
2. **Data Hash Validation** (`agents/ensemble_architect.py`):
   - Validates all models in registry trained on same data version before blending
   - Filters out stale-data models automatically, logs warning
3. **Wilcoxon Gate** (`tools/wilcoxon_gate.py`):
   - Wilcoxon signed-rank test for statistically rigorous model comparison (p < 0.05)
   - Non-parametric — no normality assumption on CV fold scores
   - Plugged into ml_optimizer model selection
4. **Graph wiring stabilisation** — routing map fix for conditional edges

**Tests:** 55 quality tests, all green.

---

### Day 14 — Compounding Advantage + Phase 2 Gate (commit `57294eb`)
**Theme:** The critic grows smarter with every competition

**Tasks completed:**
1. **Critic Vector 8: Historical Failures** (`agents/red_team_critic.py`):
   - Queries `critic_failure_patterns` ChromaDB collection for structurally similar past competitions
   - Flags features matching known failure modes with confidence-based severity (≥0.85 → CRITICAL, ≥0.70 → HIGH, ≥0.50 → MEDIUM)
   - Compounding advantage — static rules stay at 7, battle-tested rules grow without bound
2. **Phase 2 Gate** (`tests/phase2_gate.py`):
   - Condition 1: Critic catches injected leakage (target-derived feature → CRITICAL)
   - Condition 2: Validation Architect blocks wrong metric (AUC on regression → error)
   - Condition 3: End-to-end CV beats Phase 1 baseline by ≥ 0.005
3. **Phase 2 Regression Test** (`tests/regression/test_phase2_regression.py`) — FROZEN, IMMUTABLE

**Bugs fixed:** Flaky ChromaDB round-trip test (query top-N on small collection).

---

### Day 15 — Phase 2 Finale (commit `688c6d8`)
**Theme:** Infrastructure for Phase 3

**Tasks completed:**
1. **Graph Singleton** (`core/professor.py`):
   - Module-level singleton with thread-safe double-checked locking
   - Prevents 2-4s recompilation overhead per invocation (catastrophic in retry loops)
   - `get_graph_cache_clear()` for test isolation
2. **Docker Sandbox** (`tools/e2b_sandbox.py`):
   - `python:3.11-slim` containers, `--network none`, `--read-only`, `--tmpfs /tmp:rw,size=512m`
   - Memory/CPU limits, `--security-opt no-new-privileges`
   - Falls back to subprocess when Docker unavailable
3. **LangFuse Observability** (`core/professor.py`):
   - Full token counts, node-level spans, tool call logging
   - Graceful degradation if keys absent — JSONL lineage always active
   - Coexists with LangSmith (different responsibilities)
4. **External Data Scout** (`agents/competition_intel.py`):
   - `run_external_data_scout()` — LLM-driven identification of useful external data
   - Gated by `state["external_data_allowed"]` — no side effects if disabled
5. **Contract test:** test_competition_intel_contract.py (IMMUTABLE)

**Known issue:** `test_data_engineer_logs_high_relevance_sources` fails — mock target mismatch (`agents.data_engineer.log_event` not imported at module level).

---

## Architecture at Phase 2 End

### Pipeline Flow
```
semantic_router → competition_intel → data_engineer → eda_agent
    → validation_architect → ml_optimizer → red_team_critic
    ├── OK/MEDIUM/HIGH → submit
    ├── CRITICAL (≤3 times) → supervisor_replan → re-enter at affected node
    └── CRITICAL (>3 times) → HITL escalation
```

### Operational Agents (10 implemented)
| Agent | Responsibility |
|-------|---------------|
| semantic_router | Task detection, DAG selection, strategy |
| competition_intel | Kaggle scraping, LLM brief, external data scout |
| data_engineer | CSV → Polars → cleaned.parquet + schema.json |
| eda_agent | Target/correlation/outlier/leakage analysis |
| validation_architect | CV strategy selection, metric/target validation |
| ml_optimizer | Optuna + LightGBM, OOM guard, Wilcoxon gate |
| red_team_critic | 8-vector quality gate, severity routing |
| ensemble_architect | Data hash validation, equal-weight blending |
| post_mortem_agent | CV/LB gap root cause, pattern memory |
| supervisor | DAG replan orchestration (max 3) |

### Stub Agents (Phase 3+)
- `feature_factory.py` — pass-through placeholder
- `publisher.py`, `pseudo_label_agent.py`, `submission_strategist.py`, `qa_gate.py` — empty

### Guards
| Guard | Status |
|-------|--------|
| circuit_breaker | Complete — 4-level escalation + HITL + resume |
| agent_retry | Complete — 3-attempt inner retry |
| service_health | Complete — provider fallback + backoff |
| context_budget, cost_tracker, lb_monitor, schema_validator, tool_constraints | Stubs |

### Memory
| Module | Status |
|--------|--------|
| chroma_client | Complete — validated embeddings |
| memory_schema | Complete — fingerprints + patterns |
| redis_state | Complete — Redis → fakeredis → dict fallback |
| memory_quality, pinecone_memory, seed_memory | Stubs |

### Tools
| Tool | Status |
|------|--------|
| data_tools, e2b_sandbox, llm_client, mlflow_tracker, submit_tools, wilcoxon_gate | Complete |
| arxiv_tool, kaggle_scraper, leakage_detector, null_importance, report_tools, stability_validator, web_search | Stubs |

---

## Key Metrics

| Metric | Value |
|--------|-------|
| CV AUC (Spaceship Titanic) | 0.8798 (+/- 0.0055) |
| LB Score | 0.78419 |
| Contract tests | 18 (IMMUTABLE) |
| Quality tests (Days 8-15) | 300+ across 8 daily test files |
| Regression tests | Frozen at Phase 1 and Phase 2 gates |

---

## Non-Negotiable Rules (carried forward)

1. **Polars only** — no Pandas anywhere in the pipeline
2. **No DataFrames in state** — file path pointers only
3. **Contract tests are IMMUTABLE** after creation day
4. **Dependencies pinned** — no `pip upgrade` during 30-day build
5. **Regression tests frozen** at phase gates
6. **Fail loud, fail early, never fail silently**
7. **Every state mutation explicit** — `{**state, "key": new_value}` pattern
8. **venv Python 3.13.2** — system Python is 3.14, never use system Python

---

## Phase 2 Gate Status: PASSED

Gate commit: `57294eb`
All 3 conditions verified. Regression test frozen. Ready for Phase 3.
