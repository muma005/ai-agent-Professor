# Professor -- Daily Build Log

---

## Day 15 -- 2026-03-15 -- Phase 2 Finale: Infrastructure for Phase 3

**Schedule status:** ON TRACK

**Tests green before starting:** YES

**The ONE thing for today:**
Lock graph compilation, add Docker sandbox, LangFuse observability, and external data scout. Close Phase 2.

**Tasks completed:**
- [x] core/professor.py -- graph singleton with thread-safe double-checked locking (`get_graph()`, `get_graph_cache_clear()`)
- [x] tools/e2b_sandbox.py -- Docker container sandbox (`python:3.11-slim`, `--network none`, `--read-only`, memory/CPU limits), subprocess fallback when Docker unavailable
- [x] core/professor.py -- LangFuse observability integration (graceful degradation if keys absent, JSONL lineage coexists)
- [x] agents/competition_intel.py -- external data scout (`run_external_data_scout()`), gated by `state["external_data_allowed"]`
- [x] core/state.py -- new fields: `external_data_allowed`, `external_data_manifest`
- [x] conftest.py -- `reset_graph_singleton` autouse fixture to prevent cross-test pollution
- [x] Contract test: test_competition_intel_contract.py (IMMUTABLE)

**Bugs found and fixed:**
- Graph recompilation on every `run_professor()` call -- 2-4s overhead per invocation, compounds in retry loops
- `conftest.py` needed graph singleton reset to prevent test cross-contamination

**Known test issue (not yet fixed):**
- `test_day15_quality.py::TestExternalDataScout::test_data_engineer_logs_high_relevance_sources` -- FAILS with `AttributeError: <module 'agents.data_engineer'> does not have the attribute 'log_event'`. The test patches `agents.data_engineer.log_event` but `data_engineer.py` does not import `log_event` at module level. Mock target mismatch.

**Test results:** 45 passed, 1 failed, 1 skipped, 1 deselected (46 warnings -- google.generativeai deprecation)

**Final commit hash:** 688c6d8

---

## Day 14 -- 2026-03-15 -- Compounding Advantage + Phase 2 Gate

**Schedule status:** ON TRACK

**Tests green before starting:** YES

**The ONE thing for today:**
Add historical failure pattern vector to critic, run Phase 2 gate, freeze regression test.

**Tasks completed:**
- [x] agents/red_team_critic.py -- Vector 8 (`historical_failures`): queries `critic_failure_patterns` ChromaDB collection for structurally similar past competitions, flags features matching known failure modes with confidence-based severity
- [x] memory/memory_schema.py -- `query_critic_failure_patterns()` function for semantic retrieval from ChromaDB
- [x] tests/phase2_gate.py -- Phase 2 gate with 3 conditions (critic catches injected leakage, validation architect blocks wrong metric, CV beats Phase 1 baseline)
- [x] tests/regression/test_phase2_regression.py -- FROZEN, IMMUTABLE

**Bugs found and fixed:**
- Flaky ChromaDB round-trip test -- query was limited to top-20 results when collection had fewer entries, fixed by querying full collection

**Final commit hash:** 57294eb

---

## Day 13 -- 2026-03-15 -- Submission Integrity

**Schedule status:** ON TRACK

**Tests green before starting:** YES

**The ONE thing for today:**
Fix silent submission bugs: train/test column misalignment, stale data hash in ensemble, and add Wilcoxon significance gate.

**Tasks completed:**
- [x] agents/ml_optimizer.py -- save `feature_order` (exact column list) to `metrics.json` after training
- [x] tools/submit_tools.py -- enforce `feature_order` at prediction time, raise on mismatch (not silent wrong prediction)
- [x] agents/ensemble_architect.py -- `data_hash` validation before blend, filters out models trained on stale data versions
- [x] tools/wilcoxon_gate.py -- Wilcoxon signed-rank test for statistically rigorous model comparison (p < 0.05)
- [x] agents/ml_optimizer.py -- Wilcoxon gate plugged into model selection (fold scores stored per Optuna trial)
- [x] core/professor.py -- graph wiring stabilisation fix (routing map)
- [x] 55 quality tests -- all green

**Bugs found and fixed:**
- Train/test column misalignment: Polars reads CSVs with different internal ordering, LightGBM silently uses wrong features (shapes match but semantics wrong)
- Ensemble blending models trained on different data versions (Kaggle mid-competition data corrections)
- Graph wiring: routing map KeyError on conditional edges (stabilisation gate fix)

**Final commit hash:** a42a938

---

## Day 12 -- 2026-03-14 -- Podium-Level Hardening

**Schedule status:** ON TRACK

**Tests green before starting:** YES (252+ tests pass)

**The ONE thing for today:**
Complete HITL human layer, fix OOM in Optuna, and add cost control to LangSmith tracing.

**Tasks completed:**
- [x] guards/circuit_breaker.py -- HITL prompt generation with 5 error classes, 3 interventions per class, terminal banner
- [x] guards/circuit_breaker.py -- resume_from_checkpoint() with AUTO/MANUAL intervention application
- [x] agents/ml_optimizer.py -- per-fold memory check via psutil, TrialPruned on threshold, del models in finally, gc_after_trial, n_jobs=1
- [x] core/professor.py -- LangSmith tracing disabled during Optuna loop, sampling rate from env var (default 0.10)
- [x] Contract tests: test_hitl_prompt_contract.py, test_resume_checkpoint_contract.py (IMMUTABLE)

**Bugs found and fixed:**
- del models in try not finally -- never runs on exception
- try/except not try/finally for tracing restore
- env var restore setting "false" when key was absent
- No error message truncation (50KB JSON per HITL event)
- Per-fold memory check missing (OOM kills process at fold 5)
- n_jobs=-1 causing 8x memory multiplier

**Final commit hash:** b9e13c8

---

## Day 11 -- 2026-03-13 -- Learning Loop

**Schedule status:** ON TRACK

**Tests green before starting:** YES (186+ tests pass)

**The ONE thing for today:**
Add robustness vector, wire critic to supervisor auto-replan, build post-mortem agent.

**Tasks completed:**
- [x] agents/red_team_critic.py -- Vector 4 (robustness): noise injection, slice audit, OOF calibration (ECE + Brier)
- [x] core/professor.py -- supervisor_replan node: CRITICAL -> auto-replan (drop features, rerun affected nodes, increment dag_version); max 3 replans before HITL
- [x] agents/post_mortem_agent.py -- CV/LB gap root cause, feature retrospective, pattern extraction -> professor_patterns_v2 + critic_failure_patterns collections
- [x] 66 new tests -- all green

**Bugs found and fixed:**
- CRITICAL verdict going straight to HITL when 80% are mechanically fixable (auto-replan handles them)
- Critic self-improvement loop was missing (post-mortem now feeds back missed issues)

**Final commit hash:** 3c267c6

---

## Day 10 -- 2026-03-12 -- Quality Conscience

**Schedule status:** ON TRACK

**Tests green before starting:** YES (110+ tests pass)

**The ONE thing for today:**
Redesign ChromaDB memory for transferable patterns, build 6-vector Red Team Critic.

**Tasks completed:**
- [x] memory/memory_schema.py -- competition fingerprints + NL embeddings for semantic retrieval; patterns replace raw hyperparams
- [x] agents/red_team_critic.py -- 6 detection vectors: shuffled target, ID-only model, adversarial classifier, preprocessing audit, PR curve (imbalanced), temporal leakage
- [x] Severity routing: CRITICAL -> hitl_required + replan_requested; HIGH/MEDIUM -> log + continue
- [x] Contract test: test_critic_contract.py (IMMUTABLE)
- [x] 53 adversarial quality tests -- all green

**Bugs found and fixed:**
- Hyperparams don't transfer between competitions (patterns do)
- Preprocessing leakage regex false positives on sklearn Pipeline objects

**Final commit hash:** 610b7af

---

## Day 9 -- 2026-03-11 -- Resilience Layer

**Schedule status:** ON TRACK

**Tests green before starting:** YES (66 tests pass)

**The ONE thing for today:**
Build circuit breaker, subprocess sandbox, service health fallbacks, and inner retry loops.

**Tasks completed:**
- [x] guards/circuit_breaker.py -- 4-level escalation: MICRO -> MACRO -> HITL -> TRIAGE
- [x] tools/e2b_sandbox.py -- replaced RestrictedPython with subprocess sandbox (supports numpy, polars, LightGBM)
- [x] guards/service_health.py -- Groq->Gemini fallback, exponential backoff, ChromaDB/Redis graceful degradation
- [x] agents/agent_retry.py -- inner retry loop (3 attempts + error context fed back to LLM) for all 8 LLM-calling agents
- [x] core/professor.py -- parallel execution groups (intelligence fan-out, model trial fan-out, critic fan-out)
- [x] 54 adversarial resilience tests -- 52 pass, 2 skip (Docker Redis not running)

**Bugs found and fixed:**
- RestrictedPython blocks numpy/LightGBM C-extensions (switched to subprocess sandbox)
- Redis silent fallback to fakeredis losing state on restart

**Final commit hash:** 010371d

---

## Day 8 -- 2026-03-10 -- Phase 2 Kickoff

**Schedule status:** ON TRACK

**Tests green before starting:** YES (58/58 contract tests pass)

**The ONE thing for today:**
Build intelligence and quality agents: Validation Architect, EDA Agent, Competition Intel.

**Tasks completed:**
- [x] core/state.py -- Phase 2 fields: task_type, data_hash (SHA-256[:16]), competition_context
- [x] agents/validation_architect.py -- deterministic CV strategy (StratifiedKFold / GroupKFold / TimeSeriesSplit / KFold) + CV/LB mismatch detection
- [x] agents/eda_agent.py -- outlier profiling, leakage fingerprinting, duplicate/ID-conflict detection
- [x] agents/competition_intel.py -- GM-CAP 1 forum scraper upgrade
- [x] Fixed ChromaDB silent fallback to random embeddings (validated bge-small-en-v1.5, 384-dim)
- [x] 57 new tests -- all green (66 total after merge)

**Bugs found and fixed:**
- ChromaDB silently using random embeddings when model load fails -- invisible corruption of all memory queries

**Final commit hash:** c95ecc5

---

## Day 7 -- 2026-03-07 -- PHASE 1 GATE

**Schedule status:** ON TRACK

**Tests green before starting:** YES (58/58 contract tests pass)

### === PHASE 1 GATE: PASSED ===

**Gate session:** spaceshi_694d438e
**Gate CV AUC:** 0.8798 (+/- 0.0055)
**Submission 0 CV:** 0.775
**Wall clock:** 20 seconds
**Kaggle submit:** 0.78419 LB score (fixed bool format)

**Gate conditions (all 5 PASS):**
- PASS  CV 0.8798 >= floor 0.7700 (Sub0: 0.775 - 0.005)
- PASS  submission.csv valid: 4277 rows, correct columns, zero nulls
- PASS  All contract tests green (58/58, pytest exit code 0)
- PASS  Wall clock: 0m 20s < 30m limit
- PASS  CV 0.8798 > 0.70 absolute floor

**Tasks completed:**
- [x] Task 1: core/state.py -- Annotated list fields (REPLACE vs ACCUMULATE reducers)
- [x] Task 2: tests/phase1_gate.py -- standalone gate script with 5 pass conditions
- [x] Task 3: tools/mlflow_tracker.py -- graceful fallback, wired into ml_optimizer
- [x] Task 4: Full gate run -- PASSED
- [x] Task 5: tests/regression/test_phase1_regression.py -- FROZEN, IMMUTABLE

**Bugs found and fixed:**
- NoneType + NoneType crash: ACCUMULATE fields (model_registry, submission_log) must init as [] not None
- model_registry duplication: ml_optimizer must return only new entry (not full list copy) with operator.add reducer

**Regression test frozen with:**
- Gate CV: 0.8798
- CV floor: 0.8498 (gate - 0.03)
- Commit hash: b60b6150276f84d8fded513cdae17793c1fed431

---

## Day 6 -- 2026-03-05

**Schedule status:** ON TRACK

**Tests green before starting:** YES (58/58 contract tests pass)

**The ONE thing for today:**
Replace submit stub with validated submission generator, add lineage logging.

**Tasks completed:**
- [x] tools/submit_tools.py -- generate_submission(), validate_existing_submission(), save_submission_log()
- [x] core/lineage.py -- append-only JSONL logger, wired into all 3 agents
- [x] core/professor.py -- replaced submit stub with real validated submission node
- [x] LangSmith tracing placeholder in .env (user fills in API key)
- [x] Full pipeline end-to-end with validated submission

**CV score today:** 0.8798 (+/- 0.0055) -- 5-fold AUC, default LightGBM
**Submission validation:** PASS -- 4277 rows, [PassengerId, Transported], dtype: bool, 0 nulls, IDs match

**Lineage trace (3 events):**
- data_engineer: cleaned_and_profiled -> clean_data_path, schema_path, data_hash
- ml_optimizer: trained_and_scored -> model_registry, cv_mean, oof_predictions_path
- submit: generated_submission -> submission_path

**What broke:** Nothing! Clean day.

**Tomorrow's ONE thing:**
Day 7 Phase 1 Gate -- upload submission.csv to Kaggle and get a real LB score.

**Final commit hash:** c694e49

---

## Day 5 -- 2026-03-05

**Schedule status:** ON TRACK

**Tests green before starting:** YES (45/45 contract tests pass)

**The ONE thing for today:**
First time `python main.py run` runs the full LangGraph graph end to end.

**Tasks completed:**
- [x] agents/semantic_router.py -- v0 linear routing, rule-based task type detection
- [x] core/professor.py -- LangGraph StateGraph with conditional edges + submit stub
- [x] Contract test: test_semantic_router_contract.py (13 tests, IMMUTABLE)
- [x] main.py _run() wired to run_professor() -- first full pipeline run

**CV score today:** 0.8798 (+/- 0.0055) -- 5-fold AUC, default LightGBM
**Submission:** 4277 rows generated via submit stub

**What broke:**
- Windows cp1252 encoding crash on Unicode arrows (->)  in print statements

**How it was fixed:**
- Replaced Unicode arrows with ASCII `->` in semantic_router.py, professor.py, main.py

**Final commit hash:** 9540a3a

---

## Day 4 -- 2026-03-04

**Schedule status:** ON TRACK

**Tests green before starting:** YES (27/27 contract tests pass)

**The ONE thing for today:**
Feed cleaned.parquet into ml_optimizer.py and get a real CV score.

**Tasks completed:**
- [x] core/metric_contract.py -- scorer registry, MetricContract dataclass
- [x] agents/ml_optimizer.py -- v0 LightGBM, StratifiedKFold(5), AUC scoring
- [x] Contract test: test_ml_optimizer_contract.py (18 tests, IMMUTABLE)
- [x] End-to-end smoke test: Data Engineer -> ML Optimizer on Spaceship Titanic

**CV score today:** 0.8798 (+/- 0.0055) -- 5-fold AUC, default LightGBM
**Public LB score:** 0.79424 (Submission 0, Day 2 -- accuracy metric)
**Fold scores:** [0.8859, 0.8764, 0.8824, 0.8836, 0.8709]

**What broke:**
- Tiny fixture (5 rows) too small for StratifiedKFold(5) -- expanded to 50 rows with learnable signal
- Sandbox can't import project modules -- moved profiling outside sandbox

**How it was fixed:**
- Generated 50-row fixture with structured signal (high spenders + young -> transported)
- Sandbox does only Polars cleaning; run_data_engineer calls profile_data outside sandbox

**Tomorrow's ONE thing:**
Build agents/semantic_router.py v0 + core/professor.py -- first time main.py run does something real.

**Final commit hash:** 706d67b

---

## Day 3 -- 2026-03-04

**Schedule status:** ON TRACK

**Tests green before starting:** YES (12/12 contract tests pass)

**The ONE thing for today:**
Feed train.csv into data_engineer.py and get cleaned.parquet + schema.json.

**Tasks completed:**
- [x] tools/data_tools.py -- Polars I/O layer (read/write/profile/hash/validate)
- [x] agents/data_engineer.py -- LangGraph node with sandbox + retry loop
- [x] Contract test: test_data_engineer_contract.py (15 tests, IMMUTABLE)
- [x] Verified on real Spaceship Titanic data (8693 rows, 0 nulls after cleaning)

**CV score today:** N/A (optimizer not built yet)

**What broke:**
- Sandbox _safe_import blocks project module imports inside sandbox code

**How it was fixed:**
- Sandbox uses only inline Polars calls; profile_data + write_json called outside sandbox

**Final commit hash:** 73fa97d

---

## Day 2 -- 2026-03-03

**Schedule status:** ON TRACK

**Tests green before starting:** YES (11/11 contract tests pass)

**The ONE thing for today:**
Build the sandbox with retries AND get a real Kaggle score on Spaceship Titanic.

**Tasks completed:**
- [x] tools/e2b_sandbox.py -- full RestrictedPython sandbox + 3-attempt retry loop
- [x] Contract test: tests/contracts/test_e2b_sandbox_contract.py (11 tests, IMMUTABLE)
- [x] Download Spaceship Titanic data
- [x] Manual Submission 0 -- default LightGBM baseline

**CV score today:** 0.7904 (+/- 0.0090) -- 5-fold accuracy, default LightGBM
**Public LB score:** 0.79424 -- Submission 0 (manual baseline)
**CV/LB gap:** 0.0038 -- healthy, no leakage detected

**What broke:**
- RestrictedPython safe_builtins missing __import__ -- added controlled _safe_import
- RestrictedPython transforms print() to _print_() -- added PrintCollector guard
- Windows cp1252 encoding issue with Polars output -- added UTF-8 stdout

**How it was fixed:**
- Added _safe_import that whitelists only ALLOWED_MODULES
- Added RestrictedPython guard functions (_print_, _getattr_, _getitem_, _getiter_, _write_)
- sys.stdout.reconfigure(encoding='utf-8') in baseline script

**Tomorrow's ONE thing:**
Build agents/data_engineer.py -- first real agent that exercises the state schema.

**Final commit hash:** (to be filled after commit)

---

## Day 1 -- 2026-03-02

**Schedule status:** ON TRACK

**Tests green before starting:** N/A (first day)

**The ONE thing for today:**
Set up the complete environment and confirm all services run.

**Tasks completed:**
- [x] Virtual environment (Python 3.13) + dependencies pinned
- [x] Fireworks AI DeepSeek-v3p2 + GLM-5 verified
- [x] Google Gemini Flash verified (with fallback)
- [x] Folder structure
- [x] tools/llm_client.py
- [x] core/state.py
- [x] main.py
- [x] RestrictedPython verified
- [x] ChromaDB verified
- [x] fakeredis verified
- [x] Git branching + pre-commit hook

**CV score today:** N/A (pipeline not wired yet)

**What broke:**
- ChromaDB pydantic v1 incompatible with Python 3.14 -- recreated venv with Python 3.13
- Git push 403 -- re-authenticated with GitHub token

**How it was fixed:**
- py -3.13 -m venv venv
- git remote set-url origin with token

**Final commit hash:** 4b4960a

---
