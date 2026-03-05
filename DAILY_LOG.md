# Professor -- Daily Build Log

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
