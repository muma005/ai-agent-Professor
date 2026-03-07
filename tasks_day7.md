# Day 7 Tasks
You are implementing Day 7 of the Professor Kaggle agent build.
Read every word of this prompt before writing a single line of code.

## The Standard You Are Being Held To

You are a principal engineer. Not a junior developer generating boilerplate.
Every function you write must:
- Handle failure paths before the happy path
- Raise immediately with an actionable error message when something is wrong
- Never return None or empty values silently
- Include type annotations and a docstring stating what it reads, writes, and raises
- Be tested against the real Spaceship Titanic data, not just a 5-row fixture

Before writing any function ask: what are all the ways this can fail?
Write those cases first. Then fill in the success path.

---

## Build Order — Do Not Deviate

Task 1 → Task 2 → Task 3 → Task 4 → Task 5

Task 5 cannot be written until Task 4 passes the gate.
Task 4 cannot be run until Tasks 1 and 2 are done.
If Task 4 fails, debug it. Do not proceed to Task 5 until the gate passes.

---

## Task 1 — FIX: LangGraph State Merge Corrupts model_registry
File: core/state.py
Priority: Critical — Must be done FIRST before any other task

### The Problem
LangGraph does not simply replace state between nodes. It uses reducers.
A plain `list` field in TypedDict uses the DEFAULT reducer which APPENDS
on every node return instead of replacing. This means:
- Every time ml_optimizer returns model_registry = [new_entry], LangGraph
  APPENDS to the existing list instead of replacing it.
- On the first gate run with any retry, model_registry has duplicates.
- On competition run 3, model_registry has 3 copies of the same model.
- The Ensemble Architect tries to blend 20 identical models by Day 15.

This WILL corrupt the Day 7 gate run if not fixed first.

### The Fix
Open core/state.py. Add these imports at the top:
  from typing import Annotated
  import operator

Then for every field in ProfessorState, classify it as either:

ACCUMULATE (grows across runs — use Annotated[list, operator.add]):
  - model_registry: every competition adds entries, never replaced wholesale
  - errors: accumulates all errors seen across the session
  - lineage_log: append-only event log

REPLACE (reset each pipeline run — use plain list):
  - dag: router sets the full route, optimizer never appends to it
  - cv_scores: optimizer replaces with current run's fold scores
  - oof_predictions: replaced each training run

Apply the correct annotation to every list field.
Verify by checking: "if this node returns this field, should it ADD TO
or REPLACE the existing value?" — answer that for each field before annotating.

After the fix, write a quick verification:
  from core.state import ProfessorState
  from langgraph.graph import StateGraph
  # confirm graph compiles without errors with new annotations
  print("State annotations verified")

---

## Task 2 — FIX: Define Phase 1 Gate Thresholds Explicitly
File: tests/phase1_gate.py
Priority: Critical — Must be done before the gate run

### The Problem
Without explicit thresholds, "did the gate pass?" is ambiguous.
If Professor scores 0.7751 and Submission 0 scored 0.7754, have you failed?
That is a 0.0003 gap from random seed variance — not a broken pipeline.
You need defined, checkable pass/fail conditions written as assertions
before you run the gate, not after.

### What to Build
Create tests/phase1_gate.py — a standalone script (not pytest) that:

1. Reads SUBMISSION_0_CV from a constant you set manually right now.
   Open your Day 2 manual submission notebook, find the CV AUC you recorded,
   set it as: SUBMISSION_0_CV = X.XXXX
   If you didn't record it, set it to 0.775 as a conservative floor.

2. Defines these exact pass conditions as Python assertions:

   Pass condition 1: Professor CV >= SUBMISSION_0_CV - 0.005
     Rationale: 0.005 buffer accounts for random seed variance between runs.
     A gap larger than this means something is wrong, not random.

   Pass condition 2: submission.csv passes validate_existing_submission()
     with zero errors against sample_submission.csv.
     Use the function already built in tools/submit_tools.py.

   Pass condition 3: pytest tests/contracts/ exits with code 0.
     Run it as a subprocess and check returncode. If any contract test
     fails the gate fails — the pipeline cannot be trusted.

   Pass condition 4: Full pipeline wall clock < 30 minutes.
     time.time() before and after run_professor(). Fail if delta > 1800s.
     Rationale: Phase 3 Optuna adds 10-20x runtime. If Phase 1 already
     takes 4 hours, Phase 3 will never finish.

   Pass condition 5: CV > 0.70 absolute floor.
     This is independent of Submission 0. A CV of 0.65 means the pipeline
     is broken regardless of what Submission 0 scored.

   Hard fail conditions (raise immediately, do not continue):
   - Any Python exception during pipeline run
   - Any null values in submission.csv
   - submission.csv missing required columns
   - model_registry empty after run

3. Prints a clear PASS / FAIL report with all condition results.
   A passing gate prints:
     ✓ CV 0.8123 >= floor 0.7700 (Submission 0: 0.7750 - 0.005 = 0.7700)
     ✓ submission.csv valid: 4277 rows, correct columns, zero nulls
     ✓ All contract tests green (pytest exit code 0)
     ✓ Wall clock: 14m 32s < 30m limit
     ✓ CV 0.8123 > 0.70 absolute floor
     === PHASE 1 GATE: PASSED ===

---

## Task 3 — Set Up MLflow Experiment Tracking
File: tools/mlflow_tracker.py
Priority: Medium — Safe to Stub if time is short

### What to Build
A thin wrapper around MLflow that the ml_optimizer calls at the end
of every training run. This is for visibility, not functionality.
The pipeline must work identically whether MLflow is available or not.

Build it with a graceful fallback:
  try:
      import mlflow
      MLFLOW_AVAILABLE = True
  except ImportError:
      MLFLOW_AVAILABLE = False

Functions to build:
  log_run(session_id, competition, model_type, params, cv_mean, cv_std,
          n_features, data_hash) -> None
    - If MLFLOW_AVAILABLE: log to experiment named after competition
    - If not: print a one-line summary to stdout and return
    - Never raises — MLflow failure must never crash the pipeline

  log_submission(session_id, submission_path, cv_mean, lb_score=None) -> None
    - Same graceful fallback pattern

Setup instructions to include as a comment at the top of the file:
  # Setup: pip install mlflow
  # Start UI: mlflow ui --port 5000
  # View at: http://localhost:5000
  # Set MLFLOW_TRACKING_URI in .env to persist across sessions

Wire log_run() into agents/ml_optimizer.py at the end of the training loop,
after metrics.json is written.

If MLflow installation causes any dependency conflicts, stub the entire
file with the graceful fallback and move on. This is Safe to Stub.

---

## Task 4 — Full End-to-End Run: Spaceship Titanic → submission.csv
File: main.py
Priority: Critical — THIS IS THE PHASE 1 GATE

This is the most important task of the day. Everything built in Days 1-6
must work together as a single connected pipeline for the first time.

### Pre-Run Checklist (Do These Before Running)

1. Verify Task 1 is done: core/state.py has Annotated list fields
2. Verify Task 2 is done: tests/phase1_gate.py exists with constants set
3. Run pytest tests/contracts/ — ALL GREEN before proceeding.
   If any contract test fails, fix it now. Do not attempt the gate run
   with failing contract tests. A failing contract test means the
   component it tests is broken, and the pipeline will fail.
4. Verify the Spaceship Titanic data is in place:
   data/spaceship_titanic/train.csv
   data/spaceship_titanic/test.csv
   data/spaceship_titanic/sample_submission.csv
5. Verify .env has LANGCHAIN_TRACING_V2 set (true or false, must exist)

### The Gate Run Command
  python main.py run \
    --competition spaceship-titanic \
    --data ./data/spaceship_titanic/train.csv \
    --budget 2.0

### Expected Console Output (Every Line Matters)
  [Professor] Session:      spaceship_XXXXXXXX
  [Professor] Competition:  spaceship-titanic
  [Professor] Data:         ./data/spaceship_titanic/train.csv
  [Professor] Budget:       $2.00
  [SemanticRouter] Task type: tabular_classification
  [SemanticRouter] Route:   data_engineer → ml_optimizer → submit
  [DataEngineer] Loaded: 8693 rows, 14 columns
  [DataEngineer] Nulls before cleaning: XXX
  [DataEngineer] Nulls after cleaning: 0
  [DataEngineer] Complete. data_hash: XXXXXXXXXXXXXXXX
  [MLOptimizer] Target column: Transported
  [MLOptimizer] Features: XX columns
  [MLOptimizer] Fold 1/5: AUC = 0.XXXX
  [MLOptimizer] Fold 2/5: AUC = 0.XXXX
  [MLOptimizer] Fold 3/5: AUC = 0.XXXX
  [MLOptimizer] Fold 4/5: AUC = 0.XXXX
  [MLOptimizer] Fold 5/5: AUC = 0.XXXX
  [MLOptimizer] CV AUC: 0.XXXX (+/- 0.XXXX)
  [Submit] Generating submission — session: spaceship_XXXXXXXX
  [SubmitTools] ✓ submission.csv valid: outputs/.../submission.csv
  [SubmitTools] Rows: 4277 | Cols: ['PassengerId', 'Transported']
  [Submit] ✓ Done. Upload to Kaggle:
    kaggle competitions submit -c spaceship-titanic \
      -f outputs/.../submission.csv \
      -m 'Professor Phase 1 baseline'
  [Professor] ✓ Complete
  [Professor] CV score:   0.XXXX
  [Professor] Submission: outputs/.../submission.csv

If any line is missing or shows an error, stop and fix before continuing.

### Failure Modes and What They Mean

Pipeline crashes at DataEngineer:
  → Check raw_data_path is the exact path to train.csv
  → Check cleaned.parquet and schema.json are writing to outputs/{session_id}/
  → Run the Data Engineer contract test in isolation

Pipeline crashes at MLOptimizer:
  → Check clean_data_path in state points to outputs/{session_id}/cleaned.parquet
  → Check schema.json has required fields: columns, types, missing_rates
  → Verify the target column is being identified (should be 'Transported')
  → Check feature matrix is not all-zero

Pipeline crashes at Submit:
  → Check test.csv exists at data/spaceship_titanic/test.csv
  → Check sample_submission.csv exists at data/spaceship_titanic/sample_submission.csv
  → Check model was saved: outputs/{session_id}/best_model.pkl must exist

CV AUC < 0.70:
  → This is a broken pipeline, not bad luck
  → Check cleaned.parquet for null contamination: pl.read_parquet(path).null_count()
  → Check the target column is boolean, not string
  → Check feature encoding — categoricals must be integer codes, not strings

CV AUC between 0.70 and 0.75:
  → Acceptable for Phase 1 with default LightGBM, no feature engineering
  → Do not tune. Do not retry. Proceed to submission upload.

submission.csv row count is not 4277:
  → test.csv has 4277 rows in Spaceship Titanic
  → The submit node is loading the wrong test file
  → Print test_df.shape before generating predictions

### After the Gate Run Passes
Run tests/phase1_gate.py to get the formal PASS result:
  python tests/phase1_gate.py

Record in DAILY_LOG.md:
  Day 7 gate:
    Session ID:     spaceship_XXXXXXXX
    CV AUC:         0.XXXX
    Submission 0:   0.XXXX
    Wall clock:     Xm Xs
    Gate status:    PASSED
    Kaggle submit:  [pending / submitted / LB score: X.XXXX]

Upload the submission to Kaggle:
  kaggle competitions submit \
    -c spaceship-titanic \
    -f outputs/{session_id}/submission.csv \
    -m "Professor Phase 1 baseline — Day 7 gate"

Record the Kaggle LB score in DAILY_LOG.md when it comes back.

---

## Task 5 — FREEZE Phase 1 Regression Test
File: tests/regression/test_phase1_regression.py
Priority: Critical — Written ONLY after Task 4 gate passes. Not before.

### What to Build
This file is written ONCE and NEVER edited after today.
It is the permanent floor that protects everything built in Phase 1.

At the top of the file, in a comment, record:
  # Written: Day 7
  # Gate CV: X.XXXX (the exact CV from today's gate run)
  # Gate session: spaceship_XXXXXXXX
  # Commit hash: [run `git rev-parse HEAD` and paste here]
  # IMMUTABLE: never edit this file after Day 7

### What to Freeze

Freeze 1 — CV Floor
  The CV from today's gate minus 0.03.
  If today's gate CV was 0.812, the floor is 0.782.
  This gives a 0.03 buffer for normal model variance.
  Any future run below this floor = regression alert.

  def test_cv_floor():
      # Re-run the pipeline on Spaceship Titanic with a fixed random seed
      # CV must be >= CV_FLOOR = [today's gate CV - 0.03]

Freeze 2 — Submission Format
  submission.csv must always have exactly 2 columns.
  Column 1 must be 'PassengerId'.
  Column 2 must be 'Transported'.
  Row count must be exactly 4277.
  Zero nulls.

  def test_submission_format():
      # validate_existing_submission() must return {"valid": True, "errors": []}

Freeze 3 — State Pointer Contract
  No raw DataFrames in state. Only string file pointers.
  This is the most important architectural invariant in the project.

  def test_state_has_only_pointers():
      for key, value in result_state.items():
          assert not isinstance(value, pl.DataFrame), \
              f"Raw DataFrame found in state key '{key}' — must be a file pointer"
          assert not isinstance(value, pd.DataFrame), \
              f"Pandas DataFrame found in state key '{key}' — Pandas not allowed"

Freeze 4 — Cost Tracker Incremented
  The pipeline made LLM calls (or will in Phase 2).
  cost_tracker must exist and be a dict with llm_calls key.

  def test_cost_tracker_incremented():
      assert "cost_tracker" in result_state
      assert "llm_calls" in result_state["cost_tracker"]
      assert isinstance(result_state["cost_tracker"]["llm_calls"], int)

Freeze 5 — All Existing Contract Tests Still Pass
  This is a meta-test. It runs the full contracts suite and fails if
  any contract test that passed on Day 7 no longer passes.

  def test_all_contract_tests_pass():
      result = subprocess.run(
          ["pytest", "tests/contracts/", "-v", "--tb=short"],
          capture_output=True, text=True
      )
      assert result.returncode == 0, \
          f"Contract tests failed:\n{result.stdout}\n{result.stderr}"

---

## End of Day Checklist

Complete these in order. Do not skip any.

  [ ] Task 1 done: core/state.py has Annotated list fields, graph compiles
  [ ] Task 2 done: tests/phase1_gate.py exists, SUBMISSION_0_CV constant set
  [ ] Task 3 done (or stubbed): tools/mlflow_tracker.py exists, graceful fallback
  [ ] All contract tests green: pytest tests/contracts/ -v
  [ ] Task 4 done: pipeline ran end-to-end without crashing
  [ ] Gate passed: python tests/phase1_gate.py printed PASSED
  [ ] Kaggle submission uploaded and LB score noted
  [ ] Task 5 done: tests/regression/test_phase1_regression.py written with
      today's CV and commit hash in the header. NEVER edit this file again.
  [ ] Commit:
        git add .
        git commit -m "Day 7: Phase 1 gate passed — CV: X.XXXX, LB: X.XXXX, all tests green"
        git push origin phase-1
  [ ] DAILY_LOG.md updated with session ID, CV, LB score, wall clock time

---

## What Done Means Today

Done is not "the code was written."

Done is:
- pytest tests/contracts/ → all green
- python tests/phase1_gate.py → PASSED printed
- Kaggle submission uploaded → LB score recorded
- tests/regression/test_phase1_regression.py frozen with commit hash

If pytest tests/contracts/ is not all green, you are not done.
If the gate script did not print PASSED, you are not done.
If the regression test is not written and frozen, you are not done.

Phase 2 does not start until all five conditions above are true.