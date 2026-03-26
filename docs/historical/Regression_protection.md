# Regression Protection Protocol
### How Phase 2 Kills Phase 1 — And How We Engineer Around It

---

> Keep this file open every single day.
> The procedures in this document are not suggestions.
> They are the difference between a project that finishes and one that dies in Phase 3.

---

## What Regression Actually Is

Regression is not a dramatic failure. It does not announce itself. It is silent, cumulative, and by the time you notice it, the damage is already compounded across weeks of work.

Here is what it looks like in practice:

```
Day 7:   Phase 1 passes gate. Pipeline runs. Spaceship Titanic
         CV score: 0.798. You merge to main. You feel good.

Day 10:  You are deep in Phase 2. You refactor data_tools.py
         to support the Validation Architect. The change seems
         small. You move on.

Day 14:  Phase 2 gate. Full pipeline run. Something is wrong.
         The Data Engineer is producing malformed schema.json.
         The Optimizer is failing silently. CV score: 0.61.
         You have no idea what happened.

Day 15:  You spend the day debugging. You find the data_tools.py
         change from Day 10. One line. It broke the schema output
         format. Every downstream agent that reads schema.json
         has been receiving malformed data for 5 days.

Day 16:  You fix it. But now you discover the Critic's feature
         correlation logic was also affected. That needs fixing too.
         And the fix introduces a new issue in the Feature Factory.

Day 18:  You are debugging three phases simultaneously.
         The code is unfamiliar — it was written two weeks ago.
         You don't remember why half of it is structured the way it is.
         The project is now ten days behind with no clear path forward.
```

This is not a story about a bad developer. This is the default outcome of multi-phase development without regression protection. It happens to everyone. The only question is whether you have the systems to catch the Day 10 change in 4 minutes instead of discovering it on Day 14.

---

## The Core Principle

```
The rule that prevents 90% of regressions:

You never find out something broke when you run the full system.
You find out the moment the specific line of code that broke it
was written.

─────────────────────────────────────────────────────────────────
Without protection:
  You write a change. You move on.
  Two phases later, you run the full system. It fails.
  You spend 3 days figuring out what Phase 2 change
  broke which Phase 1 component.
  The original code is unfamiliar.
  The fix introduces new regressions.

With protection:
  You write a change. You save the file.
  Tests run automatically. 30 seconds later:
  "FAIL: test_data_engineer_contract — schema.json
   missing 'missing_rates' field. Changed in data_tools.py line 47."
  You fix one line. You move on.
  Total time lost: 4 minutes.
─────────────────────────────────────────────────────────────────
```

The cost of a regression is determined entirely by when it is discovered:

```
Discovered at the moment of introduction:    4 minutes
Discovered at end of day during integration: 30 minutes
Discovered at phase gate:                    4 hours
Discovered two phases later:                 4 days
Discovered at Day 30 final run:              project over
```

Every system in this document exists to push discovery as close to the moment of introduction as possible.

---

## The Three-Layer Protection System

### Layer 1 — Contract Tests (The Foundation)

Every agent has a public contract. The contract defines exactly three things:

1. What inputs the agent accepts
2. What outputs the agent must produce
3. What state mutations the agent must make

The contract test is written on the day the agent is built. It is never edited again for the duration of the project. The agent's internal logic can be completely rewritten in Phase 3. The contract never changes because the contract is the agent's interface to the rest of the system — and interfaces do not change.

```python
# tests/contracts/test_data_engineer_contract.py
# ─────────────────────────────────────────────
# Written:  Day 3
# Author:   [you]
# Status:   IMMUTABLE — never edit this file after Day 3
# ─────────────────────────────────────────────
#
# CONTRACT: Data Engineer
#
#   INPUT:   raw_data_path (str) — must point to existing file
#   OUTPUT:  outputs/{session_id}/cleaned.parquet — must exist
#            outputs/{session_id}/schema.json — must have:
#              columns (list), types (dict), missing_rates (dict)
#   STATE:   clean_data_path set to str pointer (not DataFrame)
#            data_hash set to SHA-256 of source file
#            cost_tracker.total_usd incremented

class TestDataEngineerContract:

    def test_accepts_valid_csv_path(self, fixture_csv):
        result = run_data_engineer(raw_data_path=fixture_csv)
        assert result["status"] == "success"

    def test_rejects_missing_path(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            run_data_engineer(raw_data_path="/nonexistent/path.csv")

    def test_produces_parquet_file(self, fixture_csv, session_dir):
        run_data_engineer(raw_data_path=fixture_csv)
        assert (session_dir / "cleaned.parquet").exists()

    def test_produces_schema_json(self, fixture_csv, session_dir):
        run_data_engineer(raw_data_path=fixture_csv)
        assert (session_dir / "schema.json").exists()

    def test_schema_has_required_fields(self, fixture_csv, session_dir):
        run_data_engineer(raw_data_path=fixture_csv)
        schema = json.loads((session_dir / "schema.json").read_text())
        assert "columns" in schema,       "schema.json missing 'columns'"
        assert "types" in schema,         "schema.json missing 'types'"
        assert "missing_rates" in schema, "schema.json missing 'missing_rates'"

    def test_state_contains_pointer_not_payload(self, fixture_csv):
        state = run_data_engineer(raw_data_path=fixture_csv)
        assert isinstance(state["clean_data_path"], str), \
            "clean_data_path must be a string path, not a DataFrame"
        assert "raw_data" not in state, \
            "raw DataFrame must never be stored in state"

    def test_data_hash_set_in_state(self, fixture_csv):
        state = run_data_engineer(raw_data_path=fixture_csv)
        assert "data_hash" in state
        assert len(state["data_hash"]) == 16  # first 16 chars of SHA-256

    def test_cost_tracker_incremented(self, fixture_csv):
        before = get_cost_tracker_total()
        run_data_engineer(raw_data_path=fixture_csv)
        after = get_cost_tracker_total()
        assert after > before, "cost_tracker must be incremented on every run"
```

**The rule:** One contract test file per agent. Written the day the agent is built. Never edited. If the contract needs to change, the agent's interface has changed — which is a major architectural decision requiring explicit acknowledgment, not a side effect of refactoring.

---

### Layer 2 — The Regression Suite (Frozen at Gate)

Contract tests verify individual agents. The regression suite verifies the system.

At each phase gate, a regression test file is written and immediately frozen. It captures the exact state of "working" at that moment: CV score floors, submission format validation, state schema integrity, all contract tests passing, and any other observable property that defines correctness for that phase.

```
tests/
├── contracts/              ← written day agent is built, never edited
│   ├── test_data_engineer_contract.py         (Day 3)
│   ├── test_ml_optimizer_contract.py          (Day 4)
│   ├── test_semantic_router_contract.py       (Day 5)
│   ├── test_e2b_sandbox_contract.py           (Day 2)
│   ├── test_validation_architect_contract.py  (Day 8)
│   ├── test_critic_contract.py               (Day 10)
│   ├── test_circuit_breaker_contract.py       (Day 9)
│   ├── test_feature_factory_contract.py       (Day 16)
│   ├── test_competition_intel_contract.py     (Day 15)
│   ├── test_ml_optimizer_optuna_contract.py   (Day 19)
│   ├── test_ensemble_architect_contract.py    (Day 22)
│   └── test_submission_strategist_contract.py (Day 23)
│
├── regression/             ← written at gate, frozen immediately, never edited
│   ├── test_phase1_regression.py    ← frozen Day 7
│   ├── test_phase2_regression.py    ← frozen Day 14
│   ├── test_phase3_regression.py    ← frozen Day 21
│   └── test_phase4_regression.py    ← frozen Day 28
│
└── integration/
    └── test_end_to_end.py  ← runs every day, must always pass
```

**What a frozen regression file actually contains:**

```python
# tests/regression/test_phase1_regression.py
# ────────────────────────────────────────────────────────────────
# FROZEN: Day 7
# Gate passed at commit: a3f9c21
# CV Score at gate: 0.798 on Spaceship Titanic
# Kaggle LB at gate: 0.794
# CV Floor: 0.760  (gate minus 5% margin — catches serious regressions)
#
# THIS FILE IS NEVER EDITED.
# If a test in this file fails, something in the system
# regressed from the Phase 1 baseline.
# Fix the regression — do not change this file.
# ────────────────────────────────────────────────────────────────

class TestPhase1Regression:

    def test_cv_score_above_phase1_floor(self):
        """CV must never drop below Phase 1 gate floor."""
        score = run_pipeline_cv_score(SPACESHIP_TITANIC_FIXTURE)
        assert score >= 0.760, \
            f"REGRESSION: CV {score:.4f} dropped below Phase 1 floor 0.760"

    def test_submission_csv_format_valid(self):
        """submission.csv must always match Spaceship Titanic format."""
        submission = run_pipeline_submission(SPACESHIP_TITANIC_FIXTURE)
        assert set(submission.columns) == {"PassengerId", "Transported"}, \
            f"REGRESSION: submission columns changed: {submission.columns.tolist()}"
        assert submission["Transported"].dtype == bool, \
            "REGRESSION: Transported column must be boolean"
        assert len(submission) == EXPECTED_TEST_ROW_COUNT

    def test_state_schema_integrity(self):
        """ProfessorState must always contain all Phase 1 required fields."""
        state = run_pipeline_get_state(SPACESHIP_TITANIC_FIXTURE)
        required_fields = [
            "session_id", "clean_data_path", "schema_path",
            "data_hash", "cost_tracker", "task_type"
        ]
        for field in required_fields:
            assert field in state, \
                f"REGRESSION: required state field '{field}' missing"

    def test_state_pointers_not_payloads(self):
        """State must never contain raw data — only string pointers."""
        state = run_pipeline_get_state(SPACESHIP_TITANIC_FIXTURE)
        assert isinstance(state["clean_data_path"], str)
        assert "raw_data" not in state
        assert "train_df" not in state

    def test_pipeline_completes_without_crash(self):
        """Full pipeline must run from CSV to submission.csv without exception."""
        result = run_pipeline_full(SPACESHIP_TITANIC_FIXTURE)
        assert result["status"] == "success", \
            f"REGRESSION: pipeline crashed with: {result.get('error')}"

    def test_all_phase1_contracts_pass(self):
        """All Phase 1 agent contracts must still be honoured."""
        results = run_contract_suite(agents=[
            "e2b_sandbox", "data_engineer",
            "ml_optimizer", "semantic_router"
        ])
        failed = [r for r in results if not r["passed"]]
        assert len(failed) == 0, \
            f"REGRESSION: contract failures: {[r['agent'] for r in failed]}"
```

**The CV floor is critical.** The test does not check that the score is exactly 0.798 — that would be too brittle and fail on legitimate improvements to randomness handling. It checks that the score never drops below 0.760 — a meaningful floor that catches something seriously broken while allowing normal variation.

---

### Layer 3 — Git Branching Strategy

Layers 1 and 2 catch regressions at the moment they are introduced. Layer 3 ensures you always have a known-working version of the system to return to, regardless of what happens on the development branch.

```
main
│
│  ← NEVER written to directly
│  ← Only receives merges after gate passes + all tests green
│  ← Every commit on main is tested, gated, working
│  ← main is always green — this is the invariant that makes
│    everything else possible
│
├── phase-1
│   Development happens here.
│   Gate passes + all tests green?
│   → Freeze regression test for Phase 1
│   → Merge phase-1 to main
│   → Delete phase-1 branch
│   → Create phase-2 from main
│
├── phase-2  (created from main after phase-1 merged)
│   Development happens here.
│   If phase-2 introduces a regression:
│   → git diff main shows EXACTLY what changed since merge
│   → git checkout main gives working system instantly
│   → You never lose the Phase 1 baseline
│
├── phase-3  (created from main after phase-2 merged)
│
└── hotfix
    If a merged phase introduces a bug discovered in production:
    → Create hotfix from main
    → Fix the specific issue only
    → Merge hotfix to main
    → Cherry-pick fix into active phase branch
    → Never debug on main directly
```

**The invariant that makes this work:** `main` is always green. The end-to-end test on `main` always produces a valid submission. If it does not, everything stops until it does. No deferrals. No "I'll fix it tomorrow." Main being broken means the project's only known-good baseline is broken. This is a stop-the-line event.

---

## The Pre-Commit Hook (Automation)

The three layers above are useless if you have to remember to run them manually. The pre-commit hook converts regression detection from a deliberate act into an automatic one. Every time you commit, the tests run. A commit that breaks any contract or regression test is rejected before it lands.

```yaml
# .pre-commit-config.yaml
# Place in project root. Run once: pre-commit install

repos:
  - repo: local
    hooks:
      - id: contract-tests
        name: Contract tests
        entry: pytest tests/contracts/ -x -q --tb=short
        language: system
        pass_filenames: false
        stages: [commit]

      - id: regression-tests
        name: Regression suite
        entry: pytest tests/regression/ -x -q --tb=short
        language: system
        pass_filenames: false
        stages: [commit]

      - id: end-to-end-test
        name: End-to-end pipeline
        entry: pytest tests/integration/test_end_to_end.py -x -q
        language: system
        pass_filenames: false
        stages: [commit]
```

**Install:**
```bash
pip install pre-commit
pre-commit install
```

From this point forward: every `git commit` runs all three test suites. A passing commit produces output like:

```
Contract tests...................................................Passed
Regression suite.................................................Passed
End-to-end pipeline..............................................Passed
[phase-2 a3f9c21] Day 10: Add Validation Architect — all tests pass
```

A failing commit produces:

```
Contract tests...................................................Failed
- hook id: contract-tests
- exit code: 1

FAILED tests/contracts/test_data_engineer_contract.py::test_schema_has_required_fields
AssertionError: schema.json missing 'missing_rates'
```

The commit is rejected. You fix the one line. You recommit. Total time: 4 minutes.

---

## The Daily Procedure

This procedure is followed every single day without exception. It takes 5 minutes at the start of the day and 2 minutes at the end.

```
────────────────────────────────────────────────────────────
START OF EVERY DAY
────────────────────────────────────────────────────────────

1. git checkout phase-N
2. git pull origin phase-N
3. Run: pytest tests/contracts/ tests/regression/ -q
4. Record in daily log: "Tests green before starting: YES / NO"

If NO — fix before writing any new code.
A day that starts on broken tests will end on broken tests.
This is not negotiable.

────────────────────────────────────────────────────────────
DURING BUILD — commit discipline
────────────────────────────────────────────────────────────

Write one logical unit of code.
Run the tests (pre-commit hook does this automatically).
Tests pass → commit with a specific message:
  "Day N: Add execute_code() retry loop to e2b_sandbox"
  "Day N: Wire Data Engineer into LangGraph state"
  "Day N: Fix schema.json missing_rates field"

Never commit 400 lines across 12 files in one go.
If something breaks later and your last commit is
"Day 10: Built Phase 2" — you have no useful information.
If your last commit is "Day 10: Add fold-aware scaler
to Data Engineer preprocessing" — you have the answer.

────────────────────────────────────────────────────────────
END OF EVERY DAY
────────────────────────────────────────────────────────────

1. All tests green?
2. Commit: "Day N: [what was built] — all tests pass"
3. Push to phase branch
4. Record in daily log: "Final commit: [hash] — tests: GREEN"

────────────────────────────────────────────────────────────
PHASE GATE DAY
────────────────────────────────────────────────────────────

1. Run full regression suite — ALL phases must be green
2. Run end-to-end test — must produce valid Kaggle submission
3. Gate score achieved?

   YES:
   → Write frozen regression test for this phase
   → Record: gate score, CV floor, dataset, commit hash
   → Merge phase-N to main
   → Create phase-N+1 from main
   → Delete phase-N branch

   NO:
   → Do NOT merge to main
   → Stay on phase branch
   → Debug until gate passes
   → Phase is not complete until the submission scores
```

---

## The Emergency Protocol

Despite the pre-commit hook, despite the daily regression runs, something will still break badly at some point. The emergency protocol converts a potential 3-day debugging spiral into a 2-hour surgical fix.

```
────────────────────────────────────────────────────────────
STEP 1 — Stop. Write no more code.
────────────────────────────────────────────────────────────
The moment you realise something is badly broken:
Stop. Close the file. Do not write another line trying to fix it.
More code on top of broken code deepens the hole.
Every line you add without understanding the root cause
is a line you will have to unwind later.

────────────────────────────────────────────────────────────
STEP 2 — Find the last green commit.
────────────────────────────────────────────────────────────
git log --oneline

Read the commit messages. Find the last one that says
"all tests pass" — or the last one you remember working.
You have this because you committed after every working state.
This is why small commits matter.

────────────────────────────────────────────────────────────
STEP 3 — Confirm the green state.
────────────────────────────────────────────────────────────
git checkout [last_green_commit] -b diagnosis

Run: pytest tests/contracts/ tests/regression/ -q
Tests pass? Good. This is your reference point.
The bug exists somewhere between this commit and HEAD.

────────────────────────────────────────────────────────────
STEP 4 — Find the exact commit that introduced it.
────────────────────────────────────────────────────────────
git diff [last_green_commit] phase-N

Read this diff carefully.
The bug is in here. Nowhere else.
There is no other possible location.
Read it like you are looking for one specific line —
because you are.

For complex diffs, use git bisect:
  git bisect start
  git bisect bad HEAD
  git bisect good [last_green_commit]
  git bisect run pytest tests/contracts/ -q
git bisect identifies the exact commit automatically.
On a project with 20 commits between good and bad: 90 seconds.

────────────────────────────────────────────────────────────
STEP 5 — Fix surgically.
────────────────────────────────────────────────────────────
Fix the specific line that introduced the regression.
Do not refactor adjacent code.
Do not improve related functions.
Do not address technical debt you notice along the way.
One commit. One change. Specifically: the regression.

Run all tests. If green: commit the fix, continue building.

────────────────────────────────────────────────────────────
STEP 6 — If not found within 2 hours: escalate.
────────────────────────────────────────────────────────────
Two hours is the ceiling for solo debugging on a single issue.
After two hours without a root cause, the cost of continuing
alone exceeds the cost of asking for help.

Paste to Claude:
  1. The exact failing test output (copy the full error)
  2. The git diff from Step 4
  3. What you were building when it broke
  4. What you have already tried

Do not summarise. Do not paraphrase. Paste the exact output.
The root cause is usually visible immediately with fresh eyes
and the complete information.
```

---

## Commit Message Convention

Commit messages are the project's searchable history. A good commit message makes `git log` useful. A bad one makes it useless.

```
FORMAT:
  Day N: [what was built or fixed] — [test status]

GOOD EXAMPLES:
  Day 3: Add execute_code() to e2b_sandbox — all tests pass
  Day 3: Add 3-attempt retry loop to execute_code — all tests pass
  Day 3: Wire e2b_sandbox into Data Engineer — all tests pass
  Day 7: GATE Phase 1 — CV 0.798, LB 0.794 — all tests pass
  Day 10: Fix schema.json missing_rates after data_tools refactor — all tests pass
  Day 14: GATE Phase 2 — CV 0.821 — all tests pass

BAD EXAMPLES:
  wip
  fix
  Day 10: Built Phase 2
  updates
  final version
  debugging stuff

The test of a good commit message:
  Can you read the git log 3 weeks from now and know
  exactly what changed in each commit and whether it worked?
  If yes: good message.
  If no: write it again.
```

---

## Quick Reference Card

Paste this into your daily log template and read it every morning.

```
┌─────────────────────────────────────────────────────────────────┐
│  REGRESSION PROTECTION — DAILY RULES                           │
│                                                                 │
│  1. Run tests before writing any code.                         │
│     pytest tests/contracts/ tests/regression/ -q               │
│     If not green: fix first. Always.                           │
│                                                                 │
│  2. Commit after every logical unit. Never 400 lines at once.  │
│     Pre-commit hook runs tests automatically.                   │
│     Red hook = fix immediately. Never bypass it.               │
│                                                                 │
│  3. main is always green.                                       │
│     Never write to main directly.                               │
│     Never merge a phase branch with failing tests.             │
│                                                                 │
│  4. Frozen files are never edited.                              │
│     tests/contracts/ — immutable after day written.            │
│     tests/regression/ — immutable after gate passed.           │
│                                                                 │
│  5. Stuck for more than 2 hours? Escalate.                     │
│     Paste: failing test + git diff + what you were building.   │
│     Do not spend a third hour alone.                           │
│                                                                 │
│  Cost of regression found TODAY:    4 minutes                  │
│  Cost of regression found PHASE 3:  4 days                     │
│  The entire system exists to keep the cost at 4 minutes.       │
└─────────────────────────────────────────────────────────────────┘
```

---

## What This Looks Like Over 30 Days

```
Days 1–7   (Phase 1)
  Contract tests written for: e2b_sandbox, data_engineer,
  ml_optimizer, semantic_router.
  Pre-commit hook catches 4 small regressions during the week.
  Total time lost to regressions: 18 minutes.
  Day 7: Phase 1 gate passes. Regression file frozen. Merged to main.

Days 8–14  (Phase 2)
  Contract tests written for: validation_architect, critic,
  circuit_breaker.
  Regression suite from Phase 1 runs on every commit.
  One significant regression on Day 11: data_tools change breaks
  schema format. Caught by pre-commit hook 30 seconds after commit.
  Fixed in 12 minutes.
  Day 14: Phase 2 gate passes. Regression file frozen. Merged to main.
  Both Phase 1 and Phase 2 regression suites now run on every commit.

Days 15–21 (Phase 3)
  Three regression tests catch issues before they compound:
  Day 16: Feature Factory output schema change breaks Data Engineer
  contract test. Fixed in 8 minutes.
  Day 18: Optuna warm start changes metric_contract format.
  Phase 2 regression test catches it. Fixed in 15 minutes.
  Day 21: Gate passes. All three regression files now run permanently.

Days 22–30 (Phase 4)
  Full regression suite (all four phases) runs on every commit.
  Zero undetected regressions.
  Day 30: Three concurrent competitions. All run without regression.
  Gate passes.

Total time lost to regressions across 30 days: ~3 hours
Total time that would have been lost without protection: ~15 days
```

---

*Document version: 1.0*
*Project: Professor AI Agent — 30-Day Build*
*Keep this file in the project root. Read it daily.*