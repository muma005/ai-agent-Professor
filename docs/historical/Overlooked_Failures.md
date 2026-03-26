# Five Overlooked Project Killers
### The Failures That Don't Announce Themselves Until It's Too Late

---

> This document covers five failure modes that are not in the main architecture,
> not in the regression protocol, and not in the daily procedures.
> Each one is capable of ending the project on its own.
> Each one is fully preventable with less than an hour of work up front.
> Read this before Day 1. Do the fixes before writing a single line of code.

---

## Overview

```
Failure                           Probability   Impact     Prevention Time
─────────────────────────────────────────────────────────────────────────
1. Dependency Hell                High          Severe     5 minutes Day 1
2. Schedule Slippage No Cut Plan  High          Severe     30 minutes Day 1
3. Polars / Pandas Mismatch       Medium        Silent     20 minutes Day 1
4. Mid-Pipeline Checkpoint Gap    Medium        Moderate   2 hours Day 16
5. First Competition Panic        Medium        Severe     20 minutes Day 1
─────────────────────────────────────────────────────────────────────────
```

Failures 1 and 2 have the highest combined risk. Both are likely to occur.
Both are project-ending without mitigation. Both are fixed before the first
line of code is written.

---

## Failure 1 — Dependency Hell

### What It Is

The Professor stack has 20+ libraries. On any day during the build, any one
of them can release a breaking update. You do not need to do anything wrong
for this to happen. You just need to be building on Day 18 when a library
maintainer merges a breaking change.

### What It Looks Like

```
Day 18 — Deep in Feature Factory.
You create a fresh virtual environment or run pip install --upgrade.
LangGraph 0.3 releases with a breaking change to conditional edges.
Your entire core/professor.py graph stops working.
The error message references internal LangGraph APIs you never called.
You spend 6 hours debugging code that worked perfectly yesterday
before realising the library changed under you.
Day 19 is now a debugging day instead of a build day.
You are two days behind with no buffer left.

Other libraries with a history of breaking API changes:
  polars    → breaks between minor versions, documented known issue
  chromadb  → collection API changed between 0.4 and 0.5
  groq SDK  → response format changed in early 2025
  langfuse  → self-hosted version must match pip package exactly
  optuna    → storage backend API changed in v3
```

### The Fix — 5 Minutes on Day 1

After installing all dependencies and confirming the environment runs:

```bash
# Step 1: Freeze exact working versions
pip freeze > requirements.txt

# Step 2: Commit immediately
git add requirements.txt
git commit -m "Day 1: Pin exact dependency versions — environment confirmed working"
```

Then manually review requirements.txt and add comments to the critical pins:

```
# requirements.txt
# ─────────────────────────────────────────────────────────────────────
# PINNED on Day 1. All versions confirmed working together.
# DO NOT upgrade any library during the 30-day build unless a
# specific bug requires it. Library updates are a Month 2 activity.
# ─────────────────────────────────────────────────────────────────────

langgraph==0.2.x          # core graph engine — pin strictly
groq==0.9.x               # primary LLM client — pin strictly
chromadb==0.5.x           # vector memory — pin strictly
polars==1.x.x             # DataFrame library — breaks between minors
langfuse==2.x.x           # observability — must match self-hosted version
lightgbm==4.x.x           # ML model — pin strictly
optuna==3.x.x             # HPO — storage API changed in v3
langgraph-checkpoint-redis # pin to match langgraph version
fakeredis==2.x.x          # test Redis substitute
restrictedpython==7.x.x   # Phase 1-2 sandbox
```

### The Rule Going Forward

```
NEVER run during the 30-day build:
  pip upgrade
  pip install --upgrade [anything]
  pip install [new_library] (without pinning immediately after)

The only exception:
  A specific library version is confirmed to have a bug
  that is blocking the current build task.
  In this case: upgrade only that library, test, pin the new version,
  commit with message: "Day N: Upgrade [library] to fix [specific bug]"

Every new library added during the build:
  pip install [library]
  pip freeze | grep [library] >> requirements.txt
  git add requirements.txt
  git commit immediately
```

### If Dependency Hell Hits Anyway

```
Symptom: Code that worked yesterday throws unfamiliar errors today.
         Error messages reference library internals you never touched.

Step 1: Check if a library updated since last working commit.
  pip list --outdated
  Compare output against requirements.txt from last green commit.

Step 2: Downgrade the changed library.
  pip install [library]==[version_in_requirements.txt]

Step 3: Run tests. If green: add explicit pin for that library.
  Commit: "Day N: Pin [library]==[version] after breaking update"

Step 4: If requirements.txt was not committed with exact versions,
  you are debugging blind. This is why Step 1 above is non-negotiable.
```

---

## Failure 2 — Schedule Slippage With No Cut Decision Tree

### What It Is

The build plan is 40+ components in 30 days at 8 hours per day. It has no
slack. A single multi-day debugging session — which will happen — puts the
entire phase calendar behind. The danger is not the slippage itself. The
danger is making cut decisions under pressure at 11pm on Day 23 when you
are behind, exhausted, and not thinking clearly. Cuts made under those
conditions are always wrong.

### What It Looks Like

```
Day 12 — A difficult LangGraph wiring bug costs 3 days to resolve.
          Phase 1 gate passes Day 10 instead of Day 7.
          The calendar is 3 days behind before Phase 2 begins.

Day 21 — Phase 3 gate. The calendar says Day 21. Reality says Day 24.
          Three days of Phase 4 work no longer exist.

Day 28 — You are looking at 9 days of planned Phase 4 work
          that now has to fit in 6 days.
          In a panic, you cut the circuit breaker.
          You cut the daily regression run (to save time).
          You cut the session_id isolation.

Day 30 — The Day 30 gate: 3 concurrent competitions.
          No session isolation: the runs corrupt each other's state.
          No circuit breaker: a failure hangs indefinitely.
          You have also lost your regression safety net.
          Gate fails. Project ends without meeting its own criteria.
```

The cuts made under panic are precisely the non-negotiables. The cut
decision tree below is written now, while you are calm, so it is available
on Day 23 when you are not.

### The Cut Decision Tree

```
─────────────────────────────────────────────────────────────────────
IF 1 DAY BEHIND at any phase gate:
─────────────────────────────────────────────────────────────────────
Cut from the Safe Stub list only. One item maximum.
Never cut from the Never Cut list.

Stub candidates in order of lowest risk:
  1. LangFuse observability → keep JSONL logging, add LangFuse Month 2
  2. Optuna warm start (GAP 11) → cold start is slower, not broken
  3. External data scout (GAP 5) → stub returns empty manifest
  4. Pseudo-label agent (GAP 9) → skip if gates not met anyway

─────────────────────────────────────────────────────────────────────
IF 2 DAYS BEHIND at Phase 2 gate (Day 14):
─────────────────────────────────────────────────────────────────────
Cut the following — all are restorable in Month 2:
  → Parallel execution (GAP 6): run sequential, add LangGraph
    Send API in Month 2. Pipeline is slower but correct.
  → External Data Scout (GAP 5): stub with empty manifest.
    Feature Factory proceeds without external sources.
  → LangFuse: keep JSONL through all 30 days.
  → Data version hashing (GAP 13): add in Month 2.
    Document the risk in README.

Do NOT cut:
  Critic (any vector), Metric Contract, circuit breaker,
  cost tracker, session_id isolation, Validation Architect,
  daily regression run, HITL escalation.

─────────────────────────────────────────────────────────────────────
IF 3+ DAYS BEHIND at Phase 2 gate (Day 14):
─────────────────────────────────────────────────────────────────────
Everything in the 2-day cut list PLUS:
  → EDA Agent (GAP 1): stub with empty eda_report.json.
    Critic still runs its 4 vectors. Feature Factory
    proceeds without EDA fingerprint. Quality degrades slightly.
  → Pseudo-label agent (GAP 9): remove from Phase 4 entirely.
  → Seed memory (GAP 4): ChromaDB starts empty. Cold start
    acknowledged in README. Add seed script in Month 2.
  → Post-mortem agent (GAP 15): manual notes replace structured
    memory writes for Month 1.

Do NOT cut under any circumstances:
  Critic, Metric Contract, circuit breaker, cost tracker,
  session_id isolation, Validation Architect, HITL escalation,
  daily regression run, schema validator (GAP 3),
  service health (GAP 10), interaction budget cap (GAP 12).

─────────────────────────────────────────────────────────────────────
IF 3+ DAYS BEHIND at Phase 3 gate (Day 21):
─────────────────────────────────────────────────────────────────────
Everything in the 3-day Phase 2 cut list PLUS:
  → Scope the Day 30 gate down.

Original Day 30 gate:
  3 competitions simultaneously, all complete without human help,
  at least 1 scores top 30%.

Revised Day 30 gate:
  1 competition, runs without human help, scores top 40%.

This is not a failure. A single-competition Professor that scores
top 40% autonomously is still an exceptional result for a 30-day
solo build. The concurrent competition capability is Month 2.

─────────────────────────────────────────────────────────────────────
WHAT IS NEVER CUT REGARDLESS OF HOW FAR BEHIND:
─────────────────────────────────────────────────────────────────────
  ✗ Inner code retry loop in every agent
  ✗ Cost tracker with thresholds
  ✗ Metric Contract enforcement
  ✗ Validation Architect
  ✗ Red Team Critic (minimum Vectors 1 and 2)
  ✗ Circuit breaker and HITL escalation
  ✗ Session_id namespace isolation
  ✗ Schema validator between agents
  ✗ Service health with fallbacks
  ✗ Daily regression run
  ✗ Vertical slice — pipeline end-to-end every day
  ✗ Feature interaction budget cap
  ✗ Preprocessing leakage check (Critic Vector 1)
```

### The Daily Slip Tracking Rule

Add one line to the daily log every single day:

```
Day N: Schedule status: ON TRACK / 1 DAY BEHIND / 2 DAYS BEHIND
```

If the status reaches "2 DAYS BEHIND" — execute the 2-day cut list
that same evening. Do not wait to see if you catch up. You will not
catch up without explicitly freeing time. Cutting the right things
on Day 12 gives you breathing room. Cutting the wrong things on Day 27
ends the project.

---

## Failure 3 — Polars / Pandas API Mismatch in LLM-Generated Code

### What It Is

Professor uses Polars throughout the pipeline. The LLM generates Python code.
Every LLM's training data is overwhelmingly Pandas — it is the dominant
DataFrame library by orders of magnitude. When an agent asks the LLM to
generate data processing code, the LLM will reach for Pandas instinctively.
In most cases it will not announce this. The code will run.

### What It Looks Like

```
Data Engineer asks the LLM to generate preprocessing code.

LLM generates:
  df = pd.read_csv(path)
  df = df.fillna(df.mean())           ← Pandas API
  df.to_parquet("cleaned.parquet")    ← Pandas parquet writer

This code runs without error in the Docker sandbox.
Pandas is installed. The parquet file is produced.

Feature Factory reads the parquet file using Polars.
The file loads. The columns look correct.

But:
  Pandas and Polars have different default type inference.
  Pandas writes nullable integer columns as float64 when NaN present.
  Polars reads them back as Float64.
  The Feature Factory generates features on Float64 columns
  that should be Int64 — statistical operations behave differently.
  The null importance filter produces slightly different results.
  The Critic does not flag this.
  The CV score is 0.004 lower than it should be.
  You never know why.

This is the silent variant. The visible variant:
  Polars raises a type error when receiving a pandas DataFrame
  from a function that was supposed to return a polars DataFrame.
  Contract test catches it. Fixed in 10 minutes.

The silent variant is the dangerous one.
```

### The Fix — Two Parts

**Part 1: Sandbox preamble injected before every generated script**

```python
# In tools/e2b_sandbox.py

SANDBOX_PREAMBLE = '''\
# ── Professor Pipeline — Library Standard ─────────────────────────
# This pipeline uses POLARS for all DataFrame operations.
# pandas is available but is NOT the project standard.
#
# CORRECT:
#   import polars as pl
#   df = pl.read_csv(path)
#   df.write_parquet("output.parquet")
#
# WRONG (do not use):
#   import pandas as pd
#   df = pd.read_csv(path)
#   df.to_parquet("output.parquet")
#
# If pandas is genuinely required for a specific library call,
# convert output back to Polars before returning:
#   polars_df = pl.from_pandas(pandas_df)
# ──────────────────────────────────────────────────────────────────
import polars as pl
import polars.selectors as cs

'''

def execute_code(code: str, session_id: str) -> dict:
    full_code = SANDBOX_PREAMBLE + code
    # ... rest of execution logic
```

**Part 2: System prompt injection for every code-generating agent**

```python
# In tools/llm_client.py — added to system prompt for coding agents

POLARS_CONSTRAINT = """
LIBRARY REQUIREMENT — READ BEFORE GENERATING ANY CODE:
This pipeline uses Polars (not Pandas) for all DataFrame operations.

Required patterns:
  Read CSV:     pl.read_csv(path)
  Read Parquet: pl.read_parquet(path)
  Write:        df.write_parquet(path)
  Filter:       df.filter(pl.col("x") > 0)
  GroupBy:      df.group_by("col").agg(pl.col("val").mean())
  Fill null:    df.fill_null(strategy="mean")

Forbidden patterns (will cause type mismatches downstream):
  pd.read_csv()        ← use pl.read_csv()
  df.to_parquet()      ← use df.write_parquet()
  df.fillna()          ← use df.fill_null()
  df.groupby()         ← use df.group_by()
  df.apply()           ← use df.map_elements() or df.map_rows()

If you use Pandas for any reason, convert before returning:
  polars_df = pl.from_pandas(pandas_df)
"""
```

**Part 3: Contract test added to every agent that generates code**

```python
# In tests/contracts/test_data_engineer_contract.py
# (and any other agent contract that produces parquet files)

def test_output_is_polars_readable(self, fixture_csv, session_dir):
    """Output parquet must be readable as a native Polars DataFrame."""
    run_data_engineer(raw_data_path=fixture_csv)
    df = pl.read_parquet(session_dir / "cleaned.parquet")
    assert isinstance(df, pl.DataFrame), \
        "Output must be a Polars DataFrame, not Pandas"

def test_no_object_dtype_columns(self, fixture_csv, session_dir):
    """No columns should have object dtype — indicates Pandas contamination."""
    run_data_engineer(raw_data_path=fixture_csv)
    df = pl.read_parquet(session_dir / "cleaned.parquet")
    object_cols = [c for c in df.columns if df[c].dtype == pl.Object]
    assert len(object_cols) == 0, \
        f"Object dtype columns detected (Pandas contamination): {object_cols}"
```

---

## Failure 4 — Mid-Pipeline Checkpoint Gap

### What It Is

The service_health.py handles API failures with retry and fallback. What it
does not handle is a rate limit that hits midway through a long-running
pipeline stage after significant computation has already been completed.
The retry exhausts. The stage raises an exception. The pipeline restarts
from the beginning of that stage. Hours of CPU computation are lost.

### What It Looks Like

```
Day 17 — Feature Factory running Rounds 1 through 5.
Round 1 (basic transforms): complete. 23 features pass.
Round 2 (domain features): complete. 31 features pass.
Round 3 (aggregations): running. 2.5 hours elapsed.

At feature 8 of 47 planned aggregations:
  Groq rate limit reached for the day.
  service_health.py retries 3 times — all fail.
  Pipeline raises GroqRateLimitExhausted.
  Circuit breaker triggers HITL escalation.
  State is saved to Redis.

You restart with fresh API quota the next morning.
Feature Factory reads state from Redis.
Feature Factory starts from Round 1.
2.5 hours of aggregation computation is gone.
Round 3 regenerates 7 features that were already validated.

At scale with the null importance filter (50 shuffles per feature):
  50 LightGBM runs × 7 features = 350 training runs redone
  On a laptop CPU: approximately 2 hours of compute
  Total loss: 2 hours of compute + 30 minutes of LLM calls
```

### The Fix — Stage Checkpointing in Feature Factory

This is the only fix in this document that requires actual code changes
beyond Day 1. It is built on Day 16 when Feature Factory is built.

```python
# In agents/feature_factory.py

def run_feature_factory(state: ProfessorState) -> ProfessorState:

    # Load checkpoint if one exists from a previous interrupted run
    checkpoint = load_checkpoint(
        session_id=state["session_id"],
        agent="feature_factory"
    ) or {
        "completed_rounds": [],
        "approved_features": [],
        "rejected_features": [],
        "current_round": 1,
        "last_saved": None
    }

    for round_num in range(1, 6):

        # Skip rounds already completed before interruption
        if round_num in checkpoint["completed_rounds"]:
            log(f"Round {round_num}: skipping — already completed in previous run")
            continue

        log(f"Round {round_num}: starting")

        # Run the round — may raise exception if rate limited
        new_features = run_round(round_num, state, checkpoint["approved_features"])

        # Filter features through null importance
        approved, rejected = apply_null_importance_filter(new_features, state)

        # Update checkpoint
        checkpoint["approved_features"].extend(approved)
        checkpoint["rejected_features"].extend(rejected)
        checkpoint["completed_rounds"].append(round_num)
        checkpoint["last_saved"] = datetime.utcnow().isoformat()

        # Save to Redis after EVERY round — before moving to next
        save_checkpoint(
            session_id=state["session_id"],
            agent="feature_factory",
            data=checkpoint
        )
        log(f"Round {round_num}: complete. {len(approved)} features approved. Checkpoint saved.")

    # Clear checkpoint on successful completion
    clear_checkpoint(state["session_id"], "feature_factory")

    return {**state, "feature_manifest": checkpoint["approved_features"]}
```

```python
# In memory/redis_state.py — add these three functions

def save_checkpoint(session_id: str, agent: str, data: dict) -> None:
    key = f"{session_id}:checkpoint:{agent}"
    redis_client.set(key, json.dumps(data))

def load_checkpoint(session_id: str, agent: str) -> dict | None:
    key = f"{session_id}:checkpoint:{agent}"
    value = redis_client.get(key)
    return json.loads(value) if value else None

def clear_checkpoint(session_id: str, agent: str) -> None:
    key = f"{session_id}:checkpoint:{agent}"
    redis_client.delete(key)
```

**Apply the same pattern to ML Optimizer** (Optuna can also be interrupted
mid-trial). Optuna has native study persistence — use it:

```python
# In agents/ml_optimizer.py

def run_ml_optimizer(state: ProfessorState) -> ProfessorState:
    study_name = f"{state['session_id']}_optuna"

    # Optuna's built-in persistence — resumes from where it left off
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///outputs/{state['session_id']}/optuna.db",
        load_if_exists=True,   # ← resumes interrupted study
        direction="maximize"
    )

    # Calculate remaining trials (not total — remaining)
    completed_trials = len(study.trials)
    remaining_trials = max(0, TARGET_TRIALS - completed_trials)

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)
    else:
        log("Optuna study already complete — loading best params from storage")

    return {**state, "best_params": study.best_params}
```

---

## Failure 5 — First Live Competition Panic

### What It Is

Spaceship Titanic is a clean, well-structured dataset built for beginners.
It always runs. It never surprises. The first real Kaggle competition will
have at least one property that Spaceship Titanic never prepared Professor
for. When Professor underperforms — and it will on its first real competition
— the natural response is to conclude that the architecture is wrong. This
response is almost always incorrect. And acting on it is almost always
project-ending.

### What It Looks Like

```
Day 21 — Phase 3 gate on a real competition.
Professor runs. Submission produced. Leaderboard score: 0.71.
Expected for top 40%: 0.78.

Gut reaction: "The architecture is wrong. The Feature Factory
is not generating the right features. LangGraph was a mistake.
I should rebuild the pipeline with a simpler approach."

What you do next determines whether the project survives:

WRONG response:
  Open a new file. Start sketching a simplified architecture.
  Spend Day 21 in a spiral of second-guessing.
  Day 22 arrives. Nothing has been debugged. Nothing has been built.
  You are now behind on Phase 4 and have made the problem worse.

RIGHT response:
  Open the diagnostic checklist.
  Work through it systematically.
  Discover: the competition uses a grouped CV strategy
  (user_id groups) that the Validation Architect defaulted
  to standard KFold for. CV score was artificially inflated.
  Real CV was 0.71 all along. Architecture is correct.
  Add group detection to Validation Architect.
  Resubmit Day 22. Score: 0.79. Top 38%. Gate passes.
```

The architecture is not wrong. It has never run on this specific data before.
A specific implementation failing on specific data is expected and fixable.
The diagnosis checklist below converts panic into a systematic 45-minute
debugging session.

### The Diagnostic Checklist — Run Before Any Architectural Conclusion

```
WHEN PROFESSOR UNDERPERFORMS ON A REAL COMPETITION:
─────────────────────────────────────────────────────────────────
Paste this checklist into the daily log and work through it
in order. Do not skip steps. Do not make architectural conclusions
before completing all 6 steps.

STEP 1 — Check the Critic output.
  Open: outputs/{session_id}/critic_verdict.json
  Are there any CRITICAL or HIGH severity flags?
  If YES: those are the bugs. Fix those first.
          Do not proceed to Step 2 until Critic flags are resolved.
  If NO: proceed to Step 2.

STEP 2 — Verify the metric.
  Open: outputs/{session_id}/metric_contract.json
  Does metric_contract.scorer match the competition evaluation metric?
  Compare against the competition's Evaluation page on Kaggle.
  If MISMATCH: fix the Metric Contract. Rerun. Check score.
  If MATCH: proceed to Step 3.

STEP 3 — Diagnose the CV / LB gap.
  Compare: cv_mean in metrics.json vs public LB score.

  CV high, LB low (gap > 0.02):
    → Likely preprocessing leakage or train/test distribution shift
    → Check Critic Vector 1 preprocessing leakage output
    → Check eda_report.json for temporal drift between train and test

  Both CV and LB low (both below expectation):
    → Features are insufficient for this domain
    → Proceed to Step 4

  CV and LB close but both below expectation:
    → Model quality or feature quality issue
    → Proceed to Step 4

STEP 4 — Check the EDA report.
  Open: outputs/{session_id}/eda_report.json
  Did Professor act on everything flagged?

  target_distribution.recommended_transform not applied?
    → Apply log transform to target. Rerun Optimizer.

  temporal_profile.train_test_drift = true?
    → Validation Architect should have caught this.
    → Force GroupKFold or TimeSeriesSplit. Rerun.

  duplicate_analysis.id_conflicts > 0?
    → Duplicate IDs with different targets in training data.
    → Investigate before any feature engineering.

  leakage_fingerprint with high correlation features?
    → These features must be removed before modeling.

STEP 5 — Check the feature manifest.
  Open: outputs/{session_id}/feature_manifest.json
  Count features that passed null importance filter.

  Fewer than 10 features passed:
    → Feature Factory starved. Domain features missing.
    → Add domain-specific features from competition forum insights.
    → Rerun Feature Factory from Round 2.

  More than 150 features passed:
    → Null importance threshold may be too permissive.
    → Raise percentile threshold from 95th to 97th.
    → Rerun filter.

  No interaction features in manifest:
    → Feature Factory may not have reached Round 3.
    → Check Feature Factory logs for early termination.

STEP 6 — Check the validation strategy.
  Open: outputs/{session_id}/validation_strategy.json
  Is the CV strategy appropriate for this competition?

  Competition has user_id or group_id column and CV uses KFold:
    → Data leaks across folds (same user in train and validation).
    → Fix: GroupKFold on user_id column.

  Competition is time-series and CV uses KFold:
    → Future data leaks into past folds.
    → Fix: TimeSeriesSplit with appropriate gap.

  Competition is heavily imbalanced and CV uses standard KFold:
    → Fix: StratifiedKFold.

─────────────────────────────────────────────────────────────────
AFTER ALL 6 STEPS:

If root cause identified:
  Fix the specific component identified.
  Do not touch components that were not identified as the cause.
  Rerun. Resubmit.

If root cause not identified after all 6 steps:
  Paste to Claude: the full critic_verdict.json,
  metric_contract.json, validation_strategy.json,
  and the competition description.
  Fresh eyes will find it.

If 6 steps completed and score is simply below a top competitor's:
  This is not an architecture failure.
  This is a domain knowledge gap that features will close.
  Proceed with Phase 4 as planned.
─────────────────────────────────────────────────────────────────
```

### The Rule Against Architectural Rewrites Mid-Build

Write this into the README and read it on any day you consider a rewrite:

```
The architecture was designed over multiple sessions with full
context. It was pressure-tested across 15 gap analyses.
It was validated against the regression and execution drift
failure modes before a line was written.

A bad competition score on Day 21 does not invalidate any of this.
It means one specific component produced a suboptimal result
on one specific dataset.

The correct response to a bad score is always:
  Run the diagnostic checklist.
  Fix the specific component identified.
  Never rewrite a component that was not identified as the cause.

A rewrite that takes 2 days and produces a 0.003 improvement
is a project-ending trade. The diagnostic checklist that takes
45 minutes and produces the same improvement is the right path.
```

---

## The Unified Pre-Build Checklist

Everything in this document that must be done before Day 1:

```
□ pip freeze > requirements.txt after environment setup
□ Add version comments to critical dependencies in requirements.txt
□ Commit requirements.txt immediately
□ Write the schedule slip tracking field into daily log template
□ Add SANDBOX_PREAMBLE to tools/e2b_sandbox.py
□ Add POLARS_CONSTRAINT to tools/llm_client.py system prompt
□ Add polars contract test to test_data_engineer_contract.py
□ Write the diagnostic checklist into README under "Competition Failures"
□ Add schedule status line to daily log template:
    "Schedule status: ON TRACK / 1 DAY BEHIND / 2 DAYS BEHIND"

Things built on the day their agent is built:
□ Day 16: Stage checkpointing in feature_factory.py
□ Day 19: Optuna study persistence in ml_optimizer.py
```

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────┐
│  FIVE OVERLOOKED FAILURES — DAILY REMINDERS                   │
│                                                                │
│  1. DEPENDENCIES                                               │
│     Never run pip upgrade during the 30-day build.            │
│     requirements.txt is pinned. It stays pinned.              │
│                                                                │
│  2. SCHEDULE SLIP                                              │
│     Log schedule status every day: ON TRACK / BEHIND.         │
│     2 days behind → execute cut list immediately.             │
│     Cut the right things now. Not the wrong things later.     │
│                                                                │
│  3. POLARS PREAMBLE                                            │
│     Every LLM-generated script starts with SANDBOX_PREAMBLE.  │
│     Contract tests check for Polars output. Always.           │
│                                                                │
│  4. CHECKPOINTING                                              │
│     Feature Factory saves to Redis after every round.         │
│     Optuna uses SQLite storage. Interruption ≠ restart.       │
│                                                                │
│  5. COMPETITION FAILURE                                        │
│     Bad score → run diagnostic checklist.                     │
│     Fix what the checklist identifies.                        │
│     Never rewrite what the checklist doesn't identify.        │
│     Architecture is not wrong. One component is off.          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

*Document version: 1.0*
*Project: Professor AI Agent — 30-Day Build*
*Keep this file in the project root alongside REGRESSION_PROTECTION.md*
*Pre-build checklist must be completed before Day 1 code is written.*