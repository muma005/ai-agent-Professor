You are performing a complete forensic audit of the Professor Agent pipeline.

Your job is to prove, with evidence from the actual code, whether each component works or does not work.
You are not allowed to assume anything works. You are not allowed to assume anything is broken.
Every verdict must be backed by a direct quote from the source file and an explanation of why it passes or fails.

Do not fix anything. Do not refactor anything. Do not suggest improvements.
Your only output is the audit report.

---

## PHASE 1 — READ THE FULL CODEBASE

Before writing a single verdict, read every file listed below in full.
Use the Read tool on each one. Do not skip any file.

Core:
  core/professor.py
  core/state.py

Agents (read every one):
  agents/data_engineer.py
  agents/eda_agent.py
  agents/validation_architect.py
  agents/feature_factory.py
  agents/ml_optimizer.py
  agents/red_team_critic.py
  agents/ensemble_architect.py
  agents/pseudo_label_agent.py
  agents/competition_intel.py
  agents/semantic_router.py
  agents/submission_strategist.py

Tools:
  tools/submit_tools.py
  tools/wilcoxon_gate.py
  tools/null_importance.py
  tools/stability_validator.py
  tools/e2b_sandbox.py

Memory:
  memory/memory_schema.py
  memory/redis_state.py

Guards:
  guards/circuit_breaker.py
  guards/service_health.py

After reading every file, proceed to Phase 2.

---

## PHASE 2 — AUDIT EACH COMPONENT

For every component below, you must answer all three questions:

  Q1. Does the code in this file actually do what its name implies?
  Q2. Does it connect correctly to the components that feed into it and the components it feeds into?
  Q3. Is there any code path that would cause it to silently produce wrong output — no crash, just wrong results?

A silent wrong output (Q3) is the most dangerous class of bug. A crash is easy to find. Silent wrong output can run for weeks undetected.

Work through every component in this exact order.

---

### COMPONENT 1: data_engineer.py

Read the file. Answer:

1. What does it write to disk? What are the exact file paths and formats?
2. What does it put into state? List every key it sets.
3. Does it correctly handle boolean columns (e.g. CryoSleep, VIP in Spaceship Titanic)?
   Quote the exact code that handles boolean encoding. If none exists, flag it.
4. Does it handle missing values? Quote the exact code.
5. What does X_train look like when it leaves this agent — Polars DataFrame, numpy array, or path string?
   Quote the exact line where it sets the output.

VERDICT: WORKING / BROKEN / SILENT-WRONG-OUTPUT
EVIDENCE: [quote the specific line(s) that prove the verdict]

---

### COMPONENT 2: feature_factory.py

Read the file. Answer:

1. What is the complete list of features it generates for a binary classification dataset?
2. Does it ever include the target column in the generated feature set?
   Search for every place the target column name is referenced. Quote each one.
3. Does it read raw CSV data directly, or only schema.json and competition_brief.json?
   Quote the exact file-read calls.
4. What does it write to state["feature_order"]? Quote the exact line.
5. Does feature_order include the target column? Prove it with a code trace.

VERDICT: WORKING / BROKEN / SILENT-WRONG-OUTPUT
EVIDENCE: [quote the specific line(s)]

---

### COMPONENT 3: ml_optimizer.py

Read the file. Answer:

1. What is the Optuna study direction for metric="accuracy"? Quote the exact line.
2. What is the Optuna study direction for metric="log_loss"? Quote the exact line.
   If the direction is hardcoded to "maximize" regardless of metric, that is a fatal bug.
3. Does HPO_OVERRIDES (n_estimators=150) apply to the final model training, or only inside _objective()?
   Trace the code path from _objective() to final model training. Quote every relevant line.
4. Does it store fold_scores in trial.user_attrs? Quote the exact line.
5. Does it store oof_predictions in the model_registry entry? Quote the exact line.
6. What does it pass to state at the end? Does it pass X_train as a DataFrame?
   Quote every state key it sets that contains data (not paths).

VERDICT: WORKING / BROKEN / SILENT-WRONG-OUTPUT
EVIDENCE: [quote the specific line(s)]

---

### COMPONENT 4: submit_tools.py

Read the file. Answer:

1. What is the exact code that converts model probability outputs into the submission column?
   Quote every line involved in the conversion.
2. For metric="accuracy" on Spaceship Titanic (target values are True/False strings):
   Does the code produce True/False strings, or 1/0 integers, or 0.87/0.23 floats?
   Trace the exact code path. Quote every transformation step.
3. Does the code read sample_submission.csv to determine the output format?
   Quote the line where it reads that file, or state "does not read sample_submission.csv".
4. For metric="logloss", does the code submit raw probabilities or hard class labels?
   Quote the exact code path.
5. Is there any int() or round() call applied to predictions before they go into the submission DataFrame?
   Search the entire file for int( and round(. Quote every instance.

VERDICT: WORKING / BROKEN / SILENT-WRONG-OUTPUT
EVIDENCE: [quote the specific line(s)]

---

### COMPONENT 5: pseudo_label_agent.py

Read the file. Answer:

1. Where does it get X_train from? Quote the exact line that reads training data.
2. Does state["X_train"] actually exist when this agent runs?
   Check ml_optimizer.py — does it ever set state["X_train"] to a DataFrame?
   Quote the ml_optimizer line that sets X_train in state, or state "ml_optimizer does not set state['X_train']".
3. If state["X_train"] does not exist, what happens when pseudo_label_agent runs?
   Quote the exact line that would crash or produce wrong output.
4. Does it respect the validation fold integrity rule (pseudo-labels added to training folds only, never validation)?
   Quote the exact code that enforces this.
5. What is the feature set it uses for pseudo-label training? Does it match the feature set used by ml_optimizer?

VERDICT: WORKING / BROKEN / SILENT-WRONG-OUTPUT
EVIDENCE: [quote the specific line(s)]

---

### COMPONENT 6: ensemble_architect.py

Read the file. Answer:

1. Does _validate_oof_present() exist and get called before blending? Quote the call site.
2. Does _validate_data_hash_consistency() get called before blending? Quote the call site.
3. Does it correctly compute OOF-weighted ensemble predictions?
   Quote the exact blending computation lines.
4. After blending, what does it put into state? List every key it sets.
5. Does it produce a submission-ready output (correctly formatted predictions for the test set)?
   Trace how test set predictions are generated from the ensemble. Quote every relevant line.

VERDICT: WORKING / BROKEN / SILENT-WRONG-OUTPUT
EVIDENCE: [quote the specific line(s)]

---

### COMPONENT 7: null_importance.py

Read the file. Answer:

1. Does Stage 1 (_run_stage1_permutation_filter) actually drop features or does it always return all features?
   Quote the exact lines that implement the drop decision.
2. Does Stage 2 run inside a single execute_code() call (persistent sandbox)?
   Quote the exact line where execute_code() is called. Is it inside a loop or called once?
3. What happens when PROFESSOR_FAST_MODE=1? Quote the exact code path.
4. Does the function correctly return a NullImportanceResult with all required fields in fast mode?
   Quote the return statement for fast mode.

VERDICT: WORKING / BROKEN / SILENT-WRONG-OUTPUT
EVIDENCE: [quote the specific line(s)]

---

### COMPONENT 8: competition_intel.py

Read the file. Answer:

1. Does run_external_data_scout() actually call an LLM or does it return an empty manifest unconditionally?
   Quote the exact lines.
2. Does competition_brief.json get written to disk? Quote the write call.
3. What does it put into state? List every key it sets.

VERDICT: WORKING / BROKEN / SILENT-WRONG-OUTPUT
EVIDENCE: [quote the specific line(s)]

---

### COMPONENT 9: red_team_critic.py

Read the file. Answer:

1. How many vectors are in VECTOR_FUNCTIONS? List their names.
2. Does vectors_checked in the verdict get built from VECTOR_FUNCTIONS.keys() dynamically,
   or is it a hardcoded list? Quote the exact line.
3. Does _check_historical_failures() exist and return OK gracefully when the ChromaDB collection is empty?
   Quote the exact code path for empty collection.
4. Do all sub-checks inside _check_robustness() run even if one sub-check throws an exception?
   Quote the try/except structure.

VERDICT: WORKING / BROKEN / SILENT-WRONG-OUTPUT
EVIDENCE: [quote the specific line(s)]

---

### COMPONENT 10: circuit_breaker.py

Read the file. Answer:

1. Does handle_escalation() call generate_hitl_prompt() at the HITL level?
   Quote the exact call site.
2. Does generate_hitl_prompt() exist in the file? If not, state "function does not exist".
3. Does resume_from_checkpoint() exist in the file? If not, state "function does not exist".
4. At failure_count=3, does the code set hitl_required=True? Quote the exact line.
5. At failure_count=1, does the code set hitl_required=False? Quote the exact line.

VERDICT: WORKING / BROKEN / SILENT-WRONG-OUTPUT
EVIDENCE: [quote the specific line(s)]

---

### COMPONENT 11: STATE CONNECTIVITY AUDIT

This is the most important part of the audit.

For each data handoff below, trace the exact code path:
  - What does the sender set in state?
  - What key name does it use?
  - What does the receiver read from state?
  - What key name does it expect?
  - Do the key names match exactly?

Handoff 1: data_engineer → feature_factory
  data_engineer sets: [list every key]
  feature_factory reads: [list every key it reads from state]
  MATCH: YES / NO / PARTIAL
  If NO or PARTIAL: quote the mismatched key names

Handoff 2: feature_factory → ml_optimizer
  feature_factory sets: [list every key]
  ml_optimizer reads: [list every key it reads from state]
  MATCH: YES / NO / PARTIAL

Handoff 3: ml_optimizer → ensemble_architect
  ml_optimizer sets: [list every key]
  ensemble_architect reads: [list every key it reads from state]
  MATCH: YES / NO / PARTIAL

Handoff 4: ml_optimizer → pseudo_label_agent
  ml_optimizer sets: [list every key]
  pseudo_label_agent reads: [list every key it reads from state]
  MATCH: YES / NO / PARTIAL

Handoff 5: ensemble_architect → submit_tools
  ensemble_architect sets: [list every key]
  submit_tools reads: [list every key it reads from state]
  MATCH: YES / NO / PARTIAL

---

## PHASE 3 — PRODUCE THE AUDIT REPORT

After completing all component audits, produce a single structured report.

Format it exactly like this:

=======================================================
PROFESSOR AGENT — PIPELINE AUDIT REPORT
=======================================================

COMPONENT VERDICTS
------------------
[component name] | WORKING / BROKEN / SILENT-WRONG-OUTPUT
  Proof: [one sentence quoting the specific evidence]

STATE CONNECTIVITY
------------------
[handoff name] | CONNECTED / DISCONNECTED / PARTIALLY CONNECTED
  Proof: [quote the mismatched key names if disconnected]

CRITICAL BUGS (must fix before any submission)
----------------------------------------------
[Bug 1 title]
  File: [exact file path]
  Line: [line number or function name]
  What it does wrong: [one precise sentence]
  Proof: [direct quote from the code]

[Bug 2 title]
  ...

NON-CRITICAL BUGS (fix after critical bugs resolved)
----------------------------------------------------
[list in same format]

CONFIRMED WORKING
-----------------
[list every component and handoff confirmed working with brief evidence]

UNKNOWN / COULD NOT VERIFY
---------------------------
[list anything you could not verify because the file was missing, empty, or unreadable]

=======================================================
END OF AUDIT
=======================================================

---

## RULES

1. Every verdict must be backed by a direct quote from the source file.
   No verdicts based on what the code is supposed to do. Only what it actually does.

2. If a function is referenced but does not exist in the file, that is a BROKEN verdict.
   Do not assume the function exists elsewhere.

3. If a state key is read with state["key"] and that key is never set by any upstream agent,
   that is a BROKEN verdict. The pipeline will crash at runtime.

4. If a state key is read with state.get("key") and the key is never set upstream,
   that is a SILENT-WRONG-OUTPUT verdict. The pipeline will use the default value silently.

5. Silent wrong output is worse than a crash. Flag every instance.

6. Do not fix anything. Do not suggest improvements. Audit only.

7. When you are done, print the total count:
   CRITICAL BUGS: N
   NON-CRITICAL BUGS: N
   WORKING COMPONENTS: N
   DISCONNECTED HANDOFFS: N