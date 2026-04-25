# BUILD PROMPT — Layer 4: Advanced + Output (Days 12-17)
# Feed to Gemini CLI with: @PROFESSOR.md @STATE.md @SANDBOX.md @HITL.md @POLARS.md @CONTRACTS.md @PROMPTS.md @PROVIDERS.md

---

## CONTEXT

Layers 0-3 are complete and passing all contract tests:
- Layer 0: ProfessorState, Self-Debugging Engine, HITL, Cost Governor, Metric Gate, Data Integrity
- Layer 1: Pre-Flight + Depth Router, Deep EDA + Artifact Export, Domain Research, Shift Detector
- Layer 2: Leakage Precheck, Data Usage Checker, Retry Guidance, SOTA Search, Problem Reframer, Feature Factory (Adaptive Gates), Creative Hypothesis
- Layer 3: Gate Config, ML Optimizer, Red Team Critic v2 (9 vectors + confirmation), Self-Reflection + Dynamic Rules, Ensemble Upgrade (4 techniques), Post-Processing Optimizer

Now you're building the final layer: advanced capabilities (pseudo-labeling, freeform sandbox) and the output pipeline (solution provenance, submission safety, publisher). After this layer, Professor v2 is feature-complete.

---

## COMMIT PLAN (8 commits)

```
Commit 1:  agents/pseudo_label.py + tests/contracts/test_pseudo_label_contract.py
Commit 2:  tools/freeform_sandbox.py + HITL /freeform command integration + tests
Commit 3:  tools/code_ledger.py (upgrade) + tools/solution_assembler.py + tests
Commit 4:  shields/submission_safety.py + tests
Commit 5:  shields/memory_hygiene.py + tests
Commit 6:  agents/publisher.py + tests
Commit 7:  agents/supervisor.py (graph wiring + replan logic) + tests
Commit 8:  Full pipeline integration test — end-to-end on mock competition
```

Every commit passes `pytest tests/contracts/ -q` including ALL Layer 0-3 tests.

---

## COMMIT 1: Pseudo-Label Architect (agents/pseudo_label.py)

### What this does

Uses the test set's distributional signal to improve training. On competitions with large test sets (10x+ training), this signal is significant. Marginal improvement (+0.1-0.5%) but consistent — and on the medal boundary, 0.2% is 50-200 rank positions.

### CONDITIONAL ACTIVATION — two gates must pass

The Pseudo-Label Architect does NOT run unconditionally. It activates only when:
1. `len(test) >= len(train)` — test set must be at least as large as train (otherwise not enough signal)
2. `critic_verdict["severity"] != "CONFIRMED_CRITICAL"` — models must be trustworthy enough to generate pseudo-labels

If either gate fails, the agent returns immediately with no state changes.

Also skipped in SPRINT mode (`pipeline_depth == "sprint"`).

### The LangGraph node function

```python
def pseudo_label_architect(state: ProfessorState) -> dict:
    """
    Semi-supervised learning with safety-gated pseudo-labeling.
    
    Reads: features_train_path, features_test_path, target_column,
           validation_strategy, gate_config, metric_name, metric_config,
           best_model_params, best_model_type, canonical_train_rows,
           canonical_test_rows, critic_verdict, pipeline_depth,
           oof_predictions_path, test_predictions_path
    Writes: pseudo_label_activated, pseudo_label_fraction,
            pseudo_label_cv_delta
    Emits: STATUS (activation check), STATUS (round result)
    """
```

### Activation check

```python
def pseudo_label_architect(state: ProfessorState) -> dict:
    # Gate 1: Skip in SPRINT mode
    if state.pipeline_depth == "sprint" or "pseudo_label" in (state.agents_skipped or []):
        emit_to_operator("⏭️ Pseudo-Labels skipped (SPRINT mode)", level="STATUS")
        return {}
    
    # Gate 2: Test set must be >= train set
    if state.canonical_test_rows < state.canonical_train_rows:
        emit_to_operator(
            f"⏭️ Pseudo-Labels skipped: test ({state.canonical_test_rows}) < train ({state.canonical_train_rows})",
            level="STATUS"
        )
        return state.validated_update("pseudo_label", {"pseudo_label_activated": False})
    
    # Gate 3: Critic must not have found CRITICAL issues
    critic_severity = state.critic_verdict.get("severity", "CLEAR")
    if critic_severity in ("CRITICAL", "CONFIRMED_CRITICAL"):
        emit_to_operator(
            f"⏭️ Pseudo-Labels skipped: Critic severity={critic_severity} — models not trustworthy",
            level="STATUS"
        )
        return state.validated_update("pseudo_label", {"pseudo_label_activated": False})
    
    emit_to_operator("🧪 Pseudo-Label Architect activated", level="STATUS")
    # ... proceed with pseudo-labeling
```

### Round 1 — Generate and filter pseudo-labels

Generate code via `run_in_sandbox()`:

```python
pseudo_label_code = f"""
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import {cv_class}

# Load data
train = pl.read_parquet("{state.features_train_path}")
test = pl.read_parquet("{state.features_test_path}")

# Load test predictions from the best model
test_preds = pl.read_parquet("{state.test_predictions_path}")

# Compute confidence per test row
# Classification: confidence = max predicted probability
# Regression: confidence = 1 - normalized_residual (use prediction certainty heuristic)
if "{state.competition_type}" in ("binary", "multiclass"):
    confidence = test_preds["prediction"].to_numpy()
    confidence = np.maximum(confidence, 1 - confidence)  # max(p, 1-p) for binary
else:
    # Regression: use prediction std across ensemble members if available
    # Fallback: all confidence = 1.0 (accept all within K%)
    confidence = np.ones(len(test_preds))

# Filter: keep top K% most confident
K = min(30, int(100 * len(train) / len(test)))  # Never exceed 30% of train size
n_pseudo = int(len(test) * K / 100)
top_indices = np.argsort(confidence)[-n_pseudo:]  # Highest confidence

# Create pseudo-labeled rows
pseudo_test = test[top_indices.tolist()]
pseudo_labels = test_preds["prediction"][top_indices.tolist()]

# For classification: round to class labels
if "{state.competition_type}" in ("binary", "multiclass"):
    pseudo_labels = pseudo_labels.round().cast(pl.Int64)

# Tag pseudo-labeled rows
pseudo_test = pseudo_test.with_columns([
    pseudo_labels.alias("{state.target_column}"),
    pl.lit(True).alias("_is_pseudo_label"),
])

# Augment training data
train_augmented = pl.concat([
    train.with_columns(pl.lit(False).alias("_is_pseudo_label")),
    pseudo_test,
])

# Train on augmented data with SAME CV strategy
# CRITICAL: pseudo-labeled rows go into TRAINING folds only, never validation
# ... cross-validation code ...

# Compare augmented vs original
print(f"PSEUDO_RESULT:{{")
print(f"  \\"original_cv\\": {{original_cv_mean}},")
print(f"  \\"augmented_cv\\": {{augmented_cv_mean}},")
print(f"  \\"n_pseudo\\": {{n_pseudo}},")
print(f"  \\"K_pct\\": {{K}},")
print(f"  \\"pseudo_fraction\\": {{n_pseudo / len(train)}}")
print(f"}}")
"""
```

### Wilcoxon gate on pseudo-label acceptance

After sandbox execution, parse the results:

```python
# Compare per-fold scores: augmented vs original
from scipy.stats import wilcoxon

wilcoxon_p = state.gate_config.get("wilcoxon_p", 0.05)
stat, p_value = wilcoxon(augmented_fold_scores, original_fold_scores, alternative="greater")

if p_value < wilcoxon_p and augmented_cv_mean > original_cv_mean:
    # ACCEPT Round 1
    round_1_accepted = True
    emit_to_operator(
        f"🧪 Pseudo-Labels Round 1 ACCEPTED: +{augmented_cv_mean - original_cv_mean:.4f} "
        f"({n_pseudo} pseudo-labels, {pseudo_fraction:.1%} of train)",
        level="STATUS"
    )
else:
    # REJECT Round 1 — revert to original data
    round_1_accepted = False
    emit_to_operator(
        f"🧪 Pseudo-Labels Round 1 REJECTED: p={p_value:.4f} > {wilcoxon_p}. "
        f"Reverting to original data.",
        level="STATUS"
    )
```

### Round 2 (only if Round 1 accepted)

Same process but:
- Uses Round 1's retrained models for predictions (better pseudo-labels)
- K% halved (stricter confidence filter)
- If Round 2 degrades: revert to Round 1 result (NOT to original)

```python
if round_1_accepted:
    K_round2 = K // 2  # Halved
    # ... same process with stricter filtering ...
    
    if round_2_cv > round_1_cv:
        final_result = "round_2"
    else:
        final_result = "round_1"  # Revert to Round 1, not original
else:
    final_result = "original"
```

### Safety caps — NON-NEGOTIABLE

These are HARDCODED, not configurable:

```python
MAX_PSEUDO_ROUNDS = 2          # Maximum 2 rounds. No exceptions.
MAX_PSEUDO_FRACTION = 0.30     # From constants.py — pseudo-labels never exceed 30% of train
```

Enforce in the code:
```python
n_pseudo = min(n_pseudo, int(state.canonical_train_rows * MAX_PSEUDO_FRACTION))
```

### Pseudo-label handling in CV

**CRITICAL:** Pseudo-labeled rows must go into TRAINING folds only, never validation folds. They are NEW rows added to the dataset, not replacements. Standard KFold/StratifiedKFold on the augmented dataset naturally handles this IF the pseudo-labeled rows are appended (not interleaved). Verify by checking that validation fold indices never include rows with `_is_pseudo_label == True`.

### State return

```python
return state.validated_update("pseudo_label", {
    "pseudo_label_activated": True,
    "pseudo_label_fraction": final_pseudo_fraction,
    "pseudo_label_cv_delta": final_cv - original_cv,
})
```

### Contract tests: tests/contracts/test_pseudo_label_contract.py

Create fixtures:
```python
@pytest.fixture
def large_test_data(tmp_path):
    """Test set 2x larger than train — pseudo-labels should activate."""
    # train: 1000 rows, test: 2000 rows
    
@pytest.fixture
def small_test_data(tmp_path):
    """Test set smaller than train — pseudo-labels should NOT activate."""
    # train: 1000 rows, test: 500 rows
```

Tests:
1. `test_activates_when_test_larger(large_test_data)` — pseudo_label_activated=True
2. `test_skips_when_test_smaller(small_test_data)` — pseudo_label_activated=False
3. `test_skips_when_critic_critical` — critic_verdict severity="CONFIRMED_CRITICAL" → not activated
4. `test_skips_in_sprint_mode` — pipeline_depth="sprint" → immediate return
5. `test_max_2_rounds` — pseudo_label implementation has max 2 rounds hardcoded
6. `test_pseudo_fraction_capped_at_30pct` — even with huge test set, fraction ≤ 0.30
7. `test_wilcoxon_gate_applied` — mock scores where augmented is worse → pseudo-labels rejected
8. `test_wilcoxon_gate_passes` — mock scores where augmented is significantly better → accepted
9. `test_pseudo_labels_tagged` — augmented data has _is_pseudo_label column
10. `test_revert_on_round2_degradation` — Round 2 worse than Round 1 → reverts to Round 1 (not original)
11. `test_cv_delta_nonnegative` — pseudo_label_cv_delta >= 0 (or pseudo_labels not activated)
12. `test_never_halts_pipeline` — any failure → returns with pseudo_label_activated=False, pipeline continues

---

## COMMIT 2: Freeform Sandbox (tools/freeform_sandbox.py)

### What this does

Closes the flexibility gap with Claude Code. The operator sends `/freeform "Try TabNet with entity embeddings"` and Professor generates a complete standalone script with zero agent constraints — but inside Professor's full safety infrastructure.

### File: tools/freeform_sandbox.py

```python
def run_freeform(
    instruction: str,
    state: ProfessorState,
    timeout: int = 1800,  # 30 minutes default
) -> dict:
    """
    Execute an operator's freeform instruction as a standalone ML script.
    
    Returns:
    {
        "run_id": str,
        "instruction": str,
        "cv_score": float or None,
        "submission_path": str or None,
        "status": "success" | "failed" | "timeout",
        "stdout": str,
        "stderr": str,
        "runtime": float,
    }
    """
```

### The freeform prompt

Build a MINIMAL prompt — no agent constraints, no Feature Factory template, no ML Optimizer repertoire:

```python
freeform_prompt = f"""You are a machine learning engineer. Write a COMPLETE standalone Python script.

COMPETITION: {state.competition_name}
METRIC: {state.metric_name} ({"higher" if state.metric_config.get("higher_is_better", True) else "lower"} is better)
TRAIN FILE: {state.features_train_path} ({state.canonical_train_rows} rows)
TEST FILE: {state.features_test_path}
TARGET: {state.target_column}
COMPETITION TYPE: {state.competition_type}

DATA SCHEMA:
{_format_schema(state.data_schema)}

CURRENT BEST CV SCORE: {state.ensemble_cv or state.cv_mean}

INSTRUCTION FROM OPERATOR:
{instruction}

REQUIREMENTS:
- Load train and test data (Polars: import polars as pl)
- Implement the operator's instruction
- Use {_format_cv_strategy(state.validation_strategy)} for cross-validation
- Print the CV score as: FREEFORM_CV_SCORE={{score}}
- Save predictions to 'freeform_submission.csv' with columns matching sample submission
- Pin random seed to 42
- The script must be COMPLETE and SELF-CONTAINED (no imports from professor)
"""
```

### Execution flow

```python
def run_freeform(instruction, state, timeout=1800):
    run_id = f"freeform_{uuid4().hex[:8]}"
    
    emit_to_operator(f"🔓 Freeform starting: {instruction[:100]}...", level="STATUS")
    
    # 1. Generate code via LLM
    response = llm_call(freeform_prompt, agent_name="freeform")
    code = extract_code(response["text"])
    
    # 2. Execute in sandbox with FULL safety infrastructure
    result = run_in_sandbox(
        code=code,
        timeout=timeout,
        agent_name="freeform",
        purpose=f"Freeform: {instruction[:200]}",
        round_num=0,
        attempt=1,
        llm_prompt=freeform_prompt,
        llm_reasoning=response.get("reasoning", ""),
        expected_row_change="none",
    )
    
    # 3. Parse results
    if result["success"]:
        cv_score = _parse_freeform_score(result["stdout"])
        submission_path = _find_submission_file(state.session_id, "freeform_submission.csv")
        status = "success"
        emit_to_operator(
            f"🔓 Freeform complete: CV={cv_score:.4f} "
            f"({'better' if cv_score > (state.ensemble_cv or 0) else 'worse'} than main pipeline)",
            level="RESULT"
        )
    else:
        cv_score = None
        submission_path = None
        status = "failed"
        emit_to_operator(
            f"🔓 Freeform FAILED after {result.get('runtime', 0):.0f}s. "
            f"Self-Debugging Engine exhausted all retries.",
            level="STATUS"
        )
    
    return {
        "run_id": run_id,
        "instruction": instruction,
        "cv_score": cv_score,
        "submission_path": submission_path,
        "status": status,
        "stdout": result["stdout"][:2000],
        "stderr": result["stderr"][:2000],
        "runtime": result.get("runtime", 0),
    }
```

### HITL command integration

Add `/freeform` command handling to the CommandListener in `tools/operator_channel.py`:

```python
# In CommandListener._classify_message():

if message.startswith("/freeform"):
    parts = message.split(" ", 1)
    if len(parts) == 1:
        # Just "/freeform" with no argument
        subcommand = parts[0].replace("/freeform", "").strip()
    else:
        subcommand = parts[1].strip()
    
    if subcommand == "status":
        # Report current freeform status
        active = [r for r in state.freeform_runs if r["status"] == "running"]
        respond(f"Freeform: {len(active)} active, {len(state.freeform_runs)} total")
    
    elif subcommand == "results":
        # Report all freeform results
        for r in state.freeform_runs:
            respond(f"  {r['run_id']}: CV={r.get('cv_score', 'N/A')} — {r['status']}")
    
    elif subcommand.startswith("include "):
        run_id = subcommand.split(" ")[1]
        # Mark for ensemble inclusion
        for r in state.freeform_runs:
            if r["run_id"] == run_id:
                r["included_in_ensemble"] = True
                respond(f"✅ {run_id} marked for ensemble inclusion")
    
    elif subcommand.startswith("exclude "):
        run_id = subcommand.split(" ")[1]
        for r in state.freeform_runs:
            if r["run_id"] == run_id:
                r["included_in_ensemble"] = False
                respond(f"❌ {run_id} excluded from ensemble")
    
    else:
        # The subcommand IS the instruction (quoted or unquoted)
        instruction = subcommand.strip('"').strip("'")
        # Queue freeform execution
        add_to_injection_queue({"type": "freeform", "instruction": instruction})
```

### Processing freeform in the pipeline

Freeform runs are processed at agent transitions (when the injection queue is checked):

```python
# In process_pending_injections():
for item in injection_queue:
    if item["type"] == "freeform":
        # Run freeform in a background thread (non-blocking)
        import threading
        thread = threading.Thread(
            target=_execute_freeform_background,
            args=(item["instruction"], state),
        )
        thread.start()
        # The main pipeline continues while freeform runs in parallel
```

### Freeform results in ensemble

When the Ensemble Architect runs, it checks `state.freeform_runs` for included results:

```python
# In ensemble_architect.py:
freeform_included = [
    r for r in (state.freeform_runs or [])
    if r.get("included_in_ensemble") and r.get("status") == "success"
]

for fr in freeform_included:
    # Load freeform predictions
    freeform_preds = pl.read_csv(fr["submission_path"])
    # Add to ensemble candidate pool
    oof_preds[f"freeform_{fr['run_id']}"] = freeform_preds["prediction"].to_numpy()
```

### Timeout warning

```python
# Inside run_freeform, monitor execution time:
if elapsed > timeout * 0.8:
    emit_to_operator(
        f"⚠️ Freeform at 80% timeout ({elapsed:.0f}s / {timeout}s)",
        level="STATUS"
    )
```

### State additions

Already in STATE.md:
```python
freeform_runs: list = Field(default_factory=list)   # owner: freeform handler
freeform_active: bool = False                        # owner: freeform handler
```

### Contract tests: tests/contracts/test_freeform_contract.py

1. `test_freeform_produces_result_dict` — result has: run_id, instruction, cv_score, submission_path, status
2. `test_freeform_uses_same_cv_strategy` — generated code uses the same CV strategy as main pipeline (check prompt content)
3. `test_freeform_goes_through_debugging_engine` — mock sandbox failure → Self-Debugging retry cascade activates
4. `test_freeform_leakage_check_runs` — leakage precheck fires on freeform code (agent_name in LEAKAGE_CHECK_AGENTS)
5. `test_freeform_logged_in_code_ledger` — Code Ledger entry created with agent_name="freeform"
6. `test_freeform_result_has_cv_score` — successful freeform → cv_score is a float
7. `test_freeform_failure_doesnt_crash` — LLM generates garbage → status="failed", pipeline continues
8. `test_multiple_freeform_independent` — two freeform runs produce separate entries in freeform_runs
9. `test_include_command_sets_flag` — /freeform include {run_id} → included_in_ensemble=True
10. `test_timeout_warning_at_80pct` — mock slow execution → warning emitted at 80% timeout
11. `test_freeform_prompt_has_schema` — prompt includes data schema and competition details

---

## COMMIT 3: Solution Provenance & Reproducibility (tools/code_ledger.py + tools/solution_assembler.py)

### What this does

After every competition run, Professor produces 3 files:
- `solution_notebook.py` — standalone script that reproduces the exact submission (zero Professor dependencies)
- `solution_writeup.md` — narrative in Kaggle gold-medal writeup format
- `requirements.txt` — pinned library versions

### File: tools/code_ledger.py (upgrade)

The Code Ledger already exists from Layer 0 (captures every sandbox execution). Upgrade it with a query interface:

```python
def get_kept_entries(session_dir: str) -> list[dict]:
    """Return all Code Ledger entries with kept=True, ordered by execution time."""
    entries = _read_ledger(session_dir)
    return [e for e in entries if e.get("kept", True)]

def get_entries_by_agent(session_dir: str, agent_name: str) -> list[dict]:
    """Return all entries for a specific agent."""
    entries = _read_ledger(session_dir)
    return [e for e in entries if e["agent"] == agent_name]

def get_reasoning_chain(session_dir: str) -> list[dict]:
    """Return kept entries with their reasoning fields, for writeup generation."""
    kept = get_kept_entries(session_dir)
    return [
        {
            "agent": e["agent"],
            "purpose": e["purpose"],
            "reasoning": e.get("llm_reasoning", ""),
            "round": e.get("round", 0),
            "success": e["success"],
        }
        for e in kept
    ]
```

### File: tools/solution_assembler.py

```python
def assemble_solution_notebook(
    state: ProfessorState,
    session_dir: str,
) -> str:
    """
    Stitch all kept Code Ledger entries into a single reproducible Python script.
    
    The script:
    - Has ZERO Professor dependencies (no imports from professor/)
    - Uses only standard ML libraries (polars, sklearn, lightgbm, xgboost, catboost, numpy, scipy)
    - Reproduces the exact submission.csv when run with train.csv and test.csv
    - Has pinned random seeds throughout
    
    Returns: path to solution_notebook.py
    """
```

**Assembly process:**

1. **Extract code from kept ledger entries:**
```python
kept_entries = get_kept_entries(session_dir)

# Group by agent in pipeline order
agent_order = ["data_engineer", "feature_factory", "creative_hypothesis",
               "ml_optimizer", "ensemble_architect", "post_processor"]

code_blocks = []
for agent in agent_order:
    agent_entries = [e for e in kept_entries if e["agent"] == agent and e["success"]]
    for entry in agent_entries:
        code_blocks.append({
            "agent": agent,
            "purpose": entry["purpose"],
            "code": entry["code"],
            "round": entry.get("round", 0),
        })
```

2. **Build the notebook script:**

```python
notebook_parts = [
    "#!/usr/bin/env python3",
    '"""',
    f"Solution Notebook — {state.competition_name}",
    f"Generated by Professor v2 on {datetime.utcnow().isoformat()}",
    f"Final CV: {state.ensemble_cv or state.cv_mean}",
    f"Metric: {state.metric_name}",
    '"""',
    "",
    "# === IMPORTS ===",
    "import polars as pl",
    "import numpy as np",
    "import lightgbm as lgb",
    "import xgboost as xgb",
    "import catboost as cb",
    "from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold",
    "from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score",
    "import optuna",
    "optuna.logging.set_verbosity(optuna.logging.WARNING)",
    "",
    "SEED = 42",
    "np.random.seed(SEED)",
    "",
]

# Add each code block with section headers
for block in code_blocks:
    notebook_parts.append(f"# === {block['agent'].upper()} — {block['purpose']} ===")
    notebook_parts.append(block["code"])
    notebook_parts.append("")

# Add submission generation
notebook_parts.extend([
    "# === SAVE SUBMISSION ===",
    "submission = pl.DataFrame({'id': test_ids, 'target': final_predictions})",
    "submission.write_csv('submission.csv')",
    "print(f'Submission saved: {len(submission)} rows')",
])
```

3. **Clean the notebook:**
- Remove Professor-specific imports (`from tools.sandbox import ...`)
- Remove `emit_to_operator()` calls
- Remove `run_in_sandbox()` wrappers (extract inner code)
- Remove Code Ledger metadata logging
- Verify all imports are standard libraries
- Verify all file paths reference local data (train.csv, test.csv)

4. **Generate requirements.txt:**
```python
def _generate_requirements(code: str) -> str:
    """Extract imports and generate pinned requirements."""
    requirements = {
        "polars": ">=0.20.0",
        "numpy": ">=1.24.0",
        "scipy": ">=1.10.0",
        "scikit-learn": ">=1.3.0",
        "lightgbm": ">=4.0.0",
        "xgboost": ">=2.0.0",
        "catboost": ">=1.2.0",
        "optuna": ">=3.4.0",
    }
    # Only include libraries actually imported in the notebook
    used = {pkg: ver for pkg, ver in requirements.items() 
            if pkg.replace("-", "_") in code or pkg in code}
    return "\n".join(f"{pkg}{ver}" for pkg, ver in used.items())
```

5. **Generate writeup:**

```python
def generate_writeup(state: ProfessorState, session_dir: str) -> str:
    """
    Generate solution_writeup.md in Kaggle gold-medal format.
    Uses reasoning fields from Code Ledger — no additional LLM call needed.
    """
    reasoning_chain = get_reasoning_chain(session_dir)
    
    writeup = f"""# {state.competition_name} — Solution Writeup

## Approach Summary
- **Problem type:** {state.competition_type}
- **Metric:** {state.metric_name}
- **Final CV:** {state.ensemble_cv or state.cv_mean:.4f}
- **Models:** {', '.join(m['model_type'] for m in state.model_configs)}
- **Ensemble method:** {state.ensemble_method}

## Data Preprocessing
{_extract_reasoning_for_agent(reasoning_chain, "data_engineer")}

## Feature Engineering
{_extract_reasoning_for_agent(reasoning_chain, "feature_factory")}
- Total features: {len(state.feature_manifest)} surviving from {state.feature_factory_rounds_completed} rounds
- Top features by importance: {_format_top_features(state)}

## Model Training
{_extract_reasoning_for_agent(reasoning_chain, "ml_optimizer")}
- Best model: {state.best_model_type} with CV {state.cv_mean:.4f} ± {state.cv_std:.4f}
- Optuna trials: {state.optuna_trials_completed}

## Ensemble
- Method: {state.ensemble_method}
- Ensemble CV: {state.ensemble_cv:.4f}
- Diversity: {_format_diversity(state.ensemble_diversity_report)}

## Post-Processing
- Applied: {state.postprocess_config.get('method', 'none')}
- Delta: {state.postprocess_cv_delta:+.4f}

## What Worked
{_extract_what_worked(reasoning_chain)}

## What Didn't Work
{_extract_what_didnt_work(reasoning_chain)}
"""
    return writeup
```

6. **Reproduction validation:**

```python
def validate_reproduction(notebook_path: str, session_dir: str, expected_submission: str) -> dict:
    """
    Run the generated notebook in a clean sandbox and verify it reproduces the submission.
    """
    result = run_in_sandbox(
        code=open(notebook_path).read(),
        timeout=600,  # 10 minutes
        agent_name="reproduction_validation",
        purpose="Verify solution_notebook.py reproduces submission.csv",
    )
    
    if result["success"]:
        # Compare generated submission with original
        original = pl.read_csv(expected_submission)
        reproduced = pl.read_csv(os.path.join(session_dir, "submission.csv"))
        
        # Check row count
        rows_match = len(original) == len(reproduced)
        # Check values (allow small floating-point differences)
        values_match = np.allclose(
            original["target"].to_numpy(), 
            reproduced["target"].to_numpy(),
            atol=1e-6
        )
        
        return {
            "reproduced": rows_match and values_match,
            "rows_match": rows_match,
            "values_match": values_match,
            "max_diff": float(np.max(np.abs(
                original["target"].to_numpy() - reproduced["target"].to_numpy()
            ))) if rows_match else None,
        }
    else:
        return {"reproduced": False, "error": result["stderr"][:500]}
```

### Contract tests: tests/contracts/test_solution_provenance_contract.py

1. `test_notebook_produced` — solution_notebook.py file exists in session directory
2. `test_notebook_has_no_professor_imports` — no `from professor`, `from tools`, `from agents` in the notebook
3. `test_notebook_has_seed_pinning` — code contains `random_state=42` or `np.random.seed(42)`
4. `test_requirements_produced` — requirements.txt exists
5. `test_requirements_only_used_libraries` — every library in requirements.txt is actually imported in the notebook
6. `test_writeup_produced` — solution_writeup.md exists
7. `test_writeup_has_required_sections` — contains: Approach Summary, Feature Engineering, Model Training, Ensemble
8. `test_writeup_contains_scores` — writeup contains the cv_mean value
9. `test_reproduction_validation_runs` — mock sandbox, verify validation function executes
10. `test_code_ledger_query_returns_kept_only` — get_kept_entries filters out rejected entries
11. `test_reasoning_chain_has_agent_and_purpose` — each entry has agent and purpose fields

---

## COMMIT 4: Submission Safety Net — Shield 7 (shields/submission_safety.py)

### What this prevents

Both final submissions are correlated (0.998) → private LB shakeup kills both. Or: submission file is corrupted (truncated write, wrong columns). Or: EWMA monitor freezes submissions during crucial final week due to noisy public LB.

### File: shields/submission_safety.py

```python
def verify_submission(
    submission_path: str,
    sample_submission_path: str,
    canonical_test_rows: int,
) -> dict:
    """
    Verify a submission file is valid before marking as final.
    
    Returns:
    {
        "valid": bool,
        "checks": {
            "row_count_ok": bool,
            "columns_ok": bool,
            "no_nans": bool,
            "range_ok": bool,
            "format_ok": bool,
        },
        "issues": [str],
    }
    """
```

Checks:
- Row count matches `canonical_test_rows`
- Column names match sample_submission exactly
- No NaN/null values in prediction columns
- Predictions in valid range (probabilities [0,1] for classification, within training target range for regression)
- File is complete (not truncated — verify last row has all columns)
- File format matches sample_submission (delimiter, quoting)

```python
def check_submission_diversity(
    submission_1_path: str,
    submission_2_path: str,
) -> dict:
    """
    Check diversity between final 2 submissions.
    
    Returns:
    {
        "correlation": float,
        "diversity_rating": "good" | "moderate" | "warning",
    }
    """
    preds_1 = pl.read_csv(submission_1_path)["target"].to_numpy()
    preds_2 = pl.read_csv(submission_2_path)["target"].to_numpy()
    
    corr = float(np.corrcoef(preds_1, preds_2)[0, 1])
    
    if corr > 0.995:
        rating = "warning"  # Nearly identical — shakeup kills both
    elif corr > 0.95:
        rating = "moderate"
    else:
        rating = "good"     # Strong diversity
    
    return {"correlation": round(corr, 4), "diversity_rating": rating}
```

```python
def estimate_lb_noise(
    n_public_rows: int,
    n_total_test_rows: int,
) -> dict:
    """
    Estimate expected noise in public LB based on sample fraction.
    
    Returns:
    {
        "public_fraction": float,
        "noise_level": "high" | "moderate" | "low",
        "gap_threshold": float,  # Adjusted CV/LB gap threshold
    }
    """
    fraction = n_public_rows / n_total_test_rows if n_total_test_rows > 0 else 0.5
    
    if fraction < 0.20:
        return {"public_fraction": fraction, "noise_level": "high", "gap_threshold": 0.01}
    elif fraction < 0.50:
        return {"public_fraction": fraction, "noise_level": "moderate", "gap_threshold": 0.005}
    else:
        return {"public_fraction": fraction, "noise_level": "low", "gap_threshold": 0.003}
```

### EWMA freeze override in endgame

```python
def check_ewma_freeze(
    days_remaining: int,
    cv_lb_gap: float,
    gap_threshold: float,
    operator_approved: bool,
) -> dict:
    """
    Determine whether to freeze submissions.
    
    In final 7 days: freeze requires operator approval (GATE, not auto).
    """
    should_freeze = cv_lb_gap > gap_threshold
    
    if should_freeze and days_remaining <= 7:
        # Final week — freeze is operator's decision, not automatic
        return {
            "freeze_recommended": True,
            "freeze_enforced": False,  # NOT enforced in endgame
            "reason": f"CV/LB gap {cv_lb_gap:.4f} > threshold {gap_threshold}. "
                      f"But in final {days_remaining} days — operator decides.",
            "requires_operator": True,
        }
    elif should_freeze:
        return {
            "freeze_recommended": True,
            "freeze_enforced": True,
            "reason": f"CV/LB gap {cv_lb_gap:.4f} > threshold {gap_threshold}.",
            "requires_operator": False,
        }
    else:
        return {
            "freeze_recommended": False,
            "freeze_enforced": False,
            "reason": "Gap within threshold.",
            "requires_operator": False,
        }
```

### Submission backup

```python
def backup_submission(submission_path: str, session_dir: str) -> str:
    """Copy submission to timestamped backup. Keep last 20."""
    backup_dir = os.path.join(session_dir, "submission_backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"submission_{timestamp}.csv")
    shutil.copy2(submission_path, backup_path)
    
    # Keep only last 20 backups
    backups = sorted(Path(backup_dir).glob("submission_*.csv"))
    for old_backup in backups[:-20]:
        old_backup.unlink()
    
    return backup_path
```

### Contract tests: tests/contracts/test_submission_safety_contract.py

1. `test_valid_submission_passes` — correct row count, columns, no NaN → valid=True
2. `test_wrong_row_count_fails` — 100 rows when expected 250 → valid=False
3. `test_nan_predictions_fails` — NaN in predictions → valid=False
4. `test_wrong_columns_fails` — different column names → valid=False
5. `test_out_of_range_fails` — probabilities > 1.0 → valid=False
6. `test_high_correlation_warning` — two submissions with corr=0.999 → diversity_rating="warning"
7. `test_low_correlation_good` — corr=0.85 → diversity_rating="good"
8. `test_endgame_freeze_requires_operator` — days_remaining=5 → requires_operator=True, freeze_enforced=False
9. `test_early_freeze_auto_enforced` — days_remaining=15 → freeze_enforced=True (no operator needed)
10. `test_backup_created` — backup file exists after backup_submission()
11. `test_backup_max_20` — create 25 backups → only 20 remain
12. `test_lb_noise_small_public` — 15% public → noise_level="high", gap_threshold=0.01

---

## COMMIT 5: Memory Hygiene — Shield 2 (shields/memory_hygiene.py)

### What this prevents

After 15 competitions, ChromaDB has learned patterns. Some are wrong. "Target encoding always helps on healthcare data" from one lucky competition. Professor applies it on the next healthcare competition where it causes overfitting. The system gets WORSE over time.

### Note: This activates after 5+ competitions. Build it now, it becomes valuable later.

### File: shields/memory_hygiene.py

```python
def validate_retrieved_pattern(
    pattern: dict,
    current_data: dict,
    gate_config: dict,
) -> dict:
    """
    Before applying a ChromaDB pattern, validate it on current data via Wilcoxon.
    
    Returns:
    {
        "validated": bool,
        "method": "wilcoxon_passed" | "wilcoxon_failed" | "untestable",
        "p_value": float or None,
    }
    """

def check_contradiction(
    new_rule: dict,
    existing_rules: list[dict],
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """
    Check if a new rule contradicts any existing rules.
    Uses cosine similarity on rule text embeddings.
    
    Returns list of conflicting rules (empty if no conflicts).
    """

def decay_unused_rules(
    rules: list[dict],
    competitions_since_last_use: int = 10,
    decay_rate: float = 0.1,
) -> list[dict]:
    """
    Rules not confirmed in last N competitions lose confidence.
    """
    for rule in rules:
        if rule.get("last_confirmed_competition", 0) < current_competition - competitions_since_last_use:
            rule["confidence"] -= decay_rate
            rule["confidence"] = max(rule["confidence"], 0.0)
    return rules

def quarantine_pattern(
    pattern: dict,
    reason: str,
    chromadb_collection: str = "failed_patterns",
) -> None:
    """
    Move a pattern to the quarantine collection.
    Agents see quarantined patterns as anti-patterns: DO NOT apply.
    """
```

### Contract tests: tests/contracts/test_memory_hygiene_contract.py

1. `test_retrieved_pattern_validated_via_wilcoxon` — pattern tested before application, not blind
2. `test_failed_pattern_quarantined` — pattern that degrades score → moved to failed_patterns
3. `test_contradiction_detected` — two semantically opposite rules flagged
4. `test_confidence_decays_on_unused` — rule unused for 10 competitions → confidence drops
5. `test_max_20_rules_enforced` — 21st rule evicts lowest-confidence
6. `test_quarantined_patterns_become_anti_patterns` — quarantined rules returned as "avoid" list

---

## COMMIT 6: Publisher (agents/publisher.py)

### What this does

Final agent in the pipeline. Generates all output files, runs reproduction validation, emits Milestone 4 (final summary).

### The LangGraph node function

```python
def publisher(state: ProfessorState) -> dict:
    """
    Final pipeline agent. Produces all outputs.
    
    Reads: EVERYTHING (needs full state for writeup + notebook)
    Writes: code_ledger_path, solution_notebook_path, solution_writeup_path,
            notebook_reproduction_validated, notebook_reproduction_diff, submission_path
    Emits: RESULT (Milestone 4 — final summary), GATE (GUIDED mode: submit?)
    """
```

### Steps

1. **Verify submission via Shield 7:**
```python
from shields.submission_safety import verify_submission, backup_submission

verification = verify_submission(
    submission_path=current_submission_path,
    sample_submission_path=state.preflight_submission_format.get("path", ""),
    canonical_test_rows=state.canonical_test_rows,
)

if not verification["valid"]:
    # Try to fix (most common: wrong column order)
    # If unfixable: emit ESCALATION
    emit_to_operator(
        f"🚨 Submission verification FAILED: {verification['issues']}",
        level="ESCALATION"
    )

# Backup
backup_path = backup_submission(current_submission_path, session_dir)
```

2. **Assemble solution notebook + writeup + requirements:**
```python
from tools.solution_assembler import assemble_solution_notebook, generate_writeup

notebook_path = assemble_solution_notebook(state, session_dir)
writeup_path = generate_writeup(state, session_dir)
requirements_path = _generate_requirements_file(notebook_path, session_dir)
```

3. **Run reproduction validation:**
```python
from tools.solution_assembler import validate_reproduction

repro = validate_reproduction(notebook_path, session_dir, current_submission_path)

if repro["reproduced"]:
    emit_to_operator("✅ Reproduction validated — notebook produces identical submission", level="STATUS")
else:
    emit_to_operator(
        f"⚠️ Reproduction mismatch: {repro.get('error', f'max_diff={repro.get(\"max_diff\", \"N/A\")}')}",
        level="STATUS"
    )
```

4. **Generate cost report:**
```python
from shields.cost_governor import get_governor

gov = get_governor()
cost_summary = gov.get_summary()
```

5. **Emit Milestone 4 (final summary):**
```
✅ RUN COMPLETE
Final CV: 0.8341
Ensemble: hill_climbing (LGB=0.45, XGB=0.35, CAT=0.20)
Post-processing: threshold_sweep (+0.0003)
Pseudo-labels: activated (+0.0012)

📁 Outputs:
  submission.csv — verified ✓
  solution_notebook.py — reproduction validated ✓
  solution_writeup.md — generated ✓
  requirements.txt — generated ✓
  code_ledger.jsonl — {n_entries} entries

💰 Cost: {total_calls} LLM calls, ${total_cost:.2f}
⏱️ Runtime: {total_runtime}

Submission backed up. /submit to finalize.
```

6. **In GUIDED mode: emit GATE for final submission approval:**
```python
if state.hitl_mode == "guided":
    response = emit_to_operator(
        "🎯 Ready to submit. Reply /submit to confirm or /iterate to run again.",
        level="GATE"
    )
```

### Contract tests: tests/contracts/test_publisher_contract.py

1. `test_submission_verified` — verify_submission called on final submission
2. `test_notebook_generated` — solution_notebook_path points to existing file
3. `test_writeup_generated` — solution_writeup_path points to existing file
4. `test_reproduction_validation_runs` — validate_reproduction called
5. `test_backup_created` — submission backup exists
6. `test_milestone_4_emitted` — RESULT message emitted with final summary
7. `test_guided_mode_gate` — hitl_mode="guided" → GATE emitted for submit confirmation
8. `test_cost_report_in_summary` — final message includes LLM call count and cost

---

## COMMIT 7: Supervisor + Graph Wiring (agents/supervisor.py + graph/builder.py)

### What this does

The Supervisor routes between agents on replan, manages dag_version, and handles the conditional edges. The graph builder wires all nodes together.

### agents/supervisor.py

```python
def supervisor(state: ProfessorState) -> dict:
    """
    Routes replan decisions. Increments dag_version.
    
    Reads: critic_verdict, dag_version, replan_target (from routing logic)
    Writes: dag_version, replan_target
    Emits: STATUS (replan reason)
    """
    new_dag_version = state.dag_version + 1
    
    # Determine replan target from critic findings
    replan_target = _determine_replan_target(state.critic_verdict)
    
    emit_to_operator(
        f"🔄 Replanning from {replan_target} (dag_version {new_dag_version}). "
        f"Reason: {state.critic_verdict.get('findings', [{}])[0].get('evidence', 'unknown')[:100]}",
        level="STATUS"
    )
    
    return state.validated_update("supervisor", {
        "dag_version": new_dag_version,
        "replan_target": replan_target,
    })


def _determine_replan_target(critic_verdict: dict) -> str:
    """Map critic findings to the earliest affected agent."""
    findings = critic_verdict.get("findings", [])
    
    for finding in findings:
        if finding.get("severity") == "CONFIRMED_CRITICAL":
            rerun = finding.get("replan_instructions", {}).get("rerun_nodes", [])
            if "feature_factory" in rerun:
                return "feature_factory"
            elif "ml_optimizer" in rerun:
                return "ml_optimizer"
    
    return "feature_factory"  # Default: restart from feature engineering
```

### graph/builder.py

```python
def build_professor_graph() -> StateGraph:
    """Build the complete Professor v2 LangGraph pipeline."""
    graph = StateGraph(ProfessorState)
    
    # === ADD ALL NODES ===
    graph.add_node("preflight_checks", run_preflight_checks)
    graph.add_node("competition_intel", competition_intel)
    graph.add_node("metric_verification_gate", run_metric_verification_gate)
    graph.add_node("data_engineer", data_engineer)
    graph.add_node("shift_detector", shift_detector)
    graph.add_node("eda_agent", eda_agent)
    graph.add_node("domain_research", domain_research)
    graph.add_node("validation_architect", validation_architect)
    graph.add_node("problem_reframer", problem_reframer)
    graph.add_node("feature_factory", feature_factory)
    graph.add_node("creative_hypothesis", creative_hypothesis)
    graph.add_node("ml_optimizer", ml_optimizer)
    graph.add_node("red_team_critic", red_team_critic)
    graph.add_node("supervisor", supervisor)
    graph.add_node("self_reflection", self_reflection)
    graph.add_node("pseudo_label", pseudo_label_architect)
    graph.add_node("ensemble_architect", ensemble_architect)
    graph.add_node("post_processor", post_processor)
    graph.add_node("publisher", publisher)
    
    # === SEQUENTIAL EDGES ===
    graph.set_entry_point("preflight_checks")
    graph.add_edge("preflight_checks", "competition_intel")
    graph.add_edge("competition_intel", "metric_verification_gate")
    graph.add_edge("metric_verification_gate", "data_engineer")
    graph.add_edge("data_engineer", "shift_detector")
    graph.add_edge("shift_detector", "eda_agent")
    graph.add_edge("eda_agent", "domain_research")
    graph.add_edge("domain_research", "validation_architect")
    graph.add_edge("validation_architect", "problem_reframer")
    graph.add_edge("problem_reframer", "feature_factory")
    graph.add_edge("feature_factory", "creative_hypothesis")
    graph.add_edge("creative_hypothesis", "ml_optimizer")
    graph.add_edge("ml_optimizer", "red_team_critic")
    
    # === CONDITIONAL: After Critic ===
    graph.add_conditional_edges(
        "red_team_critic",
        _route_after_critic,
        {
            "replan": "supervisor",
            "continue": "self_reflection",
        }
    )
    
    # === CONDITIONAL: After Supervisor ===
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s.replan_target,
        {
            "feature_factory": "feature_factory",
            "ml_optimizer": "ml_optimizer",
            "problem_reframer": "problem_reframer",
        }
    )
    
    # === REMAINING SEQUENTIAL ===
    graph.add_edge("self_reflection", "pseudo_label")
    graph.add_edge("pseudo_label", "ensemble_architect")
    graph.add_edge("ensemble_architect", "post_processor")
    graph.add_edge("post_processor", "publisher")
    graph.add_edge("publisher", END)
    
    return graph


def _route_after_critic(state: ProfessorState) -> str:
    """Decide: replan or continue after Critic."""
    verdict = state.critic_verdict
    severity = verdict.get("severity", "CLEAR")
    dag_version = state.dag_version or 0
    
    if severity == "CONFIRMED_CRITICAL" and dag_version < MAX_REPLAN_CYCLES:
        return "replan"
    else:
        return "continue"
```

### HITL injection hook

Add a wrapper that checks for pending HITL injections at every node transition:

```python
def _wrap_with_hitl_check(func):
    """Decorator that processes HITL queue before running the agent."""
    def wrapper(state: ProfessorState) -> dict:
        # Check pause/abort
        if state.pipeline_paused:
            emit_to_operator("⏸️ Pipeline paused. Send /resume.", level="STATUS")
            _wait_for_resume()
        if state.pipeline_aborted:
            raise PipelineAborted()
        
        # Process pending injections
        state = process_pending_injections(state)
        
        # Run the actual agent
        return func(state)
    
    return wrapper
```

Apply to all nodes:
```python
graph.add_node("feature_factory", _wrap_with_hitl_check(feature_factory))
# ... for all nodes
```

### Contract tests: tests/contracts/test_supervisor_contract.py

1. `test_dag_version_increments` — replan increments dag_version by 1
2. `test_max_replan_cycles_respected` — dag_version >= MAX_REPLAN_CYCLES → route to "continue" not "replan"
3. `test_replan_target_from_findings` — CONFIRMED_CRITICAL on feature → replan_target="feature_factory"
4. `test_clear_verdict_continues` — severity="CLEAR" → routes to self_reflection
5. `test_hitl_pause_blocks` — pipeline_paused=True → wrapper blocks until resumed
6. `test_graph_has_all_nodes` — compiled graph contains all 18 agent nodes
7. `test_entry_point_is_preflight` — graph entry point is "preflight_checks"
8. `test_graph_reaches_end` — mock all agents to return empty → graph reaches END without error

---

## COMMIT 8: Full Pipeline Integration Test

### tests/contracts/test_full_pipeline_integration.py

The final test. Run the ENTIRE Professor v2 pipeline on a mock competition.

```python
@pytest.fixture
def mock_competition(tmp_path):
    """Complete mock competition with train, test, sample_submission."""
    np.random.seed(42)
    n_train, n_test = 2000, 500
    
    train = pl.DataFrame({
        "id": range(n_train),
        "feat_1": np.random.normal(0, 1, n_train).tolist(),
        "feat_2": np.random.exponential(2, n_train).tolist(),
        "feat_3": np.random.choice(["A", "B", "C"], n_train).tolist(),
        "target": np.random.randint(0, 2, n_train).tolist(),
    })
    test = pl.DataFrame({
        "id": range(n_test),
        "feat_1": np.random.normal(0, 1, n_test).tolist(),
        "feat_2": np.random.exponential(2, n_test).tolist(),
        "feat_3": np.random.choice(["A", "B", "C"], n_test).tolist(),
    })
    sample_sub = pl.DataFrame({"id": range(n_test), "target": [0.5] * n_test})
    
    train.write_csv(tmp_path / "train.csv")
    test.write_csv(tmp_path / "test.csv")
    sample_sub.write_csv(tmp_path / "sample_submission.csv")
    return tmp_path


def test_full_pipeline_produces_submission(mock_competition):
    """End-to-end: competition files in → submission.csv out."""
    # Build and compile graph
    graph = build_professor_graph()
    app = graph.compile()
    
    initial_state = ProfessorState(
        session_id="test-e2e",
        competition_name="test-binary-classification",
        raw_data_path=str(mock_competition),
        hitl_mode="autonomous",  # No blocking
        hitl_channels=[],        # No HITL output
    )
    
    # Mock LLM calls and potentially sandbox for speed
    with patch("tools.llm_provider.llm_call") as mock_llm:
        mock_llm.return_value = {"text": "...", "reasoning": "", 
                                  "input_tokens": 100, "output_tokens": 200,
                                  "model": "test", "cost_usd": 0.001}
        
        final_state = app.invoke(initial_state)
    
    # Verify core outputs exist
    assert final_state.submission_path != ""
    assert final_state.cv_mean > 0
    assert final_state.metric_verified  # Metric gate passed
    assert final_state.preflight_passed  # Pre-flight passed
    assert len(final_state.feature_manifest) > 0  # Features generated
    assert len(final_state.model_configs) > 0  # Models trained
    assert final_state.ensemble_method != ""  # Ensemble selected
    assert final_state.solution_notebook_path != ""  # Notebook generated


def test_full_pipeline_no_state_violations(mock_competition):
    """Full pipeline run produces zero ValidationError or OwnershipError."""
    # Same as above but verify no exceptions logged
    

def test_full_pipeline_all_milestones_emitted(mock_competition):
    """All 4 milestones + Milestone 0 emitted during full run."""
    with patch("tools.operator_channel.emit_to_operator") as mock_emit:
        # Run pipeline
        # Verify at least 5 CHECKPOINT/GATE emissions (M0, M1, M2, M3, M4)


def test_sprint_mode_completes_faster(mock_competition):
    """SPRINT mode skips agents and completes in fewer LLM calls."""
    # Run in STANDARD mode, count LLM calls
    # Run in SPRINT mode, count LLM calls
    # Verify SPRINT < STANDARD


def test_critic_replan_works(mock_competition):
    """Inject leakage → Critic catches → replan → second pass succeeds."""
    # Mock Feature Factory to introduce leakage on first pass
    # Mock it to fix leakage on second pass (after replan)
    # Verify dag_version increments
    # Verify final submission is clean
```

---

## WHAT NOT TO DO

- Do NOT let pseudo-labels exceed 30% of training size. HARDCODED cap.
- Do NOT activate pseudo-labels when test < train or Critic found CRITICAL.
- Do NOT let freeform scripts bypass the leakage precheck or output validation.
- Do NOT include Professor imports in solution_notebook.py. It must be standalone.
- Do NOT auto-freeze submissions in the final 7 days. Operator decides.
- Do NOT let memory hygiene blindly apply patterns. Every pattern goes through Wilcoxon.
- Do NOT skip reproduction validation. If the notebook doesn't reproduce the submission, log a warning.
- Do NOT modify any Layer 0-3 files except documented integration points (ensemble reads freeform results, HITL processes /freeform commands).
- Do NOT use Pandas. Polars only. Convert to numpy at the model boundary.
- Do NOT let the publisher crash if any single output generation fails. Generate what you can, log failures for the rest.