# BUILD PROMPT — Layer 2: Feature Engineering (Days 6-8)
# Feed to Gemini CLI with: @PROFESSOR.md @STATE.md @SANDBOX.md @HITL.md @POLARS.md @CONTRACTS.md @PROMPTS.md @PROVIDERS.md

---

## CONTEXT

Layers 0 and 1 are complete and passing all contract tests:
- Layer 0: ProfessorState (typed, ownership-enforced), Self-Debugging Engine (4-layer retry), HITL (CLI+Telegram), Cost Governor, Metric Verification Gate, Data Integrity Checkpoints
- Layer 1: Pre-Flight Checks + Complexity-Gated Depth, Deep EDA + Artifact Export, Domain Research Engine, Shift Detector

Now you're building the feature engineering layer — the agents that generate, validate, and refine the features the model will train on. This is where Professor's competitive edge lives. Every feature passes through statistical gates. No exceptions.

All Layer 2 components read context from Layer 1 (eda_insights_summary, domain_brief, shifted_features, technique_brief) and write through `validated_update()`. They execute code through `run_in_sandbox()` with full Self-Debugging Engine support. They emit through `emit_to_operator()`.

---

## COMMIT PLAN (7 commits)

```
Commit 1:  guards/leakage_precheck.py + sandbox integration + tests
Commit 2:  guards/data_usage_checker.py + data_engineer integration + tests
Commit 3:  agents/_retry_utils.py + retry loop integration across agents + tests
Commit 4:  agents/competition_intel.py SOTA search extension + tests
Commit 5:  agents/problem_reframer.py + tests
Commit 6:  agents/feature_factory.py (with Adaptive Gates from Fix 1) + tests
Commit 7:  agents/creative_hypothesis.py + tests
```

Every commit passes `pytest tests/contracts/ -q` including ALL Layer 0 + Layer 1 tests. No exceptions. If an earlier test breaks, fix before committing.

---

## COMMIT 1: Pre-Execution Leakage Check (guards/leakage_precheck.py)

### What this prevents

LLM-generated code fits a StandardScaler on the full dataset before the CV split. The model trains on leaked data. The Critic catches it 15 minutes later AFTER training. Total waste: 15 minutes + one replan cycle. Pre-execution check catches it in <1ms BEFORE execution.

### File 1: guards/leakage_precheck.py

```python
def check_code_for_leakage(code: str) -> dict:
    """
    Scan generated code for data leakage patterns BEFORE execution.
    
    Returns:
    {
        "leakage_detected": bool,
        "line": int or None,
        "code_line": str or None,
        "description": str or None,
        "fix_suggestion": str or None,
    }
    """
```

**Danger patterns — regex list with descriptions:**
```python
DANGER_PATTERNS = [
    (r"\.fit_transform\(X\)", "fit_transform on variable named X (likely full dataset)"),
    (r"\.fit_transform\(df\)", "fit_transform on variable named df (likely full dataset)"),
    (r"\.fit_transform\(data\)", "fit_transform on variable named data (likely full dataset)"),
    (r"\.fit\(X\).*\.transform", "fit on X then transform — X may be full dataset"),
    (r"StandardScaler\(\)\.fit\((?!.*train)", "StandardScaler fit on non-train variable"),
    (r"MinMaxScaler\(\)\.fit\((?!.*train)", "MinMaxScaler fit on non-train variable"),
    (r"SimpleImputer\(\)\.fit\((?!.*train)", "SimpleImputer fit on non-train variable"),
    (r"TargetEncoder\(\)\.fit\((?!.*train)", "TargetEncoder fit outside CV fold"),
    (r"LabelEncoder\(\)\.fit\(.*(?:concat|vstack)", "LabelEncoder fit on combined train+test"),
    (r"\.fit\(.*(?:concat|vstack|rbind)", "fit on concatenated train+test data"),
]
```

**Safe pattern whitelist — lines matching these are NOT flagged:**
```python
SAFE_PATTERNS = [
    r"Pipeline\(",               # sklearn Pipeline handles fit/transform correctly
    r"ColumnTransformer\(",      # same
    r"cross_val_score\(",        # sklearn cross-validation handles internally
    r"cross_val_predict\(",      # same
    r"make_pipeline\(",          # same
    r"\.fit\(X_train",           # Explicit train-only fitting
    r"\.fit\(train_",            # Explicit train prefix
    r"\.fit_transform\(X_train", # Explicit train-only fit_transform
]
```

**Detection logic:**
For each line in the code:
1. Check if any danger pattern matches this line
2. If match found: check surrounding context (4 lines above, 3 lines below) for any safe pattern
3. If safe pattern found in context: skip (this is inside a Pipeline or explicit train split)
4. If no safe pattern: return leakage_detected=True with the specific line, description, and fix suggestion

Fix suggestion is always: "Use .fit() on training data only (X_train), or use sklearn Pipeline which handles fold-correct fitting internally."

### File 2: Integration into tools/sandbox.py

Add the pre-execution check inside `run_in_sandbox()`, BEFORE the code executes:

```python
# Only check for agents that generate training code
LEAKAGE_CHECK_AGENTS = {"data_engineer", "feature_factory", "ml_optimizer", 
                         "creative_hypothesis", "post_processor"}

if agent_name in LEAKAGE_CHECK_AGENTS:
    from guards.leakage_precheck import check_code_for_leakage
    leakage = check_code_for_leakage(code)
    if leakage["leakage_detected"]:
        # Do NOT execute the code. Return as failure.
        return {
            "success": False,
            "stdout": "",
            "stderr": (
                f"PRE-EXECUTION LEAKAGE DETECTED: {leakage['description']}\n"
                f"Line {leakage['line']}: {leakage['code_line']}\n"
                f"Fix: {leakage['fix_suggestion']}\n"
                f"Code was NOT executed to prevent wasted compute."
            ),
            "runtime": 0.0,
            "entry_id": _next_entry_id(),
            "diagnostics": {"leakage_precheck": leakage},
            "integrity_ok": True,
            "pre_execution_blocked": True,
        }
```

The Self-Debugging Engine sees this as a failure and generates a retry with the specific leakage fix instruction. The error message tells the LLM EXACTLY what to fix: which line, what pattern, and how to correct it.

**Key design decision:** The pre-check does NOT modify code. It blocks execution and lets the retry cascade handle the fix. This keeps the separation clean — leakage_precheck only detects, never transforms.

**Relationship to Critic Vector 1d:** Pre-execution catches obvious patterns BEFORE compute. Critic Vector 1d catches subtler patterns AFTER training (leakage from dynamic variable assignment, multi-file data flow, etc.). They are defense-in-depth.

### Contract tests: tests/contracts/test_leakage_precheck_contract.py

1. `test_scaler_on_full_data_blocked` — `StandardScaler().fit_transform(X)` → leakage_detected=True
2. `test_scaler_on_train_allowed` — `StandardScaler().fit_transform(X_train)` → leakage_detected=False
3. `test_pipeline_not_blocked` — `Pipeline([('scaler', StandardScaler())])` → leakage_detected=False
4. `test_cross_val_score_not_blocked` — code inside `cross_val_score()` context → leakage_detected=False
5. `test_concat_fit_blocked` — `LabelEncoder().fit(pd.concat([train, test]))` → leakage_detected=True
6. `test_blocked_result_has_line_number` — line field is populated with correct line number
7. `test_blocked_result_has_fix_suggestion` — fix_suggestion is non-empty
8. `test_non_training_agents_skip_check` — agent_name="eda_agent" → no leakage check runs, code executes normally
9. `test_precheck_cost_under_1ms` — 200-line code block checked in < 1 millisecond

---

## COMMIT 2: Data Usage Checker (guards/data_usage_checker.py)

### What this prevents

Competition provides 5 data files. Data Engineer only loads train.csv. Nobody notices metadata.csv (45KB) containing useful categorical mappings. MLE-STAR demonstrated this: their data usage checker caught cases where ignored datasets improved scores.

### File: guards/data_usage_checker.py

```python
DATA_EXTENSIONS = {".csv", ".parquet", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".feather"}
IGNORE_FILES = {"sample_submission.csv", "samplesubmission.csv", "sample_sub.csv"}

def check_data_usage(
    data_dir: str,
    generated_code: str,
) -> dict:
    """
    Compare available data files against what the generated code references.
    
    Returns:
    {
        "total_data_files": int,
        "used_files": list[str],
        "unused_files": list[str],
        "all_data_used": bool,
    }
    """
```

Logic:
1. Walk `data_dir` with `os.listdir()`. For each file:
   - Check extension against `DATA_EXTENSIONS`
   - Skip if filename (lowercased) is in `IGNORE_FILES`
   - Add to `data_files` list
2. For each data file, check if it's referenced in `generated_code`:
   - Check exact filename: `"train.csv" in code`
   - Check stem (without extension): `"train" in code` — handles cases like `pl.read_csv("train")`
   - Check with quotes: `f'"{fname}"' in code or f"'{fname}'" in code`
   - If any match: it's "used". Otherwise: "unused".
3. Return the report

### Integration into agents/data_engineer.py

After the Data Engineer generates and executes its code, call the checker:

```python
from guards.data_usage_checker import check_data_usage

usage_report = check_data_usage(
    data_dir=state.raw_data_path,
    generated_code=generated_code,
)

if usage_report["unused_files"]:
    # Inject into downstream prompts — Feature Factory and Creative Hypothesis
    # will see: "These data files exist but weren't used: [metadata.csv, extra_data.parquet]"
    unused_warning = (
        f"WARNING: These data files exist but were NOT used by the Data Engineer: "
        f"{usage_report['unused_files']}. Consider whether joining or using them "
        f"could improve the model. Do NOT use them blindly — only if relevant."
    )
    emit_to_operator(f"📁 Unused data files: {usage_report['unused_files']}", level="STATUS")
```

**Key design decision:** The checker FLAGS unused files. It does NOT auto-load them. The decision to use supplementary data is made by the LLM with domain context. Auto-loading unknown files creates data quality risks.

**State additions:**
```python
data_usage_report: dict = Field(default_factory=dict)  # owner: data_engineer
```

### Contract tests: tests/contracts/test_data_usage_checker_contract.py

Create test fixtures with `tmp_path`:

1. `test_all_files_used` — code references all data files → all_data_used=True
2. `test_unused_file_flagged` — code references train.csv only, metadata.csv exists → metadata.csv in unused_files
3. `test_sample_submission_ignored` — sample_submission.csv never appears in data_files or unused_files
4. `test_non_data_extensions_ignored` — README.md, LICENSE.txt not in data_files
5. `test_stem_matching_works` — code contains `"train"` (no extension) → train.csv is marked as used
6. `test_empty_directory` — no data files → all_data_used=True, total_data_files=0, no crash
7. `test_report_has_required_keys` — total_data_files, used_files, unused_files, all_data_used all present

---

## COMMIT 3: Error-Classified Retry Guidance (agents/_retry_utils.py)

### What this prevents

v1's retry loop feeds raw tracebacks back to the LLM: "PREVIOUS ATTEMPT FAILED. DO NOT REPEAT." The LLM sees a 500-line LightGBM traceback and guesses. This fix classifies the error FIRST, then gives the LLM targeted guidance: "This is a data quality issue. Most likely: missing column. Check column names against schema."

Professor already has `_classify_error()` in the circuit breaker. This wires it into the retry prompt.

### File: agents/_retry_utils.py

Build a `RETRY_GUIDANCE` dict mapping each error class to specific, actionable guidance:

```python
RETRY_GUIDANCE = {
    "data_quality": (
        "ERROR CLASSIFICATION: Data quality issue.\n"
        "MOST LIKELY CAUSES:\n"
        "- Missing column (check column names against data schema)\n"
        "- Type mismatch (categorical treated as numeric or vice versa)\n"
        "- NaN/null values in unexpected places\n"
        "- Empty dataframe after filtering or join\n"
        "FIX STRATEGY: Print df.columns and df.dtypes FIRST. "
        "Verify column exists before accessing. "
        "Add .drop_nulls() or .fill_null() before operations that don't accept nulls."
    ),
    "model_failure": (
        "ERROR CLASSIFICATION: Model training failure.\n"
        "MOST LIKELY CAUSES:\n"
        "- Hyperparameters too aggressive (learning_rate too high, max_depth too deep)\n"
        "- Feature matrix contains inf or NaN values\n"
        "- Target variable has wrong format or type\n"
        "- Too few samples for the model complexity\n"
        "FIX STRATEGY: Use conservative defaults: n_estimators=100, max_depth=6, "
        "learning_rate=0.1. Add np.isfinite() check on feature matrix before training. "
        "Verify target dtype matches task type."
    ),
    "memory": (
        "ERROR CLASSIFICATION: Memory exhaustion.\n"
        "MOST LIKELY CAUSES:\n"
        "- Dataset too large for available RAM\n"
        "- Too many features created simultaneously\n"
        "- Model with n_jobs=-1 using all cores\n"
        "FIX STRATEGY: Set n_jobs=1. Reduce n_estimators to 100. "
        "Process features in batches of 10. "
        "Use float32 instead of float64: df = df.cast({col: pl.Float32 for col in float_cols})."
    ),
    "api_timeout": (
        "ERROR CLASSIFICATION: External API timeout.\n"
        "This is NOT a code bug. The LLM API rate limit was hit or network is slow.\n"
        "FIX STRATEGY: Reduce prompt size. Truncate context. "
        "The service_health wrapper handles retries automatically. "
        "Do NOT change the actual ML code — the issue is infrastructure, not logic."
    ),
    "unknown": (
        "ERROR CLASSIFICATION: Unclassified error.\n"
        "FIX STRATEGY: Read the traceback carefully. Identify the exact line. "
        "Print the variable state on the line BEFORE the failure. "
        "Do NOT change unrelated code — isolate the fix to the failing operation."
    ),
}
```

Build the retry prompt builder:

```python
def build_retry_prompt(
    error: Exception,
    traceback_str: str,
    attempt: int,
    agent_name: str,
) -> str:
    """
    Build a targeted retry prompt based on error classification.
    Prepended to the agent's system prompt on retry.
    """
    # Classify using existing circuit breaker function
    error_class = _classify_error(agent_name, error)
    guidance = RETRY_GUIDANCE.get(error_class, RETRY_GUIDANCE["unknown"])
    
    # Truncate traceback — LightGBM/XGBoost produce 200-500 line C++ traces
    # Keep first 5 lines (Python call stack) and last 25 lines (actual error)
    tb_lines = traceback_str.strip().split("\n")
    if len(tb_lines) > 30:
        tb_truncated = "\n".join(
            tb_lines[:5] + ["... (truncated %d lines) ..." % (len(tb_lines) - 30)] + tb_lines[-25:]
        )
    else:
        tb_truncated = traceback_str
    
    return (
        f"\n{'='*60}\n"
        f"ATTEMPT {attempt} FAILED. DO NOT REPEAT THE SAME MISTAKE.\n"
        f"{'='*60}\n\n"
        f"{guidance}\n\n"
        f"ACTUAL ERROR (truncated):\n{tb_truncated}\n\n"
        f"CRITICAL: Fix the SPECIFIC issue identified above. "
        f"Do not rewrite unrelated code.\n"
        f"{'='*60}\n"
    )
```

### Integration

This replaces the generic retry prefix in every agent's retry path. The Self-Debugging Engine (Layer 2 in sandbox.py) already handles the retry cascade. `build_retry_prompt` is called inside the Layer 2 error classification step when generating the retry LLM call.

Wire it in `tools/sandbox.py` where the Layer 2 retry prompt is built:

```python
# In the retry cascade, when Layer 2 activates:
from agents._retry_utils import build_retry_prompt

retry_prefix = build_retry_prompt(
    error=captured_exception,
    traceback_str=diagnostics["error"]["traceback"],
    attempt=current_attempt,
    agent_name=agent_name,
)
# Prepend to the original prompt for the LLM retry call
retry_prompt = retry_prefix + "\n" + original_prompt
```

### Contract tests: tests/contracts/test_retry_guidance_contract.py

1. `test_keyerror_classified_as_data_quality` — KeyError → guidance contains "Missing column"
2. `test_memory_error_classified_as_memory` — MemoryError → guidance contains "n_jobs=1"
3. `test_timeout_classified_as_api_timeout` — TimeoutError → guidance contains "NOT a code bug"
4. `test_unknown_error_gets_generic` — random RuntimeError → guidance contains "Read the traceback"
5. `test_traceback_truncated` — 500-line traceback → output ≤ 35 lines (5 + marker + 25 + headers)
6. `test_short_traceback_preserved` — 10-line traceback → all 10 lines present
7. `test_attempt_number_in_output` — attempt=3 → output contains "ATTEMPT 3"
8. `test_original_error_text_present` — the actual exception message appears in the output

---

## COMMIT 4: SOTA Technique Search (agents/competition_intel.py extension)

### What this prevents

DeepSeek-R1's training cutoff means it defaults to LightGBM with standard params, never suggests RealMLP, TabPFN, or recent ensemble techniques. MLE-STAR showed that web search for current model architectures accounted for a significant portion of their 25.8% → 63.6% medal improvement.

### Extension to agents/competition_intel.py

Add a new function `_search_sota_techniques()` called within the existing Competition Intel agent:

```python
def _search_sota_techniques(
    task_type: str,
    target_type: str,
    metric_name: str,
    n_rows: int,
    domain: str,
) -> dict:
    """
    Search the web for current best-practice ML approaches.
    
    Returns:
    {
        "recommended_models": [{"name": str, "library": str, "rationale": str}],
        "hyperparameter_hints": [{"model": str, "param": str, "value": str, "rationale": str}],
        "preprocessing_hints": [str],
        "ensemble_hints": [str],
        "sources_consulted": int,
        "confidence": "high" | "medium" | "low",
    }
    """
```

**Step 1 — Generate 3 search queries:**
```python
queries = [
    f"best {task_type} model {metric_name} kaggle 2025 2026",
    f"{task_type} {target_type} state of the art approach",
    f"kaggle {metric_name} winning solution technique",
]
```

**Step 2 — Execute web search:**
Use whatever web search tool is available. If no web search: return empty dict (graceful degradation). Fetch top 3 results per query. Extract text content.

**Step 3 — LLM synthesis:**
Feed all search results to `llm_call()` with a structured prompt demanding JSON output:

```
You are reading ML technique search results for a {task_type} 
competition evaluated on {metric_name}.

Extract ONLY:
1. Model architectures mentioned (with library + version if stated)
2. Key hyperparameter recommendations specific to this task type
3. Preprocessing techniques specific to this metric
4. Ensemble strategies mentioned

Ignore: generic advice, outdated approaches (pre-2024),
approaches requiring multi-GPU or TPU.

Respond with ONLY valid JSON (no markdown, no explanation):
{
    "recommended_models": [...],
    "hyperparameter_hints": [...],
    "preprocessing_hints": [...],
    "ensemble_hints": [...],
    "sources_consulted": <number>,
    "confidence": "high" | "medium" | "low"
}
```

**Step 4 — Validate and store:**
Parse JSON. If parsing fails: return empty dict with confidence="low". Validate that recommended model libraries are in the known installable set (lightgbm, xgboost, catboost, sklearn, optuna, tabnet — NOT pytorch, tensorflow for v2). Remove any entries requiring unavailable libraries.

**Downstream injection:**
- `ml_optimizer` prompt: "Consider these SOTA approaches: {recommended_models}. Start with the highest-confidence recommendation."
- `feature_factory` prompt: "Current preprocessing best practices: {preprocessing_hints}"
- `ensemble_architect` prompt: "Current ensemble strategies: {ensemble_hints}"

The technique_brief is a SUGGESTION in the prompt, not a hard override. The Wilcoxon gate still validates that the recommended approach beats the baseline.

**State additions:**
```python
technique_brief: dict = Field(default_factory=dict)  # owner: competition_intel
technique_brief_path: str = ""                         # owner: competition_intel
```

**Graceful degradation:** If web search is unavailable (no tool, rate limited, network error), `technique_brief` is empty dict. Pipeline continues with LLM's internal knowledge. This is v1 behavior — no regression.

### Contract tests: tests/contracts/test_technique_search_contract.py

Mock web search to return controlled results. Mock llm_call to return controlled JSON.

1. `test_technique_brief_produced` — after search, technique_brief is non-empty dict
2. `test_required_keys_present` — recommended_models, confidence, sources_consulted all exist
3. `test_recommended_models_have_name_and_library` — each entry has name + library fields
4. `test_graceful_degradation_no_search` — mock web_search returning empty → technique_brief is empty dict, no crash
5. `test_pipeline_runs_without_brief` — technique_brief is empty → all downstream agents run normally
6. `test_no_gpu_approaches` — technique_brief never contains pytorch or tensorflow model recommendations
7. `test_invalid_json_from_llm_handled` — mock llm_call returning invalid JSON → empty dict, no crash
8. `test_confidence_field_valid` — confidence is one of "high", "medium", "low"

---

## COMMIT 5: Problem Reframer (agents/problem_reframer.py)

### What this prevents

Solving the stated problem when a different formulation has a strictly higher score ceiling. Example: RMSLE competition where log-transforming the target and training on MSE produces strictly better results. v1 takes the competition at face value and never discovers this.

### The LangGraph node function

```python
def problem_reframer(state: ProfessorState) -> dict:
    """
    Evaluate alternative problem formulations.
    
    Reads: data_schema, competition_type, metric_name, metric_config,
           eda_insights_summary, eda_modality_flags, domain_brief,
           clean_data_path, target_column, validation_strategy
    Writes: reframings, active_reframing, reframing_tested
    Emits: STATUS (reframings found), STATUS (quick CV result)
    """
```

### Strategy Catalog — 5 strategies with precise activation conditions

**Strategy 1 — Target Transform:**
```python
def _check_target_transform(state):
    """
    Condition: target is continuous with skewness > 1.0 OR metric is RMSLE/MSLE
    """
    activate = False
    candidates = []
    
    # Check metric
    if state.metric_name in ["rmsle", "msle", "mean_squared_log_error"]:
        candidates.append({
            "strategy": "target_transform",
            "description": "Log-transform target for RMSLE optimization",
            "transform": "np.log1p(target)",
            "inverse": "np.expm1(prediction)",
            "rationale": "RMSLE is mathematically equivalent to RMSE on log-space. "
                         "Training on log-transformed target directly optimizes RMSLE.",
            "expected_impact": "high",
            "priority": 1,
        })
        activate = True
    
    # Check skewness from EDA
    target_stats = state.eda_report.get("target_distribution", {})
    skewness = target_stats.get("skewness", 0)
    if abs(skewness) > 1.0 and state.competition_type == "regression":
        transform = "np.log1p" if skewness > 0 else "np.sqrt"  # Right-skew vs left-skew
        candidates.append({
            "strategy": "target_transform",
            "description": f"Transform skewed target (skew={skewness:.2f})",
            "transform": f"{transform}(target)",
            "inverse": "np.expm1(prediction)" if "log" in transform else "prediction**2",
            "rationale": f"Target skewness {skewness:.2f} means residuals are heteroscedastic. "
                         f"Transforming stabilizes variance and improves regression models.",
            "expected_impact": "medium",
            "priority": 2,
        })
        activate = True
    
    return activate, candidates
```

**Strategy 2 — Problem Type Shift:**
```python
def _check_type_shift(state):
    """
    Condition: regression with discrete-ish target OR classification with ordinal targets
    """
    candidates = []
    
    # Regression with few unique target values → might be ordinal classification
    if state.competition_type == "regression":
        target_nunique = state.eda_report.get("target_distribution", {}).get("n_unique", 999)
        if target_nunique <= 20:
            candidates.append({
                "strategy": "type_shift",
                "description": f"Treat regression as classification ({target_nunique} unique values)",
                "rationale": "Target has only {target_nunique} unique values — this is effectively "
                             "an ordinal classification problem, not continuous regression.",
                "expected_impact": "medium",
                "priority": 3,
            })
    
    # Classification with ordinal targets → ordinal regression might help
    if state.competition_type == "multiclass":
        # Check if classes are ordered integers
        target_values = state.eda_report.get("target_distribution", {}).get("unique_values", [])
        if all(isinstance(v, (int, float)) for v in target_values):
            candidates.append({
                "strategy": "type_shift",
                "description": "Treat multiclass as ordinal regression",
                "rationale": "Target classes are ordered integers. Ordinal regression preserves "
                             "the ordering and penalizes far-off predictions more.",
                "expected_impact": "medium",
                "priority": 4,
            })
    
    return len(candidates) > 0, candidates
```

**Strategy 3 — Unit of Analysis Change:**
```python
def _check_unit_change(state):
    """
    Condition: group structure detected (user_id, session_id, patient_id)
    """
    # Check if validation_strategy uses GroupKFold (implies group structure)
    cv_type = state.validation_strategy.get("cv_type", "")
    group_col = state.validation_strategy.get("group_col", "")
    
    if "group" in cv_type.lower() and group_col:
        return True, [{
            "strategy": "unit_change",
            "description": f"Aggregate to {group_col}-level, then disaggregate",
            "rationale": f"Group structure on '{group_col}' means predictions at "
                         f"group level might be more stable. Train on group aggregates, "
                         f"then map back to row level.",
            "expected_impact": "low",
            "priority": 5,
        }]
    return False, []
```

**Strategy 4 — Loss Engineering:**
```python
def _check_loss_engineering(state):
    """
    Condition: metric has no direct sklearn scorer OR rewards specific behaviors
    """
    # Metrics where custom loss helps
    loss_opportunities = {
        "qwk": {
            "description": "Custom QWK-aligned loss for LightGBM",
            "rationale": "QWK penalizes far-off predictions quadratically. MSE/logloss don't "
                         "capture this. A custom loss aligned with QWK can improve 1-3%.",
            "expected_impact": "medium",
        },
        "map_at_k": {
            "description": "Custom ranking loss for MAP@K optimization",
            "rationale": "MAP@K rewards ranking quality, not calibration. A ranking-aware "
                         "loss (LambdaRank) directly optimizes what MAP@K measures.",
            "expected_impact": "high",
        },
    }
    
    if state.metric_name in loss_opportunities:
        opp = loss_opportunities[state.metric_name]
        return True, [{
            "strategy": "loss_engineering",
            "priority": 2,
            **opp,
        }]
    return False, []
```

**Strategy 5 — Multi-Model Segmentation:**
```python
def _check_segmentation(state):
    """
    Condition: EDA shows bimodal target OR distinct subpopulations
    """
    if "target" in (state.eda_modality_flags or []):
        return True, [{
            "strategy": "segmentation",
            "description": "Segment data by target mode, train separate models",
            "rationale": "Target is bimodal — likely two distinct subpopulations. "
                         "A single model averages across them. Two segment-specific "
                         "models can capture each population's patterns.",
            "expected_impact": "medium",
            "priority": 3,
        }]
    return False, []
```

### The main function flow

1. Run all 5 strategy checks. Collect all candidates.
2. Sort by priority (lower = higher priority).
3. Take top 3 candidates as `reframings`.
4. For the #1 candidate: run a QUICK CV test.

**Quick CV test:**
Generate code via `llm_call()` that implements the reframing and runs a 50-tree LightGBM with the same CV strategy. Execute via `run_in_sandbox()`. Compare against baseline score.

```python
# Quick CV prompt
quick_cv_prompt = f"""
Implement this problem reframing and test it:
Reframing: {top_reframing["description"]}
Transform: {top_reframing.get("transform", "N/A")}
Inverse: {top_reframing.get("inverse", "N/A")}

Load data from: {state.clean_data_path}
Target column: {state.target_column}
CV strategy: {state.validation_strategy}
Metric: {state.metric_name} (ORIGINAL metric, not transformed)

Train LightGBM(n_estimators=50, random_state=42) with and without the reframing.
Print BOTH scores as JSON: {{"baseline_cv": float, "reframed_cv": float}}

CRITICAL: Score BOTH on the ORIGINAL competition metric, not the transformed metric.
"""
```

5. Parse results. If `reframed_cv > baseline_cv`: set `active_reframing` to the winning reframing. Otherwise: `active_reframing` remains empty.

**Safety rule:** Quick CV scores on the ORIGINAL competition metric, never a transformed proxy. If competition evaluates RMSLE, quick CV reports RMSLE — not MSE on log-space.

### State return

```python
return state.validated_update("problem_reframer", {
    "reframings": sorted_candidates[:3],
    "active_reframing": winning_reframing or {},
    "reframing_tested": True,
})
```

Note: Check STATE.md — the field names might be `reframe_applied` and `reframe_details` instead. Use whatever STATE.md specifies. If there's a mismatch between this prompt and STATE.md, STATE.md wins.

### Contract tests: tests/contracts/test_problem_reframer_contract.py

1. `test_returns_list_of_reframings` — reframings is a list
2. `test_each_has_required_keys` — strategy, description, rationale, expected_impact, priority
3. `test_ranked_by_priority` — first item has lowest priority number
4. `test_no_reframings_acceptable` — standard binary classification with balanced classes → reframings may be empty (this is correct)
5. `test_skewed_regression_triggers_target_transform` — regression target with skew=3.5 → target_transform appears
6. `test_rmsle_triggers_log_transform` — metric="rmsle" → log1p transform appears with priority 1
7. `test_bimodal_target_triggers_segmentation` — "target" in eda_modality_flags → segmentation appears
8. `test_qwk_triggers_loss_engineering` — metric="qwk" → loss_engineering appears
9. `test_never_halts_pipeline` — pipeline continues regardless of whether reframings exist
10. `test_quick_cv_uses_original_metric` — mock sandbox, verify the quick CV code computes the ORIGINAL metric
11. `test_active_reframing_empty_when_rejected` — reframing scores worse than baseline → active_reframing is empty dict

---

## COMMIT 6: Feature Factory with Adaptive Gates (agents/feature_factory.py)

### What this builds

The core feature generation agent. Multi-round feature engineering with statistical gates (Wilcoxon + null importance) that determine which features survive. Includes Fix 1's adaptive gate thresholds.

This is the LARGEST single component. Read the architecture documents for Fix 1A (adaptive thresholds by dataset size), Fix 1B (feature parking KEPT/PARKED/REJECTED), and Fix 1C (gate calibration feedback loop) before implementing.

### The LangGraph node function

```python
def feature_factory(state: ProfessorState) -> dict:
    """
    Multi-round feature generation with statistical gates.
    
    Reads: clean_data_path, target_column, data_schema, validation_strategy,
           eda_insights_summary, eda_mutual_info, domain_brief, technique_brief,
           shifted_features, hitl_feature_hints, dynamic_rules_active,
           pipeline_depth (controls max rounds), data_usage_report
    Writes: feature_manifest, feature_factory_rounds_completed,
            features_train_path, features_test_path
    Emits: STATUS (each round start), RESULT (each round end with CV delta),
           CHECKPOINT (all rounds done — Milestone 2)
    """
```

### Per-round flow

For each round (max rounds from pipeline_depth: SPRINT=2, STANDARD=3, MARATHON=5):

**Step 1 — Build the feature generation prompt:**

Include in the prompt:
- Data schema (column names, types)
- `eda_insights_summary` — what the data looks like, high-MI pairs
- `domain_brief.feature_recipes` — domain-specific feature ideas
- `technique_brief.preprocessing_hints` — SOTA preprocessing
- `shifted_features` — features to AVOID building interactions on
- `hitl_feature_hints` — operator-suggested features (if any)
- `dynamic_rules_active` — learned rules from past competitions
- Existing features from previous rounds — "DO NOT recreate these"
- `data_usage_report.unused_files` — "Consider these supplementary files"
- Round-specific direction: Round 1 = basic transforms, Round 2 = interactions, Round 3 = domain features, Round 4-5 = creative

The prompt follows PROMPTS.md structure: Role → Context → Task → Constraints → Output format.

**Constraints section MUST include:**
```
1. Use Polars ONLY (import polars as pl), NEVER Pandas
2. Do NOT use .apply() or .map_elements() — use vectorized expressions
3. Do NOT access the target column in test data
4. Do NOT fit encoders on test data — train only
5. Pin random seeds to 42
6. All new column names must be unique
7. Handle null values explicitly
8. Output dataframe must have EXACTLY {canonical_train_rows} rows
```

**Step 2 — Generate code via `llm_call()`**

**Step 3 — Execute via `run_in_sandbox()`** with full metadata:
```python
result = run_in_sandbox(
    code=generated_code,
    agent_name="feature_factory",
    purpose=f"Round {round_num}: {feature_direction}",
    round_num=round_num,
    attempt=1,
    llm_prompt=prompt,
    llm_reasoning=response.get("reasoning", ""),
    expected_row_change="none",
)
```

**Step 4 — Apply statistical gates to each new feature:**

### Adaptive Gate Thresholds (Fix 1A)

Gate thresholds adapt based on dataset size:

```python
def _get_gate_thresholds(n_rows: int) -> dict:
    """
    Adaptive thresholds that account for statistical power.
    Small datasets: relaxed thresholds (low power, high noise)
    Large datasets: strict thresholds (high power, genuine signal)
    """
    if n_rows < 5000:
        return {
            "null_importance_percentile": 90,  # Relaxed from 95
            "wilcoxon_alpha": 0.10,             # Relaxed from 0.05
            "null_importance_shuffles": 30,     # Reduced from 50
        }
    elif n_rows < 50000:
        return {
            "null_importance_percentile": 95,   # Default
            "wilcoxon_alpha": 0.05,             # Default
            "null_importance_shuffles": 50,     # Default
        }
    else:
        return {
            "null_importance_percentile": 97,   # Strict
            "wilcoxon_alpha": 0.01,             # Strict
            "null_importance_shuffles": 50,     # Same
        }
```

### Null Importance Gate

For each new feature, test whether it carries genuine signal:

```python
def _null_importance_test(feature_values, target_values, model, n_shuffles, percentile):
    """
    Shuffle the feature N times, compute importance each time.
    If real importance > Nth percentile of shuffled importances: PASS.
    """
    # Real importance
    model.fit(feature_values.reshape(-1, 1), target_values)
    real_importance = model.feature_importances_[0]
    
    # Shuffled importances
    shuffled_importances = []
    for _ in range(n_shuffles):
        shuffled = np.random.permutation(feature_values)
        model.fit(shuffled.reshape(-1, 1), target_values)
        shuffled_importances.append(model.feature_importances_[0])
    
    threshold = np.percentile(shuffled_importances, percentile)
    passed = real_importance > threshold
    p_value = np.mean(np.array(shuffled_importances) >= real_importance)
    
    return passed, p_value, real_importance, threshold
```

### Wilcoxon Signed-Rank Gate

Test whether adding the feature significantly improves CV score:

```python
def _wilcoxon_gate(scores_with_feature, scores_without_feature, alpha):
    """
    Wilcoxon signed-rank test on per-fold CV scores.
    H0: adding the feature has no effect.
    Reject H0 (p < alpha) → feature helps.
    """
    from scipy.stats import wilcoxon
    stat, p_value = wilcoxon(scores_with_feature, scores_without_feature, alternative="greater")
    return p_value < alpha, p_value
```

### Feature Parking (Fix 1B)

Three-state system instead of binary keep/reject:

```python
def _classify_feature(null_imp_passed, wilcoxon_passed, null_imp_pvalue, wilcoxon_pvalue):
    """
    KEPT: Both gates pass → feature enters the model
    PARKED: One gate passes, one fails marginally → stored for later reconsideration
    REJECTED: Both gates fail or one fails decisively → discarded
    """
    if null_imp_passed and wilcoxon_passed:
        return "KEPT"
    
    # Parking conditions: one gate passed AND the other is marginal
    if null_imp_passed and not wilcoxon_passed and wilcoxon_pvalue < 0.15:
        return "PARKED"  # Null importance OK, Wilcoxon marginal
    if not null_imp_passed and wilcoxon_passed and null_imp_pvalue < 0.10:
        return "PARKED"  # Wilcoxon OK, null importance marginal
    
    return "REJECTED"
```

Parked features are stored in state and can be reconsidered after later rounds (when more features provide better context for the gates).

### Gate Calibration Feedback (Fix 1C)

Track gate pass rates across rounds:
```python
# After each round:
gate_stats = {
    "round": round_num,
    "features_tested": n_tested,
    "kept": n_kept,
    "parked": n_parked,
    "rejected": n_rejected,
    "pass_rate": n_kept / max(n_tested, 1),
}
```

If pass_rate is 0% for 2 consecutive rounds: gates are too strict. Log warning, consider relaxing thresholds for next round. If pass_rate is >80%: gates might be too loose. Log info. This feedback is for operator awareness via HITL, not automatic adjustment (automatic adjustment would be a second-order problem).

### Operator feature hints

Check `state.hitl_feature_hints` for pending suggestions:
```python
pending_hints = [h for h in (state.hitl_feature_hints or []) if h["status"] == "pending"]
for hint in pending_hints:
    # Include in the next round's prompt as priority
    # After testing: update hint status to "accepted" or "rejected"
    hint["status"] = "tried"
    # Feature goes through the SAME gates as generated features
    # If it fails gates: hint["status"] = "rejected", hint["rejection_reason"] = "..."
```

### Code Ledger integration

Every sandbox execution is automatically captured by the Code Ledger (built in Layer 0). When a feature is rejected by the gates:
```python
from tools.code_ledger import mark_rejected
mark_rejected(entry_id, reason=f"Null importance: p={p_value:.4f} > {threshold}")
```

### State return

```python
return state.validated_update("feature_factory", {
    "feature_manifest": all_kept_features,  # list of dicts with name, source, importance, round, status
    "feature_factory_rounds_completed": rounds_completed,
    "features_train_path": f"outputs/{state.session_id}/features_train.parquet",
    "features_test_path": f"outputs/{state.session_id}/features_test.parquet",
})
```

### Contract tests: tests/contracts/test_feature_factory_contract.py

1. `test_returns_feature_manifest_list` — feature_manifest is a list
2. `test_features_have_required_fields` — each feature has: name, source, importance, round, status
3. `test_status_is_valid` — status in ["KEPT", "PARKED", "REJECTED"]
4. `test_kept_features_pass_both_gates` — every KEPT feature has null_imp_passed=True AND wilcoxon_passed=True
5. `test_rejected_features_fail_gates` — every REJECTED feature fails at least one gate decisively
6. `test_parked_features_have_marginal_values` — PARKED features have one passing and one marginal gate
7. `test_adaptive_thresholds_small_data` — n_rows=3000 → wilcoxon_alpha=0.10 (relaxed)
8. `test_adaptive_thresholds_large_data` — n_rows=100000 → wilcoxon_alpha=0.01 (strict)
9. `test_operator_hints_go_through_gates` — operator-suggested feature that fails null importance → rejected, NOT kept
10. `test_rounds_match_pipeline_depth` — SPRINT → 2 rounds max, STANDARD → 3, MARATHON → 5
11. `test_shifted_features_avoided` — features in shifted_features list are NOT used as interaction bases
12. `test_code_ledger_entries_created` — every sandbox execution has a Code Ledger entry
13. `test_rejected_entries_marked_in_ledger` — rejected features have kept=False with rejection reason
14. `test_feature_train_parquet_produced` — features_train_path points to existing parquet file
15. `test_row_count_preserved` — output parquet has canonical_train_rows rows
16. `test_hitl_milestone_2_emitted` — after all rounds, CHECKPOINT emitted with feature summary

---

## COMMIT 7: Creative Hypothesis Engine (agents/creative_hypothesis.py)

### What this prevents

Feature ceiling. Feature Factory is algorithmic: groupby aggregations, target encoding, interactions. This produces a ceiling — typically top 20-30%. Gold-medal features are often non-obvious: domain-specific ratios, residuals, features derived from understanding WHY the data looks the way it does.

### Pipeline position

SEQUENTIAL after Feature Factory, BEFORE ML Optimizer. Creative needs the systematic feature manifest to know what's already covered. Running in parallel would duplicate work.

### The LangGraph node function

```python
def creative_hypothesis(state: ProfessorState) -> dict:
    """
    Domain-informed feature generation using 4 reasoning modes.
    
    Reads: feature_manifest, eda_report, eda_insights_summary, eda_mutual_info,
           domain_brief, technique_brief, clean_data_path, target_column,
           canonical_train_rows, validation_strategy, dynamic_rules_active
    Writes: creative_features_generated, creative_features_accepted
    Emits: STATUS (start), RESULT (features generated)
    """
```

### Four Reasoning Modes

Each mode generates feature hypotheses via `llm_call()` with different prompts:

**Mode 1 — Domain Expert Simulation:**
Prompt uses `domain_brief.known_relationships` and `domain_brief.column_semantics`:
```
You are a senior {domain_classification} practitioner.
Given these known domain relationships: {known_relationships}
And these column meanings: {column_semantics}

Generate 3-5 feature engineering ideas that a domain expert would create.
Each must have:
- name: feature name
- formula: exact Polars expression
- rationale: domain reasoning for why this helps
- source_columns: which columns are used

Focus on domain-specific ratios, thresholds, and interaction patterns 
that a pure ML approach would miss.
Existing features (DO NOT duplicate): {feature_manifest_names}
```

**Mode 2 — Past Winner Pattern Replay:**
Query ChromaDB for similar competition fingerprints (if available):
```python
if chromadb_available:
    similar_patterns = chromadb.query(
        competition_type=state.competition_type,
        domain=state.domain_classification,
        n_results=5,
    )
    # Extract feature engineering approaches from validated patterns
    # Adapt column names to current schema
```
If ChromaDB empty or unavailable: skip this mode.

**Mode 3 — Non-Obvious Interactions:**
For the top 20 features by importance (from `eda_quick_baseline_importance`):
```
Generate pairwise features for these top-20 columns:
- Ratios: A / (B + 1e-8)
- Differences: A - B
- Products: A * B
- Categorical crosses: concat(A, "_", B) for categorical pairs

Only generate interactions between columns with MI > 0.1.
Skip pairs that are already in the feature manifest.
```

**Mode 4 — Alternative Feature Spaces:**
Generate features from dimensionality reduction and clustering:
```
Create these alternative features:
- PCA top 5 components from numeric features
- K-Means cluster membership (k=5, k=10, k=20)
- Isolation Forest anomaly score
- Deviation from group mean for each categorical group
```

**Time budget per mode:** If any mode exceeds 5 minutes in sandbox execution, skip remaining computations in that mode. Return whatever was computed.

### Statistical Gates — SAME as Feature Factory

Every creative feature passes through the SAME gates: null importance + Wilcoxon with the same adaptive thresholds. Same KEPT/PARKED/REJECTED three-state classification.

```python
# Reuse Feature Factory's gate functions:
from agents.feature_factory import _null_importance_test, _wilcoxon_gate, _classify_feature, _get_gate_thresholds
```

Creative features that pass gates are tagged `source="creative"` in the manifest and merged into the feature set.

### Deduplication

Before computing any creative feature, check against `feature_manifest`:
```python
existing_names = {f["name"] for f in state.feature_manifest}
for hypothesis in generated_hypotheses:
    if hypothesis["name"] in existing_names:
        continue  # Skip — already exists from Feature Factory
```

### State return

```python
# Merge accepted creative features into the main manifest
updated_manifest = list(state.feature_manifest) + [
    {**f, "source": "creative"} for f in accepted_features
]

return state.validated_update("creative_hypothesis", {
    "creative_features_generated": all_hypotheses,  # All generated, including rejected
    "creative_features_accepted": accepted_features,  # Only KEPT features
})
```

Note: Creative Hypothesis writes to `creative_features_generated` and `creative_features_accepted` (its own fields). The updated `feature_manifest` that includes creative features needs to be handled at the graph level or by having Feature Factory own manifest updates. Check STATE.md for the correct ownership pattern.

### Contract tests: tests/contracts/test_creative_hypothesis_contract.py

1. `test_produces_hypothesis_list` — creative_features_generated is a list of dicts
2. `test_each_has_required_fields` — name, rationale, mode, tested, kept, null_importance_pvalue, wilcoxon_pvalue
3. `test_kept_features_pass_both_gates` — every kept=True feature has passing p-values
4. `test_no_gate_bypass` — no feature has kept=True with failing p-values (impossible by construction)
5. `test_features_tagged_source_creative` — accepted features have source="creative"
6. `test_no_duplicates_with_feature_factory` — no accepted feature name matches existing feature_manifest names
7. `test_domain_mode_uses_domain_brief` — mock domain_brief with known relationships → hypotheses reference those relationships
8. `test_empty_domain_doesnt_crash` — domain_classification="general" with empty domain_brief → mode 1 skips, no crash
9. `test_empty_chromadb_doesnt_crash` — ChromaDB unavailable → mode 2 skips, no crash
10. `test_time_budget_respected` — mock slow sandbox execution → mode completes within timeout, doesn't hang
11. `test_never_halts_pipeline` — if all hypotheses fail gates, returns empty lists, pipeline continues
12. `test_sprint_mode_skipped` — pipeline_depth="sprint" → creative_hypothesis returns empty immediately

---

## INTEGRATION NOTES

### How Layer 2 reads from Layer 1

| Layer 2 Component | Reads from Layer 1 |
|---|---|
| Leakage Precheck | (none — operates on raw code strings) |
| Data Usage Checker | raw_data_path (from data_engineer) |
| Retry Guidance | (none — operates on exceptions) |
| SOTA Search | competition_type, metric_name (from competition_intel) |
| Problem Reframer | eda_insights_summary, eda_modality_flags, domain_brief, metric_name, validation_strategy |
| Feature Factory | eda_insights_summary, eda_mutual_info, domain_brief, technique_brief, shifted_features, data_usage_report |
| Creative Hypothesis | feature_manifest (from feature_factory), domain_brief, eda_quick_baseline_importance |

### Dependency chain within Layer 2

```
Commits 1-3 (guards + retry): Independent of each other, integrate into sandbox/agents
Commit 4 (SOTA search): Independent, extends competition_intel
Commit 5 (Problem Reframer): Depends on EDA + domain + validation strategy from Layer 1
Commit 6 (Feature Factory): Depends on everything above — uses leakage check, data usage, retry guidance, SOTA hints
Commit 7 (Creative Hypothesis): Depends on Feature Factory (reads its manifest, reuses its gate functions)
```

Build in this order. Do not skip ahead.

---

## WHAT NOT TO DO

- Do NOT use Pandas anywhere. Polars only. Check POLARS.md.
- Do NOT skip statistical gates for ANY feature, including operator-suggested ones. Gates are NON-NEGOTIABLE.
- Do NOT auto-load unused data files. Only FLAG them.
- Do NOT let the SOTA technique brief override model selection. It's a SUGGESTION in the prompt.
- Do NOT modify Layer 0 or Layer 1 files except for the specific integration points documented above (sandbox.py leakage check, competition_intel SOTA extension).
- Do NOT run all 4 creative hypothesis modes if pipeline_depth is "sprint". Sprint skips creative_hypothesis entirely.
- Do NOT use RandomForest for null importance testing. Use LightGBM(n_estimators=50) — fast and accurate.
- Do NOT change the gate threshold constants from PROFESSOR.md unless using the adaptive function.
- Do NOT let any feature through without a Code Ledger entry. Every sandbox call passes provenance metadata.