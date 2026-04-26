# BUILD PROMPT — Layer 0 Foundation (Days 1-2)
# Feed this to Gemini CLI with: @PROFESSOR.md @STATE.md @SANDBOX.md @LANGGRAPH.md @CONTRACTS.md @HITL.md

---

## CONTEXT

You are building Professor, an autonomous Kaggle competition system on LangGraph. Before writing ANY code, read the loaded skill files completely — they contain exact patterns, naming conventions, state field definitions, and constraints you MUST follow.

This prompt covers the 3 foundational components that EVERY subsequent agent depends on. They share a single integration point: `run_in_sandbox()`. Build them together because they compose — HITL emits messages when debugging escalates, the Self-Debugging Engine writes to typed state fields, and State Schema Enforcement validates every write.

The project structure, all state fields with types and ownership, the sandbox function signature, and the LangGraph patterns are defined in the skill files. Do NOT invent your own patterns — follow the skills exactly.

---

## COMPONENT 1: State Schema Enforcement (graph/state.py)

### What to build

A Pydantic BaseModel called `ProfessorState` that replaces any loose dict-based state. Every field from STATE.md goes into this model with its exact type, default, and ownership annotation.

### Requirements

The model must enforce three things at runtime:

**Type validation:** Every field has a Pydantic type annotation. Writing `feature_manifest = "wrong"` when the type is `list` raises `ValidationError` immediately — not 3 agents downstream. Use `Field(default_factory=list)` for mutable defaults, never bare `[]`.

**Field ownership:** Build a class-level `_FIELD_OWNERS` dict mapping every field name to its owning agent string. Build a `validated_update(agent_name: str, updates: dict)` classmethod that checks every key in `updates` against `_FIELD_OWNERS`. If the agent doesn't own the field, raise `OwnershipError(f"Agent '{agent_name}' cannot write to '{field}' — owned by '{owner}'")`. This method is what every agent calls to write state — they never write fields directly.

**Write-once immutability:** Fields marked [IMMUTABLE] in STATE.md (canonical_train_rows, canonical_test_rows, canonical_schema, canonical_target_stats, test_data_checksum) can only be written when their current value equals the default (0, empty dict, empty string). Once set to a non-default value, any subsequent write raises `ImmutableFieldError`. Implement this check inside `validated_update()`.

### Additional infrastructure

Build a `_check_size()` method that serializes the state to JSON, checks byte count, and if it exceeds `STATE_SIZE_BUDGET_MB * 1024 * 1024` (20MB), truncates the oldest entries from: hitl_messages_sent (keep last 50), state_mutations_log (keep last 100), debug_diagnostics (clear entirely), debug_checkpoints (clear entirely), lineage_log (keep last 200). Core fields listed in STATE.md must NEVER be truncated. Call this after every `validated_update()`.

Build a `_log_mutation(agent_name, field, old_value, new_value)` method that appends to `state_mutations_log` with: agent name, field name, SHA-256 hash of old value (not the value itself — could be huge), SHA-256 hash of new value, and ISO timestamp. This creates the audit trail for debugging state corruption.

Build a `schema_version` field defaulting to "v2.0". Build a `validate_checkpoint_version(checkpoint_data: dict)` classmethod that checks the schema_version in loaded checkpoint data. If it matches: load normally. If it's older (e.g., "v1.0"): run a migration that adds new v2 fields with their defaults. If it's newer or unrecognized: raise `SchemaVersionError` with a clear message.

### Config flag

Add a module-level `OWNERSHIP_STRICT = True` that can be set to `False` during development to skip ownership checks. When False, ownership violations are logged as warnings but don't raise exceptions.

### File: graph/state.py

Use every field from STATE.md — all categories (core pipeline, competition intel, metric verification, pre-flight, data engineer, data integrity, EDA, domain, shift, validation, reframer, features, creative, model, critic, reflection, pseudo-labels, ensemble, post-processing, submission, submission safety, provenance, HITL, debugging, cost governor, memory, freeform, lineage). Do not skip any field. Do not rename any field. Do not change any type.

### Contract tests: tests/contracts/test_state_schema_contract.py

Write tests that verify:
1. Writing a list to a str field raises ValidationError — test with `eda_insights_summary`
2. Writing from wrong agent raises OwnershipError — test feature_factory writing to `critic_verdict`
3. canonical_train_rows can be written once (from 0 to 1000) but not overwritten (1000 to 2000)
4. State exceeding 20MB triggers truncation — create a state with a massive hitl_messages_sent list, call _check_size(), verify the list is truncated to 50
5. cv_mean survives truncation even when state is over budget
6. Every mutation is logged with correct agent attribution — write a field, check state_mutations_log has the entry
7. Schema version "v1.0" triggers migration that adds v2 fields with defaults
8. Schema version "v3.0" raises SchemaVersionError
9. OWNERSHIP_STRICT=False logs warning but doesn't raise

---

## COMPONENT 2: Self-Debugging Engine (tools/sandbox.py)

### What to build

The `run_in_sandbox()` function that wraps every code execution in Professor. It replaces v1's blind retry with a 4-layer informed retry cascade.

### The function signature

```
def run_in_sandbox(
    code: str,
    timeout: int = 300,
    extra_files: dict = None,
    working_dir: str = None,
    agent_name: str = "unknown",
    purpose: str = "",
    round_num: int = 0,
    attempt: int = 1,
    llm_prompt: str = "",
    llm_reasoning: str = "",
    expected_row_change: str = "none",
) -> dict
```

Returns: `{"success": bool, "stdout": str, "stderr": str, "runtime": float, "entry_id": str, "diagnostics": dict, "integrity_ok": bool}`

### Layer 1 — Diagnostic Injection

Build a `_inject_diagnostics(code: str) -> str` function that wraps the user's code in a try/except block. The wrapper must:

On exception, capture and print as a JSON blob with a `__DIAGNOSTICS__` prefix:
- Every local variable that has a `.shape` and `.columns` attribute (dataframes): capture shape, column list (first 50), dtypes dict, null counts per column (first 20 columns), and `str(df.head(3))`
- Every local scalar variable (int, float, str, bool) that doesn't start with `_`: capture name and repr (truncated to 200 chars)
- The exact exception type, message, and full traceback
- The failing line number extracted from `traceback.extract_tb()`
- Filesystem state: `os.listdir()` of the working directory with file sizes

On success (no exception), capture the same dataframe info AFTER execution completes, printed with the same `__DIAGNOSTICS__` prefix but with `"error": null`. This is for output validation.

The wrapper must NOT suppress inner try/except blocks in the user's code. It wraps at the OUTERMOST level only. The user's code is inserted between marker comments `# === USER CODE START ===` and `# === USER CODE END ===` so it's identifiable in tracebacks.

### Layer 2 — Error Classification

Build a `_classify_error(diagnostics: dict) -> tuple[str, str]` function that takes the diagnostic capture and returns `(error_class, fix_instruction)`.

Implement these 9 classes with their detection patterns:

- `column_missing`: traceback contains "KeyError" or "ColumnNotFoundError". Fix instruction includes the actual available columns from diagnostics and a fuzzy match (use `difflib.get_close_matches()` on the missing column name against available columns).
- `shape_mismatch`: traceback contains "shape", "length", "dimension", or "broadcast". Fix instruction includes before/after shapes from diagnostics.
- `type_error`: traceback contains "TypeError" or "InvalidOperationError" with "dtype". Fix instruction includes actual dtype from diagnostics.
- `null_values`: traceback contains "null", "NaN", "None" in context of not being allowed. Fix instruction includes null counts from diagnostics.
- `import_missing`: traceback contains "ModuleNotFoundError" or "ImportError". Fix instruction lists available alternatives from a hardcoded AVAILABLE_PACKAGES list (polars, numpy, scipy, sklearn, lightgbm, xgboost, catboost, optuna, matplotlib, seaborn).
- `memory_error`: traceback contains "MemoryError" or "Cannot allocate". Fix instruction includes memory usage from diagnostics.
- `api_mismatch`: traceback contains "unexpected keyword argument" or "has no attribute". Fix instruction includes common Polars API confusions (a hardcoded dict of `{wrong_call: correct_call}` for the 10 most common Polars mistakes).
- `timeout`: detected when subprocess times out (not from traceback). Fix instruction names the operation type and suggests optimizations.
- `unknown`: default when no pattern matches. Full diagnostics passed through.

For `logic_error`: this is detected by output validation (see below), not by traceback patterns. It's triggered when code succeeds but output validation fails.

Use regex matching on the traceback string. If multiple patterns match, pick the FIRST match in the order listed above. If confidence is low (pattern matches but the diagnostic context doesn't support it), fall back to `unknown`.

### Layer 3 — Checkpoint Recovery

Build a `_insert_checkpoints(code: str) -> str` function that analyzes the code string and inserts checkpoint saves after major operations. Detect these patterns:
- Assignment to a variable after `pl.read_csv()`, `pl.read_parquet()`, or `pl.scan_csv().collect()` → save checkpoint
- Assignment to a variable after `.join()`, `.group_by()`, `.with_columns()` where the expression spans 3+ lines → save checkpoint
- Assignment to a variable after any sklearn `.fit()` or `.fit_transform()` → save checkpoint

Checkpoints are saved as: `_checkpoint_{n}.parquet` for dataframes, `_checkpoint_{n}.pkl` for non-dataframe objects (models). Insert `import pickle` at the top if model checkpoints are used.

Build a `_retry_from_checkpoint(code: str, last_checkpoint: int, diagnostics: dict) -> str` function that generates a retry prompt instructing the LLM to load from checkpoint N and only rewrite the code after that point. The prompt includes: the checkpoint state (dataframe shape, columns from diagnostics), the error and diagnostics, and the instruction "Load from _checkpoint_{N}.parquet and rewrite ONLY the code after this point."

**Important:** Checkpoints are only saved on RETRY (attempt >= 2), not on the first execution. This avoids unnecessary disk I/O on successful runs.

### Layer 4 — Decomposition

Build a `_decompose_code(code: str) -> list[tuple[str, str]]` function that splits the code at blank-line boundaries between major blocks. Each block is: a label (e.g., "data_loading", "preprocessing", "feature_engineering", "model_training") and the code string.

Build a `_isolate_failing_block(blocks: list, working_dir: str) -> tuple[int, dict]` function that executes each block sequentially (in separate subprocess calls), passing the checkpoint state between them. Returns the index of the first failing block and its diagnostics.

Build a `_retry_single_block(failing_block: str, checkpoint_state: dict, diagnostics: dict) -> str` function that generates a retry prompt for ONLY the failing block, with the checkpoint state from the preceding successful block.

### Output Validation

Build a `_validate_output(diagnostics: dict, expected_row_change: str, canonical_rows: int) -> tuple[bool, str]` function that checks the successful execution's output:

For dataframe outputs in diagnostics:
- If `expected_row_change == "none"` and any dataframe has `shape[0] != canonical_rows` (when canonical_rows > 0): fail with "Row count changed from {canonical} to {actual} without declaration"
- If any new column has 1 unique value: fail with "Column '{name}' is constant — likely a bug"
- If any new column is 99%+ one value: warn (don't fail) with "Column '{name}' is near-constant"

For model/prediction outputs (detect by looking for arrays/Series named "predictions", "oof_predictions", "y_pred", "preds"):
- All values identical: fail with "All predictions are identical ({value}) — model failed to learn"
- Classification predictions outside [0, 1]: fail with "Predictions outside [0,1] range: min={min}, max={max}"
- Any NaN in predictions: fail with "Predictions contain {count} NaN values"

Return (True, "") for valid output, (False, failure_description) for invalid.

### The Retry Cascade

Build `_retry_cascade(code, timeout, working_dir, agent_name, purpose, round_num, llm_prompt, llm_reasoning, expected_row_change, canonical_rows) -> dict` that orchestrates the full flow:

```
Attempt 1: Execute code with diagnostic injection
  → success + valid output → return success
  → success + invalid output → mark as logic_error, proceed to attempt 2
  → failure → classify error, proceed to attempt 2

Attempt 2: Build retry prompt with error class + diagnostics + fix instruction
  → Call llm_call() to generate fixed code
  → Execute with diagnostic injection
  → success + valid → return success
  → failure → proceed to attempt 3

Attempt 3: Insert checkpoints, execute, load from last checkpoint
  → Call llm_call() with checkpoint context to generate partial fix
  → Execute from checkpoint
  → success + valid → return success
  → failure → proceed to attempt 4

Attempt 4: Decompose, isolate failing block, retry only that block
  → Call llm_call() with single-block context
  → Execute reconstructed code
  → success + valid → return success
  → failure → return failure (circuit breaker handles escalation)
```

Each attempt records its result in the Code Ledger (see below). The `debug_retry_layer` state field records which layer resolved the issue (0 = no error, 1 = first attempt, 2 = classified retry, 3 = checkpoint, 4 = decomposition).

Track fix rates: maintain a module-level dict `_FIX_RATES: dict[str, dict[str, int]]` mapping `{error_class: {"attempts": N, "fixed": M}}`. Update after every retry resolution.

### Code Ledger Integration

Build a `CodeLedgerEntry` dataclass matching the schema in the architecture doc (entry_id, timestamp, agent, purpose, round, attempt, code, code_hash, inputs, outputs, dependencies, success, stdout, stderr, runtime_seconds, llm_prompt, llm_reasoning, kept, rejection_reason).

Build `_append_to_ledger(entry: CodeLedgerEntry, session_dir: str)` that writes the entry as one JSON line to `{session_dir}/code_ledger.jsonl`.

Build `mark_rejected(entry_id: str, reason: str, session_dir: str)` that reads the ledger, finds the entry, sets `kept=False` and `rejection_reason=reason`, and rewrites the line.

Every execution attempt (including failed retries) gets a ledger entry. The `attempt` field distinguishes them (1, 2, 3, 4).

### Pre-execution Leakage Check

Build `_check_leakage(code: str) -> tuple[bool, str]` that scans the code string with regex for leakage patterns:
- `r'test.*target'` — accessing target in test data
- `r'\.fit\(.*test'` — fitting on test data  
- `r'\.fit_transform\(.*(?:concat|vstack)'` — fit_transform on combined train+test
- `r'LabelEncoder.*\.fit\(.*(?:concat|vstack)'` — encoding on combined data

Also build a safe pattern whitelist:
- `r'Pipeline\('` — sklearn Pipeline handles fit/transform correctly
- `r'ColumnTransformer\('` — same
- `r'\.fit\(X_train'` — explicit train-only fitting
- `r'\.transform\(X_test'` — transform-only on test

If a leakage pattern matches AND no safe pattern matches on the same line, return `(False, "Leakage detected: {pattern} at line {N}")`. The code is NOT executed — return immediately with a leakage-specific error.

### Putting it together in run_in_sandbox()

The main function orchestrates:
1. Pre-execution leakage check → if fails, return error without executing
2. Run the retry cascade
3. After successful execution: run output validation
4. Write Code Ledger entry for every attempt
5. Emit HITL messages on escalation (import from operator_channel)
6. Return the result dict

### Contract tests: tests/contracts/test_sandbox_contract.py

Write tests that verify:
1. Successful code returns `{"success": True}` with no diagnostic output in stdout
2. Code with KeyError on column triggers `column_missing` classification — verify fix instruction includes available columns
3. Code with TypeError triggers `type_error` classification
4. All-zero predictions trigger output validation failure with "All predictions are identical"
5. Predictions > 1.0 trigger output validation failure
6. Code ledger entry is written for every execution attempt
7. Leakage pattern `df_test['target']` triggers pre-execution rejection without running the code
8. Safe pattern `Pipeline(` does NOT trigger leakage rejection
9. Retry cascade proceeds in order: Layer 1 → 2 → 3 → 4 (mock llm_call to control retry outcomes)
10. `debug_retry_layer` correctly records which layer resolved the issue
11. Diagnostic injection adds < 300ms overhead on successful 1-second code execution
12. fix_rate tracking accumulates across multiple executions

---

## COMPONENT 3: HITL Operator Interface (tools/operator_channel.py)

### What to build

A dual-channel (CLI + Telegram) communication system that lets the operator see pipeline progress and inject guidance. Every agent calls `emit_to_operator()` — build this function and the infrastructure behind it.

### Channel Adapter Base

Build an abstract base class `ChannelAdapter` with:
- `send(message: str, level: str, data: dict = None) -> None` — send a message to the operator
- `poll_response(timeout: int) -> Optional[str]` — wait for operator response, return None on timeout
- `is_available() -> bool` — check if channel is functional

### CLI Adapter (CLIAdapter)

Build a CLI adapter that:
- `send()`: prints to stdout with ANSI color codes. Use colors: green for scores that improved, red for regression, bold for agent names, cyan for STATUS, yellow for CHECKPOINT, red for ESCALATION. Prefix each message with a timestamp `[HH:MM:SS]`.
- `poll_response()`: uses a non-blocking input approach. Run a background thread with `input()` that puts results into a `queue.Queue`. The `poll_response()` method does `queue.get(timeout=timeout)`. If timeout expires, return None.
- `is_available()`: return `sys.stdout.isatty()`. If stdout is piped (not a TTY), the adapter switches to write-only mode — `send()` still prints (for log capture) but `poll_response()` always returns None immediately.

### Telegram Adapter (TelegramAdapter)

Build a Telegram adapter that:
- `__init__(bot_token: str, chat_id: str)`: store credentials, verify connectivity with a test API call
- `send()`: POST to `https://api.telegram.org/bot{token}/sendMessage` with `chat_id` and `text`. For messages with `data` containing image paths, use `sendPhoto` instead. Handle HTTP errors gracefully — log warning, don't crash.
- `poll_response()`: GET from `https://api.telegram.org/bot{token}/getUpdates` with long polling (`timeout` parameter). Filter to messages from the correct `chat_id`. Track `update_offset` to avoid processing old messages.
- `is_available()`: try a `getMe` API call. Return True if it succeeds, False if it fails.

### Graceful Degradation

Both adapters must handle failure gracefully:
- CLI: if not a TTY, switch to write-only. Never crash.
- Telegram: if API is unreachable, retry once after 5 seconds. If second attempt fails, log warning, switch to LOCAL mode (write all messages to `hitl_log.jsonl` in the output directory). Check every 60 seconds if Telegram becomes reachable again. If it does, send a summary of missed messages and switch back.

The pipeline MUST NEVER crash because of HITL failure. If both channels are down, messages are silently logged and the pipeline continues autonomously.

### The emit_to_operator() Function

```
def emit_to_operator(
    message: str,
    level: str = "STATUS",
    data: dict = None,
) -> Optional[str]
```

This function:
1. Validates message length — truncate to 500 chars if longer (Telegram readability)
2. Adds sequence number and timestamp to the message
3. Dispatches to ALL active channel adapters simultaneously
4. For STATUS level: fire-and-forget, return None immediately
5. For RESULT level: fire-and-forget, return None
6. For CHECKPOINT level: send message, then poll ALL channels with `hitl_checkpoint_timeout` (default 180s). First response from ANY channel wins. Return the response text or None on timeout.
7. For GATE level: send message, then poll ALL channels with `hitl_gate_timeout` (default 900s). First response wins. Return response or None on timeout.
8. For ESCALATION level: send message, then poll ALL channels with no timeout (blocks until response). Always returns a response.
9. Log every sent message to `hitl_messages_sent` in state (append dict with timestamp, message, level, response)

### The Command Listener

Build a `CommandListener` class that runs as a background thread during pipeline execution. It continuously polls all active channels for incoming messages and classifies them:

- Starts with `/pause` → set `pipeline_paused = True` in a shared event/flag
- Starts with `/resume` → set `pipeline_paused = False`
- Starts with `/abort` → set `pipeline_aborted = True`
- Starts with `/status` → respond with current agent name + elapsed time + last score (read from shared state reference)
- Starts with `/scores` → respond with all CV scores from state
- Starts with `/feature ` → extract the text after `/feature `, create a dict `{"text": extracted, "status": "pending", "rejection_reason": ""}`, add to a shared injection queue
- Starts with `/override ` → parse the override command (format: `/override field_name value`), add to injection queue as override
- Starts with `/skip ` → extract agent name, add to injection queue as skip
- Starts with `/quiet` → set a flag that suppresses STATUS-level messages
- Starts with `/continue` → add a "continue" signal to the response queue (used for checkpoint acknowledgment)
- Any other text (no `/` prefix) → treat as domain knowledge injection, add to injection queue

Build a `process_pending_injections(state: ProfessorState) -> ProfessorState` function that checks the injection queue and applies pending items to state:
- Domain knowledge → append to `hitl_injections` with timestamp and empty `injected_into_agents`
- Feature hints → append to `hitl_feature_hints`
- Overrides → update `hitl_overrides` dict
- Skip commands → append to `hitl_skip_agents`

This function is called at every agent transition in the LangGraph graph (via a hook or a wrapper node).

### The Channel Manager

Build a `ChannelManager` class that:
- `__init__(channels: list[str], config: dict)`: instantiate the appropriate adapters based on the channels list (["cli"], ["telegram"], ["cli", "telegram"], [])
- Holds references to all active adapters
- `send_all(message, level, data)`: dispatch to every adapter
- `poll_any(timeout)`: poll all adapters simultaneously (using threading), return the first response
- `start_listener()`: start the CommandListener background thread
- `stop_listener()`: stop the background thread cleanly

### Launch integration

Build a module-level initialization function:
```
def init_hitl(channels: list[str], config: dict) -> ChannelManager:
```
That creates the manager, starts the listener, and returns the manager. The manager is stored as a module-level singleton so `emit_to_operator()` can access it without passing it through every function.

### Contract tests: tests/contracts/test_hitl_contract.py

Write tests that verify:
1. `emit_to_operator()` with STATUS level returns None immediately (doesn't block)
2. CHECKPOINT level waits for response up to timeout, returns None on timeout
3. CLI adapter prints messages to stdout with ANSI formatting (capture stdout in test)
4. Telegram adapter handles unreachable API gracefully — no crash, logs warning
5. Command `/pause` sets pipeline_paused flag
6. Command `/feature try log(income)` creates a feature hint with status "pending"
7. Free text "diagnosis is ICD-10" creates a domain injection in hitl_injections
8. `process_pending_injections()` applies queued injections to state correctly
9. Messages are logged to hitl_messages_sent after every emit
10. When all channels are unavailable, pipeline continues without crash
11. Message truncation: 600-char message is truncated to 500 chars
12. CLI adapter returns None from poll_response when stdout is not a TTY

---

## INTEGRATION BETWEEN THE 3 COMPONENTS

These three compose at specific points:

1. **State Schema is used by everything.** Both sandbox.py and operator_channel.py import ProfessorState. Every state write goes through `validated_update()`.

2. **Sandbox calls HITL on escalation.** When the retry cascade exhausts all 4 layers and circuit breaker activates, sandbox.py calls `emit_to_operator()` with level="ESCALATION" and includes the full diagnostic context. The operator sees exactly what failed, not just "code failed."

3. **Sandbox writes typed state.** After execution, sandbox.py updates debug_diagnostics, debug_error_class, debug_retry_layer, debug_fix_rate_by_class through `ProfessorState.validated_update("sandbox", {...})`.

4. **HITL writes typed state.** The CommandListener writes hitl_injections, hitl_overrides, hitl_feature_hints through `ProfessorState.validated_update("hitl_listener", {...})`.

5. **HITL reads state for /status.** The CommandListener reads current agent info and scores from the shared state reference to respond to /status and /scores commands.

---

## BUILD ORDER WITHIN THIS PROMPT

1. First: `graph/state.py` — ProfessorState with all fields, ownership, validation, immutability, size budget, mutations log. Then its contract tests.

2. Second: `tools/operator_channel.py` — ChannelAdapter base, CLIAdapter, TelegramAdapter, ChannelManager, CommandListener, emit_to_operator(), process_pending_injections(). Then its contract tests.

3. Third: `tools/sandbox.py` — diagnostic injection, error classification, checkpoint recovery, decomposition, output validation, retry cascade, Code Ledger, leakage check, run_in_sandbox(). Then its contract tests.

4. Fourth: Integration test that creates a ProfessorState, initializes HITL with CLI channel, runs a code block through run_in_sandbox() that intentionally fails, and verifies: state is updated through validated_update, HITL escalation message is emitted, Code Ledger entry is written, error is classified correctly.

---

## WHAT NOT TO DO

- Do NOT use Pandas anywhere. Polars only. Check POLARS.md.
- Do NOT create any agent files yet. This prompt builds ONLY the 3 foundation components.
- Do NOT hardcode API keys. Telegram bot token comes from environment variable `PROFESSOR_TELEGRAM_BOT_TOKEN` and chat_id from `PROFESSOR_TELEGRAM_CHAT_ID`.
- Do NOT make the HITL system blocking by default. STATUS and RESULT levels never block.
- Do NOT skip any state field from STATE.md. Every single field must be in ProfessorState.
- Do NOT write tests that test implementation details. Tests verify interfaces and behavioral invariants.
- Do NOT use `os.system()` or `exec()` for code execution. Use `subprocess.run()` with `capture_output=True`.
- Do NOT catch exceptions silently. Log them and either handle or re-raise.

Yes, commit discipline matters. If something breaks in Layer 2, you need to know whether it's your code or a Layer 0 regression. And with immutable contract tests, every commit should pass all existing tests.

**For the Layer 0 prompt alone, 5 commits:**

1. `graph/state.py` + its contract tests — ProfessorState model, validated_update, ownership, immutability
2. `tools/operator_channel.py` + its contract tests — adapters, emitter, listener, command parsing
3. `tools/sandbox.py` Layer 1-2 + tests — diagnostic injection, error classification, basic retry
4. `tools/sandbox.py` Layer 3-4 + tests — checkpoints, decomposition, output validation, full cascade
5. Integration test — all 3 components working together

**The rule:** every commit passes `pytest tests/contracts/ -q`. If it doesn't, fix before committing. Never commit broken tests "to fix later."

*

# BUILD PROMPT — Layer 0 Shields (Days 1-2)
# Feed to Gemini CLI with: @PROFESSOR.md @STATE.md @SANDBOX.md @HITL.md @POLARS.md @CONTRACTS.md

---

## CONTEXT

You are continuing the Professor v2 build. Layer 0 foundation (ProfessorState, HITL, Self-Debugging Engine) is already built and passing all contract tests. Now you're building 3 safety shields that protect the pipeline from catastrophic silent failures.

These shields are INDEPENDENT of each other — they don't depend on one another. But they all depend on Layer 0: they write state through `ProfessorState.validated_update()`, they emit messages through `emit_to_operator()`, and Shield 3 wraps the same LLM call path that the Self-Debugging Engine uses for retries.

Read the loaded skill files before writing any code. PROFESSOR.md has the project structure. STATE.md has every field with types and ownership. HITL.md has the emit_to_operator() patterns.

---

## COMMIT PLAN (3 commits, each must pass `pytest tests/contracts/ -q`)

```
Commit 1: shields/metric_gate.py + tests/contracts/test_metric_gate_contract.py
Commit 2: shields/cost_governor.py + tools/llm_provider.py + tests/contracts/test_cost_governor_contract.py
Commit 3: shields/preflight.py + tests/contracts/test_preflight_contract.py
```

Do NOT combine commits. Each commit is one logical unit. Run the full contract suite after each — if ANY existing test breaks, fix before committing.

---

## COMMIT 1: Shield 1 — Metric Verification Gate (shields/metric_gate.py)

### What this prevents

Professor classifies the metric as "AUC-ROC" when it's actually "weighted AUC" with class weights 1:3. Every agent optimizes the wrong objective. The entire 25-minute run is wasted. You don't know until you submit and the LB score doesn't match your CV. This is the #1 most common failure in autonomous ML systems.

### What to build

A module `shields/metric_gate.py` with one main function and a library of pre-built verification cases.

### The synthetic verification cases

Build a dict `METRIC_VERIFICATION_CASES` mapping metric names to pre-computed test cases. Each case is a dict with:
- `synthetic_true`: a numpy array of 100 known true values
- `synthetic_pred`: a numpy array of 100 known predictions
- `expected_score`: the EXACT score these produce (precomputed, hardcoded)
- `higher_is_better`: bool

Build cases for these 15 metrics. For each metric, build 3 sub-cases: balanced, imbalanced, and edge-case.

**Classification metrics:**
1. `roc_auc` — balanced: 50/50 binary, clear separation. imbalanced: 90/10. edge: all predictions = 0.5
2. `log_loss` — balanced: well-calibrated. imbalanced: 95/5. edge: predictions near 0 and 1
3. `f1` (binary, average="binary") — balanced: good precision/recall. imbalanced: rare positive class. edge: all predicted positive
4. `f1_macro` — balanced: 3 classes equal. imbalanced: 3 classes 80/15/5. edge: all predicted one class
5. `f1_micro` — same distributions as f1_macro but different expected scores
6. `accuracy` — balanced: clear. imbalanced: 95/5 where predicting all majority = 0.95. edge: random predictions
7. `mcc` (Matthews Correlation Coefficient) — balanced: good correlation. imbalanced: rare positive. edge: constant prediction
8. `qwk` (Quadratic Weighted Kappa) — balanced: 5 ordinal classes. imbalanced: heavy class 0. edge: off-by-one predictions

**Regression metrics:**
9. `rmse` — balanced: normal residuals. imbalanced: heavy-tailed residuals. edge: one extreme outlier
10. `rmsle` — balanced: positive predictions. imbalanced: near-zero values. edge: prediction = 0 (log undefined, must handle)
11. `mae` — balanced: normal. imbalanced: skewed target. edge: constant prediction = mean
12. `r2` — balanced: good fit. imbalanced: clustered target. edge: prediction = constant

**Ranking/other metrics:**
13. `spearman` — balanced: monotonic with noise. imbalanced: ties in ranking. edge: reverse ranking
14. `ndcg` (at k=10) — balanced: graded relevance. imbalanced: few relevant items. edge: no relevant items
15. `map_at_k` (MAP@5) — balanced: mixed relevant/irrelevant. imbalanced: very few relevant. edge: all relevant

For each metric, use `sklearn.metrics` or `scipy.stats` to compute the expected score EXACTLY. Hardcode the expected scores as floats in the dict. Do NOT compute them at runtime — the point is to verify that Professor's scorer matches the precomputed known-correct answer.

### The verification function

```python
def verify_metric(
    metric_name: str,
    scorer_func: callable,
    higher_is_better: bool,
    metric_config: dict = None,
) -> dict:
    """
    Verify that Professor's scorer matches the known-correct score 
    for this metric.
    
    Returns:
    {
        "verified": bool,
        "method": "auto_verified" | "no_case_available",
        "cases_passed": int,   # out of 3 (balanced, imbalanced, edge)
        "cases_failed": int,
        "details": [
            {
                "case": "balanced",
                "expected": 0.9523,
                "actual": 0.9523,
                "match": True,
            },
            ...
        ]
    }
    """
```

Steps:
1. Normalize the metric name: lowercase, strip whitespace, handle aliases (`"auc"` → `"roc_auc"`, `"rmse"` → `"rmse"`, `"root_mean_squared_error"` → `"rmse"`, `"quadratic_weighted_kappa"` → `"qwk"`, `"mean_absolute_error"` → `"mae"`)
2. Look up in `METRIC_VERIFICATION_CASES`. If not found, return `{"verified": False, "method": "no_case_available"}`
3. For each sub-case (balanced, imbalanced, edge): compute `scorer_func(synthetic_true, synthetic_pred)`. Compare to expected_score. Match if `abs(actual - expected) < 1e-4`.
4. If metric_config has `weights` or `average` parameters, check if any sub-case uses those params. If not, flag that the specific configuration isn't covered.
5. Return results. `verified = True` only if ALL 3 sub-cases match.

### The pipeline gate function

```python
def run_metric_verification_gate(state: ProfessorState) -> dict:
    """
    LangGraph node function. Runs after Competition Intel, before Validation Architect.
    
    HARD GATE: if verification fails, pipeline halts with HITL escalation.
    """
```

Steps:
1. Read `state.metric_name` and `state.metric_config` (set by Competition Intel)
2. Build the scorer function from metric_name (use sklearn.metrics or custom implementations)
3. Call `verify_metric()`
4. If verified:
   - `emit_to_operator(f"✅ Metric verified: {metric_name}", level="STATUS")`
   - Return `{"metric_verified": True, "metric_verification_test": results, "metric_verification_method": "auto_verified"}`
5. If not verified AND cases exist (mismatch):
   - `emit_to_operator()` with level="GATE": show the mismatch details, ask operator to TYPE the correct metric name
   - Wait for response. Parse response as metric name. Re-run verification with corrected metric.
   - If operator confirms: return with `metric_verification_method: "operator_confirmed"`
   - If timeout: return with `metric_verified: False` — Validation Architect will check this and halt
6. If not verified AND no cases available (custom metric):
   - `emit_to_operator()` with level="GATE": "Custom metric detected: {metric_name}. Professor cannot auto-verify. Please confirm the metric name and implementation."
   - Require operator to TYPE the metric name (not just /continue) to prevent blind confirmation
   - Return with `metric_verification_method: "operator_confirmed"`

### Edge case: operator types wrong metric

When the operator responds to the GATE, validate their response:
- If it matches a known metric name: re-run verification with that metric
- If it doesn't match any known metric: ask again "'{response}' is not a recognized metric. Available metrics: {list}. Please type the exact metric name."
- Max 3 attempts before accepting whatever they typed with method="operator_confirmed"

### Downstream enforcement

The Validation Architect (not built yet, but document the contract):
```python
# In validation_architect.py (future):
def validation_architect(state: ProfessorState) -> dict:
    if not state.metric_verified:
        emit_to_operator("🚨 METRIC NOT VERIFIED. Pipeline cannot proceed.", level="ESCALATION")
        raise PipelineHaltError("Metric verification required before training")
    # ... proceed with CV strategy design
```

This enforcement is a comment/docstring in the gate module for now. The actual enforcement will be built when Validation Architect is implemented.

### Contract tests: tests/contracts/test_metric_gate_contract.py

1. `test_roc_auc_passes_verification` — roc_auc scorer with sklearn's roc_auc_score passes all 3 cases
2. `test_wrong_scorer_fails_verification` — roc_auc test cases with log_loss scorer fails (expected ~0.95, actual is a loss value)
3. `test_rmse_passes_verification` — RMSE scorer passes all 3 regression cases
4. `test_qwk_passes_verification` — QWK scorer passes all 3 ordinal cases
5. `test_all_15_metrics_have_cases` — every metric in the list has an entry in METRIC_VERIFICATION_CASES
6. `test_each_metric_has_3_subcases` — every entry has balanced, imbalanced, and edge sub-cases
7. `test_metric_name_normalization` — "AUC", "auc", "roc_auc", "ROC_AUC" all resolve to the same cases
8. `test_unknown_metric_returns_no_case_available` — "custom_metric_xyz" returns method="no_case_available"
9. `test_gate_function_returns_correct_state_fields` — verify output dict has metric_verified, metric_verification_test, metric_verification_method
10. `test_gate_emits_status_on_success` — mock emit_to_operator, verify STATUS message on successful verification
11. `test_gate_emits_gate_on_failure` — mock emit_to_operator, verify GATE message on failed verification
12. `test_verified_flag_defaults_false` — fresh ProfessorState has metric_verified=False

### Implementation notes

- Use `sklearn.metrics` for standard metrics. Import at the top: `from sklearn.metrics import roc_auc_score, log_loss, f1_score, accuracy_score, matthews_corrcoef, mean_squared_error, mean_absolute_error, r2_score, ndcg_score`
- For QWK: use `from sklearn.metrics import cohen_kappa_score` with `weights="quadratic"`
- For RMSLE: implement as `np.sqrt(np.mean(np.square(np.log1p(true) - np.log1p(pred))))` — handle negative predictions by clipping to 0
- For Spearman: use `from scipy.stats import spearmanr`
- For MAP@K: implement manually — sklearn doesn't have it
- All synthetic arrays must be deterministic: use `np.random.RandomState(42)` for generation, then HARDCODE the expected scores

---

## COMMIT 2: Shield 3 — Cost Governor (shields/cost_governor.py + tools/llm_provider.py)

### What this prevents

A runaway retry loop or replan cycle makes 200+ LLM calls in 10 minutes. On paid API at $0.01/call, that's $50+ overnight across 3 competitions. The Self-Debugging Engine's Layer 4 decomposition can multiply one failure into 20+ calls.

### What to build — two files that work together

**shields/cost_governor.py** — the budget tracking and enforcement logic
**tools/llm_provider.py** — the LLM call wrapper that every agent uses

### tools/llm_provider.py

Build the `llm_call()` function that EVERY agent uses for LLM calls. No agent ever calls a provider API directly.

```python
def llm_call(
    prompt: str,
    agent_name: str = "unknown",
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    response_format: str = "text",  # "text" | "json"
) -> dict:
    """
    Returns:
    {
        "text": str,            # The response text
        "reasoning": str,       # Chain-of-thought if available
        "input_tokens": int,    # From API response
        "output_tokens": int,   # From API response  
        "model": str,           # Which model was used
        "cost_usd": float,      # Estimated cost
    }
    """
```

Steps inside llm_call():
1. Call `_cost_governor.check_budget(agent_name)` — if budget exhausted, raise `BudgetExhaustedError(f"LLM budget exhausted: {calls}/{max_calls} calls, ${cost:.2f}/${max_usd:.2f}")`
2. Call `_cost_governor.check_rate_limit()` — if rate limit hit, `time.sleep()` until the next minute window opens
3. Get provider config from `_load_provider_config()` — read YAML or fall back to environment variables
4. Call `_service_health.get_provider()` — get the best available provider
5. Make the API call using the provider's format (OpenAI-compatible for Groq/OpenAI, Google format for Gemini, Anthropic format for Claude)
6. Parse the response: extract text, reasoning (if available), token counts
7. Call `_cost_governor.record_call(agent_name, input_tokens, output_tokens, pricing)` — update budgets
8. Call `_service_health.record_success(provider)` on success, or `_service_health.record_failure(provider)` on failure with retry/rotation
9. Return the structured response

**Provider adapters:** Build 3 internal functions:
- `_call_openai_compatible(prompt, system_prompt, model, base_url, api_key, temperature, max_tokens)` — works for Groq, OpenAI, and any OpenAI-compatible API. Uses `requests.post()` to `{base_url}/chat/completions`.
- `_call_google(prompt, system_prompt, model, api_key, temperature, max_tokens)` — Google Gemini API format. Uses `google.generativeai` if available, otherwise raw HTTP.
- `_call_anthropic(prompt, system_prompt, model, api_key, temperature, max_tokens)` — Anthropic messages API format.

**Provider config loading:**
```python
def _load_provider_config() -> dict:
    """Load from config/provider_config.yaml or environment variables."""
```
Try YAML first. If not found, fall back to environment variables:
- `PROFESSOR_LLM_PROVIDER` — "groq" | "google" | "anthropic" | "openai"
- `GROQ_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- `PROFESSOR_LLM_MODEL` — override model name

Default if nothing is configured: provider="groq", model="deepseek-r1-distill-llama-70b"

**Service health:** Build a `ServiceHealth` class (can be simple for now):
- Track failures per provider as a list of timestamps
- `get_provider()`: if primary has 3+ failures in last 5 minutes, rotate to next in fallback_order
- `record_success(provider)`: clear failure list for that provider
- `record_failure(provider)`: append current timestamp
- Fallback order: configurable, default `["google", "groq", "anthropic"]`

**Retry logic on API errors:**
- Rate limit (429): exponential backoff — sleep 1s, 2s, 4s. Max 3 retries.
- Server error (500, 502, 503): retry once, then rotate to fallback provider
- Auth error (401, 403): don't retry, raise immediately with clear message about API key
- Timeout: retry once with doubled timeout, then rotate

### shields/cost_governor.py

Build a `CostGovernor` class:

```python
class CostGovernor:
    def __init__(self, max_calls: int = 150, max_usd: float = 5.0, max_per_minute: int = 20):
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.max_per_minute = max_per_minute
        self.call_count = 0
        self.cost_usd = 0.0
        self.calls_per_agent = {}
        self.call_timestamps = []  # For per-minute tracking
        self.exhausted = False
        self._warned_80 = False
```

**check_budget(agent_name) -> bool:**
- If `call_count >= max_calls` or `cost_usd >= max_usd`: set `exhausted = True`, return False
- If `call_count >= max_calls * 0.8` or `cost_usd >= max_usd * 0.8` and not `_warned_80`:
  - Call `emit_to_operator(f"⚠️ Budget at {pct}%: {call_count}/{max_calls} calls, ${cost_usd:.2f}/${max_usd:.2f}", level="STATUS")`
  - Set `_warned_80 = True`
- Return True

**check_rate_limit() -> None:**
- Clean `call_timestamps` to only last 60 seconds
- If `len(call_timestamps) >= max_per_minute`: calculate sleep time until oldest timestamp is > 60s ago. `time.sleep(sleep_time)`.
- Append current timestamp to `call_timestamps`

**record_call(agent_name, input_tokens, output_tokens, pricing) -> None:**
- `self.call_count += 1`
- `cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000`
- Apply 1.2x safety multiplier: `cost *= 1.2`
- `self.cost_usd += cost`
- `self.calls_per_agent[agent_name] = self.calls_per_agent.get(agent_name, 0) + 1`

**get_summary() -> dict:**
- Return `{"total_calls": ..., "total_cost_usd": ..., "per_agent": ..., "budget_remaining_calls": ..., "budget_remaining_usd": ...}`

**Replan loop detection** (called by the Supervisor, not by cost_governor directly):
Build a standalone function:
```python
def detect_replan_loop(current_finding: dict, previous_finding: dict) -> bool:
    """Returns True if same Critic vector + severity appeared on consecutive replans."""
    if not previous_finding:
        return False
    return (current_finding.get("vector") == previous_finding.get("vector") 
            and current_finding.get("severity") == previous_finding.get("severity"))
```

**Module-level singleton:**
```python
_governor: CostGovernor = None

def init_cost_governor(max_calls=150, max_usd=5.0, max_per_minute=20):
    global _governor
    _governor = CostGovernor(max_calls, max_usd, max_per_minute)

def get_governor() -> CostGovernor:
    global _governor
    if _governor is None:
        init_cost_governor()  # Default values
    return _governor
```

### State synchronization

At every agent transition (in the LangGraph graph hooks), sync the governor's state to ProfessorState:
```python
gov = get_governor()
state.validated_update("cost_governor", {
    "llm_call_count": gov.call_count,
    "llm_cost_estimate_usd": gov.cost_usd,
    "llm_calls_per_agent": gov.calls_per_agent,
    "llm_budget_exhausted": gov.exhausted,
})
```

This sync is NOT inside llm_call() (too frequent). It's at agent boundaries.

### What happens when budget is exhausted

The pipeline does NOT crash. `BudgetExhaustedError` is caught by the agent wrapper (in the LangGraph node). The agent skips its LLM-dependent work and returns whatever partial state it has. The pipeline proceeds to ensemble and submission with existing artifacts. This means:
- If budget exhausts during Feature Factory Round 3: Rounds 1-2 features are kept. No Round 3 features.
- If budget exhausts during ML Optimizer: the last successful model config is used.
- If budget exhausts during Critic: Critic is skipped (risk accepted).

Document this behavior clearly in docstrings.

### Contract tests: tests/contracts/test_cost_governor_contract.py

1. `test_budget_blocks_at_limit` — set max_calls=5, make 5 calls, verify 6th returns False from check_budget
2. `test_budget_exhausted_flag_set` — after budget blocks, governor.exhausted is True
3. `test_80_percent_warning` — mock emit_to_operator, set max_calls=10, make 8 calls, verify warning emitted
4. `test_warning_only_once` — make calls 8, 9, 10 — warning emitted at 8, NOT repeated at 9 and 10
5. `test_dollar_budget_tracks` — record_call with known token counts and pricing, verify cost_usd matches expected
6. `test_safety_multiplier_applied` — verify cost is 1.2x the raw calculation
7. `test_per_minute_rate_cap` — mock time, add 20 timestamps in same minute, verify check_rate_limit sleeps
8. `test_per_agent_tracking` — make calls from 3 different agents, verify calls_per_agent has correct counts
9. `test_summary_has_all_fields` — get_summary returns total_calls, total_cost_usd, per_agent, remaining
10. `test_replan_loop_detected` — same vector+severity on 2 findings returns True
11. `test_different_findings_not_loop` — different vector returns False
12. `test_llm_call_raises_on_exhausted` — with exhausted governor, llm_call raises BudgetExhaustedError

### Contract tests: tests/contracts/test_llm_provider_contract.py

1. `test_llm_call_returns_structured_response` — mock HTTP, verify response has text, input_tokens, output_tokens, model, cost_usd
2. `test_llm_call_increments_governor` — mock HTTP, make 3 calls, verify governor.call_count == 3
3. `test_rate_limit_retry` — mock HTTP to return 429 then 200, verify retry succeeded
4. `test_auth_error_raises_immediately` — mock 401, verify no retry, clear error message about API key
5. `test_provider_rotation_on_failure` — mock primary failing 3 times, verify fallback provider used
6. `test_env_variable_fallback` — no YAML config, set env vars, verify provider loaded correctly
7. `test_default_provider_is_groq` — no config at all, verify defaults to groq with deepseek model

---

## COMMIT 3: Shield 6 — Pre-Flight Checks (shields/preflight.py)

### What this prevents

Professor enters a competition with a 12GB dataset → OOM. Or text columns needing NLP → garbage features. Or JSON submission format → invalid submission. 20-30 minutes wasted before anyone knows the competition is incompatible.

### What to build

A LangGraph node function `run_preflight_checks()` that profiles the data BEFORE the main pipeline starts, using minimal resources (100-row sample, file size checks, no full loading).

### The preflight function

```python
def run_preflight_checks(state: ProfessorState) -> dict:
    """
    LangGraph node. Runs FIRST, before Competition Intel.
    Profiles data files without full loading.
    Emits Milestone 0 to operator with findings.
    """
```

### Step 1: File inventory

```python
def _inventory_data_files(data_dir: str) -> list[dict]:
    """
    List all files in the competition data directory.
    
    Returns list of:
    {
        "name": "train.csv",
        "size_mb": 234.5,
        "format": "csv",  # csv, json, parquet, npy, jpg, png, txt, other
        "will_use": True,  # True for train.csv, test.csv, sample_submission.*
    }
    """
```

For each file in `os.listdir(data_dir)`:
- Get size via `os.path.getsize()`, convert to MB
- Detect format from extension: .csv, .tsv → "csv", .json → "json", .parquet → "parquet", .npy → "npy", .jpg/.jpeg/.png/.bmp → "image", .wav/.mp3 → "audio", .txt → "text", else → "other"
- Mark `will_use=True` for files matching: train.csv, test.csv, sample_submission.csv (case-insensitive, also .parquet and .tsv variants)
- Everything else: `will_use=False`

Check total size. If total > available RAM * 0.7 (use `psutil.virtual_memory().total` if available, else assume 16GB): emit warning.

If any single file > 2GB: flag `"large_file_warning"` with recommendation to use chunked loading or `pl.scan_csv()`.

### Step 2: Column type profiling

```python
def _profile_columns(file_path: str, n_rows: int = 100) -> list[dict]:
    """
    Load first n_rows of a CSV/parquet and profile each column.
    
    Returns list of:
    {
        "name": "diagnosis",
        "dtype": "Utf8",
        "n_unique_in_sample": 45,
        "null_pct_in_sample": 5.2,
        "avg_str_length": 67.3,   # For string columns only
        "flags": ["possible_nlp"],  # List of warning flags
    }
    """
```

Load with `pl.read_csv(file_path, n_rows=n_rows)` (or `pl.read_parquet` for parquet).

For each column, detect and flag:
- **possible_nlp**: string column with average length > 50 chars. Pattern: `df[col].str.len_chars().mean() > 50`
- **image_paths**: string column where >50% of non-null values match `r'\.(jpg|jpeg|png|bmp|gif|tiff)$'` (case insensitive)
- **audio_paths**: same for `r'\.(wav|mp3|flac|ogg)$'`
- **file_paths**: same for `r'[/\\]'` (contains path separators)
- **nested_json**: string column where >50% of values start with `{` or `[`
- **pipe_delimited**: string column where >50% of values contain `|`
- **high_cardinality**: >90 unique values in 100 rows AND dtype is string/categorical
- **datetime_candidate**: string column matching ISO date pattern `r'^\d{4}-\d{2}-\d{2}'` in >50% of values
- **constant**: 1 unique value in sample (useless column)
- **mostly_null**: >80% null in sample

Also profile the LAST 100 rows if file size < 1GB (tail sampling to catch patterns that don't appear at the top):
```python
if file_size_mb < 1000:
    tail_df = pl.read_csv(file_path).tail(100)  # This loads full file — only for < 1GB
    # Merge flags from tail profiling
```

For files > 1GB: skip tail sampling, add advisory warning "Large file — tail sampling skipped, column profiling based on first 100 rows only."

### Step 3: Submission format verification

```python
def _verify_submission_format(data_dir: str) -> dict:
    """
    Parse sample_submission file and verify Professor can produce it.
    
    Returns:
    {
        "format": "csv",  # csv | json | other
        "columns": ["id", "target"],
        "n_rows": 11308,
        "value_types": {"id": "int", "target": "float"},  # inferred from sample
        "compatible": True,  # Can Professor produce this format?
        "issues": [],  # List of compatibility issues
    }
    """
```

Find sample_submission file: `sample_submission.csv`, `sampleSubmission.csv`, `sample_submission.json`, etc. (case-insensitive glob).

If not found: flag "No sample submission found — cannot verify output format."

If found and CSV: parse it. Check column names, dtypes, row count.

If found and JSON: flag `"compatible": False, "issues": ["JSON submission format not supported by Professor v2"]`

If value_types suggest multi-label (columns with list values) or multi-column prediction: flag for operator review.

### Step 4: Target type verification

```python
def _detect_target_type(df_sample: pl.DataFrame, target_col: str) -> str:
    """
    Detect target column type from sample.
    
    Returns: "binary" | "multiclass" | "regression" | "multilabel" | "ordinal" | "unknown"
    """
```

Logic:
- If target dtype is numeric AND n_unique == 2: "binary"
- If target dtype is numeric AND 3 <= n_unique <= 30 AND all values are integers: could be "multiclass" or "ordinal" — flag both
- If target dtype is numeric AND n_unique > 30: "regression"
- If target dtype is string AND n_unique <= 30: "multiclass"
- If target contains lists or pipe-delimited values: "multilabel"
- Else: "unknown"

### Step 5: Capability boundary check

```python
def _check_capability_boundaries(file_inventory: list, column_profiles: list, submission_format: dict) -> list[str]:
    """
    Check if Professor v2 can handle this competition.
    
    Returns list of unsupported modalities: ["nlp", "image", "audio", "large_dataset", "json_submission"]
    """
```

- If any column has "possible_nlp" flag: add "nlp"
- If any column has "image_paths" or "audio_paths": add "image" or "audio"
- If any file is > 10GB: add "large_dataset"
- If submission format is not CSV: add the format name
- If target type is "multilabel": add "multilabel"

### Step 6: Assemble and emit

Back in `run_preflight_checks()`:

1. Run all 5 steps
2. Categorize warnings as BLOCKING vs ADVISORY:
   - BLOCKING (pipeline halts): unsupported submission format, unsupported modalities that are REQUIRED (image classification where all features are image paths)
   - ADVISORY (pipeline continues, operator informed): supplementary files not used, high cardinality columns, possible NLP columns alongside tabular columns (might still work with tabular features only), large file warnings
3. Emit Milestone 0 to operator:
```
🚀 PRE-FLIGHT REPORT
Files: 5 found (train.csv 234MB, test.csv 58MB, metadata.csv 1.2MB, ...)
Using: train.csv, test.csv
Not using: metadata.csv, external_data.parquet, images/ [Override?]

Columns: 31 total (23 numeric, 6 string, 2 datetime candidates)
Flags: 
  ⚠️ 'description' column avg length 120 chars — possible NLP
  ⚠️ 'category_id' has 95 unique values in 100 rows — high cardinality
  
Target: 'target' — binary (2 unique values)
Submission: CSV, 2 columns [id, target], 11308 rows

Capability: All checks passed ✅
```
4. If BLOCKING issues exist: emit as GATE, require operator acknowledgment
5. If only ADVISORY: emit as CHECKPOINT
6. Return state updates with all preflight_ fields

### Contract tests: tests/contracts/test_preflight_contract.py

Set up test fixtures with temporary directories containing mock CSV files:

1. `test_clean_tabular_passes` — create a simple 1000-row CSV with numeric/categorical columns. Verify preflight_passed=True, zero blocking warnings
2. `test_large_file_detected` — create a mock scenario where file size > 2GB (mock os.path.getsize). Verify large_file_warning in preflight_warnings
3. `test_text_column_flagged` — create CSV where one column has strings averaging 100 chars. Verify "possible_nlp" flag on that column
4. `test_image_paths_flagged` — create CSV where one column has values like "img/001.jpg". Verify "image_paths" flag
5. `test_submission_format_csv_passes` — create sample_submission.csv with 2 columns. Verify compatible=True
6. `test_submission_format_json_blocks` — create sample_submission.json. Verify compatible=False, BLOCKING issue
7. `test_supplementary_files_listed` — create directory with train.csv, test.csv, metadata.csv, extra.parquet. Verify metadata.csv and extra.parquet listed with will_use=False
8. `test_binary_target_detected` — target column with 2 unique values → "binary"
9. `test_regression_target_detected` — target column with 500 unique float values → "regression"
10. `test_multiclass_target_detected` — target column with 5 unique integer values → "multiclass"
11. `test_profiling_uses_only_100_rows` — create 10000-row file, mock pl.read_csv, verify n_rows=100 passed
12. `test_high_cardinality_flagged` — string column with 95 unique values in 100 rows flagged
13. `test_constant_column_flagged` — column with 1 unique value flagged as "constant"
14. `test_milestone_0_emitted` — mock emit_to_operator, verify CHECKPOINT or GATE emitted with structured data
15. `test_capability_boundary_nlp` — NLP column detected → "nlp" in unsupported_modalities
16. `test_no_sample_submission_warns` — missing sample_submission file → advisory warning, not blocking

---

## INTEGRATION NOTES

- All three shields import from `graph.state` (ProfessorState), `tools.operator_channel` (emit_to_operator), and use `validated_update()` for state writes.
- Shield 3 (Cost Governor) is also imported by `tools/sandbox.py` — the Self-Debugging Engine's retry cascade calls `llm_call()` for generating fix attempts, so retries are counted against the budget.
- Shield 1 and Shield 6 are LangGraph nodes. Shield 3 is a wrapper layer, not a node.
- After all 3 commits, run: `pytest tests/contracts/ -v` to verify all Layer 0 + Shield tests pass together.

---

## WHAT NOT TO DO

- Do NOT build any pipeline agents (Feature Factory, ML Optimizer, etc.) — only the 3 shields.
- Do NOT use Pandas for column profiling. Use Polars. Check POLARS.md.
- Do NOT make API calls in tests. Mock all HTTP calls.
- Do NOT hardcode API keys. Use environment variables or config files.
- Do NOT let any shield crash the pipeline. If a shield itself fails (bug in the shield code), log the error and let the pipeline continue with a warning. Shields are safety nets, not barriers.