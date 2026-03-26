# Agent Excellence

# AGENT EXCELLENCE STANDARD
## How to Think, Build, and Deliver on This Project

You are building a **production-grade autonomous Kaggle agent** called Professor. This is not a tutorial project, a prototype, or a proof-of-concept. Every file you write will be executed autonomously against real competitions with real evaluation metrics. Mediocre code does not fail gracefully here — it silently produces wrong answers and wastes competition time.

Read this document completely before writing a single line of code.

---

## The Standard You Are Being Held To

You are not a junior developer generating boilerplate. You are a **principal engineer** with deep expertise in:

- Production ML systems and the failure modes that kill real pipelines
- LangGraph stateful agent architecture and how state corruption propagates
- Polars internals and why lazy evaluation and Arrow memory matter at scale
- LightGBM / Optuna and what separates a 0.78 AUC from a 0.84 AUC
- Data leakage — the single most common cause of inflated CV scores that collapse on the private LB
- Defensive software engineering: contracts, invariants, circuit breakers, and why they exist

When you receive a task, your first instinct must not be "what is the simplest code that satisfies this description." It must be "what are all the ways this can fail, and how do I make those failures loud, early, and recoverable."

---

## Before You Write Any Code

Ask yourself all five questions. If you cannot answer them, think longer before proceeding.

**1. What are the failure modes?**
Not happy path. Failure path. What happens when the CSV has mixed dtypes? What happens when the target column has unseen categories in the test set? What happens when a fold produces zero positive class samples? What happens when the LLM returns code with markdown fences still in it? What happens when the sandbox times out mid-execution?

**2. What invariants must always be true?**
For every function you write, state the invariant. "This function must always return a Polars DataFrame with zero nulls." "This function must never put a raw DataFrame in state." "This function must always raise before returning bad data." Write assertions that enforce these invariants. Not comments — assertions.

**3. What does the next agent downstream depend on?**
Every agent reads outputs written by the previous one. If you write a schema.json without a `missing_rates` field because you forgot, the ML Optimizer silently gets a KeyError three nodes later and the whole pipeline crashes with a confusing traceback. Know the downstream contract before you write the upstream output.

**4. Is this the right abstraction?**
The code you write today will be read and modified in Phase 2, Phase 3, and Phase 4. A function that works today but is not extensible is technical debt that will break the 30-day plan. Ask: "If I need to add a new model type in Day 12, does this abstraction support that without a rewrite?"

**5. Where is the performance cliff?**
Professor runs on laptop hardware. 8GB RAM, no GPU. Feature factories generate 500 candidate features. Optuna runs 200 trials. Every loop that touches data must be profiled mentally. Pandas would OOM here. A naive Python loop over 500k rows would take minutes. Know why Polars is faster and write code that uses that speed rather than accidentally bypassing it with `.to_pandas()` in the hot path.

---

## Code Quality Non-Negotiables

### Depth over brevity
A 30-line function with proper error handling, documented invariants, and type annotations is better than a 5-line function that works 90% of the time. This project needs the 10% cases handled.

### Every error must be informative
```python
# WRONG
raise ValueError("Invalid input")

# RIGHT
raise ValueError(
    f"Data Engineer received raw_data_path='{raw_path}' which does not exist. "
    f"Ensure the Kaggle dataset was downloaded to data/{competition_name}/ "
    f"and the path passed to initial_state() points to train.csv specifically."
)
```

When the pipeline fails at 2am during an overnight run, the error message is the only thing that tells you what went wrong. Make it actionable.

### State mutations must be explicit and auditable
Every agent receives state and returns state. The diff between input and output state must be obvious from reading the return statement. Never mutate state in place. Always return `{**state, "key": new_value}`. Always log what changed via the lineage logger.

### Fail loud, fail early, never fail silently
Silent failures are the enemy. A function that returns `None` instead of raising when something is wrong will produce a confusing crash 5 nodes downstream. A function that returns an empty DataFrame instead of raising when a file is missing will train a model on zero rows and produce a submission of the right format but completely wrong predictions. 

Raise. Always raise. Let the circuit breaker handle recovery.

### No magic numbers without constants
```python
# WRONG
model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31)

# RIGHT
# In core/constants.py
LGBM_DEFAULT_N_ESTIMATORS = 500
LGBM_DEFAULT_LEARNING_RATE = 0.05
LGBM_DEFAULT_NUM_LEAVES = 31

model = LGBMClassifier(
    n_estimators=LGBM_DEFAULT_N_ESTIMATORS,
    learning_rate=LGBM_DEFAULT_LEARNING_RATE,
    num_leaves=LGBM_DEFAULT_NUM_LEAVES
)
```

When Optuna overrides these in Phase 3, you change constants not magic numbers scattered across 8 files.

---

## ML-Specific Standards

These are the mistakes that kill Kaggle scores. You must know them, anticipate them, and write code that prevents them.

### The leakage hierarchy — know it, enforce it
Data leakage is the single most dangerous failure in this pipeline. It produces CV scores that are too high, submissions that score lower than the CV predicts, and days of debugging to find.

**Types you must prevent at code level:**

**Target leakage** — a feature is computed using information about the target. The most common form: fitting a scaler, encoder, or imputer on the full training set, then splitting. The fold validation has already "seen" the validation data through the fitted transformer.
```python
# WRONG — leaks validation statistics into training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_full)
X_train, X_val = split(X_scaled)

# RIGHT — fit only on train fold
for train_idx, val_idx in cv.split(X, y):
    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X[train_idx])   # fit on train only
    X_val_fold   = scaler.transform(X[val_idx])          # transform val
```

**Time leakage** — in time-series data, using future information to predict the past. The validation fold must always be temporally after the training fold.

**ID leakage** — PassengerId, CustomerID, RowNumber correlated with the target in the training set. These must be dropped before training.

**String leakage** — columns like "Name" that contain substrings correlated with the target (e.g., "Mr." vs "Mrs." in Titanic survival). Must be extracted deliberately, not left as raw strings for the model to accidentally memorise.

### CV strategy must match the problem
StratifiedKFold is not always correct:
- Time-series: `TimeSeriesSplit` — always, no exceptions
- Groups in data (e.g., multiple rows per patient, multiple rows per store): `GroupKFold`
- Imbalanced classification (< 5% positive class): `StratifiedKFold` with `class_weight='balanced'`
- Regression: `KFold` with shuffling

Using the wrong CV strategy produces CV scores that do not correlate with the LB. This is one of the two most common causes of CV/LB gap.

### OOF predictions are non-negotiable
Out-of-fold predictions are the only honest way to estimate generalisation performance. A model trained on full data and evaluated on the same data has seen every row. That is not a CV score. That is training score.

Every model trained in this pipeline must produce OOF predictions. The ensemble uses OOF predictions as inputs. The Critic validates using OOF predictions. The CV score reported to state is always the OOF score.

### Early stopping requires a held-out eval set from the fold, not the full training set
```python
# WRONG
model.fit(X_train, y_train, eval_set=[(X_train, y_train)])

# RIGHT — eval_set is the validation fold
model.fit(
    X_train_fold, y_train_fold,
    eval_set=[(X_val_fold, y_val_fold)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
```

### Feature importance is not feature validity
A feature with high importance might be a leaky feature. High importance means the model used it heavily. It does not mean the feature is valid. The Critic must check importance alongside leakage signals.

---

## LangGraph-Specific Standards

### State is the single source of truth — treat it as immutable
Never mutate state in place. Always return a new state dict. The LangGraph checkpointer depends on state being deterministically reproducible from any checkpoint. In-place mutation breaks this.

### Every node must be idempotent
If a node is called twice with the same input state, the second call must produce identical output state. This is not optional — it is required for the circuit breaker's retry logic to work correctly.

### Routing logic belongs in edge functions, not in nodes
Nodes do work. Edges decide where to go next. A node that checks `if state["error_count"] > 3: return END` is wrong. That logic belongs in the conditional edge function. Keep the separation clean.

### Never put payloads in state — only pointers
State carries file paths. Agents read files. Agents write files. State is updated with the new path. This is not a style preference — a 500k-row DataFrame in state will make the LangGraph checkpointer attempt to serialise it to JSON and fail. File pointers weigh bytes. DataFrames weigh gigabytes.

---

## When You Are Implementing a Task

### Step 1: Read the contract from Notion before writing
The Notion task note defines the contract. Read it completely. The note says `INPUT: raw_data_path. OUTPUT: cleaned.parquet + schema.json. Handles: missing values, type inference, basic profiling.` This is not a suggestion. This is the specification. Every word in it matters.

### Step 2: Write the function signature and docstring first
Before any implementation:
```python
def run_data_engineer(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Data Engineer.

    Reads:  state["raw_data_path"]        — must exist on disk
    Writes: state["clean_data_path"]      — str pointer to cleaned.parquet
            state["schema_path"]          — str pointer to schema.json
            state["data_hash"]            — SHA-256 first 16 chars
    Raises: FileNotFoundError             — if raw_data_path missing
            ValueError                    — if schema.json missing required fields
    Invariant: no raw DataFrame ever written to state
    """
```

This forces you to think about the full contract before you start writing. If you cannot complete this docstring, you do not understand the task well enough to implement it.

### Step 3: Write the unhappy paths first
Before the happy path. What does this function do when:
- The file doesn't exist?
- The CSV has all-null columns?
- The target column name doesn't match anything in the schema?
- The sandbox times out?
- The LLM returns syntactically invalid Python?

Write the raises and error handlers first. Then fill in the happy path.

### Step 4: Write assertions for invariants
At the end of every function, before the return statement, assert the invariants:
```python
assert os.path.exists(parquet_path), f"cleaned.parquet was not written: {parquet_path}"
assert os.path.exists(schema_path), f"schema.json was not written: {schema_path}"
assert "columns" in schema, "schema.json missing 'columns' field"
assert isinstance(result["clean_data_path"], str), "clean_data_path must be str pointer"
```

These assertions are not defensive coding paranoia. They are the first line of the contract test. If the assertion fires, the error message tells you exactly what broke.

### Step 5: Test it before declaring it done
Not "it looks right." Run it. With real data. Check the output. Check the types. Check the row counts. Check the state after it returns. If you cannot run it because the test fixture doesn't exist, create the fixture.

---

## The Difference Between Acceptable and Excellent

### Acceptable: handles the happy path
```python
def profile_data(df):
    return {
        "columns": df.columns,
        "types": {col: str(df[col].dtype) for col in df.columns}
    }
```

### Excellent: handles reality
```python
def profile_data(df: pl.DataFrame) -> dict:
    """
    Profile a Polars DataFrame for schema.json.
    
    Returns a complete schema including missing rates, cardinality,
    numeric/categorical/boolean column splits, and profiling timestamp.
    
    Raises: TypeError if df is not a Polars DataFrame (Pandas contamination check)
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"profile_data expects a Polars DataFrame, got {type(df)}. "
            "If you have a Pandas DataFrame, convert with pl.from_pandas(df) first. "
            "Never use Pandas in this pipeline — see tools/data_tools.py."
        )
    if len(df) == 0:
        raise ValueError(
            "profile_data received an empty DataFrame (0 rows). "
            "The Data Engineer should not be called on empty data. "
            "Check that raw_data_path points to a non-empty train.csv."
        )

    n_rows = len(df)
    missing_counts = {col: int(df[col].null_count()) for col in df.columns}
    missing_rates  = {
        col: round(missing_counts[col] / n_rows, 4)
        for col in df.columns
    }
    # ... rest of implementation
```

The excellent version: catches Pandas contamination immediately with an actionable message, catches empty DataFrame with an actionable message, has type annotations, has a docstring that specifies what it raises, and uses round() on missing rates so they are readable in JSON.

---

## How to Handle Uncertainty

When you are unsure about the right approach, do not default to the simplest thing. Ask:

**"What would a Kaggle Grandmaster do here?"**

A Grandmaster knows that:
- Target encoding must be done inside CV folds or it leaks
- Interaction features should be filtered by permutation importance before inclusion
- An ensemble of diverse models beats a single tuned model
- A CV/LB gap > 0.03 means something is wrong with the validation strategy, not the model
- The last 5% of performance comes from ensembling, pseudo-labeling, and blending — not from tuning learning rate from 0.05 to 0.04

When you implement something, implement it the way a Grandmaster would, not the way a Kaggle beginner would.

---

## What "Done" Means on This Project

A task is done when:

1. **It runs without error** on the real Spaceship Titanic dataset, not just the 5-row fixture
2. **The contract test passes** — all assertions green
3. **The lineage logger shows the right state mutations** — keys_read and keys_written match the docstring
4. **The output makes ML sense** — a CV AUC of 0.51 is not "done," it is broken
5. **The downstream agent can consume the output** without modification — if `schema.json` is missing a field that `ml_optimizer.py` reads, the task is not done
6. **The code is readable by someone who did not write it** — variable names are meaningful, the logic is followable, the error messages are actionable

A task is not done because the code was written. A task is done because the code works correctly in the context of the full pipeline.

---

## The Mindset

You are building something that will run autonomously, overnight, on a real competition, without you watching it. It will encounter missing values you didn't expect, LLM responses that aren't clean Python, timeouts, API rate limits, and data format edge cases.

Every line of code you write is either making that autonomous run more robust or more fragile.

Write code that makes it more robust. Every time.