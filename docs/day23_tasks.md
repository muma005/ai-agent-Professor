# Day 23 — Submission Strategist, Publisher, QA Gate
## Implementation Prompt for Qwen Code

---

## BEFORE YOU WRITE A SINGLE LINE

Read these files completely. Do not start until you have read all of them.

```
CLAUDE.md
AGENTS.md
core/state.py
core/professor.py
agents/ensemble_architect.py     ← understand what it puts in state
tools/submit_tools.py            ← understand how submission.csv is currently built
```

After reading, write down:
1. What keys does ensemble_architect put in state? List them.
2. What does submit_tools.py currently do with predictions before writing submission.csv?
3. Does a submission_log.json exist anywhere in outputs/? What is its format?

Do not proceed until you have answered all three from the actual code.

---

## TASK 1 — Build `agents/submission_strategist.py`

This agent runs after ensemble_architect. It selects the final submission pair, monitors the EWMA of CV/LB gap over time, and writes submission.csv.

### The submission log

All EWMA logic depends on a persistent submission log. The log lives at:

```
outputs/{session_id}/submission_log.json
```

Format — a list of submission records, append-only:

```json
[
  {
    "submission_number": 1,
    "session_id": "abc123",
    "competition_id": "spaceship-titanic",
    "timestamp": "2026-03-01T14:22:00Z",
    "cv_score": 0.8121,
    "lb_score": null,
    "cv_lb_gap": null,
    "model_used": "lgbm_optuna_1234",
    "ensemble_accepted": true,
    "n_models_in_ensemble": 2,
    "submission_path": "outputs/abc123/submission.csv",
    "is_final_pair_submission": false
  }
]
```

`lb_score` and `cv_lb_gap` are null at write time — they are filled in by the harness or manually after Kaggle scores the submission.

The log is read by the EWMA monitor and written to on every submission. If the log does not exist, create it as an empty list and write the first record.

### The EWMA monitor

**Purpose:** Detect when CV/LB gap is systematically worsening across submissions, which means the pipeline is overfitting to CV rather than generalising.

**Activation condition:** Monitor only activates after `MIN_SUBMISSIONS_BEFORE_MONITOR = 5` submissions with non-null `lb_score` entries in the log. Before that threshold, the monitor always returns `{"active": False, "action": "monitor_not_yet_active"}`.

**EWMA computation:**

```python
EWMA_ALPHA = 0.3   # smoothing factor — higher = more weight on recent gaps

def compute_ewma_gap(gap_history: list[float], alpha: float = EWMA_ALPHA) -> float:
    """
    gap_history: list of cv_lb_gap values in chronological order (oldest first).
    Returns the EWMA of gaps. First value initialises the EWMA.
    """
    ewma = gap_history[0]
    for gap in gap_history[1:]:
        ewma = alpha * gap + (1 - alpha) * ewma
    return ewma
```

**Freeze trigger — either condition fires the freeze:**

Condition A: `current_ewma > 2.0 * initial_ewma`
- `initial_ewma` = EWMA computed from the first 5 submissions with non-null lb_score
- `current_ewma` = EWMA computed from all submissions with non-null lb_score

Condition B: Gap increased in 5 of the last 7 submissions (count consecutive `gap[i] > gap[i-1]` instances in the last 7 non-null gap entries).

**When freeze fires:**

```python
state["submission_freeze_active"] = True
state["submission_freeze_reason"] = "ewma_exceeded_2x_initial" | "gap_increasing_5_of_7"
```

Log a WARNING. Do not crash. Do not halt the pipeline. The strategist continues and writes the submission — the freeze is a signal, not a blocker. The human decides whether to act on it.

### The final submission pair

Regardless of EWMA state, the strategist always selects two submissions:

**Submission A — best CV:**
The model with the highest `stability_score` in `model_registry`. If `ensemble_accepted=True` in state, use the ensemble OOF; otherwise use the best single model.

**Submission B — most diverse from A:**
The model in `model_registry` whose `oof_predictions` has the lowest Pearson correlation with Submission A's OOF predictions. This is the model that is right where A is wrong.

Both submissions are written as separate CSV files:
```
outputs/{session_id}/submission_a.csv   ← best CV
outputs/{session_id}/submission_b.csv   ← most diverse from A
outputs/{session_id}/submission.csv     ← same as submission_a.csv (default)
```

### Submission format validation

Before writing any submission CSV, validate it against `data/sample_submission.csv`:

```python
def validate_submission(submission_df, sample_submission_path, spec):
    sample = pl.read_csv(sample_submission_path)

    # Check 1: column names match exactly
    assert list(submission_df.columns) == list(sample.columns), (
        f"Column mismatch. Expected {list(sample.columns)}, "
        f"got {list(submission_df.columns)}"
    )

    # Check 2: row count matches test set
    assert len(submission_df) == len(sample), (
        f"Row count mismatch. Expected {len(sample)}, got {len(submission_df)}"
    )

    # Check 3: id column values match (same PassengerIds in same order)
    id_col = spec["id_column"]
    assert submission_df[id_col].to_list() == sample[id_col].to_list(), (
        "ID column values or order do not match sample_submission.csv"
    )

    # Check 4: target column dtype matches sample
    target_col = spec["target_column"]
    assert submission_df[target_col].dtype == sample[target_col].dtype, (
        f"Target column dtype mismatch. Expected {sample[target_col].dtype}, "
        f"got {submission_df[target_col].dtype}"
    )
```

If validation fails, raise `ValueError` with the full message. Do not write the file.

### State outputs

```python
state["submission_a_path"]         # str — path to submission_a.csv
state["submission_b_path"]         # str — path to submission_b.csv
state["submission_path"]           # str — path to submission.csv (same as A)
state["submission_a_model"]        # str — model name used for submission A
state["submission_b_model"]        # str — model name used for submission B
state["submission_b_correlation_with_a"]  # float — Pearson correlation of B OOF with A OOF
state["submission_log_path"]       # str — path to submission_log.json
state["submission_freeze_active"]  # bool — True if EWMA freeze triggered
state["submission_freeze_reason"]  # str — reason code or "" if not frozen
state["ewma_current"]              # float | None — current EWMA gap value
state["ewma_initial"]              # float | None — initial 5-submission EWMA
state["n_submissions_with_lb"]     # int — submissions in log with non-null lb_score
```

### Lineage event

```python
log_event(state, action="submission_strategist_complete", agent="submission_strategist",
details={
    "submission_a_model": state["submission_a_model"],
    "submission_b_model": state["submission_b_model"],
    "submission_b_correlation": state["submission_b_correlation_with_a"],
    "freeze_active": state["submission_freeze_active"],
    "freeze_reason": state["submission_freeze_reason"],
    "n_submissions_with_lb": state["n_submissions_with_lb"],
})
```

---

## TASK 2 — Build `agents/publisher.py`

The publisher produces a structured HTML report after every pipeline run. It never hallucinates statistics. Every number comes from `metrics.json`. The LLM writes narrative only in designated slots.

### Template structure

The HTML template has named slots marked with `{{SLOT_NAME}}`. Numbers are injected programmatically before the LLM is called. The LLM only fills narrative slots — it never sees raw numbers to summarise, only receives pre-formatted strings to weave into prose.

```python
TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Professor Agent — Competition Report</title></head>
<body>

<h1>{{COMPETITION_NAME}}</h1>
<p class="timestamp">Generated: {{TIMESTAMP}}</p>

<section id="performance">
  <h2>Performance</h2>
  <table>
    <tr><td>CV Score</td><td>{{CV_SCORE}}</td></tr>
    <tr><td>CV Std</td><td>{{CV_STD}}</td></tr>
    <tr><td>Winning Model</td><td>{{WINNING_MODEL_TYPE}}</td></tr>
    <tr><td>Features Used</td><td>{{N_FEATURES_FINAL}}</td></tr>
    <tr><td>Features Dropped (Null Importance)</td><td>{{N_FEATURES_DROPPED}}</td></tr>
    <tr><td>Ensemble Accepted</td><td>{{ENSEMBLE_ACCEPTED}}</td></tr>
    <tr><td>Pseudo-labels Applied</td><td>{{PSEUDO_LABELS_APPLIED}}</td></tr>
    <tr><td>Runtime</td><td>{{RUNTIME_SECONDS}}s</td></tr>
  </table>
</section>

<section id="narrative">
  <h2>Analysis</h2>
  <p>{{NARRATIVE_PERFORMANCE}}</p>
</section>

<section id="critic">
  <h2>Critic Findings</h2>
  <p>Overall severity: {{CRITIC_SEVERITY}}</p>
  <p>{{NARRATIVE_CRITIC}}</p>
</section>

<section id="next">
  <h2>Recommended Next Steps</h2>
  <p>{{NARRATIVE_NEXT_STEPS}}</p>
</section>

</body>
</html>
"""

NUMERIC_SLOTS = {
    "CV_SCORE", "CV_STD", "N_FEATURES_FINAL", "N_FEATURES_DROPPED",
    "RUNTIME_SECONDS", "N_SUBMISSIONS_WITH_LB",
}

NARRATIVE_SLOTS = {
    "NARRATIVE_PERFORMANCE", "NARRATIVE_CRITIC", "NARRATIVE_NEXT_STEPS",
}
```

### Injection logic

```python
def fill_numeric_slots(template: str, metrics: dict, state: dict) -> str:
    """
    Injects all non-narrative values from metrics.json and state.
    Every NUMERIC_SLOT must be filled. If a value is missing, inject "N/A".
    Never calls an LLM for this step.
    """
    replacements = {
        "COMPETITION_NAME":       state.get("competition_name", "Unknown"),
        "TIMESTAMP":              datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "CV_SCORE":               f"{metrics.get('cv_mean', 'N/A'):.5f}",
        "CV_STD":                 f"{metrics.get('cv_std', 'N/A'):.5f}",
        "WINNING_MODEL_TYPE":     state.get("winning_model_type", "N/A"),
        "N_FEATURES_FINAL":       str(state.get("n_features_final", "N/A")),
        "N_FEATURES_DROPPED":     str(
            state.get("stage1_drop_count", 0) + state.get("stage2_drop_count", 0)
        ),
        "ENSEMBLE_ACCEPTED":      str(state.get("ensemble_accepted", False)),
        "PSEUDO_LABELS_APPLIED":  str(state.get("pseudo_labels_applied", False)),
        "RUNTIME_SECONDS":        str(state.get("total_runtime_seconds", "N/A")),
        "CRITIC_SEVERITY":        state.get("critic_severity", "unchecked"),
    }
    for slot, value in replacements.items():
        template = template.replace("{{" + slot + "}}", value)
    return template
```

### LLM narrative generation

After numeric slots are filled, call the LLM once per narrative slot. The LLM prompt for each slot includes the already-filled template as context but receives explicit instructions not to invent numbers.

```python
NARRATIVE_PROMPT = """
You are writing one section of a Kaggle competition analysis report.
The report already contains all statistics — they were injected programmatically.
Your job is to write 2-3 sentences of narrative for the {slot_name} section.

DO NOT invent or repeat any numbers. The numbers are already in the report.
DO NOT write phrases like "with a CV score of X" — the CV score is already displayed in the table.
Write only observations, interpretations, and recommendations in plain English.

Context from this run:
- Competition: {competition_name}
- CV score: {cv_score}
- Winning model: {winning_model_type}
- Features used: {n_features_final}
- Critic severity: {critic_severity}
- Ensemble accepted: {ensemble_accepted}

Write the narrative for: {slot_name}
"""
```

### State outputs

```python
state["report_path"]    # str — path to outputs/{session_id}/report.html
state["report_written"] # bool — True if report written successfully
```

If LLM call fails for any narrative slot, inject `"[Narrative unavailable]"` into that slot. Never crash. The report must always be written even with partial narrative.

---

## TASK 3 — Build `agents/qa_gate.py`

The QA gate runs after publisher. It checks three things using only deterministic code — no LLM judgment.

### Check 1: All template slots filled

Read the HTML report from `state["report_path"]`. Search for any remaining `{{SLOT_NAME}}` patterns. If any are found, the report failed — at least one slot was not filled.

```python
import re

def check_no_unfilled_slots(report_html: str) -> list[str]:
    """Returns list of unfilled slot names. Empty list = all filled."""
    pattern = re.compile(r"\{\{([A-Z_]+)\}\}")
    return pattern.findall(report_html)
```

### Check 2: No orphan numbers in narrative

An orphan number is a decimal number (e.g. `0.8121`, `0.003`, `32.5`) appearing inside a narrative `<p>` tag but not inside a `<table>`. Numbers belong in tables, not in LLM-generated narrative.

```python
def check_no_orphan_numbers_in_narrative(report_html: str) -> list[str]:
    """
    Extracts text from narrative <p> tags (those not inside <table>).
    Returns list of decimal numbers found in narrative text.
    Integers like "5" or "10" are allowed — only decimals (containing ".") are flagged.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(report_html, "html.parser")

    # Remove all table content
    for table in soup.find_all("table"):
        table.decompose()

    narrative_text = " ".join(p.get_text() for p in soup.find_all("p"))
    decimal_pattern = re.compile(r"\b\d+\.\d+\b")
    return decimal_pattern.findall(narrative_text)
```

If `beautifulsoup4` is not in requirements.txt, add it. Import it with `from bs4 import BeautifulSoup`.

### Check 3: Submission CSV matches sample_submission format exactly

```python
def check_submission_format(submission_path: str, sample_path: str) -> list[str]:
    """
    Returns list of format violations. Empty list = format correct.
    Checks: column names, column count, row count, id column order, target dtype.
    """
    errors = []
    sub    = pl.read_csv(submission_path)
    sample = pl.read_csv(sample_path)

    if list(sub.columns) != list(sample.columns):
        errors.append(
            f"Column names mismatch: expected {list(sample.columns)}, "
            f"got {list(sub.columns)}"
        )
    if len(sub) != len(sample):
        errors.append(
            f"Row count mismatch: expected {len(sample)}, got {len(sub)}"
        )
    if not errors:  # only check dtypes if column names match
        for col in sample.columns:
            if sub[col].dtype != sample[col].dtype:
                errors.append(
                    f"Column '{col}' dtype mismatch: "
                    f"expected {sample[col].dtype}, got {sub[col].dtype}"
                )
    return errors
```

### Gate verdict

```python
def run_qa_gate(state: ProfessorState) -> ProfessorState:
    failures = []

    # Check 1: unfilled slots
    report_path = state.get("report_path")
    if report_path and Path(report_path).exists():
        html = Path(report_path).read_text()
        unfilled = check_no_unfilled_slots(html)
        if unfilled:
            failures.append(f"Unfilled template slots: {unfilled}")
        orphans = check_no_orphan_numbers_in_narrative(html)
        if orphans:
            failures.append(f"Orphan numbers in narrative: {orphans}")
    else:
        failures.append("Report file not found or not written.")

    # Check 3: submission format
    submission_path = state.get("submission_path")
    sample_path     = state.get("sample_submission_path",
                                f"data/{state['competition_name']}/sample_submission.csv")
    if submission_path and Path(submission_path).exists():
        fmt_errors = check_submission_format(submission_path, sample_path)
        failures.extend(fmt_errors)
    else:
        failures.append("submission.csv not found.")

    passed = len(failures) == 0

    state["qa_passed"]   = passed
    state["qa_failures"] = failures

    if not passed:
        logger.warning(
            f"[qa_gate] QA FAILED — {len(failures)} issue(s):\n" +
            "\n".join(f"  - {f}" for f in failures)
        )
    else:
        logger.info("[qa_gate] QA PASSED — all checks green.")

    log_event(state, action="qa_gate_complete", agent="qa_gate", details={
        "passed":   passed,
        "failures": failures,
        "n_checks": 3,
    })

    return state
```

### What not to do

- Do not call an LLM anywhere in QA gate. All checks are deterministic code.
- Do not raise on QA failure — set `state["qa_passed"] = False` and continue.
- Do not modify submission.csv — QA gate is read-only.

---

## COMMIT SEQUENCE

One commit per task:

```
git commit -m "Day 23: agents/submission_strategist.py — EWMA monitor, final pair, format validation"
git commit -m "Day 23: agents/publisher.py — structured template, programmatic slot injection"
git commit -m "Day 23: agents/qa_gate.py — deterministic format, slot, orphan number checks"
```

---

## VERIFICATION BEFORE EACH COMMIT

```bash
python -c "from agents.submission_strategist import run_submission_strategist; print('OK')"
python -c "from agents.publisher import run_publisher; print('OK')"
python -c "from agents.qa_gate import run_qa_gate; print('OK')"

pytest tests/contracts/test_submission_strategist_contract.py -v --tb=short
pytest tests/test_day23_quality.py -v --tb=short
pytest tests/contracts/ -v --tb=short
pytest tests/regression/ -v --tb=short
```

All four pytest commands must show zero failures before committing.