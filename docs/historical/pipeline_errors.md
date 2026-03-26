# Pipeline Errors — Smoke Run Log

> Last updated: 2026-03-24
> Source: Run 1 (`logs_safe.txt`), Run 2 (end-to-end smoke with `PYTHONIOENCODING=utf-8`, 2 Optuna trials)
> Dataset: `tiny_spaceship` — 100 rows

---

## ERROR 1 — Unicode Encoding Crash (CRITICAL — FIXED BY ENV VAR)

**Where:** `agents/data_engineer.py` — inside `run_data_engineer`
**Symptom:**
```
'charmap' codec can't encode character '\u2713' in position 15: character maps to <undefined>
```
**Root cause:** Print statements use `✓` (U+2713). Windows console defaults to `cp1252` which cannot encode it.
**Fix:** Set `PYTHONIOENCODING=utf-8` before running. Confirmed working in Run 2.

---

## ERROR 2 — task_type Integrity Gate Failure (DOWNSTREAM of Error 1)

**Where:** `guards/pipeline_integrity.py` — `check_post_data_engineer()`
**Symptom:**
```
[FAIL] task_type_valid: task_type='tabular_classification' not in {'multiclass', 'regression', 'binary'}
```
**Root cause:** SemanticRouter sets `task_type='tabular_classification'`. DataEngineer overwrites it with `'binary'` — but only if it doesn't crash first (Error 1).
**Fix:** Fix Error 1. Confirmed: with `PYTHONIOENCODING=utf-8`, DataEngineer sets `task_type='binary'` and this error disappears.

---

## ERROR 3 — All Round 2 LLM Features Rejected (MEDIUM)

**Where:** `agents/feature_factory.py` — Round 2 domain feature generation
**Symptom:**
```
[FeatureFactory] Suppressed invalid AST round 2 feature total_spending: Syntax error in expression: Sum of all expenditure columns (RoomService + FoodCourt + ShoppingMall + Spa + VRDeck)
[FeatureFactory] Suppressed invalid AST round 2 feature age_bin: Syntax error in expression: Bin Age into categories ...
... (15 features, ALL rejected)
```
**Root cause:** The LLM (DeepSeek) returns **English descriptions** instead of valid Polars expressions. The AST validator correctly rejects them all.
**Consequence:** Round 2 contributes ZERO features. Pipeline continues but with weaker feature set.
**Fix needed:** Improve the LLM prompt in `_generate_round2_features()` to demand Polars code, not descriptions. Or add few-shot examples.

---

## ERROR 4 — Bool Multiply Fails in Feature Factory (MEDIUM)

**Where:** `agents/feature_factory.py` — Round 5 feature generation
**Symptom:**
```
[FeatureFactory] Suppressed invalid AST round 5 feature CryoSleep_x_VIP: `multiply` operation not supported for dtype `bool`
```
**Root cause:** `CryoSleep` and `VIP` are boolean columns. Polars does not support `bool * bool`. Need `.cast(pl.Int8)` first.
**Consequence:** Interaction features between boolean columns silently fail. Pipeline continues.
**Fix needed:** Cast boolean columns to int before multiplication in Round 5 feature generation.

---

## ERROR 5 — NullImportance Stage 2 Sandbox Crash (MEDIUM)

**Where:** `tools/null_importance.py` — Stage 2 (50-shuffle null distribution)
**Symptom:**
```
[NullImportance] Stage 2 sandbox raised: Code failed after 1 attempts.
Final error: ImportError
Final traceback:
Import of 'sys' is not allowed in sandbox. Blocked modules: ctypes, ftplib, http, multiprocessing, pty, resource, shutil, signal, smtplib, socket, subprocess, sys, urllib.
Returning all survivors (no Stage 2 filtering).
```
**Root cause:** The sandbox code generated for Stage 2 imports `sys` (for progress printing). The restricted sandbox blocks it.
**Consequence:** Stage 2 filtering is skipped entirely. All Stage 1 survivors pass through unfiltered. Features that should be dropped are kept.
**Fix needed:** Remove `import sys` / `print(..., file=sys.stderr)` from the Stage 2 sandbox code template. Use plain `print()` instead.

---

## ERROR 6 — Feature Names Lost in ML Optimizer (LOW)

**Where:** `agents/ml_optimizer.py` — during Optuna CV and prediction
**Symptom:**
```
UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
```
(repeated hundreds of times)
**Root cause:** Model is fitted on a DataFrame (with column names), but predictions are made on a numpy array (no names). LGBM warns about the mismatch.
**Consequence:** Predictions are still correct (column order matches). But the warnings flood stderr and obscure real errors.
**Fix needed:** Either convert X to numpy before fitting, or keep as DataFrame throughout. Pick one consistently.

---

## ERROR 7 — WilcoxonGate Fallback: Too Few Folds (LOW)

**Where:** `tools/wilcoxon_gate.py`
**Symptom:**
```
[WilcoxonGate] Only 3 folds — minimum 5 required for reliable Wilcoxon test. Falling back to mean comparison.
```
**Root cause:** With 100 rows, StratifiedKFold produces only 3 folds (or the fold count is set to 3). Wilcoxon signed-rank test needs >= 5 paired observations.
**Consequence:** Statistical significance testing is bypassed. Mean comparison is used instead — less rigorous but functional.
**Fix needed:** Not a bug per se — just a data-size limitation. On real datasets (1000+ rows) with 5+ folds, this works correctly.

---

## ERROR 8 — Pipeline Crashed at Exit Code 1 (UNKNOWN — output truncated)

**Where:** Unknown — likely `red_team_critic` or `submit` node
**Symptom:** Process exited with code 1. Output was truncated after ~20K chars of sklearn warnings.
**Root cause:** Unknown. The crash happened after ml_optimizer completed (warnings prove it ran). The next nodes are `red_team_critic` → `submit`.
**Fix needed:** Re-run with stderr redirected to file to see the actual traceback.

---

## ERROR 9 — google.generativeai Deprecation Warning (LOW)

**Where:** `tools/llm_client.py:6`
**Symptom:**
```
All support for the `google.generativeai` package has ended.
Please switch to the `google.genai` package as soon as possible.
```
**Impact:** Non-fatal today. Will break when Google removes the package.
**Fix needed:** Migrate to `google.genai`.

---

## ERROR 10 — Docker Not Available (LOW / KNOWN)

**Where:** `tools/e2b_sandbox.py`
**Symptom:**
```
[sandbox] Docker not available — falling back to subprocess sandbox.
```
**Impact:** Lower isolation but functional.
**Fix needed:** Install Docker Desktop or accept subprocess mode.

---

## Confirmed Working (from Run 2 evidence)

| Component | Status | Evidence |
|-----------|--------|----------|
| SemanticRouter | PASS | Routes correctly to full DAG |
| CompetitionIntel | PASS | Completes (0 public notebooks — expected for fake competition) |
| DataEngineer | PASS | `target=Transported`, `task_type=binary`, cleaned.parquet written (with utf-8 fix) |
| EDA Agent | PASS | Ran (no errors in output) |
| ValidationArchitect | PASS | Ran (no errors in output) |
| FeatureFactory | PARTIAL | Round 1 works, Round 2 all rejected, Round 5 bool crash |
| NullImportance | PARTIAL | Stage 1 works, Stage 2 sandbox crash (falls back gracefully) |
| MLOptimizer | PASS | Optuna ran, model trained, OOF predictions generated |
| WilcoxonGate | PARTIAL | Falls back to mean comparison (too few folds) |
| Integrity Gates | PASS | POST_DATA_ENGINEER passes with utf-8 fix |
| Circuit Breaker | PASS | Correctly detects failures and escalates |
| agent_retry | PASS | 3-attempt retry works correctly |

---

## Priority Fix Order

| # | Error | Severity | Effort |
|---|-------|----------|--------|
| 1 | Unicode encoding (Error 1) | CRITICAL | 1 min — env var |
| 2 | Pipeline crash at exit (Error 8) | CRITICAL | Unknown — need traceback |
| 3 | NullImportance `sys` import (Error 5) | MEDIUM | 5 min — remove sys from template |
| 4 | Round 2 LLM features all rejected (Error 3) | MEDIUM | 15 min — fix prompt |
| 5 | Bool multiply in Round 5 (Error 4) | MEDIUM | 5 min — cast to int |
| 6 | Feature names warning spam (Error 6) | LOW | 5 min — use numpy consistently |
| 7 | WilcoxonGate fallback (Error 7) | LOW | Not a bug on small data |
| 8 | google.generativeai deprecation (Error 9) | LOW | 30 min — SDK migration |
| 9 | Docker not available (Error 10) | LOW | External — install Docker |
