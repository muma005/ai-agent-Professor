# BUILD PROMPT — Layer 1: Intelligence Gathering (Days 3-5)
# Feed to Gemini CLI with: @PROFESSOR.md @STATE.md @SANDBOX.md @HITL.md @POLARS.md @CONTRACTS.md @PROMPTS.md

---

## CONTEXT

Layer 0 is complete and passing all contract tests: ProfessorState (typed, ownership-enforced), Self-Debugging Engine (4-layer retry cascade), HITL (CLI + Telegram), Cost Governor (budget caps + rate limiting), Metric Verification Gate, Data Integrity Checkpoints.

Now you're building the intelligence-gathering layer — the agents that understand the competition and data BEFORE any feature engineering or model training begins. These agents run early in the pipeline and produce context that every downstream agent consumes.

All 4 components in this prompt write state through `validated_update()`, emit messages through `emit_to_operator()`, and execute code through `run_in_sandbox()`. They all depend on Layer 0. They do NOT depend on each other — Pre-Flight, EDA, and Domain Research can be built and tested independently.

---

## COMMIT PLAN (6 commits)

```
Commit 1:  shields/preflight.py + tests/contracts/test_preflight_contract.py
Commit 2:  agents/eda_agent.py (Deep EDA upgrade) + tests/contracts/test_eda_v2_contract.py
Commit 3:  tools/eda_plots.py (Artifact Export) — extends Commit 2's EDA agent
Commit 4:  agents/domain_research.py + tests/contracts/test_domain_research_contract.py
Commit 5:  agents/shift_detector.py + tests/contracts/test_shift_detector_contract.py
Commit 6:  Integration test — all 4 components run in sequence on mock data, state flows correctly
```

Every commit passes `pytest tests/contracts/ -q` including ALL Layer 0 tests. If a Layer 0 test breaks, fix before committing. Layer 0 contracts are IMMUTABLE.

---

## COMMIT 1: Pre-Flight Checks (shields/preflight.py)

### What this prevents

Professor enters a competition with a 12GB dataset → OOM. Or text columns requiring NLP → garbage features. Or JSON submission format → invalid submission. 20-30 minutes of pipeline wasted before anyone discovers incompatibility.

### The LangGraph node function

```python
def run_preflight_checks(state: ProfessorState) -> dict:
    """
    FIRST node in the pipeline. Runs before Competition Intel.
    Profiles data without full loading. Emits Milestone 0.
    
    Reads: raw_data_path (directory containing competition files)
    Writes: preflight_data_files, preflight_warnings, preflight_data_size_mb,
            preflight_submission_format, preflight_target_type,
            preflight_unsupported_modalities, preflight_passed
    Emits: CHECKPOINT (Milestone 0) with full pre-flight report
    """
```

### Sub-function 1: _inventory_data_files(data_dir: str) -> list[dict]

Walk the data directory with `os.listdir()` (non-recursive — competition data is flat). For each file:

- `name`: filename
- `size_mb`: `os.path.getsize(path) / (1024 * 1024)`, round to 1 decimal
- `format`: detect from extension. Map: `.csv`/`.tsv` → "csv", `.json` → "json", `.parquet` → "parquet", `.npy`/`.npz` → "numpy", `.jpg`/`.jpeg`/`.png`/`.bmp`/`.gif` → "image", `.wav`/`.mp3`/`.flac` → "audio", `.txt` → "text", `.zip`/`.tar`/`.gz` → "archive", everything else → "other"
- `will_use`: True if filename matches (case-insensitive): `train.csv`, `train.parquet`, `train.tsv`, `test.csv`, `test.parquet`, `test.tsv`, `sample_submission.csv`, `sample_submission.parquet`. False for everything else.

Compute `total_size_mb = sum(f["size_mb"] for f in files)`.

Check available RAM: try `import psutil; available_ram_mb = psutil.virtual_memory().total / (1024*1024)`. If psutil unavailable, assume 16384 MB (16GB). If `total_size_mb > available_ram_mb * 0.7`: create a warning `{"type": "large_dataset", "column": None, "description": f"Total data {total_size_mb:.0f}MB exceeds 70% of available RAM ({available_ram_mb:.0f}MB). Recommend chunked loading or cloud compute."}`.

If any single file > 2048 MB (2GB): create a warning `{"type": "large_file", "column": None, "description": f"File '{name}' is {size_mb:.0f}MB. Will use pl.scan_csv() for lazy evaluation."}`.

### Sub-function 2: _profile_columns(file_path: str, n_rows: int = 100) -> list[dict]

Detect file format from extension. Load first `n_rows`:
- CSV/TSV: `pl.read_csv(file_path, n_rows=n_rows)` (for TSV: add `separator="\t"`)
- Parquet: `pl.read_parquet(file_path, n_rows=n_rows)` — parquet supports row-level reads natively
- Other formats: skip profiling, return empty list with advisory warning

For each column in the loaded sample, build a profile dict:
```python
{
    "name": col_name,
    "dtype": str(df[col_name].dtype),   # e.g., "Int64", "Utf8", "Float64"
    "n_unique_in_sample": df[col_name].n_unique(),
    "null_pct_in_sample": round(df[col_name].null_count() / len(df) * 100, 1),
    "flags": [],   # populated below
}
```

**Flag detection (applied per column):**

For string (Utf8) columns:
- Compute `avg_len = df[col_name].str.len_chars().mean()`. If `avg_len` is not None and `avg_len > 50`: append flag `{"type": "possible_nlp", "description": f"Avg string length {avg_len:.0f} chars — may require NLP processing"}`
- Sample 10 non-null values. If >50% match `r'\.(jpg|jpeg|png|bmp|gif|tiff|webp)$'` (case-insensitive): append `{"type": "image_paths", "description": "Column contains image file paths"}`
- If >50% match `r'\.(wav|mp3|flac|ogg|aac)$'`: append `{"type": "audio_paths", "description": "Column contains audio file paths"}`
- If >50% match `r'^[\[{]'` (start with `[` or `{`): append `{"type": "nested_json", "description": "Column contains JSON/nested structures"}`
- If >50% contain `|` (pipe character): append `{"type": "pipe_delimited", "description": "Column contains pipe-delimited lists — possible multi-label"}`
- If >50% match `r'^\d{4}-\d{2}-\d{2}'`: append `{"type": "datetime_candidate", "description": "Column contains ISO date strings — parse as datetime"}`

For ALL columns:
- If `n_unique_in_sample == 1`: append `{"type": "constant", "description": "Only 1 unique value — drop candidate"}`
- If `null_pct_in_sample > 80`: append `{"type": "mostly_null", "description": f"{null_pct_in_sample}% null — consider dropping or engineering missingness feature"}`
- If dtype is Utf8 and `n_unique_in_sample > 90` (out of 100 rows): append `{"type": "high_cardinality", "description": f"{n_unique_in_sample} unique values in 100 rows — high cardinality categorical"}`

**Tail sampling (for files < 1GB):**
If file size < 1024 MB: also load the LAST 100 rows. For CSV: `pl.read_csv(file_path).tail(n_rows)` — yes this loads the full file, only for < 1GB. Merge any NEW flags from tail profiling that weren't in head profiling. This catches patterns like "first 100 rows are numeric IDs, but rows 5000+ have text descriptions."

For files >= 1GB: skip tail sampling. Add an advisory note: `"Large file — column profiling based on first 100 rows only."`

### Sub-function 3: _verify_submission_format(data_dir: str) -> dict

Find sample submission file. Search for (case-insensitive glob): `sample_submission.*`, `sampleSubmission.*`, `sample_sub.*`. Accept `.csv`, `.parquet`, `.json`.

If not found: return `{"format": "unknown", "columns": [], "n_rows": 0, "value_types": {}, "compatible": True, "issues": ["No sample submission file found — cannot verify output format"]}`.

If found:
- Detect format from extension
- If CSV: `df = pl.read_csv(path)`. Extract column names, row count. Infer value_types per column: check if values are floats in [0,1] → "probability", integers in {0,1} → "binary_class", integers with >2 unique → "multiclass", floats outside [0,1] → "continuous".
- If JSON: return `{"format": "json", "compatible": False, "issues": ["JSON submission format — Professor v2 outputs CSV only"]}`. This is a BLOCKING issue.
- If parquet: same as CSV but read with `pl.read_parquet()`.

Check compatibility: `compatible = (format in ["csv", "parquet"])`. List any issues.

### Sub-function 4: _detect_target_type(df_sample: pl.DataFrame, target_col: str) -> str

If `target_col` not in `df_sample.columns`: return "unknown" (target column not identified yet — Competition Intel will find it).

Examine the target column in the sample:
- If dtype is numeric and `n_unique == 2`: return "binary"
- If dtype is numeric and `3 <= n_unique <= 30` and all values are integer-like (`df[target_col].cast(pl.Int64, strict=False).is_not_null().all()`): return "multiclass" (could also be ordinal — flag for review)
- If dtype is numeric and `n_unique > 30`: return "regression"
- If dtype is Utf8 and `n_unique <= 30`: return "multiclass"
- If dtype is Utf8 and any value contains `|` or starts with `[`: return "multilabel"
- Else: return "unknown"

### Sub-function 5: _check_capability_boundaries(inventory, profiles, submission_fmt) -> list[str]

Collect unsupported modalities:
- If ANY column has flag type "image_paths": append "image"
- If ANY column has flag type "audio_paths": append "audio"
- If ANY column has flag type "possible_nlp" AND no tabular columns exist (all columns are text): append "nlp" (if text columns coexist with numeric/categorical, it's tabular-with-text — Professor can handle the tabular part)
- If submission format is not compatible: append submission format name
- If ANY file > 10GB: append "extreme_large_dataset"
- If target type is "multilabel": append "multilabel"

### Assembling the report and emitting Milestone 0

Categorize all warnings:
- **BLOCKING** (pipeline halts, requires operator acknowledgment): unsupported submission format, unsupported modalities where ALL data is that modality (pure image classification), extreme_large_dataset
- **ADVISORY** (pipeline continues, operator informed): large_file, large_dataset, possible_nlp alongside tabular, high_cardinality, constant columns, mostly_null, unused supplementary files

Build the Milestone 0 message. Structure it as:
```
🚀 PRE-FLIGHT REPORT

📁 Files: {n_files} found
   ✅ Using: train.csv ({size}MB), test.csv ({size}MB)
   ❓ Not using: metadata.csv ({size}MB), images/ ({size}MB)
   
📊 Columns: {n_cols} total ({n_numeric} numeric, {n_string} string, {n_datetime} datetime candidates)
   
⚠️ Flags:
   - 'description' avg length 120 chars — possible NLP
   - 'category_id' has 95 unique in 100 rows — high cardinality
   - 'unused_col' is constant — drop candidate

🎯 Target: '{target_col}' — {target_type}
📤 Submission: {format}, {n_cols} columns, {n_rows} rows — {compatible_status}

{capability_status}
```

If BLOCKING issues: emit as GATE. Pipeline halts until operator responds. The operator gets options: "(a) abort this competition, (b) proceed with tabular features only, (c) override and continue anyway."

If ADVISORY only: emit as CHECKPOINT with 3-minute timeout. If operator responds with domain knowledge or file overrides, incorporate. If timeout, continue.

### State return

```python
return state.validated_update("preflight", {
    "preflight_data_files": inventory,
    "preflight_warnings": all_warnings,
    "preflight_data_size_mb": total_size_mb,
    "preflight_submission_format": submission_format,
    "preflight_target_type": target_type,
    "preflight_unsupported_modalities": unsupported,
    "preflight_passed": len(blocking_warnings) == 0,
})
```

### Contract tests: tests/contracts/test_preflight_contract.py

Use `tmp_path` pytest fixture to create temporary directories with mock CSV files for each test.

```python
@pytest.fixture
def clean_tabular_dir(tmp_path):
    """Standard tabular competition — should pass cleanly."""
    train = pl.DataFrame({
        "id": range(1000),
        "age": [25 + i % 50 for i in range(1000)],
        "income": [30000.0 + i * 100 for i in range(1000)],
        "category": ["A", "B", "C", "D"] * 250,
        "target": [0, 1] * 500,
    })
    test = train.drop("target").head(250)
    sample_sub = pl.DataFrame({"id": range(250), "target": [0.5] * 250})
    
    train.write_csv(tmp_path / "train.csv")
    test.write_csv(tmp_path / "test.csv")
    sample_sub.write_csv(tmp_path / "sample_submission.csv")
    return tmp_path


@pytest.fixture  
def nlp_heavy_dir(tmp_path):
    """Competition with text columns — should flag NLP."""
    train = pl.DataFrame({
        "id": range(100),
        "text": ["This is a long text description that exceeds fifty characters easily for testing " * 2] * 100,
        "target": [0, 1] * 50,
    })
    train.write_csv(tmp_path / "train.csv")
    pl.DataFrame({"id": range(50), "target": [0.5] * 50}).write_csv(tmp_path / "sample_submission.csv")
    return tmp_path


@pytest.fixture
def supplementary_files_dir(tmp_path):
    """Competition with extra files that Professor doesn't auto-use."""
    # ... train.csv, test.csv, sample_submission.csv, metadata.csv, external_features.parquet
    return tmp_path
```

Tests:
1. `test_clean_tabular_passes(clean_tabular_dir)` — preflight_passed=True, zero blocking warnings, preflight_warnings may have advisories but no blockers
2. `test_text_column_flagged(nlp_heavy_dir)` — column "text" has flag type "possible_nlp"
3. `test_binary_target_detected(clean_tabular_dir)` — preflight_target_type == "binary"
4. `test_regression_target_detected` — create fixture with continuous target, verify "regression"
5. `test_multiclass_target_detected` — create fixture with 5-class target, verify "multiclass"
6. `test_submission_csv_compatible(clean_tabular_dir)` — submission format compatible=True
7. `test_submission_json_blocks` — create sample_submission.json, verify compatible=False and BLOCKING
8. `test_supplementary_files_listed(supplementary_files_dir)` — metadata.csv in inventory with will_use=False
9. `test_constant_column_flagged` — column with 1 unique value gets "constant" flag
10. `test_high_cardinality_flagged` — string column with 95 unique in 100 rows gets "high_cardinality" flag
11. `test_mostly_null_flagged` — column with 90% null gets "mostly_null" flag
12. `test_image_paths_flagged` — column with "/img/001.jpg" values gets "image_paths" flag and "image" in unsupported_modalities
13. `test_profiling_sample_size` — mock pl.read_csv, verify n_rows=100 is passed
14. `test_missing_sample_submission_warns` — no sample_submission file → advisory warning, NOT blocking
15. `test_no_crash_on_empty_directory` — empty data dir → warnings but no crash
16. `test_milestone_0_emitted` — mock emit_to_operator, verify CHECKPOINT emitted with structured data
17. `test_state_fields_correct_types` — all returned state fields match ProfessorState types

---

## COMMIT 2: Deep EDA Agent (agents/eda_agent.py)

### What this upgrades

v1 EDA produces 8 keys (target distribution, correlations, outlier profile, duplicate analysis, temporal profile, leakage fingerprint, drop candidates, summary). These tell you WHAT the data looks like but not WHAT IT MEANS for downstream decisions.

v2 adds 4 new output sections that give downstream agents actionable context.

### The LangGraph node function

```python
def eda_agent(state: ProfessorState) -> dict:
    """
    Enhanced EDA with Deep Analysis + Artifact Export.
    
    Reads: clean_data_path, clean_test_path, target_column, 
           canonical_train_rows, data_schema, domain_brief (if available)
    Writes: eda_report (v1 8 keys — PRESERVED), eda_insights_summary,
            eda_mutual_info, eda_vif_report, eda_modality_flags,
            eda_plots_paths, eda_plots_delivered, eda_quick_baseline_importance
    Emits: STATUS (start), RESULT (stats complete), CHECKPOINT (Milestone 1 with plots)
    """
```

### Part A: v1 EDA (preserve exactly)

If v1 EDA code already exists, DO NOT modify it. The v1 contract tests are FROZEN. Run the existing v1 logic to produce the 8 original keys in `eda_report`. If v1 code doesn't exist yet, implement the 8 keys:

1. `target_distribution`: value counts, class balance percentages, basic stats (mean, std, min, max, median for regression)
2. `correlations`: Pearson correlation of all numeric features with target, top 10 pairs
3. `outlier_profile`: per-numeric-column: count of values beyond 3 std from mean
4. `duplicate_analysis`: number of exact duplicate rows, percentage
5. `temporal_profile`: if any datetime column exists, check if data is sorted by time, date range
6. `leakage_fingerprint`: check for columns perfectly correlated with target (correlation > 0.99), columns with names suspiciously close to target
7. `drop_candidates`: constant columns, >95% null columns, duplicate columns
8. `summary`: text summary of key findings

All of this runs in the sandbox via `run_in_sandbox()` with agent_name="eda_agent".

### Part B: v2 Deep Analysis (4 new sections)

Generate a SINGLE code block that computes all 4 sections. Execute via `run_in_sandbox()`. The code reads the clean parquet file and outputs JSON results.

**Section 1 — Statistical Profiling:**

For each numeric column: compute skewness (`df[col].skew()`), kurtosis (use scipy: `from scipy.stats import kurtosis`), and modality count. For modality: use KDE to find peaks, or simpler: create a histogram with 50 bins, find bins with local maxima (count > both neighbors). If >1 peak: `is_multimodal = True`.

Recommended transform per column: if `abs(skewness) > 2`: recommend "log" (if all positive) or "sqrt" (if all non-negative) or "yeo-johnson" (if negative values exist). If `abs(skewness) < 0.5`: "none". Otherwise: "box-cox" or "yeo-johnson".

Output: `{"statistical_profiling": {col_name: {"skewness": float, "kurtosis": float, "n_modes": int, "is_multimodal": bool, "recommended_transform": str}}}`

**Section 2 — Mutual Information:**

Compute MI between each numeric feature and the target. Use `from sklearn.feature_selection import mutual_info_classif, mutual_info_regression`. Choose the right function based on `state.competition_type` (binary/multiclass → classif, regression → regression).

If >50 numeric columns: only compute MI for top 50 by variance (to avoid O(n²) cost).

Also compute top 10 pairwise feature interactions: for top 20 features by target MI, compute pairwise MI (feature_A vs feature_B). Sort by MI descending. These are the interaction candidates Feature Factory should prioritize.

Output: `{"mutual_info": {"target_mi": [{col: float}], "top_interactions": [{"feature_a": str, "feature_b": str, "pairwise_mi": float}]}}`

**Section 3 — Multicollinearity (VIF):**

Compute VIF for all numeric columns. Use `from statsmodels.stats.outliers_influence import variance_inflation_factor`. If statsmodels unavailable, fall back to manual computation: `VIF_j = 1 / (1 - R²_j)` where R²_j is from regressing feature j on all other features.

**CRITICAL: Handle singular matrices.** If the feature matrix is singular (happens with perfectly correlated features), catch the `LinAlgError` and set VIF = infinity for those columns. Log which columns caused the singularity.

If >50 numeric columns: compute VIF only for top 50 by variance.

Flag columns with VIF > 10 as multicollinear.

Output: `{"vif_report": {"scores": {col: float}, "high_vif_columns": [col], "threshold": 10.0}}`

**Section 4 — Modality Flags:**

For each numeric feature, check if the distribution is multimodal (from Section 1). Collect all multimodal features into a list.

Also check the TARGET for multimodality. A multimodal target is a strong signal for the Problem Reframer (suggests segmented modeling).

Output: `{"modality_flags": ["feature_a", "feature_c", "target"]}`

### Part C: insights_summary generation

After Parts A and B complete, call `llm_call()` to generate the insights_summary paragraph. This is NOT generated in the sandbox — it's an LLM synthesis call.

**Prompt:**
```
You are a senior data scientist analyzing a competition dataset.
Generate a single paragraph (150-250 words) summarizing the most 
important findings for downstream feature engineering and modeling.

REQUIREMENTS:
- Reference exact numbers (skewness values, MI scores, VIF scores)
- Name specific columns
- State actionable implications ("feature X has MI 0.35 with target, 
  prioritize features derived from X")
- Flag risks ("features A and B have VIF 45, their interaction will 
  be redundant")
- NO generic statements like "this is an interesting dataset"

DATA:
Target: {target_column} ({competition_type})
Top 5 features by MI: {top_5_mi}
High VIF columns: {high_vif}
Multimodal features: {modality_flags}
Skewed features (|skew| > 2): {skewed}
Class balance: {class_balance}
Missing value summary: {missing_summary}
Duplicate rows: {duplicates}
Leakage risk: {leakage_fingerprint}
```

Store the response as `eda_insights_summary`. This paragraph gets injected into the system prompts of Feature Factory, Creative Hypothesis, Problem Reframer, and Red Team Critic.

### State return

```python
return state.validated_update("eda_agent", {
    "eda_report": v1_report,  # 8 original keys preserved
    "eda_insights_summary": insights_paragraph,
    "eda_mutual_info": mi_results,
    "eda_vif_report": vif_results,
    "eda_modality_flags": modality_list,
    "eda_quick_baseline_importance": baseline_importance,  # From Commit 3
    "eda_plots_paths": plot_paths,  # From Commit 3
    "eda_plots_delivered": False,  # Set True after HITL delivery in Commit 3
})
```

### Contract tests: tests/contracts/test_eda_v2_contract.py

NEW file — does NOT modify any v1 contract tests.

Create a test fixture:
```python
@pytest.fixture
def mock_eda_data(tmp_path):
    """1000-row dataset with known properties for testing."""
    np.random.seed(42)
    df = pl.DataFrame({
        "id": range(1000),
        "normal_feat": np.random.normal(0, 1, 1000).tolist(),
        "skewed_feat": np.random.exponential(2, 1000).tolist(),  # Right-skewed
        "correlated_a": np.random.normal(0, 1, 1000).tolist(),
        "correlated_b": None,  # Will be set to correlated_a + small noise
        "multimodal_feat": None,  # Will be bimodal
        "high_card_cat": [f"cat_{i}" for i in range(1000)],
        "target": [0, 1] * 500,
    })
    # correlated_b = correlated_a + noise (VIF should be high)
    # multimodal_feat = mixture of two normals
    # ... set up the fixture with known statistical properties
    df.write_parquet(tmp_path / "clean_train.parquet")
    return tmp_path, df
```

Tests:
1. `test_insights_summary_nonempty` — eda_insights_summary is a non-empty string with >50 chars
2. `test_insights_summary_mentions_target` — insights_summary contains the target column name
3. `test_insights_summary_contains_numbers` — insights_summary contains at least 3 numeric values (regex `r'\d+\.\d+'`)
4. `test_mutual_info_has_all_numeric_features` — target_mi list has an entry for every numeric column
5. `test_mutual_info_values_nonnegative` — all MI values >= 0
6. `test_top_interactions_present` — top_interactions list exists, each entry has feature_a, feature_b, pairwise_mi
7. `test_vif_has_all_numeric_features` — VIF scores dict has entry for every numeric column
8. `test_vif_threshold_is_10` — threshold field is 10.0
9. `test_high_vif_detected` — correlated_a and correlated_b appear in high_vif_columns (they're correlated)
10. `test_vif_handles_singular_matrix` — add a perfectly duplicated column, verify no crash, VIF = inf for that column
11. `test_modality_flags_is_list` — eda_modality_flags is a list (may be empty)
12. `test_multimodal_feature_detected` — multimodal_feat appears in modality_flags
13. `test_skewed_feature_gets_transform_recommendation` — skewed_feat gets recommended_transform != "none"
14. `test_normal_feature_gets_none_transform` — normal_feat gets recommended_transform == "none"
15. `test_v1_keys_still_present` — eda_report dict has all 8 original v1 keys
16. `test_state_ownership` — verify eda_agent writes only to eda_* fields

---

## COMMIT 3: EDA Artifact Export (tools/eda_plots.py)

### What this adds

7 diagnostic plots generated in the sandbox, saved as PNG files, delivered to the operator via HITL at Milestone 1. Closes the real-time reasoning gap with Claude Code.

### The plot generation function

```python
def generate_eda_plots(
    data_path: str,
    target_col: str,
    competition_type: str,
    mi_scores: dict,
    session_dir: str,
    max_sample_rows: int = 10000,
) -> list[dict]:
    """
    Generate 7 diagnostic plots. Returns list of 
    {"path": str, "name": str, "caption": str}.
    
    All plots generated via run_in_sandbox() using matplotlib/seaborn.
    """
```

Build a SINGLE code block that generates all 7 plots. Execute via `run_in_sandbox()` with `agent_name="eda_plots"`. The code:
- Loads data from `data_path`
- If rows > `max_sample_rows`: stratified sample to `max_sample_rows` (stratify on target for classification)
- Generates each plot with `matplotlib.pyplot` and `seaborn`
- Saves each as PNG at 150 DPI with `figsize=(10, 6)`, `fontsize=12`, `linewidth=2`
- Uses a consistent style: `plt.style.use('seaborn-v0_8-whitegrid')` or fall back to `'ggplot'`

**Plot 1 — Target Distribution (`target_distribution.png`):**
- Classification: `sns.countplot()` with value counts on bars. If >10 classes, group rare classes (<2%) into "Other".
- Regression: `sns.histplot()` with KDE overlay. Mark mean and median with vertical lines.
- Caption: "Target distribution. {type}. Balance: {majority_pct}%/{minority_pct}%." or "Target distribution. Range: [{min:.2f}, {max:.2f}]. Mean: {mean:.2f}, Median: {median:.2f}."

**Plot 2 — Correlation Heatmap (`correlation_heatmap.png`):**
- Select top 15 features by absolute correlation with target. Compute pairwise correlation matrix.
- `sns.heatmap()` with `annot=True`, `fmt=".2f"`, `cmap="RdBu_r"`, `center=0`.
- Caption: "Top 15 features by target correlation. Red = positive, Blue = negative."

**Plot 3 — Missing Value Heatmap (`missing_values.png`):**
- Compute null percentage per column. Sort descending. Show only columns with >0% null.
- Horizontal bar chart: `plt.barh()` with percentage labels.
- If no nulls: skip this plot, add a note "No missing values."
- Caption: "Missing values by column. {n_cols} columns have nulls."

**Plot 4 — Top Feature Distributions (`feature_distributions.png`):**
- Select top 6 features by MI score.
- 2×3 subplot grid.
- Classification: `sns.histplot()` with `hue=target_col` and `kde=True` for each feature.
- Regression: `sns.histplot()` with color gradient by target value.
- Caption: "Top 6 features by mutual information, colored by target."

**Plot 5 — Feature Importance (`feature_importance.png`):**
- Train a quick LightGBM with 100 rounds, default params, `verbose=-1`.
- Extract `model.feature_importances_`. Sort descending. Plot top 20.
- Horizontal bar chart with importance values.
- Store importance dict as `eda_quick_baseline_importance`.
- Caption: "LightGBM 100-round baseline. Top 20 feature importances."

**Plot 6 — Target vs Top Features (`target_vs_features.png`):**
- Top 3 features by MI.
- 1×3 subplot grid.
- Classification: box plots of each feature grouped by target class.
- Regression: scatter plots with target on y-axis, feature on x-axis, `alpha=0.3`.
- Caption: "Target vs top 3 features. Look for separability (classification) or linearity (regression)."

**Plot 7 — Data Quality Summary (`data_quality.png`):**
- A single visual with text annotations showing:
  - Row count, column count
  - Duplicate row percentage
  - Total null percentage
  - Number of constant columns
  - Number of high-cardinality categoricals (>100 unique)
  - Memory usage estimate
- Use `plt.text()` to create an infographic-style summary.
- Caption: "Data quality snapshot."

### Plot delivery via HITL

After generating all plots, build the Milestone 1 message:

```python
# Build caption summary (always sent, even if operator doesn't open images)
caption_block = "\n".join([f"  {p['name']}: {p['caption']}" for p in plots])

milestone_1_msg = f"""📋 COMPETITION BRIEF + EDA
{competition_brief_text}

📊 EDA PLOTS ({len(plots)} generated):
{caption_block}

Reply with domain knowledge, feature ideas, or /continue"""

# Emit as CHECKPOINT
response = emit_to_operator(milestone_1_msg, level="CHECKPOINT", data={"plots": plot_paths})
```

For CLI: print file paths so operator can open them locally.
For Telegram: send each plot as a photo message using the Telegram `sendPhoto` API. The TelegramAdapter needs a `send_photo(photo_path, caption)` method — add it to the adapter.

### Quick baseline importance (separate sandbox run)

The LightGBM baseline for Plot 5 also stores its feature importances in state. This gives downstream agents (especially Feature Factory) immediate signal about which features the model finds useful BEFORE the full feature engineering begins.

```python
# Inside the plot generation code:
import lightgbm as lgb

# Prepare data
feature_cols = [c for c in df.columns if c not in [target_col, id_col]]
X = df.select(feature_cols).to_numpy()
y = df[target_col].to_numpy()

# Quick train
params = {"n_estimators": 100, "verbose": -1, "random_state": 42}
if competition_type in ["binary", "multiclass"]:
    model = lgb.LGBMClassifier(**params)
else:
    model = lgb.LGBMRegressor(**params)
model.fit(X, y)

# Extract importance
importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
# Save as JSON for state consumption
```

### TelegramAdapter extension

Add to `tools/operator_channel.py`:
```python
class TelegramAdapter(ChannelAdapter):
    # ... existing methods ...
    
    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send an image file as a Telegram photo message."""
        try:
            with open(photo_path, "rb") as photo:
                response = requests.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": caption[:1024]},
                    files={"photo": photo},
                    timeout=30,
                )
            return response.status_code == 200
        except Exception as e:
            self._log_warning(f"Failed to send photo: {e}")
            return False
```

### Contract tests: tests/contracts/test_eda_plots_contract.py

1. `test_7_plots_generated` — all 7 PNG files created in session directory
2. `test_plot_files_are_valid_png` — each file starts with PNG magic bytes `\x89PNG`
3. `test_plot_file_sizes_reasonable` — each file between 10KB and 5MB (catches empty files and exploded renders)
4. `test_captions_present` — each plot dict has non-empty "caption" string
5. `test_quick_baseline_importance_populated` — eda_quick_baseline_importance has entries for feature columns
6. `test_importance_values_nonnegative` — all importance values >= 0
7. `test_sampling_applied_on_large_data` — create 50K-row fixture, verify only 10K rows used for plotting (check code or mock)
8. `test_no_crash_on_all_numeric` — dataset with only numeric columns generates plots without error
9. `test_no_crash_on_many_classes` — 50-class target doesn't crash bar chart (groups into "Other")
10. `test_no_crash_on_no_nulls` — missing value plot gracefully skips when no nulls exist
11. `test_correlation_heatmap_max_15_features` — even with 100 features, heatmap shows only top 15
12. `test_milestone_1_emitted` — mock emit_to_operator, verify CHECKPOINT emitted after plot generation

---

## COMMIT 4: Domain Research Engine (agents/domain_research.py)

### What this does

Professor sees "column HbA1c_level, float64, range 3.5-14.2" and builds groupby aggregations. A doctor sees it and knows: threshold 6.5 = diabetes. The Domain Research Engine bridges this gap by acquiring domain knowledge and structuring it for downstream agents.

### The LangGraph node function

```python
def domain_research(state: ProfessorState) -> dict:
    """
    Acquires domain knowledge via 4 channels.
    Runs IN PARALLEL with Competition Intel (zero added latency).
    
    Reads: competition_name, data_schema, eda_insights_summary (if available)
    Writes: domain_brief, domain_classification, domain_templates_applied
    Emits: STATUS (domain found), STATUS (channels completed)
    """
```

### Stage 1 — Domain Classification (two passes)

**Pass 1: Deterministic keyword scoring.** Build a dict of 8 domain keyword lists:

```python
DOMAIN_KEYWORDS = {
    "healthcare": ["patient", "diagnosis", "icd", "hba1c", "glucose", "bmi", "cholesterol",
                    "blood_pressure", "heart_rate", "medication", "clinical", "hospital",
                    "disease", "symptom", "treatment", "lab_result", "medical"],
    "finance": ["transaction", "credit", "debit", "balance", "loan", "interest_rate",
                "portfolio", "stock", "revenue", "profit", "loss", "risk_score",
                "fraud", "default", "payment", "account", "bank"],
    "retail": ["product", "customer", "purchase", "order", "price", "quantity",
               "sku", "category", "store", "sales", "inventory", "discount",
               "cart", "session", "click", "conversion"],
    "geospatial": ["latitude", "longitude", "lat", "lng", "zip_code", "postal",
                    "city", "state", "country", "region", "distance", "elevation",
                    "coordinate", "location", "address", "geo"],
    "energy": ["consumption", "kwh", "voltage", "current", "power", "solar",
               "wind", "temperature", "humidity", "meter", "grid", "load"],
    "manufacturing": ["sensor", "vibration", "pressure", "rpm", "cycle",
                       "defect", "quality", "machine", "assembly", "tolerance"],
    "transport": ["trip", "route", "vehicle", "speed", "distance", "pickup",
                   "dropoff", "eta", "fare", "driver", "passenger"],
    "natural_science": ["species", "habitat", "concentration", "ph", "wavelength",
                         "spectrum", "sample", "experiment", "observation"],
}
```

Score each domain: count how many of its keywords appear in (a) column names (case-insensitive partial match) and (b) competition_name. The domain with the highest score wins. If top score < 3: `primary_domain = "general"`.

**Pass 2: LLM narrowing (only if domain != "general").** Call `llm_call()`:

```
Given this competition in the {primary_domain} domain:
Competition: {competition_name}
Columns: {column_names}

What is the specific sub-domain? Reply with ONLY the sub-domain 
description in one sentence. Example: "diabetes screening from 
routine blood panels" or "credit card fraud detection in 
e-commerce transactions."
```

Store as `domain_sub_classification`.

### Stage 2 — Knowledge Acquisition (4 channels)

**Channel 1 — LLM Domain Reasoning (always runs):**

Call `llm_call()` with a structured prompt requesting JSON output:

```
You are a senior {primary_domain} practitioner with 15 years experience.
The competition is: {competition_name}
Sub-domain: {sub_classification}
Columns and types: {schema}

Respond with a JSON object containing:
{
    "column_semantics": {
        "column_name": {
            "meaning": "Real-world meaning of this column",
            "valid_range": [min, max] or null,
            "clinical_thresholds": [{"value": 6.5, "meaning": "diabetes threshold"}] or [],
            "domain_importance": "high" | "medium" | "low"
        }
    },
    "known_relationships": [
        {"features": ["col_a", "col_b"], "relationship": "Description", "feature_recipe": "col_a / col_b"}
    ],
    "feature_recipes": [
        {
            "name": "recipe_name",
            "formula": "Exact Polars expression or description",
            "rationale": "Why this feature helps",
            "source_columns": ["col_a", "col_b"]
        }
    ],
    "domain_constraints": [
        {"type": "impossible_value", "column": "age", "constraint": "must be > 0 and < 150"}
    ],
    "evaluation_context": "What this metric means in domain terms"
}

RULES:
- Only reference columns that exist in the schema
- Feature recipes must use columns that exist
- Be specific — exact thresholds, exact formulas, exact ranges
- If you're not sure about a column's meaning, say "unknown"
```

Parse the JSON response. Validate: every column in `column_semantics` exists in the schema. Every `source_columns` entry in `feature_recipes` exists. Remove any entries referencing non-existent columns.

**Channel 2 — Web Search (if web search available):**

This depends on whether Professor has web search capability. If `tools.web_search` exists:
- Search for `"{sub_classification} key predictive factors"`
- Search for `"{sub_classification} feature engineering kaggle"`
- Extract key findings from top 3 results
- Cross-reference with Channel 1: validate or contradict LLM's thresholds and relationships

If web search is not available: skip. Log: "Channel 2 (web search) skipped — not available."

**Channel 3 — Academic papers (if arxiv accessible):**

If arxiv search is available:
- Search for `"{sub_classification} prediction model"`
- Extract methodology and key features from abstracts
- Focus on: which features were most predictive? What model worked best?

If not available: skip.

**Channel 4 — Past Competition Memory (if ChromaDB has entries):**

If `tools.chromadb_memory` has entries:
- Query with `sub_classification` text
- Retrieve validated patterns from similar past competitions
- These are HIGH confidence because they've been Wilcoxon-validated

If ChromaDB is empty or unavailable: skip.

**Graceful degradation:** Only Channel 1 is required. Channels 2-4 enrich and validate. If any channel fails (exception, timeout, not available), it's skipped with a log message. Pipeline continues with whatever channels succeeded.

### Stage 3 — Structuring the Domain Brief

Merge results from all channels into `domain_brief`:

```python
domain_brief = {
    "primary_domain": primary_domain,
    "sub_classification": sub_classification,
    "column_semantics": channel_1["column_semantics"],
    "known_relationships": channel_1["known_relationships"],
    "feature_recipes": [],  # Merged from all channels
    "domain_constraints": channel_1["domain_constraints"],
    "evaluation_context": channel_1["evaluation_context"],
    "domain_summary": "",  # Generated below
}
```

For `feature_recipes`: merge from all channels. Each recipe gets a `source` field: "llm_reasoning", "web_research", "academic_paper", "past_memory". Recipes corroborated by 2+ channels get `verified: True`. Recipes from Channel 1 only get `verified: False`.

For `domain_summary`: a 2-3 sentence summary suitable for LLM prompts: "This is a {sub_classification} problem. Key domain features include {top_3_recipes}. Critical thresholds: {top_thresholds}."

### Pre-built domain templates

Build a dict `DOMAIN_TEMPLATES` with pre-validated knowledge for the most common Kaggle domains:

```python
DOMAIN_TEMPLATES = {
    "diabetes_screening": {
        "column_patterns": {"hba1c": ..., "glucose": ..., "bmi": ...},
        "thresholds": [
            {"column": "hba1c", "value": 6.5, "meaning": "diabetes diagnosis"},
            {"column": "glucose", "value": 126, "meaning": "fasting glucose diabetes"},
            {"column": "bmi", "value": 30, "meaning": "obesity threshold"},
        ],
        "recipes": [
            {"name": "hba1c_glucose_ratio", "formula": "pl.col('hba1c') / pl.col('glucose')", 
             "rationale": "Insulin resistance proxy"},
        ],
    },
    "house_price_prediction": { ... },
    "credit_fraud_detection": { ... },
    "customer_churn": { ... },
    "time_series_forecasting": { ... },
}
```

Start with 3-5 templates. They grow over time as Professor competes in more competitions. If the sub_classification matches a template key (fuzzy match), load the template and merge with Channel 1 output. Template entries get `source: "pre_built_template"` and `verified: True`.

### State return

```python
return state.validated_update("domain_research", {
    "domain_brief": domain_brief,
    "domain_classification": primary_domain,
    "domain_templates_applied": applied_template_names,
})
```

### Contract tests: tests/contracts/test_domain_research_contract.py

Mock `llm_call()` to return a known JSON response for all tests. Mock web search as unavailable.

1. `test_healthcare_classification` — columns ["patient_id", "hba1c", "glucose", "bmi", "target"] → primary_domain = "healthcare"
2. `test_finance_classification` — columns ["transaction_id", "amount", "credit_score", "fraud"] → primary_domain = "finance"
3. `test_general_classification` — columns ["col_1", "col_2", "col_3", "target"] with score < 3 → "general"
4. `test_domain_brief_has_required_sections` — all 6 sections present in domain_brief
5. `test_column_semantics_reference_existing_columns` — every column in semantics exists in schema
6. `test_feature_recipes_reference_existing_columns` — every source_column in recipes exists in schema
7. `test_invalid_columns_removed` — LLM returns a recipe referencing "nonexistent_col" → recipe is removed
8. `test_channel_1_failure_doesnt_crash` — mock llm_call to raise exception → domain_brief is minimal but valid
9. `test_general_domain_returns_minimal_brief` — domain "general" returns brief with empty recipes and "unknown" semantics
10. `test_domain_summary_nonempty` — domain_summary is non-empty string
11. `test_template_applied_for_healthcare` — healthcare competition with hba1c column → template applied
12. `test_state_ownership` — domain_research writes only to domain_* fields
13. `test_cost_governor_respected` — verify llm_call is called (mock), not raw API calls
14. `test_recipes_have_source_field` — every recipe has a "source" field

---

## COMMIT 5: Distribution Shift Detector (agents/shift_detector.py)

### What this does

Detects when the test set comes from a different distribution than the training set. This is common in Kaggle (temporal splits, geographic splits, different hospitals). If shift is severe, the model's CV score won't correlate with LB score.

### The LangGraph node function

```python
def shift_detector(state: ProfessorState) -> dict:
    """
    Detects train/test distribution shift.
    
    Reads: clean_data_path, clean_test_path, data_schema
    Writes: shift_report, shift_severity, shift_sample_weights
    Emits: STATUS (result)
    """
```

### Detection approach

Generate a code block executed via `run_in_sandbox()`:

**Step 1 — Adversarial Validation:** 
- Label all training rows as 0, all test rows as 1
- Train LightGBM to distinguish train from test
- If AUC > 0.7: shift is detectable (the model can tell train from test apart)
- If AUC > 0.85: severe shift
- Extract feature importances: which features are most different between train and test?

**Step 2 — Per-Feature KS + PSI Tests:**
- For each numeric feature: Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`)
- For each categorical feature: Population Stability Index (PSI)
- KS p-value < 0.01 or PSI > 0.25: significant shift on that feature

**Step 3 — Sample Weights (if shift detected):**
- If adversarial AUC > 0.7: compute sample weights for training data
- Use adversarial model's predicted probabilities: `weight = p(test) / p(train)`
- Clip weights to [0.1, 10.0] to prevent extreme values
- These weights can be used by ML Optimizer for importance-weighted training

### Severity classification

- `shift_severity = "none"`: adversarial AUC < 0.6
- `shift_severity = "low"`: adversarial AUC 0.6-0.7
- `shift_severity = "medium"`: adversarial AUC 0.7-0.85
- `shift_severity = "high"`: adversarial AUC > 0.85

### State return

```python
return state.validated_update("shift_detector", {
    "shift_report": {
        "adversarial_auc": auc_score,
        "top_shifted_features": top_features,
        "ks_results": ks_dict,
        "psi_results": psi_dict,
    },
    "shift_severity": severity,
    "shift_sample_weights": weights_list if severity in ["medium", "high"] else [],
})
```

### Contract tests: tests/contracts/test_shift_detector_contract.py

1. `test_no_shift_detected` — identical train/test distributions → severity "none" and AUC < 0.6
2. `test_severe_shift_detected` — train and test from completely different distributions → severity "high"
3. `test_sample_weights_produced_on_shift` — medium/high severity produces non-empty weights list
4. `test_sample_weights_clipped` — all weights between 0.1 and 10.0
5. `test_no_weights_on_no_shift` — severity "none" → empty weights list
6. `test_shift_report_has_required_fields` — adversarial_auc, top_shifted_features, ks_results all present
7. `test_ks_results_per_numeric_feature` — every numeric feature has a KS test result
8. `test_handles_no_test_data_gracefully` — if test data is missing/empty, returns severity "unknown" and doesn't crash
9. `test_state_ownership` — shift_detector writes only to shift_* fields

---

## COMMIT 6: Integration Test

### tests/contracts/test_layer1_integration.py

Create a comprehensive test that runs all 4 Layer 1 components in sequence on a mock dataset:

```python
def test_layer1_full_sequence(mock_competition_dir):
    """
    Run: preflight → eda_agent → domain_research → shift_detector
    Verify state flows correctly between them.
    """
    state = ProfessorState(
        session_id="test-layer1",
        competition_name="test-diabetes-prediction",
        raw_data_path=str(mock_competition_dir),
        target_column="target",
    )
    
    # Mock LLM calls and sandbox for speed
    with patch("tools.llm_provider.llm_call") as mock_llm, \
         patch("tools.sandbox.run_in_sandbox") as mock_sandbox:
        
        # Configure mocks to return valid responses
        mock_llm.return_value = {"text": mock_insights, "reasoning": "", ...}
        mock_sandbox.return_value = {"success": True, "stdout": mock_output, ...}
        
        # Run sequence
        updates = run_preflight_checks(state)
        state = state.copy(update=updates)
        assert state.preflight_passed == True
        
        updates = eda_agent(state)
        state = state.copy(update=updates)
        assert state.eda_insights_summary != ""
        assert len(state.eda_mutual_info.get("target_mi", [])) > 0
        
        updates = domain_research(state)
        state = state.copy(update=updates)
        assert state.domain_classification != ""
        assert "column_semantics" in state.domain_brief
        
        updates = shift_detector(state)
        state = state.copy(update=updates)
        assert state.shift_severity in ["none", "low", "medium", "high"]
    
    # Verify no ownership violations occurred
    # Verify all state fields have correct types
    # Verify no state mutations from wrong agents
```

Also test:
- `test_preflight_failure_doesnt_block_eda` — if preflight has advisory warnings, EDA still runs
- `test_domain_research_uses_eda_insights` — if eda_insights_summary exists when domain_research runs, it's included in the domain prompt
- `test_all_hitl_emissions` — verify at least: 1 CHECKPOINT from preflight (Milestone 0), 1 CHECKPOINT from EDA (Milestone 1), 2+ STATUS from domain_research

---

## WHAT NOT TO DO

- Do NOT modify any Layer 0 files (state.py, sandbox.py, operator_channel.py, cost_governor.py, metric_gate.py). They're built and tested.
- Do NOT modify v1 EDA contract tests. Create test_eda_v2_contract.py as a NEW file.
- Do NOT use Pandas. All data operations in Polars. Check POLARS.md.
- Do NOT make real LLM calls in tests. Mock everything.
- Do NOT generate plots interactively (no plt.show()). Only plt.savefig().
- Do NOT let any component crash the pipeline. If domain research fails, return minimal brief. If EDA plots fail, skip plots and continue with stats. If shift detector fails, return severity "unknown".
- Do NOT compute MI or VIF on more than 50 features. Cap at top 50 by variance for performance.
- Do NOT let the insights_summary be generic. The prompt demands exact numbers and specific column names. If the LLM returns generic text, log a warning but accept it — fixing the prompt is iteration, not a code bug.