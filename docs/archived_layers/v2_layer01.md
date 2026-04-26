# BUILD PROMPT — Layer 1 Continuation: Shift Detector (CORRECTED) + Complexity-Gated Depth
# Feed to Gemini CLI with: @PROFESSOR.md @STATE.md @SANDBOX.md @HITL.md @POLARS.md @CONTRACTS.md

---

## IMPORTANT — CORRECTIONS TO PREVIOUS LAYER 1 PROMPT

The previous Layer 1 prompt (BUILD_PROMPT_LAYER1.md) contained errors in Commit 5 (Shift Detector). 
**DO NOT use the Commit 5 from that prompt.** Use THIS prompt instead. The errors were:

1. WRONG: Used LightGBM for adversarial validation → CORRECT: LogisticRegression(max_iter=200, solver="lbfgs")
2. WRONG: Severity levels none/low/medium/high → CORRECT: clean/mild/severe (3 levels, not 4)
3. WRONG: Adversarial AUC thresholds 0.6/0.7/0.85 → CORRECT: 0.55/0.65 (clean < 0.55, mild 0.55-0.65, severe > 0.65)
4. WRONG: State fields shift_severity/shift_sample_weights → CORRECT: shift_severity/shifted_features/sample_weights_path + shift_report_path
5. MISSING: Per-feature remediation (train with/without each drifted feature to decide REMOVE vs KEEP)
6. MISSING: Jensen-Shannon divergence for categorical features
7. MISSING: PSI computation alongside KS test with dual threshold requirement

These are architectural decisions that exist for specific reasons. LogisticRegression is used because RandomForest memorizes high-cardinality categoricals and produces AUC 0.56-0.58 on noise — the same false-positive problem fixed in Critic Fix 2A. The dual threshold (KS + PSI) prevents false positives from KS test's oversensitivity on large datasets.

---

## COMMIT PLAN (2 commits, continuing from Layer 1)

```
Commit 5 (REPLACES previous Commit 5): agents/shift_detector.py + tests/contracts/test_shift_detector_contract.py
Commit 7: graph/depth_router.py + tests/contracts/test_depth_router_contract.py
```

Commit 6 (integration test from previous prompt) should run AFTER both of these are done.
Updated integration test should include shift_detector and depth_router.

---

## COMMIT 5 (CORRECTED): Distribution Shift Detector (agents/shift_detector.py)

### What this prevents

Training a model on train distribution that collapses on test distribution. The #1 silent killer in competitions with temporal, geographic, or adversarial splits. v1's Critic catches this AFTER training (wasting 10-20 minutes). The Shift Detector catches it BEFORE training and injects remediation (sample weights, feature flagging) into the pipeline.

### Pipeline position

After `data_engineer`, before `eda_agent`. Needs only `clean_data_path` and test data. No other v2 dependencies.

### The LangGraph node function

```python
def shift_detector(state: ProfessorState) -> dict:
    """
    Detects train/test distribution shift BEFORE any model training.
    
    Reads: clean_data_path, clean_test_path, data_schema
    Writes: shift_report, shift_report_path, shift_severity, 
            shifted_features, sample_weights_path
    Emits: STATUS (result summary)
    
    Pipeline position: after data_engineer, before eda_agent
    """
```

### The algorithm — generate as a SINGLE code block for run_in_sandbox()

**Step 1 — Adversarial Classifier (global shift detection):**

```
Combine train rows (label=0) and test rows (label=1)
Train LogisticRegression(max_iter=200, random_state=42, solver="lbfgs")
3-fold cross-validation on is_train prediction
adversarial_auc = mean(fold_aucs)
```

**WHY LogisticRegression, NOT RandomForest or LightGBM:**
RF and LightGBM memorize high-cardinality categorical leaf patterns on shuffled data, producing AUC 0.56-0.58 on clean data where there IS no shift. This is the exact same false-positive problem we fixed in Critic Fix 2A. LogisticRegression doesn't have this problem because it doesn't create categorical splits. This is a deliberate architectural choice, not a simplification.

Implementation details:
- Combine train and test into one dataframe. Add column `_is_test` (0 for train, 1 for test).
- Select only numeric features (LR can't handle categoricals directly). For categorical features with < 20 unique values, one-hot encode. For > 20 unique, skip (they'd dominate and cause false positives).
- Standardize features: `from sklearn.preprocessing import StandardScaler`. Fit on combined data.
- `from sklearn.linear_model import LogisticRegression`
- `from sklearn.model_selection import cross_val_predict, StratifiedKFold`
- 3-fold stratified CV: `cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
- Get OOF predictions: `oof_probs = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]`
- Compute AUC: `from sklearn.metrics import roc_auc_score; adversarial_auc = roc_auc_score(y, oof_probs)`
- Extract feature importances from a final LR fit on all data: `lr.fit(X, y); importances = abs(lr.coef_[0])`

**Step 2 — Per-Feature Drift Tests:**

For each NUMERIC feature:
```python
from scipy.stats import ks_2samp
ks_stat, ks_pvalue = ks_2samp(train_col.drop_nulls().to_numpy(), test_col.drop_nulls().to_numpy())
psi = _compute_psi(train_col, test_col, bins=10)
```

PSI computation:
```python
def _compute_psi(train_col, test_col, bins=10):
    """Population Stability Index — industry standard for drift detection."""
    # Bin the training data into deciles
    breakpoints = np.percentile(train_col.drop_nulls().to_numpy(), np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Count proportions in each bin
    train_counts = np.histogram(train_col.drop_nulls().to_numpy(), bins=breakpoints)[0]
    test_counts = np.histogram(test_col.drop_nulls().to_numpy(), bins=breakpoints)[0]
    
    # Avoid division by zero
    train_pct = (train_counts + 1) / (sum(train_counts) + bins)
    test_pct = (test_counts + 1) / (sum(test_counts) + bins)
    
    psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
    return float(psi)
```

For each CATEGORICAL feature:
```python
def _jensen_shannon_divergence(train_col, test_col):
    """Jensen-Shannon divergence for categorical distribution comparison."""
    from scipy.spatial.distance import jensenshannon
    
    # Get value counts as probability distributions
    all_values = set(train_col.drop_nulls().to_list() + test_col.drop_nulls().to_list())
    train_counts = train_col.drop_nulls().value_counts()
    test_counts = test_col.drop_nulls().value_counts()
    
    # Build aligned probability vectors
    train_probs = []
    test_probs = []
    for val in sorted(all_values):
        train_probs.append(train_counts.filter(pl.col(col_name) == val).select("count").item(0, default=0))
        test_probs.append(test_counts.filter(pl.col(col_name) == val).select("count").item(0, default=0))
    
    # Normalize
    train_probs = np.array(train_probs, dtype=float)
    test_probs = np.array(test_probs, dtype=float)
    train_probs = (train_probs + 1) / (train_probs.sum() + len(all_values))  # Laplace smoothing
    test_probs = (test_probs + 1) / (test_probs.sum() + len(all_values))
    
    return float(jensenshannon(train_probs, test_probs) ** 2)  # Squared for comparability
```

**Drift flag criteria — DUAL THRESHOLD for numeric (prevents false positives):**
```
Numeric feature flagged as DRIFTED if:
    ks_pvalue < 0.001 AND psi > 0.25
    
    WHY both: KS-test alone is oversensitive on large datasets (n > 50K) where
    tiny distributional differences reach p < 0.001. PSI > 0.25 is the industry 
    standard for "significant drift." Requiring BOTH prevents false positives 
    from statistical power alone.

Categorical feature flagged as DRIFTED if:
    js_divergence > 0.1
```

**Step 3 — Severity Classification:**

```
adversarial_auc < 0.55  → severity = "clean"
adversarial_auc 0.55-0.65 → severity = "mild"  
adversarial_auc > 0.65  → severity = "severe"
```

Three levels, not four. The thresholds are conservative: 0.55 means the LR can barely distinguish train from test (essentially random with slight bias). 0.65 means there's a real distributional difference the model can exploit.

**Step 4 — Remediation Strategy (MILD or SEVERE only):**

For each DRIFTED feature:
```
Train quick LightGBM(n_estimators=50) WITH the feature → score_with
Train quick LightGBM(n_estimators=50) WITHOUT the feature → score_without
If score_without >= score_with: recommendation = "remove"
Else: recommendation = "keep_with_weighting"
```

Sample weights (if SEVERE):
```
Train the adversarial LR on full data
weights_per_row = P(test | row) / P(train | row) for each training row
Clip weights to [0.1, 10.0]
Save to outputs/{session_id}/sample_weights.parquet
```

**WHY clip to [0.1, 10.0]:** Extreme density ratios (100:1) cause training instability — single rows dominate the loss. The range [0.1, 10.0] allows 100x variation which is sufficient for most shift corrections. The Wilcoxon gate in ML Optimizer later verifies that weighted training actually beats unweighted. If it doesn't, weights are discarded.

**Step 5 — Write shift_report.json:**

```python
shift_report = {
    "adversarial_auc": float(adversarial_auc),
    "severity": severity,
    "drifted_features": [
        {
            "feature": feature_name,
            "drift_type": "ks" | "js",  # numeric vs categorical
            "ks_stat": float,      # for numeric
            "ks_pvalue": float,    # for numeric
            "psi": float,          # for numeric
            "js_divergence": float, # for categorical
            "recommendation": "remove" | "keep_with_weighting",
        }
    ],
    "n_drifted": int,
    "n_total_features": int,
    "drift_ratio": float,  # n_drifted / n_total_features
    "sample_weights_generated": bool,
    "sample_weights_path": str,  # path to parquet file, or ""
    "remediation_strategy": "none" | "flag_only" | "weight_and_flag",
    "checked_at": str,  # ISO timestamp
}
```

### Downstream injection — who reads what

| Agent | Reads | Effect |
|---|---|---|
| validation_architect | shift_severity | SEVERE → consider adversarial-split-aware CV |
| eda_agent | shifted_features | Highlight drifted features in EDA report |
| feature_factory | shifted_features | Avoid building interactions on drifted columns |
| ml_optimizer | sample_weights_path | Pass weights to model.fit() if present |
| red_team_critic | shift_severity | Cross-reference with Critic Vector 1c (Fix 2A) |

### LangGraph wiring

```python
graph.add_node("shift_detector", shift_detector)
graph.add_edge("data_engineer", "shift_detector")
graph.add_edge("shift_detector", "eda_agent")
```

SEVERE does NOT auto-halt the pipeline. Default is proceed with sample weighting. The only halt condition is if the weight computation itself fails (numerical issues, degenerate weights) — in that case, emit STATUS warning and continue WITHOUT weights.

### State return

```python
return state.validated_update("shift_detector", {
    "shift_report": shift_report,
    "shift_report_path": f"outputs/{state.session_id}/shift_report.json",
    "shift_severity": severity,  # "clean" | "mild" | "severe"
    "shifted_features": [f["feature"] for f in drifted_features],
    "sample_weights_path": weights_path if severity == "severe" else "",
})
```

### IMPORTANT: State fields differ from STATE.md

The current STATE.md has `shift_sample_weights: list` but the architecture specifies `sample_weights_path: str` (a file path, not the weights array — we never put raw arrays in state). Update STATE.md to match:

```python
# CORRECT state fields for shift detector:
shift_report: dict = Field(default_factory=dict)
shift_report_path: str = ""
shift_severity: str = "unchecked"     # "unchecked" | "clean" | "mild" | "severe"
shifted_features: list = Field(default_factory=list)   # List of feature names
sample_weights_path: str = ""          # Path to parquet file, not the array itself
```

If `shift_sample_weights: list` exists in STATE.md from earlier, REPLACE it with `sample_weights_path: str`. Raw arrays in state violate the "state pointers not data" rule.

### Contract tests: tests/contracts/test_shift_detector_contract.py

Create fixtures with KNOWN shift properties:

```python
@pytest.fixture
def clean_data(tmp_path):
    """Train and test from the SAME distribution — should return 'clean'."""
    np.random.seed(42)
    n = 2000
    df = pl.DataFrame({
        "feat_1": np.random.normal(0, 1, n).tolist(),
        "feat_2": np.random.normal(5, 2, n).tolist(),
        "feat_3": np.random.choice(["A", "B", "C"], n).tolist(),
        "target": np.random.randint(0, 2, n).tolist(),
    })
    train = df.head(1500)
    test = df.tail(500).drop("target")
    train.write_parquet(tmp_path / "clean_train.parquet")
    test.write_parquet(tmp_path / "clean_test.parquet")
    return tmp_path


@pytest.fixture
def shifted_data(tmp_path):
    """Train and test from DIFFERENT distributions — should return 'severe'."""
    np.random.seed(42)
    train = pl.DataFrame({
        "feat_1": np.random.normal(0, 1, 1500).tolist(),
        "feat_2": np.random.normal(5, 2, 1500).tolist(),
        "feat_3": np.random.choice(["A", "B", "C"], 1500).tolist(),
        "target": np.random.randint(0, 2, 1500).tolist(),
    })
    test = pl.DataFrame({
        "feat_1": np.random.normal(3, 1, 500).tolist(),    # SHIFTED mean
        "feat_2": np.random.normal(10, 2, 500).tolist(),   # SHIFTED mean
        "feat_3": np.random.choice(["A", "D", "E"], 500).tolist(),  # SHIFTED categories
    })
    train.write_parquet(tmp_path / "clean_train.parquet")
    test.write_parquet(tmp_path / "clean_test.parquet")
    return tmp_path
```

Tests:
1. `test_clean_data_returns_clean(clean_data)` — adversarial_auc < 0.55, severity == "clean"
2. `test_shifted_data_returns_severe(shifted_data)` — adversarial_auc > 0.65, severity == "severe"
3. `test_shift_report_has_all_required_keys` — all 10 keys present in shift_report
4. `test_adversarial_auc_between_0_and_1` — 0 <= adversarial_auc <= 1
5. `test_severity_is_valid_enum` — severity in ["clean", "mild", "severe"]
6. `test_sample_weights_generated_when_severe(shifted_data)` — sample_weights_path is non-empty, file exists
7. `test_no_weights_when_clean(clean_data)` — sample_weights_path is empty
8. `test_sample_weights_clipped` — all weights in [0.1, 10.0] when loaded from parquet
9. `test_shifted_features_populated(shifted_data)` — shifted_features list is non-empty
10. `test_shifted_features_empty_when_clean(clean_data)` — shifted_features list is empty
11. `test_uses_logistic_regression` — verify the adversarial model is LogisticRegression, NOT RandomForest or LightGBM (check the generated code or mock the import)
12. `test_dual_threshold_for_numeric` — a feature with ks_pvalue < 0.001 but PSI < 0.1 is NOT flagged (KS alone isn't enough)
13. `test_state_has_no_raw_arrays` — sample_weights_path is a string (file path), NOT a list of floats
14. `test_state_ownership` — shift_detector writes only to shift_* and sample_weights_path fields
15. `test_pipeline_continues_on_severe` — severity "severe" does NOT halt the pipeline (no GATE emitted)
16. `test_graceful_failure` — if shift detection crashes (e.g., all-null features), return severity "unchecked" and continue

---

## COMMIT 7: Complexity-Gated Pipeline Depth (graph/depth_router.py)

### What this prevents

Professor's 15-agent pipeline is overkill for a 1,000-row Playground competition. Claude Code loads, trains, submits in 20 minutes. Professor takes 25-30 minutes for the same result. And for the multi-run strategy (Operator Playbook), 5 STANDARD runs = 2.5 hours. With depth gating: 1 STANDARD + 4 SPRINT = 1h40m.

### Pipeline position

Runs INSIDE Pre-Flight Checks (Shield 6). After data profiling completes, before the main pipeline starts. Uses the profiling results to classify complexity and select depth.

### The depth classification function

```python
def classify_pipeline_depth(
    preflight_data_files: list,
    preflight_warnings: list, 
    preflight_target_type: str,
    preflight_data_size_mb: float,
    n_rows: int,
    n_features: int,
    metric_name: str,
    operator_override: str = None,
) -> dict:
    """
    Classify competition complexity and select pipeline depth.
    
    Returns:
    {
        "depth": "sprint" | "standard" | "marathon",
        "auto_detected": True | False,
        "reason": "rows=1200, features=14, metric=auc → sprint",
        "agents_skipped": ["competition_intel", "domain_research", ...],  # For SPRINT
        "optuna_trials": 50 | 100 | 200,
        "feature_rounds": 2 | 3 | 5 | 7,
        "critic_vectors": 4 | 9,  # Core only vs full
    }
    """
```

**SPRINT triggers when ALL of:**
- `n_rows < 10_000`
- `n_features < 30`
- `metric_name` is in standard set: `["roc_auc", "log_loss", "rmse", "rmsle", "mae", "f1", "accuracy", "r2", "mcc"]`
- `preflight_target_type` in `["binary", "multiclass", "regression"]`
- No BLOCKING preflight warnings
- No unsupported modalities

**MARATHON triggers when ANY of:**
- `n_rows > 100_000`
- `n_features > 200`
- `metric_name` not in standard set (custom metric)
- Unsupported modalities present but operator chose to proceed
- `preflight_data_size_mb > 500`
- `operator_override == "marathon"`

**STANDARD:** everything else. This is the DEFAULT. When in doubt, STANDARD.

**Conservative auto-detection:** An ambiguous competition (15K rows, 25 features, standard metric) gets STANDARD, not SPRINT. Only clear-cut simple competitions get SPRINT. The risk of SPRINT missing signal is worse than the cost of STANDARD's extra 15 minutes.

### What each depth configures

| Setting | SPRINT | STANDARD | MARATHON |
|---|---|---|---|
| Competition Intel | SKIPPED | Full | Full |
| Domain Research | SKIPPED | Full | Full + extended |
| EDA | Basic (v1 8-key only) | Deep (v2 4 sections + plots) | Deep + extra analysis |
| Shift Detector | SKIPPED | Full | Full |
| Feature Factory rounds | 2 | 3 | 5-7 |
| Creative Hypothesis | SKIPPED | Full | Full |
| Problem Reframer | SKIPPED | Full | Full |
| Optuna trials | 50 | 100 | 200 |
| Critic vectors | 4 core only | All 9 | All 9 |
| Pseudo-Labels | SKIPPED | Conditional | Always attempted |
| Post-Processing | SKIPPED | Full | Full |
| Ensemble | Simple average only | Hill climbing | Hill climbing + search |

**SPRINT skips these agents entirely:** competition_intel, domain_research, shift_detector, creative_hypothesis, problem_reframer, pseudo_label, post_processor.

**SPRINT Critic runs only 4 core vectors:** Vector 1a (target leakage), Vector 1b (preprocessing leakage), Vector 1c (train/test gap), Vector 3 (overfitting). Skips: Vector 1d (preprocessing audit), Vector 2 (feature stability), Vector 5 (metric gaming), Vector 6 (p-hacking), and any LLM-based vectors.

### Operator override

The operator can force any depth at launch:
```
professor run --competition titanic --depth sprint
professor run --competition healthcare --depth marathon
```

Or mid-run via HITL:
```
/depth marathon    → remaining agents use marathon settings
/depth sprint      → remaining non-essential agents are skipped
```

When operator overrides: `auto_detected = False`, depth is whatever they chose, regardless of data profile.

### Integration with Pre-Flight Checks

The depth classification runs at the END of `run_preflight_checks()`, AFTER profiling is complete. Pre-Flight needs the profiling results (n_rows, n_features, metric, target type) to classify depth.

```python
# At the end of run_preflight_checks():

# Get row count from profiling (first 100 rows loaded, but file metadata gives full count)
n_rows = _estimate_row_count(train_file_path)  # Use file size / avg row size from sample
n_features = len(column_profiles)

# Classify depth
depth_result = classify_pipeline_depth(
    preflight_data_files=inventory,
    preflight_warnings=all_warnings,
    preflight_target_type=target_type,
    preflight_data_size_mb=total_size_mb,
    n_rows=n_rows,
    n_features=n_features,
    metric_name=state.metric_name or "unknown",
    operator_override=_check_operator_depth_override(state),
)

# Include in state return
return state.validated_update("preflight", {
    # ... existing preflight fields ...
    "pipeline_depth": depth_result["depth"],
    "pipeline_depth_auto_detected": depth_result["auto_detected"],
    "pipeline_depth_reason": depth_result["reason"],
    "agents_skipped": depth_result["agents_skipped"],
})
```

### How agents read depth

Every agent checks `state.pipeline_depth` at the start:

```python
def domain_research(state: ProfessorState) -> dict:
    # Check if we're in SPRINT mode
    if state.pipeline_depth == "sprint" or "domain_research" in (state.agents_skipped or []):
        emit_to_operator("⏭️ Domain Research skipped (SPRINT mode)", level="STATUS")
        return {}  # No state changes
    
    # ... full domain research logic ...
```

This pattern is already in the universal agent template in PROFESSOR.md (check `hitl_skip_agents`). The depth router adds agents to the skip list, and the existing skip mechanism handles the rest.

### Row count estimation

Full row count is needed for depth classification but we DON'T want to load the entire file (that's what Pre-Flight is trying to avoid). Estimate:

```python
def _estimate_row_count(file_path: str) -> int:
    """Estimate row count without loading the full file."""
    if file_path.endswith(".parquet"):
        # Parquet metadata contains exact row count
        return pl.scan_parquet(file_path).collect().height
        # Actually for parquet we can read metadata:
        # import pyarrow.parquet as pq
        # return pq.read_metadata(file_path).num_rows
    
    # For CSV: sample-based estimation
    file_size = os.path.getsize(file_path)
    sample = pl.read_csv(file_path, n_rows=100)
    avg_row_bytes = file_size_of_sample / 100  # approximate
    return int(file_size / avg_row_bytes)
```

This is an estimate. Off by 10-20% is fine — the depth thresholds have enough margin (10K for SPRINT, 100K for MARATHON) that a 20% error doesn't change the classification.

### Milestone 0 includes depth

Update the Pre-Flight Milestone 0 message to include the depth decision:

```
🚀 PRE-FLIGHT REPORT
...
⚡ Pipeline Depth: SPRINT (auto-detected)
   Reason: rows≈1200, features=14, metric=auc
   Skipping: Competition Intel, Domain Research, Shift Detector, 
             Creative Hypothesis, Problem Reframer, Pseudo-Labels, Post-Processing
   Optuna trials: 50 | Feature rounds: 2 | Critic: 4 core vectors
   
   Override with /depth standard or /depth marathon
```

### State additions

These fields already exist in STATE.md:
- `pipeline_depth` (str) — "sprint" | "standard" | "marathon"
- `pipeline_depth_auto_detected` (bool)
- `pipeline_depth_reason` (str)

Add if missing:
- `agents_skipped` (list) — populated by depth router, agents check this

### Contract tests: tests/contracts/test_depth_router_contract.py

```python
@pytest.fixture
def simple_competition():
    """1000 rows, 14 features, binary, standard metric — should be SPRINT."""
    return {
        "preflight_data_files": [{"name": "train.csv", "size_mb": 0.5}],
        "preflight_warnings": [],
        "preflight_target_type": "binary",
        "preflight_data_size_mb": 0.8,
        "n_rows": 1000,
        "n_features": 14,
        "metric_name": "roc_auc",
    }


@pytest.fixture
def complex_competition():
    """200K rows, 300 features, custom metric — should be MARATHON."""
    return {
        "preflight_data_files": [{"name": "train.csv", "size_mb": 800}],
        "preflight_warnings": [],
        "preflight_target_type": "regression",
        "preflight_data_size_mb": 900,
        "n_rows": 200_000,
        "n_features": 300,
        "metric_name": "custom_weighted_f1",
    }


@pytest.fixture
def ambiguous_competition():
    """15K rows, 25 features — ambiguous, should default to STANDARD."""
    return {
        "preflight_data_files": [{"name": "train.csv", "size_mb": 50}],
        "preflight_warnings": [],
        "preflight_target_type": "binary",
        "preflight_data_size_mb": 55,
        "n_rows": 15_000,
        "n_features": 25,
        "metric_name": "roc_auc",
    }
```

Tests:
1. `test_simple_gets_sprint(simple_competition)` — depth == "sprint"
2. `test_complex_gets_marathon(complex_competition)` — depth == "marathon"
3. `test_ambiguous_gets_standard(ambiguous_competition)` — depth == "standard" (conservative)
4. `test_sprint_skips_correct_agents(simple_competition)` — agents_skipped contains: competition_intel, domain_research, shift_detector, creative_hypothesis, problem_reframer, pseudo_label, post_processor
5. `test_sprint_optuna_50(simple_competition)` — optuna_trials == 50
6. `test_marathon_optuna_200(complex_competition)` — optuna_trials == 200
7. `test_standard_optuna_100(ambiguous_competition)` — optuna_trials == 100
8. `test_sprint_critic_4_vectors(simple_competition)` — critic_vectors == 4
9. `test_standard_critic_9_vectors(ambiguous_competition)` — critic_vectors == 9
10. `test_sprint_feature_rounds_2(simple_competition)` — feature_rounds == 2
11. `test_marathon_feature_rounds_5_to_7(complex_competition)` — feature_rounds in [5, 6, 7]
12. `test_operator_override_respected` — pass operator_override="sprint" on complex competition → depth == "sprint", auto_detected == False
13. `test_custom_metric_triggers_marathon` — metric_name="custom_xyz" → marathon regardless of size
14. `test_reason_string_populated` — reason is non-empty and contains key values (rows, features, metric)
15. `test_blocking_warnings_prevent_sprint` — competition with BLOCKING preflight warnings → NOT sprint
16. `test_depth_result_has_all_fields` — returned dict has: depth, auto_detected, reason, agents_skipped, optuna_trials, feature_rounds, critic_vectors

---

## UPDATED INTEGRATION TEST (Commit 6 — revise from previous prompt)

Add to the Layer 1 integration test:

```python
def test_shift_detector_in_sequence(clean_data_dir):
    """Shift detector runs after data_engineer, produces valid severity."""
    state = ProfessorState(
        session_id="test-integration",
        clean_data_path=str(clean_data_dir / "clean_train.parquet"),
        clean_test_path=str(clean_data_dir / "clean_test.parquet"),
        data_schema={"feat_1": "Float64", "feat_2": "Float64", "feat_3": "Utf8"},
    )
    
    result = shift_detector(state)
    state = state.copy(update=result)
    
    assert state.shift_severity in ["clean", "mild", "severe", "unchecked"]
    assert isinstance(state.shifted_features, list)
    assert isinstance(state.sample_weights_path, str)  # String, not list


def test_depth_router_reads_preflight(clean_tabular_dir):
    """Depth router uses preflight results to classify correctly."""
    state = ProfessorState(session_id="test-depth")
    
    # Run preflight first
    state = state.copy(update=run_preflight_checks(state))
    
    # Verify depth was set
    assert state.pipeline_depth in ["sprint", "standard", "marathon"]
    assert state.pipeline_depth_reason != ""
    assert isinstance(state.agents_skipped, list)


def test_sprint_agents_check_skip_list():
    """In SPRINT mode, skipped agents return empty dict immediately."""
    state = ProfessorState(
        session_id="test-sprint-skip",
        pipeline_depth="sprint",
        agents_skipped=["domain_research", "creative_hypothesis"],
    )
    
    # Domain research should skip
    result = domain_research(state)
    assert result == {} or result == state.validated_update("domain_research", {})
```

---

## FULL COMMIT SEQUENCE FOR LAYER 1 (CORRECTED)

```
Commit 1:  shields/preflight.py (WITHOUT depth router) + tests
Commit 2:  agents/eda_agent.py (Deep EDA) + tests
Commit 3:  tools/eda_plots.py (Artifact Export) + tests
Commit 4:  agents/domain_research.py + tests
Commit 5:  agents/shift_detector.py (CORRECTED: LR, dual threshold, 3 severities) + tests
Commit 6:  graph/depth_router.py + update preflight.py to call it + tests
Commit 7:  Integration test — full Layer 1 sequence with depth routing + shift detection
```

Each commit passes `pytest tests/contracts/ -q`. No exceptions.

---

## WHAT NOT TO DO

- Do NOT use RandomForest or LightGBM for adversarial validation. Use LogisticRegression. The architecture document explains why.
- Do NOT use 4 severity levels. Use 3: clean/mild/severe. The thresholds are 0.55 and 0.65.
- Do NOT store raw weight arrays in state. Store the FILE PATH to the parquet file.
- Do NOT flag a feature as drifted with ONLY KS test. Require BOTH KS < 0.001 AND PSI > 0.25 for numeric features.
- Do NOT auto-halt on severe shift. Proceed with sample weighting. Only halt if weight computation itself fails.
- Do NOT classify ambiguous competitions as SPRINT. Default to STANDARD. Only clear-cut simple competitions get SPRINT.
- Do NOT skip depth classification when operator doesn't specify. Auto-detection always runs. Operator override only replaces the result.