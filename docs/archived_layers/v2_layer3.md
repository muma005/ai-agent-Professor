# BUILD PROMPT — Layer 3: Model + Validation + Ensemble (Days 9-14)
# Feed to Gemini CLI with: @PROFESSOR.md @STATE.md @SANDBOX.md @HITL.md @POLARS.md @CONTRACTS.md @PROMPTS.md @PROVIDERS.md

---

## CONTEXT

Layers 0-2 are complete and passing all contract tests:
- Layer 0: ProfessorState, Self-Debugging Engine, HITL, Cost Governor, Metric Gate, Data Integrity
- Layer 1: Pre-Flight + Depth Router, Deep EDA + Artifact Export, Domain Research, Shift Detector
- Layer 2: Leakage Precheck, Data Usage Checker, Retry Guidance, SOTA Search, Problem Reframer, Feature Factory (with Adaptive Gates + Parking), Creative Hypothesis

Now you're building the model training, adversarial validation, learning, and ensemble layer. This is where Professor's competitive advantage compounds — the Critic catches pipeline flaws, Self-Reflection learns across competitions, and the Ensemble extracts maximum value from diverse models.

Every component in this layer depends heavily on Layer 2's outputs (feature_manifest, features_train_path) and Layer 1's context (validation_strategy, eda_insights_summary, shift_report). All components write state through `validated_update()`, execute code through `run_in_sandbox()`, and emit through `emit_to_operator()`.

---

## COMMIT PLAN (8 commits)

```
Commit 1:  tools/gate_config.py + validation_architect gate integration + tests
Commit 2:  agents/ml_optimizer.py + tests
Commit 3:  agents/red_team_critic.py — Fix 2A (4 vector upgrades) + 2 new vectors (5, 6) + tests
Commit 4:  Critic Fix 2B — confirmation check before replan + tests
Commit 5:  agents/self_reflection.py + dynamic golden rules + tests
Commit 6:  agents/ensemble_architect.py — hill climbing upgrade + tests
Commit 7:  agents/post_processor.py + tests
Commit 8:  Integration test — full Layer 3 sequence on mock data
```

Every commit passes `pytest tests/contracts/ -q` including ALL Layer 0-2 tests. No exceptions.

---

## COMMIT 1: Gate Config (tools/gate_config.py)

### What this does

Single source of truth for all statistical gate thresholds. Adaptive by dataset size. Called by Validation Architect once, written to state, read by Feature Factory, Creative Hypothesis, ML Optimizer, and Ensemble Architect.

### File: tools/gate_config.py

```python
def get_gate_config(n_rows: int) -> dict:
    """
    Adaptive gate thresholds based on dataset size.
    
    Small data: relaxed thresholds (low statistical power, high noise)
    Medium data: default thresholds
    Large data: strict thresholds (high power, genuine signal only)
    
    Returns dict with: wilcoxon_p, null_importance_percentile,
    null_importance_shuffles, cv_folds, regime
    """
```

Implement exactly these regimes:

```python
if n_rows < 1500:
    return {
        "wilcoxon_p": 0.10,
        "null_importance_percentile": 90,
        "null_importance_shuffles": 30,
        "cv_folds": 5,           # Capped — 1500/7 = 214 rows per fold is too thin
        "regime": "very_small",
    }
elif n_rows < 5000:
    return {
        "wilcoxon_p": 0.10,
        "null_importance_percentile": 90,
        "null_importance_shuffles": 30,
        "cv_folds": 7,           # More folds = more paired observations for Wilcoxon
        "regime": "small",
    }
elif n_rows <= 50000:
    return {
        "wilcoxon_p": 0.05,
        "null_importance_percentile": 95,
        "null_importance_shuffles": 50,
        "cv_folds": 5,
        "regime": "medium",
    }
else:
    return {
        "wilcoxon_p": 0.02,
        "null_importance_percentile": 97,
        "null_importance_shuffles": 50,
        "cv_folds": 5,
        "regime": "large",
    }
```

**Why these specific numbers:**
- At n=5000 with 5 folds, Wilcoxon has n=5 pairs. Minimum p-value is 1/2^5 = 0.03. With p<0.05, a feature that improves 4/5 folds gets killed. With p<0.10, it passes (correctly).
- At n=5000 with 7 folds, Wilcoxon has n=7 pairs. Much more power. But 5000/7 = 714 rows per fold — still stable for LightGBM.
- At n<1500, 7 folds gives 214 rows per fold — too thin. Cap at 5.
- At n>50000, fold scores are very stable. p<0.02 is achievable and avoids noise features that get lucky on 1-2 folds.

### Integration: Validation Architect calls this

When Validation Architect runs (not built yet in this prompt, but document the contract):
```python
# In validation_architect.py:
from tools.gate_config import get_gate_config

gate_config = get_gate_config(state.canonical_train_rows)
# ... include gate_config in state return
```

**State addition:**
```python
gate_config: dict = Field(default_factory=dict)  # owner: validation_architect
```

Feature Factory (already built in Layer 2) should read `state.gate_config` instead of hardcoded thresholds. If Layer 2 was built with hardcoded thresholds, THIS commit updates Feature Factory to read from state. If `gate_config` is empty (backward compatibility), fall back to medium defaults.

### Contract tests: tests/contracts/test_gate_config_contract.py

1. `test_very_small_data` — n_rows=1000 → wilcoxon_p=0.10, cv_folds=5 (capped)
2. `test_small_data` — n_rows=3000 → wilcoxon_p=0.10, cv_folds=7
3. `test_medium_data` — n_rows=20000 → wilcoxon_p=0.05, cv_folds=5
4. `test_large_data` — n_rows=100000 → wilcoxon_p=0.02, percentile=97
5. `test_boundary_5000` — n_rows=5000 → regime="medium" (5000 is medium, not small)
6. `test_boundary_50000` — n_rows=50000 → regime="medium" (50000 is medium boundary)
7. `test_boundary_50001` — n_rows=50001 → regime="large"
8. `test_all_keys_present` — every return dict has: wilcoxon_p, null_importance_percentile, null_importance_shuffles, cv_folds, regime
9. `test_feature_factory_reads_gate_config` — Feature Factory uses state.gate_config thresholds, not hardcoded values (mock state with custom gate_config, verify the custom thresholds are used)

---

## COMMIT 2: ML Optimizer (agents/ml_optimizer.py)

### What this does

Trains models with Optuna HPO on the feature-engineered dataset. Produces OOF predictions for ensemble, test predictions for submission. Respects pipeline_depth for trial count. Uses sample weights from Shift Detector if available.

### The LangGraph node function

```python
def ml_optimizer(state: ProfessorState) -> dict:
    """
    Model training with Optuna hyperparameter optimization.
    
    Reads: features_train_path, features_test_path, target_column,
           validation_strategy, gate_config, metric_name, metric_config,
           technique_brief, shift_severity, sample_weights_path,
           active_reframing, pipeline_depth, canonical_train_rows
    Writes: model_configs, best_model_type, best_model_params,
            cv_scores, cv_mean, cv_std, oof_predictions_path,
            test_predictions_path, optuna_trials_completed
    Emits: STATUS (start), RESULT (Milestone 3 — model report)
    """
```

### Model portfolio

Professor trains 3 model families (always, regardless of technique_brief suggestions):

1. **LightGBM** — primary model, typically strongest on tabular
2. **XGBoost** — second model, different regularization profile
3. **CatBoost** — third model, handles categoricals natively

If `technique_brief.recommended_models` suggests additional models (e.g., RealMLP, TabNet), add them as optional candidates BUT only if the library is importable in the sandbox. Verify with a try/import in the generated code.

### Per-model Optuna optimization

Generate code via `llm_call()` for each model family. The code must:

1. Load features from `features_train_path` (parquet)
2. Set up CV using `state.validation_strategy` (fold type, n_splits, group_col if applicable)
3. Apply target transform if `state.active_reframing` has a transform
4. Apply sample weights if `state.sample_weights_path` is non-empty
5. Define Optuna objective function with the model's hyperparameter search space
6. Run `optuna.create_study().optimize(objective, n_trials=N)` where N comes from pipeline_depth:
   - SPRINT: 50 trials
   - STANDARD: 100 trials
   - MARATHON: 200 trials
7. Train best trial on full training data
8. Generate OOF predictions via cross-validation with best params
9. Generate test predictions
10. Apply inverse transform if reframing was active
11. Save OOF predictions and test predictions as parquet

**Hyperparameter search spaces (hardcoded, not LLM-generated):**

Include these in the prompt as CONSTRAINTS so the LLM doesn't invent its own spaces:

```python
# LightGBM search space
LGBM_SPACE = {
    "n_estimators": ("int", 100, 1000),
    "max_depth": ("int", 3, 12),
    "learning_rate": ("float_log", 0.01, 0.3),
    "num_leaves": ("int", 15, 127),
    "min_child_samples": ("int", 5, 100),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.3, 1.0),
    "reg_alpha": ("float_log", 1e-8, 10.0),
    "reg_lambda": ("float_log", 1e-8, 10.0),
}

# XGBoost search space
XGB_SPACE = {
    "n_estimators": ("int", 100, 1000),
    "max_depth": ("int", 3, 10),
    "learning_rate": ("float_log", 0.01, 0.3),
    "min_child_weight": ("int", 1, 100),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.3, 1.0),
    "gamma": ("float_log", 1e-8, 5.0),
    "reg_alpha": ("float_log", 1e-8, 10.0),
    "reg_lambda": ("float_log", 1e-8, 10.0),
}

# CatBoost search space
CAT_SPACE = {
    "iterations": ("int", 100, 1000),
    "depth": ("int", 3, 10),
    "learning_rate": ("float_log", 0.01, 0.3),
    "l2_leaf_reg": ("float_log", 1e-8, 10.0),
    "bagging_temperature": ("float", 0.0, 1.0),
    "random_strength": ("float_log", 1e-8, 10.0),
}
```

### Multi-seed stability validation

After Optuna finds the best params for each model:

```python
# Train with 3 seeds and check stability
seeds = [42, 142, 242]
seed_scores = []
for seed in seeds:
    params_with_seed = {**best_params, "random_state": seed}
    score = cross_val_score_with_params(params_with_seed, X, y, cv)
    seed_scores.append(score)

mean_score = np.mean(seed_scores)
std_score = np.std(seed_scores)
stability_penalty = std_score * STABILITY_PENALTY  # 1.5 from constants
adjusted_score = mean_score - stability_penalty
```

Use `adjusted_score` (not raw mean) for model ranking. This penalizes models that are lucky on one seed.

### Sample weight integration

If `state.sample_weights_path` is non-empty (Shift Detector found distribution shift):
```python
# Load weights
weights = pl.read_parquet(sample_weights_path)["weight"].to_numpy()

# Pass to model training
lgbm_model.fit(X_train, y_train, sample_weight=weights_train)
xgb_model.fit(X_train, y_train, sample_weight=weights_train)
catboost_model.fit(X_train, y_train, sample_weight=weights_train)

# Wilcoxon gate: weighted vs unweighted
# Only use weights if weighted model beats unweighted
```

### State return

```python
return state.validated_update("ml_optimizer", {
    "model_configs": [
        {"model_type": "lightgbm", "params": lgbm_best, "cv_score": lgbm_score, "cv_std": lgbm_std},
        {"model_type": "xgboost", "params": xgb_best, "cv_score": xgb_score, "cv_std": xgb_std},
        {"model_type": "catboost", "params": cat_best, "cv_score": cat_score, "cv_std": cat_std},
    ],
    "best_model_type": best_model_name,
    "best_model_params": best_params,
    "cv_scores": per_fold_scores,  # List of per-fold scores for best model
    "cv_mean": best_adjusted_score,
    "cv_std": best_std,
    "oof_predictions_path": f"outputs/{state.session_id}/oof_predictions.parquet",
    "test_predictions_path": f"outputs/{state.session_id}/test_predictions.parquet",
    "optuna_trials_completed": total_trials_across_models,
})
```

### HITL Milestone 3

After all models trained:
```
🎯 MODEL REPORT
Best model: LightGBM
CV: 0.8312 ± 0.0045 (stability-adjusted)
Multi-seed: [0.8334, 0.8298, 0.8305] → std=0.0019

All models:
  LightGBM:  0.8312 ± 0.0045
  XGBoost:   0.8278 ± 0.0038
  CatBoost:  0.8265 ± 0.0052

Optuna trials: 300 total (100 per model)
Sample weights: Applied (shift_severity=mild)

Reply /submit or /iterate
```

### Contract tests: tests/contracts/test_ml_optimizer_contract.py

1. `test_3_model_configs_produced` — model_configs has exactly 3 entries (LGB, XGB, CAT)
2. `test_model_configs_have_required_fields` — each has: model_type, params, cv_score, cv_std
3. `test_best_model_type_valid` — best_model_type in ["lightgbm", "xgboost", "catboost"]
4. `test_cv_scores_is_list_of_floats` — cv_scores is list, all entries are float
5. `test_cv_mean_matches_scores` — cv_mean approximately equals mean of cv_scores (within stability penalty)
6. `test_oof_predictions_file_exists` — oof_predictions_path points to existing parquet
7. `test_test_predictions_file_exists` — test_predictions_path points to existing parquet
8. `test_oof_row_count_matches_train` — OOF predictions have canonical_train_rows rows
9. `test_test_row_count_matches_test` — test predictions have canonical_test_rows rows
10. `test_sprint_uses_50_trials` — pipeline_depth="sprint" → 50 trials per model
11. `test_standard_uses_100_trials` — pipeline_depth="standard" → 100 trials per model
12. `test_marathon_uses_200_trials` — pipeline_depth="marathon" → 200 trials per model
13. `test_multi_seed_stability` — verify 3 seeds are used, std is computed
14. `test_stability_penalty_applied` — adjusted_score < raw_mean when std > 0
15. `test_sample_weights_used_when_available` — mock sample_weights_path, verify weights passed to fit
16. `test_milestone_3_emitted` — mock emit_to_operator, verify RESULT emitted with model report

---

## COMMIT 3: Red Team Critic — Fix 2A + 2 New Vectors (agents/red_team_critic.py)

### What this changes

Upgrade the Critic from 7 vectors to 9 vectors, with 4 existing vectors patched to reduce false positives.

### Vector upgrades (Fix 2A)

**Vector 1a — Shuffled target: swap RandomForest for LogisticRegression:**

```python
# BEFORE (v1):
model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42, n_jobs=-1)

# AFTER (v2):
model = LogisticRegression(max_iter=200, random_state=42, solver="lbfgs")
```

WHY: RF memorizes high-cardinality categorical leaf patterns, producing AUC 0.56-0.58 on shuffled targets. This is RF noise, not leakage. LR cannot memorize categoricals. If LR on shuffled targets gets AUC > 0.55, it's genuine linear leakage.

Prepare features for LR: select only numeric features. For categoricals with <20 unique values, one-hot encode. For >20 unique, skip (would cause dimensionality explosion). Standardize all features with `StandardScaler`.

**Vector 1c — Cross-reference with Shift Detector:**

```python
adversarial_auc = float(np.mean(scores))
shift_severity = state.shift_severity

if adversarial_auc > 0.75:
    if shift_severity in ("mild", "severe"):
        # Drift is KNOWN — Shift Detector already handling it
        verdict = "HIGH"  # Downgraded from CRITICAL
        evidence = (f"Adversarial AUC {adversarial_auc:.4f} but shift_severity={shift_severity}. "
                    f"Drift is known and handled by sample weighting. Not a pipeline bug.")
        replan_instructions = {"remove_features": [], "rerun_nodes": []}
    else:
        # Shift Detector said clean but adversarial AUC is high — genuinely suspicious
        verdict = "CRITICAL"
        evidence = (f"Adversarial AUC {adversarial_auc:.4f} with shift_severity=clean. "
                    f"Unexpected shift detected post-training — likely feature engineering introduced it.")
        replan_instructions = {"remove_features": top_drift_features, "rerun_nodes": ["feature_factory", "ml_optimizer"]}
```

**Vector 1d — Safe-pattern whitelist:**

Reuse the same `SAFE_PATTERNS` list from the pre-execution leakage check (guards/leakage_precheck.py). When Vector 1d scans the Code Ledger for preprocessing patterns, check surrounding context against safe patterns before flagging.

```python
from guards.leakage_precheck import SAFE_PATTERNS

for line_num, line in enumerate(code_lines):
    if _matches_danger_pattern(line):
        context = code_lines[max(0, line_num-3):line_num+4]
        context_str = "\n".join(context)
        if any(re.search(safe, context_str) for safe in SAFE_PATTERNS):
            continue  # Safe context — skip
        findings.append(...)
```

**Vector 4 — Type-aware noise injection:**

```python
for feature in top_features:
    col_data = X_test[feature]
    
    if _is_binary(col_data):
        # Binary: random bit flip at 10% rate
        mask = np.random.random(len(col_data)) < 0.10
        noisy = col_data.copy()
        noisy[mask] = 1 - noisy[mask]
        noise_type = "bit_flip_10pct"
    elif _is_categorical_encoded(col_data):
        # Integer-encoded categorical: random swap to different category
        mask = np.random.random(len(col_data)) < 0.10
        unique_vals = np.unique(col_data[~np.isnan(col_data)])
        noisy = col_data.copy()
        for i in np.where(mask)[0]:
            noisy[i] = np.random.choice(unique_vals)
        noise_type = "category_swap_10pct"
    else:
        # Continuous: Gaussian noise (existing approach)
        sigma = float(np.std(col_data)) * 0.10
        noise = np.random.normal(0, sigma, len(col_data))
        noisy = col_data + noise
        noise_type = "gaussian_10pct"

def _is_binary(col):
    unique = np.unique(col[~np.isnan(col)])
    return len(unique) <= 2

def _is_categorical_encoded(col):
    unique = np.unique(col[~np.isnan(col)])
    return len(unique) <= 20 and np.all(unique == unique.astype(int))
```

### Vector 5 — Metric Gaming Detection (NEW)

Three sub-checks executed as a single code block via `run_in_sandbox()`:

**Sub-check A — Prediction Distribution Audit:**
```python
oof_preds = load_oof_predictions()
# Check for prediction spikes
hist, edges = np.histogram(oof_preds, bins=50)
max_bin_pct = hist.max() / hist.sum()
if max_bin_pct > 0.30:
    # >30% of predictions in one bin — suspicious
    subcheck_a = "HIGH"
else:
    subcheck_a = "OK"
```

**Sub-check B — Baseline Comparison:**
```python
# Majority class baseline
majority_pred = np.full(len(y_true), mode(y_true))
majority_score = metric_func(y_true, majority_pred)

# Best single feature logistic regression
best_single_score = 0
for col in feature_cols[:20]:  # Top 20 by importance
    lr = LogisticRegression(max_iter=200)
    lr_score = cross_val_score(lr, X[:, col:col+1], y, cv=3, scoring=scorer).mean()
    best_single_score = max(best_single_score, lr_score)

if model_cv_score <= majority_score:
    subcheck_b = "CRITICAL"  # Model doesn't beat random guessing
elif model_cv_score <= best_single_score:
    subcheck_b = "HIGH"  # Model doesn't beat a single feature
else:
    subcheck_b = "OK"
```

**Sub-check C — Calibration Audit (classification only):**
```python
if competition_type in ["binary", "multiclass"]:
    from sklearn.calibration import calibration_curve
    fraction_pos, mean_predicted = calibration_curve(y_true, oof_preds, n_bins=10)
    ece = np.mean(np.abs(fraction_pos - mean_predicted))
    
    if ece > 0.15:
        subcheck_c = "HIGH"
    else:
        subcheck_c = "OK"
else:
    subcheck_c = "SKIP"  # Not applicable to regression
```

**Severity logic:**
- Any sub-check CRITICAL → Vector 5 overall = CRITICAL
- Any sub-check HIGH (and none CRITICAL) → Vector 5 overall = HIGH
- All OK/SKIP → Vector 5 overall = OK

### Vector 6 — P-Hacking Detection (NEW)

```python
# Take top 5 Optuna trials by CV score
top_trials = sorted(state.model_configs[0]["optuna_trials"], key=lambda t: t["score"], reverse=True)[:5]

# Re-run each with 3 different seeds
seed_stds = []
for trial in top_trials:
    trial_scores = []
    for seed_offset in [0, 100, 200]:
        params = {**trial["params"], "random_state": 42 + seed_offset}
        score = quick_cv(params, X, y, cv, scorer)  # 50-tree quick CV
        trial_scores.append(score)
    
    trial_std = np.std(trial_scores)
    seed_stds.append(trial_std)

mean_std = np.mean(seed_stds)
max_std = np.max(seed_stds)

if mean_std > 0.010:
    verdict = "CRITICAL"  # Entire top of Optuna leaderboard is unstable
    evidence = f"Mean seed std across top 5 trials: {mean_std:.4f}. HPO results driven by seed luck."
elif max_std > 0.015:
    verdict = "HIGH"  # At least one trial is unstable
    evidence = f"Trial with std {max_std:.4f} across seeds. Score swings >1.5% by seed."
else:
    verdict = "OK"
```

**Why 0.015 and 0.010:** A trial scoring 0.82 on one seed and 0.805 on another (std ~0.015) has a 1.5% swing — larger than most feature engineering gains. If the entire top-5 averages std > 0.01, the Optuna leaderboard is noise.

**Cost concern:** Re-running 5 trials × 3 seeds = 15 model trainings. Mitigate by using FAST config: `n_estimators=50` (not the full trial's 500+). This checks RELATIVE stability, not absolute performance.

### Critic orchestration — all 9 vectors

```python
def red_team_critic(state: ProfessorState) -> dict:
    """
    9-vector adversarial validation of the pipeline.
    ALL vectors run. None skipped. A Critic that skips vectors creates false confidence.
    
    Exception: SPRINT mode runs only 4 core vectors (1a, 1b, 1c, 3).
    """
    
    # Determine which vectors to run
    if state.pipeline_depth == "sprint":
        vectors_to_run = ["1a", "1b", "1c", "3"]  # Core only
    else:
        vectors_to_run = ["1a", "1b", "1c", "1d", "1e", "1f", "3", "4", "5", "6"]
    
    findings = []
    for vector_id in vectors_to_run:
        result = _run_vector(vector_id, state)
        findings.append(result)
    
    # Determine overall severity
    severities = [f["severity"] for f in findings]
    if "CRITICAL" in severities:
        overall = "CRITICAL"
    elif "HIGH" in severities:
        overall = "HIGH"
    else:
        overall = "CLEAR"
    
    # ... emit results, return state
```

### State additions

```python
critic_gaming_flags: list = Field(default_factory=list)   # Vector 5 findings, owner: red_team_critic
critic_phacking_flags: list = Field(default_factory=list)  # Vector 6 findings, owner: red_team_critic
```

### Contract tests: tests/contracts/test_critic_v2_contract.py

NEW file — v1 contracts are FROZEN.

**Fix 2A tests:**
1. `test_vector_1a_uses_logistic_regression` — verify model is LR, not RF
2. `test_high_cardinality_no_false_critical` — 10 high-cardinality categoricals + shuffled targets → NOT CRITICAL
3. `test_known_drift_downgrades_to_high` — shift_severity="severe" + adversarial AUC=0.80 → HIGH, not CRITICAL
4. `test_unknown_drift_stays_critical` — shift_severity="clean" + adversarial AUC=0.80 → CRITICAL
5. `test_safe_pattern_not_flagged` — Pipeline() inside cross_val_score → NOT flagged by Vector 1d
6. `test_unsafe_pattern_still_flagged` — scaler.fit_transform(X) before KFold → still flagged
7. `test_binary_feature_bit_flip` — binary feature → noise_type "bit_flip_10pct"
8. `test_categorical_feature_swap` — integer-encoded categorical → noise_type "category_swap_10pct"

**New vector tests:**
9. `test_vectors_checked_has_9_entries` — vectors_checked list has 9 entries (STANDARD mode)
10. `test_sprint_runs_4_core_vectors` — pipeline_depth="sprint" → only 4 vectors run
11. `test_vector5_majority_class_critical` — all-majority predictions → CRITICAL (fails baseline)
12. `test_vector5_good_model_ok` — well-spread predictions beating both baselines → OK
13. `test_vector5_prediction_spike_high` — >30% of predictions at one value → HIGH
14. `test_vector6_stable_trials_ok` — 5 trials with std < 0.005 across seeds → OK
15. `test_vector6_unstable_trials_critical` — 5 trials with mean std > 0.015 → CRITICAL
16. `test_vector6_uses_fast_config` — re-run uses n_estimators=50, not full trial params
17. `test_all_v1_vectors_still_run` — v1 vectors 1b, 1e, 1f, 3 still produce results
18. `test_v1_critic_contracts_pass` — all existing v1 critic contract tests unchanged

---

## COMMIT 4: Critic Fix 2B — Confirmation Check (agents/red_team_critic.py continued)

### What this prevents

A single noisy vector run returns CRITICAL. Supervisor replans. 25 minutes wasted. The CRITICAL was random noise from one unlucky seed. Fix 2B re-runs the vector with a different seed AND a different model. Both must confirm CRITICAL for the replan to proceed.

### Implementation — add to the end of red_team_critic()

```python
# After all vectors complete:
critical_findings = [f for f in findings if f["severity"] == "CRITICAL"]

if critical_findings:
    confirmed_findings = []
    
    for finding in critical_findings:
        # Confirmation check 1: re-run with different seed
        recheck_1 = _run_single_vector(
            finding["vector"], state,
            seed_override=42 + 1000
        )
        
        # Confirmation check 2: re-run with different model
        recheck_2 = _run_single_vector(
            finding["vector"], state,
            model_override="logistic_regression"
        )
        
        confirmed_by_seed = recheck_1["severity"] == "CRITICAL"
        confirmed_by_model = recheck_2["severity"] == "CRITICAL"
        
        if confirmed_by_seed and confirmed_by_model:
            finding["severity"] = "CONFIRMED_CRITICAL"
            finding["confirmation"] = "both_checks_passed"
            confirmed_findings.append(finding)
        elif confirmed_by_seed or confirmed_by_model:
            finding["severity"] = "HIGH"  # Downgrade
            finding["confirmation"] = "one_check_passed"
        else:
            finding["severity"] = "FALSE_POSITIVE"
            finding["confirmation"] = "neither_check_passed"
    
    # Only CONFIRMED_CRITICAL triggers replan
    overall = "CONFIRMED_CRITICAL" if confirmed_findings else "HIGH"
    replan_requested = len(confirmed_findings) > 0
else:
    replan_requested = False
```

**Cost:** ~60 seconds per CRITICAL finding (2 re-runs).
**Savings:** Prevents 25-minute replan cycles when CRITICAL was noise.
**Expected net:** False CRITICALs outnumber true CRITICALs. Large net time savings.

### Contract tests: tests/contracts/test_critic_confirmation_contract.py

1. `test_confirmation_runs_on_critical` — CRITICAL finding → 2 re-runs execute (different seed + different model)
2. `test_both_confirm_gives_confirmed_critical` — both re-runs return CRITICAL → CONFIRMED_CRITICAL
3. `test_one_confirms_downgrades_to_high` — one re-run returns CRITICAL, other doesn't → HIGH
4. `test_neither_confirms_gives_false_positive` — both re-runs return OK → FALSE_POSITIVE
5. `test_confirmed_critical_triggers_replan` — CONFIRMED_CRITICAL → replan_requested=True
6. `test_false_positive_no_replan` — all findings downgraded → replan_requested=False
7. `test_genuine_leakage_confirmed` — inject target copy column → CONFIRMED_CRITICAL after both checks
8. `test_confirmation_uses_different_seed` — verify seed_override is 42+1000, not 42
9. `test_high_findings_skip_confirmation` — HIGH severity findings don't get confirmation checks (only CRITICAL does)

---

## COMMIT 5: Self-Reflection Agent (agents/self_reflection.py)

### What this does

Learns from every Critic verdict. Extracts rules. Rules accumulate across competitions, get validated, get promoted or demoted. After 5+ competitions, Professor starts each new competition with learned patterns.

### The LangGraph node function

```python
def self_reflection(state: ProfessorState) -> dict:
    """
    Immediate reflection after every Critic verdict.
    Extracts lessons, manages dynamic rule lifecycle.
    
    Reads: critic_verdict, cv_mean, feature_manifest, model_configs,
           dynamic_rules_active, eda_insights_summary
    Writes: reflection_notes, dynamic_rules_active, dynamic_rules_pending
    Emits: STATUS (reflection complete)
    """
```

### Mode 1 — Immediate Reflection (runs every time)

After Critic verdict is received:

1. **Root cause analysis via `llm_call()`:**
```
Analyze this Critic verdict and identify the root cause pattern:

Verdict: {critic_verdict}
Competition: {competition_name}
CV Score: {cv_mean}
Features used: {len(feature_manifest)}

Classify the root cause into ONE of:
- preprocessing_leakage
- feature_noise
- distribution_shift
- metric_gaming
- overfitting
- calibration_failure
- data_quality
- none (Critic was CLEAR)

Then write a RULE that would prevent this failure class in future competitions.
The rule should be:
- Specific enough to be actionable
- General enough to apply across competitions
- Testable (you can verify whether it was followed)

Respond with JSON:
{
    "failure_class": str,
    "root_cause": str,
    "rule_text": str,
    "applies_to_agents": [str],
    "confidence_justification": str
}
```

2. **Rule lifecycle management:**

```python
# New rule from this verdict
new_rule = {
    "rule_id": f"rule_{uuid4().hex[:8]}",
    "text": llm_result["rule_text"],
    "failure_class": llm_result["failure_class"],
    "confidence": 0.50,  # Born at 0.50
    "source_competition": state.session_id,
    "created_at": datetime.utcnow().isoformat(),
    "validated_count": 0,
    "violated_count": 0,
    "applies_to_agents": llm_result["applies_to_agents"],
}
```

3. **Update existing rules based on this verdict:**

```python
for rule in existing_rules:
    if rule["failure_class"] == llm_result["failure_class"]:
        if critic_verdict["severity"] == "CLEAR":
            # Critic CLEAR and rule predicted this class → rule was followed correctly
            rule["confidence"] += 0.10
            rule["validated_count"] += 1
        elif critic_verdict["severity"] in ("CRITICAL", "CONFIRMED_CRITICAL"):
            # Critic found the same class → rule wasn't followed or didn't help
            rule["confidence"] -= 0.15
            rule["violated_count"] += 1
```

4. **Manage active rules:**

```python
# Sort by confidence descending
all_rules = sorted(existing_rules + [new_rule], key=lambda r: r["confidence"], reverse=True)

# Promote top rules with confidence > 0.80
active_rules = [r for r in all_rules if r["confidence"] > 0.80][:20]  # Max 20

# Demote rules below 0.40
for rule in all_rules:
    if rule["confidence"] < 0.40:
        rule["status"] = "deprecated"
    elif rule["confidence"] >= 0.80:
        rule["status"] = "active"
    else:
        rule["status"] = "pending"

# Conflict check against hard-coded Golden Rules
GOLDEN_RULES = [
    "If CV and public LB are not correlated — STOP.",
    "Never select complex over simple unless Wilcoxon p < threshold.",
    "Agents pass pointers, never payloads.",
    "Same failure twice = strategy problem.",
    "Every forum insight is a hypothesis, not a fact.",
    "Final 2 submissions: best CV + most different model.",
]

for rule in active_rules:
    # Quick LLM check: does this rule contradict any Golden Rule?
    # If yes: rule is NEVER promoted regardless of confidence
    # This is a simple string similarity + LLM judgment call
    pass
```

### Mode 2 — Post-Competition Retrospective

This runs MANUALLY after competition closes (not in the main pipeline). Build it as a separate function:

```python
def post_competition_retrospective(state: ProfessorState, private_score: float) -> dict:
    """
    After competition ends and private LB is revealed.
    Updates rule confidence based on actual outcome.
    """
    cv_lb_gap = abs(state.cv_mean - private_score)
    
    for rule in state.dynamic_rules_active:
        if rule["failure_class"] in critic_findings_classes:
            if cv_lb_gap < 0.01:
                # Critic flagged but private held — likely false positive
                rule["confidence"] -= 0.20
            else:
                # Critic flagged and private collapsed — true positive
                rule["confidence"] += 0.15
    
    # Store to ChromaDB for cross-competition learning
    # ...
```

### Downstream injection of active rules

Active rules (confidence > 0.80) are injected into downstream agent prompts:

```python
# In any agent's prompt building:
if state.dynamic_rules_active:
    rules_text = "\n".join([
        f"RULE (confidence {r['confidence']:.2f}): {r['text']}" 
        for r in state.dynamic_rules_active
        if agent_name in r.get("applies_to_agents", [])
    ])
    prompt += f"\n\nRULES FROM PAST COMPETITIONS:\n{rules_text}"
```

### State return

```python
return state.validated_update("self_reflection", {
    "reflection_notes": existing_notes + [new_reflection],
    "dynamic_rules_active": active_rules,
    "dynamic_rules_pending": pending_rules,
})
```

### Contract tests: tests/contracts/test_self_reflection_contract.py

1. `test_reflection_log_grows` — after running, reflection_notes has one more entry
2. `test_rules_have_required_fields` — rule_id, text, failure_class, confidence, source_competition, timestamps
3. `test_new_rules_born_at_0_50` — fresh rule has confidence=0.50
4. `test_confidence_increases_on_validation` — rule validated → confidence += 0.10
5. `test_confidence_decreases_on_violation` — rule violated → confidence -= 0.15
6. `test_active_rules_above_0_80` — every rule in dynamic_rules_active has confidence > 0.80
7. `test_max_20_active_rules` — even with 30 eligible rules, active list is capped at 20
8. `test_deprecated_rules_archived` — rule with confidence < 0.40 has status="deprecated"
9. `test_golden_rule_conflict_blocked` — rule contradicting "never select complex over simple unless Wilcoxon" → never promoted
10. `test_clear_verdict_validates_rules` — CLEAR critic verdict + matching failure class → confidence increases
11. `test_never_halts_pipeline` — self_reflection always returns, never raises
12. `test_empty_rules_on_first_competition` — no existing rules → new rule created, active list may be empty (confidence=0.50 < 0.80)

---

## COMMIT 6: Hill Climbing Ensemble Upgrade (agents/ensemble_architect.py)

### What this upgrades

v1 blends LGB/XGB/CatBoost — same model family with >0.95 prediction correlation. Marginal gain. v2 adds 4 techniques that extract real diversity.

### The LangGraph node function

```python
def ensemble_architect(state: ProfessorState) -> dict:
    """
    Ensemble with diversity pruning, greedy selection, 
    hill climbing weights, and rank averaging.
    
    Reads: model_configs, oof_predictions_path, test_predictions_path,
           cv_scores, metric_name, validation_strategy, gate_config
    Writes: ensemble_method, ensemble_weights, ensemble_cv,
            ensemble_diversity_report
    Emits: STATUS (start), CHECKPOINT (ensemble report — ask operator)
    """
```

### Step 1 — Load all OOF predictions

Each model produced OOF predictions during ML Optimizer. Load all of them:
```python
# Each model's OOF predictions stored as columns in one parquet
# Or as separate files — adapt based on ml_optimizer's output format
oof_preds = {}  # {"lightgbm": np.array, "xgboost": np.array, "catboost": np.array}
```

### Step 2 — Diversity analysis

```python
def _compute_diversity_report(oof_preds: dict) -> dict:
    """Pairwise prediction correlation for all model pairs."""
    models = list(oof_preds.keys())
    correlations = {}
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                corr = np.corrcoef(oof_preds[m1], oof_preds[m2])[0, 1]
                correlations[f"{m1}_vs_{m2}"] = round(corr, 4)
    return {"pairwise_correlations": correlations, "models": models}
```

### Step 3 — Diversity pruning

```python
def _diversity_prune(oof_preds: dict, model_scores: dict, threshold: float = 0.98) -> dict:
    """Remove models with correlation > threshold. Keep the higher-scoring one."""
    models = list(oof_preds.keys())
    pruned = set()
    
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j and m1 not in pruned and m2 not in pruned:
                corr = np.corrcoef(oof_preds[m1], oof_preds[m2])[0, 1]
                if corr > threshold:
                    # Drop the lower-scoring one
                    drop = m2 if model_scores[m1] >= model_scores[m2] else m1
                    pruned.add(drop)
    
    return {k: v for k, v in oof_preds.items() if k not in pruned}
```

### Step 4 — Try all 4 ensemble techniques

**Technique 1 — Simple weighted average (baseline):**
```python
def _simple_blend(oof_preds, y_true, scorer):
    """Equal-weight average of all models."""
    avg = np.mean(list(oof_preds.values()), axis=0)
    return scorer(y_true, avg)
```

**Technique 2 — Greedy forward selection:**
```python
def _greedy_forward(oof_preds, y_true, scorer, wilcoxon_p):
    """Start with best model. Add models that improve Wilcoxon-tested."""
    models = sorted(oof_preds.keys(), key=lambda m: scorer(y_true, oof_preds[m]), reverse=True)
    selected = [models[0]]
    
    for candidate in models[1:]:
        current_blend = np.mean([oof_preds[m] for m in selected], axis=0)
        new_blend = np.mean([oof_preds[m] for m in selected + [candidate]], axis=0)
        
        # Wilcoxon test: does adding this model help?
        # Compare per-fold scores
        current_score = scorer(y_true, current_blend)
        new_score = scorer(y_true, new_blend)
        
        if new_score > current_score:
            # Quick Wilcoxon on fold-level scores
            selected.append(candidate)
    
    return selected, scorer(y_true, np.mean([oof_preds[m] for m in selected], axis=0))
```

**Technique 3 — Hill climbing weights:**
```python
def _hill_climb_weights(oof_preds, y_true, scorer, n_iterations=1000):
    """Perturb weights, accept if score improves."""
    models = list(oof_preds.keys())
    n_models = len(models)
    weights = np.ones(n_models) / n_models  # Start equal
    
    preds_matrix = np.array([oof_preds[m] for m in models])  # (n_models, n_samples)
    best_score = scorer(y_true, weights @ preds_matrix)
    best_weights = weights.copy()
    
    for _ in range(n_iterations):
        # Perturb: pick random model, shift weight ±0.05
        idx = np.random.randint(n_models)
        delta = np.random.choice([-0.05, 0.05])
        new_weights = best_weights.copy()
        new_weights[idx] += delta
        new_weights = np.clip(new_weights, 0, 1)
        new_weights /= new_weights.sum()  # Renormalize
        
        score = scorer(y_true, new_weights @ preds_matrix)
        if score > best_score:
            best_score = score
            best_weights = new_weights.copy()
    
    return dict(zip(models, best_weights.tolist())), best_score
```

**Technique 4 — Rank averaging:**
```python
def _rank_average(oof_preds, y_true, scorer):
    """Convert predictions to percentile ranks, then average."""
    from scipy.stats import rankdata
    
    ranked = {}
    for model, preds in oof_preds.items():
        ranked[model] = rankdata(preds) / len(preds)  # Percentile ranks [0, 1]
    
    avg_ranks = np.mean(list(ranked.values()), axis=0)
    return scorer(y_true, avg_ranks)
```

### Step 5 — Select best technique

```python
results = {
    "simple_blend": simple_score,
    "greedy_forward": greedy_score,
    "hill_climbing": hill_score,
    "rank_average": rank_score,
}

# Also compare against best single model
best_single = max(scorer(y_true, oof_preds[m]) for m in oof_preds)

# Pick best technique that beats best single model
best_technique = max(results, key=results.get)
best_ensemble_score = results[best_technique]

if best_ensemble_score <= best_single:
    # No ensemble beats single model → use best single (Occam's razor)
    ensemble_method = "single_model"
    ensemble_cv = best_single
else:
    ensemble_method = best_technique
    ensemble_cv = best_ensemble_score
```

### Generate final test predictions

Apply the winning technique to test predictions:
```python
# Load test predictions for each model
# Apply same weights/selection/ranking
# Save final blended test predictions
```

### HITL emission

```
⚔️ ENSEMBLE REPORT
Best single model: LightGBM (0.8312)
Simple blend: 0.8334 (+0.0022)
Greedy forward: 0.8338 (+0.0026) — selected LGB + XGB
Hill climbing: 0.8341 (+0.0029) — weights: LGB=0.45, XGB=0.35, CAT=0.20
Rank average: 0.8336 (+0.0024)

Selected: hill_climbing (0.8341)
Diversity: LGB↔XGB=0.953, LGB↔CAT=0.938, XGB↔CAT=0.961

Reply with approval or /iterate
```

### State return

```python
return state.validated_update("ensemble_architect", {
    "ensemble_method": ensemble_method,
    "ensemble_weights": weights_dict,
    "ensemble_cv": best_ensemble_score,
    "ensemble_diversity_report": diversity_report,
})
```

### Contract tests: tests/contracts/test_ensemble_v2_contract.py

1. `test_diversity_report_has_correlations` — pairwise_correlations has entries for all model pairs
2. `test_ensemble_method_valid` — ensemble_method in ["simple_blend", "greedy_forward", "hill_climbing", "rank_average", "single_model"]
3. `test_ensemble_beats_or_equals_single` — ensemble_cv >= best single model cv (by construction)
4. `test_single_model_when_no_improvement` — if no technique beats single → ensemble_method="single_model"
5. `test_hill_climbing_converges` — weights sum to ~1.0, all weights in [0, 1]
6. `test_rank_average_in_0_1` — rank-averaged predictions in [0, 1] range
7. `test_greedy_forward_selects_subset` — selected models is a subset of all models
8. `test_diversity_pruning_removes_corr_above_98` — two nearly identical models → one pruned
9. `test_v1_ensemble_contracts_pass` — all existing v1 contracts unchanged

---

## COMMIT 7: Post-Processing Optimizer (agents/post_processor.py)

### What this does

Metric-specific output optimization. Threshold sweep for binary classification, OptimizedRounder for QWK, Platt scaling for probability calibration, prediction clipping for bounded targets.

### The LangGraph node function

```python
def post_processor(state: ProfessorState) -> dict:
    """
    Metric-specific post-processing on ensemble predictions.
    
    Reads: ensemble_cv, oof_predictions_path, test_predictions_path,
           metric_name, competition_type, domain_brief
    Writes: postprocess_config, postprocess_cv_delta
    Emits: STATUS
    """
```

### Metric-specific catalog

```python
POSTPROCESS_CATALOG = {
    # Binary classification
    "roc_auc": ["threshold_sweep"],  # AUC doesn't benefit from post-processing, but threshold for submission
    "log_loss": ["platt_scaling", "isotonic_calibration"],
    "f1": ["threshold_sweep"],
    "accuracy": ["threshold_sweep"],
    "mcc": ["threshold_sweep"],
    
    # Multiclass
    "f1_macro": ["temperature_scaling"],
    "f1_micro": ["temperature_scaling"],
    
    # Regression
    "rmse": ["prediction_clipping"],
    "rmsle": ["prediction_clipping_positive"],  # Clip to >= 0
    "mae": ["prediction_clipping"],
    
    # Ordinal
    "qwk": ["optimized_rounder"],
    
    # Ranking
    "ndcg": [],  # No standard post-processing
    "map_at_k": [],
}
```

**Threshold sweep (binary):**
```python
def _threshold_sweep(oof_preds, y_true, scorer, n_points=200):
    thresholds = np.linspace(0.01, 0.99, n_points)
    best_threshold = 0.5
    best_score = scorer(y_true, (oof_preds > 0.5).astype(int))
    
    for t in thresholds:
        binary_preds = (oof_preds > t).astype(int)
        score = scorer(y_true, binary_preds)
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold, best_score
```

**OptimizedRounder (QWK):**
```python
def _optimized_rounder(oof_preds, y_true, n_classes):
    """Find optimal rounding boundaries for ordinal predictions."""
    from scipy.optimize import minimize
    
    initial_boundaries = np.arange(0.5, n_classes - 0.5, 1.0)
    
    def qwk_loss(boundaries):
        rounded = np.digitize(oof_preds, sorted(boundaries))
        return -cohen_kappa_score(y_true, rounded, weights="quadratic")
    
    result = minimize(qwk_loss, initial_boundaries, method="Nelder-Mead")
    optimal_boundaries = sorted(result.x)
    return optimal_boundaries
```

**Platt scaling:**
```python
def _platt_scaling(oof_preds, y_true):
    """Calibrate probabilities using logistic regression on OOF predictions."""
    from sklearn.calibration import CalibratedClassifierCV
    # Or simple LR: fit LR on oof_preds → y_true, apply to test preds
    lr = LogisticRegression()
    lr.fit(oof_preds.reshape(-1, 1), y_true)
    return lr  # Apply lr.predict_proba on test predictions
```

**Prediction clipping:**
```python
def _prediction_clipping(oof_preds, y_true, domain_constraints=None):
    """Clip predictions to valid range."""
    min_target = y_true.min()
    max_target = y_true.max()
    
    # Use domain constraints if available
    if domain_constraints:
        for constraint in domain_constraints:
            if constraint["type"] == "target_range":
                min_target = max(min_target, constraint.get("min", min_target))
                max_target = min(max_target, constraint.get("max", max_target))
    
    return np.clip(oof_preds, min_target, max_target)
```

### Apply and validate

For each applicable post-processing technique:
1. Apply to OOF predictions
2. Score on competition metric
3. Compare to pre-post-processing score
4. Select the best improvement (or none if no improvement)

```python
postprocess_cv_delta = postprocessed_score - ensemble_cv
if postprocess_cv_delta > 0:
    # Apply same post-processing to test predictions
    # Save updated predictions
    pass
else:
    # No improvement — skip post-processing
    postprocess_config = {"method": "none", "delta": 0.0}
```

### Contract tests: tests/contracts/test_post_processor_contract.py

1. `test_postprocess_config_has_method` — postprocess_config has "method" key
2. `test_cv_delta_is_float` — postprocess_cv_delta is a float
3. `test_threshold_sweep_finds_optimal` — inject known-optimal threshold, verify it's found
4. `test_optimized_rounder_for_qwk` — metric="qwk" → optimized_rounder method used
5. `test_platt_scaling_for_logloss` — metric="log_loss" → platt_scaling considered
6. `test_clipping_for_rmse` — metric="rmse" → prediction_clipping applied
7. `test_no_postprocessing_when_no_improvement` — if all techniques score worse → method="none"
8. `test_predictions_stay_valid_after_postprocessing` — no NaN, correct row count, valid range
9. `test_sprint_mode_skipped` — pipeline_depth="sprint" → post_processor skipped

---

## COMMIT 8: Integration Test

### tests/contracts/test_layer3_integration.py

```python
def test_layer3_full_sequence(mock_features_dir):
    """
    Run: gate_config → ml_optimizer → red_team_critic → self_reflection 
         → ensemble_architect → post_processor
    Verify state flows correctly between them.
    """
    state = ProfessorState(
        session_id="test-layer3",
        features_train_path=str(mock_features_dir / "features_train.parquet"),
        features_test_path=str(mock_features_dir / "features_test.parquet"),
        target_column="target",
        competition_type="binary",
        metric_name="roc_auc",
        canonical_train_rows=5000,
        canonical_test_rows=1250,
        validation_strategy={"cv_type": "StratifiedKFold", "n_splits": 5},
        feature_manifest=[{"name": "feat_1", "source": "feature_factory"}],
        pipeline_depth="standard",
        shift_severity="clean",
        gate_config=get_gate_config(5000),
    )
    
    with patch("tools.llm_provider.llm_call") as mock_llm, \
         patch("tools.sandbox.run_in_sandbox") as mock_sandbox:
        
        # ML Optimizer
        updates = ml_optimizer(state)
        state = state.copy(update=updates)
        assert len(state.model_configs) == 3
        assert state.cv_mean > 0
        
        # Critic
        updates = red_team_critic(state)
        state = state.copy(update=updates)
        assert "severity" in state.critic_verdict
        assert len(state.critic_verdict.get("findings", [])) >= 4  # At least core vectors
        
        # Self-Reflection
        updates = self_reflection(state)
        state = state.copy(update=updates)
        assert len(state.reflection_notes) >= 1
        
        # Ensemble
        updates = ensemble_architect(state)
        state = state.copy(update=updates)
        assert state.ensemble_method != ""
        assert state.ensemble_cv >= state.cv_mean  # Ensemble >= single model
        
        # Post-Processing
        updates = post_processor(state)
        state = state.copy(update=updates)
        assert "method" in state.postprocess_config


def test_critic_replan_flow():
    """Critic CONFIRMED_CRITICAL → replan_requested=True → Supervisor replans."""
    # Inject known leakage
    # Verify CONFIRMED_CRITICAL after confirmation checks
    # Verify replan targets correct agents


def test_sprint_skips_correct_components():
    """SPRINT mode: Critic 4 vectors, post_processor skipped."""
    state = ProfessorState(pipeline_depth="sprint", ...)
    # Run critic → verify 4 vectors only
    # Run post_processor → verify immediate return
```

---

## WHAT NOT TO DO

- Do NOT use RandomForest for Critic Vector 1a. Use LogisticRegression. The architecture document explains why in detail.
- Do NOT skip confirmation checks on CRITICAL findings. Every CRITICAL gets 2 re-runs before triggering replan.
- Do NOT let dynamic rules exceed 20 active. Cap and evict lowest-confidence.
- Do NOT let auto-calibration relax Wilcoxon threshold beyond p=0.15. Hard cap.
- Do NOT let rules contradict Golden Rules. Check before promoting.
- Do NOT use Optuna for ensemble weight optimization. Hill climbing is more robust when n_models < 10.
- Do NOT skip multi-seed stability validation. Every model gets 3 seeds.
- Do NOT apply STABILITY_PENALTY constant as anything other than 1.5. It's locked.
- Do NOT modify Layer 0-2 files except where explicitly documented (Feature Factory gate_config integration).
- Do NOT use Pandas. All data operations in Polars. sklearn/scipy accept numpy arrays — convert at the model boundary.