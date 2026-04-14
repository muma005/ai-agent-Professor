# Professor — Autonomous Kaggle Agent

Stateful Hierarchical Multi-Agent Graph for production-grade data science.

**Current build phase: Day 22 — Ensemble Architect + Phase 3 Regression Freeze**

---

## Time Savings Analysis

A serious competition run has roughly these time buckets. Here's what Professor does to each:

| Task | Typical time | Professor's effect | Time remaining |
|---|---|---|---|
| EDA — distributions, correlations, outliers | 4–8h | **Fully automated** → `eda_report.json` | ~30min to review |
| Forum + notebook intel gathering | 3–6h/week | **Fully automated** → `intel_brief.json` | ~20min to review |
| CV strategy selection | 1–3h | **Fully automated** → `validation_strategy.json` | ~10min to review |
| Preprocessing implementation | 3–5h | **Mostly automated** (data_engineer generates code) | ~1h reviewing output |
| Leakage detection | 2–4h | **Fully automated** → critic runs 6 vectors | ~15min reviewing verdict |
| Hyperparameter tuning | 10–20h (compute + monitoring) | **Fully automated** (Optuna, runs overnight) | ~30min reviewing results |
| Standard feature engineering | 8–15h | **Partially automated** (feature_factory generates, iterates) | ~3–5h domain work |
| Ensemble design + blending | 3–5h | **Mostly automated** (ensemble_architect) | ~1h reviewing weights |
| Submission timing + risk decisions | 2–4h | **Advised, not decided** (submission_strategist) | ~2h (unchanged) |
| Post-mortem + documentation | 2–4h | **Fully automated** (post_mortem_agent, Day 11) | ~20min reviewing |
| HITL decision-making | 0h (doesn't exist now) | **New task created** | ~1–3h per competition |

**Rough total time: ~40–75h per competition → ~10–18h with Professor**

That's approximately **70–75% reduction in execution time.**

---

## What Changes Most About Your Role

**Before Professor:** Most of your time is execution. Running pandas, debugging CV splits, writing feature engineering scripts, waiting for tuning runs, re-running after leakage is discovered at 2am.

**After Professor:** Your time is judgment. Reviewing reports Professor generated. Injecting the domain insight Professor can't have. Making the calls Professor flags as requiring a human. Reading the one forum post that changes everything.

The difference between a top 3 finish and a top 15 finish is almost never EDA quality or CV strategy. It's one good feature that nobody else found, or one submission timing decision that avoided a shakeup. Professor removes the work that separates serious competitors from casual ones. It doesn't remove the work that separates gold medalists from silver medalists — that's still you.

---

## Quick Start

```bash
# Activate virtual environment
venv\Scripts\activate           # Windows
source venv/bin/activate        # Mac/Linux

# Verify environment
python main.py check

# Start a competition run
python main.py run --competition spaceship-titanic --data ./data/spaceship_titanic/
```

## Architecture

Professor is a LangGraph-based multi-agent system where specialised agents handle each phase of a Kaggle competition pipeline:

| Node | Agent | Role |
|------|-------|------|
| 0 | Semantic Router | Supervisor — plans, routes, world model |
| 1 | Competition Intel | Research + external data scout |
| 2 | Validation Architect | CV strategy + Metric Contract |
| 3 | Data Engineer | CSV → Parquet + data hash |
| 3b | EDA Agent | Target dist, correlations, outliers, leakage fingerprint |
| 4 | Feature Factory | Hypothesis-driven features + interaction cap |
| 5 | ML Optimizer | Model portfolio + Optuna + calibration |
| 6 | Red Team Critic | 4-vector adversarial audit |
| 7 | Ensemble Architect | Diversity-first stacking + Wilcoxon gate |
| 8 | Publisher | Structured report + slot injection |
| 9 | Submission Strategist | EWMA LB gap + competition context update |

## Day 22 — Ensemble Architect

The Ensemble Architect (`agents/ensemble_architect.py`) runs after `ml_optimizer` completes and all model variants are in `model_registry`. It implements:

1. **Data hash validation** — filters stale models before any computation
2. **OOF validation** — verifies prediction shapes match training data
3. **Diversity pruning** — greedy selection; rejects any model with Pearson correlation > 0.98 with already-selected models
4. **Holdout split** — 80/20 stratified split (seed=42); holdout never used in weight optimisation
5. **Constrained Optuna weights** — softmax normalisation, clip ≥ 0.05, renormalise; `n_trials=50`, `n_jobs=1`, `gc_after_trial=True`
6. **Stacking meta-learner** — `LogisticRegression(C=0.1)` or `Ridge(alpha=10.0)` with 5-fold CV; selects whichever beats the other on holdout
7. **Wilcoxon validation gate** — ensemble must significantly beat best single model (p < 0.05) or falls back to single model
8. **Holdout scoring** — final unbiased quality estimate
9. **Lineage logging** — full selection metadata for auditability

### Contract Tests

24 immutable contract tests in `tests/contracts/test_ensemble_architect_contract.py` (class: `TestEnsembleArchitectContract`) verify:
- Hash validation runs before anything else
- OOF validation before weight optimisation
- Diversity pruning completes before Optuna starts (timestamp-proven)
- No pair with correlation > 0.98 in final ensemble
- Weights sum to 1.0 (tolerance 1e-6)
- No weight below 0.05
- Holdout indices have zero overlap with optimisation pool
- Wilcoxon gate called exactly once
- ensemble_oof length equals len(y_train)
- All required state keys present and correctly typed

### Phase 3 Regression Freeze

`tests/regression/test_phase3_regression.py` freezes 7 capability floors after Phase 3 gate:
- Phase 1/2 CV floors still hold
- CV score floor from gate result (gate_cv - 0.020)
- Null importance filter is running and removing features
- Optuna stability ranking beats peak ranking
- All 4 core Critic vectors fire on injected failures
- Wilcoxon gate rejects noise-level differences
- Ensemble diversity pruning enforced (no pair > 0.98)

## Competition Failure Diagnostic Checklist

When Professor underperforms on a real competition, run this
before any architectural conclusion:

1. Check critic_verdict.json — any CRITICAL/HIGH flags? Fix those first.
2. Verify metric_contract.json matches competition metric.
3. Diagnose CV/LB gap — high CV + low LB = leakage.
4. Check eda_report.json — any unaddressed flags?
5. Check feature_manifest.json — too few or too many features?
6. Check validation_strategy.json — wrong CV split for this data type?

Fix the specific component identified. Nothing else.
Architecture is not wrong. One component is off.

## Emergency Regression Protocol

1. Stop. Write no more code.
2. git log --oneline — find last green commit.
3. git checkout [hash] -b diagnosis — confirm tests pass.
4. git diff [hash] phase-N — bug is in this diff, nowhere else.
5. Fix one line. One commit. Run tests.
6. If not found in 2 hours — paste failing test + diff to Claude.

## Project Structure

```
professor-agent/
├── core/          # State, graph builder, metric contract, lineage
├── agents/        # Every node in the graph
├── tools/         # Callable functions agents use (LLM client, sandbox, etc.)
├── guards/        # Safety systems (circuit breaker, cost tracker, etc.)
├── memory/        # Redis state + ChromaDB vector memory
├── tests/         # contracts/ regression/ integration/
├── data/          # Local data — never commit large files
├── outputs/       # Logs, models, predictions, submissions, reports
└── notebooks/     # Manual sanity checks
```

## Dependencies

All dependencies are pinned in `requirements.txt`. **DO NOT** run `pip upgrade` during the 30-day build. Library updates are a Month 2 activity,
