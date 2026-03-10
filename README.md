# Professor — Autonomous Kaggle Agent
Good question to ask now, while we can still adjust the build plan if the answer surprises you. Let me give you a precise answer, not an optimistic one.

---

## The Honest Breakdown

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

## But Here's What That Number Hides

The 25–30% that remains is not arbitrary leftovers. It's the work that actually determines whether you finish top 5% or top 15%.

**What you still own entirely:**

**1. Domain knowledge injection.** Professor's feature factory generates statistically-motivated features. It doesn't know that in wildfire prediction, `wind_speed × slope_aspect` matters more than either variable alone, or that `days_since_last_rain` is a better predictor than raw humidity. That kind of insight comes from you reading the problem, thinking about the physical process, and injecting it. This is still 3–5 hours of creative work per competition and it's disproportionately high-leverage.

**2. External dataset decisions.** Professor's intel scraper will tell you which external datasets competitors are using. Downloading, understanding, joining, and cleaning those datasets is still mostly you. Professor can help once the data is on disk, but it can't retrieve it for you.

**3. HITL decisions.** Professor is designed to halt and ask rather than proceed when something is wrong. Every HITL alert requires you to read the reason, look at the evidence, decide whether to override or fix, and resume. This is a new task that didn't exist before Professor, and it's genuinely important — these are the moments where bad models get stopped before they become bad submissions.

**4. Reading novel competition-specific quirks.** Forums have signal that the LLM synthesis misses — the subtle post from a host that clarifies an edge case in the metric, or the early notebook from a GM that has a non-obvious data structure insight. The intel brief gives you 80% of what's there. The remaining 20% still requires you to read, and that 20% sometimes decides the leaderboard.

**5. Final submission calls.** Professor can tell you the shakeup risk is high and the conservative strategy is to not push. You decide whether to trust that or push anyway. The machine advises, you decide.

---

## What Changes Most About Your Role

The bigger shift isn't percentage of tasks — it's the *type* of work you do.

**Before Professor:** Most of your time is execution. Running pandas, debugging CV splits, writing feature engineering scripts, waiting for tuning runs, re-running after leakage is discovered at 2am.

**After Professor:** Your time is judgment. Reviewing reports Professor generated. Injecting the domain insight Professor can't have. Making the calls Professor flags as requiring a human. Reading the one forum post that changes everything.

The difference between a top 3 finish and a top 15 finish is almost never EDA quality or CV strategy. It's one good feature that nobody else found, or one submission timing decision that avoided a shakeup. Professor removes the work that separates serious competitors from casual ones. It doesn't remove the work that separates gold medalists from silver medalists — that's still you.

---



That last point is a strategic call worth several leaderboard positions, and it's exactly the kind of thing that stays with you.
Stateful Hierarchical Multi-Agent Graph for production-grade data science.

Good question to ask now, while we can still adjust the build plan if the answer surprises you. Let me give you a precise answer, not an optimistic one.

---

## The Honest Breakdown

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

## But Here's What That Number Hides

The 25–30% that remains is not arbitrary leftovers. It's the work that actually determines whether you finish top 5% or top 15%.

**What you still own entirely:**

**1. Domain knowledge injection.** Professor's feature factory generates statistically-motivated features. It doesn't know that in wildfire prediction, `wind_speed × slope_aspect` matters more than either variable alone, or that `days_since_last_rain` is a better predictor than raw humidity. That kind of insight comes from you reading the problem, thinking about the physical process, and injecting it. This is still 3–5 hours of creative work per competition and it's disproportionately high-leverage.

**2. External dataset decisions.** Professor's intel scraper will tell you which external datasets competitors are using. Downloading, understanding, joining, and cleaning those datasets is still mostly you. Professor can help once the data is on disk, but it can't retrieve it for you.

**3. HITL decisions.** Professor is designed to halt and ask rather than proceed when something is wrong. Every HITL alert requires you to read the reason, look at the evidence, decide whether to override or fix, and resume. This is a new task that didn't exist before Professor, and it's genuinely important — these are the moments where bad models get stopped before they become bad submissions.

**4. Reading novel competition-specific quirks.** Forums have signal that the LLM synthesis misses — the subtle post from a host that clarifies an edge case in the metric, or the early notebook from a GM that has a non-obvious data structure insight. The intel brief gives you 80% of what's there. The remaining 20% still requires you to read, and that 20% sometimes decides the leaderboard.

**5. Final submission calls.** Professor can tell you the shakeup risk is high and the conservative strategy is to not push. You decide whether to trust that or push anyway. The machine advises, you decide.

---

## What Changes Most About Your Role

The bigger shift isn't percentage of tasks — it's the *type* of work you do.

**Before Professor:** Most of your time is execution. Running pandas, debugging CV splits, writing feature engineering scripts, waiting for tuning runs, re-running after leakage is discovered at 2am.

**After Professor:** Your time is judgment. Reviewing reports Professor generated. Injecting the domain insight Professor can't have. Making the calls Professor flags as requiring a human. Reading the one forum post that changes everything.

The difference between a top 3 finish and a top 15 finish is almost never EDA quality or CV strategy. It's one good feature that nobody else found, or one submission timing decision that avoided a shakeup. Professor removes the work that separates serious competitors from casual ones. It doesn't remove the work that separates gold medalists from silver medalists — that's still you.

---

That last point is a strategic call worth several leaderboard positions, and it's exactly the kind of thing that stays with you.

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
| 7 | Ensemble Architect | Diversity-first stacking |
| 8 | Publisher | Structured report + slot injection |
| 9 | Submission Strategist | EWMA LB gap + competition context update |

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

All dependencies are pinned in `requirements.txt`. **DO NOT** run `pip upgrade` during the 30-day build. Library updates are a Month 2 activity.
