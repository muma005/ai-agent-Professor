# Professor — Autonomous Kaggle Agent

Stateful Hierarchical Multi-Agent Graph for production-grade data science.

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
