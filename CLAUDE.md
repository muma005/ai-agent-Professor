# CLAUDE.md — Professor Agent Instructions

## Lightning AI — Cloud Compute Offload

Your laptop gives Lightning the data and the Python script.
Lightning brings back a JSON.
Your laptop reads the JSON and continues.

| Task | File | Local time | Lightning CPU | Spot-safe |
|---|---|---|---|---|
| EDA (correlations + mutual info) | eda_agent.py | 10-30 min | 2-5 min | YES |
| Feature testing Stage 1 | feature_factory.py | 20-60 min | 5-15 min | YES |
| Optuna HPO | ml_optimizer.py | 60-180 min | 8-25 min | YES |
| Null importance Stage 2 | null_importance.py | 15-45 min | 5-8 min | YES |
| Stability validator | stability_validator.py | 20-40 min | 5-10 min | YES |
| Critic permutation + adversarial | red_team_critic.py | 15-45 min | 4-12 min | YES |
| Pseudo-labeling (3 iters) | pseudo_label_agent.py | 15-45 min | 5-15 min | YES |
| Ensemble meta-learner | ensemble_architect.py | 5-20 min | 2-5 min | YES |
| Harness (full pipeline × 3) | harness_runner.py | 3-18 hours | 1-4 hours | NO |

### How to use

1. Set credentials in `.env` (see `.env.example`).
2. Enable one task at a time. Start with `LIGHTNING_OFFLOAD_OPTUNA=1`.
3. Confirm it works before enabling more.

### Hard rules

- All Lightning paths have local fallback.
- Setting all flags to 0 makes Professor run exactly as before.
- `n_jobs=1` in every `study.optimize()` call. Never change this.
- `gc_after_trial=True` in every `study.optimize()` call. Never change this.
- `interruptible=True` for all tasks except `run_harness.py`.
- Test data (`test.csv`) never goes to Lightning for Optuna/stability/EDA.
- All existing tests must pass with all Lightning flags set to 0.

### Environment variables

```bash
# Required credentials
LIGHTNING_API_KEY=
LIGHTNING_USER_ID=
LIGHTNING_USERNAME=
LIGHTNING_STUDIO_NAME=professor-agent-studio
LIGHTNING_TEAMSPACE=

# Per-task offload flags (0=local, 1=Lightning)
LIGHTNING_OFFLOAD_OPTUNA=0
LIGHTNING_OFFLOAD_NULL_IMPORTANCE=0
LIGHTNING_OFFLOAD_EDA=0
LIGHTNING_OFFLOAD_FEATURE_TESTING=0
LIGHTNING_OFFLOAD_STABILITY=0
LIGHTNING_OFFLOAD_CRITIC=0
LIGHTNING_OFFLOAD_PSEUDO_LABEL=0
LIGHTNING_OFFLOAD_ENSEMBLE=0
LIGHTNING_OFFLOAD_HARNESS=0

# Per-task machine type (CPU | L4 | L40S | A100)
LIGHTNING_OPTUNA_MACHINE=CPU
LIGHTNING_NULL_IMPORTANCE_MACHINE=CPU
LIGHTNING_EDA_MACHINE=CPU
LIGHTNING_FEATURE_TESTING_MACHINE=CPU
LIGHTNING_STABILITY_MACHINE=CPU
LIGHTNING_CRITIC_MACHINE=CPU
LIGHTNING_PSEUDO_LABEL_MACHINE=CPU
LIGHTNING_ENSEMBLE_MACHINE=CPU
LIGHTNING_HARNESS_MACHINE=CPU
```

### Architecture

```
┌────────────────┐        ┌─────────────────────┐
│  Your Laptop   │        │   Lightning AI       │
│  (Orchestrator)│───────▶│   (Compute Worker)   │
│                │  data  │                      │
│  Reads JSON ◀──│────────│  Returns JSON result │
│  Continues...  │        │  (scores, configs)   │
└────────────────┘        └─────────────────────┘
```

### Verification

```bash
# Verify imports work without credentials
python -c "
from tools.lightning_runner import is_lightning_configured, run_on_lightning
print('is_configured:', is_lightning_configured())
result = run_on_lightning('any.py', {}, 'test')
print('never_raises:', 'success' in result)
"

# Run Lightning-specific tests
pytest tests/test_lightning_integration.py -v
```
