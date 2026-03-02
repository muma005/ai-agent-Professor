# Day 1 Tasks
Now let me get the Task Board.Let me fetch a few of the individual task pages for the full implementation details.Good — I have everything. Let me now compile the complete Day 1 plan with full implementation.

---

# Day 1 — Complete Implementation Plan

**The ONE thing that must work by end of today:**
Environment is fully set up, all APIs tested, git protection active, and `python main.py` runs without errors.

---

## The 14 Tasks — In Execution Order

```
MORNING BLOCK (06:30 – 09:30)  — Setup + APIs + Structure
  Task 1:  Create project folder + virtual environment
  Task 2:  Install all dependencies + pin versions
  Task 3:  Get Groq API key + test DeepSeek-R1 70B
  Task 4:  Get Google AI Studio key + test Gemini 2.0 Flash
  Task 5:  Install Ollama + pull local fallback model
  Task 6:  Create .env file + .gitignore

AFTERNOON BLOCK (10:00 – 12:30)  — Core Files
  Task 7:  Create full folder structure
  Task 8:  Build tools/llm_client.py
  Task 9:  Write core/state.py
  Task 10: Write main.py CLI entry point

INTEGRATION BLOCK (13:30 – 15:30)  — Services + Verification
  Task 11: Verify RestrictedPython sandbox
  Task 12: Verify ChromaDB persistent storage
  Task 13: Verify fakeredis state
  Task 14: Launch MLflow dashboard

PROTECTION BLOCK (15:30 – 17:00)  — Git + Safeguards
  Task 15: Git branching setup + pre-commit hook
  Task 16: Add SANDBOX_PREAMBLE + POLARS_CONSTRAINT
  Task 17: Write daily log template + README sections
```

---

## Task 1 — Create Project Folder + Virtual Environment

```bash
# Create the project
mkdir professor-agent
cd professor-agent

# Virtual environment
python -m venv venv

# Activate (use this every single day)
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows
```

---

## Task 2 — Install All Dependencies + Pin Versions

```bash
pip install \
  langgraph \
  langchain-core \
  groq \
  google-generativeai \
  polars \
  pyarrow \
  lightgbm \
  optuna \
  scikit-learn \
  chromadb \
  fakeredis \
  mlflow \
  RestrictedPython \
  pre-commit \
  pytest \
  python-dotenv \
  kaggle \
  pandas \
  numpy \
  shap \
  matplotlib

# Immediately pin every version — do this before anything else
pip freeze > requirements.txt
git add requirements.txt
```

Open `requirements.txt` and add this header comment:

```
# ─────────────────────────────────────────────────────────
# PINNED Day 1. All versions confirmed working together.
# DO NOT run pip upgrade during the 30-day build.
# Library updates are a Month 2 activity.
# ─────────────────────────────────────────────────────────
```

---

## Task 3 — Groq API Key + Test DeepSeek-R1 70B

Go to [console.groq.com](https://console.groq.com) → Create account → API Keys → Create key.

Copy the key. Then test it:

```python
# test_groq.py — run this, delete after confirmed working
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="deepseek-r1-distill-llama-70b",
    messages=[{"role": "user", "content": "Reply with: GROQ_WORKING"}],
    max_tokens=20
)
print(response.choices[0].message.content)
# Expected output: GROQ_WORKING
```

```bash
python test_groq.py
# Must print: GROQ_WORKING
# If it does: Groq is live. Delete test_groq.py.
```

---

## Task 4 — Google AI Studio Key + Test Gemini 2.0 Flash

Go to [aistudio.google.com](https://aistudio.google.com) → Get API key → Create API key.

```python
# test_gemini.py — run this, delete after confirmed working
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Reply with: GEMINI_WORKING")
print(response.text)
# Expected output: GEMINI_WORKING
```

```bash
python test_gemini.py
# Must print: GEMINI_WORKING
# Delete test_gemini.py after.
```

---

## Task 5 — Install Ollama + Local Fallback Model

```bash
# Install Ollama (Mac)
brew install ollama

# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Pull the fallback model
# If you have 16GB RAM:
ollama pull deepseek-r1:14b

# If you only have 8GB RAM:
ollama pull deepseek-r1:7b

# Test it
ollama run deepseek-r1:14b "Reply with: OLLAMA_WORKING"
# Must print: OLLAMA_WORKING
```

---

## Task 6 — Create `.env` + `.gitignore`

```bash
# .env — never commit this file
touch .env
```

```bash
# .env contents
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

```bash
# .gitignore
.env
venv/
__pycache__/
*.pyc
*.pkl
*.parquet
data/
memory/chroma/
outputs/models/
outputs/predictions/
.DS_Store
mlruns/
*.db
```

---

## Task 7 — Create Full Folder Structure

```bash
# Run this entire block — creates every folder and file
mkdir -p core agents tools guards memory tests/contracts \
         tests/regression tests/integration \
         data/spaceship_titanic notebooks \
         outputs/logs outputs/briefings outputs/models \
         outputs/predictions outputs/charts \
         outputs/submissions outputs/reports

# Create all __init__.py files
touch core/__init__.py agents/__init__.py tools/__init__.py \
      guards/__init__.py memory/__init__.py tests/__init__.py \
      tests/contracts/__init__.py tests/regression/__init__.py \
      tests/integration/__init__.py

# Create placeholder files so structure is visible in git
touch core/state.py core/professor.py core/metric_contract.py core/lineage.py
touch agents/semantic_router.py agents/competition_intel.py \
      agents/validation_architect.py agents/data_engineer.py \
      agents/eda_agent.py agents/feature_factory.py \
      agents/ml_optimizer.py agents/red_team_critic.py \
      agents/ensemble_architect.py agents/publisher.py \
      agents/submission_strategist.py agents/qa_gate.py \
      agents/pseudo_label_agent.py agents/post_mortem_agent.py
touch tools/llm_client.py tools/e2b_sandbox.py tools/data_tools.py \
      tools/submit_tools.py tools/null_importance.py \
      tools/wilcoxon_gate.py tools/mlflow_tracker.py
touch guards/circuit_breaker.py guards/cost_tracker.py \
      guards/schema_validator.py guards/service_health.py \
      guards/tool_constraints.py guards/lb_monitor.py
touch memory/redis_state.py memory/chroma_memory.py \
      memory/memory_quality.py memory/seed_memory.py
touch outputs/.gitkeep data/.gitkeep

# Verify structure
find . -type f -name "*.py" | head -40
```

---

## Task 8 — Build `tools/llm_client.py`

This is the most important file you write today. Every agent calls this. No agent ever touches the Groq or Gemini SDK directly.

```python
# tools/llm_client.py

import os
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
import requests
import json

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────
_groq_client = None
_gemini_configured = False

def _get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client

def _get_gemini(model_name: str):
    global _gemini_configured
    if not _gemini_configured:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _gemini_configured = True
    return genai.GenerativeModel(model_name)

# ── Polars constraint injected into all code-generating prompts ───
POLARS_CONSTRAINT = """
LIBRARY REQUIREMENT: This pipeline uses Polars not Pandas.
CORRECT:   pl.read_csv()  df.write_parquet()  df.fill_null()  df.group_by()
INCORRECT: pd.read_csv()  df.to_parquet()     df.fillna()     df.groupby()
If pandas is required for a specific library call:
  convert back with pl.from_pandas(df) before returning.
"""

# ── Main call function ────────────────────────────────────────────
def call_llm(
    prompt: str,
    system: str = "",
    model: str = "groq-deepseek",
    max_tokens: int = 4096,
    is_coding_task: bool = False
) -> str:
    """
    Unified LLM interface. All agents call this. Never call Groq/Gemini directly.

    Models:
      "groq-deepseek"  → DeepSeek-R1 70B on Groq (default, all reasoning agents)
      "groq-llama"     → Llama 3.3 70B on Groq (routing, fast tasks)
      "gemini-flash"   → Gemini 2.0 Flash (Publisher, QA Gate)
      "ollama"         → DeepSeek-R1 14b local (fallback only)
    """

    # Inject Polars constraint into all coding task prompts
    if is_coding_task:
        system = POLARS_CONSTRAINT + "\n\n" + system

    # ── Groq: DeepSeek-R1 70B ─────────────────────────────────────
    if model == "groq-deepseek":
        try:
            return _call_groq(
                prompt=prompt,
                system=system,
                model_name="deepseek-r1-distill-llama-70b",
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"[llm_client] Groq DeepSeek failed: {e}. Falling back to Gemini.")
            return _call_gemini(prompt, system, max_tokens)

    # ── Groq: Llama 3.3 70B ───────────────────────────────────────
    elif model == "groq-llama":
        try:
            return _call_groq(
                prompt=prompt,
                system=system,
                model_name="llama-3.3-70b-versatile",
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"[llm_client] Groq Llama failed: {e}. Falling back to DeepSeek.")
            return _call_groq(prompt, system, "deepseek-r1-distill-llama-70b", max_tokens)

    # ── Gemini 2.0 Flash ──────────────────────────────────────────
    elif model == "gemini-flash":
        try:
            return _call_gemini(prompt, system, max_tokens)
        except Exception as e:
            print(f"[llm_client] Gemini failed: {e}. Falling back to Groq.")
            return _call_groq(prompt, system, "deepseek-r1-distill-llama-70b", max_tokens)

    # ── Ollama local fallback ──────────────────────────────────────
    elif model == "ollama":
        return _call_ollama(prompt, system)

    else:
        raise ValueError(f"Unknown model: {model}. Use groq-deepseek, groq-llama, gemini-flash, or ollama.")


def _call_groq(prompt: str, system: str, model_name: str, max_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_groq().chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, system: str, max_tokens: int) -> str:
    model = _get_gemini("gemini-2.0-flash")
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    response = model.generate_content(
        full_prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": 0.1}
    )
    return response.text


def _call_ollama(prompt: str, system: str) -> str:
    payload = {
        "model": "deepseek-r1:14b",
        "messages": [],
        "stream": False
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})

    response = requests.post("http://localhost:11434/api/chat", json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]
```

**Test it immediately:**

```python
# Quick test — paste in terminal
from tools.llm_client import call_llm
result = call_llm("Reply with: LLM_CLIENT_WORKING", model="groq-deepseek")
print(result)
# Must print something containing: LLM_CLIENT_WORKING
```

---

## Task 9 — Write `core/state.py`

```python
# core/state.py

from typing import TypedDict, Optional, Any
import uuid
from datetime import datetime


class CostTracker(TypedDict):
    total_usd: float
    groq_tokens_in: int
    groq_tokens_out: int
    gemini_tokens: int
    llm_calls: int
    budget_usd: float
    warning_threshold: float   # 0.70 — warn at 70% budget
    throttle_threshold: float  # 0.85 — throttle at 85%
    triage_threshold: float    # 0.95 — HITL at 95%


class CompetitionContext(TypedDict):
    competition_name: str
    days_remaining: Optional[int]
    current_rank: Optional[int]
    total_teams: Optional[int]
    submission_count: int
    submission_limit: int
    public_lb_score: Optional[float]
    best_cv_score: Optional[float]
    lb_cv_gap: Optional[float]
    shakeup_risk: Optional[str]   # "low" | "medium" | "high"


class ProfessorState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────
    session_id: str              # namespaces ALL resources for this run
    created_at: str

    # ── Competition ───────────────────────────────────────────────
    competition_name: str
    task_type: str               # "tabular_classification" | "tabular_regression" | "timeseries" | "auto"
    competition_context: Optional[CompetitionContext]

    # ── Data (pointers only — never raw DataFrames in state) ──────
    raw_data_path: str
    clean_data_path: Optional[str]
    schema_path: Optional[str]
    eda_report_path: Optional[str]
    data_hash: Optional[str]     # SHA-256 of source file, first 16 chars

    # ── Feature Engineering ───────────────────────────────────────
    feature_manifest: Optional[list]
    feature_factory_checkpoint: Optional[dict]

    # ── Validation ────────────────────────────────────────────────
    cv_strategy: Optional[dict]
    metric_contract: Optional[dict]
    cv_scores: Optional[list]
    cv_mean: Optional[float]

    # ── Models ────────────────────────────────────────────────────
    model_registry: Optional[list]
    best_params: Optional[dict]
    optuna_study_path: Optional[str]

    # ── Critic ────────────────────────────────────────────────────
    critic_verdict: Optional[dict]

    # ── Ensemble ──────────────────────────────────────────────────
    ensemble_weights: Optional[dict]
    oof_predictions_path: Optional[str]
    test_predictions_path: Optional[str]

    # ── Submission ────────────────────────────────────────────────
    submission_path: Optional[str]
    submission_log: Optional[list]

    # ── Routing ───────────────────────────────────────────────────
    dag: Optional[list]
    current_node: Optional[str]
    next_node: Optional[str]
    error_count: int
    escalation_level: str        # "micro" | "macro" | "hitl" | "triage"

    # ── Budget ────────────────────────────────────────────────────
    cost_tracker: CostTracker

    # ── Output ────────────────────────────────────────────────────
    report_path: Optional[str]
    lineage_log_path: Optional[str]


def initial_state(
    competition: str,
    data_path: str,
    budget_usd: float = 2.00,
    task_type: str = "auto"
) -> ProfessorState:
    """Create a fresh state for a new competition run."""

    session_id = f"{competition[:8].replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

    return ProfessorState(
        session_id=session_id,
        created_at=datetime.utcnow().isoformat(),
        competition_name=competition,
        task_type=task_type,
        competition_context=None,
        raw_data_path=data_path,
        clean_data_path=None,
        schema_path=None,
        eda_report_path=None,
        data_hash=None,
        feature_manifest=None,
        feature_factory_checkpoint=None,
        cv_strategy=None,
        metric_contract=None,
        cv_scores=None,
        cv_mean=None,
        model_registry=None,
        best_params=None,
        optuna_study_path=None,
        critic_verdict=None,
        ensemble_weights=None,
        oof_predictions_path=None,
        test_predictions_path=None,
        submission_path=None,
        submission_log=None,
        dag=None,
        current_node=None,
        next_node=None,
        error_count=0,
        escalation_level="micro",
        cost_tracker=CostTracker(
            total_usd=0.0,
            groq_tokens_in=0,
            groq_tokens_out=0,
            gemini_tokens=0,
            llm_calls=0,
            budget_usd=budget_usd,
            warning_threshold=0.70,
            throttle_threshold=0.85,
            triage_threshold=0.95
        ),
        report_path=None,
        lineage_log_path=f"outputs/logs/{session_id}.jsonl"
    )
```

---

## Task 10 — Write `main.py`

```python
# main.py

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="Professor — Autonomous Kaggle Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run --competition spaceship-titanic --data ./data/spaceship_titanic/
  python main.py run --competition spaceship-titanic --data ./data/spaceship_titanic/ --budget 2.00
  python main.py status --session spaceship_abc123
  python main.py history
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── run command ───────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Start a competition run")
    run_parser.add_argument("--competition", required=True, help="Competition name")
    run_parser.add_argument("--data", required=True, help="Path to data folder")
    run_parser.add_argument("--budget", type=float, default=2.00, help="Budget in USD (default: 2.00)")
    run_parser.add_argument("--task-type", default="auto",
                            choices=["auto", "tabular_classification",
                                     "tabular_regression", "timeseries"],
                            help="Task type (default: auto-detect)")

    # ── status command ────────────────────────────────────────────
    status_parser = subparsers.add_parser("status", help="Check status of a running session")
    status_parser.add_argument("--session", required=True, help="Session ID")

    # ── history command ───────────────────────────────────────────
    subparsers.add_parser("history", help="List all past runs")

    # ── check command (verify environment) ───────────────────────
    subparsers.add_parser("check", help="Verify environment — run this on Day 1")

    args = parser.parse_args()

    if args.command == "run":
        _run(args)

    elif args.command == "status":
        _status(args.session)

    elif args.command == "history":
        _history()

    elif args.command == "check":
        _check_environment()

    else:
        parser.print_help()


def _run(args):
    from core.state import initial_state

    # Validate data path
    if not os.path.exists(args.data):
        print(f"[ERROR] Data path does not exist: {args.data}")
        sys.exit(1)

    state = initial_state(
        competition=args.competition,
        data_path=args.data,
        budget_usd=args.budget,
        task_type=args.task_type
    )

    print(f"[Professor] Starting session: {state['session_id']}")
    print(f"[Professor] Competition:      {state['competition_name']}")
    print(f"[Professor] Data path:        {state['raw_data_path']}")
    print(f"[Professor] Budget:           ${state['cost_tracker']['budget_usd']:.2f}")
    print(f"[Professor] Task type:        {state['task_type']}")
    print(f"[Professor] Log:              {state['lineage_log_path']}")
    print()

    # Graph will be wired starting Day 2
    # For now: confirm state initialises correctly
    print(f"[Professor] State initialised. Pipeline coming Day 2.")


def _status(session_id: str):
    log_path = f"outputs/logs/{session_id}.jsonl"
    if not os.path.exists(log_path):
        print(f"[ERROR] No session found: {session_id}")
        return

    with open(log_path) as f:
        lines = f.readlines()
    print(f"[Status] Session {session_id}: {len(lines)} log entries")


def _history():
    log_dir = "outputs/logs"
    if not os.path.exists(log_dir):
        print("[History] No runs yet.")
        return
    sessions = [f.replace(".jsonl", "") for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    if not sessions:
        print("[History] No completed runs.")
    for s in sessions:
        print(f"  {s}")


def _check_environment():
    """Verify all Day 1 setup is correct. Run: python main.py check"""
    import importlib
    print("[Check] Verifying Professor environment...\n")
    ok = True

    # Check API keys
    for key in ["GROQ_API_KEY", "GOOGLE_API_KEY"]:
        val = os.getenv(key)
        if val:
            print(f"  ✓ {key} present")
        else:
            print(f"  ✗ {key} MISSING — add to .env")
            ok = False

    # Check critical libraries
    libs = ["langgraph", "groq", "google.generativeai", "polars",
            "lightgbm", "optuna", "chromadb", "fakeredis", "mlflow",
            "RestrictedPython"]
    for lib in libs:
        try:
            importlib.import_module(lib.replace("-", "_"))
            print(f"  ✓ {lib}")
        except ImportError:
            print(f"  ✗ {lib} NOT INSTALLED")
            ok = False

    # Check folder structure
    required_dirs = ["core", "agents", "tools", "guards", "memory",
                     "tests/contracts", "outputs/logs", "data"]
    for d in required_dirs:
        if os.path.exists(d):
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/ MISSING")
            ok = False

    print()
    if ok:
        print("[Check] ✓ Environment ready. Start building.")
    else:
        print("[Check] ✗ Fix the issues above before proceeding.")


if __name__ == "__main__":
    main()
```

---

## Task 11-13 — Verify Services

```python
# Paste each block in a Python terminal and confirm output

# ── RestrictedPython ──────────────────────────────────────────────
from RestrictedPython import compile_restricted, safe_globals
code = compile_restricted("result = 2 + 2", "<string>", "exec")
glb = dict(safe_globals)
exec(code, glb)
print(glb.get("result"))  # Must print: 4

# ── ChromaDB ──────────────────────────────────────────────────────
import chromadb
client = chromadb.PersistentClient(path="memory/chroma")
col = client.get_or_create_collection("test")
col.add(documents=["test document"], ids=["id1"])
result = col.query(query_texts=["test"], n_results=1)
print(result["documents"])  # Must print: [['test document']]
# ChromaDB is working and persisting to disk ✓

# ── fakeredis ─────────────────────────────────────────────────────
import fakeredis
r = fakeredis.FakeRedis()
r.set("test_key", "test_value")
print(r.get("test_key").decode())  # Must print: test_value
```

---

## Task 14 — Launch MLflow

```bash
mlflow ui --port 5000 &
# Open browser: http://localhost:5000
# Must see the MLflow dashboard
# This runs in background — no experiments yet, that's fine
```

---

## Task 15 — Git Setup + Pre-commit Hook

```bash
# Initialise git (if not done)
git init
git add .
git commit -m "Day 1: Initial project structure — environment confirmed working"

# Create main branch protection
git checkout -b main 2>/dev/null || git checkout main
git checkout -b phase-1

# Verify branching
git branch
# Should show: * phase-1   main
```

Create `.pre-commit-config.yaml` in project root:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: contract-tests
        name: Contract tests
        entry: pytest tests/contracts/ -x -q --tb=short
        language: system
        pass_filenames: false
        stages: [commit]

      - id: regression-tests
        name: Regression suite
        entry: pytest tests/regression/ -x -q --tb=short
        language: system
        pass_filenames: false
        stages: [commit]
```

```bash
pip install pre-commit
pre-commit install
# Output: pre-commit installed at .git/hooks/pre-commit
```

Add a placeholder test so the hook has something to run:

```python
# tests/contracts/test_placeholder.py
def test_placeholder_passes():
    """Placeholder — replaced by real contract tests starting Day 2."""
    assert True
```

```bash
# Verify hook works
git add .
git commit -m "Day 1: Add pre-commit hook + placeholder test — all tests pass"
# Should run: Contract tests... Passed
```

---

## Task 16 — Add `SANDBOX_PREAMBLE` to `tools/e2b_sandbox.py`

```python
# tools/e2b_sandbox.py
# Day 1: scaffold with preamble. Full implementation Day 2.

from RestrictedPython import compile_restricted, safe_globals

SANDBOX_PREAMBLE = """\
import polars as pl
import polars.selectors as cs
import numpy as np
# ── Library standard: Polars not Pandas ──────────────────────────
# CORRECT:   pl.read_csv()  df.write_parquet()  df.fill_null()
# INCORRECT: pd.read_csv()  df.to_parquet()     df.fillna()
# If pandas required: convert back with pl.from_pandas(df)
# ─────────────────────────────────────────────────────────────────
"""

def execute_code(code: str, session_id: str) -> dict:
    """
    Execute Python code in RestrictedPython sandbox.
    Full implementation Day 2.
    For today: just confirm the scaffold runs.
    """
    full_code = SANDBOX_PREAMBLE + code

    try:
        compiled = compile_restricted(full_code, "<sandbox>", "exec")
        glb = dict(safe_globals)
        glb["__builtins__"] = safe_globals["__builtins__"]
        exec(compiled, glb)
        return {"status": "success", "output": glb}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

---

## Task 17 — Daily Log Template + README Sections

Create `DAILY_LOG.md`:

```markdown
# Professor — Daily Build Log

---

## Day 1 — [DATE]

**Schedule status:** ON TRACK / 1 DAY BEHIND / 2 DAYS BEHIND

**Tests green before starting:** YES / NO

**The ONE thing for today:**
Set up the complete environment and confirm all services run.

**Tasks completed:**
- [ ] Virtual environment + dependencies
- [ ] pip freeze > requirements.txt
- [ ] Groq API key + DeepSeek-R1 test
- [ ] Google AI Studio key + Gemini test
- [ ] Ollama + local model
- [ ] Folder structure
- [ ] tools/llm_client.py
- [ ] core/state.py
- [ ] main.py
- [ ] RestrictedPython verified
- [ ] ChromaDB verified
- [ ] fakeredis verified
- [ ] MLflow dashboard live
- [ ] Git branching + pre-commit hook

**CV score today:** N/A (pipeline not wired yet, Day 2)

**What broke:**

**How it was fixed:**

**Tomorrow's ONE thing:**
Build tools/e2b_sandbox.py and run the manual Submission 0 on Spaceship Titanic.

**Final commit hash:**

---
```

Add this to `README.md`:

```markdown
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
```

---

## End of Day 1 — Verification

```bash
# Run the environment check
python main.py check

# Expected output:
#   ✓ GROQ_API_KEY present
#   ✓ GOOGLE_API_KEY present
#   ✓ langgraph
#   ✓ groq
#   ✓ google.generativeai
#   ✓ polars
#   ✓ lightgbm
#   ✓ optuna
#   ✓ chromadb
#   ✓ fakeredis
#   ✓ mlflow
#   ✓ RestrictedPython
#   ✓ core/  agents/  tools/  guards/  memory/  tests/  outputs/  data/
#   ✓ Environment ready. Start building.

# Run main.py
python main.py run --competition spaceship-titanic \
  --data ./data/spaceship_titanic/

# Expected output:
#   [Professor] Starting session: spaceship_abc123de
#   [Professor] Competition:      spaceship-titanic
#   [Professor] State initialised. Pipeline coming Day 2.

# Confirm git is clean
git status        # should show: nothing to commit
git log --oneline # should show Day 1 commit on phase-1 branch
git branch        # should show: * phase-1   main
```

If `python main.py check` prints all green and `python main.py run` initialises without error — **Day 1 is complete.**

---

## Day 2 Preview

Tomorrow's ONE thing: build `tools/e2b_sandbox.py` with the full retry loop, then run the manual Submission 0 on Spaceship Titanic by hand (no Professor, just a notebook) to confirm the dataset format and establish a baseline score to beat.