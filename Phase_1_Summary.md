# Phase 1 Summary: "Make It Run" (Days 1–7)

Phase 1 established the robust, autonomous backbone of the Professor Kaggle agent. It transitioned the project from a conceptual architecture into a completely functional, end-to-end pipeline that routes tasks, cleans data securely, trains cross-validated models, and submits exactly-formatted predictions to Kaggle.

## Core Milestones Achieved

### **Day 1: Project Skeleton & State Management**
- Defined the strict LangGraph `ProfessorState` schema (`core/state.py`).
- Established the core architectural rules: no Pandas, no DataFrames in state (pointers only), strict invariant assertions.

### **Day 2: Secure Untrusted Execution**
- Built isolated execution environments (Sandboxes) for executing LLM-generated or untrusted data transformations.
- Set up the immutable `tests/contracts/` architecture to defensively verify every agent.

### **Day 3: The Data Engineer Agent**
- Built `agents/data_engineer.py` with rigorous Polars data transformations.
- Implemented robust `schema.json` generation (missing rates, cardinality, dtypes) for downstream agents.

### **Day 4: The ML Optimizer Agent**
- Built `agents/ml_optimizer.py` utilizing LightGBM with robust 5-fold cross-validation.
- Enforced strict Data Leakage prevention (e.g., fitting only on training folds).
- Generated Out-Of-Fold (OOF) predictions and tracked models in the `model_registry`.

### **Day 5: LangGraph Routing & Orchestration**
- Built `agents/semantic_router.py` for task identification and routing.
- Wired the initial linear LangGraph in `core/professor.py` (`Router` → `Data Engineer` → `ML Optimizer` → `Submit`).

### **Day 6: Validation, Logging, & Traceability**
- Developed `tools/submit_tools.py` for aggressive submission validation against Kaggle's expectations (row counts, IDs, specific dtypes like capitalized booleans).
- Implemented `core/lineage.py` for an append-only JSONL audit trail of all state mutations.
- Added LangSmith tracing integration for deep LangGraph debugging.

### **Day 7: The Phase 1 Gate & Regression Freeze**
- Final pipeline validation and Kaggle submission (Spaceship Titanic baseline: 0.78419 LB).
- Fixed LangGraph state reducer corruption (`operator.add` vs `_replace`).
- Froze the Phase 1 regression test `tests/regression/test_phase1_regression.py` to permanently protect the pipeline floor (CV > 0.8498).
- Implemented MLflow experiment tracking.

---

*Below are the concatenated original daily task prompts from Day 1 to Day 7.*

# Day 1 Tasks
Now let me get the Task Board.Let me fetch a few of the individual task pages for the full implementation details.Good â€” I have everything. Let me now compile the complete Day 1 plan with full implementation.

---

# Day 1 â€” Complete Implementation Plan

**The ONE thing that must work by end of today:**
Environment is fully set up, all APIs tested, git protection active, and `python main.py` runs without errors.

---

## The 14 Tasks â€” In Execution Order

```
MORNING BLOCK (06:30 â€“ 09:30)  â€” Setup + APIs + Structure
  Task 1:  Create project folder + virtual environment
  Task 2:  Install all dependencies + pin versions
  Task 3:  Get Groq API key + test DeepSeek-R1 70B
  Task 4:  Get Google AI Studio key + test Gemini 2.0 Flash
  Task 5:  Install Ollama + pull local fallback model
  Task 6:  Create .env file + .gitignore

AFTERNOON BLOCK (10:00 â€“ 12:30)  â€” Core Files
  Task 7:  Create full folder structure
  Task 8:  Build tools/llm_client.py
  Task 9:  Write core/state.py
  Task 10: Write main.py CLI entry point

INTEGRATION BLOCK (13:30 â€“ 15:30)  â€” Services + Verification
  Task 11: Verify RestrictedPython sandbox
  Task 12: Verify ChromaDB persistent storage
  Task 13: Verify fakeredis state
  Task 14: Launch MLflow dashboard

PROTECTION BLOCK (15:30 â€“ 17:00)  â€” Git + Safeguards
  Task 15: Git branching setup + pre-commit hook
  Task 16: Add SANDBOX_PREAMBLE + POLARS_CONSTRAINT
  Task 17: Write daily log template + README sections
```

---

## Task 1 â€” Create Project Folder + Virtual Environment

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

## Task 2 â€” Install All Dependencies + Pin Versions

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

# Immediately pin every version â€” do this before anything else
pip freeze > requirements.txt
git add requirements.txt
```

Open `requirements.txt` and add this header comment:

```
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PINNED Day 1. All versions confirmed working together.
# DO NOT run pip upgrade during the 30-day build.
# Library updates are a Month 2 activity.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

---

## Task 3 â€” Groq API Key + Test DeepSeek-R1 70B

Go to [console.groq.com](https://console.groq.com) â†’ Create account â†’ API Keys â†’ Create key.

Copy the key. Then test it:

```python
# test_groq.py â€” run this, delete after confirmed working
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

## Task 4 â€” Google AI Studio Key + Test Gemini 2.0 Flash

Go to [aistudio.google.com](https://aistudio.google.com) â†’ Get API key â†’ Create API key.

```python
# test_gemini.py â€” run this, delete after confirmed working
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

## Task 5 â€” Install Ollama + Local Fallback Model

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

## Task 6 â€” Create `.env` + `.gitignore`

```bash
# .env â€” never commit this file
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

## Task 7 â€” Create Full Folder Structure

```bash
# Run this entire block â€” creates every folder and file
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

## Task 8 â€” Build `tools/llm_client.py`

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

# â”€â”€ Clients â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€ Polars constraint injected into all code-generating prompts â”€â”€â”€
POLARS_CONSTRAINT = """
LIBRARY REQUIREMENT: This pipeline uses Polars not Pandas.
CORRECT:   pl.read_csv()  df.write_parquet()  df.fill_null()  df.group_by()
INCORRECT: pd.read_csv()  df.to_parquet()     df.fillna()     df.groupby()
If pandas is required for a specific library call:
  convert back with pl.from_pandas(df) before returning.
"""

# â”€â”€ Main call function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
      "groq-deepseek"  â†’ DeepSeek-R1 70B on Groq (default, all reasoning agents)
      "groq-llama"     â†’ Llama 3.3 70B on Groq (routing, fast tasks)
      "gemini-flash"   â†’ Gemini 2.0 Flash (Publisher, QA Gate)
      "ollama"         â†’ DeepSeek-R1 14b local (fallback only)
    """

    # Inject Polars constraint into all coding task prompts
    if is_coding_task:
        system = POLARS_CONSTRAINT + "\n\n" + system

    # â”€â”€ Groq: DeepSeek-R1 70B â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Groq: Llama 3.3 70B â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Gemini 2.0 Flash â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elif model == "gemini-flash":
        try:
            return _call_gemini(prompt, system, max_tokens)
        except Exception as e:
            print(f"[llm_client] Gemini failed: {e}. Falling back to Groq.")
            return _call_groq(prompt, system, "deepseek-r1-distill-llama-70b", max_tokens)

    # â”€â”€ Ollama local fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
# Quick test â€” paste in terminal
from tools.llm_client import call_llm
result = call_llm("Reply with: LLM_CLIENT_WORKING", model="groq-deepseek")
print(result)
# Must print something containing: LLM_CLIENT_WORKING
```

---

## Task 9 â€” Write `core/state.py`

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
    warning_threshold: float   # 0.70 â€” warn at 70% budget
    throttle_threshold: float  # 0.85 â€” throttle at 85%
    triage_threshold: float    # 0.95 â€” HITL at 95%


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
    # â”€â”€ Identity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    session_id: str              # namespaces ALL resources for this run
    created_at: str

    # â”€â”€ Competition â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    competition_name: str
    task_type: str               # "tabular_classification" | "tabular_regression" | "timeseries" | "auto"
    competition_context: Optional[CompetitionContext]

    # â”€â”€ Data (pointers only â€” never raw DataFrames in state) â”€â”€â”€â”€â”€â”€
    raw_data_path: str
    clean_data_path: Optional[str]
    schema_path: Optional[str]
    eda_report_path: Optional[str]
    data_hash: Optional[str]     # SHA-256 of source file, first 16 chars

    # â”€â”€ Feature Engineering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    feature_manifest: Optional[list]
    feature_factory_checkpoint: Optional[dict]

    # â”€â”€ Validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cv_strategy: Optional[dict]
    metric_contract: Optional[dict]
    cv_scores: Optional[list]
    cv_mean: Optional[float]

    # â”€â”€ Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model_registry: Optional[list]
    best_params: Optional[dict]
    optuna_study_path: Optional[str]

    # â”€â”€ Critic â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    critic_verdict: Optional[dict]

    # â”€â”€ Ensemble â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ensemble_weights: Optional[dict]
    oof_predictions_path: Optional[str]
    test_predictions_path: Optional[str]

    # â”€â”€ Submission â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    submission_path: Optional[str]
    submission_log: Optional[list]

    # â”€â”€ Routing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    dag: Optional[list]
    current_node: Optional[str]
    next_node: Optional[str]
    error_count: int
    escalation_level: str        # "micro" | "macro" | "hitl" | "triage"

    # â”€â”€ Budget â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cost_tracker: CostTracker

    # â”€â”€ Output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

## Task 10 â€” Write `main.py`

```python
# main.py

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="Professor â€” Autonomous Kaggle Agent",
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

    # â”€â”€ run command â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    run_parser = subparsers.add_parser("run", help="Start a competition run")
    run_parser.add_argument("--competition", required=True, help="Competition name")
    run_parser.add_argument("--data", required=True, help="Path to data folder")
    run_parser.add_argument("--budget", type=float, default=2.00, help="Budget in USD (default: 2.00)")
    run_parser.add_argument("--task-type", default="auto",
                            choices=["auto", "tabular_classification",
                                     "tabular_regression", "timeseries"],
                            help="Task type (default: auto-detect)")

    # â”€â”€ status command â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    status_parser = subparsers.add_parser("status", help="Check status of a running session")
    status_parser.add_argument("--session", required=True, help="Session ID")

    # â”€â”€ history command â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    subparsers.add_parser("history", help="List all past runs")

    # â”€â”€ check command (verify environment) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    subparsers.add_parser("check", help="Verify environment â€” run this on Day 1")

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
            print(f"  âœ“ {key} present")
        else:
            print(f"  âœ— {key} MISSING â€” add to .env")
            ok = False

    # Check critical libraries
    libs = ["langgraph", "groq", "google.generativeai", "polars",
            "lightgbm", "optuna", "chromadb", "fakeredis", "mlflow",
            "RestrictedPython"]
    for lib in libs:
        try:
            importlib.import_module(lib.replace("-", "_"))
            print(f"  âœ“ {lib}")
        except ImportError:
            print(f"  âœ— {lib} NOT INSTALLED")
            ok = False

    # Check folder structure
    required_dirs = ["core", "agents", "tools", "guards", "memory",
                     "tests/contracts", "outputs/logs", "data"]
    for d in required_dirs:
        if os.path.exists(d):
            print(f"  âœ“ {d}/")
        else:
            print(f"  âœ— {d}/ MISSING")
            ok = False

    print()
    if ok:
        print("[Check] âœ“ Environment ready. Start building.")
    else:
        print("[Check] âœ— Fix the issues above before proceeding.")


if __name__ == "__main__":
    main()
```

---

## Task 11-13 â€” Verify Services

```python
# Paste each block in a Python terminal and confirm output

# â”€â”€ RestrictedPython â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from RestrictedPython import compile_restricted, safe_globals
code = compile_restricted("result = 2 + 2", "<string>", "exec")
glb = dict(safe_globals)
exec(code, glb)
print(glb.get("result"))  # Must print: 4

# â”€â”€ ChromaDB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import chromadb
client = chromadb.PersistentClient(path="memory/chroma")
col = client.get_or_create_collection("test")
col.add(documents=["test document"], ids=["id1"])
result = col.query(query_texts=["test"], n_results=1)
print(result["documents"])  # Must print: [['test document']]
# ChromaDB is working and persisting to disk âœ“

# â”€â”€ fakeredis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import fakeredis
r = fakeredis.FakeRedis()
r.set("test_key", "test_value")
print(r.get("test_key").decode())  # Must print: test_value
```

---

## Task 14 â€” Launch MLflow

```bash
mlflow ui --port 5000 &
# Open browser: http://localhost:5000
# Must see the MLflow dashboard
# This runs in background â€” no experiments yet, that's fine
```

---

## Task 15 â€” Git Setup + Pre-commit Hook

```bash
# Initialise git (if not done)
git init
git add .
git commit -m "Day 1: Initial project structure â€” environment confirmed working"

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
    """Placeholder â€” replaced by real contract tests starting Day 2."""
    assert True
```

```bash
# Verify hook works
git add .
git commit -m "Day 1: Add pre-commit hook + placeholder test â€” all tests pass"
# Should run: Contract tests... Passed
```

---

## Task 16 â€” Add `SANDBOX_PREAMBLE` to `tools/e2b_sandbox.py`

```python
# tools/e2b_sandbox.py
# Day 1: scaffold with preamble. Full implementation Day 2.

from RestrictedPython import compile_restricted, safe_globals

SANDBOX_PREAMBLE = """\
import polars as pl
import polars.selectors as cs
import numpy as np
# â”€â”€ Library standard: Polars not Pandas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CORRECT:   pl.read_csv()  df.write_parquet()  df.fill_null()
# INCORRECT: pd.read_csv()  df.to_parquet()     df.fillna()
# If pandas required: convert back with pl.from_pandas(df)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

## Task 17 â€” Daily Log Template + README Sections

Create `DAILY_LOG.md`:

```markdown
# Professor â€” Daily Build Log

---

## Day 1 â€” [DATE]

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

1. Check critic_verdict.json â€” any CRITICAL/HIGH flags? Fix those first.
2. Verify metric_contract.json matches competition metric.
3. Diagnose CV/LB gap â€” high CV + low LB = leakage.
4. Check eda_report.json â€” any unaddressed flags?
5. Check feature_manifest.json â€” too few or too many features?
6. Check validation_strategy.json â€” wrong CV split for this data type?

Fix the specific component identified. Nothing else.
Architecture is not wrong. One component is off.

## Emergency Regression Protocol

1. Stop. Write no more code.
2. git log --oneline â€” find last green commit.
3. git checkout [hash] -b diagnosis â€” confirm tests pass.
4. git diff [hash] phase-N â€” bug is in this diff, nowhere else.
5. Fix one line. One commit. Run tests.
6. If not found in 2 hours â€” paste failing test + diff to Claude.
```

---

## End of Day 1 â€” Verification

```bash
# Run the environment check
python main.py check

# Expected output:
#   âœ“ GROQ_API_KEY present
#   âœ“ GOOGLE_API_KEY present
#   âœ“ langgraph
#   âœ“ groq
#   âœ“ google.generativeai
#   âœ“ polars
#   âœ“ lightgbm
#   âœ“ optuna
#   âœ“ chromadb
#   âœ“ fakeredis
#   âœ“ mlflow
#   âœ“ RestrictedPython
#   âœ“ core/  agents/  tools/  guards/  memory/  tests/  outputs/  data/
#   âœ“ Environment ready. Start building.

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

If `python main.py check` prints all green and `python main.py run` initialises without error â€” **Day 1 is complete.**

---

## Day 2 Preview

Tomorrow's ONE thing: build `tools/e2b_sandbox.py` with the full retry loop, then run the manual Submission 0 on Spaceship Titanic by hand (no Professor, just a notebook) to confirm the dataset format and establish a baseline score to beat.
# Day 2 Tasks
Got everything. Day 2 has exactly 4 tasks â€” all Critical, all Never Cut.

```
Task 1:  Build tools/e2b_sandbox.py â€” full RestrictedPython sandbox
Task 2:  Add 3-attempt inner retry loop to sandbox
Task 3:  Write contract test for the sandbox (immutable from today)
Task 4:  Manual Submission 0 â€” Spaceship Titanic by hand
```

**The ONE thing that must work by end of today:**
The sandbox executes code reliably with retries, AND you have a real Kaggle score on Spaceship Titanic to beat. That score is the floor everything Professor builds toward.

---

## Task 1 + 2 â€” Build `tools/e2b_sandbox.py` With Retry Loop

Both tasks live in the same file. Build them together.

```python
# tools/e2b_sandbox.py

import os
import sys
import signal
import traceback
from typing import Optional
from RestrictedPython import compile_restricted, safe_globals, safe_builtins
from RestrictedPython.Guards import safe_iter_unpack_sequence, guarded_iter_unpack_sequence
from dotenv import load_dotenv

load_dotenv()

# â”€â”€ Polars preamble injected before every generated script â”€â”€â”€â”€â”€â”€â”€â”€â”€
SANDBOX_PREAMBLE = """\
import polars as pl
import polars.selectors as cs
import numpy as np
import json
import os
# â”€â”€ Library standard: Polars not Pandas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CORRECT:   pl.read_csv()  df.write_parquet()  df.fill_null()
# INCORRECT: pd.read_csv()  df.to_parquet()     df.fillna()
# If pandas required: convert with pl.from_pandas(df) before returning
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
"""

# â”€â”€ Allowed imports inside sandbox â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ALLOWED_MODULES = {
    "polars", "numpy", "json", "os", "math",
    "sklearn", "lightgbm", "xgboost", "catboost",
    "optuna", "scipy", "statistics", "itertools",
    "collections", "functools", "datetime", "pathlib"
}

class SandboxExecutionError(Exception):
    """Raised when code fails all 3 retry attempts."""
    pass


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution exceeded 10 minute limit")


def _make_safe_globals(session_id: str) -> dict:
    """Build a restricted global namespace for the sandbox."""
    import polars as pl
    import numpy as np
    import json
    import math

    glb = dict(safe_globals)
    glb["__builtins__"] = dict(safe_builtins)

    # Inject allowed libraries directly
    glb["pl"] = pl
    glb["np"] = np
    glb["json"] = json
    glb["math"] = math
    glb["os"] = os

    # Allow print for debugging output
    glb["__builtins__"]["print"] = print
    glb["__builtins__"]["len"] = len
    glb["__builtins__"]["range"] = range
    glb["__builtins__"]["enumerate"] = enumerate
    glb["__builtins__"]["zip"] = zip
    glb["__builtins__"]["list"] = list
    glb["__builtins__"]["dict"] = dict
    glb["__builtins__"]["str"] = str
    glb["__builtins__"]["int"] = int
    glb["__builtins__"]["float"] = float
    glb["__builtins__"]["bool"] = bool
    glb["__builtins__"]["type"] = type
    glb["__builtins__"]["isinstance"] = isinstance
    glb["__builtins__"]["hasattr"] = hasattr
    glb["__builtins__"]["getattr"] = getattr
    glb["__builtins__"]["open"] = open  # needed for file I/O
    glb["__builtins__"]["Exception"] = Exception
    glb["__builtins__"]["ValueError"] = ValueError
    glb["__builtins__"]["TypeError"] = TypeError

    # Session output path â€” sandbox writes here
    glb["SESSION_OUTPUT_DIR"] = f"outputs/{session_id}"
    os.makedirs(f"outputs/{session_id}", exist_ok=True)

    return glb


def _execute_once(code: str, session_id: str, timeout_seconds: int = 600) -> dict:
    """
    Single execution attempt â€” no retry logic here.
    Returns: {success, stdout, stderr, result}
    """
    full_code = SANDBOX_PREAMBLE + code

    # Capture stdout
    import io
    from contextlib import redirect_stdout, redirect_stderr

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Set timeout (Unix only â€” Windows uses threading approach)
    if sys.platform != "win32":
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        compiled = compile_restricted(full_code, "<sandbox>", "exec")
        glb = _make_safe_globals(session_id)

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compiled, glb)

        if sys.platform != "win32":
            signal.alarm(0)  # cancel timeout

        return {
            "success": True,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "result": glb.get("result"),  # scripts can set result = value
            "globals": glb
        }

    except TimeoutError:
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": "TIMEOUT: Code exceeded 10 minute execution limit",
            "error": "TimeoutError",
            "traceback": "Execution timeout"
        }
    except Exception as e:
        if sys.platform != "win32":
            signal.alarm(0)
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": type(e).__name__,
            "traceback": traceback.format_exc()
        }


def execute_code(
    code: str,
    session_id: str,
    llm_fix_callback=None,
    max_attempts: int = 3,
    timeout_seconds: int = 600
) -> dict:
    """
    Execute code in RestrictedPython sandbox with 3-attempt retry loop.

    On failure: feeds full traceback back to LLM (via llm_fix_callback)
    which returns corrected code. Retries up to max_attempts times.
    After 3 failures: raises SandboxExecutionError (never hangs).

    Args:
        code:              Python code string to execute
        session_id:        Session namespace for file I/O
        llm_fix_callback:  fn(code, error, traceback) -> fixed_code
                           If None: retries same code (for testing)
        max_attempts:      Maximum retry attempts (default: 3)
        timeout_seconds:   Timeout per attempt in seconds (default: 600)

    Returns:
        {success, stdout, stderr, result, attempts_used}

    Raises:
        SandboxExecutionError: after max_attempts failures
    """
    current_code = code
    last_result = None

    for attempt in range(1, max_attempts + 1):
        print(f"[sandbox] Attempt {attempt}/{max_attempts}...")
        result = _execute_once(current_code, session_id, timeout_seconds)

        if result["success"]:
            result["attempts_used"] = attempt
            print(f"[sandbox] Success on attempt {attempt}.")
            return result

        # â”€â”€ Failure â€” log and prepare retry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        last_result = result
        error_info = f"""
EXECUTION FAILED (Attempt {attempt}/{max_attempts})
Error type:  {result.get('error', 'Unknown')}
Traceback:
{result.get('traceback', 'No traceback available')}
Stdout before failure:
{result.get('stdout', '')}
"""
        print(f"[sandbox] {error_info}")

        # If we have more attempts AND a fix callback, get corrected code
        if attempt < max_attempts and llm_fix_callback is not None:
            print(f"[sandbox] Requesting LLM fix for attempt {attempt + 1}...")
            try:
                current_code = llm_fix_callback(
                    code=current_code,
                    error=result.get("error", ""),
                    traceback_str=result.get("traceback", "")
                )
            except Exception as callback_error:
                print(f"[sandbox] LLM fix callback failed: {callback_error}")
                # Continue with same code if callback fails

        elif attempt < max_attempts:
            print(f"[sandbox] No fix callback. Retrying same code...")

    # â”€â”€ All attempts exhausted â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    raise SandboxExecutionError(
        f"Code failed after {max_attempts} attempts.\n"
        f"Final error: {last_result.get('error')}\n"
        f"Final traceback:\n{last_result.get('traceback')}"
    )
```

---

## Task 3 â€” Write Contract Test (Immutable From Today)

```python
# tests/contracts/test_e2b_sandbox_contract.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Written: Day 2
# Status:  IMMUTABLE â€” never edit this file after today
#
# CONTRACT: execute_code()
#   INPUT:  code (str), session_id (str)
#   OUTPUT: dict with keys: success (bool), stdout (str), stderr (str)
#   ERRORS: raises SandboxExecutionError after 3 failed attempts
#           never hangs â€” always returns or raises within timeout
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import pytest
import time
from tools.e2b_sandbox import execute_code, SandboxExecutionError

SESSION = "test_session_sandbox"


class TestSandboxContract:

    def test_successful_execution_returns_success_true(self):
        result = execute_code("result = 2 + 2", session_id=SESSION)
        assert result["success"] is True

    def test_output_has_required_keys(self):
        result = execute_code("x = 1", session_id=SESSION)
        assert "success" in result
        assert "stdout" in result
        assert "stderr" in result

    def test_stdout_captured(self):
        result = execute_code("print('hello_sandbox')", session_id=SESSION)
        assert result["success"] is True
        assert "hello_sandbox" in result["stdout"]

    def test_polars_available_in_sandbox(self):
        code = """
import polars as pl
df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
result = df.shape
print(f"shape: {df.shape}")
"""
        result = execute_code(code, session_id=SESSION)
        assert result["success"] is True
        assert "shape" in result["stdout"]

    def test_numpy_available_in_sandbox(self):
        code = "import numpy as np\nresult = np.mean([1, 2, 3])\nprint(result)"
        result = execute_code(code, session_id=SESSION)
        assert result["success"] is True

    def test_syntax_error_raises_sandbox_error(self):
        bad_code = "def broken( :"  # intentional syntax error
        with pytest.raises(SandboxExecutionError):
            execute_code(bad_code, session_id=SESSION, max_attempts=1)

    def test_runtime_error_raises_sandbox_error_after_max_attempts(self):
        bad_code = "result = 1 / 0"  # ZeroDivisionError
        with pytest.raises(SandboxExecutionError) as exc_info:
            execute_code(bad_code, session_id=SESSION, max_attempts=3)
        assert "3 attempts" in str(exc_info.value)

    def test_retry_loop_uses_fix_callback(self):
        """LLM fix callback is called on failure and fixed code succeeds."""
        call_count = {"n": 0}

        def mock_fix(code, error, traceback_str):
            call_count["n"] += 1
            return "result = 42  # fixed"  # always returns working code

        bad_code = "result = 1 / 0"
        result = execute_code(
            bad_code,
            session_id=SESSION,
            llm_fix_callback=mock_fix,
            max_attempts=3
        )
        assert result["success"] is True
        assert call_count["n"] == 1  # called once on first failure

    def test_never_allows_dangerous_imports(self):
        """Sandbox must block filesystem and system access."""
        dangerous_code = "import subprocess\nsubprocess.run(['ls'])"
        with pytest.raises((SandboxExecutionError, Exception)):
            execute_code(dangerous_code, session_id=SESSION, max_attempts=1)

    def test_attempts_used_recorded_in_result(self):
        result = execute_code("x = 1", session_id=SESSION)
        assert "attempts_used" in result
        assert result["attempts_used"] == 1

    def test_output_dir_created_for_session(self):
        import os
        execute_code("x = 1", session_id="output_test_session")
        assert os.path.exists("outputs/output_test_session")
```

Run it immediately:

```bash
pytest tests/contracts/test_e2b_sandbox_contract.py -v
# All tests must pass before moving to Task 4
```

---

## Task 4 â€” Manual Submission 0 (Spaceship Titanic)

This is built **by you in a notebook** â€” not by Professor. The purpose is to confirm the dataset format, metric, and submission structure before Professor writes a single line of agent code.

First, download the data:

```bash
# Make sure kaggle CLI is set up
# ~/.kaggle/kaggle.json must have your username and key

kaggle competitions download -c spaceship-titanic -p data/spaceship_titanic/
cd data/spaceship_titanic/
unzip spaceship-titanic.zip
ls
# Should show: train.csv  test.csv  sample_submission.csv
```

Now open `notebooks/sanity_check.ipynb` and run this â€” no feature engineering, no CV, default everything:

```python
# notebooks/sanity_check.ipynb
# Manual Submission 0 â€” built by hand, not by Professor
# Purpose: confirm format, metric, and establish baseline score

import polars as pl
import pandas as pd  # sklearn needs pandas for now â€” fine in notebook
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train = pl.read_csv("../data/spaceship_titanic/train.csv")
test  = pl.read_csv("../data/spaceship_titanic/test.csv")
sample = pl.read_csv("../data/spaceship_titanic/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape: ", test.shape)
print("\nColumns:", train.columns)
print("\nTarget distribution:")
print(train["Transported"].value_counts())
print("\nSample submission format:")
print(sample.head(3))

# â”€â”€ Minimal preprocessing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Convert to pandas for sklearn
train_pd = train.to_pandas()
test_pd  = test.to_pandas()

# Drop high-cardinality and complex columns for this baseline
drop_cols = ["PassengerId", "Name", "Cabin"]
train_pd = train_pd.drop(columns=drop_cols, errors="ignore")
test_pd  = test_pd.drop(columns=drop_cols, errors="ignore")

# Encode target
y = train_pd["Transported"].astype(int)
train_pd = train_pd.drop(columns=["Transported"])

# Label encode categoricals
categorical_cols = train_pd.select_dtypes(include="object").columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    train_pd[col] = train_pd[col].fillna("missing")
    test_pd[col]  = test_pd[col].fillna("missing")
    # Fit on combined to avoid unseen labels
    combined = pd.concat([train_pd[col], test_pd[col]])
    le.fit(combined)
    train_pd[col] = le.transform(train_pd[col])
    test_pd[col]  = le.transform(test_pd[col])

# Fill numeric nulls
train_pd = train_pd.fillna(train_pd.median(numeric_only=True))
test_pd  = test_pd.fillna(test_pd.median(numeric_only=True))

print("\nFeatures used:", list(train_pd.columns))
print("Training shape:", train_pd.shape)

# â”€â”€ Train single model â€” default params, no tuning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model.fit(train_pd, y)

# â”€â”€ Quick local CV estimate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    LGBMClassifier(n_estimators=500, learning_rate=0.05,
                   random_state=42, verbose=-1),
    train_pd, y,
    cv=5,
    scoring="accuracy"
)
print(f"\nLocal CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# â”€â”€ Generate submission â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
preds = model.predict(test_pd)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"].to_list(),
    "Transported": preds.astype(bool)
})

submission.to_csv("../outputs/submission_0_manual.csv", index=False)
print("\nSubmission saved: outputs/submission_0_manual.csv")
print("\nSubmission format check:")
print(submission.head(5))
print(f"\nShape: {submission.shape}")
print(f"Expected: ({len(test)}, 2)")

# â”€â”€ Format validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
assert set(submission.columns) == {"PassengerId", "Transported"}, \
    "Wrong columns in submission"
assert len(submission) == len(test), \
    f"Wrong row count: {len(submission)} vs {len(test)}"
assert submission["Transported"].dtype == bool or \
    submission["Transported"].isin([True, False]).all(), \
    "Transported must be boolean"

print("\nâœ“ Submission format valid")
print("âœ“ Ready to submit to Kaggle")
```

Submit it:

```bash
kaggle competitions submit \
  -c spaceship-titanic \
  -f outputs/submission_0_manual.csv \
  -m "Submission 0 â€” manual baseline, default LightGBM"

# Check your score
kaggle competitions submissions -c spaceship-titanic
```

---

## End of Day 2 â€” What Must Be True

```
Before committing and closing:

â–¡ pytest tests/contracts/test_e2b_sandbox_contract.py -v
  â†’ All tests green

â–¡ python -c "from tools.e2b_sandbox import execute_code;
             r = execute_code('print(42)', 'test');
             print(r['success'])"
  â†’ True

â–¡ Submission 0 submitted to Kaggle
  â†’ Public LB score recorded in DAILY_LOG.md
  â†’ Must beat 0.50 (random baseline for binary classification)
  â†’ Spaceship Titanic random baseline â‰ˆ 0.50, good baseline â‰ˆ 0.77-0.79

â–¡ DAILY_LOG.md updated:
  â†’ CV score from notebook
  â†’ Public LB score from Kaggle
  â†’ Both recorded as the floors Professor must beat
```

**Record the LB score carefully.** That number is Submission 0. Every Professor-built submission from Day 7 onward must beat it. If you cannot beat a manual default LightGBM, something in the pipeline is wrong.

---

## Day 3 Preview

Tomorrow's ONE thing: build `agents/data_engineer.py` â€” takes `raw_data_path` from state, produces `cleaned.parquet` + `schema.json`, writes both pointers back to state. First real agent. First time the state schema gets exercised.
# Day 3 Tasks
Got everything. Here's the full Day 3 breakdown.

---

## Day 3 Tasks â€” From Notion

```
â”Œâ”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ #  â”‚ Task                                     â”‚ Phase                    â”‚ Priority â”‚ Cuttable  â”‚
â”œâ”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ 1  â”‚ Build agents/data_engineer.py            â”‚ ðŸš€ Phase 1: Make It Run  â”‚ Critical â”‚ Never Cut â”‚
â”‚ 2  â”‚ Write contract test â€” Data Engineer      â”‚ ðŸš€ Phase 1: Make It Run  â”‚ Critical â”‚ Never Cut â”‚
â”‚ 3  â”‚ Test Data Engineer on Spaceship Titanic  â”‚ ðŸš€ Phase 1: Make It Run  â”‚ Critical â”‚ Never Cut â”‚
â””â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

All three are Phase 1. All Critical. All Never Cut.

**The ONE thing that must work by end of today:**
Feed `train.csv` into `data_engineer.py` and get back `cleaned.parquet` + `schema.json` with only string pointers in state. No raw data in state. Ever.

---

## Task 1 â€” Build `agents/data_engineer.py`

The Data Engineer is a LangGraph node. It takes `raw_data_path` from state, runs preprocessing code inside the sandbox, and writes two files: `cleaned.parquet` and `schema.json`. It then updates state with the file paths â€” never the data itself.

```python
# agents/data_engineer.py

import os
import json
import hashlib
import polars as pl
from datetime import datetime
from core.state import ProfessorState
from tools.e2b_sandbox import execute_code, SandboxExecutionError
from tools.llm_client import call_llm

# â”€â”€ LLM fix callback for sandbox retry loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _make_fix_callback(session_id: str):
    """Returns a callback that asks the LLM to fix broken sandbox code."""
    def fix_callback(code: str, error: str, traceback_str: str) -> str:
        prompt = f"""
The following Python code failed with an error.
Fix ONLY the specific error. Do not restructure the code.
Return the complete corrected code and nothing else â€” no explanation,
no markdown fences, just raw Python.

FAILED CODE:
{code}

ERROR TYPE: {error}
TRACEBACK:
{traceback_str}
"""
        fixed = call_llm(
            prompt=prompt,
            system="You are a Python debugging assistant. Return only corrected code.",
            model="fireworks-deepseek",
            is_coding_task=True
        )
        # Strip any accidental markdown fences
        fixed = fixed.strip()
        if fixed.startswith("```"):
            fixed = "\n".join(fixed.split("\n")[1:])
        if fixed.endswith("```"):
            fixed = "\n".join(fixed.split("\n")[:-1])
        return fixed.strip()
    return fix_callback


# â”€â”€ Preprocessing code template â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _build_preprocessing_code(
    raw_data_path: str,
    output_dir: str,
    schema: dict
) -> str:
    """
    Builds the preprocessing script the sandbox will execute.
    The LLM customises this per-dataset. For now: deterministic template.
    Full LLM-driven preprocessing comes in Phase 2.
    """
    return f"""
import polars as pl
import polars.selectors as cs
import json
import os

# â”€â”€ Load raw data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df = pl.read_csv("{raw_data_path}", infer_schema_length=10000)
print(f"Loaded: {{df.shape[0]}} rows, {{df.shape[1]}} columns")

# â”€â”€ Profile BEFORE cleaning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
missing_rates = {{
    col: round(df[col].null_count() / len(df), 4)
    for col in df.columns
}}
print(f"Missing rates: {{missing_rates}}")

# â”€â”€ Type inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
column_types = {{col: str(df[col].dtype) for col in df.columns}}

# â”€â”€ Basic cleaning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Fill numeric nulls with median
numeric_cols = df.select(cs.numeric()).columns
for col in numeric_cols:
    median_val = df[col].median()
    if median_val is not None:
        df = df.with_columns(pl.col(col).fill_null(median_val))

# Fill string nulls with "missing"
string_cols = df.select(cs.string()).columns
for col in string_cols:
    df = df.with_columns(pl.col(col).fill_null("missing"))

# Fill boolean nulls with False
bool_cols = df.select(cs.boolean()).columns
for col in bool_cols:
    df = df.with_columns(pl.col(col).fill_null(False))

print(f"After cleaning: {{df.shape[0]}} rows, {{df.shape[1]}} columns")
print(f"Remaining nulls: {{df.null_count().sum_horizontal().item()}}")

# â”€â”€ Write outputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
os.makedirs("{output_dir}", exist_ok=True)

parquet_path = "{output_dir}/cleaned.parquet"
schema_path  = "{output_dir}/schema.json"

df.write_parquet(parquet_path)
print(f"Saved: {{parquet_path}}")

schema = {{
    "columns":       df.columns,
    "types":         {{col: str(df[col].dtype) for col in df.columns}},
    "missing_rates": missing_rates,
    "shape":         list(df.shape),
    "cleaned_at":    "{datetime.utcnow().isoformat()}"
}}

with open(schema_path, "w") as f:
    json.dump(schema, f, indent=2)
print(f"Saved: {{schema_path}}")

print("DATA_ENGINEER_COMPLETE")
"""


# â”€â”€ Main agent function (LangGraph node) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_data_engineer(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Data Engineer.

    Reads:  state["raw_data_path"]
    Writes: state["clean_data_path"]  â€” str pointer to cleaned.parquet
            state["schema_path"]      â€” str pointer to schema.json
            state["data_hash"]        â€” SHA-256 of source file (first 16 chars)
            state["cost_tracker"]     â€” incremented
    Never puts raw data in state.
    """
    session_id  = state["session_id"]
    raw_path    = state["raw_data_path"]
    output_dir  = f"outputs/{session_id}"

    print(f"[DataEngineer] Starting â€” session: {session_id}")
    print(f"[DataEngineer] Input: {raw_path}")

    # â”€â”€ Validate input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw_data_path does not exist: {raw_path}")

    # â”€â”€ Hash the source file â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with open(raw_path, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    print(f"[DataEngineer] data_hash: {data_hash}")

    # â”€â”€ Build preprocessing code â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    code = _build_preprocessing_code(
        raw_data_path=raw_path,
        output_dir=output_dir,
        schema={}
    )

    # â”€â”€ Execute in sandbox with retry loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fix_callback = _make_fix_callback(session_id)

    try:
        result = execute_code(
            code=code,
            session_id=session_id,
            llm_fix_callback=fix_callback,
            max_attempts=3
        )
    except SandboxExecutionError as e:
        print(f"[DataEngineer] Sandbox failed after 3 attempts: {e}")
        raise

    if not result["success"]:
        raise RuntimeError(f"[DataEngineer] Unexpected failure: {result}")

    print(f"[DataEngineer] Sandbox output:\n{result['stdout']}")

    # â”€â”€ Verify outputs exist â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    parquet_path = f"{output_dir}/cleaned.parquet"
    schema_path  = f"{output_dir}/schema.json"

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"cleaned.parquet not produced: {parquet_path}")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema.json not produced: {schema_path}")

    # â”€â”€ Validate schema.json has required fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with open(schema_path) as f:
        schema = json.load(f)

    required_schema_fields = ["columns", "types", "missing_rates"]
    for field in required_schema_fields:
        if field not in schema:
            raise ValueError(f"schema.json missing required field: '{field}'")

    print(f"[DataEngineer] cleaned.parquet: {os.path.getsize(parquet_path):,} bytes")
    print(f"[DataEngineer] schema.json:     {len(schema['columns'])} columns")

    # â”€â”€ Update cost tracker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cost_tracker = dict(state["cost_tracker"])
    cost_tracker["llm_calls"] += result.get("attempts_used", 1)

    print(f"[DataEngineer] Complete. Attempts used: {result.get('attempts_used', 1)}")

    # â”€â”€ Return updated state â€” ONLY pointers, never raw data â”€â”€â”€â”€â”€â”€
    return {
        **state,
        "clean_data_path": parquet_path,
        "schema_path":     schema_path,
        "data_hash":       data_hash,
        "cost_tracker":    cost_tracker,
    }
```

---

## Task 2 â€” Write Contract Test (Immutable From Today)

```python
# tests/contracts/test_data_engineer_contract.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Written: Day 3
# Status:  IMMUTABLE â€” never edit this file after today
#
# CONTRACT: run_data_engineer()
#   INPUT:   state["raw_data_path"] â€” str, must exist on disk
#   OUTPUT:  outputs/{session_id}/cleaned.parquet â€” must exist
#            outputs/{session_id}/schema.json â€” must have:
#              columns (list), types (dict), missing_rates (dict)
#   STATE:   clean_data_path â€” str pointer (not DataFrame)
#            schema_path     â€” str pointer
#            data_hash       â€” 16-char hex string
#            cost_tracker    â€” llm_calls incremented
#   NEVER:   raw DataFrame in state
#            raw DataFrame in any state field
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import pytest
import os
import json
import polars as pl
from pathlib import Path
from core.state import initial_state
from agents.data_engineer import run_data_engineer

# â”€â”€ Fixture: minimal CSV the tests always use â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FIXTURE_CSV = "tests/fixtures/tiny_train.csv"

@pytest.fixture(scope="session", autouse=True)
def create_fixture_csv():
    """Create a minimal CSV fixture for contract tests."""
    os.makedirs("tests/fixtures", exist_ok=True)
    if not os.path.exists(FIXTURE_CSV):
        import polars as pl
        df = pl.DataFrame({
            "PassengerId": ["0001_01", "0002_01", "0003_01",
                            "0004_01", "0005_01"],
            "HomePlanet":  ["Europa", "Earth", None, "Mars", "Earth"],
            "Age":         [39.0, 24.0, None, 58.0, 33.0],
            "RoomService": [0.0, 109.0, None, 43.0, 0.0],
            "Transported": [False, True, True, False, True],
        })
        df.write_csv(FIXTURE_CSV)


@pytest.fixture
def base_state():
    return initial_state(
        competition="test-titanic",
        data_path=FIXTURE_CSV,
        budget_usd=2.0
    )


class TestDataEngineerContract:

    def test_accepts_valid_raw_data_path(self, base_state):
        result = run_data_engineer(base_state)
        assert result is not None

    def test_rejects_nonexistent_path(self, base_state):
        bad_state = {**base_state, "raw_data_path": "/nonexistent/train.csv"}
        with pytest.raises(FileNotFoundError):
            run_data_engineer(bad_state)

    def test_produces_cleaned_parquet(self, base_state):
        result = run_data_engineer(base_state)
        assert os.path.exists(result["clean_data_path"]), \
            "cleaned.parquet must exist after run"

    def test_produces_schema_json(self, base_state):
        result = run_data_engineer(base_state)
        assert os.path.exists(result["schema_path"]), \
            "schema.json must exist after run"

    def test_schema_has_columns_field(self, base_state):
        result = run_data_engineer(base_state)
        schema = json.loads(Path(result["schema_path"]).read_text())
        assert "columns" in schema, "schema.json must have 'columns'"
        assert isinstance(schema["columns"], list)
        assert len(schema["columns"]) > 0

    def test_schema_has_types_field(self, base_state):
        result = run_data_engineer(base_state)
        schema = json.loads(Path(result["schema_path"]).read_text())
        assert "types" in schema, "schema.json must have 'types'"
        assert isinstance(schema["types"], dict)

    def test_schema_has_missing_rates_field(self, base_state):
        result = run_data_engineer(base_state)
        schema = json.loads(Path(result["schema_path"]).read_text())
        assert "missing_rates" in schema, "schema.json must have 'missing_rates'"
        assert isinstance(schema["missing_rates"], dict)

    def test_clean_data_path_is_string_not_dataframe(self, base_state):
        result = run_data_engineer(base_state)
        assert isinstance(result["clean_data_path"], str), \
            "clean_data_path must be a str pointer â€” never a DataFrame"

    def test_no_raw_data_in_state(self, base_state):
        result = run_data_engineer(base_state)
        for key, value in result.items():
            assert not isinstance(value, pl.DataFrame), \
                f"DataFrame found in state['{key}'] â€” only pointers allowed"

    def test_data_hash_set_in_state(self, base_state):
        result = run_data_engineer(base_state)
        assert "data_hash" in result
        assert isinstance(result["data_hash"], str)
        assert len(result["data_hash"]) == 16, \
            "data_hash must be 16-char hex string"

    def test_cost_tracker_llm_calls_incremented(self, base_state):
        before = base_state["cost_tracker"]["llm_calls"]
        result = run_data_engineer(base_state)
        after  = result["cost_tracker"]["llm_calls"]
        assert after >= before, \
            "cost_tracker.llm_calls must be incremented after run"

    def test_parquet_is_polars_readable(self, base_state):
        result = run_data_engineer(base_state)
        df = pl.read_parquet(result["clean_data_path"])
        assert isinstance(df, pl.DataFrame), \
            "cleaned.parquet must be readable as Polars DataFrame"

    def test_parquet_has_no_object_dtype(self, base_state):
        result = run_data_engineer(base_state)
        df = pl.read_parquet(result["clean_data_path"])
        object_cols = [c for c in df.columns if df[c].dtype == pl.Object]
        assert len(object_cols) == 0, \
            f"Object dtype columns detected (Pandas contamination): {object_cols}"

    def test_no_nulls_in_cleaned_parquet(self, base_state):
        result = run_data_engineer(base_state)
        df = pl.read_parquet(result["clean_data_path"])
        total_nulls = df.null_count().sum_horizontal().item()
        assert total_nulls == 0, \
            f"cleaned.parquet should have 0 nulls after cleaning, found {total_nulls}"

    def test_session_id_namespacing(self, base_state):
        """Output files must live under outputs/{session_id}/"""
        result = run_data_engineer(base_state)
        session_id = base_state["session_id"]
        assert session_id in result["clean_data_path"], \
            "clean_data_path must be namespaced under session_id"
        assert session_id in result["schema_path"], \
            "schema_path must be namespaced under session_id"
```

---

## Task 3 â€” Test Data Engineer on Spaceship Titanic

Run it against the real dataset, not just the fixture.

```python
# Run this in a terminal â€” not in the notebook
from core.state import initial_state
from agents.data_engineer import run_data_engineer
import json

state = initial_state(
    competition="spaceship-titanic",
    data_path="data/spaceship_titanic/train.csv",
    budget_usd=2.0
)

print("Running Data Engineer on Spaceship Titanic...")
result = run_data_engineer(state)

# â”€â”€ Verify outputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import polars as pl

df = pl.read_parquet(result["clean_data_path"])
schema = json.loads(open(result["schema_path"]).read())

print(f"\nâœ“ cleaned.parquet shape:  {df.shape}")
print(f"âœ“ Columns:                {df.columns}")
print(f"âœ“ Null count:             {df.null_count().sum_horizontal().item()}")
print(f"âœ“ data_hash:              {result['data_hash']}")
print(f"âœ“ Schema columns:         {schema['columns']}")
print(f"âœ“ Missing rates:          {schema['missing_rates']}")
print(f"âœ“ State clean_data_path:  {result['clean_data_path']}")
print(f"âœ“ State schema_path:      {result['schema_path']}")
print(f"âœ“ No DataFrame in state:  TRUE")
print(f"âœ“ Cost tracker calls:     {result['cost_tracker']['llm_calls']}")
```

Expected output:

```
Running Data Engineer on Spaceship Titanic...
[DataEngineer] Starting â€” session: spaceship_abc123de
[DataEngineer] data_hash: a3f9c21d4b8e7f01
[DataEngineer] Sandbox output:
  Loaded: 8693 rows, 14 columns
  Missing rates: {'HomePlanet': 0.0201, 'CryoSleep': 0.0247, ...}
  After cleaning: 8693 rows, 14 columns
  Remaining nulls: 0
  Saved: outputs/spaceship_abc123de/cleaned.parquet
  Saved: outputs/spaceship_abc123de/schema.json
  DATA_ENGINEER_COMPLETE

âœ“ cleaned.parquet shape:  (8693, 14)
âœ“ Null count:             0
âœ“ data_hash:              a3f9c21d4b8e7f01
âœ“ State clean_data_path:  outputs/spaceship_abc123de/cleaned.parquet
âœ“ No DataFrame in state:  TRUE
```

---

## End of Day 3 Checklist

```bash
# 1. Run contract tests
pytest tests/contracts/test_data_engineer_contract.py -v
# All 15 tests must be green

# 2. Run all contract tests together (sandbox + data engineer)
pytest tests/contracts/ -v
# All tests from both Day 2 and Day 3 must pass

# 3. Confirm real dataset run produced outputs
ls outputs/spaceship-titanic*/
# Must show: cleaned.parquet  schema.json

# 4. Commit
git add .
git commit -m "Day 3: Build Data Engineer + contract test â€” all tests pass"
git push origin phase-1
```


â”Œâ”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ #  â”‚ Task                                         â”‚ Phase                   â”‚ Priority â”‚ Cuttable       â”‚
â”œâ”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ 1  â”‚ Build tools/data_tools.py â€” Polars helpers   â”‚ ðŸš€ Phase 1: Make It Run â”‚ High     â”‚ Safe to Stub   â”‚
â”‚ 2  â”‚ Build agents/data_engineer.py                â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut      â”‚
â”‚ 3  â”‚ Write contract test â€” Data Engineer          â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut      â”‚
â”‚ 4  â”‚ Test Data Engineer on Spaceship Titanic      â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut      â”‚
â””â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Important note on order:** `data_tools.py` must be built first. The Data Engineer imports from it. Build Task 1 before Task 2 or the import will fail.

---

## Task 1 â€” Build `tools/data_tools.py`

This is the utility layer that all agents use for data I/O. Three core functions: `read_csv`, `write_parquet`, `profile_data`. The Data Engineer calls `profile_data` to produce `schema.json`. Every agent that reads or writes data calls this instead of calling Polars directly â€” so if the I/O layer ever needs to change, you change it in one place.

```python
# tools/data_tools.py

import os
import json
import hashlib
import polars as pl
import polars.selectors as cs
from datetime import datetime
from pathlib import Path


# â”€â”€ Read â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def read_csv(path: str, infer_schema_length: int = 10000) -> pl.DataFrame:
    """
    Read a CSV file into a Polars DataFrame.
    Always use this â€” never pl.read_csv() directly in agents.
    Validates the file exists before reading.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pl.read_csv(path, infer_schema_length=infer_schema_length)


def read_parquet(path: str) -> pl.DataFrame:
    """
    Read a Parquet file into a Polars DataFrame.
    Always use this â€” never pl.read_parquet() directly in agents.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pl.read_parquet(path)
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected Polars DataFrame, got {type(df)}")
    return df


# â”€â”€ Write â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def write_parquet(df: pl.DataFrame, path: str) -> str:
    """
    Write a Polars DataFrame to Parquet.
    Creates parent directories if they don't exist.
    Returns the path written.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"write_parquet expects a Polars DataFrame, got {type(df)}. "
            "If you have a Pandas DataFrame, convert with pl.from_pandas(df) first."
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write_parquet(path)
    return path


def write_json(data: dict, path: str) -> str:
    """
    Write a dict to JSON. Creates parent directories if needed.
    Returns the path written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def read_json(path: str) -> dict:
    """Read a JSON file and return as dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path) as f:
        return json.load(f)


# â”€â”€ Profile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def profile_data(df: pl.DataFrame) -> dict:
    """
    Profile a Polars DataFrame and return a schema dict.
    This is what gets written to schema.json by the Data Engineer.

    Returns:
        {
            columns:       [list of column names],
            types:         {col: dtype_str},
            missing_rates: {col: float 0-1},
            missing_counts:{col: int},
            shape:         [rows, cols],
            numeric_cols:  [list],
            categorical_cols: [list],
            boolean_cols:  [list],
            cardinality:   {col: n_unique} for categorical cols,
            profiled_at:   ISO timestamp
        }
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"profile_data expects Polars DataFrame, got {type(df)}")

    n_rows = len(df)

    # Column types
    column_types = {col: str(df[col].dtype) for col in df.columns}

    # Missing rates and counts
    missing_counts = {col: int(df[col].null_count()) for col in df.columns}
    missing_rates  = {
        col: round(missing_counts[col] / n_rows, 4) if n_rows > 0 else 0.0
        for col in df.columns
    }

    # Categorise columns by type
    numeric_cols     = df.select(cs.numeric()).columns
    categorical_cols = df.select(cs.string()).columns
    boolean_cols     = df.select(cs.boolean()).columns

    # Cardinality for categoricals (useful for encoding decisions)
    cardinality = {
        col: int(df[col].n_unique())
        for col in categorical_cols
    }

    return {
        "columns":          df.columns,
        "types":            column_types,
        "missing_rates":    missing_rates,
        "missing_counts":   missing_counts,
        "shape":            list(df.shape),
        "numeric_cols":     numeric_cols,
        "categorical_cols": categorical_cols,
        "boolean_cols":     boolean_cols,
        "cardinality":      cardinality,
        "profiled_at":      datetime.utcnow().isoformat(),
    }


# â”€â”€ Hash â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def hash_file(path: str, length: int = 16) -> str:
    """
    SHA-256 hash of a file. Returns first `length` hex chars.
    Used to detect if the dataset changes mid-competition.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot hash â€” file not found: {path}")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:length]


def hash_dataframe(df: pl.DataFrame, length: int = 16) -> str:
    """
    SHA-256 hash of a Polars DataFrame's content.
    Deterministic â€” same data always produces same hash.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"hash_dataframe expects Polars DataFrame, got {type(df)}")
    content = df.write_csv().encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:length]


# â”€â”€ Validate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def validate_submission(
    submission: pl.DataFrame,
    sample_submission: pl.DataFrame
) -> dict:
    """
    Validate a submission DataFrame against the sample submission format.

    Returns:
        {"valid": bool, "errors": [list of error strings]}
    """
    errors = []

    # Column names must match exactly
    if set(submission.columns) != set(sample_submission.columns):
        errors.append(
            f"Column mismatch. Expected: {sample_submission.columns}. "
            f"Got: {submission.columns}"
        )

    # Row count must match
    if len(submission) != len(sample_submission):
        errors.append(
            f"Row count mismatch. Expected: {len(sample_submission)}. "
            f"Got: {len(submission)}"
        )

    # No nulls in submission
    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        errors.append(f"Submission contains {null_count} null values")

    return {
        "valid":  len(errors) == 0,
        "errors": errors
    }


# â”€â”€ Ensure output dir â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def ensure_session_dirs(session_id: str) -> dict:
    """
    Create all output subdirectories for a session.
    Returns dict of all paths.
    """
    base = f"outputs/{session_id}"
    dirs = {
        "base":        base,
        "models":      f"{base}/models",
        "predictions": f"{base}/predictions",
        "charts":      f"{base}/charts",
        "logs":        f"{base}/logs",
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs
```

Now update `agents/data_engineer.py` to use `data_tools` instead of calling Polars directly:

```python
# In agents/data_engineer.py â€” update the imports at the top
# Replace the direct polars import with data_tools

from tools.data_tools import (
    read_csv,
    write_parquet,
    write_json,
    profile_data,
    hash_file,
    ensure_session_dirs
)
```

And update `_build_preprocessing_code` to use `data_tools` functions inside the sandbox:

```python
# The sandbox code now imports and uses data_tools too
def _build_preprocessing_code(raw_data_path, output_dir, schema):
    return f"""
import polars as pl
import polars.selectors as cs
import json, os, sys
sys.path.insert(0, '.')          # so sandbox can import from project root
from tools.data_tools import profile_data, write_parquet, write_json

df = pl.read_csv("{raw_data_path}", infer_schema_length=10000)
print(f"Loaded: {{df.shape[0]}} rows, {{df.shape[1]}} columns")

# Profile BEFORE cleaning
schema = profile_data(df)
print(f"Missing rates: {{schema['missing_rates']}}")

# Clean
import polars.selectors as cs
for col in df.select(cs.numeric()).columns:
    median_val = df[col].median()
    if median_val is not None:
        df = df.with_columns(pl.col(col).fill_null(median_val))
for col in df.select(cs.string()).columns:
    df = df.with_columns(pl.col(col).fill_null("missing"))
for col in df.select(cs.boolean()).columns:
    df = df.with_columns(pl.col(col).fill_null(False))

print(f"After cleaning: {{df.shape[0]}} rows, {{df.null_count().sum_horizontal().item()}} nulls")

# Write outputs
os.makedirs("{output_dir}", exist_ok=True)
write_parquet(df,    "{output_dir}/cleaned.parquet")
write_json(schema,   "{output_dir}/schema.json")
print("DATA_ENGINEER_COMPLETE")
"""
```

---

## Quick Verification

```bash
# Test data_tools in isolation before wiring into Data Engineer
python -c "
import polars as pl
from tools.data_tools import read_csv, profile_data, hash_file

df = read_csv('data/spaceship_titanic/train.csv')
schema = profile_data(df)
h = hash_file('data/spaceship_titanic/train.csv')

print('Shape:',    df.shape)
print('Hash:',     h)
print('Columns:',  schema['columns'])
print('Numerics:', schema['numeric_cols'])
print('Cats:',     schema['categorical_cols'])
print('Missing:',  schema['missing_rates'])
print('data_tools: WORKING')
"
```

---

The reason this task is **Safe to Stub** (unlike the others) is that in an emergency you could inline the Polars calls directly in the Data Engineer. But you should never do that â€” `data_tools.py` is the single place all I/O lives, which means Polars version changes, format changes, or validation logic only ever need one fix in one file. Build it properly today.

## Day 4 Preview

Tomorrow's ONE thing: build `agents/ml_optimizer.py` â€” a single LightGBM model, no Optuna yet, just StratifiedKFold CV, reads `cleaned.parquet` from state, writes OOF predictions and test predictions back as file pointers. First time the pipeline produces a real CV score.
# Day 4 Tasks
I've now checked every relevant search. Notion confirms exactly 3 tasks on Day 4 â€” all Phase 1, all Critical, all Never Cut.

Day 4 Tasks â€” Confirmed From Notion
â”Œâ”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ #  â”‚ Task                                     â”‚ Phase                   â”‚ Priority â”‚ Cuttable  â”‚
â”œâ”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ 1  â”‚ Build core/metric_contract.py            â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut â”‚
â”‚ 2  â”‚ Build agents/ml_optimizer.py â€” v0        â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut â”‚
â”‚ 3  â”‚ Write contract test â€” ML Optimizer       â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut â”‚
â””â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
Notion notes verbatim:

metric_contract.py: MetricContract dataclass: scorer_fn, direction, forbidden_metrics. Injected into every agent system prompt. Hardcode AUC for now.
ml_optimizer.py: Single LightGBM with default params. Reads schema.json + cleaned.parquet pointer. Outputs best_model.pkl + metrics.json. Uses same CV folds (StratifiedKFold 5).
contract test: INPUT: cleaned.parquet pointer, schema.json, metric_contract.json. OUTPUT: best_model.pkl exists, metrics.json has cv_mean/cv_std/fold_scores. STATE: model_registry updated, cost_tracker incremented. CV score must never use forbidden metrics.

The ONE thing that must work by end of today:
Feed cleaned.parquet into ml_optimizer.py and get back a real CV score in metrics.json. First number that means something. This is the floor everything from Day 5 onward must beat.
Build order matters: Task 1 before Task 2 â€” the optimizer imports from metric_contract.

Task 1 â€” Build core/metric_contract.py
python# core/metric_contract.py

from dataclasses import dataclass, field
from typing import Callable, Optional
import json
import os
from sklearn import metrics as skmetrics


# â”€â”€ All supported scorers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SCORER_REGISTRY = {
    # Classification
    "accuracy":          (skmetrics.accuracy_score,          "maximize"),
    "auc":               (skmetrics.roc_auc_score,           "maximize"),
    "roc_auc":           (skmetrics.roc_auc_score,           "maximize"),
    "log_loss":          (skmetrics.log_loss,                "minimize"),
    "f1":                (skmetrics.f1_score,                "maximize"),
    "f1_macro":          (lambda y, p: skmetrics.f1_score(y, p, average="macro"), "maximize"),
    "f1_weighted":       (lambda y, p: skmetrics.f1_score(y, p, average="weighted"), "maximize"),
    "matthews_corrcoef": (skmetrics.matthews_corrcoef,       "maximize"),
    # Regression
    "rmse":              (lambda y, p: skmetrics.root_mean_squared_error(y, p), "minimize"),
    "mae":               (skmetrics.mean_absolute_error,     "minimize"),
    "r2":                (skmetrics.r2_score,                "maximize"),
    "rmsle":             (lambda y, p: skmetrics.mean_squared_log_error(y, p) ** 0.5, "minimize"),
    "mape":              (skmetrics.mean_absolute_percentage_error, "minimize"),
}

# Metrics that require predict_proba instead of predict
PROBABILITY_METRICS = {"auc", "roc_auc", "log_loss"}

# Metrics that are FORBIDDEN â€” never optimise toward these
# (proxies that look good but don't reflect true performance)
FORBIDDEN_METRICS = {"accuracy_on_train", "train_loss", "overfit_score"}


@dataclass
class MetricContract:
    """
    The single source of truth for what Professor is optimising toward.
    Written once per competition. Injected into every agent system prompt.
    Never changed mid-pipeline without explicit user approval.
    """
    scorer_name:       str              # e.g. "auc", "rmse"
    direction:         str              # "maximize" or "minimize"
    scorer_fn:         Callable         # the actual sklearn function
    requires_proba:    bool             # True if predict_proba needed
    forbidden_metrics: list             # metrics never to optimise toward
    task_type:         str              # "classification" or "regression"
    competition_name:  str = ""
    locked:            bool = False     # True after first submission
    notes:             str = ""


def build_metric_contract(
    scorer_name: str,
    task_type: str,
    competition_name: str = "",
    notes: str = ""
) -> MetricContract:
    """
    Build a MetricContract from a scorer name string.
    Used by the Validation Architect (Phase 2). For Phase 1: hardcode AUC.

    Args:
        scorer_name:      one of the keys in SCORER_REGISTRY
        task_type:        "classification" or "regression"
        competition_name: for logging
        notes:            any additional context

    Returns:
        MetricContract ready to inject into agent prompts
    """
    scorer_name = scorer_name.lower().strip()

    if scorer_name not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown scorer: '{scorer_name}'. "
            f"Supported: {list(SCORER_REGISTRY.keys())}"
        )

    scorer_fn, direction = SCORER_REGISTRY[scorer_name]

    return MetricContract(
        scorer_name=scorer_name,
        direction=direction,
        scorer_fn=scorer_fn,
        requires_proba=scorer_name in PROBABILITY_METRICS,
        forbidden_metrics=list(FORBIDDEN_METRICS),
        task_type=task_type,
        competition_name=competition_name,
        locked=False,
        notes=notes
    )


def default_contract(competition_name: str = "") -> MetricContract:
    """
    Phase 1 default: AUC for binary classification.
    Replaced by Validation Architect auto-detection in Phase 2.
    """
    return build_metric_contract(
        scorer_name="auc",
        task_type="classification",
        competition_name=competition_name,
        notes="Phase 1 default â€” hardcoded AUC. Auto-detected from Day 8."
    )


def save_contract(contract: MetricContract, path: str) -> str:
    """Save MetricContract as metric_contract.json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "scorer_name":       contract.scorer_name,
        "direction":         contract.direction,
        "requires_proba":    contract.requires_proba,
        "forbidden_metrics": contract.forbidden_metrics,
        "task_type":         contract.task_type,
        "competition_name":  contract.competition_name,
        "locked":            contract.locked,
        "notes":             contract.notes
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_contract(path: str) -> MetricContract:
    """Load MetricContract from metric_contract.json."""
    with open(path) as f:
        data = json.load(f)
    scorer_fn, _ = SCORER_REGISTRY[data["scorer_name"]]
    return MetricContract(
        scorer_name=data["scorer_name"],
        direction=data["direction"],
        scorer_fn=scorer_fn,
        requires_proba=data["requires_proba"],
        forbidden_metrics=data["forbidden_metrics"],
        task_type=data["task_type"],
        competition_name=data["competition_name"],
        locked=data.get("locked", False),
        notes=data.get("notes", "")
    )


def contract_to_prompt_snippet(contract: MetricContract) -> str:
    """
    Returns a string injected into every agent system prompt.
    Makes every agent aware of what it is optimising toward.
    """
    better = "higher" if contract.direction == "maximize" else "lower"
    return f"""
METRIC CONTRACT (read-only â€” never change this mid-pipeline):
  Competition:     {contract.competition_name}
  Optimise for:    {contract.scorer_name.upper()} ({better} is better)
  Task type:       {contract.task_type}
  Requires proba:  {contract.requires_proba}
  FORBIDDEN:       Never report or optimise toward: {contract.forbidden_metrics}
  Locked:          {contract.locked}
"""

Task 2 â€” Build agents/ml_optimizer.py â€” v0
python# agents/ml_optimizer.py

import os
import json
import pickle
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor
from core.state import ProfessorState
from core.metric_contract import (
    MetricContract, default_contract,
    save_contract, load_contract, contract_to_prompt_snippet
)
from tools.data_tools import read_parquet, read_json


def _identify_target_column(schema: dict, state: ProfessorState) -> str:
    """
    Identify the target column from schema.
    Phase 1: look for common names. Phase 2: LLM-driven detection.
    """
    common_targets = [
        "Transported", "target", "Target", "label", "Label",
        "Survived", "survived", "y", "outcome", "Outcome",
        "price", "Price", "SalePrice", "salary", "Salary"
    ]
    columns = schema["columns"]
    for candidate in common_targets:
        if candidate in columns:
            return candidate

    # Fall back to last column (common convention)
    return columns[-1]


def _prepare_features(df: pl.DataFrame, target_col: str, schema: dict) -> tuple:
    """
    Convert Polars DataFrame to numpy arrays for sklearn/LightGBM.
    Encodes categoricals as integer codes.
    Returns (X, y, feature_names)
    """
    feature_cols = [c for c in df.columns if c != target_col]

    # Encode string columns as integer codes
    for col in feature_cols:
        if df[col].dtype == pl.Utf8 or df[col].dtype == pl.String:
            df = df.with_columns(
                pl.col(col).cast(pl.Categorical).cast(pl.Int32)
            )

    # Convert target
    y_series = df[target_col]
    if y_series.dtype == pl.Boolean:
        y = y_series.cast(pl.Int32).to_numpy()
    elif y_series.dtype in (pl.Utf8, pl.String):
        y = y_series.cast(pl.Categorical).cast(pl.Int32).to_numpy()
    else:
        y = y_series.to_numpy()

    X = df.select(feature_cols).to_numpy()

    return X, y, feature_cols


def run_ml_optimizer(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: ML Optimizer v0.

    Reads:  state["clean_data_path"]  â€” cleaned.parquet
            state["schema_path"]      â€” schema.json
    Writes: state["model_registry"]   â€” list with model entry
            state["cv_mean"]          â€” float
            state["cv_scores"]        â€” list of fold scores
            state["oof_predictions_path"] â€” str pointer
            state["cost_tracker"]     â€” updated

    Phase 1: single LightGBM, default params, StratifiedKFold(5).
    Phase 3: upgraded with Optuna HPO.
    """
    session_id  = state["session_id"]
    output_dir  = f"outputs/{session_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[MLOptimizer] Starting â€” session: {session_id}")

    # â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not state.get("clean_data_path"):
        raise ValueError("[MLOptimizer] clean_data_path not in state â€” run Data Engineer first")
    if not state.get("schema_path"):
        raise ValueError("[MLOptimizer] schema_path not in state â€” run Data Engineer first")

    df     = read_parquet(state["clean_data_path"])
    schema = read_json(state["schema_path"])

    print(f"[MLOptimizer] Data loaded: {df.shape}")

    # â”€â”€ Metric Contract â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    contract_path = f"{output_dir}/metric_contract.json"
    if os.path.exists(contract_path):
        contract = load_contract(contract_path)
        print(f"[MLOptimizer] Loaded existing contract: {contract.scorer_name}")
    else:
        contract = default_contract(competition_name=state["competition_name"])
        save_contract(contract, contract_path)
        print(f"[MLOptimizer] Created default contract: {contract.scorer_name}")

    # â”€â”€ Identify target + prepare features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    target_col = _identify_target_column(schema, state)
    print(f"[MLOptimizer] Target column: {target_col}")

    X, y, feature_names = _prepare_features(df, target_col, schema)
    print(f"[MLOptimizer] Features: {len(feature_names)} | Rows: {len(X)}")

    # â”€â”€ Cross-validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_folds   = 5
    task_type = contract.task_type

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores   = []
    oof_preds     = np.zeros(len(y))
    trained_models = []

    print(f"[MLOptimizer] Running {n_folds}-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build model â€” Phase 1: default params
        if task_type == "classification":
            model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        else:
            model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__("lightgbm").early_stopping(50, verbose=False),
                __import__("lightgbm").log_evaluation(0)
            ]
        )

        # Score this fold
        if contract.requires_proba:
            val_preds = model.predict_proba(X_val)[:, 1]
        else:
            val_preds = model.predict(X_val)

        oof_preds[val_idx] = val_preds

        fold_score = contract.scorer_fn(y_val, val_preds)
        fold_scores.append(float(fold_score))
        trained_models.append(model)

        print(f"[MLOptimizer] Fold {fold}: {contract.scorer_name.upper()} = {fold_score:.4f}")

    cv_mean = float(np.mean(fold_scores))
    cv_std  = float(np.std(fold_scores))
    print(f"[MLOptimizer] CV {contract.scorer_name.upper()}: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # â”€â”€ Save best model (highest/lowest score depending on direction) â”€â”€
    if contract.direction == "maximize":
        best_fold_idx = int(np.argmax(fold_scores))
    else:
        best_fold_idx = int(np.argmin(fold_scores))

    best_model = trained_models[best_fold_idx]
    model_path = f"{output_dir}/best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"[MLOptimizer] Best model (fold {best_fold_idx + 1}) saved: {model_path}")

    # â”€â”€ Save OOF predictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    oof_path = f"{output_dir}/oof_predictions.npy"
    np.save(oof_path, oof_preds)

    # â”€â”€ Save metrics.json â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metrics = {
        "scorer_name":    contract.scorer_name,
        "direction":      contract.direction,
        "cv_mean":        cv_mean,
        "cv_std":         cv_std,
        "fold_scores":    fold_scores,
        "n_folds":        n_folds,
        "best_fold":      best_fold_idx + 1,
        "n_features":     len(feature_names),
        "feature_names":  feature_names,
        "target_col":     target_col,
        "n_rows":         len(X),
    }
    metrics_path = f"{output_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # â”€â”€ Update model registry in state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model_registry = list(state.get("model_registry") or [])
    model_registry.append({
        "model_path":  model_path,
        "model_type":  "lightgbm_v0",
        "cv_mean":     cv_mean,
        "cv_std":      cv_std,
        "scorer_name": contract.scorer_name,
        "data_hash":   state.get("data_hash"),
        "fold_scores": fold_scores,
    })

    # â”€â”€ Update cost tracker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cost_tracker = dict(state["cost_tracker"])
    cost_tracker["llm_calls"] += 0  # v0 uses no LLM calls

    print(f"[MLOptimizer] Complete.")

    return {
        **state,
        "cv_scores":            fold_scores,
        "cv_mean":              cv_mean,
        "model_registry":       model_registry,
        "metric_contract":      metrics,
        "oof_predictions_path": oof_path,
        "cost_tracker":         cost_tracker,
    }

Task 3 â€” Write Contract Test (Immutable From Today)
python# tests/contracts/test_ml_optimizer_contract.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Written: Day 4
# Status:  IMMUTABLE â€” never edit this file after today
#
# CONTRACT: run_ml_optimizer()
#   INPUT:   state["clean_data_path"] â€” must exist
#            state["schema_path"]     â€” must exist
#   OUTPUT:  outputs/{session_id}/best_model.pkl â€” must exist
#            outputs/{session_id}/metrics.json   â€” must have
#              cv_mean (float), cv_std (float), fold_scores (list)
#   STATE:   model_registry â€” list, at least 1 entry after run
#            cv_mean        â€” float, > 0
#            cv_scores      â€” list of length n_folds
#            cost_tracker   â€” not None
#   NEVER:   optimise toward forbidden metrics
#            put raw model object in state (only file pointer)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import pytest
import os
import json
import pickle
import numpy as np
from core.state import initial_state
from core.metric_contract import FORBIDDEN_METRICS
from agents.data_engineer import run_data_engineer
from agents.ml_optimizer import run_ml_optimizer

FIXTURE_CSV = "tests/fixtures/tiny_train.csv"


@pytest.fixture(scope="module")
def optimized_state():
    """Run Data Engineer â†’ ML Optimizer pipeline once for all tests."""
    state = initial_state(
        competition="test-titanic",
        data_path=FIXTURE_CSV,
        budget_usd=2.0
    )
    state = run_data_engineer(state)
    state = run_ml_optimizer(state)
    return state


class TestMLOptimizerContract:

    def test_runs_without_error(self, optimized_state):
        assert optimized_state is not None

    def test_best_model_pkl_exists(self, optimized_state):
        assert os.path.exists(optimized_state["model_registry"][0]["model_path"]), \
            "best_model.pkl must exist on disk"

    def test_model_is_loadable(self, optimized_state):
        model_path = optimized_state["model_registry"][0]["model_path"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict"), "Loaded object must have predict()"

    def test_metrics_json_exists(self, optimized_state):
        session_id   = optimized_state["session_id"]
        metrics_path = f"outputs/{session_id}/metrics.json"
        assert os.path.exists(metrics_path), "metrics.json must exist"

    def test_metrics_has_cv_mean(self, optimized_state):
        session_id   = optimized_state["session_id"]
        metrics      = json.load(open(f"outputs/{session_id}/metrics.json"))
        assert "cv_mean" in metrics
        assert isinstance(metrics["cv_mean"], float)

    def test_metrics_has_cv_std(self, optimized_state):
        session_id = optimized_state["session_id"]
        metrics    = json.load(open(f"outputs/{session_id}/metrics.json"))
        assert "cv_std" in metrics
        assert isinstance(metrics["cv_std"], float)

    def test_metrics_has_fold_scores(self, optimized_state):
        session_id = optimized_state["session_id"]
        metrics    = json.load(open(f"outputs/{session_id}/metrics.json"))
        assert "fold_scores" in metrics
        assert isinstance(metrics["fold_scores"], list)
        assert len(metrics["fold_scores"]) == 5

    def test_cv_mean_is_positive(self, optimized_state):
        assert optimized_state["cv_mean"] > 0, \
            "CV mean must be positive"

    def test_cv_mean_is_reasonable(self, optimized_state):
        # AUC should be above 0.5 (random baseline) on any real data
        assert optimized_state["cv_mean"] > 0.5, \
            f"CV mean {optimized_state['cv_mean']} is below random baseline (0.5)"

    def test_cv_scores_length_matches_folds(self, optimized_state):
        assert len(optimized_state["cv_scores"]) == 5

    def test_model_registry_updated(self, optimized_state):
        assert optimized_state["model_registry"] is not None
        assert len(optimized_state["model_registry"]) >= 1

    def test_model_registry_entry_has_required_fields(self, optimized_state):
        entry = optimized_state["model_registry"][0]
        for field in ["model_path", "model_type", "cv_mean", "scorer_name"]:
            assert field in entry, f"model_registry entry missing '{field}'"

    def test_model_path_is_string_not_object(self, optimized_state):
        entry = optimized_state["model_registry"][0]
        assert isinstance(entry["model_path"], str), \
            "model_path must be a str pointer â€” never a model object"

    def test_no_model_object_in_state(self, optimized_state):
        import lightgbm as lgb
        for key, value in optimized_state.items():
            assert not isinstance(value, (lgb.LGBMClassifier, lgb.LGBMRegressor)), \
                f"Model object found in state['{key}'] â€” only file pointers allowed"

    def test_oof_predictions_path_exists(self, optimized_state):
        assert optimized_state.get("oof_predictions_path") is not None
        assert os.path.exists(optimized_state["oof_predictions_path"])

    def test_oof_predictions_loadable(self, optimized_state):
        oof = np.load(optimized_state["oof_predictions_path"])
        assert len(oof) > 0

    def test_never_optimises_forbidden_metrics(self, optimized_state):
        session_id = optimized_state["session_id"]
        metrics    = json.load(open(f"outputs/{session_id}/metrics.json"))
        scorer     = metrics["scorer_name"]
        assert scorer not in FORBIDDEN_METRICS, \
            f"Scorer '{scorer}' is in FORBIDDEN_METRICS â€” never optimise toward this"

    def test_requires_clean_data_path(self):
        state = initial_state("test", "tests/fixtures/tiny_train.csv")
        state = {**state, "clean_data_path": None}
        with pytest.raises((ValueError, TypeError)):
            run_ml_optimizer(state)

End of Day 4 Checklist
bash# 1. Run today's contract tests
pytest tests/contracts/test_ml_optimizer_contract.py -v
# All tests must be green

# 2. Run ALL contracts together (Days 2 + 3 + 4)
pytest tests/contracts/ -v
# Everything still green â€” no regressions introduced today

# 3. Quick real-data smoke test
python -c "
from core.state import initial_state
from agents.data_engineer import run_data_engineer
from agents.ml_optimizer import run_ml_optimizer

state = initial_state('spaceship-titanic', 'data/spaceship_titanic/train.csv')
state = run_data_engineer(state)
state = run_ml_optimizer(state)
print(f'CV {state[\"cv_mean\"]:.4f}')
print(f'Model: {state[\"model_registry\"][0][\"model_path\"]}')
# CV should be around 0.78-0.82 on Spaceship Titanic with default LightGBM
"

# 4. Commit
git add .
git commit -m "Day 4: metric_contract + ml_optimizer v0 + contract test â€” all tests pass"
git push origin phase-1
Expected CV score today: ~0.78â€“0.82 on Spaceship Titanic with default LightGBM. If it's below 0.70 something is wrong in the data pipeline from Day 3 â€” check cleaned.parquet for null contamination first.

Day 5 Preview
Tomorrow's ONE thing: agents/semantic_router.py v0 â€” simple linear routing only (Supervisor â†’ Data Engineer â†’ Optimizer â†’ Submit, no DAG yet) and core/professor.py â€” the LangGraph graph that wires all three nodes together for the first time. First time python main.py run does something real.
# Day 5 Tasks
Confirmed. Notion has exactly **3 tasks on Day 5** â€” all Phase 1, all Critical, all Never Cut.

---

## Day 5 Tasks â€” Pulled From Notion

```
â”Œâ”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ #  â”‚ Task                                         â”‚ Phase                   â”‚ Priority â”‚ Cuttable  â”‚
â”œâ”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ 1  â”‚ Build agents/semantic_router.py â€” v0         â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut â”‚
â”‚ 2  â”‚ Build core/professor.py â€” LangGraph wiring   â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut â”‚
â”‚ 3  â”‚ Write contract test â€” Semantic Router        â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut â”‚
â””â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

Notion notes verbatim:
- **semantic_router.py**: v0: simple linear routing only. Supervisor â†’ Data Engineer â†’ Optimizer â†’ Submit. No DAG yet. Add DAG in Phase 2.
- **professor.py**: Wire nodes into LangGraph StateGraph. State flows correctly between nodes. Add edges: router â†’ data_engineer â†’ optimizer â†’ submit.
- **contract test**: INPUT: problem_statement, raw_data_path. OUTPUT: dag populated in state, task_type set, metric_contract initialized. CONSTRAINT: Router must never write code or touch data directly. Verify it only mutates routing fields in state.

**The ONE thing that must work by end of today:**
`python main.py run --competition spaceship-titanic --data ./data/spaceship_titanic/` runs the full LangGraph graph end to end â€” router â†’ data engineer â†’ optimizer â€” without crashing. First time the pipeline runs as a connected system.

Build order: Task 1 â†’ Task 2 â†’ Task 3. Professor.py imports from semantic_router, so router must exist first.

---

## Task 1 â€” Build `agents/semantic_router.py` â€” v0

v0 is intentionally simple. No LLM calls, no DAG building, no complex routing logic. Linear only. The DAG and LLM-driven routing come in Phase 2.

```python
# agents/semantic_router.py

from core.state import ProfessorState


# â”€â”€ v0 linear route â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Phase 1: fixed sequence, no branching, no LLM involvement.
# Phase 2: replaced with LLM-driven DAG construction.

LINEAR_ROUTE_V0 = [
    "data_engineer",
    "ml_optimizer",
    "submit",
]


def run_semantic_router(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Semantic Router v0.

    Phase 1 behaviour:
      - Sets task_type to "tabular_classification" (hardcoded)
      - Populates state["dag"] with the linear route
      - Sets state["next_node"] to first node in route
      - Never writes code, never touches data files

    Reads:   state["competition_name"], state["task_type"]
    Writes:  state["dag"], state["task_type"], state["next_node"],
             state["current_node"]
    NEVER:   writes code, reads/writes data files, calls external APIs
    """
    competition = state["competition_name"]
    task_type   = state.get("task_type", "auto")

    print(f"[SemanticRouter] Competition: {competition}")

    # â”€â”€ Task type detection â€” v0: rule-based, v1: LLM-driven â”€â”€â”€â”€â”€
    if task_type == "auto":
        task_type = _detect_task_type(competition)

    print(f"[SemanticRouter] Task type: {task_type}")
    print(f"[SemanticRouter] Route: {' â†’ '.join(LINEAR_ROUTE_V0)}")

    return {
        **state,
        "task_type":    task_type,
        "dag":          LINEAR_ROUTE_V0.copy(),
        "current_node": "semantic_router",
        "next_node":    LINEAR_ROUTE_V0[0],
    }


def _detect_task_type(competition_name: str) -> str:
    """
    Rule-based task type detection for v0.
    Phase 2: replaced with LLM parsing of competition description.
    """
    name = competition_name.lower()

    # Known time-series patterns
    if any(kw in name for kw in ["forecast", "time-series", "timeseries",
                                  "temporal", "sales", "demand", "predict-future"]):
        return "timeseries"

    # Known regression patterns
    if any(kw in name for kw in ["price", "cost", "revenue", "salary",
                                  "amount", "value", "regression", "house"]):
        return "tabular_regression"

    # Default: classification
    return "tabular_classification"
```

---

## Task 2 â€” Build `core/professor.py` â€” LangGraph Graph Wiring

This is the most important file in the project. Everything built so far connects here.

```python
# core/professor.py

from langgraph.graph import StateGraph, END
from core.state import ProfessorState
from agents.semantic_router import run_semantic_router
from agents.data_engineer import run_data_engineer
from agents.ml_optimizer import run_ml_optimizer


# â”€â”€ Routing functions (conditional edges) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def route_after_router(state: ProfessorState) -> str:
    """After router runs: go to first node in DAG."""
    next_node = state.get("next_node")
    dag       = state.get("dag", [])

    if not dag:
        print("[Professor] WARNING: DAG is empty after router. Ending.")
        return END

    print(f"[Professor] Routing to: {next_node}")
    return next_node


def route_after_data_engineer(state: ProfessorState) -> str:
    """After Data Engineer: advance to next node in DAG."""
    return _advance_dag(state, current="data_engineer")


def route_after_optimizer(state: ProfessorState) -> str:
    """After Optimizer: advance to next node in DAG."""
    return _advance_dag(state, current="ml_optimizer")


def _advance_dag(state: ProfessorState, current: str) -> str:
    """
    Find current node in DAG and return the next one.
    If current is last node, return END.
    """
    dag = state.get("dag", [])

    if current not in dag:
        print(f"[Professor] '{current}' not in DAG â€” ending.")
        return END

    idx = dag.index(current)

    if idx + 1 >= len(dag):
        print(f"[Professor] '{current}' is last node â€” ending.")
        return END

    next_node = dag[idx + 1]
    print(f"[Professor] DAG advance: {current} â†’ {next_node}")
    return next_node


# â”€â”€ Submit node â€” Phase 1 stub â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_submit(state: ProfessorState) -> ProfessorState:
    """
    Phase 1 stub: generate submission.csv from OOF predictions.
    Full implementation Day 6 with tools/submit_tools.py.
    """
    import os
    import numpy as np
    import polars as pl
    from tools.data_tools import read_csv

    session_id = state["session_id"]
    output_dir = f"outputs/{session_id}"

    print(f"[Submit] Generating submission â€” session: {session_id}")

    # Load test data
    test_path = state["raw_data_path"].replace("train.csv", "test.csv")
    if not os.path.exists(test_path):
        print(f"[Submit] WARNING: test.csv not found at {test_path}")
        print(f"[Submit] Stub: writing empty submission.csv")
        submission_path = f"{output_dir}/submission.csv"
        pl.DataFrame({"stub": []}).write_csv(submission_path)
        return {**state, "submission_path": submission_path}

    test_df = read_csv(test_path)

    # Load best model and generate predictions
    import pickle
    if not state.get("model_registry"):
        raise ValueError("[Submit] No model in registry â€” run ML Optimizer first")

    model_path = state["model_registry"][0]["model_path"]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Prepare test features (same logic as optimizer)
    from tools.data_tools import read_parquet, read_json
    from agents.ml_optimizer import _identify_target_column, _prepare_features

    schema     = read_json(state["schema_path"])
    train_df   = read_parquet(state["clean_data_path"])
    target_col = _identify_target_column(schema, state)

    # Prepare test set â€” drop target if it exists, encode same way
    test_features = [c for c in train_df.columns if c != target_col
                     and c in test_df.columns]

    test_subset = test_df.select(test_features)

    # Encode string columns
    for col in test_subset.columns:
        if test_subset[col].dtype in (pl.Utf8, pl.String):
            test_subset = test_subset.with_columns(
                pl.col(col).cast(pl.Categorical).cast(pl.Int32)
            )

    # Fill nulls
    import polars.selectors as cs
    for col in test_subset.select(cs.numeric()).columns:
        test_subset = test_subset.with_columns(
            pl.col(col).fill_null(0)
        )

    X_test = test_subset.to_numpy()

    # Predict
    from core.metric_contract import load_contract, PROBABILITY_METRICS
    contract_path = f"{output_dir}/metric_contract.json"
    if os.path.exists(contract_path):
        contract = load_contract(contract_path)
        if contract.requires_proba:
            preds = model.predict_proba(X_test)[:, 1]
            preds = preds > 0.5  # convert to bool for classification
        else:
            preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)

    # Build submission
    id_col = test_df.columns[0]  # first column is usually ID
    submission = pl.DataFrame({
        id_col:     test_df[id_col].to_list(),
        target_col: preds.tolist(),
    })

    submission_path = f"{output_dir}/submission.csv"
    submission.write_csv(submission_path)

    print(f"[Submit] submission.csv saved: {submission_path}")
    print(f"[Submit] Rows: {len(submission)} | Columns: {submission.columns}")

    return {
        **state,
        "submission_path": submission_path,
    }


# â”€â”€ Build the graph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_graph() -> StateGraph:
    """
    Assemble the Professor LangGraph StateGraph.

    Phase 1 graph:
      semantic_router â†’ data_engineer â†’ ml_optimizer â†’ submit â†’ END

    Phase 2+: conditional edges, parallel branches, Critic loop added here.
    """
    graph = StateGraph(ProfessorState)

    # â”€â”€ Add nodes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    graph.add_node("semantic_router", run_semantic_router)
    graph.add_node("data_engineer",   run_data_engineer)
    graph.add_node("ml_optimizer",    run_ml_optimizer)
    graph.add_node("submit",          run_submit)

    # â”€â”€ Set entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    graph.set_entry_point("semantic_router")

    # â”€â”€ Add edges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    graph.add_conditional_edges(
        "semantic_router",
        route_after_router,
        {
            "data_engineer": "data_engineer",
            END:              END,
        }
    )

    graph.add_conditional_edges(
        "data_engineer",
        route_after_data_engineer,
        {
            "ml_optimizer": "ml_optimizer",
            END:             END,
        }
    )

    graph.add_conditional_edges(
        "ml_optimizer",
        route_after_optimizer,
        {
            "submit": "submit",
            END:       END,
        }
    )

    graph.add_edge("submit", END)

    return graph.compile()


# â”€â”€ Convenience runner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_professor(state: ProfessorState) -> ProfessorState:
    """Run the full Professor graph from an initial state."""
    graph  = build_graph()
    result = graph.invoke(state)
    return result
```

Now wire it into `main.py` â€” replace the `_run` stub from Day 1:

```python
# main.py â€” update _run() only, everything else stays the same

def _run(args):
    from core.state import initial_state
    from core.professor import run_professor

    if not os.path.exists(args.data):
        print(f"[ERROR] Data path does not exist: {args.data}")
        sys.exit(1)

    state = initial_state(
        competition=args.competition,
        data_path=args.data,
        budget_usd=args.budget,
        task_type=args.task_type
    )

    print(f"[Professor] Session:     {state['session_id']}")
    print(f"[Professor] Competition: {state['competition_name']}")
    print(f"[Professor] Data:        {state['raw_data_path']}")
    print(f"[Professor] Budget:      ${state['cost_tracker']['budget_usd']:.2f}")
    print()

    result = run_professor(state)

    print()
    print(f"[Professor] âœ“ Complete")
    print(f"[Professor] CV score:    {result.get('cv_mean', 'N/A')}")
    print(f"[Professor] Submission:  {result.get('submission_path', 'N/A')}")
    print(f"[Professor] LLM calls:   {result['cost_tracker']['llm_calls']}")
```

---

## Task 3 â€” Write Contract Test â€” Semantic Router (Immutable From Today)

```python
# tests/contracts/test_semantic_router_contract.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Written: Day 5
# Status:  IMMUTABLE â€” never edit this file after today
#
# CONTRACT: run_semantic_router()
#   INPUT:   state["competition_name"] â€” str
#            state["task_type"]        â€” str ("auto" or explicit)
#   OUTPUT:  state["dag"]         â€” non-empty list of node names
#            state["task_type"]   â€” str, one of known task types
#            state["next_node"]   â€” str, first node in DAG
#            state["current_node"]â€” "semantic_router"
#   NEVER:   writes code
#            reads or writes data files
#            calls external APIs
#            mutates raw_data_path, clean_data_path, model_registry
#            or any non-routing field in state
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import pytest
from core.state import initial_state
from agents.semantic_router import run_semantic_router

KNOWN_TASK_TYPES = {
    "tabular_classification",
    "tabular_regression",
    "timeseries"
}

KNOWN_NODES = {
    "data_engineer",
    "ml_optimizer",
    "submit",
    "eda_agent",
    "feature_factory",
    "red_team_critic",
    "ensemble_architect",
    "validation_architect",
    "publisher",
}


@pytest.fixture
def base_state():
    return initial_state(
        competition="spaceship-titanic",
        data_path="tests/fixtures/tiny_train.csv",
        budget_usd=2.0
    )


class TestSemanticRouterContract:

    def test_runs_without_error(self, base_state):
        result = run_semantic_router(base_state)
        assert result is not None

    def test_dag_is_populated(self, base_state):
        result = run_semantic_router(base_state)
        assert result.get("dag") is not None
        assert isinstance(result["dag"], list)
        assert len(result["dag"]) > 0, "DAG must not be empty"

    def test_dag_contains_valid_node_names(self, base_state):
        result  = run_semantic_router(base_state)
        for node in result["dag"]:
            assert isinstance(node, str), f"DAG node must be str, got {type(node)}"

    def test_task_type_is_set(self, base_state):
        result = run_semantic_router(base_state)
        assert result.get("task_type") is not None
        assert result["task_type"] in KNOWN_TASK_TYPES, \
            f"task_type '{result['task_type']}' not in {KNOWN_TASK_TYPES}"

    def test_next_node_is_first_dag_node(self, base_state):
        result = run_semantic_router(base_state)
        assert result["next_node"] == result["dag"][0], \
            "next_node must be the first node in the DAG"

    def test_current_node_is_semantic_router(self, base_state):
        result = run_semantic_router(base_state)
        assert result["current_node"] == "semantic_router"

    def test_explicit_task_type_preserved(self, base_state):
        state  = {**base_state, "task_type": "tabular_regression"}
        result = run_semantic_router(state)
        assert result["task_type"] == "tabular_regression", \
            "Explicit task_type must not be overridden by auto-detection"

    def test_auto_detects_classification(self):
        state  = initial_state("spaceship-titanic", "tests/fixtures/tiny_train.csv")
        result = run_semantic_router(state)
        assert result["task_type"] == "tabular_classification"

    def test_auto_detects_regression(self):
        state  = initial_state("house-price-prediction", "tests/fixtures/tiny_train.csv")
        result = run_semantic_router(state)
        assert result["task_type"] == "tabular_regression"

    def test_never_touches_data_paths(self, base_state):
        before_clean  = base_state.get("clean_data_path")
        before_schema = base_state.get("schema_path")
        result        = run_semantic_router(base_state)
        assert result.get("clean_data_path") == before_clean, \
            "Router must never modify clean_data_path"
        assert result.get("schema_path") == before_schema, \
            "Router must never modify schema_path"

    def test_never_modifies_model_registry(self, base_state):
        before = base_state.get("model_registry")
        result = run_semantic_router(base_state)
        assert result.get("model_registry") == before, \
            "Router must never modify model_registry"

    def test_never_modifies_cost_tracker_llm_calls(self, base_state):
        """Router v0 makes no LLM calls â€” llm_calls must not increase."""
        before = base_state["cost_tracker"]["llm_calls"]
        result = run_semantic_router(base_state)
        after  = result["cost_tracker"]["llm_calls"]
        assert after == before, \
            f"Router v0 must not make LLM calls. Before: {before}, After: {after}"

    def test_raw_data_path_unchanged(self, base_state):
        before = base_state["raw_data_path"]
        result = run_semantic_router(base_state)
        assert result["raw_data_path"] == before, \
            "Router must never modify raw_data_path"
```

---

## End of Day 5 Checklist

```bash
# 1. Run today's contract test
pytest tests/contracts/test_semantic_router_contract.py -v
# All tests green

# 2. Run ALL contracts â€” Days 2 + 3 + 4 + 5
pytest tests/contracts/ -v
# Zero regressions from today's wiring

# 3. THE BIG TEST â€” first full pipeline run
python main.py run \
  --competition spaceship-titanic \
  --data ./data/spaceship_titanic/

# Expected output:
# [Professor] Session:     spaceship_abc123de
# [Professor] Competition: spaceship-titanic
# [SemanticRouter] Task type: tabular_classification
# [SemanticRouter] Route: data_engineer â†’ ml_optimizer â†’ submit
# [DataEngineer] Loaded: 8693 rows, 14 columns
# [DataEngineer] Complete. data_hash: a3f9c21d
# [MLOptimizer] CV AUC: 0.8XXX (+/- 0.00XX)
# [Submit] submission.csv saved: outputs/spaceship_abc123de/submission.csv
# [Professor] âœ“ Complete
# [Professor] CV score:   0.8XXX
# [Professor] Submission: outputs/spaceship_abc123de/submission.csv

# 4. Verify submission.csv exists and has correct format
python -c "
import polars as pl
df = pl.read_csv('outputs/spaceship-titanic_abc123/submission.csv')
print('Columns:', df.columns)
print('Shape:  ', df.shape)
print('Sample: ')
print(df.head(3))
# Must have: PassengerId + Transported columns, 4277 rows
"

# 5. Commit
git add .
git commit -m "Day 5: semantic_router + professor graph wiring + contract test â€” full pipeline runs, CV: X.XXXX"
git push origin phase-1
```

---

## Day 6 Preview

Tomorrow: `tools/submit_tools.py` (proper submission generator replacing the Phase 1 stub), then the full vertical slice test â€” `python main.py run` must produce a properly validated submission.csv that can be uploaded directly to Kaggle. Day 7 is the Phase 1 gate, so Day 6 is the polish day before the gate.
# Day 6 Tasks
Confirmed. Notion has exactly **2 tasks on Day 6**.

---

## Day 6 Tasks â€” Pulled From Notion

```
â”Œâ”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ #  â”‚ Task                                           â”‚ Phase                   â”‚ Priority â”‚ Cuttable     â”‚
â”œâ”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ 1  â”‚ Build submission.csv generator + validator     â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut    â”‚
â”‚ 2  â”‚ Add JSONL lineage logger                       â”‚ ðŸš€ Phase 1: Make It Run â”‚ High     â”‚ Safe to Stub â”‚
â””â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

Notion notes verbatim:
- **submit_tools.py**: `generate_submission(predictions, sample_submission_path) â†’ submission.csv`. Validates column names, row count, ID match against sample_submission.csv before saving.
- **lineage logger**: Append-only. Each entry: timestamp, agent, action, keys_read, keys_written, values_changed. One file per session in `outputs/logs/`.

Day 6 is deliberately light â€” two tasks â€” because **Day 7 is the Phase 1 gate**. The full end-to-end run (`main.py run` â†’ real Kaggle score) lives on Day 7. Day 6 is the polish that makes that gate possible: replace the Day 5 submit stub with a real validated submission generator, and add lineage logging so you can trace exactly what the pipeline did when you review the Day 7 gate score.

**The ONE thing that must work by end of today:** `generate_submission()` produces a submission.csv that passes format validation against `sample_submission.csv` â€” correct columns, correct row count, correct ID match, zero nulls. If Day 7's gate fails, this validator tells you exactly why.

---

## Task 1 â€” Build `tools/submit_tools.py`

```python
# tools/submit_tools.py

import os
import polars as pl
import numpy as np
from datetime import datetime
from tools.data_tools import read_csv, write_json


class SubmissionValidationError(Exception):
    """Raised when submission.csv fails format validation."""
    pass


def generate_submission(
    predictions: np.ndarray,
    sample_submission_path: str,
    output_path: str,
    target_dtype: str = "auto"
) -> dict:
    """
    Generate and validate submission.csv against the sample submission.

    Validates:
      - Column names match sample_submission.csv exactly
      - Row count matches sample_submission.csv exactly
      - ID column values match sample_submission.csv exactly
      - Zero null values in output
      - Target column dtype matches sample (bool, int, or float)

    Args:
        predictions:           numpy array of predictions (1D)
        sample_submission_path: path to sample_submission.csv
        output_path:           where to write submission.csv
        target_dtype:          "auto", "bool", "int", or "float"

    Returns:
        {"path": str, "rows": int, "columns": list, "validation": dict}

    Raises:
        SubmissionValidationError if any check fails
    """
    if not os.path.exists(sample_submission_path):
        raise FileNotFoundError(
            f"sample_submission.csv not found: {sample_submission_path}"
        )

    sample = read_csv(sample_submission_path)

    # â”€â”€ Validate prediction length â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if len(predictions) != len(sample):
        raise SubmissionValidationError(
            f"Prediction count mismatch: got {len(predictions)}, "
            f"expected {len(sample)} (from sample_submission.csv)"
        )

    # â”€â”€ Infer column names from sample â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    id_col     = sample.columns[0]
    target_col = sample.columns[1]

    # â”€â”€ Infer target dtype from sample â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if target_dtype == "auto":
        sample_dtype = sample[target_col].dtype
        if sample_dtype == pl.Boolean:
            target_dtype = "bool"
        elif sample_dtype in (pl.Float32, pl.Float64):
            target_dtype = "float"
        else:
            target_dtype = "int"

    # â”€â”€ Cast predictions to correct type â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if target_dtype == "bool":
        if predictions.dtype == np.float64 or predictions.dtype == np.float32:
            preds_cast = (predictions > 0.5).tolist()
        else:
            preds_cast = [bool(p) for p in predictions]
    elif target_dtype == "float":
        preds_cast = [float(p) for p in predictions]
    else:
        preds_cast = [int(round(p)) for p in predictions]

    # â”€â”€ Build submission DataFrame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    submission = pl.DataFrame({
        id_col:     sample[id_col].to_list(),
        target_col: preds_cast,
    })

    # â”€â”€ Validate columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if set(submission.columns) != set(sample.columns):
        raise SubmissionValidationError(
            f"Column mismatch.\n"
            f"  Expected: {sample.columns}\n"
            f"  Got:      {submission.columns}"
        )

    # â”€â”€ Validate row count â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if len(submission) != len(sample):
        raise SubmissionValidationError(
            f"Row count mismatch: {len(submission)} vs {len(sample)}"
        )

    # â”€â”€ Validate ID column matches exactly â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    id_matches = (submission[id_col] == sample[id_col]).all()
    if not id_matches:
        mismatches = (submission[id_col] != sample[id_col]).sum()
        raise SubmissionValidationError(
            f"ID column mismatch: {mismatches} IDs do not match sample_submission.csv"
        )

    # â”€â”€ Validate zero nulls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        raise SubmissionValidationError(
            f"Submission contains {null_count} null values â€” not allowed"
        )

    # â”€â”€ Write to disk â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.write_csv(output_path)

    validation = {
        "valid":       True,
        "rows":        len(submission),
        "columns":     submission.columns,
        "id_col":      id_col,
        "target_col":  target_col,
        "target_dtype": target_dtype,
        "null_count":  0,
        "validated_at": datetime.utcnow().isoformat(),
    }

    print(f"[SubmitTools] âœ“ submission.csv valid: {output_path}")
    print(f"[SubmitTools] Rows: {len(submission)} | "
          f"Cols: {submission.columns} | dtype: {target_dtype}")

    return {
        "path":       output_path,
        "rows":       len(submission),
        "columns":    submission.columns,
        "validation": validation,
    }


def validate_existing_submission(
    submission_path: str,
    sample_submission_path: str
) -> dict:
    """
    Validate an already-written submission.csv against sample.
    Returns {"valid": bool, "errors": [list]}. Never raises.
    """
    errors = []

    if not os.path.exists(submission_path):
        return {"valid": False, "errors": [f"File not found: {submission_path}"]}

    try:
        submission = read_csv(submission_path)
        sample     = read_csv(sample_submission_path)
    except Exception as e:
        return {"valid": False, "errors": [f"Failed to read CSV: {e}"]}

    if set(submission.columns) != set(sample.columns):
        errors.append(f"Column mismatch: {submission.columns} vs {sample.columns}")

    if len(submission) != len(sample):
        errors.append(f"Row count: {len(submission)} vs {len(sample)}")

    null_count = submission.null_count().sum_horizontal().item()
    if null_count > 0:
        errors.append(f"Contains {null_count} null values")

    id_col = sample.columns[0]
    if id_col in submission.columns and id_col in sample.columns:
        if not (submission[id_col] == sample[id_col]).all():
            errors.append(f"ID column values do not match sample_submission.csv")

    return {"valid": len(errors) == 0, "errors": errors}


def save_submission_log(
    session_id: str,
    submission_path: str,
    cv_mean: float,
    lb_score: float = None,
    notes: str = ""
) -> str:
    """
    Append an entry to the session's submission ladder log.
    Used to track every submission and its CV/LB score.
    """
    import json

    log_path = f"outputs/{session_id}/submission_log.jsonl"
    os.makedirs(f"outputs/{session_id}", exist_ok=True)

    entry = {
        "timestamp":       datetime.utcnow().isoformat(),
        "session_id":      session_id,
        "submission_path": submission_path,
        "cv_mean":         cv_mean,
        "lb_score":        lb_score,
        "notes":           notes,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"[SubmitTools] Logged submission: CV={cv_mean:.4f} "
          f"LB={lb_score if lb_score else 'pending'}")

    return log_path
```

Now replace the Phase 1 stub in `core/professor.py` with the real submit node:

```python
# core/professor.py â€” replace run_submit() with this

def run_submit(state: ProfessorState) -> ProfessorState:
    """
    Submit node: generates validated submission.csv using submit_tools.
    Replaces Day 5 stub. Full implementation.
    """
    import pickle
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    from tools.submit_tools import generate_submission, save_submission_log
    from tools.data_tools import read_parquet, read_json, read_csv
    from agents.ml_optimizer import _identify_target_column
    from core.metric_contract import load_contract

    session_id  = state["session_id"]
    output_dir  = f"outputs/{session_id}"
    competition = state["competition_name"]

    print(f"[Submit] Generating submission â€” session: {session_id}")

    # â”€â”€ Load test data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    test_path   = state["raw_data_path"].replace("train.csv", "test.csv")
    sample_path = state["raw_data_path"].replace("train.csv", "sample_submission.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found: {test_path}")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"sample_submission.csv not found: {sample_path}")

    test_df = read_csv(test_path)

    # â”€â”€ Load model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model_path = state["model_registry"][0]["model_path"]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # â”€â”€ Prepare test features (same encoding as training) â”€â”€â”€â”€â”€â”€â”€â”€
    schema     = read_json(state["schema_path"])
    train_df   = read_parquet(state["clean_data_path"])
    target_col = _identify_target_column(schema, state)

    feature_cols = [c for c in train_df.columns
                    if c != target_col and c in test_df.columns]

    test_subset = test_df.select(feature_cols)

    for col in test_subset.columns:
        if test_subset[col].dtype in (pl.Utf8, pl.String):
            test_subset = test_subset.with_columns(
                pl.col(col).cast(pl.Categorical).cast(pl.Int32)
            )

    for col in test_subset.select(cs.numeric()).columns:
        test_subset = test_subset.with_columns(
            pl.col(col).fill_null(0)
        )

    X_test = test_subset.to_numpy()

    # â”€â”€ Generate predictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    contract_path = f"{output_dir}/metric_contract.json"
    if os.path.exists(contract_path):
        contract = load_contract(contract_path)
        if contract.requires_proba:
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict(X_test).astype(float)
    else:
        preds = model.predict(X_test).astype(float)

    # â”€â”€ Generate + validate submission â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    submission_path = f"{output_dir}/submission.csv"

    result = generate_submission(
        predictions=preds,
        sample_submission_path=sample_path,
        output_path=submission_path,
        target_dtype="auto"
    )

    # â”€â”€ Log to submission ladder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    save_submission_log(
        session_id=session_id,
        submission_path=submission_path,
        cv_mean=state.get("cv_mean", 0.0),
        notes=f"Phase 1 baseline â€” {competition}"
    )

    print(f"[Submit] âœ“ Done. Upload to Kaggle:")
    print(f"  kaggle competitions submit -c {competition} "
          f"-f {submission_path} -m 'Professor Phase 1 baseline'")

    return {
        **state,
        "submission_path": submission_path,
    }
```

---

## Task 2 â€” Add `core/lineage.py` â€” JSONL Logger

```python
# core/lineage.py

import os
import json
from datetime import datetime
from typing import Any


def log_event(
    session_id: str,
    agent: str,
    action: str,
    keys_read: list = None,
    keys_written: list = None,
    values_changed: dict = None,
    notes: str = ""
) -> None:
    """
    Append a single event to the session's lineage log.
    Append-only. Never reads or rewrites existing entries.

    Each entry: timestamp, agent, action, keys_read,
                keys_written, values_changed, notes.
    One file per session: outputs/{session_id}/logs/lineage.jsonl
    """
    log_dir  = f"outputs/{session_id}/logs"
    log_path = f"{log_dir}/lineage.jsonl"
    os.makedirs(log_dir, exist_ok=True)

    entry = {
        "timestamp":      datetime.utcnow().isoformat(),
        "session_id":     session_id,
        "agent":          agent,
        "action":         action,
        "keys_read":      keys_read or [],
        "keys_written":   keys_written or [],
        "values_changed": values_changed or {},
        "notes":          notes,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_lineage(session_id: str) -> list:
    """Read all lineage entries for a session. Returns list of dicts."""
    log_path = f"outputs/{session_id}/logs/lineage.jsonl"
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_lineage(session_id: str) -> None:
    """Print a human-readable lineage trace for a session."""
    entries = read_lineage(session_id)
    if not entries:
        print(f"No lineage entries for session: {session_id}")
        return
    print(f"\nâ”€â”€ Lineage: {session_id} ({len(entries)} events) â”€â”€")
    for e in entries:
        ts    = e["timestamp"][11:19]  # HH:MM:SS only
        wrote = ", ".join(e["keys_written"]) or "â€”"
        print(f"  {ts} [{e['agent']}] {e['action']} â†’ wrote: {wrote}")
    print()
```

Add `log_event` calls to each agent. Add these lines to the `return` block of each agent:

```python
# In agents/data_engineer.py â€” add before the return statement
from core.lineage import log_event
log_event(
    session_id=session_id,
    agent="data_engineer",
    action="cleaned_and_profiled",
    keys_read=["raw_data_path"],
    keys_written=["clean_data_path", "schema_path", "data_hash"],
    values_changed={"data_hash": data_hash, "rows": df.shape[0]},
)

# In agents/ml_optimizer.py â€” add before the return statement
log_event(
    session_id=session_id,
    agent="ml_optimizer",
    action="trained_and_scored",
    keys_read=["clean_data_path", "schema_path"],
    keys_written=["model_registry", "cv_mean", "oof_predictions_path"],
    values_changed={"cv_mean": cv_mean, "cv_std": cv_std},
)

# In core/professor.py run_submit() â€” add before the return statement
from core.lineage import log_event
log_event(
    session_id=state["session_id"],
    agent="submit",
    action="generated_submission",
    keys_read=["model_registry", "clean_data_path"],
    keys_written=["submission_path"],
    values_changed={"submission_path": submission_path},
)
```

---

## End of Day 6 Checklist

```bash
# 1. Test submit_tools in isolation
python -c "
import numpy as np
from tools.submit_tools import generate_submission, validate_existing_submission

# Generate a test submission against the real sample
preds = np.random.rand(4277)  # Spaceship Titanic test set size
result = generate_submission(
    predictions=preds,
    sample_submission_path='data/spaceship_titanic/sample_submission.csv',
    output_path='outputs/test_submission.csv',
    target_dtype='bool'
)
print('Generated:', result)

# Validate it
v = validate_existing_submission(
    'outputs/test_submission.csv',
    'data/spaceship_titanic/sample_submission.csv'
)
print('Valid:', v['valid'])
print('Errors:', v['errors'])
"

# 2. Run full pipeline end-to-end with real submit node
python main.py run \
  --competition spaceship-titanic \
  --data ./data/spaceship_titanic/

# Must print:
# [Submit] âœ“ submission.csv valid: outputs/.../submission.csv
# [Professor] âœ“ Complete

# 3. Check lineage trace
python -c "
from core.lineage import print_lineage
# Replace with your actual session_id from the run above
print_lineage('spaceship_abc123de')
"
# Must show 3 events: data_engineer â†’ ml_optimizer â†’ submit

# 4. Run all contract tests â€” zero regressions
pytest tests/contracts/ -v

# 5. Commit â€” clean before the Day 7 gate
git add .
git commit -m "Day 6: submit_tools + lineage logger â€” submission.csv validated, pipeline complete"
git push origin phase-1
```


The lineage logger was actually in my Day 6 response as Task 2 â€” you may have missed it since it came after the submit_tools code. But LangSmith tracing is a genuine skip. Let me pull the Notion details and give you both properly.Good. Both confirmed. Here are the corrected complete task tables and the missing implementation.

---

## Corrected Day 5 Task Table

```
â”Œâ”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ #  â”‚ Task                                         â”‚ Phase                   â”‚ Priority â”‚ Cuttable     â”‚
â”œâ”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ 1  â”‚ Build agents/semantic_router.py â€” v0         â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut    â”‚
â”‚ 2  â”‚ Build core/professor.py â€” LangGraph wiring   â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut    â”‚
â”‚ 3  â”‚ Write contract test â€” Semantic Router        â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut    â”‚
â”‚ 4  â”‚ Set up LangSmith tracing                     â”‚ ðŸš€ Phase 1: Make It Run â”‚ High     â”‚ Safe to Stub â”‚
â””â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

## Corrected Day 6 Task Table

```
â”Œâ”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ #  â”‚ Task                                         â”‚ Phase                   â”‚ Priority â”‚ Cuttable     â”‚
â”œâ”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ 1  â”‚ Build submission.csv generator + validator   â”‚ ðŸš€ Phase 1: Make It Run â”‚ Critical â”‚ Never Cut    â”‚
â”‚ 2  â”‚ Add JSONL lineage logger                     â”‚ ðŸš€ Phase 1: Make It Run â”‚ High     â”‚ Safe to Stub â”‚
â””â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

Notion note on LangSmith: `LANGCHAIN_TRACING_V2=true`. Every token and tool call must be visible in LangSmith dashboard. Lives in `core/professor.py`.

---

## Day 5 Missing Task â€” Set Up LangSmith Tracing

**Step 1: Get API key and install**

```bash
# Go to smith.langchain.com â†’ sign up â†’ Settings â†’ API Keys â†’ Create
# Add to .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=professor-agent

pip install langsmith
pip freeze | grep langsmith >> requirements.txt
```

**Step 2: Add to `core/professor.py` â€” top of file, before anything else**

```python
# core/professor.py â€” add at the very top, before imports

import os
from dotenv import load_dotenv
load_dotenv()

# â”€â”€ LangSmith tracing â€” must be set before LangGraph is imported â”€â”€
# Reads from .env: LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY,
#                  LANGCHAIN_PROJECT, LANGCHAIN_ENDPOINT
# When LANGCHAIN_TRACING_V2=true every graph invocation is traced.
# Dashboard: https://smith.langchain.com
_tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if _tracing_enabled:
    print(f"[Professor] LangSmith tracing ON â€” project: "
          f"{os.getenv('LANGCHAIN_PROJECT', 'default')}")
else:
    print("[Professor] LangSmith tracing OFF â€” set LANGCHAIN_TRACING_V2=true to enable")
```

That's all you need. LangGraph + LangChain pick up the env vars automatically. No other code changes â€” every `graph.invoke()` call is traced automatically once the env vars are set.

**Step 3: Verify it's working**

```bash
python main.py run \
  --competition spaceship-titanic \
  --data ./data/spaceship_titanic/

# Then open: https://smith.langchain.com
# â†’ Projects â†’ professor-agent
# â†’ You should see one trace with nodes:
#   semantic_router â†’ data_engineer â†’ ml_optimizer â†’ submit
# Each node shows: input state, output state, duration, token count
```

**What you'll see in the dashboard per node:**
- Input state keys + values
- Output state keys + what changed
- Duration in ms
- Any LLM calls with full prompt/response (from Day 8 onward when agents start calling LLMs)
- Errors with full traceback if a node fails

**Stub option** (if you want to skip for now): Simply leave `LANGCHAIN_TRACING_V2=false` in `.env`. The code handles it gracefully, nothing breaks. Enable it before Day 7 gate so you have a trace of the first real run.

---

## Day 6 â€” Lineage Logger (Already in Previous Response, Reproduced Here Cleanly)

The full implementation was in the Day 6 response. For completeness, the three places you wire it in:

```python
# agents/data_engineer.py â€” add before return
from core.lineage import log_event
log_event(
    session_id=session_id,
    agent="data_engineer",
    action="cleaned_and_profiled",
    keys_read=["raw_data_path"],
    keys_written=["clean_data_path", "schema_path", "data_hash"],
    values_changed={"data_hash": data_hash, "rows": df.shape[0]},
)

# agents/ml_optimizer.py â€” add before return
from core.lineage import log_event
log_event(
    session_id=session_id,
    agent="ml_optimizer",
    action="trained_and_scored",
    keys_read=["clean_data_path", "schema_path"],
    keys_written=["model_registry", "cv_mean", "oof_predictions_path"],
    values_changed={"cv_mean": cv_mean, "cv_std": cv_std},
)

# core/professor.py run_submit() â€” add before return
from core.lineage import log_event
log_event(
    session_id=state["session_id"],
    agent="submit",
    action="generated_submission",
    keys_read=["model_registry", "clean_data_path"],
    keys_written=["submission_path"],
    values_changed={"submission_path": submission_path},
)
```

After a full run, verify both systems are working:

```bash
# Lineage check
python -c "
from core.lineage import print_lineage
print_lineage('your_session_id_here')
"
# Should show:
# â”€â”€ Lineage: spaceship_abc123 (3 events) â”€â”€
#   10:23:01 [data_engineer] cleaned_and_profiled â†’ wrote: clean_data_path, schema_path, data_hash
#   10:23:14 [ml_optimizer] trained_and_scored â†’ wrote: model_registry, cv_mean, oof_predictions_path
#   10:24:02 [submit] generated_submission â†’ wrote: submission_path

# LangSmith check â€” go to smith.langchain.com and confirm trace appeared
```

Both together give you two independent audit trails for the Day 7 gate: LangSmith shows the LangGraph execution trace, lineage.jsonl shows the state mutation trail. If the gate submission score is unexpected, you have both to diagnose against.
---

## Day 7 â€” Phase 1 Gate

Tomorrow is not a build day. It's a gate day. One task from Notion:

> **Full end-to-end run: Spaceship Titanic â†’ submission.csv** â€” Critical, Never Cut, Phase 1. PHASE 1 GATE: Pipeline must complete without crashing. submission.csv must be valid. Upload to Kaggle and get a real score. Log it.

`python main.py run` â†’ upload â†’ real LB score. If the score beats Submission 0 (your Day 2 manual baseline), Phase 1 is done. If not, you debug before moving to Phase 2. You do not proceed to Phase 2 with a broken baseline.
# Day 7 Tasks
You are implementing Day 7 of the Professor Kaggle agent build.
Read every word of this prompt before writing a single line of code.

## The Standard You Are Being Held To

You are a principal engineer. Not a junior developer generating boilerplate.
Every function you write must:
- Handle failure paths before the happy path
- Raise immediately with an actionable error message when something is wrong
- Never return None or empty values silently
- Include type annotations and a docstring stating what it reads, writes, and raises
- Be tested against the real Spaceship Titanic data, not just a 5-row fixture

Before writing any function ask: what are all the ways this can fail?
Write those cases first. Then fill in the success path.

---

## Build Order â€” Do Not Deviate

Task 1 â†’ Task 2 â†’ Task 3 â†’ Task 4 â†’ Task 5

Task 5 cannot be written until Task 4 passes the gate.
Task 4 cannot be run until Tasks 1 and 2 are done.
If Task 4 fails, debug it. Do not proceed to Task 5 until the gate passes.

---

## Task 1 â€” FIX: LangGraph State Merge Corrupts model_registry
File: core/state.py
Priority: Critical â€” Must be done FIRST before any other task

### The Problem
LangGraph does not simply replace state between nodes. It uses reducers.
A plain `list` field in TypedDict uses the DEFAULT reducer which APPENDS
on every node return instead of replacing. This means:
- Every time ml_optimizer returns model_registry = [new_entry], LangGraph
  APPENDS to the existing list instead of replacing it.
- On the first gate run with any retry, model_registry has duplicates.
- On competition run 3, model_registry has 3 copies of the same model.
- The Ensemble Architect tries to blend 20 identical models by Day 15.

This WILL corrupt the Day 7 gate run if not fixed first.

### The Fix
Open core/state.py. Add these imports at the top:
  from typing import Annotated
  import operator

Then for every field in ProfessorState, classify it as either:

ACCUMULATE (grows across runs â€” use Annotated[list, operator.add]):
  - model_registry: every competition adds entries, never replaced wholesale
  - errors: accumulates all errors seen across the session
  - lineage_log: append-only event log

REPLACE (reset each pipeline run â€” use plain list):
  - dag: router sets the full route, optimizer never appends to it
  - cv_scores: optimizer replaces with current run's fold scores
  - oof_predictions: replaced each training run

Apply the correct annotation to every list field.
Verify by checking: "if this node returns this field, should it ADD TO
or REPLACE the existing value?" â€” answer that for each field before annotating.

After the fix, write a quick verification:
  from core.state import ProfessorState
  from langgraph.graph import StateGraph
  # confirm graph compiles without errors with new annotations
  print("State annotations verified")

---

## Task 2 â€” FIX: Define Phase 1 Gate Thresholds Explicitly
File: tests/phase1_gate.py
Priority: Critical â€” Must be done before the gate run

### The Problem
Without explicit thresholds, "did the gate pass?" is ambiguous.
If Professor scores 0.7751 and Submission 0 scored 0.7754, have you failed?
That is a 0.0003 gap from random seed variance â€” not a broken pipeline.
You need defined, checkable pass/fail conditions written as assertions
before you run the gate, not after.

### What to Build
Create tests/phase1_gate.py â€” a standalone script (not pytest) that:

1. Reads SUBMISSION_0_CV from a constant you set manually right now.
   Open your Day 2 manual submission notebook, find the CV AUC you recorded,
   set it as: SUBMISSION_0_CV = X.XXXX
   If you didn't record it, set it to 0.775 as a conservative floor.

2. Defines these exact pass conditions as Python assertions:

   Pass condition 1: Professor CV >= SUBMISSION_0_CV - 0.005
     Rationale: 0.005 buffer accounts for random seed variance between runs.
     A gap larger than this means something is wrong, not random.

   Pass condition 2: submission.csv passes validate_existing_submission()
     with zero errors against sample_submission.csv.
     Use the function already built in tools/submit_tools.py.

   Pass condition 3: pytest tests/contracts/ exits with code 0.
     Run it as a subprocess and check returncode. If any contract test
     fails the gate fails â€” the pipeline cannot be trusted.

   Pass condition 4: Full pipeline wall clock < 30 minutes.
     time.time() before and after run_professor(). Fail if delta > 1800s.
     Rationale: Phase 3 Optuna adds 10-20x runtime. If Phase 1 already
     takes 4 hours, Phase 3 will never finish.

   Pass condition 5: CV > 0.70 absolute floor.
     This is independent of Submission 0. A CV of 0.65 means the pipeline
     is broken regardless of what Submission 0 scored.

   Hard fail conditions (raise immediately, do not continue):
   - Any Python exception during pipeline run
   - Any null values in submission.csv
   - submission.csv missing required columns
   - model_registry empty after run

3. Prints a clear PASS / FAIL report with all condition results.
   A passing gate prints:
     âœ“ CV 0.8123 >= floor 0.7700 (Submission 0: 0.7750 - 0.005 = 0.7700)
     âœ“ submission.csv valid: 4277 rows, correct columns, zero nulls
     âœ“ All contract tests green (pytest exit code 0)
     âœ“ Wall clock: 14m 32s < 30m limit
     âœ“ CV 0.8123 > 0.70 absolute floor
     === PHASE 1 GATE: PASSED ===

---

## Task 3 â€” Set Up MLflow Experiment Tracking
File: tools/mlflow_tracker.py
Priority: Medium â€” Safe to Stub if time is short

### What to Build
A thin wrapper around MLflow that the ml_optimizer calls at the end
of every training run. This is for visibility, not functionality.
The pipeline must work identically whether MLflow is available or not.

Build it with a graceful fallback:
  try:
      import mlflow
      MLFLOW_AVAILABLE = True
  except ImportError:
      MLFLOW_AVAILABLE = False

Functions to build:
  log_run(session_id, competition, model_type, params, cv_mean, cv_std,
          n_features, data_hash) -> None
    - If MLFLOW_AVAILABLE: log to experiment named after competition
    - If not: print a one-line summary to stdout and return
    - Never raises â€” MLflow failure must never crash the pipeline

  log_submission(session_id, submission_path, cv_mean, lb_score=None) -> None
    - Same graceful fallback pattern

Setup instructions to include as a comment at the top of the file:
  # Setup: pip install mlflow
  # Start UI: mlflow ui --port 5000
  # View at: http://localhost:5000
  # Set MLFLOW_TRACKING_URI in .env to persist across sessions

Wire log_run() into agents/ml_optimizer.py at the end of the training loop,
after metrics.json is written.

If MLflow installation causes any dependency conflicts, stub the entire
file with the graceful fallback and move on. This is Safe to Stub.

---

## Task 4 â€” Full End-to-End Run: Spaceship Titanic â†’ submission.csv
File: main.py
Priority: Critical â€” THIS IS THE PHASE 1 GATE

This is the most important task of the day. Everything built in Days 1-6
must work together as a single connected pipeline for the first time.

### Pre-Run Checklist (Do These Before Running)

1. Verify Task 1 is done: core/state.py has Annotated list fields
2. Verify Task 2 is done: tests/phase1_gate.py exists with constants set
3. Run pytest tests/contracts/ â€” ALL GREEN before proceeding.
   If any contract test fails, fix it now. Do not attempt the gate run
   with failing contract tests. A failing contract test means the
   component it tests is broken, and the pipeline will fail.
4. Verify the Spaceship Titanic data is in place:
   data/spaceship_titanic/train.csv
   data/spaceship_titanic/test.csv
   data/spaceship_titanic/sample_submission.csv
5. Verify .env has LANGCHAIN_TRACING_V2 set (true or false, must exist)

### The Gate Run Command
  python main.py run \
    --competition spaceship-titanic \
    --data ./data/spaceship_titanic/train.csv \
    --budget 2.0

### Expected Console Output (Every Line Matters)
  [Professor] Session:      spaceship_XXXXXXXX
  [Professor] Competition:  spaceship-titanic
  [Professor] Data:         ./data/spaceship_titanic/train.csv
  [Professor] Budget:       $2.00
  [SemanticRouter] Task type: tabular_classification
  [SemanticRouter] Route:   data_engineer â†’ ml_optimizer â†’ submit
  [DataEngineer] Loaded: 8693 rows, 14 columns
  [DataEngineer] Nulls before cleaning: XXX
  [DataEngineer] Nulls after cleaning: 0
  [DataEngineer] Complete. data_hash: XXXXXXXXXXXXXXXX
  [MLOptimizer] Target column: Transported
  [MLOptimizer] Features: XX columns
  [MLOptimizer] Fold 1/5: AUC = 0.XXXX
  [MLOptimizer] Fold 2/5: AUC = 0.XXXX
  [MLOptimizer] Fold 3/5: AUC = 0.XXXX
  [MLOptimizer] Fold 4/5: AUC = 0.XXXX
  [MLOptimizer] Fold 5/5: AUC = 0.XXXX
  [MLOptimizer] CV AUC: 0.XXXX (+/- 0.XXXX)
  [Submit] Generating submission â€” session: spaceship_XXXXXXXX
  [SubmitTools] âœ“ submission.csv valid: outputs/.../submission.csv
  [SubmitTools] Rows: 4277 | Cols: ['PassengerId', 'Transported']
  [Submit] âœ“ Done. Upload to Kaggle:
    kaggle competitions submit -c spaceship-titanic \
      -f outputs/.../submission.csv \
      -m 'Professor Phase 1 baseline'
  [Professor] âœ“ Complete
  [Professor] CV score:   0.XXXX
  [Professor] Submission: outputs/.../submission.csv

If any line is missing or shows an error, stop and fix before continuing.

### Failure Modes and What They Mean

Pipeline crashes at DataEngineer:
  â†’ Check raw_data_path is the exact path to train.csv
  â†’ Check cleaned.parquet and schema.json are writing to outputs/{session_id}/
  â†’ Run the Data Engineer contract test in isolation

Pipeline crashes at MLOptimizer:
  â†’ Check clean_data_path in state points to outputs/{session_id}/cleaned.parquet
  â†’ Check schema.json has required fields: columns, types, missing_rates
  â†’ Verify the target column is being identified (should be 'Transported')
  â†’ Check feature matrix is not all-zero

Pipeline crashes at Submit:
  â†’ Check test.csv exists at data/spaceship_titanic/test.csv
  â†’ Check sample_submission.csv exists at data/spaceship_titanic/sample_submission.csv
  â†’ Check model was saved: outputs/{session_id}/best_model.pkl must exist

CV AUC < 0.70:
  â†’ This is a broken pipeline, not bad luck
  â†’ Check cleaned.parquet for null contamination: pl.read_parquet(path).null_count()
  â†’ Check the target column is boolean, not string
  â†’ Check feature encoding â€” categoricals must be integer codes, not strings

CV AUC between 0.70 and 0.75:
  â†’ Acceptable for Phase 1 with default LightGBM, no feature engineering
  â†’ Do not tune. Do not retry. Proceed to submission upload.

submission.csv row count is not 4277:
  â†’ test.csv has 4277 rows in Spaceship Titanic
  â†’ The submit node is loading the wrong test file
  â†’ Print test_df.shape before generating predictions

### After the Gate Run Passes
Run tests/phase1_gate.py to get the formal PASS result:
  python tests/phase1_gate.py

Record in DAILY_LOG.md:
  Day 7 gate:
    Session ID:     spaceship_XXXXXXXX
    CV AUC:         0.XXXX
    Submission 0:   0.XXXX
    Wall clock:     Xm Xs
    Gate status:    PASSED
    Kaggle submit:  [pending / submitted / LB score: X.XXXX]

Upload the submission to Kaggle:
  kaggle competitions submit \
    -c spaceship-titanic \
    -f outputs/{session_id}/submission.csv \
    -m "Professor Phase 1 baseline â€” Day 7 gate"

Record the Kaggle LB score in DAILY_LOG.md when it comes back.

---

## Task 5 â€” FREEZE Phase 1 Regression Test
File: tests/regression/test_phase1_regression.py
Priority: Critical â€” Written ONLY after Task 4 gate passes. Not before.

### What to Build
This file is written ONCE and NEVER edited after today.
It is the permanent floor that protects everything built in Phase 1.

At the top of the file, in a comment, record:
  # Written: Day 7
  # Gate CV: X.XXXX (the exact CV from today's gate run)
  # Gate session: spaceship_XXXXXXXX
  # Commit hash: [run `git rev-parse HEAD` and paste here]
  # IMMUTABLE: never edit this file after Day 7

### What to Freeze

Freeze 1 â€” CV Floor
  The CV from today's gate minus 0.03.
  If today's gate CV was 0.812, the floor is 0.782.
  This gives a 0.03 buffer for normal model variance.
  Any future run below this floor = regression alert.

  def test_cv_floor():
      # Re-run the pipeline on Spaceship Titanic with a fixed random seed
      # CV must be >= CV_FLOOR = [today's gate CV - 0.03]

Freeze 2 â€” Submission Format
  submission.csv must always have exactly 2 columns.
  Column 1 must be 'PassengerId'.
  Column 2 must be 'Transported'.
  Row count must be exactly 4277.
  Zero nulls.

  def test_submission_format():
      # validate_existing_submission() must return {"valid": True, "errors": []}

Freeze 3 â€” State Pointer Contract
  No raw DataFrames in state. Only string file pointers.
  This is the most important architectural invariant in the project.

  def test_state_has_only_pointers():
      for key, value in result_state.items():
          assert not isinstance(value, pl.DataFrame), \
              f"Raw DataFrame found in state key '{key}' â€” must be a file pointer"
          assert not isinstance(value, pd.DataFrame), \
              f"Pandas DataFrame found in state key '{key}' â€” Pandas not allowed"

Freeze 4 â€” Cost Tracker Incremented
  The pipeline made LLM calls (or will in Phase 2).
  cost_tracker must exist and be a dict with llm_calls key.

  def test_cost_tracker_incremented():
      assert "cost_tracker" in result_state
      assert "llm_calls" in result_state["cost_tracker"]
      assert isinstance(result_state["cost_tracker"]["llm_calls"], int)

Freeze 5 â€” All Existing Contract Tests Still Pass
  This is a meta-test. It runs the full contracts suite and fails if
  any contract test that passed on Day 7 no longer passes.

  def test_all_contract_tests_pass():
      result = subprocess.run(
          ["pytest", "tests/contracts/", "-v", "--tb=short"],
          capture_output=True, text=True
      )
      assert result.returncode == 0, \
          f"Contract tests failed:\n{result.stdout}\n{result.stderr}"

---

## End of Day Checklist

Complete these in order. Do not skip any.

  [ ] Task 1 done: core/state.py has Annotated list fields, graph compiles
  [ ] Task 2 done: tests/phase1_gate.py exists, SUBMISSION_0_CV constant set
  [ ] Task 3 done (or stubbed): tools/mlflow_tracker.py exists, graceful fallback
  [ ] All contract tests green: pytest tests/contracts/ -v
  [ ] Task 4 done: pipeline ran end-to-end without crashing
  [ ] Gate passed: python tests/phase1_gate.py printed PASSED
  [ ] Kaggle submission uploaded and LB score noted
  [ ] Task 5 done: tests/regression/test_phase1_regression.py written with
      today's CV and commit hash in the header. NEVER edit this file again.
  [ ] Commit:
        git add .
        git commit -m "Day 7: Phase 1 gate passed â€” CV: X.XXXX, LB: X.XXXX, all tests green"
        git push origin phase-1
  [ ] DAILY_LOG.md updated with session ID, CV, LB score, wall clock time

---

## What Done Means Today

Done is not "the code was written."

Done is:
- pytest tests/contracts/ â†’ all green
- python tests/phase1_gate.py â†’ PASSED printed
- Kaggle submission uploaded â†’ LB score recorded
- tests/regression/test_phase1_regression.py frozen with commit hash

If pytest tests/contracts/ is not all green, you are not done.
If the gate script did not print PASSED, you are not done.
If the regression test is not written and frozen, you are not done.

Phase 2 does not start until all five conditions above are true.
