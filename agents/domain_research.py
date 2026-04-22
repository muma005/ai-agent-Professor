# agents/domain_research.py

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from core.state import ProfessorState
from core.lineage import log_event
from tools.llm_provider import llm_call, _safe_json_loads
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "domain_researcher"

# ── Domain Templates (Milestone 2) ───────────────────────────────────────────

DOMAIN_TEMPLATES = {
    "tabular_finance": {
        "key_metrics": ["Sharpe Ratio", "Log Loss", "Precision"],
        "common_features": ["Lagged returns", "Volatility", "Rolling averages", "Order book imbalance"],
        "risks": ["Look-ahead bias", "Non-stationarity", "Regime change"]
    },
    "healthcare": {
        "key_metrics": ["AUC-ROC", "F1-Score", "Recall", "Brier Score"],
        "common_features": ["Age buckets", "Co-morbidities", "Lab value deltas", "Interaction of drug/age"],
        "risks": ["Data leakage (ID-based)", "Class imbalance", "Ethical bias"]
    },
    "ecommerce": {
        "key_metrics": ["RMSE", "MAE", "MAPE", "NDCG"],
        "common_features": ["User-Item history", "Recency", "Frequency", "Monetary (RFM)", "Session depth"],
        "risks": ["Popularity bias", "Seasonality", "Churn definition drift"]
    }
}

# ── Sub-function 1: Domain Classification ───────────────────────────────────

def _classify_domain(competition_name: str, brief_text: str) -> str:
    """Keyword-based classification + LLM narrowing."""
    text = (competition_name + " " + brief_text).lower()
    
    # Keyword scores
    scores = {
        "tabular_finance": sum(1 for k in ["bank", "stock", "price", "market", "trade", "finance", "crypto"] if k in text),
        "healthcare": sum(1 for k in ["patient", "drug", "disease", "clinical", "health", "medical", "diagnosis"] if k in text),
        "ecommerce": sum(1 for k in ["customer", "sale", "click", "buy", "recommen", "retail", "product"] if k in text)
    }
    
    top_domain = max(scores, key=scores.get)
    if scores[top_domain] > 0:
        return top_domain
    return "generic_tabular"

# ── Agent Node ───────────────────────────────────────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_domain_research(state: ProfessorState) -> ProfessorState:
    """
    Intelligence Layer: 3-Stage Domain Research Engine.
    """
    session_id = state.get("session_id", "default")
    output_dir = Path(f"outputs/{session_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{AGENT_NAME}] Starting research...")

    comp_name = state.get("competition_name", "unknown")
    brief = state.get("competition_brief", {})
    brief_text = brief.get("problem_summary", "")

    # Stage 1: Classify
    domain = _classify_domain(comp_name, brief_text)
    template = DOMAIN_TEMPLATES.get(domain, {"key_metrics": [], "common_features": [], "risks": []})

    # Stage 2: Knowledge Acquisition (LLM Domain Reasoning)
    prompt = f"""You are a world-class domain expert in {domain}.
Kaggle Competition: {comp_name}
Context: {brief_text}

Research the typical feature engineering and validation strategies for this domain.
Provide a JSON with:
1. "engineering_ideas": List[str] (At least 5 specific domain-driven features)
2. "critical_risks": List[str] (Specific leakage or drift risks for this domain)
3. "validation_strategy": str (How do experts validate this type of data?)
4. "external_data_suggestions": List[str]
"""
    try:
        response = llm_call(prompt, system_prompt=f"You are a domain authority in {domain}.")
        research_json = _safe_json_loads(response)
    except:
        # Fallback to template
        research_json = {
            "engineering_ideas": template["common_features"],
            "critical_risks": template["risks"],
            "validation_strategy": "KFold with domain-specific grouping",
            "external_data_suggestions": []
        }

    # Stage 3: Structuring Domain Brief
    domain_brief = {
        "domain": domain,
        "knowledge": research_json,
        "template_context": template,
        "scraped_at": datetime.now(timezone.utc).isoformat()
    }
    
    brief_path = output_dir / "domain_research_brief.json"
    with open(brief_path, "w") as f:
        json.dump(domain_brief, f, indent=2)

    # Update State
    updates = {
        "competition_context": domain_brief,
        "intel_brief_path": str(brief_path)
    }

    log_event(
        session_id=session_id,
        agent=AGENT_NAME,
        action="domain_research_complete",
        keys_written=list(updates.keys())
    )

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
