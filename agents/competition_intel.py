# agents/competition_intel.py

import os
import json
import time
import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
from core.state import ProfessorState
from tools.llm_client import call_llm
from core.lineage import log_event

def _fetch_notebooks(competition: str) -> list:
    """Fetch top public notebooks via Kaggle API."""
    api = KaggleApi()
    api.authenticate()
    
    notebooks = []
    try:
        # Rate limit / backoff simulation
        for attempt in range(3):
            try:
                # Get top 15 kernels sorted by votes
                kernels = api.kernels_list(competition=competition, sort_by="voteCount", page_size=15)
                for k in kernels:
                    notebooks.append({
                        "title": getattr(k, 'title', ''),
                        "author": getattr(k, 'author', ''),
                        "votes": getattr(k, 'totalVotes', 0),
                        "ref": getattr(k, 'ref', '')
                    })
                break
            except Exception as e:
                time.sleep(2 ** attempt)
    except Exception as e:
        print(f"[CompetitionIntel] Failed to fetch notebooks: {e}")
        
    return notebooks

def _synthesize_brief(notebooks: list, competition: str) -> dict:
    source_count = len(notebooks)
    
    if source_count == 0:
        return {
            "critical_findings": [],
            "proven_features": [],
            "known_leaks": [],
            "external_datasets": [],
            "dominant_approach": "unknown",
            "cv_strategy_hint": "unknown",
            "forbidden_techniques": [],
            "shakeup_risk": "medium",
            "source_post_count": 0,
            "scraped_at": datetime.datetime.utcnow().isoformat()
        }

    # Format the notebook data into a prompt
    text_data = f"Competition: {competition}\n\nTop Public Notebooks:\n"
    for i, nb in enumerate(notebooks[:10]):
        text_data += f"{i+1}. Title: {nb['title']} | Author: {nb['author']} | Votes: {nb['votes']}\n"
        
    sys_prompt = (
        "You are a Kaggle Grandmaster reading competition public notebooks and forum signals.\n"
        "Extract only actionable competitive intelligence — not general ML advice.\n"
        "For critical_findings: only include things that are non-obvious and will affect the final score.\n"
        "For known_leaks: only include leaks that have been confirmed by multiple sources or by the host.\n"
        "For proven_features: only include features that improved scores in shared notebooks.\n"
        "Be specific. No vague statements like 'feature engineering is important'.\n"
        "Output ONLY valid JSON matching the schema below. No preamble, no explanation.\n\n"
        "Schema:\n"
        "{\n"
        '  "critical_findings": ["str"],\n'
        '  "proven_features": ["str"],\n'
        '  "known_leaks": ["str"],\n'
        '  "external_datasets": ["str"],\n'
        '  "dominant_approach": "str",\n'
        '  "cv_strategy_hint": "str",\n'
        '  "forbidden_techniques": ["str"],\n'
        '  "shakeup_risk": "low" | "medium" | "high",\n'
        f'  "source_post_count": {source_count},\n'
        '  "scraped_at": "timestamp"\n'
        "}"
    )

    try:
        response = call_llm(sys_prompt, text_data, model="deepseek")
        
        # Parse JSON
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(response[start:end+1])
            parsed["source_post_count"] = source_count
            parsed["scraped_at"] = datetime.datetime.utcnow().isoformat()
            
            # Ensure all required keys exist
            required = ['critical_findings','proven_features','known_leaks','external_datasets',
                        'dominant_approach','cv_strategy_hint','forbidden_techniques','shakeup_risk']
            for k in required:
                if k not in parsed:
                    if k in ('critical_findings','proven_features','known_leaks','external_datasets','forbidden_techniques'):
                        parsed[k] = []
                    else:
                        parsed[k] = "unknown"
                        
            return parsed
    except Exception as e:
        print(f"[CompetitionIntel] LLM synthesis failed, returning fallback: {e}")

    # Fallback
    return {
        "critical_findings": ["Use groups if present to avoid LB shakeup."],
        "proven_features": ["target_encoded_features"],
        "known_leaks": [],
        "external_datasets": [],
        "dominant_approach": "LightGBM or XGBoost trees",
        "cv_strategy_hint": "StratifiedKFold or GroupKFold",
        "forbidden_techniques": [],
        "shakeup_risk": "medium",
        "source_post_count": source_count,
        "scraped_at": datetime.datetime.utcnow().isoformat()
    }


def run_competition_intel(state: ProfessorState) -> ProfessorState:
    """
    LangGraph node: Competition Intel (GM-CAP 1)
    
    Reads:  state["competition_name"]
    Writes: intel_brief.json, competition_brief.json
            state["intel_brief_path"]
            state["competition_brief_path"]
            state["competition_brief"]
    """
    session_id = state["session_id"]
    output_dir = f"outputs/{session_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    comp_name = state.get("competition_name")
    print(f"[CompetitionIntel] Starting — session: {session_id} for {comp_name}")
    
    # 1. Scrape data
    notebooks = []
    if comp_name:
        notebooks = _fetch_notebooks(comp_name)
    
    print(f"[CompetitionIntel] Found {len(notebooks)} public notebooks.")
        
    # 2. Synthesize using LLM
    brief = _synthesize_brief(notebooks, comp_name or "unknown")
    
    # 3. Save
    intel_path = f"{output_dir}/intel_brief.json"
    with open(intel_path, "w") as f:
        json.dump(brief, f, indent=2)
        
    # For now, competition_brief is the same as intel_brief
    comp_brief_path = f"{output_dir}/competition_brief.json"
    with open(comp_brief_path, "w") as f:
        json.dump(brief, f, indent=2)
        
    log_event(
        session_id=session_id,
        agent="competition_intel",
        action="scraped_intelligence",
        keys_read=["competition_name"],
        keys_written=["intel_brief_path", "competition_brief_path"],
        values_changed={"source_post_count": brief["source_post_count"]},
    )
    
    # Increment LLM cost tracking
    cost_tracker = dict(state.get("cost_tracker", {}))
    llm_calls = cost_tracker.get("llm_calls", 0) + 1
    total_usd = cost_tracker.get("total_cost_usd", 0.0) + 0.05
    cost_tracker["llm_calls"] = llm_calls
    cost_tracker["total_cost_usd"] = total_usd
    
    print(f"[CompetitionIntel] Complete.")
    
    return {
        **state,
        "intel_brief_path": intel_path,
        "competition_brief_path": comp_brief_path,
        "competition_brief": brief,
        "cost_tracker": cost_tracker
    }
