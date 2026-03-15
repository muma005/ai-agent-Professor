# agents/competition_intel.py

import os
import json
import time
import logging
import datetime
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from core.state import ProfessorState
from tools.llm_client import call_llm
from core.lineage import log_event
from guards.agent_retry import with_agent_retry

logger = logging.getLogger(__name__)

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


@with_agent_retry("CompetitionIntel")
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
        try:
            notebooks = _fetch_notebooks(comp_name)
        except Exception as e:
            print(f"[CompetitionIntel] Kaggle API failed: {e}. Continuing with empty notebooks.")
            notebooks = []
    
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

    updated_state = {
        **state,
        "intel_brief_path": intel_path,
        "competition_brief_path": comp_brief_path,
        "competition_brief": brief,
        "cost_tracker": cost_tracker
    }

    # Day 15: External Data Scout — runs after competition_brief.json is written
    updated_state = run_external_data_scout(updated_state)

    return updated_state


# ── Day 15: External Data Scout ───────────────────────────────────

EXTERNAL_DATA_PROMPT = """
You are an expert Kaggle competitor. Given the competition brief below, identify external data sources that could improve model performance.

Competition: {competition_name}
Task type: {task_type}
Target: {target_description}
Domain: {domain}
Features available: {feature_summary}

Return a JSON object with the following structure:
{{
  "external_sources": [
    {{
      "name": "Human-readable name of the source",
      "type": "pretrained_model | public_dataset | external_signal",
      "description": "One sentence: what it contains and why it is relevant",
      "source_url": "URL or package name where it can be obtained",
      "relevance_score": 0.0 to 1.0,
      "join_strategy": "How to join to training data (key column or embedding approach)",
      "acquisition_method": "pip install X | wget URL | kaggle datasets download ...",
      "competition_precedent": "Known competition where this source helped (or null)"
    }}
  ],
  "recommended_sources": ["name1", "name2"],
  "total_sources_found": N,
  "scout_notes": "Any caveats or important considerations"
}}

Only include sources with relevance_score >= 0.6. Sort by relevance_score descending.
If no relevant external data exists, return an empty external_sources list.
Do NOT invent sources. Only recommend sources you are confident exist.
"""


def run_external_data_scout(state: ProfessorState) -> ProfessorState:
    """
    Searches for relevant external data sources.
    Only runs if external_data_allowed=True.
    Writes external_data_manifest.json.
    Never blocks the pipeline — returns empty manifest on any failure.
    """
    if not state.get("external_data_allowed", False):
        logger.info("[competition_intel] External data not allowed — scout skipped.")
        empty_manifest = {"external_sources": [], "total_sources_found": 0}
        state = {**state, "external_data_manifest": empty_manifest}
        _write_manifest(state, empty_manifest)
        return state

    competition_brief = state.get("competition_brief", {})
    prompt = EXTERNAL_DATA_PROMPT.format(
        competition_name=state.get("competition_name", "unknown"),
        task_type=competition_brief.get("task_type", "unknown"),
        target_description=competition_brief.get("target_description", "unknown"),
        domain=competition_brief.get("domain", "unknown"),
        feature_summary=", ".join(state.get("feature_names", [])[:20]),
    )

    try:
        response = call_llm(prompt, "Return only valid JSON.", model="deepseek")
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found in LLM response")
        manifest = json.loads(response[start:end + 1])

        # Coerce relevance_score to float
        for source in manifest.get("external_sources", []):
            source["relevance_score"] = float(source.get("relevance_score", 0))

        _validate_manifest_schema(manifest)

        # Filter recommended_sources to only high-relevance (>= 0.6)
        source_names = {s["name"] for s in manifest.get("external_sources", [])
                        if s.get("relevance_score", 0) >= 0.6}
        manifest["recommended_sources"] = [
            name for name in manifest.get("recommended_sources", [])
            if name in source_names
        ]
        manifest["total_sources_found"] = len(manifest.get("external_sources", []))

    except Exception as e:
        logger.warning(f"[competition_intel] External data scout failed: {e}. Returning empty manifest.")
        manifest = {
            "external_sources": [],
            "total_sources_found": 0,
            "scout_error": str(e)[:200],
        }

    _write_manifest(state, manifest)
    state = {**state, "external_data_manifest": manifest}

    n = manifest.get("total_sources_found", 0)
    logger.info(f"[competition_intel] External data scout complete: {n} sources found.")
    return state


def _validate_manifest_schema(manifest: dict) -> None:
    """Raises ValueError if manifest structure is invalid."""
    required_keys = {"external_sources", "total_sources_found"}
    missing = required_keys - set(manifest.keys())
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")

    for source in manifest.get("external_sources", []):
        required_source_keys = {"name", "type", "relevance_score", "join_strategy", "acquisition_method"}
        missing_source = required_source_keys - set(source.keys())
        if missing_source:
            raise ValueError(f"Source '{source.get('name', '?')}' missing keys: {missing_source}")

        score = float(source.get("relevance_score", -1))
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"relevance_score must be 0.0-1.0. Got: {score}")


def _write_manifest(state: ProfessorState, manifest: dict) -> None:
    """Writes manifest to disk. Silent on failure."""
    try:
        path = Path(f"outputs/{state['session_id']}/external_data_manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
    except Exception as e:
        logger.warning(f"[competition_intel] Could not write manifest: {e}")
