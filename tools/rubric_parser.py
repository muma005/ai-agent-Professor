import os
import re
import json
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple, Union

from core.state import ProfessorState
from tools.llm_provider import llm_call
from tools.operator_channel import emit_to_operator

logger = logging.getLogger(__name__)

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RubricCriterion:
    """Single judging criterion extracted from the competition page."""
    name: str                           # e.g., "Clinical Relevance"
    weight: int                         # Points: e.g., 25
    max_points: int                     # Same as weight for most competitions
    description: str = ""               # What the judges look for
    top_score_description: str = ""     # What earns full marks
    bottom_score_description: str = ""  # What earns minimum
    scoring_levels: list = field(default_factory=list) # [{range: "21-25", description: "..."}, ...]


@dataclass
class HackathonRubric:
    """Complete parsed rubric from a hackathon competition page."""
    competition_name: str = "Unknown"
    total_points: int = 0                                       
    criteria: list = field(default_factory=list)                                          
    submission_requirements: list = field(default_factory=list)                           
    writeup_template: dict = field(default_factory=dict)                                  
    data_policy: str = "any_public"                                        
    recommended_datasets: list = field(default_factory=list)                              
    deliverable_type: str = "notebook_and_writeup"                                   
    tracks: list = field(default_factory=list)                                            
    prizes: list = field(default_factory=list)                                            
    deadline: str = "unknown"                                           
    raw_text_hash: str = ""                                      


@dataclass
class EffortPlan:
    """
    Maps rubric weights to concrete pipeline configuration.
    Every downstream hackathon agent reads this to calibrate its depth.
    """
    thesis_depth: str           # "deep" | "standard" | "light"
    technical_depth: str        # "marathon" | "standard" | "sprint"
    writeup_depth: str          # "deep" | "standard" | "light"
    external_data_priority: str # "high" | "medium" | "skip"
    visualization_count: int    # 3 | 5 | 7
    narrative_polish_passes: int # 1 | 2 | 3
    
    # Weight breakdown (for transparency — shown to operator)
    weight_breakdown: dict      # {"technical": 30, "novelty": 10, "documentation": 20, ...}
    allocation_reasoning: str   # Human-readable explanation of why each depth was chosen


# ── Function 1: parse_rubric() ──────────────────────────────────────────────

def parse_rubric(competition_text: str) -> HackathonRubric:
    """
    Extract structured rubric from raw competition page text.
    """
    if not competition_text or not competition_text.strip():
        return _build_rubric_from_merged(_default_rubric(""), "")

    # Pass 1: Deterministic
    partial = _deterministic_extract(competition_text)
    
    # Decide if LLM pass is needed
    needs_llm = (
        len(partial["criteria"]) < 3 or  # Not enough criteria found
        any(not c.get("description") for c in partial["criteria"]) or  # Missing descriptions
        not partial["submission_requirements"] or  # No requirements found
        not partial.get("max_writeup_words")  # No word limit found
    )
    
    if needs_llm:
        llm_result = _llm_extract(competition_text, partial)
        merged = _merge_results(partial, llm_result)
    else:
        merged = partial
    
    # Validate and build final rubric
    rubric = _build_rubric_from_merged(merged, competition_text)
    
    # Add text hash for cache validation
    rubric.raw_text_hash = hashlib.sha256(competition_text.encode()).hexdigest()[:16]
    
    return rubric


def _deterministic_extract(text: str) -> dict:
    """
    Try to extract rubric from structured text patterns.
    """
    result = {
        "criteria": [],
        "submission_requirements": [],
        "writeup_sections": [],
        "recommended_datasets": [],
        "max_writeup_words": None,
        "prizes": [],
    }
    
    # Pattern 1: Criteria extraction
    criteria_patterns = [
        # "Name (N points)" or "Name (N pts)"
        r'(?:^|\n)\s*\d*\.?\s*([A-Z][A-Za-z\s&/]+?)\s*\((\d+)\s*(?:points?|pts)\)',
        # "Name — N points" or "Name: N points"
        r'(?:^|\n)\s*\d*\.?\s*([A-Z][A-Za-z\s&/]+?)\s*[—:\-]\s*(\d+)\s*(?:points?|pts)',
        # Table format: "| Name | N |" or "Name\tN"
        r'(?:^|\n)\s*\|?\s*([A-Z][A-Za-z\s&/]+?)\s*\|?\s*(\d+)\s*\|?',
        # bullet format: "- Name: N points"
        r'(?:^|\n)\s*[\-\*]\s*([A-Z][A-Za-z\s&/]+?)\s*[—:\-]\s*(\d+)\s*(?:points?|pts)',
    ]
    
    for pattern in criteria_patterns:
        matches = re.findall(pattern, text)
        if len(matches) >= 3:
            result["criteria"] = [] # Clear if we found a better match
            for name, points in matches:
                name = name.strip().rstrip('.')
                try:
                    points = int(points)
                except ValueError:
                    continue
                if 5 <= points <= 50 and name.lower() not in ("total",):
                    result["criteria"].append({
                        "name": name,
                        "weight": points,
                        "max_points": points,
                    })
            if len(result["criteria"]) >= 3:
                break
    
    # Pattern 2: Submission requirements
    requirement_keywords = {
        "notebook": ["notebook", "kaggle notebook", "code notebook"],
        "writeup": ["writeup", "write-up", "project writeup", "report"],
        "cover_image": ["cover image", "cover photo", "thumbnail"],
        "project_link": ["project link", "github", "repository", "demo link", "working product", "portfolio link"],
    }
    text_lower = text.lower()
    for req_type, keywords in requirement_keywords.items():
        if any(kw in text_lower for kw in keywords):
            result["submission_requirements"].append(req_type)
    
    # Pattern 3: Word limit
    word_limit_match = re.search(r'(?:not exceed|under|maximum|max)\s*(\d{1,5})\s*words', text_lower)
    if word_limit_match:
        result["max_writeup_words"] = int(word_limit_match.group(1))
    
    # Pattern 4: Prizes
    prize_patterns = [
        r'(?:1st|first)\s*(?:place)?\s*[:\-—]?\s*\$?([\d,]+)',
        r'(?:2nd|second)\s*(?:place)?\s*[:\-—]?\s*\$?([\d,]+)',
        r'(?:3rd|third)\s*(?:place)?\s*[:\-—]?\s*\$?([\d,]+)',
    ]
    for i, pattern in enumerate(prize_patterns):
        match = re.search(pattern, text_lower)
        if match:
            amount = match.group(1).replace(',', '')
            result["prizes"].append({"place": f"{i+1}", "amount": f"${amount}"})
    
    # Pattern 5: Recommended datasets
    dataset_patterns = [
        r'(MIMIC[\w\-]*)',
        r'(NHAMCS)',
        r'(UCI\s+(?:ML\s+)?[Rr]epository)',
    ]
    for pattern in dataset_patterns:
        match = re.search(pattern, text)
        if match:
            # deduplicate
            if not any(d["name"] == match.group(1) for d in result["recommended_datasets"]):
                result["recommended_datasets"].append({
                    "name": match.group(1),
                    "url": "",
                    "description": "",
                })
    
    return result


def _llm_extract(competition_text: str, partial_result: dict) -> dict:
    """
    Use LLM to extract/complete rubric from competition text.
    """
    existing_criteria_str = ""
    if partial_result["criteria"]:
        existing_criteria_str = (
            f"\n\nI've already extracted these criteria deterministically:\n"
            f"{json.dumps(partial_result['criteria'], indent=2)}\n"
            f"Verify these are correct and add any missing descriptions."
        )
    
    prompt = f"""Read this hackathon competition page and extract the complete judging rubric.

COMPETITION TEXT:
---
{competition_text[:8000]}
---
{existing_criteria_str}

Extract as JSON. Include ALL of these fields. If a field can't be determined, use the default shown:

{{
    "competition_name": "<name of the competition>",
    "criteria": [
        {{
            "name": "<criterion name>",
            "weight": <points as integer>,
            "max_points": <same as weight>,
            "description": "<what judges look for — 1-2 sentences>",
            "top_score_description": "<what earns top marks — from rubric if available, else infer>",
            "bottom_score_description": "<what earns minimum — from rubric if available, else infer>",
            "scoring_levels": [
                {{"range": "21-25", "description": "..."}},
                {{"range": "14-20", "description": "..."}},
                {{"range": "7-13", "description": "..."}},
                {{"range": "0-6", "description": "..."}}
            ]
        }}
    ],
    "submission_requirements": ["notebook", "writeup"],
    "writeup_template": {{
        "sections": ["problem_statement", "methodology", "results", "limitations"],
        "max_words": 2000
    }},
    "data_policy": "any_public",
    "recommended_datasets": [
        {{"name": "Dataset Name", "url": "url if mentioned", "description": "brief description"}}
    ],
    "deliverable_type": "notebook_and_writeup",
    "tracks": [],
    "prizes": [{{"place": "1st", "amount": "$5,000"}}],
    "deadline": "unknown"
}}

RULES:
- If no explicit rubric exists, INFER criteria from the description. Common hackathon criteria:
  Technical Quality, Novelty/Innovation, Presentation/Documentation, Domain Relevance, Impact.
  Assign equal weights (20 pts each for 5 criteria = 100 total).
- If criteria exist but no descriptions, infer what the judges would look for.
- scoring_levels: extract from rubric if available. If not, create 4 reasonable levels.
- writeup_template sections: extract required sections from submission guidelines.
- data_policy: "any_public" if participants can use any data, "provided_only" if restricted.
- Respond with ONLY valid JSON. No markdown fences, no explanation.
"""
    
    response_text = llm_call(
        prompt=prompt,
        temperature=0.1,
        json_mode=True,
    )
    
    try:
        extracted = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        # Fallback to robust parsing logic
        from tools.llm_provider import _safe_json_loads
        try:
            extracted = _safe_json_loads(response_text)
        except Exception:
            extracted = _default_rubric(competition_text)
    
    return extracted


def _merge_results(deterministic: dict, llm: dict) -> dict:
    """
    Merge deterministic and LLM extraction results.
    Deterministic wins for weights and word limits.
    """
    merged = {}
    
    # Criteria
    if deterministic["criteria"]:
        det_names = {c["name"].lower().strip(): c for c in deterministic["criteria"]}
        llm_names = {c["name"].lower().strip(): c for c in llm.get("criteria", [])}
        
        merged_criteria = []
        for det_key, det_crit in det_names.items():
            llm_match = None
            for llm_key, llm_crit in llm_names.items():
                if det_key in llm_key or llm_key in det_key:
                    llm_match = llm_crit
                    break
            
            if llm_match:
                merged_criteria.append({
                    **llm_match,
                    "weight": det_crit["weight"],
                    "max_points": det_crit["weight"],
                })
            else:
                merged_criteria.append(det_crit)
        
        for llm_key, llm_crit in llm_names.items():
            if not any(llm_key in dk or dk in llm_key for dk in det_names):
                merged_criteria.append(llm_crit)
        
        merged["criteria"] = merged_criteria
    else:
        merged["criteria"] = llm.get("criteria", [])
    
    # Other fields
    merged["submission_requirements"] = list(set(
        deterministic["submission_requirements"] + 
        llm.get("submission_requirements", ["notebook", "writeup"])
    ))
    
    # Word limit from deterministic takes precedence
    max_words = deterministic.get("max_writeup_words") or llm.get("writeup_template", {}).get("max_words", 2000)
    
    merged["recommended_datasets"] = (
        deterministic.get("recommended_datasets") or
        llm.get("recommended_datasets", [])
    )
    merged["prizes"] = deterministic.get("prizes") or llm.get("prizes", [])
    
    merged["competition_name"] = llm.get("competition_name", "Unknown Competition")
    merged["writeup_template"] = llm.get("writeup_template", {"sections": [], "max_words": max_words})
    merged["writeup_template"]["max_words"] = max_words # Ensure word limit is merged
    
    merged["data_policy"] = llm.get("data_policy", "any_public")
    merged["deliverable_type"] = llm.get("deliverable_type", "notebook_and_writeup")
    merged["tracks"] = llm.get("tracks", [])
    merged["deadline"] = llm.get("deadline", "unknown")
    
    return merged


def _default_rubric(competition_text: str) -> dict:
    """Fallback rubric."""
    return {
        "competition_name": "Unknown Competition",
        "criteria": [
            {"name": "Technical Quality", "weight": 20, "max_points": 20,
             "description": "Sound methodology, clean code, appropriate approach",
             "top_score_description": "Rigorous and appropriate methodology",
             "bottom_score_description": "Flawed methodology or broken code",
             "scoring_levels": []},
            {"name": "Novelty & Innovation", "weight": 20, "max_points": 20,
             "description": "Fresh perspective, underexplored angle, creative approach",
             "top_score_description": "Genuinely novel approach",
             "bottom_score_description": "Entirely derivative",
             "scoring_levels": []},
            {"name": "Documentation & Presentation", "weight": 20, "max_points": 20,
             "description": "Clear writeup, well-structured, reproducible",
             "top_score_description": "Thorough and clear documentation",
             "bottom_score_description": "Missing or unclear documentation",
             "scoring_levels": []},
            {"name": "Domain Relevance", "weight": 20, "max_points": 20,
             "description": "Addresses a real problem in the domain",
             "top_score_description": "Sharply defined, domain-grounded problem",
             "bottom_score_description": "Negligible domain relevance",
             "scoring_levels": []},
            {"name": "Insight & Impact", "weight": 20, "max_points": 20,
             "description": "Meaningful findings, practical implications",
             "top_score_description": "Actionable findings with clear impact",
             "bottom_score_description": "No meaningful findings",
             "scoring_levels": []},
        ],
        "submission_requirements": ["notebook", "writeup"],
        "writeup_template": {"sections": ["problem_statement", "methodology", "results", "limitations"], "max_words": 2000},
        "data_policy": "any_public",
        "recommended_datasets": [],
        "deliverable_type": "notebook_and_writeup",
        "tracks": [],
        "prizes": [],
        "deadline": "unknown",
    }


def _build_rubric_from_merged(merged: dict, original_text: str) -> HackathonRubric:
    """Validate and construct final HackathonRubric."""
    criteria = merged.get("criteria", [])
    
    # 1. Deduplicate criteria before any validation
    seen_names = set()
    deduped = []
    for c in criteria:
        name_key = c.get("name", "").lower().strip()
        if name_key and name_key not in seen_names:
            seen_names.add(name_key)
            deduped.append(c)
    criteria = deduped

    # 2. Validation: Every criterion must have a name and positive weight
    criteria = [c for c in criteria if c.get("name") and c.get("weight", 0) > 0]
    
    # 3. Total points validation and normalization
    total = sum(c.get("weight", 0) for c in criteria)
    
    # If points are provided but not close to 100, normalize to 100
    if total > 0 and (total < 95 or total > 105):
        for c in criteria:
            c["weight"] = int(round(c.get("weight", 0) * 100 / total))
            c["max_points"] = c["weight"]
        total = sum(c["weight"] for c in criteria)
    
    # Final fallback if still 0
    if total == 0:
        default = _default_rubric(original_text)
        return HackathonRubric(
            competition_name=merged.get("competition_name", default["competition_name"]),
            total_points=100,
            criteria=default["criteria"],
            submission_requirements=merged.get("submission_requirements", default["submission_requirements"]),
            writeup_template=merged.get("writeup_template", default["writeup_template"]),
            data_policy=merged.get("data_policy", default["data_policy"]),
            recommended_datasets=merged.get("recommended_datasets", default["recommended_datasets"]),
            deliverable_type=merged.get("deliverable_type", default["deliverable_type"]),
            tracks=merged.get("tracks", default["tracks"]),
            prizes=merged.get("prizes", default["prizes"]),
            deadline=merged.get("deadline", default["deadline"]),
            raw_text_hash="",
        )
    
    for c in criteria:
        c.setdefault("description", "")
        c.setdefault("top_score_description", "")
        c.setdefault("bottom_score_description", "")
        c.setdefault("scoring_levels", [])
        c.setdefault("max_points", c["weight"])
    
    return HackathonRubric(
        competition_name=merged.get("competition_name", "Unknown"),
        total_points=total,
        criteria=criteria,
        submission_requirements=merged.get("submission_requirements", ["notebook", "writeup"]),
        writeup_template=merged.get("writeup_template", {"sections": [], "max_words": 2000}),
        data_policy=merged.get("data_policy", "any_public"),
        recommended_datasets=merged.get("recommended_datasets", []),
        deliverable_type=merged.get("deliverable_type", "notebook_and_writeup"),
        tracks=merged.get("tracks", []),
        prizes=merged.get("prizes", []),
        deadline=merged.get("deadline", "unknown"),
        raw_text_hash="",
    )


# ── Function 2: build_effort_plan() ─────────────────────────────────────────

def build_effort_plan(rubric: HackathonRubric) -> EffortPlan:
    """Map rubric criteria weights to concrete pipeline configuration."""
    DIMENSION_KEYWORDS = {
        "technical": ["technical", "methodology", "code", "model", "approach", "reproducib", "implementation", "execution"],
        "novelty": ["novel", "impact", "creative", "original", "innovative", "fresh", "unique", "originality"],
        "documentation": ["document", "writeup", "write-up", "clarity", "presentation", "writing", "report", "quality"],
        "domain": ["clinical", "domain", "relevance", "problem", "real-world", "practical", "motivation"],
        "insight": ["insight", "finding", "result", "analysis", "discover", "interpret", "meaningful"],
    }
    
    dimension_weights = {dim: 0.0 for dim in DIMENSION_KEYWORDS}
    unclassified_weight = 0.0
    
    for criterion in rubric.criteria:
        name_lower = criterion["name"].lower()
        desc_lower = (criterion.get("description", "") or "").lower()
        search_text = name_lower + " " + desc_lower
        
        matched = False
        for dimension, keywords in DIMENSION_KEYWORDS.items():
            if any(kw in search_text for kw in keywords):
                dimension_weights[dimension] += float(criterion["weight"])
                matched = True
                break
        
        if not matched:
            unclassified_weight += float(criterion["weight"])
    
    if unclassified_weight > 0:
        per_dim = unclassified_weight / len(DIMENSION_KEYWORDS)
        for dim in dimension_weights:
            dimension_weights[dim] += per_dim
    
    total_weight = sum(dimension_weights.values())
    if total_weight == 0:
        total_weight = 100.0
    
    tech_pct = dimension_weights["technical"] / total_weight
    novelty_pct = dimension_weights["novelty"] / total_weight
    doc_pct = dimension_weights["documentation"] / total_weight
    domain_pct = dimension_weights["domain"] / total_weight
    insight_pct = dimension_weights["insight"] / total_weight
    
    if tech_pct > 0.28:
        technical_depth = "marathon"
    elif tech_pct > 0.15:
        technical_depth = "standard"
    else:
        technical_depth = "sprint"
    
    thesis_input = domain_pct + novelty_pct
    if thesis_input > 0.30:
        thesis_depth = "deep"
    elif thesis_input > 0.15:
        thesis_depth = "standard"
    else:
        thesis_depth = "light"
    
    # 20% weight should be "deep" according to test
    if doc_pct > 0.18:
        writeup_depth = "deep"
    elif doc_pct > 0.08:
        writeup_depth = "standard"
    else:
        writeup_depth = "light"
    
    data_input = novelty_pct + insight_pct
    if data_input > 0.20:
        external_data_priority = "high"
    elif data_input > 0.10:
        external_data_priority = "medium"
    else:
        external_data_priority = "skip"
    
    if insight_pct > 0.14:
        visualization_count = 7
    elif insight_pct > 0.08:
        visualization_count = 5
    else:
        visualization_count = 3
    
    if doc_pct > 0.18:
        narrative_polish_passes = 3
    elif doc_pct > 0.08:
        narrative_polish_passes = 2
    else:
        narrative_polish_passes = 1
    
    reasoning_parts = []
    reasoning_parts.append(f"Technical weight {tech_pct:.0%} → {technical_depth} pipeline depth")
    reasoning_parts.append(f"Domain+Novelty weight {thesis_input:.0%} → {thesis_depth} thesis generation")
    reasoning_parts.append(f"Documentation weight {doc_pct:.0%} → {writeup_depth} writeup ({narrative_polish_passes} polish passes)")
    reasoning_parts.append(f"Novelty+Insight weight {data_input:.0%} → {external_data_priority} external data priority")
    reasoning_parts.append(f"Insight weight {insight_pct:.0%} → {visualization_count} narrative visualizations")
    
    return EffortPlan(
        thesis_depth=thesis_depth,
        technical_depth=technical_depth,
        writeup_depth=writeup_depth,
        external_data_priority=external_data_priority,
        visualization_count=visualization_count,
        narrative_polish_passes=narrative_polish_passes,
        weight_breakdown={
            "technical": round(dimension_weights["technical"], 1),
            "novelty": round(dimension_weights["novelty"], 1),
            "documentation": round(dimension_weights["documentation"], 1),
            "domain": round(dimension_weights["domain"], 1),
            "insight": round(dimension_weights["insight"], 1),
        },
        allocation_reasoning="\n".join(reasoning_parts),
    )


# ── Function 3: run_rubric_parser() ──────────────────────────────────────────

def run_rubric_parser(state: ProfessorState) -> dict:
    """LangGraph node function."""
    emit_to_operator("📋 Parsing hackathon rubric...", level="STATUS")
    
    # Check both potential field names for compatibility
    competition_text = state.get("competition_description") or state.get("competition_brief", {}).get("text")
    
    if not competition_text or len(competition_text) < 100:
        emit_to_operator(
            "📋 Competition description is too short for rubric parsing. "
            "Please paste the full competition page text (including rubric and submission requirements).",
            level="GATE",
        )
        injections = [inj["text"] for inj in (state.hitl_injections or []) 
                      if len(inj["text"]) > 200]
        if injections:
            competition_text = injections[-1]
        else:
            emit_to_operator("⚠️ No competition text available. Using default rubric.", level="STATUS")
            rubric = HackathonRubric(**_default_rubric(""), raw_text_hash="")
            effort = build_effort_plan(rubric)
            return _build_state_return(rubric, effort)
    
    rubric = parse_rubric(competition_text)
    effort = build_effort_plan(rubric)
    
    criteria_display = "\n".join([
        f"  {c['name']}: {c['weight']} pts ({c['weight']*100//max(rubric.total_points, 1)}%)"
        f" → {_effort_for_criterion(c, effort)}"
        for c in rubric.criteria
    ])
    
    requirements_display = ", ".join(rubric.submission_requirements)
    datasets_display = ", ".join([d["name"] for d in rubric.recommended_datasets]) or "none specified"
    writeup_sections = ", ".join(rubric.writeup_template.get("sections", [])) or "not specified"
    
    message = f"""📋 RUBRIC ANALYSIS

Competition: {rubric.competition_name}
Total points: {rubric.total_points}

Criteria breakdown:
{criteria_display}

Effort allocation:
  Pipeline depth: {effort.technical_depth.upper()}
  Thesis depth: {effort.thesis_depth.upper()}
  Writeup polish: {effort.writeup_depth.upper()} ({effort.narrative_polish_passes} passes)
  External data: {effort.external_data_priority.upper()} priority
  Narrative visualizations: {effort.visualization_count}

Required deliverables: {requirements_display}
Writeup sections: {writeup_sections}
Word limit: {rubric.writeup_template.get('max_words', 'not specified')}
Data policy: {rubric.data_policy}
Recommended datasets: {datasets_display}
Prizes: {', '.join(p.get('amount', '?') for p in rubric.prizes) or 'not specified'}

Effort plan looks correct? Reply /continue or describe adjustments."""
    
    response = emit_to_operator(message, level="CHECKPOINT")
    
    if response and response.strip().lower() != "/continue":
        effort = _apply_operator_adjustments(effort, response)
        emit_to_operator(f"📋 Effort plan adjusted: {response[:100]}", level="STATUS")
    
    return _build_state_return(rubric, effort)


def _effort_for_criterion(criterion: dict, effort: EffortPlan) -> str:
    """Map a criterion to what effort dimension it drives."""
    name_lower = criterion["name"].lower()
    if any(kw in name_lower for kw in ["technical", "method", "code", "model"]):
        return f"{effort.technical_depth} pipeline"
    elif any(kw in name_lower for kw in ["novel", "impact", "creative", "originality"]):
        return f"{effort.thesis_depth} thesis + {effort.external_data_priority} data search"
    elif any(kw in name_lower for kw in ["document", "writeup", "clarity", "presentation"]):
        return f"{effort.writeup_depth} writeup ({effort.narrative_polish_passes} passes)"
    elif any(kw in name_lower for kw in ["clinical", "domain", "relevance"]):
        return f"{effort.thesis_depth} thesis depth"
    elif any(kw in name_lower for kw in ["insight", "finding", "result"]):
        return f"{effort.visualization_count} visualizations"
    return "general effort"


def _apply_operator_adjustments(effort: EffortPlan, response: str) -> EffortPlan:
    """Simple keyword matching for operator adjustments."""
    response_lower = response.lower()
    if "marathon" in response_lower:
        effort.technical_depth = "marathon"
    elif "sprint" in response_lower:
        effort.technical_depth = "sprint"
    
    if "more visual" in response_lower or "more plot" in response_lower:
        effort.visualization_count = min(effort.visualization_count + 2, 10)
    
    if "more polish" in response_lower:
        effort.narrative_polish_passes = min(effort.narrative_polish_passes + 1, 5)
    
    if "skip external" in response_lower or "no external" in response_lower:
        effort.external_data_priority = "skip"
    
    return effort


def _build_state_return(rubric: HackathonRubric, effort: EffortPlan) -> dict:
    """Build the state update dict."""
    return {
        "hackathon_rubric": asdict(rubric),
        "hackathon_effort_plan": asdict(effort),
        "hackathon_mode": True,
        "hackathon_writeup_template": rubric.writeup_template,
    }
