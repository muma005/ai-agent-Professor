import json
import re
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple, Union

from core.state import ProfessorState
from tools.llm_provider import llm_call
from tools.operator_channel import emit_to_operator

logger = logging.getLogger(__name__)

AGENT_NAME = "thesis_generator"

@dataclass
class ThesisCandidate:
    """A single thesis proposal for the hackathon."""
    thesis_id: int                      # 1-5
    statement: str                      # One-sentence thesis: "ESI undertriages elderly patients..."
    angle: str                          # What makes this different from the obvious approach
    target_audience: str                # Who uses this finding: "ED nurses", "triage algorithm designers"
    
    data_plan: dict                     # {
                                        #   "primary_dataset": str,
                                        #   "external_needed": list[str],
                                        #   "join_strategy": str
                                        # }
    
    condition_variable: str             # The variable creating the conditional split: "age_group × presentation_type"
    hypothesis: str                     # Testable prediction: "Patients >65 with atypical cardiac are undertriaged 2x"
    
    estimated_scores: dict              # {criterion_name: {"score": int, "justification": str}}
    estimated_total: int                # Sum of estimated scores
    
    feasibility: str                    # "high" | "medium" | "low"
    risk: str                           # "MIMIC-IV-ED may not have enough atypical cardiac cases"
    
    # Enriched by the generator (not from LLM directly)
    rubric_alignment_score: float = 0.0 # How well this thesis aligns with the TOP-weighted criterion (0-1)
    condition_type: str = "categorical" # "categorical" | "temporal" | "spatial" | "threshold" | "interaction"


def _generate_thesis_candidates(state: ProfessorState) -> List[ThesisCandidate]:
    """
    Generate 5 thesis candidates using domain research + EDA + rubric context.
    """
    # 1. Rubric context
    rubric = state.get("hackathon_rubric") or {}
    criteria = rubric.get("criteria", [])
    
    top_criterion = max(criteria, key=lambda c: c.get("weight", 0)) if criteria else None
    top_criterion_name = top_criterion["name"] if top_criterion else "Technical Quality"
    top_criterion_weight = top_criterion.get("weight", 20) if top_criterion else 20
    top_criterion_top_desc = top_criterion.get("top_score_description", "") if top_criterion else ""
    
    criteria_block = ""
    for c in criteria:
        criteria_block += (
            f"  - {c['name']} ({c['weight']} pts): {c.get('description', '')}\n"
            f"    TOP SCORE ({c.get('scoring_levels', [{}])[0].get('range', 'max')}): "
            f"{c.get('top_score_description', 'Not specified')}\n"
        )
    
    # 2. Domain context
    domain_brief = state.get("domain_brief") or {}
    domain_classification = state.get("domain_classification") or "general"
    column_semantics = domain_brief.get("column_semantics", {})
    known_relationships = domain_brief.get("known_relationships", [])
    domain_summary = domain_brief.get("domain_summary", "")
    
    # 3. EDA context
    eda_summary = state.get("eda_insights_summary") or ""
    mi_data = state.get("eda_mutual_info") or {}
    modality_flags = state.get("eda_modality_flags") or []
    schema = state.get("data_schema") or {}
    
    schema_block = "\n".join([f"  {col}: {dtype}" for col, dtype in list(schema.items())[:40]])
    if len(schema) > 40:
        schema_block += f"\n  ... ({len(schema) - 40} more columns)"
    
    # 4. Competition context
    competition_desc = state.get("competition_description") or ""
    recommended_datasets = rubric.get("recommended_datasets", [])
    datasets_block = "\n".join([
        f"  - {d.get('name', '?')}: {d.get('description', 'No description')}"
        for d in recommended_datasets
    ]) or "  None specified — participant chooses data"
    
    # 5. Operator injections
    operator_context = ""
    for inj in (state.get("hitl_injections") or []):
        if isinstance(inj, dict):
            operator_context += f"  - {inj.get('text', '')}\n"
    
    # 6. Top MI features
    mi_top = mi_data.get("target_mi", [])
    mi_block = ""
    if mi_top:
        top_5_mi = sorted(mi_top, key=lambda x: list(x.values())[0] if isinstance(x, dict) else 0, reverse=True)[:5]
        mi_block = "Top 5 features by mutual information with target:\n"
        for item in top_5_mi:
            if isinstance(item, dict):
                for col, score in item.items():
                    mi_block += f"  {col}: MI={score:.3f}\n"
    
    prompt = f"""You are simultaneously:
1. A senior practitioner in {domain_classification} with 15 years of experience
2. A data scientist who consistently wins Kaggle hackathons
3. A judge evaluating submissions against the rubric below

You must generate thesis candidates that score HIGH — not generic analyses that everyone else will do.

=== COMPETITION ===
{competition_desc[:3000]}

=== JUDGING RUBRIC (what the judges ACTUALLY score on) ===
{criteria_block}

THE TOP-WEIGHTED CRITERION IS: "{top_criterion_name}" ({top_criterion_weight} pts)
To score 21-25 on this criterion: {top_criterion_top_desc}
Your thesis MUST score maximum on this criterion. Everything else is secondary.

=== DATA SCHEMA ===
{schema_block}

=== EDA INSIGHTS ===
{eda_summary}

{mi_block}

Multimodal features (bimodal distributions): {modality_flags if modality_flags else 'None detected'}

=== DOMAIN KNOWLEDGE ===
Domain: {domain_classification}
{domain_summary}

Known relationships:
{json.dumps(known_relationships[:10], indent=2) if known_relationships else '  None identified'}

Column semantics:
{json.dumps(dict(list(column_semantics.items())[:15]), indent=2) if column_semantics else '  Not available'}

=== RECOMMENDED/AVAILABLE DATASETS ===
{datasets_block}

=== OPERATOR CONTEXT ===
{operator_context if operator_context else 'No additional context from operator.'}

=== YOUR TASK ===

Generate EXACTLY 5 thesis candidates. Each thesis must:

1. BE SPECIFIC — not "predict triage severity" but "ESI undertriages elderly patients presenting 
   with atypical cardiac symptoms because vital sign thresholds are calibrated for typical presentations"

2. BE CONDITIONAL — express as "X behaves differently under condition Y" or "Z is systematically 
   biased for population W." The condition_variable is the split that creates the comparison.

3. BE TESTABLE — you must be able to compute the conditional effect from available data.

4. MATTER TO PRACTITIONERS — target specific people who would change behavior.

5. SCORE HIGH ON "{top_criterion_name}" — this is {top_criterion_weight}% of the total score.

For each thesis, estimate scores on EVERY criterion in the rubric.

Respond with ONLY valid JSON (no markdown fences, no explanation before or after):
[
    {{
        "thesis_id": 1,
        "statement": "One-sentence thesis statement — specific, conditional, testable",
        "angle": "What makes this different from what 90% of teams will do",
        "target_audience": "Specific practitioners who would use this finding",
        "data_plan": {{
            "primary_dataset": "Which dataset to use as the foundation",
            "external_needed": ["Specific external data that strengthens the thesis"],
            "join_strategy": "How to combine primary + external data"
        }},
        "condition_variable": "The specific variable(s) creating the conditional comparison",
        "hypothesis": "Specific testable prediction with expected direction and magnitude",
        "estimated_scores": {{
            {', '.join(f'"{c["name"]}": {{"score": <int>, "justification": "<why this score>"}}' for c in criteria)}
        }},
        "estimated_total": <sum of scores>,
        "feasibility": "high" | "medium" | "low",
        "risk": "What could go wrong"
    }},
    ... (4 more)
]

RANK by estimated_total descending (highest first).
"""

    response_text = llm_call(
        prompt=prompt,
        temperature=0.8,
        max_tokens=6000,
    )
    
    return _parse_thesis_response(response_text, criteria)


def _parse_thesis_response(response_text: str, rubric_criteria: list) -> List[ThesisCandidate]:
    """Parse and validate LLM response."""
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    try:
        raw_list = json.loads(text)
    except json.JSONDecodeError:
        raw_list = _extract_json_objects(text)
        if not raw_list:
            return _fallback_candidates(rubric_criteria)
    
    if not isinstance(raw_list, list):
        raw_list = [raw_list]
    
    candidates = []
    criterion_names = {c["name"] for c in rubric_criteria}
    for i, raw in enumerate(raw_list[:5]):
        candidates.append(_validate_candidate(raw, i + 1, criterion_names, rubric_criteria))
    
    while len(candidates) < 5:
        candidates.append(_make_placeholder_candidate(len(candidates) + 1, rubric_criteria))
    
    candidates.sort(key=lambda c: c.estimated_total, reverse=True)
    for i, c in enumerate(candidates):
        c.thesis_id = i + 1
    return candidates


def _validate_candidate(raw: dict, thesis_id: int, criterion_names: set, rubric_criteria: list) -> ThesisCandidate:
    """Validate single candidate and fill defaults."""
    statement = raw.get("statement", f"Thesis {thesis_id} — statement not generated")
    angle = raw.get("angle", "Not specified")
    target_audience = raw.get("target_audience", "Data scientists and domain practitioners")
    
    data_plan = raw.get("data_plan", {})
    if not isinstance(data_plan, dict):
        data_plan = {"primary_dataset": str(data_plan), "external_needed": [], "join_strategy": ""}
    data_plan.setdefault("primary_dataset", "Competition data")
    data_plan.setdefault("external_needed", [])
    data_plan.setdefault("join_strategy", "")
    
    condition_variable = raw.get("condition_variable", "Not specified")
    hypothesis = raw.get("hypothesis", "Not specified")
    feasibility = raw.get("feasibility", "medium")
    if feasibility not in ("high", "medium", "low"):
        feasibility = "medium"
    risk = raw.get("risk", "No risk assessment")
    
    estimated_scores = raw.get("estimated_scores", {})
    validated_scores = {}
    total = 0
    for criterion in rubric_criteria:
        cname = criterion["name"]
        max_pts = criterion.get("max_points", criterion.get("weight", 20))
        entry = estimated_scores.get(cname, {})
        if isinstance(entry, dict):
            score = entry.get("score", max_pts // 2)
            justification = entry.get("justification", "")
        else:
            score = int(entry) if isinstance(entry, (int, float)) else max_pts // 2
            justification = ""
        score = max(0, min(int(score), max_pts))
        validated_scores[cname] = {"score": score, "justification": justification}
        total += score
        
    condition_type = _classify_condition(condition_variable)
    top_criterion = max(rubric_criteria, key=lambda c: c.get("weight", 0)) if rubric_criteria else None
    rubric_alignment = 0.0
    if top_criterion:
        top_name = top_criterion["name"]
        top_max = top_criterion.get("max_points", top_criterion.get("weight", 20))
        rubric_alignment = validated_scores.get(top_name, {}).get("score", 0) / max(top_max, 1)

    return ThesisCandidate(
        thesis_id=thesis_id, statement=statement, angle=angle, target_audience=target_audience,
        data_plan=data_plan, condition_variable=condition_variable, hypothesis=hypothesis,
        estimated_scores=validated_scores, estimated_total=total, feasibility=feasibility,
        risk=risk, rubric_alignment_score=round(rubric_alignment, 2), condition_type=condition_type
    )


def _classify_condition(condition_text: str) -> str:
    """Classify condition type."""
    t = condition_text.lower()
    if any(kw in t for kw in ["time", "hour", "day", "night", "season", "temporal"]): return "temporal"
    if any(kw in t for kw in ["age", "group", "category", "gender", "race", "type"]): return "categorical"
    if any(kw in t for kw in ["region", "location", "geographic", "site"]): return "spatial"
    if any(kw in t for kw in ["threshold", "above", "below", "greater", "less", "cutoff"]): return "threshold"
    if any(kw in t for kw in ["×", "x ", "interaction", " and "]): return "interaction"
    return "categorical"


def _extract_json_objects(text: str) -> list:
    """Extract JSON from malformed text."""
    objs = []
    depth, start = 0, None
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0: start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try: objs.append(json.loads(text[start:i+1]))
                except: pass
                start = None
    return objs


def _fallback_candidates(rubric_criteria: list) -> List[ThesisCandidate]:
    """Fallback list."""
    return [_make_placeholder_candidate(i+1, rubric_criteria) for i in range(5)]


def _make_placeholder_candidate(thesis_id: int, rubric_criteria: list) -> ThesisCandidate:
    """Minimal placeholder."""
    scores = {c["name"]: {"score": c.get("weight", 20) // 3, "justification": "Fallback"} for c in rubric_criteria}
    return ThesisCandidate(
        thesis_id=thesis_id, statement=f"Placeholder thesis {thesis_id} — Please provide your own via HITL.",
        angle="N/A", target_audience="N/A", data_plan={"primary_dataset": "TBD", "external_needed": [], "join_strategy": ""},
        condition_variable="TBD", hypothesis="TBD", estimated_scores=scores, 
        estimated_total=sum(s["score"] for s in scores.values()), feasibility="low", risk="N/A"
    )


def _evaluate_custom_thesis(thesis_text: str, rubric: dict, state: ProfessorState) -> ThesisCandidate:
    """Evaluate operator-provided thesis."""
    criteria = rubric.get("criteria", [])
    criteria_block = "\n".join([f"  - {c['name']} ({c['weight']} pts): {c.get('description', '')}" for c in criteria])
    prompt = f"""Evaluate this thesis for a hackathon.
THESIS: "{thesis_text}"
RUBRIC:
{criteria_block}
DOMAIN: {state.get('domain_classification', 'general')}
Respond with ONLY valid JSON:
{{
    "statement": "{thesis_text}",
    "angle": "Analysis angle",
    "target_audience": "Stakeholders",
    "data_plan": {{"primary_dataset": "Foundation data", "external_needed": [], "join_strategy": ""}},
    "condition_variable": "Implied condition",
    "hypothesis": "Testable prediction",
    "estimated_scores": {{{', '.join(f'"{c["name"]}": {{"score": <int>, "justification": "..."}}' for c in criteria)}}},
    "estimated_total": <sum>,
    "feasibility": "high" | "medium" | "low",
    "risk": "Potential pitfall"
}}
"""
    res_text = llm_call(prompt=prompt, temperature=0.2)
    raw = _safe_json_parse(res_text)
    cand = _validate_candidate(raw, 0, {c["name"] for c in criteria}, criteria)
    cand.thesis_id = 0
    return cand


def _safe_json_parse(text: str) -> dict:
    """Clean and parse JSON."""
    t = text.strip()
    if "```json" in t: t = t.split("```json")[1].split("```")[0]
    elif "```" in t: t = t.split("```")[1].split("```")[0]
    t = re.sub(r',\s*}', '}', t.strip())
    t = re.sub(r',\s*]', ']', t)
    try: return json.loads(t)
    except: return {}


def _present_theses_to_operator(candidates: List[ThesisCandidate], state: ProfessorState) -> str:
    """HITL GATE presentation."""
    rubric = state.get("hackathon_rubric") or {}
    criteria = rubric.get("criteria", [])
    thesis_display = ""
    for c in candidates:
        score_lines = ""
        for criterion in criteria:
            name = criterion["name"]
            entry = c.estimated_scores.get(name, {})
            score = entry.get("score", "?")
            max_pts = criterion.get("max_points", criterion.get("weight", 20))
            score_lines += f"   {name}: {score}/{max_pts} — {entry.get('justification', '')[:77]}...\n"
        
        icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(c.feasibility, "⚪")
        thesis_display += f"""
{c.thesis_id}. [{icon} EST: {c.estimated_total}/{rubric.get('total_points', 100)}] "{c.statement}"
   Angle: {c.angle}
   Condition: {c.condition_variable} ({c.condition_type})
   Hypothesis: {c.hypothesis}
   Score breakdown:
{score_lines}
"""
    message = f"""🔬 THESIS PROPOSALS
{thesis_display}
---
Select thesis (1-5), or reply:
  /thesis select N     — select thesis N
  /thesis custom "..." — provide your own
  /continue            — auto-select #1"""
    return emit_to_operator(message, level="GATE")


def _parse_operator_selection(response: str, candidates: List[ThesisCandidate], rubric: dict, state: ProfessorState) -> dict:
    """Parse operator response."""
    if not response or response.strip() == "/continue":
        return {"active_thesis": asdict(candidates[0]), "thesis_selected_by": "auto"}
    
    r = response.strip()
    if r.isdigit() and 1 <= int(r) <= len(candidates):
        return {"active_thesis": asdict(candidates[int(r)-1]), "thesis_selected_by": "operator"}
    
    if r.startswith("/thesis select"):
        parts = r.split()
        if len(parts) >= 3 and parts[2].isdigit():
            idx = int(parts[2])
            if 1 <= idx <= len(candidates):
                return {"active_thesis": asdict(candidates[idx-1]), "thesis_selected_by": "operator"}
                
    if r.startswith("/thesis custom") or len(r) > 20:
        text = r.replace("/thesis custom", "").strip().strip('"').strip("'")
        emit_to_operator(f"🔬 Evaluating custom thesis: \"{text[:80]}...\"", level="STATUS")
        cand = _evaluate_custom_thesis(text, rubric, state)
        return {"active_thesis": asdict(cand), "thesis_selected_by": "operator"}
        
    return {"active_thesis": asdict(candidates[0]), "thesis_selected_by": "auto"}


def thesis_generator(state: ProfessorState) -> dict:
    """LangGraph node for thesis generation."""
    if not state.get("hackathon_mode"): return {}
    emit_to_operator("🔬 Generating thesis candidates...", level="STATUS")
    
    effort = state.get("hackathon_effort_plan") or {}
    candidates = _generate_thesis_candidates(state)
    
    # Deep mode not fully implemented (requires diverse batch logic), using top 5 for now
    
    response = _present_theses_to_operator(candidates, state)
    rubric = state.get("hackathon_rubric") or {}
    selection = _parse_operator_selection(response, candidates, rubric, state)
    
    active = selection["active_thesis"]
    emit_to_operator(f"✅ Active thesis ({selection['thesis_selected_by']}): \"{active.get('statement', '?')[:100]}\"", level="STATUS")
    
    return {
        "thesis_candidates": [asdict(c) for c in candidates],
        "active_thesis": active,
        "thesis_selected_by": selection["thesis_selected_by"]
    }
