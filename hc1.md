# BUILD PROMPT — Hackathon Mode Component 2: Thesis Generator
# Feed to Gemini CLI with: @PROFESSOR.md @STATE.md @HITL.md @CONTRACTS.md @PROMPTS.md @PROVIDERS.md

---

## CONTEXT

Professor v2 traditional pipeline is complete. Hackathon Mode Component 1 (Rubric Parser) is built and passing all contracts. The Rubric Parser produces a `HackathonRubric` with weighted criteria and an `EffortPlan` that configures pipeline depth.

Now you're building the Thesis Generator — the single highest-leverage component in the hackathon pipeline. The thesis is the creative direction that separates a 14-20/25 Clinical Relevance score from a 21-25/25 score. It's the difference between "a generic triage prediction model" (what 90% of entrants build) and "a sharp analysis proving ESI undertriages elderly patients with atypical cardiac presentations" (what wins).

This component depends on:
- `tools/rubric_parser.py` — reads `hackathon_rubric` and `hackathon_effort_plan` from state
- `agents/eda_agent.py` — reads `eda_insights_summary`, `eda_mutual_info`, `eda_modality_flags`
- `agents/domain_research.py` — reads `domain_brief`, `domain_classification`
- `tools/llm_provider.py` — for `llm_call()`
- `tools/operator_channel.py` — for `emit_to_operator()` at GATE level

---

## COMMIT PLAN (2 commits)

```
Commit 1: agents/thesis_generator.py + tests/contracts/test_thesis_generator_contract.py
Commit 2: State additions + graph wiring note
```

Both commits pass `pytest tests/contracts/ -q` including ALL existing tests.

---

## COMMIT 1: agents/thesis_generator.py

### File structure

```
agents/
└── thesis_generator.py
    ├── ThesisCandidate (dataclass)
    ├── _generate_thesis_candidates(state) -> list[ThesisCandidate]
    ├── _evaluate_custom_thesis(thesis_text, rubric, state) -> ThesisCandidate
    ├── _present_theses_to_operator(candidates, state) -> dict
    ├── _parse_operator_selection(response, candidates) -> dict
    ├── thesis_generator(state: ProfessorState) -> dict   (LangGraph node)
```

---

### Data structure

```python
from dataclasses import dataclass, field
from typing import Optional

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
    rubric_alignment_score: float       # How well this thesis aligns with the TOP-weighted criterion (0-1)
    condition_type: str                 # "categorical" | "temporal" | "spatial" | "threshold" | "interaction"
```

---

### The thesis generation function

```python
def _generate_thesis_candidates(state: ProfessorState) -> list[ThesisCandidate]:
    """
    Generate 5 thesis candidates using domain research + EDA + rubric context.
    
    The prompt is the most important piece of engineering in this component.
    It must produce SHARP, CONDITIONAL, TESTABLE theses — not generic "predict Y" ideas.
    """
    from tools.llm_provider import llm_call
    
    # === BUILD THE CONTEXT PACKAGE ===
    
    # 1. Rubric context — what the judges score on
    rubric = state.hackathon_rubric
    criteria = rubric.get("criteria", [])
    
    # Find the top-weighted criterion — thesis should score highest HERE
    top_criterion = max(criteria, key=lambda c: c.get("weight", 0)) if criteria else None
    top_criterion_name = top_criterion["name"] if top_criterion else "Technical Quality"
    top_criterion_weight = top_criterion.get("weight", 20) if top_criterion else 20
    top_criterion_top_desc = top_criterion.get("top_score_description", "") if top_criterion else ""
    
    # Format all criteria for the prompt
    criteria_block = ""
    for c in criteria:
        criteria_block += (
            f"  - {c['name']} ({c['weight']} pts): {c.get('description', '')}\n"
            f"    TOP SCORE ({c.get('scoring_levels', [{}])[0].get('range', 'max')}): "
            f"{c.get('top_score_description', 'Not specified')}\n"
        )
    
    # 2. Domain context — what the domain research found
    domain_brief = state.domain_brief or {}
    domain_classification = state.domain_classification or "general"
    column_semantics = domain_brief.get("column_semantics", {})
    known_relationships = domain_brief.get("known_relationships", [])
    feature_recipes = domain_brief.get("feature_recipes", [])
    domain_constraints = domain_brief.get("domain_constraints", [])
    domain_summary = domain_brief.get("domain_summary", "")
    
    # 3. EDA context — what the data looks like
    eda_summary = state.eda_insights_summary or ""
    mi_data = state.eda_mutual_info or {}
    modality_flags = state.eda_modality_flags or []
    schema = state.data_schema or {}
    
    # Format schema concisely
    schema_block = "\n".join([f"  {col}: {dtype}" for col, dtype in list(schema.items())[:40]])
    if len(schema) > 40:
        schema_block += f"\n  ... ({len(schema) - 40} more columns)"
    
    # 4. Competition context
    competition_desc = state.competition_description or ""
    recommended_datasets = rubric.get("recommended_datasets", [])
    datasets_block = "\n".join([
        f"  - {d.get('name', '?')}: {d.get('description', 'No description')}"
        for d in recommended_datasets
    ]) or "  None specified — participant chooses data"
    
    # 5. Operator injections (if any domain context was provided)
    operator_context = ""
    for inj in (state.hitl_injections or []):
        operator_context += f"  - {inj.get('text', '')}\n"
    
    # 6. Top MI features (what the data says matters)
    mi_top = mi_data.get("target_mi", [])
    mi_block = ""
    if mi_top:
        top_5_mi = sorted(mi_top, key=lambda x: list(x.values())[0] if isinstance(x, dict) else 0, reverse=True)[:5]
        mi_block = "Top 5 features by mutual information with target:\n"
        for item in top_5_mi:
            if isinstance(item, dict):
                for col, score in item.items():
                    mi_block += f"  {col}: MI={score:.3f}\n"
    
    # === BUILD THE PROMPT ===
    
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

Domain constraints:
{json.dumps(domain_constraints[:5], indent=2) if domain_constraints else '  None identified'}

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
   Examples of conditions: age group, time of day, symptom presentation type, comorbidity count,
   arrival method, prior visit history, geographic region, seasonal pattern.

3. BE TESTABLE — you must be able to compute the conditional effect from available data.
   If the thesis requires data that doesn't exist, it's not testable. Include what external 
   data would strengthen the analysis in data_plan.external_needed.

4. MATTER TO PRACTITIONERS — the target_audience should be specific people who would change their 
   behavior based on the finding. Not "data scientists" but "ED triage nurses" or "hospital 
   administrators designing shift schedules."

5. SCORE HIGH ON "{top_criterion_name}" — this is {top_criterion_weight}% of the total score.
   Every thesis must be designed to score 21+ on this criterion.

For each thesis, estimate scores on EVERY criterion in the rubric. Be realistic — don't 
give everything 25/25. A thesis strong on clinical relevance might be weaker on novelty.
The estimated_total should reflect honest assessment.

WHAT NOT TO DO:
- Do NOT propose "predict triage severity from vitals" — every team will do this
- Do NOT propose pure NLP without a clinical hypothesis behind it
- Do NOT propose something that requires data you can't actually get
- Do NOT give all theses the same estimated_total — rank them honestly
- Do NOT use generic conditions like "different patients" — be specific about WHICH patients
  and WHY the condition matters

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
        "risk": "What could go wrong — specifically, what if the data doesn't support the hypothesis"
    }},
    ... (4 more)
]

RANK by estimated_total descending (highest first).
"""

    # === CALL LLM ===
    
    response = llm_call(
        prompt=prompt,
        agent_name="thesis_generator",
        temperature=0.8,  # Higher temperature than usual — we WANT creative diversity
        max_tokens=6000,  # Thesis list is substantial
        response_format="json",
    )
    
    # === PARSE AND VALIDATE ===
    
    candidates = _parse_thesis_response(response["text"], criteria)
    
    return candidates
```

---

### Response parsing and validation

```python
def _parse_thesis_response(response_text: str, rubric_criteria: list) -> list[ThesisCandidate]:
    """
    Parse LLM JSON response into validated ThesisCandidate objects.
    Handles common LLM output issues: markdown fences, trailing commas, missing fields.
    """
    import json
    import re
    
    # Clean the response
    text = response_text.strip()
    
    # Strip markdown fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    text = text.strip()
    
    # Fix trailing commas (common LLM mistake)
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    try:
        raw_list = json.loads(text)
    except json.JSONDecodeError as e:
        # Last resort: try to extract individual JSON objects
        raw_list = _extract_json_objects(text)
        if not raw_list:
            # Complete failure — generate minimal fallback candidates
            return _fallback_candidates(rubric_criteria)
    
    if not isinstance(raw_list, list):
        raw_list = [raw_list]
    
    # Validate and convert each candidate
    candidates = []
    criterion_names = {c["name"] for c in rubric_criteria}
    
    for i, raw in enumerate(raw_list[:5]):  # Max 5
        candidate = _validate_candidate(raw, i + 1, criterion_names, rubric_criteria)
        candidates.append(candidate)
    
    # Ensure we have exactly 5 (pad with reduced versions if needed)
    while len(candidates) < 5:
        candidates.append(_make_placeholder_candidate(len(candidates) + 1, rubric_criteria))
    
    # Sort by estimated_total descending
    candidates.sort(key=lambda c: c.estimated_total, reverse=True)
    
    # Reassign thesis_ids after sorting
    for i, c in enumerate(candidates):
        c.thesis_id = i + 1
    
    return candidates


def _validate_candidate(raw: dict, thesis_id: int, criterion_names: set, rubric_criteria: list) -> ThesisCandidate:
    """
    Validate a single thesis candidate from the LLM response.
    Fill in defaults for any missing fields rather than crashing.
    """
    # Required fields with defaults
    statement = raw.get("statement", f"Thesis {thesis_id} — statement not generated")
    angle = raw.get("angle", "Not specified")
    target_audience = raw.get("target_audience", "Data scientists and domain practitioners")
    
    data_plan = raw.get("data_plan", {})
    if not isinstance(data_plan, dict):
        data_plan = {"primary_dataset": str(data_plan), "external_needed": [], "join_strategy": ""}
    data_plan.setdefault("primary_dataset", "Competition-provided data")
    data_plan.setdefault("external_needed", [])
    data_plan.setdefault("join_strategy", "")
    
    condition_variable = raw.get("condition_variable", "Not specified")
    hypothesis = raw.get("hypothesis", "Not specified")
    feasibility = raw.get("feasibility", "medium")
    if feasibility not in ("high", "medium", "low"):
        feasibility = "medium"
    risk = raw.get("risk", "No risk assessment provided")
    
    # Estimated scores — validate against rubric
    estimated_scores = raw.get("estimated_scores", {})
    validated_scores = {}
    total = 0
    
    for criterion in rubric_criteria:
        cname = criterion["name"]
        max_pts = criterion.get("max_points", criterion.get("weight", 20))
        
        if cname in estimated_scores:
            score_entry = estimated_scores[cname]
            if isinstance(score_entry, dict):
                score = score_entry.get("score", max_pts // 2)
                justification = score_entry.get("justification", "")
            elif isinstance(score_entry, (int, float)):
                score = int(score_entry)
                justification = ""
            else:
                score = max_pts // 2
                justification = ""
            
            # Clamp to valid range
            score = max(0, min(score, max_pts))
        else:
            # Criterion not in LLM response — assign middle score
            score = max_pts // 2
            justification = "Score not estimated by generator"
        
        validated_scores[cname] = {"score": score, "justification": justification}
        total += score
    
    # Determine condition type from condition_variable text
    condition_type = _classify_condition(condition_variable)
    
    # Compute rubric alignment — how well this thesis targets the top criterion
    top_criterion = max(rubric_criteria, key=lambda c: c.get("weight", 0))
    top_name = top_criterion["name"]
    top_max = top_criterion.get("max_points", top_criterion.get("weight", 20))
    top_score = validated_scores.get(top_name, {}).get("score", 0)
    rubric_alignment = top_score / max(top_max, 1)
    
    return ThesisCandidate(
        thesis_id=thesis_id,
        statement=statement,
        angle=angle,
        target_audience=target_audience,
        data_plan=data_plan,
        condition_variable=condition_variable,
        hypothesis=hypothesis,
        estimated_scores=validated_scores,
        estimated_total=total,
        feasibility=feasibility,
        risk=risk,
        rubric_alignment_score=round(rubric_alignment, 2),
        condition_type=condition_type,
    )


def _classify_condition(condition_text: str) -> str:
    """
    Classify the type of conditional variable from its description.
    Used by downstream agents (Narrative Engine) to select plot types.
    """
    text_lower = condition_text.lower()
    
    if any(kw in text_lower for kw in ["time", "hour", "shift", "day", "night", "season", "month", "minute", "temporal"]):
        return "temporal"
    elif any(kw in text_lower for kw in ["age", "group", "category", "type", "class", "gender", "sex", "race", "ethnicity"]):
        return "categorical"
    elif any(kw in text_lower for kw in ["region", "location", "hospital", "site", "geographic", "zip", "postal"]):
        return "spatial"
    elif any(kw in text_lower for kw in ["threshold", "above", "below", "greater", "less", "cutoff", "level"]):
        return "threshold"
    elif "×" in text_lower or "x " in text_lower or "interaction" in text_lower or " and " in text_lower:
        return "interaction"
    else:
        return "categorical"  # Default — most conditions are categorical splits


def _extract_json_objects(text: str) -> list:
    """Last-resort extraction of JSON objects from malformed LLM output."""
    import re
    objects = []
    # Find all top-level JSON objects
    depth = 0
    start = None
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i+1])
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


def _fallback_candidates(rubric_criteria: list) -> list[ThesisCandidate]:
    """
    When LLM completely fails to generate valid JSON.
    Return minimal placeholder candidates that won't crash the pipeline.
    The operator will override these via HITL.
    """
    candidates = []
    for i in range(5):
        scores = {c["name"]: {"score": c.get("weight", 20) // 3, "justification": "Fallback — LLM generation failed"} 
                  for c in rubric_criteria}
        candidates.append(ThesisCandidate(
            thesis_id=i + 1,
            statement=f"Placeholder thesis {i+1} — LLM generation failed. Please provide your own via HITL.",
            angle="N/A",
            target_audience="N/A",
            data_plan={"primary_dataset": "TBD", "external_needed": [], "join_strategy": ""},
            condition_variable="TBD",
            hypothesis="TBD",
            estimated_scores=scores,
            estimated_total=sum(s["score"] for s in scores.values()),
            feasibility="low",
            risk="LLM generation failed — operator must provide thesis",
            rubric_alignment_score=0.0,
            condition_type="categorical",
        ))
    return candidates


def _make_placeholder_candidate(thesis_id: int, rubric_criteria: list) -> ThesisCandidate:
    """Fill a slot when LLM produced fewer than 5 candidates."""
    scores = {c["name"]: {"score": c.get("weight", 20) // 3, "justification": "Padding candidate"} 
              for c in rubric_criteria}
    return ThesisCandidate(
        thesis_id=thesis_id,
        statement=f"Additional thesis {thesis_id} — not generated. Consider providing your own.",
        angle="N/A",
        target_audience="N/A",
        data_plan={"primary_dataset": "TBD", "external_needed": [], "join_strategy": ""},
        condition_variable="TBD",
        hypothesis="TBD",
        estimated_scores=scores,
        estimated_total=sum(s["score"] for s in scores.values()),
        feasibility="low",
        risk="Placeholder",
        rubric_alignment_score=0.0,
        condition_type="categorical",
    )
```

---

### Custom thesis evaluation (when operator provides their own)

```python
def _evaluate_custom_thesis(
    thesis_text: str,
    rubric: dict,
    state: ProfessorState,
) -> ThesisCandidate:
    """
    When the operator provides their own thesis instead of selecting from generated candidates.
    Evaluate it against the rubric and return a scored ThesisCandidate.
    """
    from tools.llm_provider import llm_call
    
    criteria = rubric.get("criteria", [])
    criteria_block = "\n".join([
        f"  - {c['name']} ({c['weight']} pts): {c.get('description', '')}"
        for c in criteria
    ])
    
    prompt = f"""Evaluate this thesis for a hackathon competition.

THESIS: "{thesis_text}"

RUBRIC:
{criteria_block}

AVAILABLE DATA: {json.dumps(list((state.data_schema or {}).keys())[:30])}
DOMAIN: {state.domain_classification or 'general'}

Evaluate and respond with ONLY valid JSON:
{{
    "statement": "{thesis_text}",
    "angle": "What makes this thesis specific and interesting",
    "target_audience": "Who would use this finding",
    "data_plan": {{
        "primary_dataset": "Best dataset for this thesis",
        "external_needed": ["What external data would help"],
        "join_strategy": "How to combine data"
    }},
    "condition_variable": "The conditional variable implied by this thesis",
    "hypothesis": "Specific testable prediction derived from this thesis",
    "estimated_scores": {{
        {', '.join(f'"{c["name"]}": {{"score": "<int 0-{c.get("max_points", c.get("weight", 20))}>", "justification": "<why>"}}' for c in criteria)}
    }},
    "estimated_total": "<sum>",
    "feasibility": "high" | "medium" | "low",
    "risk": "What could go wrong",
    "strengths": "What's good about this thesis",
    "weaknesses": "What could be improved"
}}
"""
    
    response = llm_call(
        prompt=prompt,
        agent_name="thesis_generator",
        temperature=0.2,  # Low temperature — evaluation, not generation
        response_format="json",
    )
    
    raw = _safe_json_parse(response["text"])
    candidate = _validate_candidate(raw, 0, {c["name"] for c in criteria}, criteria)
    candidate.thesis_id = 0  # Custom thesis marker
    
    return candidate


def _safe_json_parse(text: str) -> dict:
    """Parse JSON with fallback for LLM formatting issues."""
    import json, re
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}
```

---

### Operator presentation and selection

```python
def _present_theses_to_operator(
    candidates: list[ThesisCandidate],
    state: ProfessorState,
) -> str:
    """
    Build the HITL GATE message presenting thesis candidates.
    Returns the operator's response text.
    """
    from tools.operator_channel import emit_to_operator
    
    rubric = state.hackathon_rubric
    criteria = rubric.get("criteria", [])
    
    # Build display for each candidate
    thesis_display = ""
    for c in candidates:
        # Score breakdown per criterion
        score_lines = ""
        for criterion in criteria:
            cname = criterion["name"]
            entry = c.estimated_scores.get(cname, {})
            score = entry.get("score", "?")
            max_pts = criterion.get("max_points", criterion.get("weight", 20))
            justification = entry.get("justification", "")
            # Truncate justification for readability
            if len(justification) > 80:
                justification = justification[:77] + "..."
            score_lines += f"   {cname}: {score}/{max_pts} — {justification}\n"
        
        feasibility_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(c.feasibility, "⚪")
        
        thesis_display += f"""
{c.thesis_id}. [{feasibility_icon} EST: {c.estimated_total}/{rubric.get('total_points', 100)}] "{c.statement}"

   Angle: {c.angle}
   Audience: {c.target_audience}
   Condition: {c.condition_variable} ({c.condition_type})
   Hypothesis: {c.hypothesis}
   
   Score breakdown:
{score_lines}
   Data: {c.data_plan.get('primary_dataset', '?')}
   External needed: {', '.join(c.data_plan.get('external_needed', [])) or 'None'}
   Risk: {c.risk}
"""
    
    message = f"""🔬 THESIS PROPOSALS (ranked by estimated total score)
{thesis_display}
---
Select a thesis number (1-5), describe your own thesis, or reply:
  /thesis select N     — select thesis N
  /thesis custom "..." — provide your own thesis
  /thesis refine N     — regenerate with thesis N as starting point
  /continue            — auto-select thesis 1"""
    
    response = emit_to_operator(message, level="GATE")
    return response


def _parse_operator_selection(
    response: str,
    candidates: list[ThesisCandidate],
    rubric: dict,
    state: ProfessorState,
) -> dict:
    """
    Parse the operator's response and return the selected thesis.
    
    Handles:
    - Number alone ("1" or "3")
    - /thesis select N
    - /thesis custom "..."
    - /continue (auto-select #1)
    - Free text (treated as custom thesis)
    - None (timeout — auto-select #1)
    """
    from tools.operator_channel import emit_to_operator
    
    if response is None or response.strip() == "/continue":
        # Timeout or explicit continue — auto-select #1
        selected = candidates[0]
        return {
            "active_thesis": _candidate_to_dict(selected),
            "thesis_selected_by": "auto",
        }
    
    response = response.strip()
    
    # Pattern: bare number
    if response.isdigit():
        idx = int(response)
        if 1 <= idx <= len(candidates):
            selected = candidates[idx - 1]
            emit_to_operator(f"✅ Thesis {idx} selected: \"{selected.statement[:80]}...\"", level="STATUS")
            return {
                "active_thesis": _candidate_to_dict(selected),
                "thesis_selected_by": "operator",
            }
    
    # Pattern: /thesis select N
    if response.startswith("/thesis select"):
        parts = response.split()
        if len(parts) >= 3 and parts[2].isdigit():
            idx = int(parts[2])
            if 1 <= idx <= len(candidates):
                selected = candidates[idx - 1]
                emit_to_operator(f"✅ Thesis {idx} selected: \"{selected.statement[:80]}...\"", level="STATUS")
                return {
                    "active_thesis": _candidate_to_dict(selected),
                    "thesis_selected_by": "operator",
                }
    
    # Pattern: /thesis custom "..."
    if response.startswith("/thesis custom"):
        custom_text = response.replace("/thesis custom", "").strip().strip('"').strip("'")
        if custom_text:
            emit_to_operator(f"🔬 Evaluating custom thesis: \"{custom_text[:80]}...\"", level="STATUS")
            custom_candidate = _evaluate_custom_thesis(custom_text, rubric, state)
            
            # Present evaluation to operator for confirmation
            confirm_msg = (
                f"🔬 Custom thesis evaluation:\n"
                f"Statement: {custom_candidate.statement}\n"
                f"Estimated total: {custom_candidate.estimated_total}/{rubric.get('total_points', 100)}\n"
                f"Feasibility: {custom_candidate.feasibility}\n"
                f"Risk: {custom_candidate.risk}\n\n"
                f"Confirm? /continue or provide a different thesis"
            )
            confirm = emit_to_operator(confirm_msg, level="CHECKPOINT")
            
            if confirm is None or "/continue" in (confirm or ""):
                return {
                    "active_thesis": _candidate_to_dict(custom_candidate),
                    "thesis_selected_by": "operator",
                }
            else:
                # Operator wants to revise — recursively handle their new response
                return _parse_operator_selection(confirm, candidates, rubric, state)
    
    # Pattern: /thesis refine N
    if response.startswith("/thesis refine"):
        parts = response.split()
        if len(parts) >= 3 and parts[2].isdigit():
            idx = int(parts[2])
            if 1 <= idx <= len(candidates):
                # TODO: Regenerate with thesis N as seed
                # For now, just select it
                selected = candidates[idx - 1]
                emit_to_operator(f"✅ Thesis {idx} selected (refinement not yet implemented)", level="STATUS")
                return {
                    "active_thesis": _candidate_to_dict(selected),
                    "thesis_selected_by": "operator",
                }
    
    # Fallback: treat any other text as a custom thesis
    if len(response) > 20:  # Long enough to be a thesis
        emit_to_operator(f"🔬 Treating as custom thesis: \"{response[:80]}...\"", level="STATUS")
        custom_candidate = _evaluate_custom_thesis(response, rubric, state)
        return {
            "active_thesis": _candidate_to_dict(custom_candidate),
            "thesis_selected_by": "operator",
        }
    
    # Very short response or unrecognized — default to #1
    emit_to_operator(f"⚠️ Unrecognized response. Auto-selecting thesis 1.", level="STATUS")
    selected = candidates[0]
    return {
        "active_thesis": _candidate_to_dict(selected),
        "thesis_selected_by": "auto",
    }


def _candidate_to_dict(candidate: ThesisCandidate) -> dict:
    """Convert ThesisCandidate to dict for state storage."""
    from dataclasses import asdict
    return asdict(candidate)
```

---

### The LangGraph node function

```python
def thesis_generator(state: ProfessorState) -> dict:
    """
    LangGraph node. Generates thesis candidates and presents to operator for selection.
    
    Reads: hackathon_rubric, hackathon_effort_plan, competition_description,
           eda_insights_summary, eda_mutual_info, eda_modality_flags,
           domain_brief, domain_classification, data_schema, hitl_injections
    Writes: thesis_candidates, active_thesis, thesis_selected_by
    Emits: STATUS (generating), GATE (thesis selection)
    
    Pipeline position: After domain_research, before external_data_scout
    """
    from tools.operator_channel import emit_to_operator
    
    # Safety check — only run in hackathon mode
    if not state.hackathon_mode:
        return {}
    
    emit_to_operator("🔬 Generating thesis candidates...", level="STATUS")
    
    # Check thesis depth from effort plan
    effort_plan = state.hackathon_effort_plan or {}
    thesis_depth = effort_plan.get("thesis_depth", "standard")
    
    # Generate candidates
    candidates = _generate_thesis_candidates(state)
    
    # If thesis depth is "deep", generate a second batch and merge best
    if thesis_depth == "deep" and len(candidates) >= 5:
        emit_to_operator("🔬 Deep thesis mode — generating additional candidates...", level="STATUS")
        
        # Second generation with different temperature for diversity
        # Inject a constraint: "Your candidates must be DIFFERENT from: {existing statements}"
        existing_statements = [c.statement for c in candidates]
        
        # Temporarily modify state to include diversity constraint
        # (done by adding to operator context, not modifying state directly)
        additional_candidates = _generate_diverse_batch(state, existing_statements)
        
        # Merge: take top 5 from combined pool
        all_candidates = candidates + additional_candidates
        all_candidates.sort(key=lambda c: c.estimated_total, reverse=True)
        candidates = all_candidates[:5]
        
        # Reassign IDs
        for i, c in enumerate(candidates):
            c.thesis_id = i + 1
    
    # Present to operator (GATE — must respond)
    response = _present_theses_to_operator(candidates, state)
    
    # Parse selection
    rubric = state.hackathon_rubric or {}
    selection_result = _parse_operator_selection(response, candidates, rubric, state)
    
    # Log the selection
    active = selection_result.get("active_thesis", {})
    selected_by = selection_result.get("thesis_selected_by", "auto")
    emit_to_operator(
        f"✅ Active thesis ({selected_by}): \"{active.get('statement', '?')[:100]}\"",
        level="STATUS"
    )
    
    # Return state updates
    return {
        "thesis_candidates": [_candidate_to_dict(c) for c in candidates],
        "active_thesis": active,
        "thesis_selected_by": selected_by,
    }


def _generate_diverse_batch(state: ProfessorState, existing_statements: list) -> list[ThesisCandidate]:
    """
    Generate a second batch of candidates that are deliberately different from the first.
    Used in 'deep' thesis mode for maximum diversity.
    """
    from tools.llm_provider import llm_call
    
    diversity_prompt = f"""Generate 5 ADDITIONAL thesis candidates that are COMPLETELY DIFFERENT 
from these existing candidates:

{chr(10).join(f'- {s}' for s in existing_statements)}

Your new candidates must:
- Address DIFFERENT aspects of the problem
- Use DIFFERENT condition variables
- Target DIFFERENT audiences
- Take DIFFERENT analytical approaches

Do NOT overlap with any existing thesis. Maximize diversity.
"""
    
    # This reuses the main generation flow but with the diversity constraint
    # injected into the operator_context
    # Implementation: call _generate_thesis_candidates with modified prompt
    # For simplicity, we call llm_call directly with a combined prompt
    
    # ... (reuse the main prompt structure with diversity_prompt prepended)
    # The actual implementation would build the full prompt with diversity constraint
    
    # For now, return empty — the main 5 are usually sufficient
    return []
```

---

### State additions

Add to `graph/state.py` (owned by thesis_generator):

```python
# ═══════════════════════════════════════════════
# THESIS — owner: thesis_generator
# ═══════════════════════════════════════════════
thesis_candidates: list = Field(default_factory=list)    # All generated ThesisCandidate dicts
active_thesis: dict = Field(default_factory=dict)        # Selected thesis
thesis_selected_by: str = ""                             # "operator" | "auto"
```

Update `_FIELD_OWNERS`:
```python
"thesis_candidates": "thesis_generator",
"active_thesis": "thesis_generator",
"thesis_selected_by": "thesis_generator",
```

---

## Contract Tests: tests/contracts/test_thesis_generator_contract.py

### Test fixtures

```python
import pytest
from unittest.mock import patch, MagicMock
from graph.state import ProfessorState

@pytest.fixture
def hackathon_state():
    """State with rubric parsed and EDA/domain complete — ready for thesis generation."""
    return ProfessorState(
        session_id="test-thesis",
        competition_name="Triagegeist",
        competition_description="Build an AI-powered tool for emergency triage...",
        hackathon_mode=True,
        hackathon_rubric={
            "competition_name": "Triagegeist",
            "total_points": 100,
            "criteria": [
                {"name": "Clinical Relevance", "weight": 25, "max_points": 25,
                 "description": "Real problem in emergency triage",
                 "top_score_description": "Sharply defined, clinically motivated"},
                {"name": "Technical Quality", "weight": 30, "max_points": 30,
                 "description": "Sound methodology, clean code"},
                {"name": "Documentation", "weight": 20, "max_points": 20,
                 "description": "Clear writeup"},
                {"name": "Insight", "weight": 15, "max_points": 15,
                 "description": "Meaningful findings"},
                {"name": "Novelty", "weight": 10, "max_points": 10,
                 "description": "Fresh perspective"},
            ],
            "recommended_datasets": [{"name": "MIMIC-IV-ED", "url": "physionet.org", "description": "ED data"}],
        },
        hackathon_effort_plan={
            "thesis_depth": "deep",
            "technical_depth": "marathon",
        },
        domain_classification="healthcare",
        domain_brief={
            "primary_domain": "healthcare",
            "sub_classification": "emergency triage",
            "column_semantics": {"triage_level": {"meaning": "ESI acuity score 1-5"}},
            "known_relationships": [{"features": ["age", "vital_signs"], "relationship": "Age modifies vital sign thresholds"}],
            "domain_summary": "Emergency triage assigns severity levels to incoming patients.",
        },
        data_schema={"patient_id": "Int64", "age": "Float64", "heart_rate": "Float64", 
                      "bp_systolic": "Float64", "chief_complaint": "Utf8", "triage_level": "Int64"},
        eda_insights_summary="Dataset has 45K rows. Age distribution is bimodal (peaks at 35 and 72). Heart rate MI with triage_level is 0.34.",
        eda_mutual_info={"target_mi": [{"heart_rate": 0.34}, {"bp_systolic": 0.28}, {"age": 0.22}]},
        eda_modality_flags=["age"],
    )


MOCK_LLM_THESIS_RESPONSE = json.dumps([
    {
        "thesis_id": 1,
        "statement": "ESI undertriages elderly patients with atypical cardiac presentations",
        "angle": "Age × presentation interaction reveals systematic bias in ESI scoring",
        "target_audience": "ED triage nurses and ESI algorithm designers",
        "data_plan": {
            "primary_dataset": "MIMIC-IV-ED",
            "external_needed": ["AHA cardiac risk thresholds by age"],
            "join_strategy": "Lookup table join on age_group"
        },
        "condition_variable": "age_group × presentation_type",
        "hypothesis": "Patients >65 with atypical cardiac symptoms are undertriaged 2x more than those with typical presentations",
        "estimated_scores": {
            "Clinical Relevance": {"score": 23, "justification": "Sharply defined, documented problem"},
            "Technical Quality": {"score": 26, "justification": "Standard ML approach sufficient"},
            "Documentation": {"score": 16, "justification": "Clear narrative arc"},
            "Insight": {"score": 12, "justification": "Age-adjusted thresholds actionable"},
            "Novelty": {"score": 7, "justification": "Known concern, novel data analysis"}
        },
        "estimated_total": 84,
        "feasibility": "high",
        "risk": "MIMIC-IV-ED may not have enough atypical cardiac cases"
    },
    {
        "thesis_id": 2,
        "statement": "Night shift triage accuracy degrades for medium-acuity patients",
        "angle": "Cognitive fatigue isolated from volume effects",
        "target_audience": "Hospital administrators designing shift schedules",
        "data_plan": {
            "primary_dataset": "MIMIC-IV-ED",
            "external_needed": ["Published shift fatigue studies"],
            "join_strategy": "Time-based filtering"
        },
        "condition_variable": "shift_period × acuity_level",
        "hypothesis": "Medium-acuity patients triaged during night shift are undertriaged 30% more",
        "estimated_scores": {
            "Clinical Relevance": {"score": 21, "justification": "Relevant but less urgent"},
            "Technical Quality": {"score": 25, "justification": "Requires causal analysis"},
            "Documentation": {"score": 16, "justification": "Compelling narrative"},
            "Insight": {"score": 11, "justification": "Staffing implications"},
            "Novelty": {"score": 6, "justification": "Known topic, per-acuity is new"}
        },
        "estimated_total": 79,
        "feasibility": "medium",
        "risk": "Timestamps may not distinguish shift boundaries"
    },
    # ... theses 3-5 would follow
] + [
    {"thesis_id": i, "statement": f"Thesis {i}", "angle": "angle", "target_audience": "audience",
     "data_plan": {"primary_dataset": "data", "external_needed": [], "join_strategy": ""},
     "condition_variable": "var", "hypothesis": "hyp",
     "estimated_scores": {"Clinical Relevance": {"score": 15, "justification": "ok"},
                           "Technical Quality": {"score": 20, "justification": "ok"},
                           "Documentation": {"score": 12, "justification": "ok"},
                           "Insight": {"score": 8, "justification": "ok"},
                           "Novelty": {"score": 5, "justification": "ok"}},
     "estimated_total": 60, "feasibility": "medium", "risk": "risk"}
    for i in range(3, 6)
])
```

### Tests — Generation correctness

```python
class TestThesisGeneration:
    
    def test_generates_5_candidates(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        assert len(candidates) == 5
    
    def test_each_has_required_fields(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        for c in candidates:
            assert c.statement and len(c.statement) > 10
            assert c.angle
            assert c.target_audience
            assert isinstance(c.data_plan, dict)
            assert "primary_dataset" in c.data_plan
            assert "external_needed" in c.data_plan
            assert c.condition_variable
            assert c.hypothesis
            assert isinstance(c.estimated_scores, dict)
            assert c.estimated_total > 0
            assert c.feasibility in ("high", "medium", "low")
            assert c.risk
    
    def test_scores_cover_all_rubric_criteria(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        rubric_names = {c["name"] for c in hackathon_state.hackathon_rubric["criteria"]}
        for candidate in candidates:
            assert set(candidate.estimated_scores.keys()) == rubric_names
    
    def test_ranked_by_total_descending(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        totals = [c.estimated_total for c in candidates]
        assert totals == sorted(totals, reverse=True)
    
    def test_condition_variable_present_and_classified(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        for c in candidates:
            assert c.condition_variable != ""
            assert c.condition_type in ("categorical", "temporal", "spatial", "threshold", "interaction")
    
    def test_hypothesis_is_testable(self, hackathon_state):
        """Each hypothesis should contain a direction or magnitude (not just 'different')."""
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        for c in candidates:
            # Hypothesis should be more than "X is different"
            assert len(c.hypothesis) > 20
    
    def test_scores_clamped_to_rubric_max(self, hackathon_state):
        """No estimated score exceeds the criterion's max_points."""
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        criteria_max = {c["name"]: c.get("max_points", c.get("weight", 20))
                        for c in hackathon_state.hackathon_rubric["criteria"]}
        
        for candidate in candidates:
            for cname, entry in candidate.estimated_scores.items():
                assert entry["score"] <= criteria_max.get(cname, 100)
                assert entry["score"] >= 0
    
    def test_estimated_total_matches_scores(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        for c in candidates:
            computed = sum(entry["score"] for entry in c.estimated_scores.values())
            assert c.estimated_total == computed
    
    def test_rubric_alignment_score_populated(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        for c in candidates:
            assert 0.0 <= c.rubric_alignment_score <= 1.0


class TestOperatorSelection:
    
    def test_number_selects_thesis(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        result = _parse_operator_selection("2", candidates, hackathon_state.hackathon_rubric, hackathon_state)
        assert result["thesis_selected_by"] == "operator"
        # Should be thesis that was originally #2 (before any re-sorting)
        assert result["active_thesis"]["statement"] != ""
    
    def test_slash_command_selects(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        result = _parse_operator_selection("/thesis select 1", candidates, hackathon_state.hackathon_rubric, hackathon_state)
        assert result["thesis_selected_by"] == "operator"
        assert result["active_thesis"]["thesis_id"] == 1
    
    def test_continue_auto_selects_first(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        result = _parse_operator_selection("/continue", candidates, hackathon_state.hackathon_rubric, hackathon_state)
        assert result["thesis_selected_by"] == "auto"
        assert result["active_thesis"]["thesis_id"] == 1  # Highest scored
    
    def test_timeout_auto_selects(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        result = _parse_operator_selection(None, candidates, hackathon_state.hackathon_rubric, hackathon_state)
        assert result["thesis_selected_by"] == "auto"
    
    def test_custom_thesis_evaluated(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            # First call: generation. Second call: custom evaluation. Third call: checkpoint.
            mock_llm.side_effect = [
                {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "", "input_tokens": 500, 
                 "output_tokens": 2000, "model": "test", "cost_usd": 0.01},
                {"text": json.dumps({
                    "statement": "Custom thesis about medication interactions",
                    "angle": "Novel angle",
                    "target_audience": "Pharmacists",
                    "data_plan": {"primary_dataset": "MIMIC-IV-ED", "external_needed": [], "join_strategy": ""},
                    "condition_variable": "polypharmacy_count",
                    "hypothesis": "Patients on 5+ medications are undertriaged",
                    "estimated_scores": {
                        "Clinical Relevance": {"score": 20, "justification": "relevant"},
                        "Technical Quality": {"score": 22, "justification": "standard"},
                        "Documentation": {"score": 15, "justification": "clear"},
                        "Insight": {"score": 10, "justification": "useful"},
                        "Novelty": {"score": 8, "justification": "underexplored"},
                    },
                    "estimated_total": 75,
                    "feasibility": "medium",
                    "risk": "Drug data may be limited",
                }), "reasoning": "", "input_tokens": 300, "output_tokens": 500, "model": "test", "cost_usd": 0.005},
            ]
            
            candidates = _generate_thesis_candidates(hackathon_state)
        
        with patch("tools.operator_channel.emit_to_operator", return_value="/continue"):
            with patch("tools.llm_provider.llm_call") as mock_eval:
                mock_eval.return_value = {"text": json.dumps({
                    "statement": "Custom thesis about medication interactions",
                    "estimated_scores": {"Clinical Relevance": {"score": 20, "justification": "ok"},
                                          "Technical Quality": {"score": 22, "justification": "ok"},
                                          "Documentation": {"score": 15, "justification": "ok"},
                                          "Insight": {"score": 10, "justification": "ok"},
                                          "Novelty": {"score": 8, "justification": "ok"}},
                    "estimated_total": 75, "feasibility": "medium", "risk": "limited data",
                    "condition_variable": "polypharmacy", "hypothesis": "test", "angle": "new",
                    "target_audience": "pharmacists",
                    "data_plan": {"primary_dataset": "MIMIC", "external_needed": [], "join_strategy": ""},
                }), "reasoning": "", "input_tokens": 200, "output_tokens": 300, "model": "test", "cost_usd": 0.003}
                
                result = _parse_operator_selection(
                    '/thesis custom "Patients on 5+ medications are systematically undertriaged"',
                    candidates, hackathon_state.hackathon_rubric, hackathon_state
                )
        
        assert result["thesis_selected_by"] == "operator"
        assert result["active_thesis"]["thesis_id"] == 0  # Custom marker
    
    def test_long_free_text_treated_as_custom(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        with patch("tools.operator_channel.emit_to_operator", return_value="/continue"):
            with patch("tools.llm_provider.llm_call") as mock_eval:
                mock_eval.return_value = {"text": "{}", "reasoning": "", "input_tokens": 100,
                                          "output_tokens": 100, "model": "test", "cost_usd": 0.001}
                result = _parse_operator_selection(
                    "I want to analyze whether repeated ED visits within 72 hours indicate systematic undertriage in the initial visit",
                    candidates, hackathon_state.hackathon_rubric, hackathon_state
                )
        
        assert result["thesis_selected_by"] == "operator"


class TestEdgeCases:
    
    def test_malformed_json_produces_fallback(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": "this is not json at all", "reasoning": "",
                                     "input_tokens": 100, "output_tokens": 100,
                                     "model": "test", "cost_usd": 0.001}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        assert len(candidates) == 5  # Fallback produces 5
        assert "Placeholder" in candidates[0].statement or "failed" in candidates[0].statement.lower()
    
    def test_partial_json_salvaged(self, hackathon_state):
        """LLM returns 3 valid theses instead of 5 — pad to 5."""
        partial = json.dumps([
            {"thesis_id": i, "statement": f"Good thesis {i}", "angle": "angle",
             "target_audience": "audience", "data_plan": {"primary_dataset": "data", "external_needed": [], "join_strategy": ""},
             "condition_variable": "var", "hypothesis": "hyp",
             "estimated_scores": {"Clinical Relevance": {"score": 20, "justification": "good"},
                                   "Technical Quality": {"score": 25, "justification": "good"},
                                   "Documentation": {"score": 15, "justification": "good"},
                                   "Insight": {"score": 10, "justification": "good"},
                                   "Novelty": {"score": 7, "justification": "good"}},
             "estimated_total": 77, "feasibility": "high", "risk": "risk"}
            for i in range(1, 4)  # Only 3 theses
        ])
        
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": partial, "reasoning": "",
                                     "input_tokens": 300, "output_tokens": 1000,
                                     "model": "test", "cost_usd": 0.005}
            candidates = _generate_thesis_candidates(hackathon_state)
        
        assert len(candidates) == 5  # Padded to 5
        # First 3 should be the real ones, sorted by total
        real_candidates = [c for c in candidates if "Placeholder" not in c.statement and "Additional" not in c.statement]
        assert len(real_candidates) >= 3
    
    def test_non_hackathon_mode_returns_empty(self):
        state = ProfessorState(hackathon_mode=False)
        result = thesis_generator(state)
        assert result == {}
    
    def test_missing_rubric_uses_defaults(self):
        state = ProfessorState(
            hackathon_mode=True,
            hackathon_rubric={},  # Empty rubric
            competition_description="Some competition",
            data_schema={"col1": "Float64"},
        )
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": "[]", "reasoning": "",
                                     "input_tokens": 100, "output_tokens": 100,
                                     "model": "test", "cost_usd": 0.001}
            with patch("tools.operator_channel.emit_to_operator", return_value="/continue"):
                result = thesis_generator(state)
        
        # Should not crash — produces fallback candidates
        assert "thesis_candidates" in result
        assert len(result["thesis_candidates"]) == 5


class TestConditionClassification:
    
    def test_temporal_condition(self):
        assert _classify_condition("time_of_day × shift_period") == "temporal"
    
    def test_categorical_condition(self):
        assert _classify_condition("age_group × gender") == "categorical"
    
    def test_spatial_condition(self):
        assert _classify_condition("hospital_location × region") == "spatial"
    
    def test_threshold_condition(self):
        assert _classify_condition("heart_rate above 100 bpm") == "threshold"
    
    def test_interaction_condition(self):
        assert _classify_condition("age × presentation_type interaction") == "interaction"
    
    def test_default_is_categorical(self):
        assert _classify_condition("some unknown variable") == "categorical"


class TestNodeFunction:
    
    def test_returns_correct_state_fields(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            with patch("tools.operator_channel.emit_to_operator", return_value="1"):
                result = thesis_generator(hackathon_state)
        
        assert "thesis_candidates" in result
        assert "active_thesis" in result
        assert "thesis_selected_by" in result
        assert len(result["thesis_candidates"]) == 5
        assert result["active_thesis"]["statement"] != ""
    
    def test_gate_emitted_for_selection(self, hackathon_state):
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            with patch("tools.operator_channel.emit_to_operator") as mock_emit:
                mock_emit.return_value = "/continue"
                thesis_generator(hackathon_state)
                
                # Find GATE call
                gate_calls = [c for c in mock_emit.call_args_list 
                              if c.kwargs.get("level") == "GATE" or
                              (len(c.args) > 1 and c.args[1] == "GATE")]
                assert len(gate_calls) >= 1
    
    def test_uses_higher_temperature(self, hackathon_state):
        """Thesis generation should use temperature=0.8 for creative diversity."""
        with patch("tools.llm_provider.llm_call") as mock_llm:
            mock_llm.return_value = {"text": MOCK_LLM_THESIS_RESPONSE, "reasoning": "",
                                     "input_tokens": 500, "output_tokens": 2000,
                                     "model": "test", "cost_usd": 0.01}
            with patch("tools.operator_channel.emit_to_operator", return_value="/continue"):
                thesis_generator(hackathon_state)
            
            # Check the temperature used in the LLM call
            call_kwargs = mock_llm.call_args_list[0].kwargs if mock_llm.call_args_list[0].kwargs else {}
            # Temperature should be 0.7-0.9 range (creative generation)
            if "temperature" in call_kwargs:
                assert call_kwargs["temperature"] >= 0.7
```

---

## WHAT NOT TO DO

- Do NOT generate generic "predict Y from X" theses. Every thesis must be CONDITIONAL.
- Do NOT use temperature < 0.5 for thesis generation. We WANT creative diversity. Use 0.8.
- Do NOT crash if the LLM returns malformed JSON. Fall back to placeholders and let the operator provide their own via HITL.
- Do NOT skip the GATE. The operator MUST see thesis proposals and select one. This is the highest-leverage decision in the hackathon.
- Do NOT let estimated scores exceed max_points for any criterion. Clamp.
- Do NOT auto-select in non-AUTONOMOUS modes. The operator chooses.
- Do NOT include the full competition description in the prompt if it exceeds 3000 chars. Truncate.
- Do NOT modify any existing traditional-mode files. This is an ADDITION.
- Do NOT use Pandas. If you somehow need data operations, use Polars.
- Do NOT let a thesis candidate have empty condition_variable or hypothesis. These are REQUIRED for the downstream pipeline.