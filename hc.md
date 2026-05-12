# BUILD PROMPT — Hackathon Mode Component 1: Rubric Parser
# Feed to Gemini CLI with: @PROFESSOR.md @STATE.md @HITL.md @CONTRACTS.md @PROMPTS.md @PROVIDERS.md

---

## CONTEXT

Professor v2's traditional pipeline is complete (Layers 0-5, all contracts passing). You are now building the first component of the Hackathon Mode extension. This is an ADDITION — you do NOT modify any existing traditional-mode files.

The Rubric Parser reads a hackathon competition page, extracts the judging criteria with weights, identifies submission requirements, and builds an effort allocation plan that configures the rest of the hackathon pipeline. Every downstream hackathon component reads the rubric parser's output to calibrate its depth.

This component depends on Layer 0 infrastructure: ProfessorState (for typed state), llm_call() (for extraction), emit_to_operator() (for presenting the parsed rubric). It does NOT depend on any Layer 1-5 agents.

---

## COMMIT PLAN (2 commits)

```
Commit 1:  tools/rubric_parser.py + tests/contracts/test_rubric_parser_contract.py
Commit 2:  State additions + HITL integration + graph node registration
```

Both commits pass `pytest tests/contracts/ -q` including ALL existing traditional-mode tests.

---

## COMMIT 1: tools/rubric_parser.py

### File structure

```
tools/
└── rubric_parser.py
    ├── HackathonRubric (dataclass)
    ├── EffortPlan (dataclass)
    ├── parse_rubric(competition_text: str) -> HackathonRubric
    ├── build_effort_plan(rubric: HackathonRubric) -> EffortPlan
    └── run_rubric_parser(state: ProfessorState) -> dict  (LangGraph node function)
```

---

### Data structures

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RubricCriterion:
    """Single judging criterion extracted from the competition page."""
    name: str                           # e.g., "Clinical Relevance"
    weight: int                         # Points: e.g., 25
    max_points: int                     # Same as weight for most competitions
    description: str                    # What the judges look for
    top_score_description: str          # What earns full marks (21-25 range description)
    bottom_score_description: str       # What earns minimum (0-6 range description)
    scoring_levels: list                # [{range: "21-25", description: "..."}, ...]


@dataclass
class HackathonRubric:
    """Complete parsed rubric from a hackathon competition page."""
    competition_name: str
    total_points: int                                       # Sum of all criteria weights
    criteria: list                                          # List of RubricCriterion dicts
    submission_requirements: list                           # ["notebook", "writeup", "cover_image", "project_link"]
    writeup_template: dict                                  # {"sections": ["clinical_problem_statement", ...], "max_words": 2000}
    data_policy: str                                        # "any_public" | "provided_only" | "specific_sources"
    recommended_datasets: list                              # [{"name": "MIMIC-IV-ED", "url": "physionet.org", "description": "..."}]
    deliverable_type: str                                   # "notebook_and_writeup" | "application" | "analysis"
    tracks: list                                            # Competition tracks if multiple exist
    prizes: list                                            # [{"place": "1st", "amount": "$5,000"}]
    deadline: str                                           # ISO date or descriptive string
    raw_text_hash: str                                      # SHA-256 of input text for cache validation


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
```

---

### Function 1: parse_rubric()

```python
def parse_rubric(competition_text: str) -> HackathonRubric:
    """
    Extract structured rubric from raw competition page text.
    
    Args:
        competition_text: The full text of the competition description page.
                          Can be scraped HTML-to-text, copy-pasted, or provided via HITL.
    
    Returns:
        HackathonRubric with all fields populated.
        Fields that can't be extracted are set to sensible defaults.
    """
```

**Implementation — two extraction passes:**

**Pass 1: Deterministic extraction (no LLM, fast, catches structured rubrics).**

Many hackathons have explicit rubric tables. Try to extract directly:

```python
def _deterministic_extract(text: str) -> dict:
    """
    Try to extract rubric from structured text patterns.
    Works when the competition has explicit scoring tables.
    """
    result = {
        "criteria": [],
        "submission_requirements": [],
        "writeup_sections": [],
        "recommended_datasets": [],
        "max_writeup_words": None,
        "prizes": [],
    }
    
    # Pattern 1: "X points" or "(X points)" or "X pts"
    # Look for lines like "Clinical Relevance (25 points)" or "1. Clinical Relevance — 25 pts"
    import re
    
    criteria_patterns = [
        # "Name (N points)" or "Name (N pts)"
        r'(?:^|\n)\s*\d*\.?\s*([A-Z][A-Za-z\s&]+?)\s*\((\d+)\s*(?:points?|pts)\)',
        # "Name — N points" or "Name: N points"
        r'(?:^|\n)\s*\d*\.?\s*([A-Z][A-Za-z\s&]+?)\s*[—:\-]\s*(\d+)\s*(?:points?|pts)',
        # Table format: "| Name | N |" or "Name\tN"
        r'(?:^|\n)\s*\|?\s*([A-Z][A-Za-z\s&]+?)\s*\|?\s*(\d+)\s*\|?',
    ]
    
    for pattern in criteria_patterns:
        matches = re.findall(pattern, text)
        if len(matches) >= 3:  # At least 3 criteria found — likely the real rubric
            for name, points in matches:
                name = name.strip().rstrip('.')
                points = int(points)
                if 5 <= points <= 50 and name not in ("Total", "total"):  # Sanity check
                    result["criteria"].append({
                        "name": name,
                        "weight": points,
                        "max_points": points,
                    })
            break  # Use the first pattern that works
    
    # Pattern 2: Submission requirements
    # Look for "must submit", "required", "each team must"
    requirement_keywords = {
        "notebook": ["notebook", "kaggle notebook", "code notebook"],
        "writeup": ["writeup", "write-up", "project writeup", "report"],
        "cover_image": ["cover image", "cover photo", "thumbnail"],
        "project_link": ["project link", "github", "repository", "demo link", "working product"],
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
            result["recommended_datasets"].append({
                "name": match.group(1),
                "url": "",  # LLM pass will fill this
                "description": "",
            })
    
    return result
```

**Pass 2: LLM extraction (fills gaps, adds descriptions, handles unstructured text).**

Only call LLM if Pass 1 is incomplete (< 3 criteria found, or descriptions missing):

```python
def _llm_extract(competition_text: str, partial_result: dict) -> dict:
    """
    Use LLM to extract/complete rubric from competition text.
    Called only when deterministic extraction is incomplete.
    """
    from tools.llm_provider import llm_call
    
    # Build prompt with what we already have
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
    
    response = llm_call(
        prompt=prompt,
        agent_name="rubric_parser",
        temperature=0.1,  # Low temperature — we want accurate extraction, not creativity
        response_format="json",
    )
    
    # Parse JSON response
    try:
        extracted = json.loads(response["text"].strip().strip('`').strip())
    except json.JSONDecodeError:
        # Try to salvage — strip markdown fences if present
        cleaned = response["text"]
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        try:
            extracted = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            # Complete failure — return defaults
            extracted = _default_rubric(competition_text)
    
    return extracted
```

**Merging deterministic + LLM results:**

```python
def parse_rubric(competition_text: str) -> HackathonRubric:
    import hashlib
    
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
```

**Merging logic — deterministic results take priority for weights:**

```python
def _merge_results(deterministic: dict, llm: dict) -> dict:
    """
    Merge deterministic and LLM extraction results.
    Deterministic wins for weights (more reliable).
    LLM wins for descriptions (can't extract deterministically).
    """
    merged = {}
    
    # Criteria: use deterministic weights if available, LLM descriptions
    if deterministic["criteria"]:
        # Match by name (fuzzy — "Technical Quality" might be "Technical" in deterministic)
        det_names = {c["name"].lower().strip(): c for c in deterministic["criteria"]}
        llm_names = {c["name"].lower().strip(): c for c in llm.get("criteria", [])}
        
        merged_criteria = []
        for det_key, det_crit in det_names.items():
            # Find matching LLM criterion
            llm_match = None
            for llm_key, llm_crit in llm_names.items():
                # Fuzzy match: check if one is substring of the other
                if det_key in llm_key or llm_key in det_key:
                    llm_match = llm_crit
                    break
            
            if llm_match:
                # Use deterministic weight, LLM descriptions
                merged_criteria.append({
                    **llm_match,
                    "weight": det_crit["weight"],
                    "max_points": det_crit["weight"],
                })
            else:
                merged_criteria.append(det_crit)
        
        # Add any LLM criteria not matched by deterministic
        for llm_key, llm_crit in llm_names.items():
            if not any(llm_key in dk or dk in llm_key for dk in det_names):
                merged_criteria.append(llm_crit)
        
        merged["criteria"] = merged_criteria
    else:
        merged["criteria"] = llm.get("criteria", [])
    
    # Other fields: prefer deterministic if present, else LLM
    merged["submission_requirements"] = (
        deterministic["submission_requirements"] or 
        llm.get("submission_requirements", ["notebook", "writeup"])
    )
    merged["max_writeup_words"] = (
        deterministic.get("max_writeup_words") or 
        llm.get("writeup_template", {}).get("max_words", 2000)
    )
    merged["recommended_datasets"] = (
        deterministic.get("recommended_datasets") or
        llm.get("recommended_datasets", [])
    )
    merged["prizes"] = deterministic.get("prizes") or llm.get("prizes", [])
    
    # Fields only from LLM
    merged["competition_name"] = llm.get("competition_name", "Unknown Competition")
    merged["writeup_template"] = llm.get("writeup_template", {"sections": [], "max_words": merged["max_writeup_words"]})
    merged["data_policy"] = llm.get("data_policy", "any_public")
    merged["deliverable_type"] = llm.get("deliverable_type", "notebook_and_writeup")
    merged["tracks"] = llm.get("tracks", [])
    merged["deadline"] = llm.get("deadline", "unknown")
    
    return merged
```

**Default rubric when extraction completely fails:**

```python
def _default_rubric(competition_text: str) -> dict:
    """
    Fallback rubric when both deterministic and LLM extraction fail.
    5 equal-weight criteria covering the standard hackathon dimensions.
    """
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
```

**Validation — catch extraction errors:**

```python
def _build_rubric_from_merged(merged: dict, original_text: str) -> HackathonRubric:
    """
    Validate and construct the final HackathonRubric.
    Catches and fixes common extraction errors.
    """
    criteria = merged.get("criteria", [])
    
    # Validation 1: Total points should be ~100. If way off, normalize.
    total = sum(c.get("weight", 0) for c in criteria)
    if total == 0:
        # No valid criteria — use defaults
        return HackathonRubric(**_default_rubric(original_text), raw_text_hash="")
    
    if total < 50 or total > 200:
        # Points seem off — might be percentages instead of points, or scaled differently
        # Normalize to 100
        for c in criteria:
            c["weight"] = round(c["weight"] * 100 / total)
            c["max_points"] = c["weight"]
        total = 100
    
    # Validation 2: Every criterion must have a name and positive weight
    criteria = [c for c in criteria if c.get("name") and c.get("weight", 0) > 0]
    
    # Validation 3: Deduplicate criteria (LLM sometimes generates duplicates)
    seen_names = set()
    deduped = []
    for c in criteria:
        name_key = c["name"].lower().strip()
        if name_key not in seen_names:
            seen_names.add(name_key)
            deduped.append(c)
    criteria = deduped
    
    # Validation 4: Ensure required fields have defaults
    for c in criteria:
        c.setdefault("description", "")
        c.setdefault("top_score_description", "")
        c.setdefault("bottom_score_description", "")
        c.setdefault("scoring_levels", [])
    
    return HackathonRubric(
        competition_name=merged.get("competition_name", "Unknown"),
        total_points=sum(c["weight"] for c in criteria),
        criteria=criteria,
        submission_requirements=merged.get("submission_requirements", ["notebook", "writeup"]),
        writeup_template=merged.get("writeup_template", {"sections": [], "max_words": 2000}),
        data_policy=merged.get("data_policy", "any_public"),
        recommended_datasets=merged.get("recommended_datasets", []),
        deliverable_type=merged.get("deliverable_type", "notebook_and_writeup"),
        tracks=merged.get("tracks", []),
        prizes=merged.get("prizes", []),
        deadline=merged.get("deadline", "unknown"),
        raw_text_hash="",  # Set by caller
    )
```

---

### Function 2: build_effort_plan()

```python
def build_effort_plan(rubric: HackathonRubric) -> EffortPlan:
    """
    Map rubric criteria weights to concrete pipeline configuration.
    
    This is the function that makes Professor adapt to different hackathons.
    Same code handles Triagegeist (30% technical) and an art hackathon (15% technical)
    differently because it reads the weights and configures proportionally.
    """
    # Step 1: Classify each criterion into an effort dimension
    # Uses keyword matching on criterion names — not LLM, not fragile
    
    DIMENSION_KEYWORDS = {
        "technical": ["technical", "methodology", "code", "model", "quality", "reproducib", "implementation"],
        "novelty": ["novel", "impact", "creative", "original", "innovative", "fresh", "unique"],
        "documentation": ["document", "writeup", "write-up", "clarity", "presentation", "writing", "report"],
        "domain": ["clinical", "domain", "relevance", "problem", "real-world", "practical", "motivation"],
        "insight": ["insight", "finding", "result", "analysis", "discover", "interpret", "meaningful"],
    }
    
    dimension_weights = {dim: 0 for dim in DIMENSION_KEYWORDS}
    unclassified_weight = 0
    
    for criterion in rubric.criteria:
        name_lower = criterion["name"].lower()
        desc_lower = (criterion.get("description", "") or "").lower()
        search_text = name_lower + " " + desc_lower
        
        matched = False
        for dimension, keywords in DIMENSION_KEYWORDS.items():
            if any(kw in search_text for kw in keywords):
                dimension_weights[dimension] += criterion["weight"]
                matched = True
                break  # Each criterion maps to ONE dimension
        
        if not matched:
            # Can't classify — distribute to all dimensions equally
            unclassified_weight += criterion["weight"]
    
    # Distribute unclassified weight equally
    if unclassified_weight > 0:
        per_dim = unclassified_weight / len(DIMENSION_KEYWORDS)
        for dim in dimension_weights:
            dimension_weights[dim] += per_dim
    
    total_weight = sum(dimension_weights.values())
    if total_weight == 0:
        total_weight = 100  # Prevent division by zero
    
    # Step 2: Map dimension weights to effort levels
    tech_pct = dimension_weights["technical"] / total_weight
    novelty_pct = dimension_weights["novelty"] / total_weight
    doc_pct = dimension_weights["documentation"] / total_weight
    domain_pct = dimension_weights["domain"] / total_weight
    insight_pct = dimension_weights["insight"] / total_weight
    
    # Technical Quality → pipeline depth
    if tech_pct > 0.28:
        technical_depth = "marathon"
    elif tech_pct > 0.15:
        technical_depth = "standard"
    else:
        technical_depth = "sprint"
    
    # Domain + Novelty → thesis generation depth
    thesis_input = domain_pct + novelty_pct
    if thesis_input > 0.30:
        thesis_depth = "deep"
    elif thesis_input > 0.15:
        thesis_depth = "standard"
    else:
        thesis_depth = "light"
    
    # Documentation → writeup polish
    if doc_pct > 0.18:
        writeup_depth = "deep"
    elif doc_pct > 0.08:
        writeup_depth = "standard"
    else:
        writeup_depth = "light"
    
    # Novelty + Insight → external data acquisition priority
    data_input = novelty_pct + insight_pct
    if data_input > 0.20:
        external_data_priority = "high"
    elif data_input > 0.10:
        external_data_priority = "medium"
    else:
        external_data_priority = "skip"
    
    # Insight → visualization count
    if insight_pct > 0.14:
        visualization_count = 7
    elif insight_pct > 0.08:
        visualization_count = 5
    else:
        visualization_count = 3
    
    # Documentation → polish passes
    if doc_pct > 0.18:
        narrative_polish_passes = 3
    elif doc_pct > 0.08:
        narrative_polish_passes = 2
    else:
        narrative_polish_passes = 1
    
    # Step 3: Build reasoning string
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
```

---

### Function 3: run_rubric_parser() — LangGraph node

```python
def run_rubric_parser(state: ProfessorState) -> dict:
    """
    LangGraph node function. Runs after Competition Intel in hackathon mode.
    
    Reads: competition_description (from competition_intel or HITL paste)
    Writes: hackathon_rubric, hackathon_effort_plan, hackathon_mode,
            hackathon_writeup_template
    Emits: STATUS (parsing), CHECKPOINT (present rubric for operator approval)
    """
    from tools.operator_channel import emit_to_operator
    
    emit_to_operator("📋 Parsing hackathon rubric...", level="STATUS")
    
    # Get competition text
    # Source 1: competition_description from Competition Intel
    # Source 2: HITL injection (operator pasted the competition page)
    competition_text = state.competition_description
    
    if not competition_text or len(competition_text) < 100:
        # Not enough text — ask operator
        emit_to_operator(
            "📋 Competition description is too short for rubric parsing. "
            "Please paste the full competition page text (including rubric and submission requirements).",
            level="GATE",
        )
        # After GATE response, competition_text comes from HITL injection
        injections = [inj["text"] for inj in (state.hitl_injections or []) 
                      if len(inj["text"]) > 200]
        if injections:
            competition_text = injections[-1]  # Use the latest long injection
        else:
            # Still nothing — use defaults
            emit_to_operator("⚠️ No competition text available. Using default rubric.", level="STATUS")
            rubric = HackathonRubric(**_default_rubric(""), raw_text_hash="")
            effort = build_effort_plan(rubric)
            return _build_state_return(rubric, effort)
    
    # Parse the rubric
    rubric = parse_rubric(competition_text)
    
    # Build effort plan
    effort = build_effort_plan(rubric)
    
    # Present to operator for approval
    criteria_display = "\n".join([
        f"  {c['name']}: {c['weight']} pts ({c['weight']*100//rubric.total_points}%)"
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
    
    # If operator provides adjustments, apply them
    if response and response.strip() != "/continue":
        effort = _apply_operator_adjustments(effort, response)
        emit_to_operator(f"📋 Effort plan adjusted: {response[:100]}", level="STATUS")
    
    return _build_state_return(rubric, effort)


def _effort_for_criterion(criterion: dict, effort: EffortPlan) -> str:
    """Map a criterion to what effort dimension it drives."""
    name_lower = criterion["name"].lower()
    if any(kw in name_lower for kw in ["technical", "method", "code", "model"]):
        return f"{effort.technical_depth} pipeline"
    elif any(kw in name_lower for kw in ["novel", "impact", "creative"]):
        return f"{effort.thesis_depth} thesis + {effort.external_data_priority} data search"
    elif any(kw in name_lower for kw in ["document", "writeup", "clarity"]):
        return f"{effort.writeup_depth} writeup ({effort.narrative_polish_passes} passes)"
    elif any(kw in name_lower for kw in ["clinical", "domain", "relevance"]):
        return f"{effort.thesis_depth} thesis depth"
    elif any(kw in name_lower for kw in ["insight", "finding", "result"]):
        return f"{effort.visualization_count} visualizations"
    return "general effort"


def _apply_operator_adjustments(effort: EffortPlan, response: str) -> EffortPlan:
    """
    Parse operator's adjustment text and modify the effort plan.
    Simple keyword matching — not a full parser.
    """
    response_lower = response.lower()
    
    # Check for explicit depth overrides
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
    from dataclasses import asdict
    return {
        "hackathon_rubric": asdict(rubric),
        "hackathon_effort_plan": asdict(effort),
        "hackathon_mode": True,
        "hackathon_writeup_template": rubric.writeup_template,
    }
```

---

## COMMIT 2: State additions + registration

### State additions to graph/state.py

Add these fields to ProfessorState (all owned by rubric_parser):

```python
# ═══════════════════════════════════════════════
# HACKATHON MODE — owner: rubric_parser
# ═══════════════════════════════════════════════
hackathon_mode: bool = False
hackathon_rubric: dict = Field(default_factory=dict)
hackathon_effort_plan: dict = Field(default_factory=dict)
hackathon_writeup_template: dict = Field(default_factory=dict)
```

Update the `_FIELD_OWNERS` dict:
```python
"hackathon_mode": "rubric_parser",
"hackathon_rubric": "rubric_parser",
"hackathon_effort_plan": "rubric_parser",
"hackathon_writeup_template": "rubric_parser",
```

### Graph registration

In `graph/hackathon_builder.py` (or as a note for future graph building):
```python
graph.add_node("rubric_parser", run_rubric_parser)
graph.add_edge("competition_intel", "rubric_parser")
graph.add_edge("rubric_parser", "data_engineer")
```

---

## Contract Tests: tests/contracts/test_rubric_parser_contract.py

### Test fixtures

```python
TRIAGEGEIST_TEXT = """
Triagegeist
Predict emergency severity and optimize triage decisions with AI.

Scoring Rubric (100 points total)
1. Clinical Relevance (25 points)
Does the submission address a real and meaningful problem in emergency triage?
Score 21-25: Problem is sharply defined, clinically motivated.
Score 14-20: Problem is relevant but partially developed.
Score 7-13: Problem relates broadly to emergency medicine.
Score 0-6: Problem has negligible clinical relevance.

2. Technical Quality (30 points)
Is the AI approach sound? Is the code clean, reproducible?
Score 25-30: Methodology is rigorous. Code is clean.
Score 17-24: Methodology is reasonable with minor gaps.
Score 9-16: Methodology has notable weaknesses.
Score 0-8: Methodology is flawed.

3. Documentation and Writeup Quality (20 points)
Is the writeup clear, complete?
Score 17-20: Thorough and clear. Reproducibility supported.
Score 11-16: Covers most sections adequately.
Score 5-10: Incomplete or unclear.
Score 0-4: Absent or insufficient.

4. Insight and Findings (15 points)
Does the submission produce meaningful findings?
Score 13-15: Meaningful, clearly communicated.
Score 8-12: Reported but limited interpretation.
Score 3-7: Superficial or overclaimed.
Score 0-2: No meaningful findings.

5. Novelty and Impact Potential (10 points)
Does the submission bring a fresh perspective?
Score 9-10: Genuinely novel.
Score 6-8: Some novel elements.
Score 3-5: Follows established approaches.
Score 0-2: No novelty.

Submission Requirements:
- Kaggle Notebook (must run end-to-end, public)
- Project Writeup (max 2000 words)
- Cover Image (560 x 280 px)
- Project Link

Recommended datasets: MIMIC-IV-ED, NHAMCS

Prizes: 1st $5,000, 2nd $3,000, 3rd $2,000
"""

MINIMAL_HACKATHON_TEXT = """
Build something cool with this data.
Best projects win prizes.
Submit a notebook and writeup.
"""

ART_HACKATHON_TEXT = """
AI Art Challenge
Create novel AI-generated artwork.

Judging (100 points):
- Creativity & Originality: 40 points
- Technical Execution: 15 points  
- Presentation & Documentation: 30 points
- Impact & Meaning: 15 points

Submit: notebook, writeup, portfolio link
"""
```

### Tests — Parse correctness

```python
class TestRubricParsing:
    
    def test_triagegeist_extracts_5_criteria(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert len(rubric.criteria) == 5
    
    def test_triagegeist_total_100_points(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert rubric.total_points == 100
    
    def test_triagegeist_clinical_relevance_25(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        clinical = next(c for c in rubric.criteria if "clinical" in c["name"].lower())
        assert clinical["weight"] == 25
    
    def test_triagegeist_technical_quality_30(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        technical = next(c for c in rubric.criteria if "technical" in c["name"].lower())
        assert technical["weight"] == 30
    
    def test_triagegeist_novelty_10(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        novelty = next(c for c in rubric.criteria if "novel" in c["name"].lower())
        assert novelty["weight"] == 10
    
    def test_triagegeist_has_scoring_levels(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        # At least some criteria should have scoring levels extracted
        criteria_with_levels = [c for c in rubric.criteria if c.get("scoring_levels")]
        assert len(criteria_with_levels) >= 3
    
    def test_triagegeist_submission_requirements(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert "notebook" in rubric.submission_requirements
        assert "writeup" in rubric.submission_requirements
        assert "cover_image" in rubric.submission_requirements
        assert "project_link" in rubric.submission_requirements
    
    def test_triagegeist_word_limit(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert rubric.writeup_template.get("max_words") == 2000
    
    def test_triagegeist_recommended_datasets(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        dataset_names = [d["name"] for d in rubric.recommended_datasets]
        assert any("MIMIC" in n for n in dataset_names)
    
    def test_triagegeist_prizes(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert len(rubric.prizes) >= 2
        assert any("5000" in p.get("amount", "") or "5,000" in p.get("amount", "") for p in rubric.prizes)
    
    def test_triagegeist_data_policy(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert rubric.data_policy == "any_public"


class TestMinimalHackathon:
    
    def test_minimal_text_produces_valid_rubric(self):
        rubric = parse_rubric(MINIMAL_HACKATHON_TEXT)
        assert len(rubric.criteria) >= 3  # At least some criteria inferred
        assert rubric.total_points > 0
    
    def test_minimal_uses_defaults(self):
        rubric = parse_rubric(MINIMAL_HACKATHON_TEXT)
        # Should have default criteria since text has no rubric
        assert "notebook" in rubric.submission_requirements


class TestArtHackathon:
    
    def test_art_hackathon_creativity_40(self):
        rubric = parse_rubric(ART_HACKATHON_TEXT)
        creativity = next(c for c in rubric.criteria if "creativ" in c["name"].lower())
        assert creativity["weight"] == 40
    
    def test_art_hackathon_technical_15(self):
        rubric = parse_rubric(ART_HACKATHON_TEXT)
        technical = next(c for c in rubric.criteria if "technical" in c["name"].lower())
        assert technical["weight"] == 15
```

### Tests — Effort allocation

```python
class TestEffortPlan:
    
    def test_triagegeist_marathon_technical(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.technical_depth == "marathon"  # 30% technical weight
    
    def test_triagegeist_deep_thesis(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.thesis_depth == "deep"  # 25% domain + 10% novelty = 35%
    
    def test_triagegeist_deep_writeup(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.writeup_depth == "deep"  # 20% documentation
    
    def test_triagegeist_high_external_data(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.external_data_priority == "high"  # 10+15=25% novelty+insight
    
    def test_triagegeist_3_polish_passes(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.narrative_polish_passes == 3  # 20% documentation
    
    def test_art_hackathon_sprint_technical(self):
        rubric = parse_rubric(ART_HACKATHON_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.technical_depth == "sprint"  # 15% technical
    
    def test_art_hackathon_deep_thesis(self):
        rubric = parse_rubric(ART_HACKATHON_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.thesis_depth == "deep"  # 40% creativity/novelty
    
    def test_weight_breakdown_sums_correctly(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        total = sum(plan.weight_breakdown.values())
        assert 95 <= total <= 105  # Allow small rounding error
    
    def test_allocation_reasoning_nonempty(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert len(plan.allocation_reasoning) > 50
        assert "pipeline" in plan.allocation_reasoning.lower()
    
    def test_equal_weights_produce_balanced_plan(self):
        """When all criteria have equal weight, no dimension dominates."""
        rubric = HackathonRubric(
            competition_name="Test",
            total_points=100,
            criteria=[
                {"name": "Technical Quality", "weight": 20, "max_points": 20},
                {"name": "Novelty", "weight": 20, "max_points": 20},
                {"name": "Documentation", "weight": 20, "max_points": 20},
                {"name": "Domain Relevance", "weight": 20, "max_points": 20},
                {"name": "Insight", "weight": 20, "max_points": 20},
            ],
            submission_requirements=["notebook"],
            writeup_template={"sections": [], "max_words": 2000},
            data_policy="any_public",
            recommended_datasets=[],
            deliverable_type="notebook_and_writeup",
            tracks=[],
            prizes=[],
            deadline="unknown",
            raw_text_hash="test",
        )
        plan = build_effort_plan(rubric)
        assert plan.technical_depth == "standard"  # 20% — between thresholds
        assert plan.thesis_depth == "deep"  # 20+20=40% domain+novelty


class TestEdgeCases:
    
    def test_empty_text_returns_defaults(self):
        rubric = parse_rubric("")
        assert len(rubric.criteria) >= 3
        assert rubric.total_points > 0
    
    def test_nonsense_text_returns_defaults(self):
        rubric = parse_rubric("asdf 1234 !@#$ random garbage text")
        assert len(rubric.criteria) >= 3
    
    def test_rubric_hash_populated(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert len(rubric.raw_text_hash) == 16
    
    def test_different_text_different_hash(self):
        rubric1 = parse_rubric(TRIAGEGEIST_TEXT)
        rubric2 = parse_rubric(ART_HACKATHON_TEXT)
        assert rubric1.raw_text_hash != rubric2.raw_text_hash
    
    def test_criteria_deduplicated(self):
        """If LLM returns duplicate criteria, they're merged."""
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        names = [c["name"].lower() for c in rubric.criteria]
        assert len(names) == len(set(names))  # No duplicates
    
    def test_normalization_when_total_off(self):
        """If extracted points sum to 200 instead of 100, normalize."""
        # This tests the _build_rubric_from_merged validation
        merged = {
            "criteria": [
                {"name": "A", "weight": 60, "max_points": 60},
                {"name": "B", "weight": 80, "max_points": 80},
                {"name": "C", "weight": 60, "max_points": 60},
            ],
            "competition_name": "Test",
        }
        rubric = _build_rubric_from_merged(merged, "")
        # Weights should be normalized to ~100
        assert 95 <= rubric.total_points <= 105


class TestNodeFunction:
    
    def test_returns_correct_state_fields(self):
        """run_rubric_parser returns all required hackathon state fields."""
        # Mock state with competition_description
        state = ProfessorState(
            competition_description=TRIAGEGEIST_TEXT,
        )
        with patch("tools.operator_channel.emit_to_operator", return_value="/continue"):
            with patch("tools.llm_provider.llm_call") as mock_llm:
                mock_llm.return_value = {"text": "{}", "reasoning": "", 
                                         "input_tokens": 100, "output_tokens": 200,
                                         "model": "test", "cost_usd": 0.001}
                result = run_rubric_parser(state)
        
        assert "hackathon_mode" in result
        assert result["hackathon_mode"] == True
        assert "hackathon_rubric" in result
        assert "hackathon_effort_plan" in result
        assert "hackathon_writeup_template" in result
    
    def test_checkpoint_emitted(self):
        """Rubric analysis is presented to operator as CHECKPOINT."""
        state = ProfessorState(competition_description=TRIAGEGEIST_TEXT)
        with patch("tools.operator_channel.emit_to_operator") as mock_emit:
            mock_emit.return_value = "/continue"
            with patch("tools.llm_provider.llm_call") as mock_llm:
                mock_llm.return_value = {"text": "{}", "reasoning": "",
                                         "input_tokens": 100, "output_tokens": 200,
                                         "model": "test", "cost_usd": 0.001}
                run_rubric_parser(state)
            
            # Find the CHECKPOINT call
            checkpoint_calls = [
                c for c in mock_emit.call_args_list
                if c.kwargs.get("level") == "CHECKPOINT" or 
                   (len(c.args) > 1 and c.args[1] == "CHECKPOINT")
            ]
            assert len(checkpoint_calls) >= 1
    
    def test_deterministic_extraction_avoids_llm_when_complete(self):
        """When deterministic extraction gets everything, LLM is not called."""
        # Triagegeist has a clear structured rubric — deterministic should get the weights
        state = ProfessorState(competition_description=TRIAGEGEIST_TEXT)
        with patch("tools.operator_channel.emit_to_operator", return_value="/continue"):
            with patch("tools.llm_provider.llm_call") as mock_llm:
                mock_llm.return_value = {"text": "{}", "reasoning": "",
                                         "input_tokens": 100, "output_tokens": 200,
                                         "model": "test", "cost_usd": 0.001}
                result = run_rubric_parser(state)
                
                # LLM may or may not be called (depends on whether deterministic
                # extraction gets descriptions). But weights should be correct
                # regardless.
                rubric = result["hackathon_rubric"]
                weights = {c["name"]: c["weight"] for c in rubric["criteria"]}
                # The key weights should be correct from deterministic extraction
                assert any(w == 30 for w in weights.values())  # Technical Quality
                assert any(w == 25 for w in weights.values())  # Clinical Relevance
```

---

## WHAT NOT TO DO

- Do NOT hardcode competition-specific logic. The parser must work on ANY hackathon text.
- Do NOT use the LLM when deterministic extraction succeeds. Save the LLM call for filling gaps.
- Do NOT let parse_rubric() crash on ANY input. Empty string, garbage text, non-English — all must return a valid rubric (defaults if necessary).
- Do NOT let the total points be 0. If extraction fails completely, use 5 equal-weight criteria.
- Do NOT modify any traditional-mode files. This is an ADDITION.
- Do NOT use Pandas. If any data manipulation is needed, use Polars.
- Do NOT let the effort plan have invalid values. Every field must be one of its defined options.
- Do NOT skip the operator CHECKPOINT. The operator must see and approve the rubric interpretation before the pipeline continues.
- Do NOT use temperature > 0.2 for the LLM extraction call. We want accurate extraction, not creative interpretation.