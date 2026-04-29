import os
import json
import re
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple, Union

from core.state import ProfessorState
from tools.llm_provider import llm_call
from tools.sandbox import run_in_sandbox
from tools.operator_channel import emit_to_operator

logger = logging.getLogger(__name__)

# ── Part A: Visualizations ──────────────────────────────────────────────────

def generate_thesis_visualizations(state: ProfessorState) -> List[Dict]:
    """
    Generate the requested number of high-quality plots to support the thesis.
    """
    thesis = state.active_thesis
    if not thesis:
        return []
        
    effort = state.hackathon_effort_plan or {}
    plot_count = effort.get("visualization_count", 3)
    
    emit_to_operator(f"📊 Generating {plot_count} narrative visualizations...", level="STATUS")
    
    # 1. Plan plots using LLM
    planning_prompt = f"""Plan {plot_count} visualizations to support this hackathon thesis.

THESIS: "{thesis.get('statement', '')}"
CONDITION: {thesis.get('condition_variable', '')}
HYPOTHESIS: {thesis.get('hypothesis', '')}

AVAILABLE DATA: load from "{state.enriched_data_path or state.clean_data_path}"
DATA SCHEMA: {json.dumps(list((state.data_schema or {}).keys())[:40])}

Requirements:
1. Plots must DIRECTLY test or illustrate the conditional hypothesis.
2. Use Seaborn/Matplotlib.
3. Each plot must have a clear "Insight Goal" (what we want the viewer to see).
4. Plots should be publication-quality (titles, labels, legends).

Respond with ONLY valid JSON array:
[
    {{
        "title": "Plot Title",
        "type": "distribution" | "correlation" | "interaction" | "time_series",
        "insight_goal": "Prove that X is higher for population Y",
        "features_needed": ["feat1", "feat2"]
    }},
    ...
]
"""
    plan_response = llm_call(prompt=planning_prompt, temperature=0.2)
    plot_specs = _parse_plot_specs(plan_response)
    
    # 2. Execute each plot in sandbox
    session_id = state.session_id or "default"
    plot_dir = f"outputs/{session_id}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    generated_plots = []
    
    for i, spec in enumerate(plot_specs[:plot_count]):
        filename = f"plot_{i+1}_{spec['type']}.png"
        filepath = os.path.join(plot_dir, filename)
        
        execution_prompt = f"""Write Python code to create a visualization.

SPECIFICATION:
- Title: {spec['title']}
- Goal: {spec['insight_goal']}
- Output: "{filepath}"

DATA: load from "{state.enriched_data_path or state.clean_data_path}" with Polars.
COLUMNS TO USE: {spec['features_needed']}

Requirements:
1. Import polars, matplotlib.pyplot, seaborn.
2. Set theme: sns.set_theme(style="whitegrid").
3. Create the plot using high-leverage visualization (violin plots, heatmaps, faceted grids).
4. Save the plot to the specified path.
5. Print JSON: {{"success": true, "path": "{filepath}"}}
"""
        code_response = llm_call(prompt=execution_prompt, temperature=0.1)
        code = _extract_code(code_response)
        
        result = run_in_sandbox(
            code=code,
            agent_name="narrative_engine",
            purpose=f"Generate plot: {spec['title']}",
        )
        
        if result["success"]:
            # Check if file actually exists
            if _validate_plot_output(filepath):
                generated_plots.append({
                    "title": spec["title"],
                    "path": filepath,
                    "insight": spec["insight_goal"],
                    "type": spec["type"]
                })
                emit_to_operator(f"📊 Created: {spec['title']}", level="STATUS")
            else:
                logger.warning(f"Plot file not found after sandbox success: {filepath}")
        else:
            logger.error(f"Plot generation failed in sandbox: {result.get('stderr', '')[:100]}")

    return generated_plots


def _parse_plot_specs(response: str) -> List[Dict]:
    """Robust parsing of plot specifications."""
    t = response.strip()
    if "```json" in t: t = t.split("```json")[1].split("```")[0]
    elif "```" in t: t = t.split("```")[1].split("```")[0]
    t = re.sub(r',\s*}', '}', t.strip())
    t = re.sub(r',\s*]', ']', t)
    try:
        data = json.loads(t)
        return data if isinstance(data, list) else [data]
    except:
        # Fallback extraction logic
        objs = []
        depth, start = 0, None
        for i, c in enumerate(t):
            if c == '{':
                if depth == 0: start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try: objs.append(json.loads(t[start:i+1]))
                    except: pass
                    start = None
        return objs


def _validate_plot_output(filepath: str) -> bool:
    """Check if plot file exists and has non-zero size."""
    # Since sandbox might be on a different path or fake, we mock existence if in test
    # In production, we'd check real path or wait for return
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


def _extract_code(text: str) -> str:
    """Extract Python code from markdown."""
    if "```python" in text: return text.split("```python")[1].split("```")[0]
    if "```" in text: return text.split("```")[1].split("```")[0]
    return text


# ── Part B: Writeup ─────────────────────────────────────────────────────────

def generate_hackathon_writeup(state: ProfessorState) -> str:
    """
    Generate a winning solution writeup in Markdown.
    """
    thesis = state.active_thesis
    if not thesis:
        return "# Hackathon Writeup\n\nNo active thesis provided."
        
    emit_to_operator("✍️ Generating hackathon writeup...", level="STATUS")
    
    # 1. Collect all evidence
    evidence = _collect_evidence(state)
    
    # 2. Get template from state or default
    template = state.hackathon_writeup_template or {
        "sections": ["problem_statement", "methodology", "findings", "limitations"],
        "max_words": 2000
    }
    
    writeup_sections = []
    
    # 3. Generate each section using LLM
    for section_name in template.get("sections", []):
        emit_to_operator(f"✍️ Drafting: {section_name.replace('_', ' ')}", level="STATUS")
        section_text = _generate_section(section_name, thesis, evidence, template)
        writeup_sections.append(f"## {section_name.replace('_', ' ').title()}\n\n{section_text}\n")
        
    # 4. Final assembly
    header = f"# {state.competition_name or 'Hackathon'} Submission\n"
    header += f"**Thesis:** {thesis.get('statement')}\n\n"
    
    full_writeup = header + "\n".join(writeup_sections)
    
    # 5. Save
    session_id = state.session_id or "default"
    output_path = f"outputs/{session_id}/hackathon_writeup.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_writeup)
        
    return output_path


def _collect_evidence(state: ProfessorState) -> Dict:
    """Gather all data points to support the writeup."""
    from tools.code_ledger import get_reasoning_chain
    
    session_id = state.session_id or "default"
    session_dir = f"outputs/{session_id}"
    
    evidence = {
        "domain_info": state.domain_brief or {},
        "eda_insights": state.eda_insights_summary or "",
        "external_data": state.external_datasets or [],
        "reasoning_chain": get_reasoning_chain(session_dir) if os.path.exists(os.path.join(session_dir, "code_ledger.jsonl")) else [],
        "plots": state.narrative_plots or [],
        "cv_score": state.ensemble_cv_score or state.cv_mean or 0.0,
        "effect_sizes": state.thesis_effect_sizes or {},
        "effort_plan": state.hackathon_effort_plan or {}
    }
    
    return evidence


def _generate_section(section_name: str, thesis: dict, evidence: dict, template: dict) -> str:
    """Generate a single section of the writeup using LLM."""
    prompt = f"""Write the "{section_name}" section for a winning hackathon writeup.

THESIS: "{thesis.get('statement', '')}"
CONDITION: {thesis.get('condition_variable', '')}
HYPOTHESIS: {thesis.get('hypothesis', '')}

EVIDENCE COLLECTED:
- Domain Context: {json.dumps(evidence['domain_info'])}
- EDA: {evidence['eda_insights'][:1000]}
- External Data: {json.dumps(evidence['external_data'])}
- Statistical Findings: {json.dumps(evidence['effect_sizes'])}
- Models/CV: {evidence['cv_score']:.4f}

Requirements:
1. Tone: Professional, insight-driven, rigorous.
2. Focus: Connect the data findings directly to the THESIS.
3. Be specific: Quote effect sizes and metrics where available.
4. Reproducibility: Mention key methodology choices from the reasoning chain.
5. Target word count for this section: {template.get('max_words', 2000) // len(template.get('sections', [1]))} words.

Respond with the Markdown content of the section only.
"""
    try:
        section_text = llm_call(prompt=prompt, temperature=0.3)
        return section_text
    except Exception as e:
        logger.error(f"Failed to generate section {section_name}: {e}")
        return f"[Section generation failed: {e}]"
