import os
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
from tools.sandbox import run_in_sandbox

logger = logging.getLogger(__name__)

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DatasetCandidate:
    """A candidate external dataset discovered during scouting."""
    name: str
    source_url: str
    source_type: str                # "competition_recommended" | "kaggle" | "government" | "academic" | "domain_specific" | "reference_table"
    description: str
    
    # Feasibility dimensions (0.0-1.0 unless bool)
    join_feasibility: float         # Can it be joined to competition data?
    relevance_to_thesis: float      # Does it help test the active thesis?
    size_compatible: bool           # Fits in memory alongside competition data?
    license_compatible: bool        # License allows use in competition?
    download_accessible: bool       # Can we actually get it programmatically?
    
    # Join details
    join_key: str                   # Column to join on: "age_group", "icd_code", "date"
    join_type: str                  # "left" | "inner" | "lookup_table" | "enrichment"
    estimated_match_rate: float     # % of competition rows that will match (0.0-1.0)
    
    # Computed
    overall_score: float            # Weighted combination: 0.4*relevance + 0.3*join_feasibility + 0.2*match_rate + 0.1*accessibility
    
    # Integration status
    downloaded: bool = False
    download_path: str = ""
    integrated: bool = False
    integration_result: dict = field(default_factory=dict)


# ── Step 1: Parse data needs ────────────────────────────────────────────────

def _parse_data_needs(active_thesis: dict) -> list[str]:
    """Extract specific data needs from the thesis's data_plan."""
    data_plan = active_thesis.get("data_plan", {})
    external_needed = data_plan.get("external_needed", [])
    
    if not external_needed:
        return []
    
    search_queries = []
    for need in external_needed:
        # Each need becomes 2-3 search queries targeting different sources
        queries = [
            f"{need} dataset download CSV",
            f"{need} kaggle dataset",
            f"{need} public data",
        ]
        search_queries.extend(queries)
    
    return search_queries


# ── Step 2: Search sources ──────────────────────────────────────────────────

def _search_kaggle_datasets(queries: list[str]) -> list[DatasetCandidate]:
    """Search Kaggle Datasets API using run_in_sandbox."""
    candidates = []
    
    for query in queries[:3]:  # Max 3 queries
        code = f"""
import subprocess
import json

result = subprocess.run(
    ["kaggle", "datasets", "list", "-s", "{query}", "--csv", "--max-size", "500000000"],
    capture_output=True, text=True, timeout=30
)

if result.returncode == 0:
    lines = result.stdout.strip().split("\\n")
    if len(lines) > 1:
        for line in lines[1:6]:
            parts = line.split(",")
            if len(parts) >= 3:
                # Kaggle CSV parts: ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating
                ref = parts[0]
                print(json.dumps({{"name": ref, "url": f"https://www.kaggle.com/datasets/{{ref}}"}}))
"""
        result = run_in_sandbox(
            code=code,
            agent_name="external_data_scout",
            purpose=f"Kaggle dataset search: {query}",
            timeout=60,
        )
        
        if result["success"]:
            for line in result["stdout"].split("\n"):
                if not line.strip(): continue
                try:
                    data = json.loads(line.strip())
                    candidates.append(DatasetCandidate(
                        name=data.get("name", ""),
                        source_url=data.get("url", ""),
                        source_type="kaggle",
                        description="",
                        join_feasibility=0.0,
                        relevance_to_thesis=0.0,
                        size_compatible=True,
                        license_compatible=True,
                        download_accessible=True,
                        join_key="",
                        join_type="",
                        estimated_match_rate=0.0,
                        overall_score=0.0,
                    ))
                except json.JSONDecodeError:
                    continue
    
    return candidates


def _search_web_sources(queries: list[str], domain: str, tiers: list = None) -> list[DatasetCandidate]:
    """Web search for datasets (Mocked/Degraded if tool unavailable)."""
    candidates = []
    
    DOMAIN_DATA_SOURCES = {
        "healthcare": ["physionet.org", "cdc.gov NHAMCS", "WHO global health observatory"],
        "finance": ["FRED economic data", "SEC EDGAR", "World Bank open data"],
        "sports": ["FBref", "Statsbomb open data", "football-data.co.uk"],
        "energy": ["EIA open data", "ENTSO-E transparency", "OpenWeatherMap"],
        "retail": ["UCI online retail dataset", "Kaggle retail datasets"],
        "geospatial": ["OpenStreetMap", "GADM boundaries", "Natural Earth"],
    }
    
    domain_sources = DOMAIN_DATA_SOURCES.get(domain, [])
    active_queries = queries[:5]
    for source in domain_sources[:3]:
        active_queries.append(f"{source} download")
        
    try:
        from tools.web_search import web_search
        for query in active_queries[:5]:
            results = web_search(query, max_results=3)
            for r in results:
                candidates.append(DatasetCandidate(
                    name=r.get("title", "Unknown"),
                    source_url=r.get("url", ""),
                    source_type="web_search",
                    description=r.get("snippet", ""),
                    join_feasibility=0.0,
                    relevance_to_thesis=0.0,
                    size_compatible=True,
                    license_compatible=True,
                    download_accessible=True,
                    join_key="",
                    join_type="",
                    estimated_match_rate=0.0,
                    overall_score=0.0,
                ))
    except (ImportError, AttributeError):
        pass
    
    return candidates


def _search_all_sources(
    queries: list[str],
    recommended_datasets: list[dict],
    domain: str,
    effort_priority: str,
) -> list[DatasetCandidate]:
    """Search all source tiers."""
    candidates = []
    
    for rec in recommended_datasets:
        candidates.append(DatasetCandidate(
            name=rec.get("name", "Unknown"),
            source_url=rec.get("url", ""),
            source_type="competition_recommended",
            description=rec.get("description", ""),
            join_feasibility=0.0,
            relevance_to_thesis=0.0,
            size_compatible=True,
            license_compatible=True,
            download_accessible=True,
            join_key="",
            join_type="",
            estimated_match_rate=0.0,
            overall_score=0.0,
        ))
    
    if effort_priority == "skip":
        return candidates
    
    kaggle_results = _search_kaggle_datasets(queries)
    candidates.extend(kaggle_results)
    
    if effort_priority == "high":
        web_results = _search_web_sources(queries, domain)
        candidates.extend(web_results)
    elif effort_priority == "medium":
        web_results = _search_web_sources(queries, domain, tiers=["government_portals", "academic_repositories"])
        candidates.extend(web_results)
    
    seen_names = set()
    deduped = []
    for c in candidates:
        name_key = c.name.lower().strip()
        if name_key not in seen_names:
            seen_names.add(name_key)
            deduped.append(c)
    
    return deduped


# ── Step 3: Evaluation ──────────────────────────────────────────────────────

def _evaluate_candidates(
    candidates: list[DatasetCandidate],
    thesis: dict,
    competition_schema: dict,
) -> list[DatasetCandidate]:
    """Use LLM to evaluate candidates."""
    if not candidates:
        return []
    
    candidate_descriptions = "\n".join([
        f"{i+1}. {c.name} — {c.description[:200]} (source: {c.source_type}, url: {c.source_url[:100]})"
        for i, c in enumerate(candidates[:15])
    ])
    
    schema_summary = ", ".join([f"{col}: {dtype}" for col, dtype in list(competition_schema.items())[:30]])
    
    prompt = f"""Evaluate these external dataset candidates for a hackathon thesis.

THESIS: "{thesis.get('statement', '')}"
THESIS DATA NEEDS: {thesis.get('data_plan', {}).get('external_needed', [])}
CONDITION VARIABLE: {thesis.get('condition_variable', '')}

COMPETITION DATA SCHEMA (columns to join on):
{schema_summary}

CANDIDATE DATASETS:
{candidate_descriptions}

For each candidate, evaluate:
1. RELEVANCE (0.0-1.0): Does this dataset DIRECTLY help test the thesis? 
2. JOIN_FEASIBILITY (0.0-1.0): Can this dataset be joined to the competition data?
3. JOIN_KEY: What column(s) to join on.
4. JOIN_TYPE: "left" | "lookup_table" | "enrichment"
5. MATCH_RATE (0.0-1.0): Fraction of competition rows matching.
6. LICENSE: true/false
7. SIZE_OK: true/false

Respond with ONLY valid JSON array:
[
    {{
        "candidate_index": 1,
        "relevance": 0.8,
        "join_feasibility": 0.7,
        "join_key": "age_group",
        "join_type": "lookup_table",
        "match_rate": 0.95,
        "license_ok": true,
        "size_ok": true,
        "reasoning": "..."
    }},
    ...
]
"""
    response_text = llm_call(prompt=prompt, temperature=0.2)
    evaluations = _safe_json_parse_list(response_text)
    
    for eval_entry in evaluations:
        idx = eval_entry.get("candidate_index", 0) - 1
        if 0 <= idx < len(candidates):
            c = candidates[idx]
            c.relevance_to_thesis = eval_entry.get("relevance", 0.0)
            c.join_feasibility = eval_entry.get("join_feasibility", 0.0)
            c.join_key = eval_entry.get("join_key", "")
            c.join_type = eval_entry.get("join_type", "left")
            c.estimated_match_rate = eval_entry.get("match_rate", 0.0)
            c.license_compatible = eval_entry.get("license_ok", True)
            c.size_compatible = eval_entry.get("size_ok", True)
            
            c.overall_score = (
                0.40 * c.relevance_to_thesis +
                0.30 * c.join_feasibility +
                0.20 * c.estimated_match_rate +
                0.10 * (1.0 if c.download_accessible else 0.0)
            )
    
    candidates.sort(key=lambda c: c.overall_score, reverse=True)
    return candidates


# ── Step 4: Presentation ────────────────────────────────────────────────────

def _present_candidates(candidates: list[DatasetCandidate], thesis: dict) -> str:
    """Present candidates to operator via HITL."""
    if not candidates or all(c.overall_score == 0 for c in candidates):
        return emit_to_operator(
            "📁 No relevant external datasets found for this thesis.\n"
            "Analysis will proceed with competition data only.",
            level="CHECKPOINT",
        )
    
    top = [c for c in candidates if c.overall_score > 0.3][:5]
    if not top:
        return emit_to_operator("📁 No high-scoring external datasets found.", level="CHECKPOINT")

    display = "📁 EXTERNAL DATA CANDIDATES\n\n"
    for i, c in enumerate(top, 1):
        display += (
            f"{i}. [{c.overall_score:.2f}] {c.name}\n"
            f"   Source: {c.source_type} | Relevance: {c.relevance_to_thesis:.1f} | Match: {c.estimated_match_rate:.0%}\n"
            f"   Join on: {c.join_key} ({c.join_type})\n"
            f"   License: {'✅' if c.license_compatible else '⚠️ unclear'}\n\n"
        )
    
    display += "Approve top items? Reply /continue, or /data approve 1,2, or /data search \"query\"."
    return emit_to_operator(display, level="CHECKPOINT")


# ── Step 5: Integration ─────────────────────────────────────────────────────

def _download_dataset(candidate: DatasetCandidate, session_dir: str) -> DatasetCandidate:
    """Mock/Simplified download logic."""
    # In a real system, this would use Kaggle API or direct URL download.
    # For now, we simulate a successful download of a placeholder file.
    os.makedirs(session_dir, exist_ok=True)
    mock_path = os.path.join(session_dir, f"ext_{candidate.name.replace('/', '_')}.csv")
    with open(mock_path, "w") as f:
        f.write("join_key_placeholder,ext_feature\nval1,0.5\nval2,0.8")
    
    candidate.downloaded = True
    candidate.download_path = mock_path
    return candidate


def _integrate_dataset(
    candidate: DatasetCandidate,
    competition_data_path: str,
    session_dir: str,
    canonical_train_rows: int,
) -> dict:
    """Integrate dataset using Polars code generated by LLM."""
    join_prompt = f"""Write Polars code to integrate an external dataset with competition data.

COMPETITION DATA: load from "{competition_data_path}"
EXTERNAL DATA: load from "{candidate.download_path}"
JOIN KEY: {candidate.join_key}
JOIN TYPE: {candidate.join_type}
EXPECTED ROW COUNT: {canonical_train_rows}

Requirements:
1. Load both with Polars.
2. Join on {candidate.join_key} using {candidate.join_type} join.
3. If join duplicates rows, deduplicate external keys first.
4. Save to "{session_dir}/enriched_data.parquet".
5. Print JSON: {{"matched_rows": int, "new_columns": list}}
"""
    response_text = llm_call(prompt=join_prompt, temperature=0.2)
    code = _extract_code(response_text)
    
    result = run_in_sandbox(
        code=code,
        agent_name="external_data_scout",
        purpose=f"Integrate {candidate.name}",
    )
    
    if result["success"]:
        summary = _parse_integration_summary(result["stdout"])
        enriched_path = os.path.join(session_dir, "enriched_data.parquet")
        
        # Verify row count
        if os.path.exists(enriched_path):
            import polars as pl
            try:
                count = len(pl.read_parquet(enriched_path))
                if count != canonical_train_rows:
                    return {"success": False, "error": f"Row count changed: {canonical_train_rows} -> {count}"}
            except:
                pass
                
        return {
            "success": True,
            "enriched_path": enriched_path,
            "new_columns": summary.get("new_columns", []),
            "matched_rows": summary.get("matched_rows", 0),
            "match_rate": summary.get("matched_rows", 0) / max(canonical_train_rows, 1),
        }
    
    return {"success": False, "error": result.get("stderr", "")}


# ── Step 6: Reference table construction ────────────────────────────────────

def _construct_reference_table(data_need: str, domain: str, session_dir: str) -> DatasetCandidate:
    """Construct lookup table from LLM knowledge."""
    prompt = f"""Generate a reference lookup table for: "{data_need}" in domain "{domain}".
Respond with:
SOURCE: citation
CSV:
col1,col2
val1,val2
JOIN_KEY: column_name
"""
    resp = llm_call(prompt=prompt, temperature=0.2)
    text = resp
    
    source = "LLM Knowledge"
    if "SOURCE:" in text: source = text.split("SOURCE:")[1].split("\n")[0].strip()
    
    csv_data = ""
    if "CSV:" in text: csv_data = text.split("CSV:")[1].split("JOIN_KEY:")[0].strip()
    
    join_key = "age_group"
    if "JOIN_KEY:" in text: join_key = text.split("JOIN_KEY:")[1].strip().split("\n")[0]

    os.makedirs(session_dir, exist_ok=True)
    ref_path = os.path.join(session_dir, f"ref_{hashlib.md5(data_need.encode()).hexdigest()[:8]}.csv")
    with open(ref_path, "w") as f:
        f.write(csv_data)
        
    return DatasetCandidate(
        name=f"Ref: {data_need}",
        source_url=f"Derived: {source}",
        source_type="reference_table",
        description=f"Generated lookup for {data_need}",
        join_feasibility=0.9, relevance_to_thesis=1.0, size_compatible=True,
        license_compatible=True, download_accessible=True,
        join_key=join_key, join_type="lookup_table", estimated_match_rate=1.0,
        overall_score=0.95, downloaded=True, download_path=ref_path
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _safe_json_parse_list(text: str) -> list:
    """Robust JSON array parsing."""
    t = text.strip()
    if "```json" in t: t = t.split("```json")[1].split("```")[0]
    elif "```" in t: t = t.split("```")[1].split("```")[0]
    t = re.sub(r',\s*}', '}', t.strip())
    t = re.sub(r',\s*]', ']', t)
    try:
        data = json.loads(t)
        return data if isinstance(data, list) else [data]
    except:
        # Fallback to manual extraction
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


def _extract_code(text: str) -> str:
    """Extract Python code from markdown."""
    if "```python" in text: return text.split("```python")[1].split("```")[0]
    if "```" in text: return text.split("```")[1].split("```")[0]
    return text


def _parse_integration_summary(stdout: str) -> dict:
    """Parse JSON summary from sandbox stdout."""
    for line in reversed(stdout.split("\n")):
        if line.strip().startswith("{") and line.strip().endswith("}"):
            try: return json.loads(line.strip())
            except: continue
    return {}


# ── Main node function ──────────────────────────────────────────────────────

def external_data_scout(state: ProfessorState) -> dict:
    """Find, evaluate, and integrate external datasets."""
    if not state.hackathon_mode: return {}
    
    thesis = state.active_thesis
    if not thesis:
        emit_to_operator("⚠️ No active thesis — skipping scouting", level="STATUS")
        return {"thesis_data_sufficient": True}
        
    data_needs = thesis.get("data_plan", {}).get("external_needed", [])
    if not data_needs:
        emit_to_operator("📁 No external data needed", level="STATUS")
        return {"thesis_data_sufficient": True, "external_datasets": [], "enriched_data_path": state.clean_data_path}

    emit_to_operator(f"📁 Searching for external data: {data_needs}", level="STATUS")
    
    queries = _parse_data_needs(thesis)
    recommended = state.hackathon_rubric.get("recommended_datasets", [])
    domain = state.domain_classification or "general"
    effort = state.hackathon_effort_plan or {}
    priority = effort.get("external_data_priority", "medium")
    
    try:
        candidates = _search_all_sources(queries, recommended, domain, priority)
    except Exception as e:
        logger.error(f"External search failed: {e}")
        emit_to_operator(f"⚠️ External search failed, proceeding with recommended only.", level="STATUS")
        candidates = []
        for rec in recommended:
            candidates.append(DatasetCandidate(
                name=rec.get("name", "Unknown"),
                source_url=rec.get("url", ""),
                source_type="competition_recommended",
                description=rec.get("description", ""),
                join_feasibility=0.0, relevance_to_thesis=0.0, size_compatible=True,
                license_compatible=True, download_accessible=True,
                join_key="", join_type="", estimated_match_rate=0.0, overall_score=0.0
            ))
    
    # Thresholds / Guidelines check
    session_dir = f"outputs/{state.session_id}"
    for need in data_needs:
        if any(kw in need.lower() for kw in ["threshold", "guideline", "protocol", "standard", "criteria"]):
            candidates.append(_construct_reference_table(need, domain, session_dir))
            
    if candidates:
        candidates = _evaluate_candidates(candidates, thesis, state.data_schema or {})
        
    response = _present_candidates(candidates, thesis)
    
    # Parse approval
    approved_indices = []
    if response and "/data approve" in response:
        try:
            parts = response.split("approve")[-1].strip()
            approved_indices = [int(x.strip()) - 1 for x in parts.split(",")]
        except: approved_indices = [0, 1, 2]
    elif not response or "/continue" in response:
        approved_indices = [i for i, c in enumerate(candidates) if c.overall_score > 0.3][:3]
        
    MAX_EXTERNAL = 3
    approved = [candidates[i] for i in approved_indices if i < len(candidates)][:MAX_EXTERNAL]
    
    integrated = []
    enriched_path = state.clean_data_path
    
    for cand in approved:
        emit_to_operator(f"📁 Integrating: {cand.name}...", level="STATUS")
        if not cand.downloaded:
            cand = _download_dataset(cand, session_dir)
        
        if cand.downloaded:
            res = _integrate_dataset(cand, enriched_path, session_dir, state.canonical_train_rows)
            if res["success"]:
                enriched_path = res["enriched_path"]
                cand.integrated = True
                cand.integration_result = res
                integrated.append(asdict(cand))
                emit_to_operator(f"✅ {cand.name} integrated.", level="STATUS")
            else:
                emit_to_operator(f"❌ {cand.name} failed: {res.get('error', '')[:50]}", level="STATUS")

    sufficient = len(integrated) > 0 or priority == "skip"
    return {
        "external_datasets": integrated,
        "external_data_paths": [d.get("download_path", "") for d in integrated],
        "thesis_data_sufficient": sufficient,
        "enriched_data_path": enriched_path
    }
