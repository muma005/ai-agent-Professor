# agents/feature_factory.py

import os
import json
import logging
import numpy as np
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from core.state import ProfessorState
from tools.llm_provider import llm_call, _safe_json_loads
from tools.sandbox import run_in_sandbox
from tools.adaptive_gater import run_adaptive_gate
from core.lineage import log_event
from guards.agent_retry import with_agent_retry
from tools.performance_monitor import timed_node

logger = logging.getLogger(__name__)

AGENT_NAME = "feature_factory"

# ── Feature Engineering System Prompt ────────────────────────────────────────

FACTORY_SYSTEM_PROMPT = """You are a Kaggle Grandmaster specializing in Python/Polars feature engineering.
Generate high-quality feature code that transforms raw data into strong signals.

RULES:
1. Output ONLY a valid Python code block using Polars (pl).
2. Use the variable 'df' as the input and output (df = df.with_columns(...)).
3. Do NOT include imports — they are handled by the sandbox.
4. Keep logic efficient. Use vectorized Polars operations.
5. Focus on the hypotheses provided in the context.
"""

# ── Hackathon Additions ──────────────────────────────────────────────────

def _build_hackathon_feature_prompt(
    state: ProfessorState,
    round_num: int,
    existing_feature_names: List[str],
) -> str:
    """
    Modified feature generation prompt for hackathon mode.
    """
    thesis = state.active_thesis
    if not thesis:
        return "" # Should not happen in hackathon mode
    
    data_path = state.enriched_data_path or state.feature_data_path or state.clean_data_path
    
    external_info = ""
    if state.external_datasets:
        external_info = "EXTERNAL DATA INTEGRATED:\n"
        for ds in state.external_datasets:
            external_info += f"  - {ds.get('name', '?')}: joined on {ds.get('join_key', '?')}, "
            external_info += f"{ds.get('integration_result', {}).get('new_columns', ['?'])} new columns\n"
    
    domain_recipes = ""
    if state.domain_brief and state.domain_brief.get("feature_recipes"):
        recipes = state.domain_brief["feature_recipes"][:5]
        domain_recipes = "DOMAIN FEATURE RECIPES (from domain research):\n"
        for r in recipes:
            domain_recipes += f"  - {r.get('name', '?')}: {r.get('formula', '?')} — {r.get('rationale', '')[:100]}\n"
    
    round_directions = {
        1: "Focus on CONDITION features — create the binary/categorical splits that define your comparison groups.",
        2: "Focus on OUTCOME and DELTA features — compute the target-related metric within each group and the gap between groups.",
        3: "Focus on MODERATOR features — cross the condition with other variables to find what amplifies or dampens the effect.",
        4: "Focus on REFINEMENT — compute second-order interactions, subgroup-specific deltas, and edge-case features.",
        5: "Focus on ROBUSTNESS — alternative operationalizations of the condition, sensitivity checks, complementary metrics.",
    }
    round_direction = round_directions.get(round_num, round_directions[3])

    prompt = f"""You are a senior data scientist testing a specific thesis for a hackathon competition.

THESIS: "{thesis.get('statement', '')}"
HYPOTHESIS: "{thesis.get('hypothesis', '')}"
CONDITION VARIABLE: {thesis.get('condition_variable', '')}
TARGET AUDIENCE: {thesis.get('target_audience', '')}

ROUND {round_num} DIRECTION: {round_direction}

Generate features in these 4 categories:

1. CONDITION FEATURES — identify WHEN/WHERE the thesis condition applies
   Create binary or categorical variables that split the data into 
   "condition present" vs "condition absent" groups.
   The condition variable is: {thesis.get('condition_variable', '')}
   
   Examples for this thesis:
   - Binary flag: is_condition_present (1 if condition applies, 0 otherwise)
   - Categorical: condition_subgroup (fine-grained split of the condition)
   - Derived: condition_severity (continuous measure of how strongly the condition applies)

2. OUTCOME FEATURES — measure the outcome WITHIN each condition group
   Compute the target-related metric separately for each condition.
   These features capture what happens INSIDE each group.
   
   Examples:
   - outcome_when_condition_present (aggregate target within condition=1 group)
   - outcome_when_condition_absent (aggregate target within condition=0 group)
   - rate_metric_by_condition (rate or proportion per group)

3. DELTA FEATURES — quantify the DIFFERENCE between groups
   This is the feature that DIRECTLY measures the thesis.
   A large, statistically significant delta PROVES the thesis.
   
   Examples:
   - condition_gap = outcome_present - outcome_absent
   - condition_ratio = outcome_present / (outcome_absent + 1e-8)
   - relative_change = (outcome_present - outcome_absent) / (outcome_absent + 1e-8)

4. MODERATOR FEATURES — find what makes the effect STRONGER or WEAKER
   Cross the condition with other variables to discover moderators.
   
   Examples:
   - condition_gap × moderator_variable (interaction)
   - condition_gap_by_subgroup (delta computed within subgroups)
   - is_high_risk_subgroup (binary flag for subgroups where the effect is strongest)

DATA: load from "{data_path}"
TARGET COLUMN: {state.target_col}
DATA SCHEMA: {json.dumps(dict(list((state.data_schema or {}).items())[:30]), indent=2)}

{external_info}

{domain_recipes}

EDA INSIGHTS: {state.eda_insights_summary[:500] if state.eda_insights_summary else 'Not available'}

EXISTING FEATURES (DO NOT duplicate):
{chr(10).join(f'  - {name}' for name in existing_feature_names[:30])}
{'  ... and ' + str(len(existing_feature_names) - 30) + ' more' if len(existing_feature_names) > 30 else ''}

CONSTRAINTS:
1. Use Polars ONLY (import polars as pl), NEVER Pandas
2. Do NOT use .apply() or .map_elements() — use vectorized expressions
3. Do NOT access the target column in test data
4. Pin random seeds to 42
5. All new column names must be unique and descriptive
6. Handle null values explicitly
7. Output dataframe must have EXACTLY {state.canonical_train_rows} rows
8. Each feature must have a CATEGORY tag in its name or a comment:
   # CATEGORY: condition | outcome | delta | moderator
9. For .over() group operations, ensure you're computing within the TRAINING data only

Return ONLY valid Python code. No markdown fences.
"""
    return prompt

def _hackathon_gate(
    feature_values: np.ndarray,
    condition_values: np.ndarray,
    gate_config: dict,
) -> Tuple[bool, float, float, dict]:
    """
    Hackathon-mode statistical gate.
    """
    from scipy.stats import mannwhitneyu
    
    # Split by condition
    mask_a = (condition_values == 1)
    mask_b = (condition_values == 0)
    
    group_a = feature_values[mask_a]
    group_b = feature_values[mask_b]
    
    # Minimum group size check
    MIN_GROUP_SIZE = 30
    if len(group_a) < MIN_GROUP_SIZE or len(group_b) < MIN_GROUP_SIZE:
        return False, 1.0, 0.0, {
            "reason": f"Insufficient group sizes: group_a={len(group_a)}, group_b={len(group_b)} (min={MIN_GROUP_SIZE})",
            "group_a_n": int(len(group_a)),
            "group_b_n": int(len(group_b)),
        }
    
    # Drop NaN
    group_a = group_a[~np.isnan(group_a)]
    group_b = group_b[~np.isnan(group_b)]
    
    if len(group_a) < MIN_GROUP_SIZE or len(group_b) < MIN_GROUP_SIZE:
        return False, 1.0, 0.0, {
            "reason": f"Insufficient non-null values: group_a={len(group_a)}, group_b={len(group_b)}",
            "group_a_n": int(len(group_a)),
            "group_b_n": int(len(group_b)),
        }
    
    # Mann-Whitney U test
    try:
        stat, p_value = mannwhitneyu(group_a, group_b, alternative="two-sided")
    except ValueError as e:
        return False, 1.0, 0.0, {"reason": f"Mann-Whitney failed: {str(e)}"}
    
    # Cohen's d
    pooled_std = np.sqrt((np.var(group_a) + np.var(group_b)) / 2)
    if pooled_std < 1e-10:
        effect_size = 0.0
    else:
        effect_size = (np.mean(group_a) - np.mean(group_b)) / pooled_std
    
    p_threshold = gate_config.get("wilcoxon_p", 0.05)
    MINIMUM_EFFECT_SIZE = 0.2
    
    passed = (p_value < p_threshold) and (abs(effect_size) > MINIMUM_EFFECT_SIZE)
    
    effect_details = {
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "effect_magnitude": (
            "large" if abs(effect_size) > 0.8 else
            "medium" if abs(effect_size) > 0.5 else
            "small" if abs(effect_size) > 0.2 else
            "negligible"
        ),
        "group_a_mean": float(np.mean(group_a)),
        "group_b_mean": float(np.mean(group_b)),
        "group_a_std": float(np.std(group_a)),
        "group_b_std": float(np.std(group_b)),
        "group_a_n": int(len(group_a)),
        "group_b_n": int(len(group_b)),
        "group_a_median": float(np.median(group_a)),
        "group_b_median": float(np.median(group_b)),
        "direction": "group_a > group_b" if np.mean(group_a) > np.mean(group_b) else "group_b > group_a",
        "mann_whitney_U": float(stat),
    }
    
    return passed, float(p_value), float(effect_size), effect_details

def _get_condition_values(
    df,  # Polars DataFrame
    thesis: dict,
    feature_manifest: list,
) -> np.ndarray:
    """
    Get the binary condition split for hackathon gate testing.
    """
    import polars as pl
    
    # 1. Check manifest for explicit 'condition' category
    condition_features = [f for f in feature_manifest if f.get("category") == "condition"]
    if condition_features:
        col = condition_features[0]["name"]
        if col in df.columns:
            vals = df[col].to_numpy()
            non_nan = vals[~np.isnan(vals)]
            if len(np.unique(non_nan)) <= 2:
                return (vals == np.nanmax(vals)).astype(int)
            else:
                return (vals > np.nanmedian(vals)).astype(int)

    # 2. Derive from condition_variable name
    cond_var = thesis.get("condition_variable", "").lower()
    for col in df.columns:
        if col.lower() in cond_var or cond_var in col.lower():
            vals = df[col].to_numpy()
            if df[col].dtype in [pl.Utf8, pl.Categorical]:
                modes = df[col].mode()
                most_common = modes[0] if len(modes) > 0 else None
                return (df[col] == most_common).to_numpy().astype(int)
            else:
                return (vals > np.nanmedian(vals)).astype(int)

    # 3. Fallback random split
    np.random.seed(42)
    return np.random.randint(0, 2, size=len(df))

def _extract_feature_categories(code: str, feature_names: list) -> dict:
    """
    Parse CATEGORY tags from code or infer from naming.
    """
    categories = {}
    lines = code.split("\n")
    current_category = "uncategorized"
    
    # Pass 1: Tag-based
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# CATEGORY:"):
            cat = stripped.replace("# CATEGORY:", "").strip().lower()
            if cat in ("condition", "outcome", "delta", "moderator"):
                current_category = cat
        
        if ".alias(" in stripped:
            match = re.search(r'\.alias\(["\']([^"\']+)["\']\)', stripped)
            if match:
                fname = match.group(1)
                if fname in feature_names and current_category != "uncategorized":
                    categories[fname] = current_category

    # Pass 2: Fallback Naming
    for fname in feature_names:
        if fname not in categories:
            fnl = fname.lower()
            if any(p in fnl for p in ["senior", "elderly", "flag", "is_", "condition", "split"]):
                categories[fname] = "condition"
            elif any(p in fnl for p in ["gap", "delta", "diff", "ratio", "change", "severity", "error"]):
                categories[fname] = "delta"
            elif any(p in fnl for p in ["moderator", "interaction", "_x_"]):
                categories[fname] = "moderator"
            elif any(p in fnl for p in ["rate", "accuracy", "outcome", "mean", "count"]):
                categories[fname] = "outcome"
            else:
                categories[fname] = "uncategorized"
    return categories

# ── Core Logic ──────────────────────────────────────────────────────────────

def _run_feature_round(
    state: ProfessorState, 
    round_num: int, 
    hypotheses: List[Dict],
    ledger: List[Dict]
) -> Tuple[bool, str, Dict]:
    """Single round of feature generation and sandbox validation."""
    
    # Check if hackathon mode
    if state.hackathon_mode:
        prompt = _build_hackathon_feature_prompt(state, round_num, [e.get("name", "") for e in ledger])
    else:
        prompt = f"""ROUND {round_num} of Feature Engineering.
Hypotheses to implement:
{json.dumps(hypotheses, indent=2)}

Previous Code Ledger (Successes and Failures):
{json.dumps(ledger[-5:], indent=2)}

Generate a single Python code block using Polars to implement these features.
"""

    try:
        sys_prompt = FACTORY_SYSTEM_PROMPT
        code_raw = llm_call(prompt, system_prompt=sys_prompt)
        if "```python" in code_raw:
            code = code_raw.split("```python")[1].split("```")[0].strip()
        else:
            code = code_raw.strip()
    except Exception as e:
        return False, f"LLM Generation failed: {e}", {}

    res = run_in_sandbox(
        code, 
        agent_name=AGENT_NAME,
        purpose=f"Round {round_num} engineering",
        round_num=round_num
    )
    
    return res["success"], code, res

# ── Agent Node ───────────────────────────────────────────────────────────────

@timed_node
@with_agent_retry(AGENT_NAME)
def run_feature_factory(state: ProfessorState) -> ProfessorState:
    """
    Intelligence Layer: Iterative 5-Round Feature Factory.
    """
    session_id = state.get("session_id", "default")
    hack_mode = state.get("hackathon_mode", False)
    
    logger.info(f"[{AGENT_NAME}] Starting iterative refinement (Hackathon={hack_mode})...")

    hypotheses = state.get("feature_candidates", [])
    if not hypotheses and not hack_mode:
        # Standard mode fallback
        feature_order = state.get("feature_order", []) or ["dummy_f1"]
        updates = {"feature_order": feature_order}
        return ProfessorState.validated_update(state, AGENT_NAME, updates)

    ledger = []
    rounds_completed = 0
    max_rounds = 5
    
    depth = state.get("pipeline_depth", "standard")
    if depth == "sprint": max_rounds = 2

    effect_sizes = {}
    features_by_cat = {"condition": [], "outcome": [], "delta": [], "moderator": [], "uncategorized": []}

    import polars as pl
    
    for r in range(1, max_rounds + 1):
        logger.info(f"[{AGENT_NAME}] Round {r}/{max_rounds} starting...")
        success, code, result = _run_feature_round(state, r, hypotheses, ledger)
        
        if success:
            rounds_completed += 1
            entry = result.get("entry", {})
            if entry: ledger.append(entry)
            
            # Extract names from sandbox result
            new_fnames = result.get("new_columns", [])
            
            if hack_mode:
                categories = _extract_feature_categories(code, new_fnames)
                
                # Update manifest-like list for entry
                round_manifest = []
                for fname in new_fnames:
                    cat = categories.get(fname, "uncategorized")
                    round_manifest.append({"name": fname, "category": cat})
                    features_by_cat[cat].append(fname)
                
                # Perform Statistical Gating
                enriched_path = result.get("output_path")
                if enriched_path and os.path.exists(enriched_path):
                    try:
                        df_round = pl.read_parquet(enriched_path)
                        condition_vals = _get_condition_values(df_round, state.active_thesis or {}, round_manifest)
                        
                        passed_round = []
                        for fname in new_fnames:
                            if fname in df_round.columns:
                                f_vals = df_round[fname].to_numpy()
                                passed, p, d, details = _hackathon_gate(f_vals, condition_vals, state.gate_config or {})
                                if passed:
                                    passed_round.append(fname)
                                    effect_sizes[fname] = details
                        
                        entry["passed_gate"] = passed_round
                    except Exception as e:
                        logger.error(f"Hackathon gating failed: {e}")
            else:
                # Standard adaptive gate
                passed, gate_reports = run_adaptive_gate(state, new_fnames)
                entry["passed_gate"] = passed
                entry["gate_reports"] = gate_reports

    # Consolidate results
    all_passed = []
    for e in ledger:
        all_passed.extend(e.get("passed_gate", []))

    updates = {
        "feature_order": [e.get("entry_id") for e in ledger if e.get("success")] or ["ff_1"],
        "features_gate_passed": list(set(all_passed)),
        "feature_data_path": state.get("enriched_data_path") or state.get("clean_data_path"),
    }
    
    for i in range(1, 6):
        updates[f"round{i}_features"] = ledger[i-1].get("code") if len(ledger) >= i else None

    if hack_mode:
        updates["thesis_effect_sizes"] = effect_sizes
        updates["thesis_features_by_category"] = features_by_cat

    log_event(
        session_id=session_id,
        agent=AGENT_NAME,
        action="factory_rounds_complete",
        values_changed={"rounds": rounds_completed}
    )

    return ProfessorState.validated_update(state, AGENT_NAME, updates)
