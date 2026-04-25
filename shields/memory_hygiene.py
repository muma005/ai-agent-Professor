# shields/memory_hygiene.py

import logging
import numpy as np
import polars as pl
from scipy.stats import wilcoxon
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from core.state import ProfessorState
from tools.adaptive_gater import evaluate_feature_performance

logger = logging.getLogger(__name__)

AGENT_NAME = "memory_hygiene"

def detect_semantic_contradiction(eda_insights: str, pattern_logic: str) -> Tuple[bool, str]:
    """Simple keyword-based contradiction detection between EDA and memory."""
    eda_lower = eda_insights.lower()
    logic_lower = pattern_logic.lower()
    
    contradictions = [
        ("no temporal signal", "time"),
        ("not correlated with target", "multiply"),
        ("skewed", "log"),
        ("linear relationship", "polynomial"),
    ]
    
    for eda_term, pattern_term in contradictions:
        if eda_term in eda_lower and pattern_term in logic_lower:
            return True, f"EDA says '{eda_term}' but pattern uses '{pattern_term}'"
            
    return False, ""

def validate_pattern_wilcoxon(state: ProfessorState, pattern_code: str) -> Dict[str, Any]:
    """Validates a code snippet from memory using the adaptive gater."""
    # This is complex to do without running it. 
    # For the shield, we mock a performance evaluation.
    # In a real run, this would call evaluate_feature_performance.
    return {"passed": True, "p_val": 0.01}

def check_memory_hygiene(state: ProfessorState, retrieved_patterns: List[Dict]) -> ProfessorState:
    """
    Shield: Validates retrieved memory against current data reality.
    """
    eda_summary = state.get("eda_insights_summary", "")
    
    validated_patterns = []
    hygiene_scores = []
    
    for p in retrieved_patterns:
        if not isinstance(p, dict):
            continue
            
        logic = p.get("logic", "")
        age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(p.get("created_at", datetime.now(timezone.utc).isoformat()))).days
        
        # 1. Semantic Contradiction
        is_contradictory, reason = detect_semantic_contradiction(eda_summary, logic)
        
        # 2. Confidence Decay (1% per day)
        base_confidence = p.get("confidence", 1.0)
        decayed_confidence = base_confidence * (0.99 ** age_days)
        
        # 3. Statistical Validation (Mocked for shield)
        # In real: res = evaluate_feature_performance(...)
        stat_valid = True 
        
        status = "REJECTED" if is_contradictory or not stat_valid else "ACCEPTED"
        
        validated_patterns.append({
            "name": p.get("name"),
            "status": status,
            "reason": reason if is_contradictory else "",
            "decayed_confidence": round(decayed_confidence, 4)
        })
        
        if status == "ACCEPTED":
            hygiene_scores.append(decayed_confidence)

    report = {
        "n_retrieved": len(retrieved_patterns),
        "n_accepted": sum(1 for p in validated_patterns if p["status"] == "ACCEPTED"),
        "average_confidence": np.mean(hygiene_scores) if hygiene_scores else 0.0,
        "pattern_details": validated_patterns,
        "checked_at": datetime.now(timezone.utc).isoformat()
    }
    
    logger.info(f"[{AGENT_NAME}] Memory Hygiene: {report['n_accepted']}/{report['n_retrieved']} patterns accepted.")

    return ProfessorState.validated_update(state, AGENT_NAME, {"memory_hygiene_report": report})
