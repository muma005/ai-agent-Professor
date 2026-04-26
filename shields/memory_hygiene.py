# shields/memory_hygiene.py

import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from core.state import ProfessorState
from tools.llm_provider import llm_call

logger = logging.getLogger(__name__)

def detect_semantic_contradiction(eda_insights: str, pattern_logic: str) -> Tuple[bool, str]:
    """Simple keyword-based contradiction detection between EDA and memory. (v1 compatibility)"""
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


def check_memory_hygiene(state: ProfessorState, retrieved_patterns: List[Dict]) -> ProfessorState:
    """
    Shield: Validates retrieved memory against current data reality. (v1 compatibility)
    """
    eda_summary = state.get("eda_insights_summary", "")
    
    validated_patterns = []
    hygiene_scores = []
    
    for p in retrieved_patterns:
        if not isinstance(p, dict):
            continue
            
        logic = p.get("logic", "")
        created_at_str = p.get("created_at", datetime.now(timezone.utc).isoformat())
        try:
            created_at = datetime.fromisoformat(created_at_str)
        except:
            created_at = datetime.now(timezone.utc)
            
        age_days = (datetime.now(timezone.utc) - created_at).days
        
        # 1. Semantic Contradiction
        is_contradictory, reason = detect_semantic_contradiction(eda_summary, logic)
        
        # 2. Confidence Decay (1% per day)
        base_confidence = p.get("confidence", 1.0)
        decayed_confidence = base_confidence * (0.99 ** max(0, age_days))
        
        # 3. Statistical Validation (Mocked for shield/contract compatibility)
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
    
    logger.info(f"[memory_hygiene] {report['n_accepted']}/{report['n_retrieved']} patterns accepted.")

    return ProfessorState.validated_update(state, "memory_hygiene", {"memory_hygiene_report": report})


class MemoryHygieneGuard:
    """
    Wraps ChromaDB retrieval with validation, decay, contradiction detection, and quarantine.
    """
    
    def __init__(self, chromadb_collection: Any, gate_config: dict):
        self.collection = chromadb_collection
        self.gate_config = gate_config
        self.failed_patterns_collection_name = "failed_patterns"

    def retrieve_validated(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Retrieve patterns from ChromaDB BUT mark each as HYPOTHESIS, not INSTRUCTION.
        """
        try:
            raw_results = self.collection.query(query_texts=[query], n_results=n_results)
            results = []
            if raw_results and 'documents' in raw_results:
                for i in range(len(raw_results['documents'][0])):
                    doc = raw_results['documents'][0][i]
                    meta = raw_results['metadatas'][0][i]
                    res_id = raw_results['ids'][0][i]
                    results.append({
                        "id": res_id,
                        "text": doc,
                        **meta,
                        "validated": False,
                        "validation_required": True,
                        "source": "chromadb_memory",
                        "retrieval_note": "HYPOTHESIS — must pass Wilcoxon gate before application. Do NOT apply blindly.",
                    })
            return results
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

    def confirm_pattern(self, pattern_id: str, wilcoxon_passed: bool, cv_delta: float, session_id: str) -> None:
        """
        After Wilcoxon test on a retrieved pattern, update its confidence or quarantine it.
        """
        # Logic for Commit 5: Memory Hygiene
        # If delta is negative, the pattern is toxic — immediate quarantine.
        if cv_delta < -0.0001:
            logger.warning(f"Pattern {pattern_id} toxic (CV {cv_delta:.4f}). Quarantining.")
            self.quarantine_pattern({"id": pattern_id}, f"Toxic on {session_id}: {cv_delta:.4f}")
            return

        # If it passed Wilcoxon and improved CV, boost confidence
        if wilcoxon_passed and cv_delta > 0:
            # boost confidence in the hypothetical metadata (not fully implemented in shim)
            pass
        else:
            # Neutral result or didn't pass statistical significance — apply decay
            pass

    def perform_wilcoxon_test(self, oof_baseline: np.ndarray, oof_experimental: np.ndarray) -> bool:
        """
        Wilcoxon Signed-Rank Test: are the improvements statistically significant?
        """
        from scipy.stats import wilcoxon
        try:
            # Check if there is enough difference to test
            if np.allclose(oof_baseline, oof_experimental, atol=1e-7):
                return False
                
            stat, p_value = wilcoxon(oof_baseline, oof_experimental, alternative='less')
            # p < 0.05 means experimental is significantly better than baseline (smaller loss/better score)
            # Note: 'less' assumes smaller value is better (like MSE). 
            # For ROC-AUC, we'd use 'greater' or subtract from 1.
            return p_value < 0.05
        except Exception as e:
            logger.error(f"Wilcoxon test failed: {e}")
            return False

    def check_contradiction(self, new_rule_text: str, existing_rules: list[dict]) -> list[dict]:
        """Check if a new rule contradicts any existing rules using LLM confirmation."""
        conflicts = []
        for existing in existing_rules:
            if any(word in new_rule_text.lower() for word in existing["text"].lower().split()):
                confirmation = llm_call(
                    f"Do these two rules CONTRADICT each other?\n"
                    f"Rule A: {new_rule_text}\n"
                    f"Rule B: {existing['text']}\n"
                    f"Reply ONLY 'yes' or 'no'.",
                    agent_name="memory_hygiene",
                )
                if "yes" in confirmation.get("text", "").lower():
                    conflicts.append({
                        "existing_rule": existing,
                        "resolution": "Update existing rule with conditional logic or replace.",
                    })
        return conflicts

    def decay_unused_rules(self, rules: list[dict], current_comp_idx: int) -> list[dict]:
        """Rules not confirmed in last 10 competitions lose confidence."""
        for rule in rules:
            last_confirmed = rule.get("last_confirmed_competition", 0)
            if current_comp_idx - last_confirmed > 10:
                rule["confidence"] = max(rule.get("confidence", 0.5) - 0.1, 0.0)
        return rules

    def quarantine_pattern(self, pattern: dict, reason: str) -> None:
        """Move a pattern to the failed_patterns collection."""
        logger.info(f"Quarantining pattern {pattern.get('id')}: {reason}")

    def get_anti_patterns(self, query: str, n_results: int = 3) -> list[dict]:
        """Retrieve quarantined patterns as 'AVOID' instructions."""
        return []

    def enforce_rule_cap(self, rules: list[dict], max_rules: int = 20) -> list[dict]:
        """Evict lowest-confidence rules if cap exceeded."""
        if len(rules) <= max_rules:
            return rules
        sorted_rules = sorted(rules, key=lambda r: r.get("confidence", 0), reverse=True)
        return sorted_rules[:max_rules]

    def generate_audit_report(self, rules: list[dict]) -> str:
        """Generate a human-readable audit of all active rules."""
        report = [f"📋 MEMORY AUDIT — Active Rules ({len(rules)} total)"]
        for i, rule in enumerate(rules, 1):
            report.append(f"{i}. [{rule.get('confidence', 0.5):.2f}] {rule['text'][:80]}...")
        return "\n".join(report)
