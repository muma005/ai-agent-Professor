# tools/llm_provider.py

import os
import json
import time
import logging
import random
from typing import Optional, Dict, List, Any, Tuple
from tools import llm_client # Existing client

logger = logging.getLogger(__name__)

# ── Provider Config ──────────────────────────────────────────────────────────

PROVIDERS = {
    "primary":   {"model": "deepseek-v3", "provider": "fireworks"},
    "secondary": {"model": "gpt-4o",      "provider": "openai"},
    "fallback":  {"model": "gemini-2.0-flash", "provider": "google"}
}

# ── Unified LLM Interface ───────────────────────────────────────────────────

def llm_call(
    prompt: str,
    system_prompt: str = "You are a Kaggle Grandmaster.",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    json_mode: bool = True,
    provider_override: str = None
) -> str:
    """
    Unified LLM call with provider rotation and exponential backoff.
    """
    providers_to_try = ["primary", "secondary", "fallback"]
    if provider_override:
        providers_to_try = [provider_override] + [p for p in providers_to_try if p != provider_override]

    last_error = ""
    for attempt_idx, p_key in enumerate(providers_to_try):
        p_cfg = PROVIDERS.get(p_key)
        if not p_cfg: continue
        
        # Exponential backoff
        for retry in range(2):
            try:
                # Stub: Using the existing client logic but wrapped
                # Note: In a real implementation, we'd map p_cfg to client calls
                response = llm_client.llm_call(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if json_mode:
                    # Validate JSON
                    _safe_json_loads(response)
                
                return response

            except Exception as e:
                last_error = str(e)
                wait_time = (2 ** retry) + random.random()
                logger.warning(f"LLM [{p_key}] attempt {retry+1} failed: {e}. Waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        logger.error(f"LLM Provider [{p_key}] exhausted. Trying next provider.")

    raise Exception(f"All LLM providers failed. Last error: {last_error}")

def _safe_json_loads(text: str) -> Dict:
    """Robust JSON extraction from LLM markdown."""
    try:
        # Try direct
        return json.loads(text)
    except:
        # Try markdown block
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
            
        # Try finding first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
            
    raise ValueError("Could not parse JSON from LLM response.")

import re # needed for _safe_json_loads
