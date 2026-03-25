# tools/llm_client.py

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import json

load_dotenv()

logger = logging.getLogger(__name__)

class TokenTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0

global_tracker = TokenTracker()

def get_token_usage() -> dict:
    return {"prompt": global_tracker.prompt_tokens, "completion": global_tracker.completion_tokens}


# ── FLAW-8.1: API Response Validation ────────────────────────────

class APIResponseValidator:
    """Validates API responses for quality and safety."""
    
    @staticmethod
    def validate_response(response: str, model: str) -> dict:
        """
        Validate API response.
        
        Args:
            response: API response text
            model: Model that generated response
        
        Returns:
            Validation result dict
        """
        issues = []
        warnings = []
        
        # Check for empty response
        if not response or not response.strip():
            issues.append("Empty response")
        
        # Check for error patterns
        error_patterns = [
            "error:", "failed:", "exception:", "traceback:",
            "rate limit", "quota exceeded", "unauthorized",
        ]
        
        response_lower = response.lower()
        for pattern in error_patterns:
            if pattern in response_lower:
                warnings.append(f"Contains error pattern: '{pattern}'")
        
        # Check for hallucinated API keys (security)
        key_patterns = [
            "sk-", "api_key=", "apikey=", "secret=",
            "FIREWORKS", "GEMINI", "GROQ",
        ]
        
        for pattern in key_patterns:
            if pattern in response:
                issues.append(f"Potential API key leakage: '{pattern}'")
        
        # Check response length
        if len(response) > 100000:  # 100K chars
            warnings.append(f"Unusually long response: {len(response)} chars")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "model": model,
            "response_length": len(response),
        }

# ── Clients ───────────────────────────────────────────────────────
_fireworks_deepseek_client = None
_fireworks_glm_client = None
_gemini_configured = False

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

def _get_fireworks_deepseek():
    global _fireworks_deepseek_client
    if _fireworks_deepseek_client is None:
        _fireworks_deepseek_client = OpenAI(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url=FIREWORKS_BASE_URL
        )
    return _fireworks_deepseek_client

def _get_fireworks_glm():
    global _fireworks_glm_client
    if _fireworks_glm_client is None:
        _fireworks_glm_client = OpenAI(
            api_key=os.getenv("FIREWORKS_GLM_API_KEY"),
            base_url=FIREWORKS_BASE_URL
        )
    return _fireworks_glm_client

def _get_gemini(model_name: str):
    global _gemini_configured
    if not _gemini_configured:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _gemini_configured = True
    return genai.GenerativeModel(model_name)

# ── Polars constraint injected into all code-generating prompts ───
POLARS_CONSTRAINT = """
LIBRARY REQUIREMENT: This pipeline uses Polars not Pandas.
CORRECT:   pl.read_csv()  df.write_parquet()  df.fill_null()  df.group_by()
INCORRECT: pd.read_csv()  df.to_parquet()     df.fillna()     df.groupby()
If pandas is required for a specific library call:
  convert back with pl.from_pandas(df) before returning.
"""


# ── FLAW-2.3 FIX: LLM Output Validation ──────────────────────────

class LLMOutputValidationError(Exception):
    """Raised when LLM output validation fails."""
    pass


def validate_llm_output(output: str, expected_type: str = "text") -> bool:
    """
    Validate LLM output before using it.
    
    Args:
        output: Raw LLM output
        expected_type: "text", "json", "code", "list"
    
    Returns:
        True if valid
    
    Raises:
        LLMOutputValidationError if invalid
    """
    if not output or not output.strip():
        raise LLMOutputValidationError("Empty output")
    
    if expected_type == "json":
        try:
            # Try to extract JSON
            start = output.find("{")
            end = output.rfind("}")
            if start == -1 or end == -1:
                raise LLMOutputValidationError("No JSON object found")
            
            json.loads(output[start:end+1])
            return True
            
        except json.JSONDecodeError as e:
            raise LLMOutputValidationError(f"Invalid JSON: {e}")
    
    elif expected_type == "code":
        # Check for suspicious patterns FIRST
        suspicious = ["__import__", "eval(", "exec(", "subprocess", "os.system"]
        for pattern in suspicious:
            if pattern in output:
                raise LLMOutputValidationError(f"Suspicious code pattern: {pattern}")
        
        # Check for common code patterns
        if not any(pattern in output for pattern in ["def ", "import ", "class ", "return "]):
            raise LLMOutputValidationError("No code detected")
        
        return True
    
    elif expected_type == "list":
        # Try to extract list
        start = output.find("[")
        end = output.rfind("]")
        if start == -1 or end == -1:
            raise LLMOutputValidationError("No list found")
        
        return True
    
    return True  # text is always valid


def call_llm_validated(
    prompt: str,
    expected_type: str = "text",
    max_retries: int = 3,
    **kwargs
) -> str:
    """
    Call LLM with output validation and retry logic.
    
    Args:
        prompt: Prompt to send
        expected_type: Expected output type
        max_retries: Maximum retry attempts
        **kwargs: Passed to call_llm
    
    Returns:
        Validated LLM output
    
    Raises:
        LLMOutputValidationError if validation fails after retries
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            output = call_llm(prompt, **kwargs)
            validate_llm_output(output, expected_type)
            return output
            
        except LLMOutputValidationError as e:
            last_error = e
            logger.warning(
                f"LLM output validation failed (attempt {attempt+1}/{max_retries}): {e}"
            )
            
            # Retry with validation hint
            prompt = f"{prompt}\n\nNOTE: Output must be valid {expected_type}. Please retry."
    
    raise LLMOutputValidationError(
        f"LLM output validation failed after {max_retries} retries: {last_error}"
    )


# ── Main call function ────────────────────────────────────────────
def call_llm(
    prompt: str,
    system: str = "",
    model: str = "deepseek",
    max_tokens: int = 4096,
    is_coding_task: bool = False
) -> str:
    """
    Unified LLM interface. All agents call this. Never call Fireworks/Gemini directly.

    Models:
      "deepseek"     -> DeepSeek-R1 on Fireworks (default, all reasoning agents)
      "glm"          -> GLM on Fireworks (routing, fast tasks)
      "gemini-flash" -> Gemini 2.0 Flash (Publisher, QA Gate)
    """

    # Inject Polars constraint into all coding task prompts
    if is_coding_task:
        system = POLARS_CONSTRAINT + "\n\n" + system

    # ── Fireworks: DeepSeek ────────────────────────────────────────
    if model == "deepseek":
        try:
            response = _call_fireworks_deepseek(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens
            )
            # FLAW-8.1: Validate response
            validation = APIResponseValidator.validate_response(response, model)
            if not validation["valid"]:
                logger.warning(f"[LLM] Response validation issues: {validation['issues']}")
            if validation["warnings"]:
                logger.debug(f"[LLM] Response warnings: {validation['warnings']}")
            return response
        except Exception as e:
            print(f"[llm_client] Fireworks DeepSeek failed: {e}. Falling back to Gemini.")
            return _call_gemini(prompt, system, max_tokens)

    # ── Fireworks: GLM ─────────────────────────────────────────────
    elif model == "glm":
        try:
            return _call_fireworks_glm(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"[llm_client] Fireworks GLM failed: {e}. Falling back to DeepSeek.")
            return _call_fireworks_deepseek(prompt, system, max_tokens)

    # ── Gemini 2.0 Flash ──────────────────────────────────────────
    elif model == "gemini-flash":
        try:
            return _call_gemini(prompt, system, max_tokens)
        except Exception as e:
            print(f"[llm_client] Gemini failed: {e}. Falling back to DeepSeek.")
            return _call_fireworks_deepseek(prompt, system, max_tokens)

    else:
        raise ValueError(f"Unknown model: {model}. Use deepseek, glm, or gemini-flash.")


def _call_fireworks_deepseek(prompt: str, system: str, max_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_fireworks_deepseek().chat.completions.create(
        model="accounts/fireworks/models/deepseek-v3p2",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1
    )
    if hasattr(response, "usage") and response.usage:
        global_tracker.prompt_tokens += getattr(response.usage, "prompt_tokens", 0)
        global_tracker.completion_tokens += getattr(response.usage, "completion_tokens", 0)
    return response.choices[0].message.content


def _call_fireworks_glm(prompt: str, system: str, max_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_fireworks_glm().chat.completions.create(
        model="accounts/fireworks/models/glm-5",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1
    )
    if hasattr(response, "usage") and response.usage:
        global_tracker.prompt_tokens += getattr(response.usage, "prompt_tokens", 0)
        global_tracker.completion_tokens += getattr(response.usage, "completion_tokens", 0)
    return response.choices[0].message.content


def _call_gemini(prompt: str, system: str, max_tokens: int) -> str:
    model = _get_gemini("gemini-2.0-flash")
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    response = model.generate_content(
        full_prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": 0.1}
    )
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        global_tracker.prompt_tokens += getattr(response.usage_metadata, "prompt_token_count", 0)
        global_tracker.completion_tokens += getattr(response.usage_metadata, "candidates_token_count", 0)
    return response.text
