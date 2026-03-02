# tools/llm_client.py

import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import json

load_dotenv()

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
            return _call_fireworks_deepseek(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens
            )
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
    return response.choices[0].message.content


def _call_gemini(prompt: str, system: str, max_tokens: int) -> str:
    model = _get_gemini("gemini-2.0-flash")
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    response = model.generate_content(
        full_prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": 0.1}
    )
    return response.text
