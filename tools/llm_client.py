# tools/llm_client.py

import os
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
import requests
import json

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────
_groq_client = None
_gemini_configured = False

def _get_groq():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client

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
    model: str = "groq-deepseek",
    max_tokens: int = 4096,
    is_coding_task: bool = False
) -> str:
    """
    Unified LLM interface. All agents call this. Never call Groq/Gemini directly.

    Models:
      "groq-deepseek"  → DeepSeek-R1 70B on Groq (default, all reasoning agents)
      "groq-llama"     → Llama 3.3 70B on Groq (routing, fast tasks)
      "gemini-flash"   → Gemini 2.0 Flash (Publisher, QA Gate)
      "ollama"         → DeepSeek-R1 14b local (fallback only)
    """

    # Inject Polars constraint into all coding task prompts
    if is_coding_task:
        system = POLARS_CONSTRAINT + "\n\n" + system

    # ── Groq: DeepSeek-R1 70B ─────────────────────────────────────
    if model == "groq-deepseek":
        try:
            return _call_groq(
                prompt=prompt,
                system=system,
                model_name="deepseek-r1-distill-llama-70b",
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"[llm_client] Groq DeepSeek failed: {e}. Falling back to Gemini.")
            return _call_gemini(prompt, system, max_tokens)

    # ── Groq: Llama 3.3 70B ───────────────────────────────────────
    elif model == "groq-llama":
        try:
            return _call_groq(
                prompt=prompt,
                system=system,
                model_name="llama-3.3-70b-versatile",
                max_tokens=max_tokens
            )
        except Exception as e:
            print(f"[llm_client] Groq Llama failed: {e}. Falling back to DeepSeek.")
            return _call_groq(prompt, system, "deepseek-r1-distill-llama-70b", max_tokens)

    # ── Gemini 2.0 Flash ──────────────────────────────────────────
    elif model == "gemini-flash":
        try:
            return _call_gemini(prompt, system, max_tokens)
        except Exception as e:
            print(f"[llm_client] Gemini failed: {e}. Falling back to Groq.")
            return _call_groq(prompt, system, "deepseek-r1-distill-llama-70b", max_tokens)

    # ── Ollama local fallback ──────────────────────────────────────
    elif model == "ollama":
        return _call_ollama(prompt, system)

    else:
        raise ValueError(f"Unknown model: {model}. Use groq-deepseek, groq-llama, gemini-flash, or ollama.")


def _call_groq(prompt: str, system: str, model_name: str, max_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _get_groq().chat.completions.create(
        model=model_name,
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


def _call_ollama(prompt: str, system: str) -> str:
    payload = {
        "model": "deepseek-r1:14b",
        "messages": [],
        "stream": False
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})

    response = requests.post("http://localhost:11434/api/chat", json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]
