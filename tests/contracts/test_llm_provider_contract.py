# tests/contracts/test_llm_provider_contract.py

import pytest
import json
from tools.llm_provider import _safe_json_loads

def test_safe_json_loads_direct():
    """Verify direct JSON parsing."""
    text = '{"key": "value"}'
    assert _safe_json_loads(text) == {"key": "value"}

def test_safe_json_loads_markdown():
    """Verify JSON extraction from markdown blocks."""
    text = "Here is the data:\n```json\n{\"score\": 0.95}\n```\nHope this helps."
    assert _safe_json_loads(text) == {"score": 0.95}

def test_safe_json_loads_messy():
    """Verify JSON extraction from messy text."""
    text = "The result is { \"success\": true } but there is more text."
    assert _safe_json_loads(text) == {"success": True}

def test_safe_json_loads_failure():
    """Verify ValueError on unparseable text."""
    with pytest.raises(ValueError):
        _safe_json_loads("No JSON here at all.")
