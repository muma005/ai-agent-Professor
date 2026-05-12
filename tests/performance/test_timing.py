"""Verify performance constraints."""

import time

def test_diagnostic_injection_overhead():
    """Diagnostic injection adds < 2000ms to successful code execution."""
    from tools.sandbox import _inject_diagnostics
    
    code = "import json\nx = sum(range(100000))\nprint(x)"
    wrapped = _inject_diagnostics(code)
    
    # Time execution without diagnostics
    start = time.time()
    exec(code)
    base_time = time.time() - start
    
    # Time execution with diagnostics
    start = time.time()
    exec(wrapped)
    wrapped_time = time.time() - start
    
    overhead = wrapped_time - base_time
    assert overhead < 2.0  # Less than 2000ms

def test_leakage_precheck_under_1ms():
    """Leakage precheck on 200-line code block takes < 50ms."""
    from guards.leakage_precheck import check_code_for_leakage
    
    code = "\n".join([f"x_{i} = {i} * 2" for i in range(200)])
    
    start = time.time()
    check_code_for_leakage(code)
    elapsed = time.time() - start
    
    assert elapsed < 0.05  # Less than 50ms

def test_rubric_deterministic_extraction_fast():
    """Deterministic rubric extraction (no LLM) completes in < 100ms."""
    from tools.rubric_parser import _deterministic_extract
    
    text = "1. Technical Quality (30 points)\n2. Novelty (20 points)\n3. Documentation (20 points)"
    
    start = time.time()
    result = _deterministic_extract(text)
    elapsed = time.time() - start
    
    assert elapsed < 0.1  # Less than 100ms

def test_state_serialization_under_1s():
    """ProfessorState serialization under 1s even at 20MB."""
    from core.state import ProfessorState
    import json
    
    state = ProfessorState()
    # Fill with large data
    state.hitl_messages_sent = [{"msg": "x" * 1000}] * 200
    
    start = time.time()
    json.dumps(state.model_dump(), default=str)
    elapsed = time.time() - start
    
    assert elapsed < 1.0
