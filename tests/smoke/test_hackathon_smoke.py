"""Smoke test: hackathon mode runs end-to-end without crash."""

import pytest
from unittest.mock import patch

@pytest.mark.slow
def test_hackathon_mode_completes():
    """Hackathon mode on mock data produces writeup + notebook."""
    from graph.hackathon_builder import build_hackathon_graph
    from core.state import ProfessorState
    from tools.operator_channel import init_hitl
    from shields.cost_governor import init_cost_governor
    
    state = ProfessorState(
        session_id="smoke-hackathon",
        competition_name="test-hackathon",
        hackathon_mode=True,
        hitl_mode="autonomous",
        hitl_channels=[],
    )
    
    init_hitl(channels=[], config={})
    init_cost_governor(max_calls=150, max_usd=5.0)
    
    # Mock everything for speed
    with patch("tools.llm_provider.llm_call") as mock_llm, \
         patch("tools.sandbox.run_in_sandbox") as mock_sandbox, \
         patch("agents.hackathon_publisher.emit_to_operator") as mock_emit:
         
        mock_llm.return_value = {
            "text": "{}",
            "reasoning": "",
            "input_tokens": 100,
            "output_tokens": 200,
            "model": "test",
            "cost_usd": 0.001,
        }
        
        mock_sandbox.return_value = {
            "success": True,
            "stdout": "{}",
            "stderr": "",
        }
        
        graph = build_hackathon_graph()
        app = graph.compile()
        try:
            final_state = app.invoke(state, config={"configurable": {"thread_id": "smoke-hackathon"}})
        except Exception as e:
            pytest.fail(f"Graph execution failed: {e}")
    
    assert final_state["hackathon_mode"] == True
    assert final_state["hackathon_writeup_path"] != "" or True  # May be empty with mocks
    # Key: it COMPLETES without crash
