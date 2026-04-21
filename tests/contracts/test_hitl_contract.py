# tests/contracts/test_hitl_contract.py

import pytest
import sys
import time
from unittest.mock import MagicMock, patch
from tools.operator_channel import init_hitl, emit_to_operator, process_pending_injections, _MANAGER
from core.state import ProfessorState

@pytest.fixture
def clean_hitl():
    """Ensure a fresh HITL manager for each test."""
    global _MANAGER
    import tools.operator_channel
    tools.operator_channel._MANAGER = None
    yield
    if tools.operator_channel._MANAGER:
        tools.operator_channel._MANAGER.stop_listener()
    tools.operator_channel._MANAGER = None

def test_emit_status_non_blocking(clean_hitl):
    """STATUS level should return immediately."""
    init_hitl(["cli"], {})
    
    start_time = time.time()
    res = emit_to_operator("Test status", level="STATUS")
    elapsed = time.time() - start_time
    
    assert res is None
    assert elapsed < 0.1

def test_emit_checkpoint_timeout(clean_hitl):
    """CHECKPOINT level should wait for response and timeout."""
    init_hitl(["cli"], {})
    
    # We mock poll_any to return None (timeout) quickly for testing
    with patch("tools.operator_channel.ChannelManager.poll_any", return_value=None):
        res = emit_to_operator("Confirm checkpoint", level="CHECKPOINT")
        assert res is None

def test_cli_adapter_output(clean_hitl, capsys):
    """CLI adapter should print formatted messages to stdout."""
    init_hitl(["cli"], {})
    emit_to_operator("Hello Operator", level="STATUS")
    
    captured = capsys.readouterr()
    assert "Hello Operator" in captured.out
    assert "[36m" in captured.out  # Cyan color code

def test_command_pause(clean_hitl):
    """Command /pause should set pipeline_paused flag."""
    manager = init_hitl(["cli"], {})
    manager.listener._classify_command("/pause")
    
    assert manager.listener.pipeline_paused is True

def test_feature_hint_injection(clean_hitl):
    """Command /feature should queue a hint."""
    manager = init_hitl(["cli"], {})
    manager.listener._classify_command("/feature log(target)")
    
    item = manager.listener.injection_queue.get()
    assert item["type"] == "feature_hint"
    assert item["text"] == "log(target)"

def test_domain_knowledge_injection(clean_hitl):
    """Free text should be treated as domain knowledge."""
    manager = init_hitl(["cli"], {})
    manager.listener._classify_command("The target is highly skewed")
    
    item = manager.listener.injection_queue.get()
    assert item["type"] == "domain"
    assert item["text"] == "The target is highly skewed"

def test_process_pending_injections(clean_hitl):
    """Injections from queue should be applied to state."""
    manager = init_hitl(["cli"], {})
    manager.listener.injection_queue.put({"type": "domain", "text": "Domain info"})
    manager.listener.pipeline_paused = True
    
    state = ProfessorState(session_id="test")
    updated_state = process_pending_injections(state)
    
    assert len(updated_state.hitl_injections) == 1
    assert updated_state.hitl_injections[0]["text"] == "Domain info"
    assert updated_state.pipeline_paused is True

def test_message_truncation(clean_hitl, capsys):
    """Messages over 500 chars should be truncated."""
    init_hitl(["cli"], {})
    long_msg = "A" * 600
    emit_to_operator(long_msg, level="STATUS")
    
    captured = capsys.readouterr()
    assert "..." in captured.out
    assert len(captured.out.split("#1 ")[1].strip()) <= 505 # Account for sequence tag

def test_cli_adapter_no_tty_poll(clean_hitl):
    """CLI poll should return None immediately if not a TTY."""
    init_hitl(["cli"], {})
    with patch("sys.stdout.isatty", return_value=False):
        from tools.operator_channel import CLIAdapter
        adapter = CLIAdapter()
        assert adapter.poll_response(timeout=10) is None
