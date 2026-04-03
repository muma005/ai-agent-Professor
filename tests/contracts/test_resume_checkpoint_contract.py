import pytest
import json
from unittest.mock import patch, MagicMock
from core.state import initial_state
from guards.circuit_breaker import resume_from_checkpoint


def _serializable_state(state):
    """Strip non-serializable objects (ProfessorConfig) from state for JSON encoding."""
    return {k: v for k, v in state.items() if k != "config"}

def test_resume_missing_session_returns_error_state():
    # redis.get returns None
    with patch("memory.redis_state.get_redis_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client
        
        result = resume_from_checkpoint("non_existent_session", 1)
        assert result.get("hitl_required") is True
        assert "No checkpoint found" in result.get("hitl_message", "")
        # should return an error dict, not raise exception

def test_resume_corrupt_json_returns_error_state():
    with patch("memory.redis_state.get_redis_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get.return_value = b"{bad_json:"
        mock_get_client.return_value = mock_client
        
        result = resume_from_checkpoint("bad_session", 1)
        assert result.get("hitl_required") is True
        assert "Checkpoint corrupt" in result.get("hitl_message", "")

def test_resume_invalid_intervention_id_returns_error_state():
    with patch("memory.redis_state.get_redis_client") as mock_get_client:
        mock_client = MagicMock()
        # Provide valid JSON but invalid intervention_id
        valid_state = initial_state("comp", "data")
        valid_state["session_id"] = "valid_session"
        payload = json.dumps({
            "state": _serializable_state(valid_state),
            "agent": "data_engineer",
            "error_class": "data_quality"
        })
        mock_client.get.return_value = payload.encode('utf-8')
        mock_get_client.return_value = mock_client

        result = resume_from_checkpoint("valid_session", 5) # 5 is invalid
        assert result.get("hitl_required") is True
        assert "intervention_id must be 1, 2, or 3" in result.get("hitl_message", "")

def test_resume_success_resets_counters():
    with patch("memory.redis_state.get_redis_client") as mock_get_client:
        with patch("guards.circuit_breaker.log_event"):
            mock_client = MagicMock()
            valid_state = initial_state("comp", "data")
            valid_state["session_id"] = "valid_session"
            valid_state["current_node_failure_count"] = 3
            payload = json.dumps({
                "state": _serializable_state(valid_state),
                "agent": "data_engineer",
                "error_class": "data_quality"
            })
            mock_client.get.return_value = payload.encode('utf-8')
            mock_get_client.return_value = mock_client

            # Use intervention 1 (Skip validation)
            result = resume_from_checkpoint("valid_session", 1)

            assert result.get("hitl_required") is False
            assert result.get("current_node_failure_count") == 0
            assert result.get("skip_data_validation") is True
            assert result.get("hitl_intervention_id") == 1
            assert "Skip validation" in result.get("hitl_intervention_label")

def test_resume_manual_intervention_no_state_change():
    with patch("memory.redis_state.get_redis_client") as mock_get_client:
        with patch("guards.circuit_breaker.log_event"):
            mock_client = MagicMock()
            valid_state = initial_state("comp", "data")
            valid_state["session_id"] = "valid_session"
            payload = json.dumps({
                "state": _serializable_state(valid_state),
                "agent": "data_engineer",
                "error_class": "data_quality"
            })
            mock_client.get.return_value = payload.encode('utf-8')
            mock_get_client.return_value = mock_client

            # Use intervention 3 (MANUAL)
            result = resume_from_checkpoint("valid_session", 3)

            assert result.get("hitl_required") is False
            # Verify specific state fields weren't changed
            assert result.get("skip_data_validation") is False
            assert result.get("null_threshold") == 1.0
