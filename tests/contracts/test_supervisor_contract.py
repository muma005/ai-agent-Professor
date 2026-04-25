# tests/contracts/test_supervisor_contract.py

import pytest
from unittest.mock import patch
from core.state import ProfessorState, initial_state
from agents.supervisor import run_supervisor

@pytest.fixture
def supervisor_state():
    state_dict = initial_state(session_id="test-supervisor")
    state = ProfessorState(**state_dict)
    state.dag = ["data_engineer", "eda_agent", "ml_optimizer"]
    return state

class TestSupervisorContract:
    """
    Contract: Supervisor Logic (Component 7)
    """

    @patch("agents.supervisor.emit_to_operator")
    def test_replan_trigger_routes_to_router(self, mock_emit, supervisor_state):
        """Verify replan_requested=True routes back to semantic_router."""
        supervisor_state.replan_requested = True
        supervisor_state.current_node = "ml_optimizer"
        
        final_state = run_supervisor(supervisor_state)
        assert final_state["next_node"] == "semantic_router"

    @patch("agents.supervisor.emit_to_operator")
    def test_replan_increments_dag_version(self, mock_emit, supervisor_state):
        """Verify dag_version is incremented on replan."""
        initial_version = supervisor_state.dag_version
        supervisor_state.replan_requested = True
        
        final_state = run_supervisor(supervisor_state)
        assert final_state["dag_version"] == initial_version + 1

    def test_normal_flow_sequencing(self, supervisor_state):
        """Verify next_node is correct for intermediate nodes."""
        supervisor_state.current_node = "data_engineer"
        final_state = run_supervisor(supervisor_state)
        assert final_state["next_node"] == "eda_agent"

    def test_final_node_routes_to_publisher(self, supervisor_state):
        """Verify last node in DAG leads to publisher."""
        supervisor_state.current_node = "ml_optimizer"
        final_state = run_supervisor(supervisor_state)
        assert final_state["next_node"] == "publisher"

    def test_preflight_routes_to_router(self, supervisor_state):
        """Verify initial preflight node leads to router."""
        supervisor_state.current_node = "preflight"
        final_state = run_supervisor(supervisor_state)
        assert final_state["next_node"] == "semantic_router"

    @patch("agents.supervisor.emit_to_operator")
    def test_replan_resets_trigger_flag(self, mock_emit, supervisor_state):
        """Verify the replan_requested flag is cleared after use."""
        supervisor_state.replan_requested = True
        final_state = run_supervisor(supervisor_state)
        assert final_state["replan_requested"] is False
