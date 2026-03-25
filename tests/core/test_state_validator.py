"""
Tests for state validation.

FLAW-3.1: State Schema Runtime Validation
FLAW-3.2: State Validation Between Agents
"""
import pytest
from core.state_validator import (
    StateValidator,
    StateValidationError,
    get_validator,
    validate_state,
    STATE_SCHEMA,
    PIPELINE_STAGE_REQUIREMENTS,
)


class TestStateValidator:
    """Test StateValidator class."""

    def test_validator_creation(self):
        """Test validator can be created."""
        validator = StateValidator()
        
        assert validator is not None
        assert validator.strict is False
        assert validator.validation_history == []

    def test_validator_strict_mode(self):
        """Test strict mode raises exceptions."""
        validator = StateValidator(strict=True)
        
        assert validator.strict is True

    def test_validate_empty_state_fails(self):
        """Test empty state fails validation."""
        validator = StateValidator()
        state = {}
        
        result = validator.validate_state(state, stage="initial")
        
        assert result is False

    def test_validate_minimal_valid_state(self):
        """Test minimal valid state passes."""
        validator = StateValidator()
        state = {
            "session_id": "test_123",
            "created_at": "2026-03-25T00:00:00Z",
            "competition_name": "test_comp",
            "task_type": "tabular",
            "cost_tracker": {"total_usd": 0.0},
        }
        
        result = validator.validate_state(state, stage="initial")
        
        assert result is True

    def test_validate_type_mismatch(self):
        """Test type mismatch detection."""
        validator = StateValidator()
        state = {
            "session_id": 123,  # Should be str
            "created_at": "2026-03-25T00:00:00Z",
            "competition_name": "test_comp",
            "task_type": "tabular",
            "cost_tracker": {"total_usd": 0.0},
        }
        
        result = validator.validate_state(state, stage="initial")
        
        assert result is False
        assert any("Type error" in e for e in validator.validation_history[-1]["errors"])

    def test_validate_stage_requirements(self):
        """Test stage requirement validation."""
        validator = StateValidator()
        state = {
            "session_id": "test_123",
            "created_at": "2026-03-25T00:00:00Z",
            "competition_name": "test_comp",
            "task_type": "tabular",
            "cost_tracker": {"total_usd": 0.0},
            # Missing post_data_engineer required keys
        }
        
        result = validator.validate_state(state, stage="post_data_engineer")
        
        assert result is False
        assert any("Missing required key" in e for e in validator.validation_history[-1]["errors"])

    def test_validate_complete_state(self):
        """Test complete state passes all validations."""
        validator = StateValidator()
        state = {
            "session_id": "test_123",
            "created_at": "2026-03-25T00:00:00Z",
            "competition_name": "test_comp",
            "task_type": "tabular",
            "cost_tracker": {"total_usd": 0.0},
            "clean_data_path": "/path/to/data.parquet",
            "schema_path": "/path/to/schema.json",
            "preprocessor_path": "/path/to/preprocessor.json",
            "data_hash": "abc123",
            "target_col": "target",
            "id_columns": [],
            "test_data_path": "/path/to/test.csv",
            "sample_submission_path": "/path/to/sample.csv",
        }
        
        result = validator.validate_state(state, stage="post_data_engineer")
        
        assert result is True


class TestValidationHistory:
    """Test validation history tracking."""

    def test_history_records_validations(self):
        """Test validation history is recorded."""
        validator = StateValidator()
        state = {
            "session_id": "test_123",
            "created_at": "2026-03-25T00:00:00Z",
            "competition_name": "test_comp",
            "task_type": "tabular",
            "cost_tracker": {"total_usd": 0.0},
        }
        
        validator.validate_state(state, stage="initial", node_name="test_node")
        validator.validate_state(state, stage="initial", node_name="test_node2")
        
        assert len(validator.validation_history) == 2

    def test_get_validation_summary(self):
        """Test validation summary generation."""
        validator = StateValidator()
        state = {
            "session_id": "test_123",
            "created_at": "2026-03-25T00:00:00Z",
            "competition_name": "test_comp",
            "task_type": "tabular",
            "cost_tracker": {"total_usd": 0.0},
        }
        
        validator.validate_state(state, stage="initial")
        validator.validate_state({}, stage="initial")  # Will fail
        
        summary = validator.get_validation_summary()
        
        assert summary["total_validations"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert "pass_rate" in summary


class TestGlobalValidator:
    """Test global validator functions."""

    def test_get_validator_singleton(self):
        """Test get_validator returns same instance."""
        v1 = get_validator()
        v2 = get_validator()
        
        assert v1 is v2

    def test_validate_state_convenience(self):
        """Test validate_state convenience function."""
        state = {
            "session_id": "test_123",
            "created_at": "2026-03-25T00:00:00Z",
            "competition_name": "test_comp",
            "task_type": "tabular",
            "cost_tracker": {"total_usd": 0.0},
        }
        
        result = validate_state(state, stage="initial")
        
        assert result is True

    def test_validate_state_strict_raises(self):
        """Test strict validation raises exception."""
        state = {}  # Invalid state
        
        with pytest.raises(StateValidationError):
            validate_state(state, stage="initial", strict=True)


class TestStateSchema:
    """Test state schema definition."""

    def test_schema_has_required_keys(self):
        """Test schema defines required keys."""
        assert "session_id" in STATE_SCHEMA
        assert "competition_name" in STATE_SCHEMA
        assert "cost_tracker" in STATE_SCHEMA

    def test_schema_has_types(self):
        """Test schema defines types for keys."""
        assert STATE_SCHEMA["session_id"]["type"] == str
        assert STATE_SCHEMA["cost_tracker"]["type"] == dict

    def test_pipeline_stages_defined(self):
        """Test pipeline stages are defined."""
        assert "initial" in PIPELINE_STAGE_REQUIREMENTS
        assert "post_data_engineer" in PIPELINE_STAGE_REQUIREMENTS
        assert "post_ml_optimizer" in PIPELINE_STAGE_REQUIREMENTS
