# tests/contracts/test_state_schema_contract.py

import pytest
from pydantic import ValidationError
from core.state import ProfessorState, OwnershipError, ImmutableFieldError, SchemaVersionError

def test_state_mapping_protocol():
    """Verify state supports dict-like access for legacy compatibility."""
    state = ProfessorState(session_id="test_session")
    
    # Read access
    assert state["session_id"] == "test_session"
    assert state.get("session_id") == "test_session"
    assert "session_id" in state
    
    # Write access (should use validated_update normally, but __setitem__ is for compatibility)
    state["competition_name"] = "test_comp"
    assert state.competition_name == "test_comp"
    assert state["competition_name"] == "test_comp"

def test_state_type_validation():
    """Verify Pydantic enforces types at runtime."""
    state = ProfessorState()
    
    # Writing a list to a str field should raise ValidationError
    with pytest.raises(ValidationError):
        # We use model_validate or similar because direct assignment might bypass 
        # some Pydantic v2 checks depending on ConfigDict, 
        # but validated_update will definitely catch it.
        ProfessorState.validated_update(state, "eda_agent", {"eda_insights_summary": ["not", "a", "string"]})

def test_field_ownership():
    """Verify only the owning agent can write to a field."""
    state = ProfessorState()
    
    # Correct owner: competition_intel owns competition_context
    updated = ProfessorState.validated_update(state, "competition_intel", {"competition_context": {"key": "val"}})
    assert updated.competition_context == {"key": "val"}
    
    # Wrong owner: feature_factory trying to write to target_col (owned by data_engineer)
    with pytest.raises(OwnershipError):
        ProfessorState.validated_update(state, "feature_factory", {"target_col": "illegal_write"})

def test_field_immutability():
    """Verify [IMMUTABLE] fields can only be set once."""
    state = ProfessorState()
    
    # First write (from default 0 to 1000) - Allowed
    state = ProfessorState.validated_update(state, "data_engineer", {"canonical_train_rows": 1000})
    assert state.canonical_train_rows == 1000
    
    # Second write (from 1000 to 2000) - Should raise error
    with pytest.raises(ImmutableFieldError):
        ProfessorState.validated_update(state, "data_engineer", {"canonical_train_rows": 2000})

def test_state_size_truncation():
    """Verify state truncates logs when exceeding budget."""
    state = ProfessorState()
    
    # Create a massive log to exceed 20MB
    # 20MB is roughly 20 million characters
    large_list = ["message" * 100] * 200000 
    
    # We update a field that is NOT in the truncation list first to check size
    state.hitl_messages_sent = large_list
    state._check_size()
    
    # hitl_messages_sent should be truncated to 50
    assert len(state.hitl_messages_sent) <= 50
    
    # Core fields like cv_mean must NOT be truncated
    state.cv_mean = 0.85
    state._check_size()
    assert state.cv_mean == 0.85

def test_mutation_logging():
    """Verify every mutation is logged with hashes."""
    state = ProfessorState()
    
    state = ProfessorState.validated_update(state, "data_engineer", {"target_col": "target"})
    
    assert len(state.state_mutations_log) >= 1
    log_entry = state.state_mutations_log[-1]
    assert log_entry["agent"] == "data_engineer"
    assert log_entry["field"] == "target_col"
    assert "new_hash" in log_entry
    assert "timestamp" in log_entry

def test_schema_version_migration():
    """Verify v1.0 data is migrated to v2.0."""
    v1_data = {
        "session_id": "v1_session",
        "state_schema_version": "v1.0"
    }
    
    migrated = ProfessorState.validate_checkpoint_version(v1_data)
    assert migrated["state_schema_version"] == "v2.0"
    
    # Recognized version
    ProfessorState.validate_checkpoint_version({"state_schema_version": "v2.0"})
    
    # Incompatible version
    with pytest.raises(SchemaVersionError):
        ProfessorState.validate_checkpoint_version({"state_schema_version": "v3.0"})
