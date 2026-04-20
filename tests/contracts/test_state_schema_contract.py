# tests/contracts/test_state_schema_contract.py

import pytest
from pydantic import ValidationError
from graph.state import ProfessorState, OwnershipError, ImmutableFieldError, SchemaVersionError

def test_type_validation_error():
    """Writing a list to a str field raises ValidationError — test with eda_insights_summary"""
    state = ProfessorState()
    with pytest.raises(ValidationError):
        # We use validated_update because agents never write fields directly
        ProfessorState.validated_update(state, "eda_agent", {"eda_insights_summary": ["not", "a", "str"]})

def test_ownership_error():
    """Writing from wrong agent raises OwnershipError — test feature_factory writing to critic_verdict"""
    state = ProfessorState()
    with pytest.raises(OwnershipError):
        # feature_factory does not own 'critic_verdict'
        ProfessorState.validated_update(state, "feature_factory", {"critic_verdict": {"severity": "HIGH"}})

def test_immutability_enforcement():
    """canonical_train_rows can be written once but not overwritten."""
    state = ProfessorState()
    # First write (from default 0 to 1000) should succeed
    state = ProfessorState.validated_update(state, "data_engineer", {"canonical_train_rows": 1000})
    assert state.canonical_train_rows == 1000
    
    # Second write should fail
    with pytest.raises(ImmutableFieldError):
        ProfessorState.validated_update(state, "data_engineer", {"canonical_train_rows": 2000})

def test_state_size_truncation():
    """State exceeding 20MB triggers truncation."""
    state = ProfessorState()
    # Create a massive list to exceed 20MB
    # Each entry is ~100 bytes, 250,000 entries ~ 25MB
    massive_list = [{"msg": "test data " * 10}] * 250000
    
    # We must set this directly to bypass validated_update logic for a moment to set up the "over budget" state
    # or just use validated_update if it allows the first big write.
    state = ProfessorState.validated_update(state, "hitl_listener", {"hitl_messages_sent": massive_list})
    
    # Verify it was truncated to 50 as per STATE.md rules
    assert len(state.hitl_messages_sent) == 50
    assert state.state_size_bytes < (20 * 1024 * 1024)

def test_core_fields_survive_truncation():
    """cv_mean survives truncation even when state is over budget."""
    state = ProfessorState()
    state = ProfessorState.validated_update(state, "ml_optimizer", {"cv_mean": 0.85})
    
    # Trigger truncation
    massive_list = [{"msg": "x"}] * 300000 
    state = ProfessorState.validated_update(state, "hitl_listener", {"hitl_messages_sent": massive_list})
    
    assert state.cv_mean == 0.85

def test_mutation_logging():
    """Every mutation is logged with correct agent attribution."""
    state = ProfessorState()
    state = ProfessorState.validated_update(state, "supervisor", {"competition_name": "titanic"})
    
    assert len(state.state_mutations_log) > 0
    last_log = state.state_mutations_log[-1]
    assert last_log["agent"] == "supervisor"
    assert last_log["field"] == "competition_name"
    assert "old_hash" in last_log
    assert "new_hash" in last_log

def test_schema_migration_v1_to_v2():
    """Schema version 'v1.0' triggers migration that adds v2 fields with defaults."""
    v1_data = {
        "session_id": "old_session",
        "state_schema_version": "v1.0",
        "cv_mean": 0.9
    }
    migrated = ProfessorState.validate_checkpoint_version(v1_data)
    assert migrated["state_schema_version"] == "v2.0"
    
    # Test instantiation
    state = ProfessorState(**migrated)
    assert state.session_id == "old_session"
    assert state.eda_insights_summary == "" # New v2 field

def test_schema_version_error():
    """Schema version 'v3.0' raises SchemaVersionError."""
    v3_data = {"state_schema_version": "v3.0"}
    with pytest.raises(SchemaVersionError):
        ProfessorState.validate_checkpoint_version(v3_data)

def test_ownership_strict_disabled():
    """OWNERSHIP_STRICT=False logs warning but doesn't raise."""
    import graph.state
    original_strict = graph.state.OWNERSHIP_STRICT
    graph.state.OWNERSHIP_STRICT = False
    try:
        state = ProfessorState()
        # feature_factory writing to supervisor field should normally fail
        state = ProfessorState.validated_update(state, "feature_factory", {"session_id": "illegal"})
        assert state.session_id == "illegal"
    finally:
        graph.state.OWNERSHIP_STRICT = original_strict
