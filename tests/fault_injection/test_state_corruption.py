"""Inject state violations and verify enforcement."""
import pytest

def test_wrong_type_raises():
    """Writing wrong type to state field raises ValidationError."""
    from core.state import ProfessorState
    from pydantic import ValidationError
    
    state = ProfessorState()
    with pytest.raises(ValidationError):
        ProfessorState.validated_update(state, "preflight", {"preflight_passed": "definitely_not_a_boolean"})

def test_ownership_violation_raises():
    """Agent writing to another agent's field raises OwnershipError."""
    from core.state import ProfessorState, OwnershipError
    
    state = ProfessorState()
    with pytest.raises(OwnershipError):
        ProfessorState.validated_update(state, "feature_factory", {"critic_verdict": {"severity": "CLEAR"}})

def test_immutable_field_rewrite_raises():
    """Writing to canonical_train_rows after first write raises ImmutableFieldError."""
    from core.state import ProfessorState, ImmutableFieldError
    
    state = ProfessorState()
    state = ProfessorState.validated_update(state, "data_engineer", {"canonical_train_rows": 1000})
    with pytest.raises(ImmutableFieldError):
        ProfessorState.validated_update(state, "data_engineer", {"canonical_train_rows": 2000})
