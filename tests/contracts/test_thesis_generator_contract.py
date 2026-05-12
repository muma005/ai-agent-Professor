import pytest
import json
from unittest.mock import patch, MagicMock
from agents.thesis_generator import (
    thesis_generator,
    _generate_thesis_candidates,
    _parse_operator_selection,
    _classify_condition,
    _validate_candidate,
    ThesisCandidate
)
from core.state import ProfessorState

@pytest.fixture
def hackathon_state():
    """State with rubric parsed and EDA/domain complete — ready for thesis generation."""
    return ProfessorState(
        session_id="test-thesis",
        competition_name="Triagegeist",
        competition_description="Build an AI-powered tool for emergency triage...",
        hackathon_mode=True,
        hackathon_rubric={
            "competition_name": "Triagegeist",
            "total_points": 100,
            "criteria": [
                {"name": "Clinical Relevance", "weight": 25, "max_points": 25,
                 "description": "Real problem in emergency triage",
                 "top_score_description": "Sharply defined, clinically motivated"},
                {"name": "Technical Quality", "weight": 30, "max_points": 30,
                 "description": "Sound methodology, clean code"},
                {"name": "Documentation", "weight": 20, "max_points": 20,
                 "description": "Clear writeup"},
                {"name": "Insight", "weight": 15, "max_points": 15,
                 "description": "Meaningful findings"},
                {"name": "Novelty", "weight": 10, "max_points": 10,
                 "description": "Fresh perspective"},
            ],
            "recommended_datasets": [{"name": "MIMIC-IV-ED", "url": "physionet.org", "description": "ED data"}],
        },
        hackathon_effort_plan={
            "thesis_depth": "deep",
            "technical_depth": "marathon",
        },
        domain_classification="healthcare",
        domain_brief={
            "primary_domain": "healthcare",
            "sub_classification": "emergency triage",
            "column_semantics": {"triage_level": {"meaning": "ESI acuity score 1-5"}},
            "known_relationships": [{"features": ["age", "vital_signs"], "relationship": "Age modifies vital sign thresholds"}],
            "domain_summary": "Emergency triage assigns severity levels to incoming patients.",
        },
        data_schema={"patient_id": "Int64", "age": "Float64", "heart_rate": "Float64", 
                      "bp_systolic": "Float64", "chief_complaint": "Utf8", "triage_level": "Int64"},
        eda_insights_summary="Dataset has 45K rows. Age distribution is bimodal (peaks at 35 and 72). Heart rate MI with triage_level is 0.34.",
        eda_mutual_info={"target_mi": [{"heart_rate": 0.34}, {"bp_systolic": 0.28}, {"age": 0.22}]},
        eda_modality_flags=["age"],
    )


MOCK_LLM_THESIS_RESPONSE = json.dumps([
    {
        "thesis_id": 1,
        "statement": "ESI undertriages elderly patients with atypical cardiac presentations",
        "angle": "Age × presentation interaction reveals systematic bias in ESI scoring",
        "target_audience": "ED triage nurses and ESI algorithm designers",
        "data_plan": {
            "primary_dataset": "MIMIC-IV-ED",
            "external_needed": ["AHA cardiac risk thresholds by age"],
            "join_strategy": "Lookup table join on age_group"
        },
        "condition_variable": "age_group × presentation_type",
        "hypothesis": "Patients >65 with atypical cardiac symptoms are undertriaged 2x more than those with typical presentations",
        "estimated_scores": {
            "Clinical Relevance": {"score": 23, "justification": "Sharply defined, documented problem"},
            "Technical Quality": {"score": 26, "justification": "Standard ML approach sufficient"},
            "Documentation": {"score": 16, "justification": "Clear narrative arc"},
            "Insight": {"score": 12, "justification": "Age-adjusted thresholds actionable"},
            "Novelty": {"score": 7, "justification": "Known concern, novel data analysis"}
        },
        "estimated_total": 84,
        "feasibility": "high",
        "risk": "MIMIC-IV-ED may not have enough atypical cardiac cases"
    },
    {
        "thesis_id": 2,
        "statement": "Night shift triage accuracy degrades for medium-acuity patients",
        "angle": "Cognitive fatigue isolated from volume effects",
        "target_audience": "Hospital administrators designing shift schedules",
        "data_plan": {
            "primary_dataset": "MIMIC-IV-ED",
            "external_needed": ["Published shift fatigue studies"],
            "join_strategy": "Time-based filtering"
        },
        "condition_variable": "shift_period × acuity_level",
        "hypothesis": "Medium-acuity patients triaged during night shift are undertriaged 30% more",
        "estimated_scores": {
            "Clinical Relevance": {"score": 21, "justification": "Relevant but less urgent"},
            "Technical Quality": {"score": 25, "justification": "Requires causal analysis"},
            "Documentation": {"score": 16, "justification": "Compelling narrative"},
            "Insight": {"score": 11, "justification": "Staffing implications"},
            "Novelty": {"score": 6, "justification": "Known topic, per-acuity is new"}
        },
        "estimated_total": 79,
        "feasibility": "medium",
        "risk": "Timestamps may not distinguish shift boundaries"
    }
] + [
    {"thesis_id": i, "statement": f"Thesis {i} is more than twenty characters long to pass test", "angle": "angle", "target_audience": "audience",
     "data_plan": {"primary_dataset": "data", "external_needed": [], "join_strategy": ""},
     "condition_variable": "var", "hypothesis": "hypothesis is also very long and specific",
     "estimated_scores": {"Clinical Relevance": {"score": 15, "justification": "ok"},
                           "Technical Quality": {"score": 20, "justification": "ok"},
                           "Documentation": {"score": 12, "justification": "ok"},
                           "Insight": {"score": 8, "justification": "ok"},
                           "Novelty": {"score": 5, "justification": "ok"}},
     "estimated_total": 60, "feasibility": "medium", "risk": "risk"}
    for i in range(3, 6)
])

@pytest.fixture(autouse=True)
def mock_llm_call():
    with patch("agents.thesis_generator.llm_call") as mock:
        # For _generate_thesis_candidates, it returns the JSON string
        mock.return_value = MOCK_LLM_THESIS_RESPONSE
        yield mock

@pytest.fixture(autouse=True)
def mock_emit():
    with patch("agents.thesis_generator.emit_to_operator") as mock:
        mock.return_value = "/continue"
        yield mock

class TestThesisGeneration:
    
    def test_generates_5_candidates(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        assert len(candidates) == 5
    
    def test_each_has_required_fields(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        for c in candidates:
            assert c.statement and len(c.statement) > 10
            assert c.angle
            assert c.target_audience
            assert isinstance(c.data_plan, dict)
            assert "primary_dataset" in c.data_plan
            assert "external_needed" in c.data_plan
            assert c.condition_variable
            assert c.hypothesis
            assert isinstance(c.estimated_scores, dict)
            assert c.estimated_total > 0
            assert c.feasibility in ("high", "medium", "low")
            assert c.risk
    
    def test_scores_cover_all_rubric_criteria(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        rubric_names = {c["name"] for c in hackathon_state.hackathon_rubric["criteria"]}
        for candidate in candidates:
            assert set(candidate.estimated_scores.keys()) == rubric_names
    
    def test_ranked_by_total_descending(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        totals = [c.estimated_total for c in candidates]
        assert totals == sorted(totals, reverse=True)
    
    def test_condition_variable_present_and_classified(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        for c in candidates:
            assert c.condition_variable != ""
            assert c.condition_type in ("categorical", "temporal", "spatial", "threshold", "interaction")
    
    def test_hypothesis_is_testable(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        for c in candidates:
            assert len(c.hypothesis) > 20
    
    def test_scores_clamped_to_rubric_max(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        criteria_max = {c["name"]: c.get("max_points", c.get("weight", 20))
                        for c in hackathon_state.hackathon_rubric["criteria"]}
        
        for candidate in candidates:
            for cname, entry in candidate.estimated_scores.items():
                assert entry["score"] <= criteria_max.get(cname, 100)
                assert entry["score"] >= 0
    
    def test_estimated_total_matches_scores(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        for c in candidates:
            computed = sum(entry["score"] for entry in c.estimated_scores.values())
            assert c.estimated_total == computed
    
    def test_rubric_alignment_score_populated(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        for c in candidates:
            assert 0.0 <= c.rubric_alignment_score <= 1.0


class TestOperatorSelection:
    
    def test_number_selects_thesis(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        result = _parse_operator_selection("2", candidates, hackathon_state.hackathon_rubric, hackathon_state)
        assert result["thesis_selected_by"] == "operator"
        assert result["active_thesis"]["statement"] != ""
    
    def test_slash_command_selects(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        result = _parse_operator_selection("/thesis select 1", candidates, hackathon_state.hackathon_rubric, hackathon_state)
        assert result["thesis_selected_by"] == "operator"
        assert result["active_thesis"]["thesis_id"] == 1
    
    def test_continue_auto_selects_first(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        result = _parse_operator_selection("/continue", candidates, hackathon_state.hackathon_rubric, hackathon_state)
        assert result["thesis_selected_by"] == "auto"
        assert result["active_thesis"]["thesis_id"] == 1  # Highest scored
    
    def test_timeout_auto_selects(self, hackathon_state):
        candidates = _generate_thesis_candidates(hackathon_state)
        
        result = _parse_operator_selection(None, candidates, hackathon_state.hackathon_rubric, hackathon_state)
        assert result["thesis_selected_by"] == "auto"
    
    def test_custom_thesis_evaluated(self, hackathon_state, mock_llm_call):
        mock_llm_call.return_value = json.dumps({
            "statement": "Custom thesis about medication interactions",
            "estimated_scores": {"Clinical Relevance": 20, "Technical Quality": 22, "Documentation": 15, "Insight": 10, "Novelty": 8},
            "estimated_total": 75, "feasibility": "medium", "risk": "low"
        })
        candidates = [_validate_candidate({}, i, set(), []) for i in range(1, 6)]
        
        result = _parse_operator_selection('/thesis custom "test"', candidates, hackathon_state.hackathon_rubric, hackathon_state)
        
        assert result["thesis_selected_by"] == "operator"
        assert result["active_thesis"]["thesis_id"] == 0
    
    def test_long_free_text_treated_as_custom(self, hackathon_state, mock_llm_call):
        mock_llm_call.return_value = json.dumps({
            "statement": "Custom long text thesis",
            "estimated_scores": {"Clinical Relevance": 10, "Technical Quality": 10, "Documentation": 10, "Insight": 10, "Novelty": 10},
            "estimated_total": 50, "feasibility": "medium", "risk": "low"
        })
        candidates = [_validate_candidate({}, i, set(), []) for i in range(1, 6)]
        result = _parse_operator_selection("I want to analyze repeated ED visits within 72 hours", candidates, hackathon_state.hackathon_rubric, hackathon_state)
        
        assert result["thesis_selected_by"] == "operator"


class TestEdgeCases:
    
    def test_malformed_json_produces_fallback(self, hackathon_state, mock_llm_call):
        mock_llm_call.return_value = "not json"
        candidates = _generate_thesis_candidates(hackathon_state)
        assert len(candidates) == 5
        assert "Placeholder" in candidates[0].statement
    
    def test_non_hackathon_mode_returns_empty(self):
        state = ProfessorState(hackathon_mode=False)
        result = thesis_generator(state)
        assert result == {}


class TestConditionClassification:
    
    def test_temporal_condition(self):
        assert _classify_condition("time_of_day") == "temporal"
    
    def test_categorical_condition(self):
        assert _classify_condition("age_group") == "categorical"


class TestNodeFunction:
    
    def test_returns_correct_state_fields(self, hackathon_state):
        with patch("agents.thesis_generator._parse_operator_selection") as mock_parse:
            mock_parse.return_value = {"active_thesis": {"statement": "test"}, "thesis_selected_by": "operator"}
            result = thesis_generator(hackathon_state)
        
        assert "thesis_candidates" in result
        assert "active_thesis" in result
        assert "thesis_selected_by" in result
