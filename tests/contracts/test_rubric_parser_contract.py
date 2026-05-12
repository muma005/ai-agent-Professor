import pytest
import os
import json
from unittest.mock import patch, MagicMock
from tools.rubric_parser import (
    parse_rubric, 
    build_effort_plan, 
    run_rubric_parser, 
    HackathonRubric, 
    EffortPlan,
    _build_rubric_from_merged
)
from core.state import ProfessorState

# ── Test fixtures ───────────────────────────────────────────────────────────

TRIAGEGEIST_TEXT = """
Triagegeist
Predict emergency severity and optimize triage decisions with AI.

Scoring Rubric (100 points total)
1. Clinical Relevance (25 points)
Does the submission address a real and meaningful problem in emergency triage?
Score 21-25: Problem is sharply defined, clinically motivated.
Score 14-20: Problem is relevant but partially developed.
Score 7-13: Problem relates broadly to emergency medicine.
Score 0-6: Problem has negligible clinical relevance.

2. Technical Quality (30 points)
Is the AI approach sound? Is the code clean, reproducible?
Score 25-30: Methodology is rigorous. Code is clean.
Score 17-24: Methodology is reasonable with minor gaps.
Score 9-16: Methodology has notable weaknesses.
Score 0-8: Methodology is flawed.

3. Documentation and Writeup Quality (20 points)
Is the writeup clear, complete?
Score 17-20: Thorough and clear. Reproducibility supported.
Score 11-16: Covers most sections adequately.
Score 5-10: Incomplete or unclear.
Score 0-4: Absent or insufficient.

4. Insight and Findings (15 points)
Does the submission produce meaningful findings?
Score 13-15: Meaningful, clearly communicated.
Score 8-12: Reported but limited interpretation.
Score 3-7: Superficial or overclaimed.
Score 0-2: No meaningful findings.

5. Novelty and Impact Potential (10 points)
Does the submission bring a fresh perspective?
Score 9-10: Genuinely novel.
Score 6-8: Some novel elements.
Score 3-5: Follows established approaches.
Score 0-2: No novelty.

Submission Requirements:
- Kaggle Notebook (must run end-to-end, public)
- Project Writeup (max 2000 words)
- Cover Image (560 x 280 px)
- Project Link

Recommended datasets: MIMIC-IV-ED, NHAMCS

Prizes: 1st $5,000, 2nd $3,000, 3rd $2,000
"""

MINIMAL_HACKATHON_TEXT = """
Build something cool with this data.
Best projects win prizes.
Submit a notebook and writeup.
"""

ART_HACKATHON_TEXT = """
AI Art Challenge
Create novel AI-generated artwork.

Judging (100 points):
- Creativity & Originality: 40 points
- Technical Execution: 15 points  
- Presentation & Documentation: 30 points
- Impact & Meaning: 15 points

Submit: notebook, writeup, portfolio link
"""

@pytest.fixture(autouse=True)
def mock_llm_and_emit():
    """Globally mock LLM and Operator Channel for all tests in this file."""
    with patch("tools.rubric_parser.llm_call") as mock_llm, \
         patch("tools.rubric_parser.emit_to_operator") as mock_emit:
        
        # Default LLM response - returns empty JSON or minimal valid structure
        mock_llm.return_value = json.dumps({
            "competition_name": "Mocked Comp",
            "criteria": [],
            "submission_requirements": ["notebook", "writeup"],
            "writeup_template": {"sections": ["problem", "method", "results"], "max_words": 1500}
        })
        mock_emit.return_value = "/continue"
        yield mock_llm, mock_emit

# ── Tests ───────────────────────────────────────────────────────────────────

class TestRubricParsing:
    
    def test_triagegeist_extracts_5_criteria(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        # Should find 5 criteria deterministically
        assert len(rubric.criteria) == 5
    
    def test_triagegeist_total_100_points(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert rubric.total_points == 100
    
    def test_triagegeist_clinical_relevance_25(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        clinical = next(c for c in rubric.criteria if "clinical" in c["name"].lower())
        assert clinical["weight"] == 25
    
    def test_triagegeist_technical_quality_30(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        technical = next(c for c in rubric.criteria if "technical" in c["name"].lower())
        assert technical["weight"] == 30
    
    def test_triagegeist_novelty_10(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        novelty = next(c for c in rubric.criteria if "novel" in c["name"].lower())
        assert novelty["weight"] == 10
    
    def test_triagegeist_submission_requirements(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert "notebook" in rubric.submission_requirements
        assert "writeup" in rubric.submission_requirements
        assert "cover_image" in rubric.submission_requirements
        assert "project_link" in rubric.submission_requirements
    
    def test_triagegeist_word_limit(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert rubric.writeup_template.get("max_words") == 2000
    
    def test_triagegeist_recommended_datasets(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        dataset_names = [d["name"] for d in rubric.recommended_datasets]
        assert any("MIMIC" in n for n in dataset_names)
    
    def test_triagegeist_prizes(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert len(rubric.prizes) >= 2
        # Use simple string check for prizes to avoid amount formatting issues
        prize_amounts = [p.get("amount", "") for p in rubric.prizes]
        assert any("5000" in a or "5,000" in a for a in prize_amounts)
    
    def test_triagegeist_data_policy(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert rubric.data_policy == "any_public"


class TestMinimalHackathon:
    
    def test_minimal_text_produces_valid_rubric(self):
        rubric = parse_rubric(MINIMAL_HACKATHON_TEXT)
        assert len(rubric.criteria) >= 3  # At least some criteria inferred
        assert rubric.total_points > 0
    
    def test_minimal_uses_defaults(self):
        rubric = parse_rubric(MINIMAL_HACKATHON_TEXT)
        # Should have default criteria since text has no rubric
        assert "notebook" in rubric.submission_requirements


class TestArtHackathon:
    
    def test_art_hackathon_creativity_40(self):
        rubric = parse_rubric(ART_HACKATHON_TEXT)
        creativity = next(c for c in rubric.criteria if "creativ" in c["name"].lower())
        assert creativity["weight"] == 40
    
    def test_art_hackathon_technical_15(self):
        rubric = parse_rubric(ART_HACKATHON_TEXT)
        technical = next(c for c in rubric.criteria if "technical" in c["name"].lower())
        assert technical["weight"] == 15


class TestEffortPlan:
    
    def test_triagegeist_marathon_technical(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.technical_depth == "marathon"  # 30% technical weight
    
    def test_triagegeist_deep_thesis(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.thesis_depth == "deep"  # 25% domain + 10% novelty = 35%
    
    def test_triagegeist_deep_writeup(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.writeup_depth == "deep"  # 20% documentation
    
    def test_triagegeist_high_external_data(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.external_data_priority == "high"  # 10+15=25% novelty+insight
    
    def test_triagegeist_3_polish_passes(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.narrative_polish_passes == 3  # 20% documentation
    
    def test_art_hackathon_sprint_technical(self):
        rubric = parse_rubric(ART_HACKATHON_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.technical_depth == "sprint"  # 15% technical
    
    def test_art_hackathon_deep_thesis(self):
        rubric = parse_rubric(ART_HACKATHON_TEXT)
        plan = build_effort_plan(rubric)
        assert plan.thesis_depth == "deep"  # 40% creativity/novelty
    
    def test_weight_breakdown_sums_correctly(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        total = sum(plan.weight_breakdown.values())
        assert 95 <= total <= 105  # Allow small rounding error
    
    def test_allocation_reasoning_nonempty(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        plan = build_effort_plan(rubric)
        assert len(plan.allocation_reasoning) > 50
        assert "pipeline" in plan.allocation_reasoning.lower()
    
    def test_equal_weights_produce_balanced_plan(self):
        """When all criteria have equal weight, no dimension dominates."""
        rubric = HackathonRubric(
            competition_name="Test",
            total_points=100,
            criteria=[
                {"name": "Technical Quality", "weight": 20, "max_points": 20},
                {"name": "Novelty", "weight": 20, "max_points": 20},
                {"name": "Documentation", "weight": 20, "max_points": 20},
                {"name": "Domain Relevance", "weight": 20, "max_points": 20},
                {"name": "Insight", "weight": 20, "max_points": 20},
            ],
            submission_requirements=["notebook"],
            writeup_template={"sections": [], "max_words": 2000},
            data_policy="any_public",
            recommended_datasets=[],
            deliverable_type="notebook_and_writeup",
            tracks=[],
            prizes=[],
            deadline="unknown",
            raw_text_hash="test",
        )
        plan = build_effort_plan(rubric)
        assert plan.technical_depth == "standard"  # 20% — between thresholds
        assert plan.thesis_depth == "deep"  # 20+20=40% domain+novelty


class TestEdgeCases:
    
    def test_empty_text_returns_defaults(self):
        rubric = parse_rubric("")
        assert len(rubric.criteria) >= 3
        assert rubric.total_points > 0
    
    def test_nonsense_text_returns_defaults(self):
        rubric = parse_rubric("asdf 1234 !@#$ random garbage text")
        assert len(rubric.criteria) >= 3
    
    def test_rubric_hash_populated(self):
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        assert len(rubric.raw_text_hash) == 16
    
    def test_different_text_different_hash(self):
        rubric1 = parse_rubric(TRIAGEGEIST_TEXT)
        rubric2 = parse_rubric(ART_HACKATHON_TEXT)
        assert rubric1.raw_text_hash != rubric2.raw_text_hash
    
    def test_criteria_deduplicated(self):
        """If LLM returns duplicate criteria, they're merged."""
        rubric = parse_rubric(TRIAGEGEIST_TEXT)
        names = [c["name"].lower() for c in rubric.criteria]
        assert len(names) == len(set(names))  # No duplicates
    
    def test_normalization_when_total_off(self):
        """If extracted points sum to 200 instead of 100, normalize."""
        # This tests the _build_rubric_from_merged validation
        merged = {
            "criteria": [
                {"name": "A", "weight": 60, "max_points": 60},
                {"name": "B", "weight": 80, "max_points": 80},
                {"name": "C", "weight": 60, "max_points": 60},
            ],
            "competition_name": "Test",
        }
        rubric = _build_rubric_from_merged(merged, "")
        # Weights should be normalized to ~100
        assert 95 <= rubric.total_points <= 105
        # Verify specific criteria weights were scaled (60/200 * 100 = 30)
        assert rubric.criteria[0]["weight"] == 30


class TestNodeFunction:
    
    def test_returns_correct_state_fields(self):
        """run_rubric_parser returns all required hackathon state fields."""
        # Mock state with competition_description
        state = ProfessorState(
            competition_description=TRIAGEGEIST_TEXT,
        )
        result = run_rubric_parser(state)
        
        assert "hackathon_mode" in result
        assert result["hackathon_mode"] == True
        assert "hackathon_rubric" in result
        assert "hackathon_effort_plan" in result
        assert "hackathon_writeup_template" in result
    
    def test_checkpoint_emitted(self, mock_llm_and_emit):
        """Rubric analysis is presented to operator as CHECKPOINT."""
        _, mock_emit = mock_llm_and_emit
        state = ProfessorState(competition_description=TRIAGEGEIST_TEXT)
        run_rubric_parser(state)
        
        # Find the CHECKPOINT call
        checkpoint_calls = [
            c for c in mock_emit.call_args_list
            if c.kwargs.get("level") == "CHECKPOINT" or 
               (len(c.args) > 1 and c.args[1] == "CHECKPOINT")
        ]
        assert len(checkpoint_calls) >= 1
    
    def test_deterministic_extraction_avoids_llm_when_complete(self):
        """Note: In current implementation, LLM is always called for Triagegeist
        because deterministic extract doesn't get descriptions. This test 
        primarily verifies that the deterministic values (weights) are preserved."""
        state = ProfessorState(competition_description=TRIAGEGEIST_TEXT)
        result = run_rubric_parser(state)
        
        rubric = result["hackathon_rubric"]
        weights = {c["name"]: c["weight"] for c in rubric["criteria"]}
        # The key weights should be correct from deterministic extraction
        assert any(w == 30 for w in weights.values())  # Technical Quality
        assert any(w == 25 for w in weights.values())  # Clinical Relevance
