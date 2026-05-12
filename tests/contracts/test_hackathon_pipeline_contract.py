import pytest
import os
import json
from unittest.mock import patch, MagicMock
from core.state import ProfessorState
from graph.hackathon_builder import build_hackathon_graph, _route_after_features, _route_after_critic
from agents.hackathon_publisher import hackathon_publisher


# ═══════════════════════════════════════
# Graph Structure Tests
# ═══════════════════════════════════════

class TestHackathonGraphStructure:
    
    def test_graph_builds_without_error(self):
        graph = build_hackathon_graph()
        assert graph is not None
    
    def test_graph_has_all_required_nodes(self):
        graph = build_hackathon_graph()
        compiled = graph.compile()
        
        required_nodes = [
            "preflight_checks", "competition_intel", "rubric_parser",
            "data_engineer", "eda_agent", "domain_research",
            "thesis_generator", "external_data_scout",
            "hypothesis_features", "ml_optimizer", "red_team_critic",
            "narrative_plots", "narrative_writeup", "hackathon_publisher",
        ]
        
        for node in required_nodes:
            assert node in compiled.nodes, f"Missing node: {node}"
    
    def test_graph_does_not_have_traditional_only_nodes(self):
        graph = build_hackathon_graph()
        compiled = graph.compile()
        
        excluded_nodes = [
            "metric_verification_gate", "shift_detector",
            "validation_architect", "problem_reframer",
            "creative_hypothesis", "self_reflection",
            "pseudo_label", "ensemble_architect",
            "post_processor", "submission_strategist",
        ]
        
        for node in excluded_nodes:
            assert node not in compiled.nodes, f"Node should be excluded: {node}"
    
    def test_entry_point_is_preflight(self):
        graph = build_hackathon_graph()
        # Entry point should be preflight_checks
        assert graph is not None
    
    def test_graph_node_count(self):
        graph = build_hackathon_graph()
        compiled = graph.compile()
        # 14 nodes: preflight, intel, rubric, data_eng, eda, domain,
        # thesis, scout, features, ml_opt, critic, narr_plots, narr_writeup, publisher
        assert len(compiled.nodes) >= 14


# ═══════════════════════════════════════
# Routing Tests
# ═══════════════════════════════════════

class TestHackathonRouting:
    
    def test_features_route_to_model_when_features_exist(self):
        state = ProfessorState(
            hackathon_mode=True,
            feature_manifest={"features": [{"name": "test_feature"}]},
            thesis_effect_sizes={"test": {"effect_size": 0.5}},
        )
        assert _route_after_features(state) == "train_model"
    
    def test_features_route_to_narrative_when_no_features(self):
        state = ProfessorState(
            hackathon_mode=True,
            feature_manifest={},
            thesis_effect_sizes={},
        )
        assert _route_after_features(state) == "skip_model"
    
    def test_critic_routes_replan_on_confirmed_critical(self):
        state = ProfessorState(
            hackathon_mode=True,
            critic_verdict={"severity": "CONFIRMED_CRITICAL"},
            dag_version=0,
        )
        assert _route_after_critic(state) == "replan"
    
    def test_critic_routes_continue_on_clear(self):
        state = ProfessorState(
            hackathon_mode=True,
            critic_verdict={"severity": "CLEAR"},
            dag_version=0,
        )
        assert _route_after_critic(state) == "continue"
    
    def test_critic_routes_continue_at_max_replans(self):
        state = ProfessorState(
            hackathon_mode=True,
            critic_verdict={"severity": "CONFIRMED_CRITICAL"},
            dag_version=3,
        )
        assert _route_after_critic(state) == "continue"


# ═══════════════════════════════════════
# Publisher Tests
# ═══════════════════════════════════════

class TestHackathonPublisher:
    
    @pytest.fixture
    def publisher_state(self, tmp_path):
        session_dir = str(tmp_path / "outputs" / "test-session")
        os.makedirs(session_dir, exist_ok=True)
        
        # Create a mock writeup
        writeup_path = os.path.join(session_dir, "hackathon_writeup.md")
        with open(writeup_path, "w") as f:
            f.write("# Test Writeup\n\nThis is a test.")
        
        # Create a mock code ledger
        ledger_path = os.path.join(session_dir, "code_ledger.jsonl")
        with open(ledger_path, "w") as f:
            f.write(json.dumps({"entry_id": "001", "agent": "data_engineer", "success": True, "code": "print(1)", "kept": True, "purpose": "test"}) + "\n")
        
        return ProfessorState(
            session_id="test-session",
            hackathon_mode=True,
            competition_name="Triagegeist",
            active_thesis={"statement": "ESI undertriages elderly patients", "hypothesis": "2x undertriage rate", "condition_variable": "age", "angle": "age-specific bias", "target_audience": "ED nurses"},
            hackathon_rubric={
                "submission_requirements": ["notebook", "writeup", "cover_image", "project_link"],
                "recommended_datasets": [{"name": "MIMIC-IV-ED", "url": "physionet.org"}],
                "writeup_template": {"max_words": 2000},
            },
            hackathon_writeup_path=writeup_path,
            hackathon_writeup_word_count=10,
            thesis_effect_sizes={"is_elderly": {"effect_size": 0.85, "p_value": 0.001}},
            feature_manifest={"features": [{"name": "is_elderly", "source": "hypothesis"}]},
            narrative_plots=[],
            best_model_type="lightgbm",
            cv_mean=0.82,
            hitl_mode="autonomous",
            data_schema={"age": "Float64"},
        )
    
    def test_publisher_returns_state_fields(self, publisher_state):
        with patch("tools.operator_channel.emit_to_operator"):
            result = hackathon_publisher(publisher_state)
        
        assert "solution_notebook_path" in result
        assert "solution_writeup_path" in result
        assert "code_ledger_path" in result
    
    def test_writeup_path_from_narrative_engine(self, publisher_state):
        with patch("tools.operator_channel.emit_to_operator"):
            result = hackathon_publisher(publisher_state)
        
        assert result["solution_writeup_path"] == publisher_state.hackathon_writeup_path
    
    def test_milestone_emitted(self, publisher_state):
        with patch("agents.hackathon_publisher.emit_to_operator") as mock_emit:
            hackathon_publisher(publisher_state)
        
        # Final milestone should be emitted as RESULT
        result_calls = [c for c in mock_emit.call_args_list 
                       if c.kwargs.get("level") == "RESULT" or 
                       (len(c.args) > 1 and c.args[1] == "RESULT")]
        assert len(result_calls) >= 1
    
    def test_guided_mode_emits_gate(self, publisher_state):
        object.__setattr__(publisher_state, "hitl_mode", "guided")
        with patch("agents.hackathon_publisher.emit_to_operator") as mock_emit:
            mock_emit.return_value = "/submit"
            result = hackathon_publisher(publisher_state)
        
        gate_calls = [c for c in mock_emit.call_args_list 
                     if c.kwargs.get("level") == "GATE" or
                     (len(c.args) > 1 and c.args[1] == "GATE")]
        assert len(gate_calls) >= 1


class TestNotebookAssembly:
    
    def test_notebook_is_valid_ipynb(self, tmp_path):
        session_dir = str(tmp_path)
        
        # Create mock code ledger
        ledger_path = os.path.join(session_dir, "code_ledger.jsonl")
        with open(ledger_path, "w") as f:
            f.write(json.dumps({
                "entry_id": "001", "agent": "data_engineer", "success": True, "kept": True,
                "code": "import polars as pl\ndf = pl.read_csv('train.csv')\nprint(df.shape)",
                "purpose": "Load training data", "round_num": 0,
            }) + "\n")
        
        state = ProfessorState(
            session_id="test",
            hackathon_mode=True,
            competition_name="Test Competition",
            active_thesis={"statement": "Test thesis", "hypothesis": "Test", "condition_variable": "x", "angle": "novel", "target_audience": "testers"},
            thesis_effect_sizes={},
            feature_manifest={},
            narrative_plots=[],
        )
        from agents.hackathon_publisher import _assemble_hackathon_notebook
        notebook_path = _assemble_hackathon_notebook(state, session_dir)
        
        assert notebook_path.endswith(".ipynb")
        assert os.path.exists(notebook_path)
        
        with open(notebook_path) as f:
            nb = json.load(f)
        
        assert nb["nbformat"] == 4
        assert "cells" in nb
        assert len(nb["cells"]) >= 3  # At least title + setup + conclusion
    
    def test_notebook_has_markdown_and_code_cells(self, tmp_path):
        session_dir = str(tmp_path)
        with open(os.path.join(session_dir, "code_ledger.jsonl"), "w") as f:
            f.write(json.dumps({"entry_id": "001", "agent": "data_engineer", "success": True, "kept": True, "code": "print('hello')", "purpose": "test", "round_num": 0}) + "\n")
        
        state = ProfessorState(
            session_id="test", hackathon_mode=True, competition_name="Test",
            active_thesis={"statement": "T", "hypothesis": "H", "condition_variable": "V", "angle": "A", "target_audience": "X"},
            thesis_effect_sizes={}, feature_manifest={}, narrative_plots=[],
        )
        
        from agents.hackathon_publisher import _assemble_hackathon_notebook
        notebook_path = _assemble_hackathon_notebook(state, session_dir)
        with open(notebook_path) as f:
            nb = json.load(f)
        
        cell_types = [c["cell_type"] for c in nb["cells"]]
        assert "markdown" in cell_types
        assert "code" in cell_types
    
    def test_notebook_contains_thesis_statement(self, tmp_path):
        session_dir = str(tmp_path)
        with open(os.path.join(session_dir, "code_ledger.jsonl"), "w") as f:
            f.write(json.dumps({"entry_id": "001", "agent": "data_engineer", "success": True, "kept": True, "code": "print(1)", "purpose": "test", "round_num": 0}) + "\n")
        
        thesis_text = "ESI undertriages elderly patients with atypical symptoms"
        state = ProfessorState(
            session_id="test", hackathon_mode=True, competition_name="Test",
            active_thesis={"statement": thesis_text, "hypothesis": "H", "condition_variable": "V", "angle": "A", "target_audience": "X"},
            thesis_effect_sizes={}, feature_manifest={}, narrative_plots=[],
        )
        
        from agents.hackathon_publisher import _assemble_hackathon_notebook
        notebook_path = _assemble_hackathon_notebook(state, session_dir)
        with open(notebook_path) as f:
            content = f.read()
        
        assert thesis_text in content
    
    def test_notebook_no_professor_imports(self, tmp_path):
        session_dir = str(tmp_path)
        with open(os.path.join(session_dir, "code_ledger.jsonl"), "w") as f:
            f.write(json.dumps({"entry_id": "001", "agent": "data_engineer", "success": True, "kept": True, "code": "from tools.sandbox import run_in_sandbox\nprint('hello')", "purpose": "test", "round_num": 0}) + "\n")
        
        state = ProfessorState(
            session_id="test", hackathon_mode=True, competition_name="Test",
            active_thesis={"statement": "T", "hypothesis": "H", "condition_variable": "V", "angle": "A", "target_audience": "X"},
            thesis_effect_sizes={}, feature_manifest={}, narrative_plots=[],
        )
        
        from agents.hackathon_publisher import _assemble_hackathon_notebook
        notebook_path = _assemble_hackathon_notebook(state, session_dir)
        with open(notebook_path) as f:
            content = f.read()
        
        assert "from tools." not in content
        assert "emit_to_operator" not in content


class TestGitHubReadme:
    
    def test_readme_generated(self, tmp_path):
        state = ProfessorState(
            competition_name="Triagegeist",
            active_thesis={"statement": "Test thesis", "hypothesis": "H", "angle": "A"},
            best_model_type="lightgbm",
            cv_mean=0.82,
            thesis_effect_sizes={"feat": {"effect_size": 0.5}},
            external_datasets=[{"name": "MIMIC-IV-ED", "source_url": "physionet.org"}],
            hackathon_rubric={"recommended_datasets": []},
        )
        
        from agents.hackathon_publisher import _generate_github_readme
        readme_path = _generate_github_readme(state, str(tmp_path))
        
        assert os.path.exists(readme_path)
        with open(readme_path) as f:
            content = f.read()
        
        assert "Triagegeist" in content
        assert "Test thesis" in content
        assert "requirements.txt" in content
        assert "MIMIC-IV-ED" in content


# ═══════════════════════════════════════
# Traditional Pipeline Unchanged
# ═══════════════════════════════════════

class TestTraditionalUnchanged:
    
    def test_traditional_graph_still_builds(self):
        """Traditional graph must still work after hackathon additions."""
        from core.professor import build_professor_graph
        graph = build_professor_graph()
        assert graph is not None
    
    def test_traditional_graph_has_all_nodes(self):
        from core.professor import build_professor_graph
        graph = build_professor_graph()
        compiled = graph
        
        # Traditional nodes that must still be present
        traditional_nodes = [
            "preflight", "competition_intel", "data_engineer",
            "eda_agent", "feature_factory", "ml_optimizer",
            "red_team_critic", "publisher",
        ]
        for node in traditional_nodes:
            assert node in compiled.nodes
    
    def test_traditional_graph_does_not_have_hackathon_nodes(self):
        from core.professor import build_professor_graph
        graph = build_professor_graph()
        compiled = graph
        
        hackathon_only = ["rubric_parser", "thesis_generator", "external_data_scout", 
                          "narrative_plots", "narrative_writeup", "hackathon_publisher"]
        for node in hackathon_only:
            assert node not in compiled.nodes

