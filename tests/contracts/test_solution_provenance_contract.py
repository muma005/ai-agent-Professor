# tests/contracts/test_solution_provenance_contract.py

import pytest
import os
import json
from unittest.mock import patch, MagicMock
from tools.code_ledger import CodeLedger, CodeLedgerEntry
from tools.solution_assembler import assemble_standalone_solution, verify_reproduction

@pytest.fixture
def mock_ledger_entries():
    return [
        CodeLedgerEntry(
            entry_id="1", timestamp="2026-04-24T10:00:00Z", agent="data_engineer",
            purpose="cleaning", round=1, attempt=1, code="df = df.drop_nulls()",
            code_hash="h1", success=True, is_winning_component=True
        ),
        CodeLedgerEntry(
            entry_id="2", timestamp="2026-04-24T10:05:00Z", agent="feature_factory",
            purpose="engineering", round=1, attempt=1, code="df = df.with_columns(pl.lit(1).alias('f1'))",
            code_hash="h2", success=True, is_winning_component=True
        )
    ]

class TestSolutionProvenanceContract:
    """
    Contract: Solution Provenance (Component 3)
    """

    def test_ledger_tracks_provenance(self, tmp_path):
        """Verify ledger correctly stores and retrieves entries."""
        # Use a real path instead of mocking os.path.join which causes recursion
        ledger = CodeLedger("test-session")
        ledger.output_dir = str(tmp_path)
        ledger.ledger_path = os.path.join(str(tmp_path), "code_ledger.json")
        
        entry_id = ledger.add_entry({
            "agent": "test", "purpose": "test", "round": 1, "attempt": 1,
            "code": "print(1)", "code_hash": "abc", "success": True
        })
        
        assert entry_id.startswith("ledger_")
        assert len(ledger.entries) == 1
        assert ledger.entries[0].is_winning_component is True
        assert os.path.exists(ledger.ledger_path)

    @patch("tools.solution_assembler.llm_call")
    def test_assembler_compiles_standalone_notebook(self, mock_llm, mock_ledger_entries, tmp_path):
        """Verify notebook contains provided code blocks and template structure."""
        mock_llm.return_value = "Mock Writeup"
        
        with patch("tools.solution_assembler.Path") as mock_path:
            mock_path.return_value = tmp_path
            res = assemble_standalone_solution(
                "test-session", mock_ledger_entries, "train.parquet", "test.parquet", "target"
            )
            
            with open(res["script"], "r") as f:
                content = f.read()
                assert "Standalone ML Solution" in content
                assert "df = df.drop_nulls()" in content
                assert "df = df.with_columns(pl.lit(1).alias('f1'))" in content
                assert 'TRAIN_PATH = "train.parquet"' in content

    @patch("tools.solution_assembler.llm_call")
    def test_notebook_has_zero_professor_imports(self, mock_llm, mock_ledger_entries, tmp_path):
        """Confirm generated code has no professor dependencies."""
        mock_llm.return_value = "Mock"
        with patch("tools.solution_assembler.Path") as mock_path:
            mock_path.return_value = tmp_path
            res = assemble_standalone_solution(
                "test", mock_ledger_entries, "tr", "te", "t"
            )
            with open(res["script"], "r") as f:
                content = f.read()
                assert "import professor" not in content
                assert "from professor" not in content

    @patch("tools.solution_assembler.run_in_sandbox")
    def test_reproduction_validation_passes(self, mock_sb, tmp_path):
        """Verify verify_reproduction calls sandbox with standalone intent."""
        mock_sb.return_value = {"success": True}
        res = verify_reproduction("print(1)", "test-session")
        assert res["success"] is True
        assert mock_sb.called
        assert mock_sb.call_args[1]["purpose"] == "Reproduction check"

    @patch("tools.solution_assembler.llm_call")
    def test_writeup_generation_called(self, mock_llm, mock_ledger_entries, tmp_path):
        """Confirm LLM is triggered for narrative summary."""
        mock_llm.return_value = "# Winning Approach"
        with patch("tools.solution_assembler.Path") as mock_path:
            mock_path.return_value = tmp_path
            assemble_standalone_solution("test", mock_ledger_entries, "tr", "te", "t")
            assert mock_llm.called
            assert "PIPELINE COMPONENTS" in mock_llm.call_args[0][0]
