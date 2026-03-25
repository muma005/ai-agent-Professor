"""
Tests for submission validation.
"""
import pytest
import tempfile
import os
import numpy as np
import polars as pl

from tools.submission_validator import (
    validate_submission_format,
    validate_submission_predictions,
    validate_submission_from_state,
    ProfessorSubmissionError
)


class TestSubmissionFormatValidation:
    """Test submission format validation."""
    
    @pytest.fixture
    def sample_submission(self, tmp_path):
        """Create sample submission."""
        sample_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0] * 20
        })
        path = tmp_path / "sample_submission.csv"
        sample_df.write_csv(path)
        return str(path)
    
    def test_valid_submission(self, sample_submission, tmp_path):
        """Test valid submission passes validation."""
        submission_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0.1, 0.3, 0.5, 0.7, 0.9] * 4  # Varying predictions
        })
        path = tmp_path / "submission.csv"
        submission_df.write_csv(path)

        result = validate_submission_format(str(path), sample_submission)
        assert result is True

    def test_column_mismatch(self, sample_submission, tmp_path):
        """Test column mismatch detection."""
        submission_df = pl.DataFrame({
            "id": list(range(20)),
            "wrong_col": [0.1, 0.3, 0.5, 0.7, 0.9] * 4
        })
        path = tmp_path / "submission.csv"
        submission_df.write_csv(path)

        with pytest.raises(ProfessorSubmissionError, match="Column mismatch"):
            validate_submission_format(str(path), sample_submission)

    def test_row_count_mismatch(self, sample_submission, tmp_path):
        """Test row count mismatch detection."""
        submission_df = pl.DataFrame({
            "id": list(range(19)),
            "target": [0.1, 0.3, 0.5, 0.7, 0.9] * 3 + [0.1, 0.2, 0.3, 0.4]  # 19 values
        })
        path = tmp_path / "submission.csv"
        submission_df.write_csv(path)

        with pytest.raises(ProfessorSubmissionError, match="Row count mismatch"):
            validate_submission_format(str(path), sample_submission)

    def test_id_mismatch(self, sample_submission, tmp_path):
        """Test ID mismatch detection."""
        submission_df = pl.DataFrame({
            "id": list(range(1, 21)),  # Different IDs
            "target": [0.1, 0.3, 0.5, 0.7, 0.9] * 4
        })
        path = tmp_path / "submission.csv"
        submission_df.write_csv(path)

        with pytest.raises(ProfessorSubmissionError, match="ID column mismatch"):
            validate_submission_format(str(path), sample_submission)
    
    def test_null_values(self, sample_submission, tmp_path):
        """Test null value detection."""
        submission_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0.5] * 19 + [None]
        })
        path = tmp_path / "submission.csv"
        submission_df.write_csv(path)
        
        with pytest.raises(ProfessorSubmissionError, match="null values"):
            validate_submission_format(str(path), sample_submission)
    
    def test_nan_values(self, sample_submission, tmp_path):
        """Test NaN value detection."""
        submission_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0.5] * 19 + [float('nan')]
        })
        path = tmp_path / "submission.csv"
        submission_df.write_csv(path)
        
        with pytest.raises(ProfessorSubmissionError, match="NaN"):
            validate_submission_format(str(path), sample_submission)
    
    def test_constant_predictions(self, sample_submission, tmp_path):
        """Test constant prediction detection."""
        submission_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0.5] * 20  # All same value
        })
        path = tmp_path / "submission.csv"
        submission_df.write_csv(path)
        
        with pytest.raises(ProfessorSubmissionError, match="no variance"):
            validate_submission_format(str(path), sample_submission)


class TestPredictionValidation:
    """Test prediction validation."""
    
    def test_valid_predictions_binary(self):
        """Test valid binary predictions."""
        preds = np.array([0.1, 0.5, 0.9])
        
        result = validate_submission_predictions(preds, task_type="binary")
        assert result is True
    
    def test_valid_predictions_multiclass(self):
        """Test valid multiclass predictions."""
        preds = np.array([0.2, 0.5, 0.3])
        
        result = validate_submission_predictions(preds, task_type="multiclass")
        assert result is True
    
    def test_nan_detection(self):
        """Test NaN detection."""
        preds = np.array([0.1, np.nan, 0.9])
        
        with pytest.raises(ProfessorSubmissionError, match="NaN"):
            validate_submission_predictions(preds)
    
    def test_inf_detection(self):
        """Test Inf detection."""
        preds = np.array([0.1, np.inf, 0.9])
        
        with pytest.raises(ProfessorSubmissionError, match="Inf"):
            validate_submission_predictions(preds)
    
    def test_range_check_binary(self):
        """Test range check for binary classification."""
        # Out of range
        preds = np.array([0.1, 1.5, 0.9])
        
        with pytest.raises(ProfessorSubmissionError, match="out of range"):
            validate_submission_predictions(preds, task_type="binary")
        
        # Negative
        preds = np.array([0.1, -0.1, 0.9])
        
        with pytest.raises(ProfessorSubmissionError, match="out of range"):
            validate_submission_predictions(preds, task_type="binary")
    
    def test_distribution_warning(self, caplog):
        """Test distribution warning for suspicious predictions."""
        # All zeros
        preds = np.array([0.0] * 20)
        
        validate_submission_predictions(preds, task_type="binary", check_distribution=True)
        
        assert "suspicious" in caplog.text.lower()
    
    def test_regression_predictions(self):
        """Test regression predictions (no range check)."""
        preds = np.array([1.0, 5.0, 10.0])
        
        result = validate_submission_predictions(preds, task_type="regression")
        assert result is True


class TestSubmissionFromState:
    """Test submission validation from state."""
    
    def test_valid_submission_from_state(self, tmp_path):
        """Test valid submission from state."""
        # Create sample submission
        sample_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0] * 20
        })
        sample_path = tmp_path / "sample_submission.csv"
        sample_df.write_csv(sample_path)

        # Create submission with varying predictions
        submission_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0.1, 0.3, 0.5, 0.7, 0.9] * 4
        })
        submission_path = tmp_path / "submission.csv"
        submission_df.write_csv(submission_path)

        state = {
            "submission_path": str(submission_path),
            "sample_submission_path": str(sample_path),
            "task_type": "binary"
        }

        result = validate_submission_from_state(state)
        assert result is True
    
    def test_missing_submission_path(self):
        """Test missing submission path."""
        state = {}
        
        with pytest.raises(ProfessorSubmissionError, match="No submission path"):
            validate_submission_from_state(state)
    
    def test_missing_sample_path(self, tmp_path):
        """Test missing sample submission path."""
        # Create submission
        submission_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0.5] * 20
        })
        submission_path = tmp_path / "submission.csv"
        submission_df.write_csv(submission_path)
        
        state = {
            "submission_path": str(submission_path)
        }
        
        with pytest.raises(ProfessorSubmissionError, match="No sample submission"):
            validate_submission_from_state(state)
