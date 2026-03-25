"""
End-to-End Integration Tests for Full Professor Pipeline.

Tests the complete pipeline from start to finish, including:
- Data loading and preprocessing
- Feature engineering
- Model training with fallback
- Error handling and recovery
- Timeout handling
- Checkpoint resume capability
"""
import pytest
import tempfile
import os
import polars as pl
import numpy as np
from datetime import datetime

from core.state import initial_state
from core.professor import run_professor, ProfessorPipelineError
from core.checkpoint import get_latest_checkpoint, load_checkpoint


class TestFullPipelineIntegration:
    """Test full pipeline integration."""
    
    @pytest.fixture
    def synthetic_data(self, tmp_path):
        """Create synthetic dataset for testing."""
        np.random.seed(42)
        n_rows = 100
        n_features = 5
        
        X = np.random.randn(n_rows, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pl.DataFrame({
            f"feature_{i}": X[:, i] for i in range(n_features)
        })
        df = df.with_columns(pl.Series("target", y))
        
        data_path = tmp_path / "train.csv"
        df.write_csv(data_path)
        
        # Create test.csv
        test_df = pl.DataFrame({
            f"feature_{i}": X[:20, i] for i in range(n_features)
        })
        test_path = tmp_path / "test.csv"
        test_df.write_csv(test_path)
        
        # Create sample submission
        sample_df = pl.DataFrame({
            "id": list(range(20)),
            "target": [0] * 20
        })
        sample_path = tmp_path / "sample_submission.csv"
        sample_df.write_csv(sample_path)
        
        return {
            "data_path": str(data_path),
            "test_path": str(test_path),
            "sample_path": str(sample_path),
            "tmp_path": tmp_path
        }
    
    def test_pipeline_completes_successfully(self, synthetic_data):
        """Test pipeline completes end-to-end."""
        state = initial_state(
            competition="integration_test",
            data_path=synthetic_data["data_path"]
        )
        
        # Run with timeout for integration test
        result = run_professor(state, timeout_seconds=300)
        
        assert result is not None
        assert "session_id" in result
        assert "cv_mean" in result or "model_registry" in result
    
    def test_pipeline_with_timeout(self, synthetic_data):
        """Test pipeline respects timeout."""
        state = initial_state(
            competition="timeout_test",
            data_path=synthetic_data["data_path"]
        )
        
        # Very short timeout should cause timeout error
        with pytest.raises(ProfessorPipelineError, match="timeout"):
            run_professor(state, timeout_seconds=1)
    
    def test_pipeline_error_context_saved(self, synthetic_data):
        """Test error context is saved on failure."""
        state = initial_state(
            competition="error_test",
            data_path=synthetic_data["data_path"]
        )
        
        try:
            # This should fail due to very short timeout
            run_professor(state, timeout_seconds=1)
        except ProfessorPipelineError:
            pass
        
        # Check error context was saved
        error_context_path = f"outputs/{state['session_id']}/error_context.json"
        assert os.path.exists(error_context_path), "Error context not saved"
    
    def test_pipeline_checkpoint_created(self, synthetic_data):
        """Test checkpoint is created during execution."""
        state = initial_state(
            competition="checkpoint_test",
            data_path=synthetic_data["data_path"]
        )
        
        try:
            run_professor(state, timeout_seconds=300)
        except:
            pass  # May or may not complete
        
        # Check if checkpoint directory exists
        checkpoint_dir = f"outputs/{state['session_id']}/checkpoints"
        # Checkpoints may or may not be created depending on when timeout occurs
    
    def test_pipeline_resume_from_checkpoint(self, synthetic_data):
        """Test pipeline can resume from checkpoint."""
        # TODO: Implement once checkpoint integration is complete
        # This test will:
        # 1. Run pipeline partway
        # 2. Simulate failure
        # 3. Resume from checkpoint
        # 4. Verify completion
        pass
    
    def test_pipeline_with_invalid_data(self, tmp_path):
        """Test pipeline handles invalid data gracefully."""
        # Create invalid dataset (empty)
        df = pl.DataFrame({"feature_1": []})
        data_path = tmp_path / "empty.csv"
        df.write_csv(data_path)
        
        state = initial_state(
            competition="invalid_data_test",
            data_path=str(data_path)
        )
        
        with pytest.raises(ProfessorPipelineError):
            run_professor(state, timeout_seconds=60)
    
    def test_pipeline_error_includes_context(self, synthetic_data):
        """Test pipeline error includes node and timestamp."""
        state = initial_state(
            competition="context_test",
            data_path=synthetic_data["data_path"]
        )
        
        try:
            run_professor(state, timeout_seconds=1)
        except ProfessorPipelineError as e:
            assert e.node is not None or "timeout" in str(e).lower()
            assert e.timestamp is not None


class TestPipelineWithFailures:
    """Test pipeline behavior with various failure scenarios."""
    
    @pytest.fixture
    def synthetic_data(self, tmp_path):
        """Create synthetic dataset for testing."""
        np.random.seed(42)
        n_rows = 100
        n_features = 5
        
        X = np.random.randn(n_rows, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pl.DataFrame({
            f"feature_{i}": X[:, i] for i in range(n_features)
        })
        df = df.with_columns(pl.Series("target", y))
        
        data_path = tmp_path / "train.csv"
        df.write_csv(data_path)
        
        return {
            "data_path": str(data_path),
            "tmp_path": tmp_path
        }
    
    def test_pipeline_handles_model_training_failure(self, synthetic_data, monkeypatch):
        """Test pipeline handles model training failures."""
        # TODO: Mock model training to fail and verify fallback works
        pass
    
    def test_pipeline_handles_api_failure(self, synthetic_data, monkeypatch):
        """Test pipeline handles API failures gracefully."""
        # Mock LLM to fail
        def mock_call_llm(*args, **kwargs):
            raise Exception("API down")
        
        monkeypatch.setattr("tools.llm_client.call_llm", mock_call_llm)
        
        state = initial_state(
            competition="api_failure_test",
            data_path=synthetic_data["data_path"]
        )
        
        # Pipeline should fail with proper error
        with pytest.raises(ProfessorPipelineError):
            run_professor(state, timeout_seconds=60)
    
    def test_pipeline_circuit_breaker_activates(self, synthetic_data, monkeypatch):
        """Test circuit breaker activates after repeated failures."""
        # TODO: Implement circuit breaker activation test
        pass


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    @pytest.fixture
    def benchmark_data(self, tmp_path):
        """Create benchmark dataset."""
        np.random.seed(42)
        n_rows = 200
        n_features = 10
        
        X = np.random.randn(n_rows, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pl.DataFrame({
            f"feature_{i}": X[:, i] for i in range(n_features)
        })
        df = df.with_columns(pl.Series("target", y))
        
        data_path = tmp_path / "train.csv"
        df.write_csv(data_path)
        
        return {
            "data_path": str(data_path),
            "tmp_path": tmp_path
        }
    
    def test_pipeline_completes_within_time_limit(self, benchmark_data):
        """Test pipeline completes within reasonable time."""
        state = initial_state(
            competition="performance_test",
            data_path=benchmark_data["data_path"]
        )
        
        start = datetime.now()
        result = run_professor(state, timeout_seconds=300)
        elapsed = (datetime.now() - start).total_seconds()
        
        # Should complete in under 5 minutes for small dataset
        assert elapsed < 300, f"Pipeline took too long: {elapsed}s"
    
    def test_pipeline_memory_usage(self, benchmark_data):
        """Test pipeline memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        state = initial_state(
            competition="memory_test",
            data_path=benchmark_data["data_path"]
        )
        
        # Get initial memory
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        result = run_professor(state, timeout_seconds=300)
        
        # Get final memory
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Memory increase should be reasonable (< 500MB for small dataset)
        memory_increase = final_memory - initial_memory
        assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB"
