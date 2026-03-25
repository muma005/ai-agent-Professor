"""
Tests for configuration management.

FLAW-2.7: Configuration Management
"""
import pytest
import os
import json
from pathlib import Path
from pydantic import ValidationError
from core.config import (
    ProfessorConfig,
    PerformanceConfig,
    BudgetConfig,
    ModelConfig,
    APIConfig,
    get_config,
    initialize_config,
    validate_config,
)


class TestPerformanceConfig:
    """Test PerformanceConfig."""

    def test_default_values(self):
        """Test default performance values."""
        perf = PerformanceConfig()
        
        assert perf.max_memory_gb == 6.0
        assert perf.timeout_seconds == 600
        assert perf.max_parallel_jobs == 4

    def test_custom_values(self):
        """Test custom performance values."""
        perf = PerformanceConfig(
            max_memory_gb=8.0,
            timeout_seconds=900,
        )
        
        assert perf.max_memory_gb == 8.0
        assert perf.timeout_seconds == 900

    def test_invalid_memory(self):
        """Test invalid memory value rejected."""
        with pytest.raises(ValidationError):
            PerformanceConfig(max_memory_gb=0)
        
        with pytest.raises(ValidationError):
            PerformanceConfig(max_memory_gb=100)  # > 32

    def test_invalid_timeout(self):
        """Test invalid timeout rejected."""
        with pytest.raises(ValidationError):
            PerformanceConfig(timeout_seconds=0)


class TestBudgetConfig:
    """Test BudgetConfig."""

    def test_default_values(self):
        """Test default budget values."""
        budget = BudgetConfig()
        
        assert budget.budget_usd == 10.0
        assert budget.warning_threshold == 0.7
        assert budget.throttle_threshold == 0.85
        assert budget.triage_threshold == 0.95

    def test_threshold_validation(self):
        """Test threshold ordering validation."""
        # Throttle must be > warning
        with pytest.raises(ValidationError):
            BudgetConfig(warning_threshold=0.9, throttle_threshold=0.8)
        
        # Triage must be > throttle
        with pytest.raises(ValidationError):
            BudgetConfig(throttle_threshold=0.9, triage_threshold=0.8)

    def test_custom_budget(self):
        """Test custom budget values."""
        budget = BudgetConfig(budget_usd=50.0)
        
        assert budget.budget_usd == 50.0


class TestModelConfig:
    """Test ModelConfig."""

    def test_default_values(self):
        """Test default model values."""
        model = ModelConfig()
        
        assert model.default_cv_folds == 5
        assert model.optuna_trials == 100
        assert model.random_seed == 42

    def test_custom_seed(self):
        """Test custom random seed."""
        model = ModelConfig(random_seed=123)
        
        assert model.random_seed == 123

    def test_invalid_cv_folds(self):
        """Test invalid CV folds rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(default_cv_folds=1)  # < 2
        
        with pytest.raises(ValidationError):
            ModelConfig(default_cv_folds=20)  # > 10


class TestProfessorConfig:
    """Test ProfessorConfig."""

    def test_default_creation(self):
        """Test default configuration creation."""
        config = ProfessorConfig()
        
        assert config.competition_name == "unknown"
        assert config.performance.max_memory_gb == 6.0
        assert config.model.optuna_trials == 100
        assert config.cache_enabled is True

    def test_custom_competition_name(self):
        """Test custom competition name."""
        config = ProfessorConfig(competition_name="titanic")
        
        assert config.competition_name == "titanic"

    def test_nested_config_access(self):
        """Test nested configuration access."""
        config = ProfessorConfig()
        
        assert config.performance.timeout_seconds == 600
        assert config.budget.warning_threshold == 0.7
        assert config.model.random_seed == 42
        assert config.api.debug_logging is False

    def test_get_summary(self):
        """Test configuration summary."""
        config = ProfessorConfig(competition_name="test_comp")
        summary = config.get_summary()
        
        assert summary["competition_name"] == "test_comp"
        assert "max_memory_gb" in summary
        assert "feature_flags" in summary

    def test_save_and_load(self, tmp_path):
        """Test configuration save and load."""
        config = ProfessorConfig(
            competition_name="test_comp",
            performance=PerformanceConfig(max_memory_gb=8.0),
        )
        
        # Save
        config_path = tmp_path / "config.json"
        config.save(str(config_path))
        
        assert config_path.exists()
        
        # Load
        loaded = ProfessorConfig.load(str(config_path))
        
        assert loaded.competition_name == "test_comp"
        assert loaded.performance.max_memory_gb == 8.0


class TestEnvironmentLoading:
    """Test environment variable loading."""

    def test_from_env_competition_name(self, monkeypatch):
        """Test loading competition name from env."""
        monkeypatch.setenv("PROFESSOR_COMPETITION_NAME", "env_comp")
        
        config = ProfessorConfig.from_env()
        
        assert config.competition_name == "env_comp"

    def test_from_env_performance(self, monkeypatch):
        """Test loading performance config from env."""
        monkeypatch.setenv("PROFESSOR_PERFORMANCE_MAX_MEMORY_GB", "12.0")
        monkeypatch.setenv("PROFESSOR_PERFORMANCE_TIMEOUT_SECONDS", "1200")
        
        config = ProfessorConfig.from_env()
        
        assert config.performance.max_memory_gb == 12.0
        assert config.performance.timeout_seconds == 1200

    def test_from_env_model(self, monkeypatch):
        """Test loading model config from env."""
        monkeypatch.setenv("PROFESSOR_MODEL_OPTUNA_TRIALS", "500")
        monkeypatch.setenv("PROFESSOR_MODEL_RANDOM_SEED", "999")
        
        config = ProfessorConfig.from_env()
        
        assert config.model.optuna_trials == 500
        assert config.model.random_seed == 999

    def test_from_env_feature_flags(self, monkeypatch):
        """Test loading feature flags from env."""
        monkeypatch.setenv("PROFESSOR_ENABLE_PSEUDO_LABELING", "true")
        monkeypatch.setenv("PROFESSOR_ENABLE_ENSEMBLE", "false")
        
        config = ProfessorConfig.from_env()
        
        assert config.enable_pseudo_labeling is True
        assert config.enable_ensemble is False

    def test_from_env_cache(self, monkeypatch):
        """Test loading cache config from env."""
        monkeypatch.setenv("PROFESSOR_CACHE_ENABLED", "false")
        monkeypatch.setenv("PROFESSOR_CACHE_TTL_HOURS", "48")
        
        config = ProfessorConfig.from_env()
        
        assert config.cache_enabled is False
        assert config.cache_ttl_hours == 48


class TestGlobalFunctions:
    """Test global configuration functions."""

    def test_get_config_singleton(self):
        """Test get_config returns same instance."""
        # Clear global state
        from core import config
        config._config = None
        
        cfg1 = get_config()
        cfg2 = get_config()
        
        assert cfg1 is cfg2

    def test_initialize_config(self):
        """Test initialize_config function."""
        # Clear global state
        from core import config
        config._config = None
        
        cfg = initialize_config(
            competition_name="init_test",
            session_id="test_123",
        )
        
        assert cfg.competition_name == "init_test"
        assert cfg.session_id == "test_123"

    def test_validate_config(self):
        """Test validate_config function."""
        # Clear global state
        from core import config
        config._config = None
        
        result = validate_config()
        
        assert result is True


class TestConfigurationLogging:
    """Test configuration logging."""

    def test_log_summary(self, caplog):
        """Test configuration summary logging."""
        import logging
        
        config = ProfessorConfig(competition_name="log_test")
        
        with caplog.at_level(logging.INFO):
            config.log_summary()
        
        assert "PROFESSOR CONFIGURATION" in caplog.text
        assert "Competition: log_test" in caplog.text
        assert "Max Memory:" in caplog.text
