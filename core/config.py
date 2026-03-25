# core/config.py

"""
Centralized configuration management for Professor pipeline.

FLAW-2.7 FIX: Configuration Management
- Centralized configuration with validation
- Environment variable integration
- Type-safe configuration objects
- Configuration documentation
- Default values with overrides
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, ValidationError

logger = logging.getLogger(__name__)


class PerformanceConfig(BaseModel):
    """Performance-related configuration."""
    
    max_memory_gb: float = Field(
        default=6.0,
        gt=0,
        le=32,
        description="Maximum memory usage in GB",
    )
    timeout_seconds: int = Field(
        default=600,
        gt=0,
        le=3600,
        description="Pipeline timeout in seconds",
    )
    max_parallel_jobs: int = Field(
        default=4,
        gt=0,
        le=16,
        description="Maximum parallel jobs",
    )
    gc_threshold_gb: float = Field(
        default=4.0,
        gt=0,
        le=32,
        description="GC trigger threshold in GB",
    )
    
    class Config:
        schema_extra = {
            "example": {
                "max_memory_gb": 6.0,
                "timeout_seconds": 600,
                "max_parallel_jobs": 4,
            }
        }


class BudgetConfig(BaseModel):
    """Budget-related configuration."""
    
    budget_usd: float = Field(
        default=10.0,
        gt=0,
        description="Total budget in USD",
    )
    warning_threshold: float = Field(
        default=0.7,
        gt=0,
        le=1,
        description="Warning threshold (0.0-1.0)",
    )
    throttle_threshold: float = Field(
        default=0.85,
        gt=0,
        le=1,
        description="Throttle threshold (0.0-1.0)",
    )
    triage_threshold: float = Field(
        default=0.95,
        gt=0,
        le=1,
        description="Triage threshold (0.0-1.0)",
    )
    
    @validator("throttle_threshold")
    def validate_throttle(cls, v, values):
        if "warning_threshold" in values and v <= values["warning_threshold"]:
            raise ValueError("throttle_threshold must be > warning_threshold")
        return v
    
    @validator("triage_threshold")
    def validate_triage(cls, v, values):
        if "throttle_threshold" in values and v <= values["throttle_threshold"]:
            raise ValueError("triage_threshold must be > throttle_threshold")
        return v


class ModelConfig(BaseModel):
    """Model training configuration."""
    
    default_cv_folds: int = Field(
        default=5,
        gt=2,
        le=10,
        description="Default CV folds",
    )
    optuna_trials: int = Field(
        default=100,
        gt=10,
        le=1000,
        description="Optuna trial count",
    )
    optuna_timeout_minutes: int = Field(
        default=30,
        gt=1,
        le=120,
        description="Optuna timeout in minutes",
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )
    
    class Config:
        schema_extra = {
            "example": {
                "default_cv_folds": 5,
                "optuna_trials": 100,
                "random_seed": 42,
            }
        }


class APIConfig(BaseModel):
    """API-related configuration."""
    
    llm_provider: str = Field(
        default="deepseek",
        description="Default LLM provider",
    )
    api_timeout_multiplier: float = Field(
        default=2.0,
        gt=0,
        description="API timeout multiplier",
    )
    api_backoff_enabled: bool = Field(
        default=True,
        description="Enable API backoff",
    )
    api_max_retries: int = Field(
        default=5,
        gt=0,
        le=10,
        description="Maximum API retries",
    )
    debug_logging: bool = Field(
        default=False,
        description="Enable debug logging",
    )


class ProfessorConfig(BaseModel):
    """
    Main Professor pipeline configuration.
    
    All configuration options are validated and documented.
    Load from environment variables or use defaults.
    """
    
    # Identity
    session_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"),
        description="Unique session identifier",
    )
    competition_name: str = Field(
        default="unknown",
        min_length=1,
        description="Competition name",
    )
    
    # Sub-configurations
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance settings",
    )
    budget: BudgetConfig = Field(
        default_factory=BudgetConfig,
        description="Budget settings",
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model training settings",
    )
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API settings",
    )
    
    # Cache settings
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching",
    )
    cache_ttl_hours: int = Field(
        default=24,
        gt=0,
        le=168,
        description="Cache TTL in hours",
    )
    
    # Feature flags
    enable_pseudo_labeling: bool = Field(
        default=False,
        description="Enable pseudo-labeling",
    )
    enable_ensemble: bool = Field(
        default=True,
        description="Enable ensemble",
    )
    enable_external_data: bool = Field(
        default=False,
        description="Enable external data",
    )
    
    class Config:
        schema_extra = {
            "title": "Professor Configuration",
            "description": "Complete configuration for Professor pipeline",
        }
    
    @classmethod
    def from_env(cls) -> "ProfessorConfig":
        """
        Load configuration from environment variables.
        
        Environment variables override defaults.
        Format: PROFESSOR_<SECTION>_<KEY>
        
        Examples:
            PROFESSOR_COMPETITION_NAME=titanic
            PROFESSOR_PERFORMANCE_MAX_MEMORY_GB=8.0
            PROFESSOR_MODEL_OPTUNA_TRIALS=200
        
        Returns:
            ProfessorConfig instance
        """
        config_data = {}
        
        # Top-level fields
        if comp_name := os.environ.get("PROFESSOR_COMPETITION_NAME"):
            config_data["competition_name"] = comp_name
        
        # Performance config
        perf_data = {}
        if val := os.environ.get("PROFESSOR_PERFORMANCE_MAX_MEMORY_GB"):
            perf_data["max_memory_gb"] = float(val)
        if val := os.environ.get("PROFESSOR_PERFORMANCE_TIMEOUT_SECONDS"):
            perf_data["timeout_seconds"] = int(val)
        if val := os.environ.get("PROFESSOR_PERFORMANCE_MAX_PARALLEL_JOBS"):
            perf_data["max_parallel_jobs"] = int(val)
        if perf_data:
            config_data["performance"] = PerformanceConfig(**perf_data)
        
        # Budget config
        budget_data = {}
        if val := os.environ.get("PROFESSOR_BUDGET_BUDGET_USD"):
            budget_data["budget_usd"] = float(val)
        if val := os.environ.get("PROFESSOR_BUDGET_WARNING_THRESHOLD"):
            budget_data["warning_threshold"] = float(val)
        if budget_data:
            config_data["budget"] = BudgetConfig(**budget_data)
        
        # Model config
        model_data = {}
        if val := os.environ.get("PROFESSOR_MODEL_DEFAULT_CV_FOLDS"):
            model_data["default_cv_folds"] = int(val)
        if val := os.environ.get("PROFESSOR_MODEL_OPTUNA_TRIALS"):
            model_data["optuna_trials"] = int(val)
        if val := os.environ.get("PROFESSOR_MODEL_RANDOM_SEED"):
            model_data["random_seed"] = int(val)
        if model_data:
            config_data["model"] = ModelConfig(**model_data)
        
        # API config
        api_data = {}
        if val := os.environ.get("PROFESSOR_API_LLM_PROVIDER"):
            api_data["llm_provider"] = val
        if val := os.environ.get("PROFESSOR_API_DEBUG_LOGGING"):
            api_data["debug_logging"] = val.lower() == "true"
        if api_data:
            config_data["api"] = APIConfig(**api_data)
        
        # Cache settings
        if val := os.environ.get("PROFESSOR_CACHE_ENABLED"):
            config_data["cache_enabled"] = val.lower() == "true"
        if val := os.environ.get("PROFESSOR_CACHE_TTL_HOURS"):
            config_data["cache_ttl_hours"] = int(val)
        
        # Feature flags
        if val := os.environ.get("PROFESSOR_ENABLE_PSEUDO_LABELING"):
            config_data["enable_pseudo_labeling"] = val.lower() == "true"
        if val := os.environ.get("PROFESSOR_ENABLE_ENSEMBLE"):
            config_data["enable_ensemble"] = val.lower() == "true"
        
        try:
            return cls(**config_data)
        except ValidationError as e:
            logger.error(f"[Config] Validation error: {e}")
            raise
    
    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save configuration
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, "w") as f:
            json.dump(self.dict(), f, indent=2, default=str)
        
        logger.info(f"[Config] Configuration saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> "ProfessorConfig":
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to configuration file
        
        Returns:
            ProfessorConfig instance
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            logger.warning(f"[Config] Configuration file not found: {path}")
            return cls()
        
        with open(path_obj, "r") as f:
            config_data = json.load(f)
        
        try:
            return cls(**config_data)
        except ValidationError as e:
            logger.error(f"[Config] Validation error loading {path}: {e}")
            raise
    
    def get_summary(self) -> dict:
        """Get configuration summary."""
        return {
            "session_id": self.session_id,
            "competition_name": self.competition_name,
            "max_memory_gb": self.performance.max_memory_gb,
            "timeout_seconds": self.performance.timeout_seconds,
            "optuna_trials": self.model.optuna_trials,
            "random_seed": self.model.random_seed,
            "cache_enabled": self.cache_enabled,
            "feature_flags": {
                "pseudo_labeling": self.enable_pseudo_labeling,
                "ensemble": self.enable_ensemble,
                "external_data": self.enable_external_data,
            },
        }
    
    def log_summary(self) -> None:
        """Log configuration summary."""
        summary = self.get_summary()
        
        logger.info("=" * 70)
        logger.info("PROFESSOR CONFIGURATION")
        logger.info("=" * 70)
        logger.info(f"Session: {summary['session_id']}")
        logger.info(f"Competition: {summary['competition_name']}")
        logger.info(f"Max Memory: {summary['max_memory_gb']} GB")
        logger.info(f"Timeout: {summary['timeout_seconds']}s")
        logger.info(f"Optuna Trials: {summary['optuna_trials']}")
        logger.info(f"Random Seed: {summary['random_seed']}")
        logger.info(f"Cache Enabled: {summary['cache_enabled']}")
        logger.info(f"Pseudo-labeling: {summary['feature_flags']['pseudo_labeling']}")
        logger.info(f"Ensemble: {summary['feature_flags']['ensemble']}")
        logger.info("=" * 70)


# Global configuration instance
_config: Optional[ProfessorConfig] = None


def get_config() -> ProfessorConfig:
    """
    Get or create global configuration instance.
    
    Returns:
        ProfessorConfig instance
    """
    global _config
    
    if _config is None:
        _config = ProfessorConfig.from_env()
        _config.log_summary()
    
    return _config


def initialize_config(
    competition_name: str,
    session_id: Optional[str] = None,
    **kwargs,
) -> ProfessorConfig:
    """
    Initialize configuration with custom values.
    
    Args:
        competition_name: Competition name
        session_id: Optional session ID
        **kwargs: Additional configuration overrides
    
    Returns:
        ProfessorConfig instance
    """
    global _config
    
    config_data = {
        "competition_name": competition_name,
    }
    
    if session_id:
        config_data["session_id"] = session_id
    
    # Apply kwargs
    config_data.update(kwargs)
    
    _config = ProfessorConfig(**config_data)
    _config.log_summary()
    
    return _config


def validate_config() -> bool:
    """
    Validate current configuration.
    
    Returns:
        True if valid
    """
    try:
        config = get_config()
        logger.info("[Config] Configuration validated successfully")
        return True
    except ValidationError as e:
        logger.error(f"[Config] Configuration validation failed: {e}")
        return False
