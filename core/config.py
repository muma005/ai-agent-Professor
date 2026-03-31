# core/config.py
"""
Professor Configuration System — Centralized control for all execution parameters.

Usage:
    config = ProfessorConfig(fast_mode=True)
    state = config.apply_to_state(initial_state)
    result = run_professor(state)

Presets:
    fast_mode=True       → Local development, ~5 min/trial
    production_mode=True → Full execution, ~1 hour/trial
    custom               → Mix and match options
"""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class SandboxConfig:
    """
    Sandbox execution configuration.
    
    Attributes:
        enabled: If False, executes code directly (faster, less isolated)
        timeout_seconds: Maximum execution time for sandbox code
        skip_import_validation: If True, allows any import (DEV MODE)
    """
    enabled: bool = True
    timeout_seconds: int = 600
    skip_import_validation: bool = False
    
    def apply_env(self):
        """Apply config to environment variables"""
        os.environ["PROFESSOR_USE_SANDBOX"] = "1" if self.enabled else "0"
        os.environ["PROFESSOR_SANDBOX_TIMEOUT"] = str(self.timeout_seconds)
        os.environ["PROFESSOR_SKIP_SANDBOX"] = "1" if not self.enabled else "0"


@dataclass
class FeatureFactoryConfig:
    """
    Feature generation configuration.
    
    Attributes:
        enabled: If False, skip feature factory entirely
        skip_llm_rounds: Skip rounds 2, 5 (LLM-generated features)
        skip_wilcoxon_gate: Skip statistical testing of features
        skip_null_importance: Skip null importance filtering
        max_interaction_features: Max features from interactions
        max_aggregation_features: Max aggregation features
    """
    enabled: bool = True
    skip_llm_rounds: bool = False
    skip_wilcoxon_gate: bool = False
    skip_null_importance: bool = False
    max_interaction_features: int = 20
    max_aggregation_features: int = 50
    
    def apply_env(self):
        """Apply config to environment variables"""
        os.environ["PROFESSOR_SKIP_LLM_ROUNDS"] = "1" if self.skip_llm_rounds else "0"
        os.environ["PROFESSOR_SKIP_WILCOXON"] = "1" if self.skip_wilcoxon_gate else "0"
        os.environ["PROFESSOR_SKIP_NULL_IMPORTANCE"] = "1" if self.skip_null_importance else "0"


@dataclass
class MLOptimizerConfig:
    """
    Model optimization configuration.
    
    Attributes:
        optuna_trials: Number of Optuna hyperparameter trials
        models_to_try: List of model types to optimize
        cv_folds: Number of cross-validation folds
        timeout_per_trial: Timeout per Optuna trial in seconds
    """
    optuna_trials: int = 30  # Reduced from 100 for fast mode
    models_to_try: List[str] = field(default_factory=lambda: ["lgbm"])
    cv_folds: int = 5
    timeout_per_trial: int = 300
    
    def apply_env(self):
        """Apply config to environment variables"""
        os.environ["PROFESSOR_OPTUNA_TRIALS"] = str(self.optuna_trials)
        os.environ["PROFESSOR_MODELS"] = ",".join(self.models_to_try)
        os.environ["PROFESSOR_CV_FOLDS"] = str(self.cv_folds)


@dataclass
class AgentSkipConfig:
    """
    Configuration for skipping entire agents.
    
    Attributes:
        skip_competition_intel: Skip forum scraping and intel gathering
        skip_eda: Skip exploratory data analysis
        skip_red_team_critic: Skip critical review
        skip_ensemble: Skip ensemble building
        skip_pseudo_label: Skip pseudo-labeling
    """
    skip_competition_intel: bool = False
    skip_eda: bool = False
    skip_red_team_critic: bool = False
    skip_ensemble: bool = False
    skip_pseudo_label: bool = False
    
    def apply_env(self):
        """Apply config to environment variables"""
        os.environ["PROFESSOR_SKIP_INTEL"] = "1" if self.skip_competition_intel else "0"
        os.environ["PROFESSOR_SKIP_EDA"] = "1" if self.skip_eda else "0"
        os.environ["PROFESSOR_SKIP_CRITIC"] = "1" if self.skip_red_team_critic else "0"
        os.environ["PROFESSOR_SKIP_ENSEMBLE"] = "1" if self.skip_ensemble else "0"


@dataclass
class ProfessorConfig:
    """
    Master configuration for Professor pipeline.
    
    Presets:
        fast_mode=True       → Local development, ~5 min/trial
            - Disables sandbox
            - Skips CompetitionIntel, EDA, RedTeamCritic
            - Skips LLM feature rounds
            - 1 Optuna trial (defaults only)
            - Single model (LightGBM)
            - 3-fold CV
        
        production_mode=True → Full execution, ~1 hour/trial
            - Full sandbox isolation
            - All agents enabled
            - 100 Optuna trials
            - 3 models (LGBM, XGB, CatBoost)
            - 5-fold CV
    
    Example:
        config = ProfessorConfig(fast_mode=True)
        state = config.apply_to_state(initial_state)
        result = run_professor(state)
    """
    # Execution mode presets
    fast_mode: bool = False
    production_mode: bool = False
    
    # Component configs
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    feature_factory: FeatureFactoryConfig = field(default_factory=FeatureFactoryConfig)
    ml_optimizer: MLOptimizerConfig = field(default_factory=MLOptimizerConfig)
    agents: AgentSkipConfig = field(default_factory=AgentSkipConfig)
    
    # Global settings
    seed: int = 42
    log_level: str = "INFO"
    checkpoint_enabled: bool = True
    
    def __post_init__(self):
        """Apply presets after initialization"""
        if self.fast_mode:
            self._apply_fast_mode()
        elif self.production_mode:
            self._apply_production_mode()
    
    def _apply_fast_mode(self):
        """Configure for fast local execution"""
        # Disable sandbox overhead
        self.sandbox.enabled = False
        self.sandbox.skip_import_validation = True
        
        # Skip expensive feature generation
        self.feature_factory.skip_llm_rounds = True
        self.feature_factory.skip_wilcoxon_gate = True
        self.feature_factory.skip_null_importance = True
        
        # Minimal model optimization
        self.ml_optimizer.optuna_trials = 1  # Just defaults
        self.ml_optimizer.models_to_try = ["lgbm"]  # Single model
        self.ml_optimizer.cv_folds = 3  # Reduced CV
        
        # Skip non-essential agents
        self.agents.skip_competition_intel = True
        self.agents.skip_eda = True
        self.agents.skip_red_team_critic = True
        self.agents.skip_ensemble = True
    
    def _apply_production_mode(self):
        """Configure for full production execution"""
        # Full sandbox isolation
        self.sandbox.enabled = True
        self.sandbox.timeout_seconds = 600
        
        # All feature generation enabled
        self.feature_factory.skip_llm_rounds = False
        self.feature_factory.skip_wilcoxon_gate = False
        self.feature_factory.skip_null_importance = False
        
        # Full model optimization
        self.ml_optimizer.optuna_trials = 100
        self.ml_optimizer.models_to_try = ["lgbm", "xgb", "catboost"]
        self.ml_optimizer.cv_folds = 5
        
        # All agents enabled
        self.agents.skip_competition_intel = False
        self.agents.skip_eda = False
        self.agents.skip_red_team_critic = False
        self.agents.skip_ensemble = False
    
    def apply_env(self):
        """Apply all config to environment variables"""
        os.environ["PROFESSOR_SEED"] = str(self.seed)
        os.environ["PROFESSOR_LOG_LEVEL"] = self.log_level
        os.environ["PROFESSOR_CHECKPOINT"] = "1" if self.checkpoint_enabled else "0"
        os.environ["PROFESSOR_FAST_MODE"] = "1" if self.fast_mode else "0"
        os.environ["PROFESSOR_PRODUCTION_MODE"] = "1" if self.production_mode else "0"
        
        self.sandbox.apply_env()
        self.feature_factory.apply_env()
        self.ml_optimizer.apply_env()
        self.agents.apply_env()
    
    def apply_to_state(self, state: dict) -> dict:
        """
        Apply config to ProfessorState.
        
        Modifies DAG to skip agents based on config.
        """
        state["config"] = self
        
        # Modify DAG based on config
        if "dag" in state:
            dag = state["dag"].copy()
            
            if self.agents.skip_competition_intel:
                dag = [n for n in dag if n != "competition_intel"]
            
            if self.agents.skip_eda:
                dag = [n for n in dag if n != "eda_agent"]
            
            if self.agents.skip_red_team_critic:
                dag = [n for n in dag if n != "red_team_critic"]
            
            if self.agents.skip_ensemble:
                dag = [n for n in dag if n != "ensemble_architect"]
            
            state["dag"] = dag
        
        return state
    
    def save(self, path: str):
        """Save config to JSON for reproducibility"""
        import json
        
        config_dict = asdict(self)
        config_dict["timestamp"] = datetime.now().isoformat()
        config_dict["version"] = "1.0.0"
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_env(cls) -> "ProfessorConfig":
        """
        Load config from environment variables.
        
        Falls back to defaults if env vars not set.
        """
        config = cls()
        
        # Check for preset modes
        if os.getenv("PROFESSOR_FAST_MODE") == "1":
            config.fast_mode = True
            config._apply_fast_mode()
        
        if os.getenv("PROFESSOR_PRODUCTION_MODE") == "1":
            config.production_mode = True
            config._apply_production_mode()
        
        # Override individual settings from env
        if os.getenv("PROFESSOR_OPTUNA_TRIALS"):
            config.ml_optimizer.optuna_trials = int(os.getenv("PROFESSOR_OPTUNA_TRIALS"))
        
        if os.getenv("PROFESSOR_SKIP_LLM_ROUNDS") == "1":
            config.feature_factory.skip_llm_rounds = True
        
        if os.getenv("PROFESSOR_SKIP_WILCOXON") == "1":
            config.feature_factory.skip_wilcoxon_gate = True
        
        if os.getenv("PROFESSOR_SKIP_EDA") == "1":
            config.agents.skip_eda = True
        
        if os.getenv("PROFESSOR_SKIP_SANDBOX") == "1":
            config.sandbox.enabled = False
        
        # Load models from env
        models_env = os.getenv("PROFESSOR_MODELS")
        if models_env:
            config.ml_optimizer.models_to_try = models_env.split(",")
        
        return config
    
    def __str__(self) -> str:
        """Human-readable config summary"""
        lines = [
            "ProfessorConfig:",
            f"  Mode: fast={self.fast_mode}, production={self.production_mode}",
            f"  Sandbox: enabled={self.sandbox.enabled}",
            f"  FeatureFactory: skip_llm={self.feature_factory.skip_llm_rounds}, skip_wilcoxon={self.feature_factory.skip_wilcoxon_gate}",
            f"  MLOptimizer: trials={self.ml_optimizer.optuna_trials}, models={self.ml_optimizer.models_to_try}",
            f"  Skipped agents: intel={self.agents.skip_competition_intel}, eda={self.agents.skip_eda}, critic={self.agents.skip_red_team_critic}",
        ]
        return "\n".join(lines)
