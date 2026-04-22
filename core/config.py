# core/config.py

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    skip_competition_intel: bool = False
    skip_eda: bool = False
    skip_red_team_critic: bool = False
    skip_pseudo_label: bool = False

class MLOptimizerConfig(BaseModel):
    models_to_try: List[str] = Field(default_factory=lambda: ["lgbm", "xgb", "catboost"])
    optuna_trials: int = 20
    timeout_seconds: int = 3600

class ProfessorConfig(BaseModel):
    fast_mode: bool = False
    agents: AgentConfig = Field(default_factory=AgentConfig)
    ml_optimizer: MLOptimizerConfig = Field(default_factory=MLOptimizerConfig)
    
    def apply_to_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Injects config into a state dictionary."""
        state_dict["config"] = self
        return state_dict
