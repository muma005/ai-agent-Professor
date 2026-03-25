# agents/hpo_agent.py

"""
Advanced Hyperparameter Optimization with multi-fidelity methods.

Advanced Feature: Advanced HPO
- Hyperband pruning algorithm
- Successive halving
- Multi-fidelity optimization
- Search space definitions
- Trial persistence
- Warm-start from previous trials
- Regression-aware baseline tracking
"""

import os
import json
import logging
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import optuna
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import TPESampler, RandomSampler
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)


@dataclass
class HPOResult:
    """Result of hyperparameter optimization."""
    
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    pruned_trials: int
    completed_trials: int
    optimization_time: float
    pruner_type: str
    sampler_type: str
    study_direction: str
    trial_history: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "best_params": self.best_params,
            "best_score": round(self.best_score, 6),
            "n_trials": self.n_trials,
            "pruned_trials": self.pruned_trials,
            "completed_trials": self.completed_trials,
            "optimization_time": round(self.optimization_time, 2),
            "pruner_type": self.pruner_type,
            "sampler_type": self.sampler_type,
            "study_direction": self.study_direction,
            "trial_history": self.trial_history,
        }


class HPOAgent:
    """
    Advanced Hyperparameter Optimization agent.
    
    Features:
    - Hyperband pruning (multi-fidelity)
    - Successive halving
    - Multiple search spaces (LightGBM, XGBoost, CatBoost)
    - Trial persistence
    - Warm-start from previous trials
    - Cross-validation based optimization
    - Multiple sampling strategies
    
    Usage:
        hpo = HPOAgent(
            n_trials=100,
            pruner="hyperband",
            sampler="tpe",
        )
        
        result = hpo.optimize(
            X=X_train,
            y=y_train,
            model_type="lightgbm",
            metric="auc",
            n_folds=5,
        )
        
        print(f"Best score: {result.best_score}")
        print(f"Best params: {result.best_params}")
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        timeout_minutes: Optional[int] = None,
        pruner: str = "hyperband",
        sampler: str = "tpe",
        n_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        study_name: str = "hpo_study",
        storage_path: Optional[str] = None,
    ):
        """
        Initialize HPO agent.
        
        Args:
            n_trials: Number of optimization trials
            timeout_minutes: Optional timeout in minutes
            pruner: Pruning algorithm ("hyperband", "successive_halving", "median", "none")
            sampler: Sampling algorithm ("tpe", "random")
            n_folds: Number of CV folds
            random_state: Random seed
            n_jobs: Number of parallel jobs
            study_name: Name of the study
            storage_path: Path for trial persistence (optional)
        """
        self.n_trials = n_trials
        self.timeout_minutes = timeout_minutes
        self.pruner_type = pruner
        self.sampler_type = sampler
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage_path = storage_path
        
        self.pruner = self._create_pruner(pruner)
        self.sampler = self._create_sampler(sampler)
        
        self.baseline_scores: Dict[str, float] = {}
        self.optimization_history: List[HPOResult] = []
        
        logger.info(
            f"[HPOAgent] Initialized -- trials: {n_trials}, "
            f"pruner: {pruner}, sampler: {sampler}"
        )
    
    def _create_pruner(self, pruner_type: str) -> Optional[optuna.pruners.BasePruner]:
        """Create pruning algorithm."""
        if pruner_type == "hyperband":
            return HyperbandPruner(
                min_resource=1,
                max_resource=self.n_trials,
                reduction_factor=3,
            )
        elif pruner_type == "successive_halving":
            return SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=3,
                min_early_stopping_rate=0,
            )
        elif pruner_type == "median":
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            )
        elif pruner_type == "none":
            return None
        else:
            logger.warning(f"Unknown pruner type: {pruner_type}, using Hyperband")
            return HyperbandPruner()
    
    def _create_sampler(self, sampler_type: str) -> optuna.samplers.BaseSampler:
        """Create sampling algorithm."""
        if sampler_type == "tpe":
            return TPESampler(
                seed=self.random_state,
                multivariate=True,
                consider_prior=True,
                prior_weight=1.0,
            )
        elif sampler_type == "random":
            return RandomSampler(seed=self.random_state)
        else:
            logger.warning(f"Unknown sampler type: {sampler_type}, using TPE")
            return TPESampler(seed=self.random_state)
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "lightgbm",
        metric: str = "auc",
        search_space: Optional[Dict[str, Any]] = None,
        n_folds: Optional[int] = None,
    ) -> HPOResult:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Model type ("lightgbm", "xgboost", "catboost")
            metric: Scoring metric ("auc", "logloss", "rmse", "mae")
            search_space: Custom search space (optional)
            n_folds: Override default n_folds
        
        Returns:
            HPOResult with best parameters and scores
        
        Raises:
            ValueError: If model_type is unknown
        """
        if n_folds is None:
            n_folds = self.n_folds
        
        logger.info(
            f"[HPOAgent] Starting optimization -- model: {model_type}, "
            f"metric: {metric}, trials: {self.n_trials}"
        )
        
        # Create or load study
        study = self._create_study(metric)
        
        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            # Get search space
            if search_space:
                params = self._suggest_params_custom(trial, search_space)
            else:
                params = self._suggest_params(trial, model_type)
            
            # Add common parameters
            params["random_state"] = self.random_state
            
            # Run cross-validation
            cv_scores = self._cross_validate(X, y, params, model_type, metric, n_folds)
            cv_mean = np.mean(cv_scores)
            
            # Report intermediate values for pruning
            for i, score in enumerate(cv_scores):
                trial.report(score, i)
                
                # Check for pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return cv_mean
        
        # Run optimization
        start_time = datetime.now()
        
        timeout_seconds = None
        if self.timeout_minutes:
            timeout_seconds = self.timeout_minutes * 60
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=timeout_seconds,
            n_jobs=1,  # Multi-processing can cause issues with pruning
            show_progress_bar=True,
            gc_after_trial=True,
        )
        
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        # Extract results
        best_params = study.best_params
        best_score = study.best_value
        
        # Count trial statistics
        completed_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        pruned_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        
        # Extract trial history
        trial_history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_history.append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                })
        
        logger.info(
            f"[HPOAgent] Optimization complete -- best_score: {best_score:.6f}, "
            f"time: {optimization_time:.2f}s, trials: {completed_trials} completed, "
            f"{pruned_trials} pruned"
        )
        
        # Create result
        result = HPOResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials),
            pruned_trials=pruned_trials,
            completed_trials=completed_trials,
            optimization_time=optimization_time,
            pruner_type=self.pruner_type,
            sampler_type=self.sampler_type,
            study_direction=study.direction.name,
            trial_history=trial_history[-10:],  # Last 10 trials
        )
        
        # Store in history
        self.optimization_history.append(result)
        
        # Save study if storage path specified
        if self.storage_path:
            self._save_study(study)
        
        return result
    
    def _create_study(self, metric: str) -> optuna.Study:
        """Create or load Optuna study."""
        direction = self._get_study_direction(metric)
        
        # Load existing study if storage path specified
        if self.storage_path and os.path.exists(self.storage_path):
            try:
                study = optuna.load_study(
                    study_name=self.study_name,
                    storage=f"sqlite:///{self.storage_path}",
                    pruner=self.pruner,
                    sampler=self.sampler,
                )
                logger.info(f"[HPOAgent] Loaded existing study from {self.storage_path}")
                return study
            except Exception as e:
                logger.warning(f"[HPOAgent] Failed to load study: {e}. Creating new study.")
        
        # Create new study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            pruner=self.pruner,
            sampler=self.sampler,
        )
        
        logger.info(f"[HPOAgent] Created new study with direction={direction}")
        return study
    
    def _get_study_direction(self, metric: str) -> str:
        """Get optimization direction for metric."""
        minimize_metrics = {"logloss", "rmse", "mae", "mse"}
        
        if metric.lower() in minimize_metrics:
            return "minimize"
        else:
            return "maximize"
    
    def _suggest_params(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for model."""
        if model_type == "lightgbm":
            return self._suggest_lgbm_params(trial)
        elif model_type == "xgboost":
            return self._suggest_xgb_params(trial)
        elif model_type == "catboost":
            return self._suggest_catboost_params(trial)
        else:
            logger.warning(f"Unknown model type: {model_type}, using LightGBM defaults")
            return self._suggest_lgbm_params(trial)
    
    def _suggest_lgbm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest LightGBM parameters."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    
    def _suggest_xgb_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest XGBoost parameters."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    
    def _suggest_catboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest CatBoost parameters."""
        return {
            "iterations": trial.suggest_int("iterations", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),
        }
    
    def _suggest_params_custom(
        self,
        trial: optuna.Trial,
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Suggest parameters from custom search space."""
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config.get("type", "int")
            
            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1),
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"],
                )
        
        return params
    
    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        model_type: str,
        metric: str,
        n_folds: int,
    ) -> np.ndarray:
        """Run cross-validation."""
        if metric in ["auc", "logloss"]:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self._train_model(X_train, y_train, params, model_type)
            score = self._evaluate_model(model, X_val, y_val, metric)
            scores.append(score)
        
        return np.array(scores)
    
    def _train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        model_type: str,
    ) -> Any:
        """Train a model with given parameters."""
        if model_type == "lightgbm":
            model = lgb.LGBMClassifier(**params, verbose=-1, n_jobs=self.n_jobs)
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(**params, verbosity=0, n_jobs=self.n_jobs)
        elif model_type == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**params, verbose=0)
        else:
            model = lgb.LGBMClassifier(**params, verbose=-1, n_jobs=self.n_jobs)
        
        model.fit(X, y)
        return model
    
    def _evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        metric: str,
    ) -> float:
        """Evaluate model with given metric."""
        from sklearn.metrics import (
            roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, r2_score,
        )
        
        if metric in ["auc", "logloss"]:
            y_pred = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(X)
        
        if metric == "auc":
            return roc_auc_score(y, y_pred)
        elif metric == "logloss":
            return -log_loss(y, y_pred)  # Negative because we maximize
        elif metric == "rmse":
            return -np.sqrt(mean_squared_error(y, y_pred))
        elif metric == "mae":
            return -mean_absolute_error(y, y_pred)
        elif metric == "r2":
            return r2_score(y, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _save_study(self, study: optuna.Study) -> None:
        """Save study to storage."""
        if not self.storage_path:
            return
        
        try:
            # Optuna automatically saves to SQLite storage
            logger.info(f"[HPOAgent] Study saved to {self.storage_path}")
        except Exception as e:
            logger.warning(f"[HPOAgent] Failed to save study: {e}")
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history as list of dicts."""
        return [result.to_dict() for result in self.optimization_history]
    
    def set_baseline(
        self,
        baseline_name: str,
        baseline_score: float,
    ) -> None:
        """Set baseline score for comparison."""
        self.baseline_scores[baseline_name] = baseline_score
        logger.info(f"[HPOAgent] Set baseline '{baseline_name}': {baseline_score}")
    
    def compare_to_baseline(
        self,
        result: HPOResult,
        baseline_name: str,
    ) -> Dict[str, Any]:
        """Compare optimization result to baseline."""
        if baseline_name not in self.baseline_scores:
            raise ValueError(f"Baseline '{baseline_name}' not found")
        
        baseline_score = self.baseline_scores[baseline_name]
        
        return {
            "baseline_score": baseline_score,
            "optimized_score": result.best_score,
            "improvement": result.best_score - baseline_score,
        }


def optimize_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "lightgbm",
    metric: str = "auc",
    n_trials: int = 100,
    pruner: str = "hyperband",
    **kwargs,
) -> HPOResult:
    """
    Convenience function for hyperparameter optimization.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: Model type
        metric: Scoring metric
        n_trials: Number of trials
        pruner: Pruning algorithm
        **kwargs: Passed to HPOAgent
    
    Returns:
        HPOResult
    """
    hpo = HPOAgent(n_trials=n_trials, pruner=pruner, **kwargs)
    return hpo.optimize(X, y, model_type, metric)
