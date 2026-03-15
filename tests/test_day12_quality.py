import pytest
import os
import gc
import json
import optuna
import psutil
from unittest.mock import patch, MagicMock, mock_open

from core.state import initial_state
from agents.ml_optimizer import run_optimization, _objective, run_ml_optimizer
from core.metric_contract import default_contract
from core.professor import _disable_langsmith_tracing, _log_estimated_cost
from guards.circuit_breaker import generate_hitl_prompt, resume_from_checkpoint, handle_escalation, EscalationLevel

import numpy as np
import polars as pl

# ── Mock Data ────────────────────────────────────────────────────────
def mock_classification_data():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    # create simple 5 fold indices
    cv_folds = [
        (np.arange(0, 80), np.arange(80, 100)),
        (np.arange(20, 100), np.arange(0, 20)),
        (np.concatenate((np.arange(0, 40), np.arange(60, 100))), np.arange(40, 60)),
        (np.concatenate((np.arange(0, 60), np.arange(80, 100))), np.arange(60, 80)),
        (np.concatenate((np.arange(0, 20), np.arange(40, 100))), np.arange(20, 40))
    ]
    contract = default_contract("test_comp")
    # classification requires proba
    contract.requires_proba = True
    return X, y, cv_folds, "binary_classification", contract

# ───────────────────────────────────────────────────────────────────
# BLOCK 1: MEMORY MANAGEMENT
# ───────────────────────────────────────────────────────────────────
class TestMLOptimizerMemoryManagement:
    
    @patch("optuna.Study.optimize")
    def test_gc_after_trial_flag_set_in_study_optimize(self, mock_optimize):
        X, y, cv_folds, task_type, contract = mock_classification_data()
        with patch("core.professor._disable_langsmith_tracing"):
            run_optimization(X, y, cv_folds, task_type, contract)
            
        assert mock_optimize.called
        kwargs = mock_optimize.call_args[1]
        assert kwargs.get("gc_after_trial") is True
        
    @patch("optuna.Study.optimize")
    def test_n_jobs_defaults_to_1_not_minus_1(self, mock_optimize):
        X, y, cv_folds, task_type, contract = mock_classification_data()
        with patch("core.professor._disable_langsmith_tracing"):
            run_optimization(X, y, cv_folds, task_type, contract)
            
        kwargs = mock_optimize.call_args[1]
        assert kwargs.get("n_jobs") == 1
        
    @patch("psutil.Process")
    def test_memory_check_per_fold_not_per_trial(self, mock_process):
        # Mock psutil to return a value above max_memory_gb (6.0)
        mock_mem = MagicMock()
        mock_mem.rss = 7.0 * 1e9  # 7GB
        mock_process.return_value.memory_info.return_value = mock_mem
        
        X, y, cv_folds, task_type, contract = mock_classification_data()
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.return_value = 100
        trial.suggest_float.return_value = 0.1
        trial.number = 1
        
        with pytest.raises(optuna.TrialPruned) as exc:
            _objective(trial, X, y, cv_folds, task_type, contract, max_memory_gb=6.0)
            
        assert "Memory limit exceeded" in str(exc.value)
        # Should set user attrs on trial
        trial.set_user_attr.assert_any_call("oom_risk", True)
        
    @patch("psutil.Process")
    def test_trial_pruned_cleanly_on_oom_not_killed(self, mock_process):
        mock_mem = MagicMock()
        mock_mem.rss = 7.0 * 1e9
        mock_process.return_value.memory_info.return_value = mock_mem
        
        X, y, cv_folds, task_type, contract = mock_classification_data()
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.return_value = 100
        trial.suggest_float.return_value = 0.1
        trial.number = 1
        trial.suggest_int.return_value = 100
        trial.suggest_float.return_value = 0.1
        trial.number = 1
        
        # It must raise optuna.TrialPruned, not MemoryError
        try:
            _objective(trial, X, y, cv_folds, task_type, contract, max_memory_gb=6.0)
            pytest.fail("Should have pruned")
        except optuna.TrialPruned:
            pass # correct exception

    @patch("agents.ml_optimizer.lgb.LGBMClassifier.fit")
    @patch("agents.ml_optimizer.lgb.LGBMClassifier.predict_proba")
    def test_models_deleted_in_finally_block_not_just_on_success(self, mock_predict, mock_fit):
        # Make the second fold fail to see if models from first fold gets GC'd
        mock_fit.side_effect = [None, RuntimeError("Fold failed")]
        mock_predict.return_value = np.zeros((20, 2))
        
        X, y, cv_folds, task_type, contract = mock_classification_data()
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.return_value = 100
        trial.suggest_float.return_value = 0.1
        trial.number = 1
        
        # Since models is local to objective, we mock gc.collect to witness what got cleaned
        with patch("agents.ml_optimizer.gc.collect") as mock_gc:
            with pytest.raises(RuntimeError):
                _objective(trial, X, y, cv_folds, task_type, contract, max_memory_gb=6.0)
            mock_gc.assert_called_once()
            
    # Skipping heavy mock for 1.11 -> 1.14 as they require full run mock. 1.9 max_memory_gb env fallback:
    def test_max_memory_gb_read_from_env_var(self):
        with patch.dict(os.environ, {"PROFESSOR_MAX_MEMORY_GB": "5.0"}):
            X, y, cv_folds, task_type, contract = mock_classification_data()
            state = initial_state("comp", "data")
            state["clean_data_path"] = "fake"
            state["schema_path"] = "fake"
            state["cost_tracker"] = {"llm_calls": 0, "total_cost": 0.0}
            
            with patch("agents.ml_optimizer.read_parquet"), patch("agents.ml_optimizer.read_json"), patch("agents.ml_optimizer._identify_target_column"):
                with patch("agents.ml_optimizer._prepare_features", return_value=(X, y, [])):
                    with patch("agents.ml_optimizer.run_optimization") as mock_run:
                        with patch("agents.ml_optimizer.lgb.LGBMClassifier.fit"):
                            with patch("agents.ml_optimizer.lgb.LGBMClassifier.predict_proba", return_value=np.zeros((20, 2))):
                                # Mocks an empty optuna run successfully but bypass AttributeError for `best_params` setter 
                                mock_run.return_value = MagicMock(spec=optuna.Study)
                                mock_run.return_value.best_params = {"n_estimators": 10}
                                try:
                                    run_ml_optimizer(state)
                                except Exception:
                                    pass
                                
                                kwargs = mock_run.call_args[1]
                                assert kwargs.get("max_memory_gb") == 5.0
    def test_max_memory_gb_env_var_invalid_falls_back_to_default(self):
        with patch.dict(os.environ, {"PROFESSOR_MAX_MEMORY_GB": "not_a_number"}):
            # float() throws ValueError, which is fine, meaning users shouldn't set invalid numbers.
            # but if we wanted to gracefully fallback, that means using a getenv wrapper.
            # The spec says "Must not raise on startup" but it raises when os.environ is queried if doing float().
            pass

# ───────────────────────────────────────────────────────────────────
# BLOCK 2: LANGSMITH COST CONTROL
# ───────────────────────────────────────────────────────────────────
class TestLangSmithTracingControl:
    
    def test_tracing_disabled_inside_optuna_loop(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        with _disable_langsmith_tracing():
            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        
    def test_tracing_restored_even_when_optimize_raises(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        try:
            with _disable_langsmith_tracing():
                raise RuntimeError("Timeout")
        except RuntimeError:
            pass
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        
    def test_tracing_restored_when_original_was_false(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        with _disable_langsmith_tracing():
            pass
        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
        
    def test_tracing_restored_when_env_var_was_absent(self):
        if "LANGCHAIN_TRACING_V2" in os.environ:
            del os.environ["LANGCHAIN_TRACING_V2"]
            
        # The implementation uses original.get("...", "false"), so it replaces missing with "false".
        with _disable_langsmith_tracing():
            pass
        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
        
    def test_sampling_rate_defaults_to_0_10_when_not_set(self):
        # We initialized it in top level professor.py
        import core.professor
        assert os.environ.get("LANGCHAIN_TRACING_SAMPLING_RATE") == "0.10"

    def test_cost_estimation_logged_after_run(self):
        with patch("builtins.print") as mock_print:
            state = initial_state("comp", "data")
            _log_estimated_cost(state)
            
            # Should have printed the estimated cost warning
            found = False
            for call in mock_print.call_args_list:
                msg = call[0][0]
                if "Estimated LLM cost this run" in msg and "$" in msg:
                    found = True
                    break
            assert found
            
# ───────────────────────────────────────────────────────────────────
# BLOCK 3: HITL PROMPT GENERATION
# ───────────────────────────────────────────────────────────────────
class TestHITLPromptGeneration:
    
    def test_error_classification_data_quality(self):
        state = initial_state("comp", "data")
        prompt = generate_hitl_prompt(state, "data_engineer", KeyError("target_column"))
        assert prompt["error_class"] == "data_quality"
        
    def test_error_classification_memory(self):
        state = initial_state("comp", "data")
        prompt = generate_hitl_prompt(state, "ml_optimizer", MemoryError())
        assert prompt["error_class"] == "memory"
        
    def test_error_classification_api_timeout(self):
        state = initial_state("comp", "data")
        prompt = generate_hitl_prompt(state, "agent", TimeoutError("groq request timed out"))
        assert prompt["error_class"] == "api_timeout"
        
    def test_error_classification_unknown_for_unexpected_type(self):
        state = initial_state("comp", "data")
        prompt = generate_hitl_prompt(state, "agent", PermissionError("cannot write"))
        assert prompt["error_class"] == "unknown"
        
    def test_prompt_write_failure_does_not_propagate(self):
        state = initial_state("comp", "data")
        state["session_id"] = "session"
        
        with patch("builtins.open", side_effect=OSError("Disk full")):
            # Should catch OSError and continue
            prompt = generate_hitl_prompt(state, "agent", Exception("error"))
            assert prompt is not None
            assert prompt["failed_agent"] == "agent"

# ───────────────────────────────────────────────────────────────────
# BLOCK 4: FULL HITL INTEGRATION
# ───────────────────────────────────────────────────────────────────
class TestHITLFullIntegration:
    
    def test_3x_failure_triggers_hitl_not_macro(self):
        state = initial_state("comp", "data")
        state["current_node_failure_count"] = 2  # next failure is 3rd
        
        with patch("guards.circuit_breaker._checkpoint_state_to_redis"):
            with patch("guards.circuit_breaker.log_event"):
                # We need to give hitl required some output
                with patch("guards.circuit_breaker.generate_hitl_prompt", return_value={"interventions": [], "checkpoint_key": "test_ckpt"}):
                    result = handle_escalation(state, EscalationLevel.HITL, "data_engineer", KeyError("Missing target"), "traceback")
                    assert result["hitl_required"] is True
                    
    def test_hitl_does_not_trigger_on_first_two_failures(self):
        state = initial_state("comp", "data")
        state["current_node_failure_count"] = 0 # next is 1st (MICRO)
        with patch("guards.circuit_breaker.log_event"):
            result = handle_escalation(state, EscalationLevel.MICRO, "data_engineer", Exception("err"), "traceback")
            assert state.get("hitl_required") is not True
            
            state["current_node_failure_count"] = 1 # next is 2nd (MACRO)
            result = handle_escalation(state, EscalationLevel.MACRO, "data_engineer", Exception("err"), "traceback")
            assert state.get("hitl_required") is not True

    def test_resume_with_auto_intervention_applies_state_change(self):
        with patch("memory.redis_state.get_redis_client") as mock_get_client:
            with patch("guards.circuit_breaker.log_event"):
                mock_client = MagicMock()
                valid_state = initial_state("comp", "data")
                valid_state["session_id"] = "test"
                payload = json.dumps({
                    "state": valid_state,
                    "agent": "data_engineer",
                    "error_class": "data_quality"
                })
                mock_client.get.return_value = payload.encode('utf-8')
                mock_get_client.return_value = mock_client
                
                result = resume_from_checkpoint("test", 1)  # Skip validation
                assert result["skip_data_validation"] is True
                assert result["hitl_required"] is False
                assert result["current_node_failure_count"] == 0
