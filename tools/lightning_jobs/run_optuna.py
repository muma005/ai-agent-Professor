import os
import json
import argparse
import time
import numpy as np
import polars as pl
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
import gc
import warnings
warnings.filterwarnings("ignore")

def optimize(trial, X, y, task_type, models_to_try, n_folds, random_state=42):
    model_type = trial.suggest_categorical("model_type", models_to_try)
    
    if model_type == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("lgbm_n_estimators", 50, 1000),
            "learning_rate": trial.suggest_float("lgbm_lr", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("lgbm_num_leaves", 15, 255),
            "max_depth": trial.suggest_int("lgbm_max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("lgbm_min_child", 5, 100),
            "feature_fraction": trial.suggest_float("lgbm_feat_frac", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("lgbm_bag_frac", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("lgbm_bag_freq", 1, 7),
            "reg_alpha": trial.suggest_float("lgbm_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("lgbm_lambda", 1e-8, 10.0, log=True),
            "verbosity": -1,
            "n_jobs": 1
        }
        if task_type == "binary":
            params["objective"] = "binary"
            ModelClass = lgb.LGBMClassifier
        elif task_type == "multiclass":
            params["objective"] = "multiclass"
            ModelClass = lgb.LGBMClassifier
        else:
            params["objective"] = "regression"
            ModelClass = lgb.LGBMRegressor
            
    elif model_type == "xgb":
        params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 1000),
            "learning_rate": trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("xgb_min_child", 1, 20),
            "subsample": trial.suggest_float("xgb_subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.4, 1.0),
            "gamma": trial.suggest_float("xgb_gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("xgb_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("xgb_lambda", 1e-8, 10.0, log=True),
            "n_jobs": 1,
            "verbosity": 0
        }
        if task_type in ("binary", "multiclass"):
            ModelClass = xgb.XGBClassifier
        else:
            ModelClass = xgb.XGBRegressor
            
    elif model_type == "cb":
        params = {
            "iterations": trial.suggest_int("cb_iterations", 50, 1000),
            "learning_rate": trial.suggest_float("cb_lr", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("cb_depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("cb_l2", 1, 100, log=True),
            "random_strength": trial.suggest_float("cb_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("cb_bagging", 0.0, 1.0),
            "min_data_in_leaf": trial.suggest_int("cb_min_data", 1, 50),
            "verbose": 0,
            "thread_count": 1
        }
        if task_type in ("binary", "multiclass"):
            ModelClass = CatBoostClassifier
        else:
            ModelClass = CatBoostRegressor

    # CV Loop
    if task_type in ("binary", "multiclass"):
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = ModelClass(**params)
        model.fit(X_train, y_train)
        
        if task_type == "binary":
            from sklearn.metrics import roc_auc_score
            preds = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, preds)
        elif task_type == "multiclass":
            from sklearn.metrics import log_loss
            preds = model.predict_proba(X_val)
            score = -log_loss(y_val, preds) # maximize negative log loss
        else:
            from sklearn.metrics import mean_squared_error
            preds = model.predict(X_val)
            score = -np.sqrt(mean_squared_error(y_val, preds))
            
        scores.append(score)
        
        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    val = np.mean(scores)
    trial.set_user_attr("fold_scores", scores)
    return val

def main():
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--target_col", required=True)
    args = parser.parse_args()

    data_dir = f"/home/{args.session_id}"
    train_path = os.path.join(data_dir, "train.csv")
    meta_path = os.path.join(data_dir, "meta.json")
    order_path = os.path.join(data_dir, "feature_order.json")
    result_path = os.path.join(data_dir, "optuna_result.json")

    result = {
        "success": False,
        "top_trials": [],
        "best_cv": 0.0,
        "n_trials_run": 0,
        "n_trials_pruned": 0,
        "runtime_seconds": 0.0
    }

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        with open(order_path, "r") as f:
            features = json.load(f)
            
        df = pl.read_csv(train_path, null_values=["", "NA", "NaN", "null"])
        
        for col in features:
            if df[col].dtype in (pl.String, pl.Categorical):
                df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().fill_null(0))
            else:
                df = df.with_columns(pl.col(col).fill_null(0))

        X = df.select(features).to_numpy()
        y = df[args.target_col].to_numpy()
        
        task_type = meta.get("task_type", "binary")
        n_trials = meta.get("n_trials", 30)
        models_to_try = meta.get("models_to_try", ["lgbm", "xgb", "cb"])
        n_folds = meta.get("cv_folds", 5)
        
        pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=1)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        
        study.optimize(
            lambda trial: optimize(trial, X, y, task_type, models_to_try, n_folds),
            n_trials=n_trials,
            n_jobs=1,               # Hard rule
            gc_after_trial=True     # Hard rule
        )
        
        # Build top 10
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        completed_trials.sort(key=lambda t: t.value, reverse=True)
        
        top_10 = []
        for t in completed_trials[:10]:
            top_10.append({
                "params": t.params,
                "mean_cv": t.value,
                "fold_scores": t.user_attrs.get("fold_scores", [])
            })
            
        result["top_trials"] = top_10
        result["best_cv"] = study.best_value if completed_trials else 0.0
        result["n_trials_run"] = len(study.trials)
        result["n_trials_pruned"] = len(pruned_trials)
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        
    finally:
        result["runtime_seconds"] = round(time.time() - t_start, 1)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
