import os
import json
import argparse
import time
import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
import gc

def main():
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--task_type", default="binary")
    args = parser.parse_args()

    data_dir = f"/home/{args.session_id}"
    train_path = os.path.join(data_dir, "train.csv")
    optuna_path = os.path.join(data_dir, "optuna_result.json")
    order_path = os.path.join(data_dir, "feature_order.json")
    result_path = os.path.join(data_dir, "stability_result.json")

    result = {
        "success": False,
        "winner": {},
        "all_results": [],
        "runtime_seconds": 0.0
    }

    try:
        if not os.path.exists(optuna_path):
            raise FileNotFoundError(f"Missing {optuna_path}")
            
        with open(optuna_path, "r") as f:
            optuna_result = json.load(f)
            
        top_trials = optuna_result.get("top_trials", [])
        if not top_trials:
            result["success"] = True
            return
            
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
        task_type = args.task_type
        
        all_results = []
        seeds = [42, 84, 126, 168, 210]
        
        for config_idx, config_data in enumerate(top_trials):
            params = config_data["params"].copy()
            model_type = params.pop("model_type", "lgbm")
            
            # Map params back logically if needed
            clean_params = {}
            for k, v in params.items():
                if k.startswith(f"{model_type}_"):
                    clean_params[k[len(model_type)+1:]] = v
                else:
                    clean_params[k] = v
                    
            if model_type == "lgbm":
                clean_params["verbosity"] = -1
                clean_params["n_jobs"] = 1
                if task_type in ("binary", "multiclass"):
                    clean_params["objective"] = task_type
                    ModelClass = lgb.LGBMClassifier
                else:
                    clean_params["objective"] = "regression"
                    ModelClass = lgb.LGBMRegressor
            elif model_type == "xgb":
                clean_params["verbosity"] = 0
                clean_params["n_jobs"] = 1
                if task_type in ("binary", "multiclass"):
                    ModelClass = xgb.XGBClassifier
                else:
                    ModelClass = xgb.XGBRegressor
            else:
                clean_params["verbose"] = 0
                clean_params["thread_count"] = 1
                if task_type in ("binary", "multiclass"):
                    ModelClass = CatBoostClassifier
                else:
                    ModelClass = CatBoostRegressor

            seed_scores = []
            
            for seed in seeds:
                if task_type in ("binary", "multiclass"):
                    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                else:
                    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
                    
                fold_scores = []
                for train_idx, val_idx in kf.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Update catboost random seed per fit
                    if model_type == "cb":
                        clean_params["random_seed"] = seed
                    # XGB / LGBM use fixed seeds typically or handle it if we pass random_state
                    if model_type in ("lgbm", "xgb"):
                        clean_params["random_state"] = seed
                        
                    model = ModelClass(**clean_params)
                    model.fit(X_train, y_train)
                    
                    if task_type == "binary":
                        from sklearn.metrics import roc_auc_score
                        preds = model.predict_proba(X_val)[:, 1]
                        score = roc_auc_score(y_val, preds)
                    elif task_type == "multiclass":
                        from sklearn.metrics import log_loss
                        preds = model.predict_proba(X_val)
                        score = -log_loss(y_val, preds)
                    else:
                        from sklearn.metrics import mean_squared_error
                        preds = model.predict(X_val)
                        score = -np.sqrt(mean_squared_error(y_val, preds))
                        
                    fold_scores.append(score)
                seed_scores.append(float(np.mean(fold_scores)))
                
            mean = float(np.mean(seed_scores))
            std = float(np.std(seed_scores))
            
            # stability_score = mean - 1.5 * std
            # If negative task (like neg_log_loss), we maximize stability score the same way.
            stability_score = mean - 1.5 * std
            
            # Restore model_type to params for returning
            ret_params = config_data["params"].copy()
            
            all_results.append({
                "params": ret_params,
                "stability_score": round(stability_score, 4),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "seed_results": [round(s, 4) for s in seed_scores]
            })

        all_results.sort(key=lambda x: x["stability_score"], reverse=True)
        
        result["all_results"] = all_results
        if all_results:
            result["winner"] = all_results[0]
            
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        
    finally:
        result["runtime_seconds"] = round(time.time() - t_start, 1)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
