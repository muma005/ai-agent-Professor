import os
import json
import argparse
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
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
    test_path = os.path.join(data_dir, "test.csv")
    registry_path = os.path.join(data_dir, "model_registry.json")
    result_path = os.path.join(data_dir, "pseudo_label_result.json")

    result = {
        "success": False,
        "pseudo_labels_applied": False,
        "iterations_completed": 0,
        "n_labels_added": 0,
        "cv_improvements": [],
        "halt_reason": "",
        "runtime_seconds": 0.0
    }

    try:
        # Load registry to find features
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                registry = json.load(f)
            winner = None
            if isinstance(registry, dict) and "models" in registry:
                models = registry.get("models", {})
                if models:
                    winner_key = list(models.keys())[0]
                    winner = models[winner_key]
            elif isinstance(registry, list) and len(registry) > 0:
                winner = registry[0]
                
            if winner and "features" in winner:
                features = winner["features"]
            else:
                features = []
        else:
            features = []

        if not os.path.exists(test_path):
            result["halt_reason"] = "no_test_data"
            result["success"] = True
            return

        df_train = pl.read_csv(train_path, null_values=["", "NA", "NaN", "null"])
        df_test = pl.read_csv(test_path, null_values=["", "NA", "NaN", "null"])
        
        if not features:
            features = [c for c in df_train.columns if c != args.target_col]
        # Keep only common
        features = [c for c in features if c in df_test.columns]
        
        for col in features:
            if df_train[col].dtype in (pl.String, pl.Categorical):
                df_train = df_train.with_columns(pl.col(col).cast(pl.Categorical).to_physical().fill_null(0))
                df_test = df_test.with_columns(pl.col(col).cast(pl.Categorical).to_physical().fill_null(0))
            else:
                df_train = df_train.with_columns(pl.col(col).fill_null(0))
                df_test = df_test.with_columns(pl.col(col).fill_null(0))

        X_train = df_train.select(features).to_numpy()
        y_train = df_train[args.target_col].to_numpy()
        X_test = df_test.select(features).to_numpy()
        
        lgb_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "verbosity": -1,
            "n_jobs": 1,
        }
        if args.task_type == "binary":
            lgb_params["objective"] = "binary"
            ModelClass = lgb.LGBMClassifier
            n_folds = 5
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif args.task_type == "multiclass":
            lgb_params["objective"] = "multiclass"
            ModelClass = lgb.LGBMClassifier
            n_folds = 5
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            lgb_params["objective"] = "regression"
            ModelClass = lgb.LGBMRegressor
            n_folds = 5
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        def cv_score(X, y, X_pseudo=None, y_pseudo=None):
            scores = []
            for train_idx, val_idx in kf.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                if X_pseudo is not None and len(X_pseudo) > 0:
                    X_tr = np.vstack((X_tr, X_pseudo))
                    y_tr = np.concatenate((y_tr, y_pseudo))
                    
                model = ModelClass(**lgb_params)
                model.fit(X_tr, y_tr)
                
                if args.task_type == "binary":
                    preds = model.predict_proba(X_val)[:, 1]
                    scores.append(roc_auc_score(y_val, preds))
                elif args.task_type == "multiclass":
                    preds = model.predict_proba(X_val)
                    scores.append(-log_loss(y_val, preds))
                else:
                    preds = model.predict(X_val)
                    scores.append(-np.sqrt(mean_squared_error(y_val, preds)))
            return np.mean(scores)
            
        baseline_score = cv_score(X_train, y_train)
        
        max_iters = 3
        current_score = baseline_score
        pseudo_labels_accumulated = []
        
        # We need an initial model to predict on test
        model_init = ModelClass(**lgb_params)
        model_init.fit(X_train, y_train)
        
        test_preds_cache = None
        if args.task_type in ("binary", "multiclass"):
            test_preds_cache = model_init.predict_proba(X_test)
        else:
            test_preds_cache = model_init.predict(X_test)
            
        for i in range(max_iters):
            if args.task_type == "binary":
                # Confident ones > 0.95 or < 0.05
                conf_pos = test_preds_cache[:, 1] > 0.95
                conf_neg = test_preds_cache[:, 1] < 0.05
                conf_mask = conf_pos | conf_neg
                if not conf_mask.any():
                    result["halt_reason"] = "no_confident_samples"
                    break
                y_pseudo = np.where(conf_pos, 1, 0)[conf_mask]
            else:
                # Basic regression: No pseudo labeling easily standard, just skip
                result["halt_reason"] = "not_supported_for_regression"
                break
                
            X_pseudo = X_test[conf_mask]
            
            new_score = cv_score(X_train, y_train, X_pseudo, y_pseudo)
            improvement = new_score - current_score
            
            if improvement > 0.0001:
                result["pseudo_labels_applied"] = True
                result["iterations_completed"] += 1
                result["n_labels_added"] = int(np.sum(conf_mask))
                result["cv_improvements"].append(round(float(improvement), 4))
                current_score = new_score
                
                # Update model and test preds for next iteration
                model_new = ModelClass(**lgb_params)
                model_new.fit(np.vstack((X_train, X_pseudo)), np.concatenate((y_train, y_pseudo)))
                test_preds_cache = model_new.predict_proba(X_test)
            else:
                result["halt_reason"] = "wilcoxon_gate_failed"
                break
                
        if not result["halt_reason"]:
            result["halt_reason"] = "max_iterations_reached"

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        
    finally:
        result["runtime_seconds"] = round(time.time() - t_start, 1)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
