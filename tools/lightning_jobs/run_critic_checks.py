import os
import json
import argparse
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split
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
    result_path = os.path.join(data_dir, "critic_checks_result.json")

    result = {
        "success": False,
        "permutation": {
            "baseline_score": 0.0,
            "importances": {},
            "negative_features": {},
            "spurious_features": []
        },
        "adversarial": {
            "auc": 0.5,
            "verdict": "OK",
            "note": ""
        },
        "runtime_seconds": 0.0
    }

    try:
        if not os.path.exists(registry_path):
            raise FileNotFoundError(f"Missing {registry_path}")
            
        with open(registry_path, "r") as f:
            registry = json.load(f)
            
        # Get features from the winning model
        winner = None
        if isinstance(registry, dict) and "models" in registry:
            # Maybe it looks like typical professor registry
            models = registry.get("models", {})
            if models:
                winner_key = list(models.keys())[0]
                winner = models[winner_key]
        elif isinstance(registry, list) and len(registry) > 0:
            winner = registry[0]
            
        if winner and "features" in winner:
            features = winner["features"]
        else:
            # Fallback if structure differs
            features = []
            
        df_train = pl.read_csv(train_path, null_values=["", "NA", "NaN", "null"])
        
        if not features:
            features = [c for c in df_train.columns if c != args.target_col]
            
        # Optional test data for adversarial
        df_test = None
        if os.path.exists(test_path):
            df_test = pl.read_csv(test_path, null_values=["", "NA", "NaN", "null"])

        # Prepare Train
        for col in features:
            if df_train[col].dtype in (pl.String, pl.Categorical):
                df_train = df_train.with_columns(pl.col(col).cast(pl.Categorical).to_physical().fill_null(0))
            else:
                df_train = df_train.with_columns(pl.col(col).fill_null(0))

        X = df_train.select(features).to_numpy()
        y = df_train[args.target_col].to_numpy()
        
        lgb_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "verbosity": -1,
            "n_jobs": 1,
        }
        if args.task_type == "binary":
            lgb_params["objective"] = "binary"
            ModelClass = lgb.LGBMClassifier
        elif args.task_type == "multiclass":
            lgb_params["objective"] = "multiclass"
            ModelClass = lgb.LGBMClassifier
        else:
            lgb_params["objective"] = "regression"
            ModelClass = lgb.LGBMRegressor

        # ----------------------------------------------------
        # 1. Permutation Importance
        # ----------------------------------------------------
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = ModelClass(**lgb_params)
        model.fit(X_tr, y_tr)
        
        def score_func(X_data, y_data):
            if args.task_type == "binary":
                preds = model.predict_proba(X_data)[:, 1]
                return roc_auc_score(y_data, preds)
            elif args.task_type == "multiclass":
                preds = model.predict_proba(X_data)
                return -log_loss(y_data, preds)
            else:
                preds = model.predict(X_data)
                return -np.sqrt(mean_squared_error(y_data, preds))

        baseline_score = score_func(X_val, y_val)
        result["permutation"]["baseline_score"] = round(float(baseline_score), 4)
        
        rng = np.random.default_rng(42)
        importances = {}
        for i, f in enumerate(features):
            X_val_shuf = X_val.copy()
            X_val_shuf[:, i] = rng.permutation(X_val_shuf[:, i])
            shuf_score = score_func(X_val_shuf, y_val)
            # drop in score = importance
            drop = baseline_score - shuf_score
            importances[f] = round(float(drop), 4)

        result["permutation"]["importances"] = importances
        
        neg_features = {f: v for f, v in importances.items() if v < 0}
        result["permutation"]["negative_features"] = neg_features
        if neg_features:
            result["permutation"]["spurious_features"] = list(neg_features.keys())

        # ----------------------------------------------------
        # 2. Adversarial Validation
        # ----------------------------------------------------
        if df_test is not None:
            # Common features
            common = [c for c in features if c in df_test.columns]
            if common:
                for col in common:
                    if df_test[col].dtype in (pl.String, pl.Categorical):
                        df_test = df_test.with_columns(pl.col(col).cast(pl.Categorical).to_physical().fill_null(0))
                    else:
                        df_test = df_test.with_columns(pl.col(col).fill_null(0))
                
                Xt = df_train.select(common).to_numpy()
                Xtest = df_test.select(common).to_numpy()
                
                # Label 0 for train, 1 for test
                yt = np.zeros(len(Xt))
                ytest = np.ones(len(Xtest))
                
                X_adv = np.vstack((Xt, Xtest))
                y_adv = np.concatenate((yt, ytest))
                
                X_adv_tr, X_adv_val, y_adv_tr, y_adv_val = train_test_split(X_adv, y_adv, test_size=0.2, random_state=42)
                
                adv_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, n_jobs=1, verbosity=-1)
                adv_model.fit(X_adv_tr, y_adv_tr)
                
                adv_preds = adv_model.predict_proba(X_adv_val)[:, 1]
                auc = roc_auc_score(y_adv_val, adv_preds)
                result["adversarial"]["auc"] = round(float(auc), 4)
                
                if auc > 0.70:
                    result["adversarial"]["verdict"] = "HIGH"
                    result["adversarial"]["note"] = "Significant distribution shift detected (AUC > 0.70)."
                else:
                    result["adversarial"]["verdict"] = "OK"
                    result["adversarial"]["note"] = "No significant distribution shift detected."

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        
    finally:
        result["runtime_seconds"] = round(time.time() - t_start, 1)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
