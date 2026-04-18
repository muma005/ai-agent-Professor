import os
import json
import argparse
import time
import numpy as np
import polars as pl
import lightgbm as lgb
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
    survivors_path = os.path.join(data_dir, "stage1_survivors.json")
    result_path = os.path.join(data_dir, "null_importance_stage2_result.json")

    result = {
        "success": False,
        "survivors": [],
        "dropped_stage2": [],
        "importance_scores": {},
        "runtime_seconds": 0.0
    }

    try:
        if not os.path.exists(survivors_path):
            raise FileNotFoundError(f"Missing {survivors_path}")
            
        with open(survivors_path, "r") as f:
            survivor_names = json.load(f)

        if not survivor_names:
            result["success"] = True
            return
            
        df = pl.read_csv(train_path, null_values=["", "NA", "NaN", "null"])
        
        for col in survivor_names:
            if df[col].dtype in (pl.String, pl.Categorical):
                df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().fill_null(0))
            else:
                df = df.with_columns(pl.col(col).fill_null(0))

        X_np = df.select(survivor_names).to_numpy()
        y = df[args.target_col].to_numpy()
        
        lgb_params = {
            "objective": "multiclass" if args.task_type == "multiclass" else "binary" if args.task_type == "binary" else "regression",
            "n_estimators": 200,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbosity": -1,
            "n_jobs": 1,
        }

        ModelClass = lgb.LGBMClassifier if args.task_type in ("binary", "multiclass") else lgb.LGBMRegressor
        
        # Actual Importances
        model_real = ModelClass(**lgb_params)
        model_real.fit(X_np, y)
        actual_importances = dict(zip(survivor_names, model_real.feature_importances_))
        del model_real
        gc.collect()

        # 50 shuffles
        n_shuffles = 50
        random_seed = 42
        rng = np.random.default_rng(seed=random_seed)
        null_records = {f: [] for f in survivor_names}
        
        for i in range(n_shuffles):
            y_shuf = rng.permutation(y)
            model_null = ModelClass(**lgb_params)
            model_null.fit(X_np, y_shuf)
            for f, imp in zip(survivor_names, model_null.feature_importances_):
                null_records[f].append(float(imp))
            del model_null
            gc.collect()

        # Compute percentiles
        threshold_percentile = 95.0
        stage2_survivors = []
        stage2_dropped = []
        
        for f in survivor_names:
            actual = actual_importances.get(f, 0.0)
            null_dist = null_records.get(f, [])
            
            if not null_dist:
                stage2_survivors.append(f)
                continue
                
            threshold = float(np.percentile(null_dist, threshold_percentile))
            if actual > threshold:
                stage2_survivors.append(f)
            else:
                stage2_dropped.append(f)
                
            result["importance_scores"][f] = round(float(actual), 4)

        result["survivors"] = stage2_survivors
        result["dropped_stage2"] = stage2_dropped
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        
    finally:
        result["runtime_seconds"] = round(time.time() - t_start, 1)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
