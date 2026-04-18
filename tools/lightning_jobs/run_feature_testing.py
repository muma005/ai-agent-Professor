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
    candidates_path = os.path.join(data_dir, "feature_candidates.json")
    result_path = os.path.join(data_dir, "feature_testing_result.json")

    result = {
        "success": False,
        "survivors": [],
        "dropped": [],
        "importance_scores": {},
        "runtime_seconds": 0.0
    }

    try:
        if not os.path.exists(candidates_path):
            raise FileNotFoundError(f"Missing {candidates_path}")
            
        with open(candidates_path, "r") as f:
            candidates = json.load(f)

        if not candidates:
            # Nothing to test
            result["success"] = True
            return
            
        df = pl.read_csv(train_path, null_values=["", "NA", "NaN", "null"])
        target_col = args.target_col
        y = df[target_col].to_numpy()
        
        # We test features INDIVIDUALLY to save memory, or all together.
        # Stage 1 null importance tests all features together vs random noise.
        # But wait, applying transforms:
        generated_features = []
        for c in candidates:
            # Support both format from Professor: "expression" string, and the one from prompt "transform" and "source_columns"
            name = c.get("name")
            expr_str = c.get("expression")
            if not expr_str:
                # Provide basic support if it's the simplified dict format
                src = c.get("source_columns", [])
                transform = c.get("transform", "")
                if transform == "sum":
                    df = df.with_columns(pl.sum_horizontal(src).alias(name))
                    generated_features.append(name)
                elif transform == "str_split_0":
                    df = df.with_columns(pl.col(src[0]).str.split("_").list.get(0).alias(name))
                    generated_features.append(name)
            else:
                # It's an AST expression string, need to eval
                try:
                    # In polars context, we can generally eval the string 
                    # but we must provide pl to eval
                    expr_obj = eval(expr_str, {"pl": pl})
                    df = df.with_columns(expr_obj.alias(name))
                    generated_features.append(name)
                except Exception as e:
                    pass

        # If we failed to generate any, exit
        if not generated_features:
            result["success"] = True
            return

        # Prepare data for 5-shuffle LightGBM test
        # We only need to test the generated features! Wait, we also need some base features to give context?
        # Typically null importance is run on the features to test.
        # Fill nulls and string encode for LGBM
        X_df = df.select(generated_features)
        
        for col in X_df.columns:
            if X_df[col].dtype in (pl.String, pl.Categorical, pl.Categorical):
                X_df = X_df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().fill_null(0))
            else:
                X_df = X_df.with_columns(pl.col(col).fill_null(0))
                
        X_np = X_df.to_numpy()
        
        lgb_params = {
            "objective": "multiclass" if args.task_type == "multiclass" else "binary" if args.task_type == "binary" else "regression",
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "verbosity": -1,
            "n_jobs": 1,
        }

        ModelClass = lgb.LGBMClassifier if args.task_type in ("binary", "multiclass") else lgb.LGBMRegressor
        
        # Actual Importances
        model_real = ModelClass(**lgb_params)
        model_real.fit(X_np, y)
        actual_importances = dict(zip(generated_features, model_real.feature_importances_))
        del model_real
        gc.collect()

        # 5 shuffles
        rng = np.random.default_rng(seed=42)
        n_shuffles = 5
        null_sums = {f: 0.0 for f in generated_features}
        
        for _ in range(n_shuffles):
            y_shuf = rng.permutation(y)
            model_null = ModelClass(**lgb_params)
            model_null.fit(X_np, y_shuf)
            for f, imp in zip(generated_features, model_null.feature_importances_):
                null_sums[f] += float(imp)
            del model_null
            gc.collect()

        # Compare ratios
        ratios = {
            f: actual_importances[f] / ( (null_sums[f] / n_shuffles) + 1e-6)
            for f in generated_features
        }
        
        threshold = float(np.percentile(list(ratios.values()), 65.0)) if ratios else 0.0
        
        survivors = []
        dropped = []
        for f in generated_features:
            if ratios[f] >= threshold:
                survivors.append(f)
            else:
                dropped.append(f)
                
            result["importance_scores"][f] = round(float(actual_importances[f]), 4)
            
        result["survivors"] = survivors
        result["dropped"] = dropped
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        
    finally:
        result["runtime_seconds"] = round(time.time() - t_start, 1)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
