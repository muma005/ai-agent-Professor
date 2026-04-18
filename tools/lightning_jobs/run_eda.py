import os
import json
import argparse
import polars as pl
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import time

def main():
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--task_type", default="binary")
    args = parser.parse_args()

    data_dir = f"/home/{args.session_id}"
    train_path = os.path.join(data_dir, "train.csv")
    result_path = os.path.join(data_dir, "eda_result.json")

    result = {
        "success": False,
        "correlation_matrix": {},
        "mutual_info_scores": {},
        "column_stats": {},
        "high_null_columns": [],
        "high_cardinality_columns": [],
        "constant_columns": [],
        "recommended_drops": [],
        "runtime_seconds": 0.0
    }

    try:
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing {train_path}")

        df = pl.read_csv(train_path, null_values=["", "NA", "NaN", "null"])
        target_col = args.target_col
        features = [c for c in df.columns if c != target_col]

        # 1. Column stats & Basic filtering
        for col in features:
            try:
                col_data = df[col]
                null_rate = col_data.null_count() / len(df)
                unique_count = col_data.n_unique()
                
                if null_rate > 0.95:
                    result["high_null_columns"].append(col)
                    result["recommended_drops"].append(col)
                if unique_count == 1:
                    result["constant_columns"].append(col)
                    result["recommended_drops"].append(col)
                if col_data.dtype in (pl.String, pl.Categorical) and unique_count > len(df) * 0.9:
                    result["high_cardinality_columns"].append(col)
                    result["recommended_drops"].append(col)

                if col_data.dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                    # compute mean, std, skew without nulls
                    clean_data = col_data.drop_nulls()
                    if len(clean_data) > 0:
                        mean = clean_data.mean()
                        std = clean_data.std()
                        # simple skew proxy
                        median = clean_data.median()
                        skew = (mean - median) / (std + 1e-6) if std else 0.0
                        result["column_stats"][col] = {
                            "mean": float(mean) if mean is not None else 0.0,
                            "std": float(std) if std is not None else 0.0,
                            "null_rate": float(null_rate),
                            "skew": float(skew)
                        }
            except Exception:
                pass

        # Valid numeric features for ML
        num_features = [c for c in features if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64) and c not in result["recommended_drops"]]
        
        # 2. Correlation Matrix (Spearman) - top 50 by variance
        if len(num_features) > 0:
            variances = []
            for c in num_features:
                var = df[c].std()
                variances.append((c, var if var is not None else 0.0))
            variances.sort(key=lambda x: x[1], reverse=True)
            top_50 = [x[0] for x in variances[:50]]
            
            # Compute pairwise correlations
            clean_df = df.select(top_50).drop_nulls().to_pandas()
            if len(clean_df) > 10:
                corr_matrix = spearmanr(clean_df).statistic
                if isinstance(corr_matrix, float):
                    if len(top_50) == 2:
                        result["correlation_matrix"][f"{top_50[0]}_vs_{top_50[1]}"] = float(corr_matrix)
                elif hasattr(corr_matrix, "shape"):
                    for i in range(len(top_50)):
                        for j in range(i+1, len(top_50)):
                            val = corr_matrix[i, j]
                            if not np.isnan(val) and abs(val) > 0.3:  # Only save significant correlations
                                result["correlation_matrix"][f"{top_50[i]}_vs_{top_50[j]}"] = round(float(val), 4)

        # 3. Mutual Information (categorical & numerical)
        clean_df_mi = df.drop_nulls().to_pandas()
        if len(clean_df_mi) > 100 and target_col in clean_df_mi.columns:
            y = clean_df_mi[target_col]
            X = clean_df_mi[features]
            
            # Encode categorical strings
            for col in X.columns:
                if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                    
            if args.task_type in ("binary", "multiclass"):
                try:
                    mi = mutual_info_classif(X, y)
                    for f, score in zip(features, mi):
                        if score > 0.01:
                            result["mutual_info_scores"][f] = round(float(score), 4)
                except Exception:
                    pass

        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
        
    finally:
        result["runtime_seconds"] = round(time.time() - t_start, 1)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
