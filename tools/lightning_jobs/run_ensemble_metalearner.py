import os
import json
import argparse
import time
import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error

def main():
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--task_type", default="binary")
    args = parser.parse_args()

    data_dir = f"/home/{args.session_id}"
    oof_path = os.path.join(data_dir, "oof_stack.npy")
    y_path = os.path.join(data_dir, "y_train.npy")
    models_path = os.path.join(data_dir, "selected_models.json")
    result_path = os.path.join(data_dir, "ensemble_metalearner_result.json")

    result = {
        "success": False,
        "meta_cv_score": 0.0,
        "meta_cv_std": 0.0,
        "meta_coefficients": [],
        "runtime_seconds": 0.0
    }

    try:
        if not os.path.exists(oof_path) or not os.path.exists(y_path):
            raise FileNotFoundError("Missing OOF or target NPY files.")
            
        X = np.load(oof_path)
        y = np.load(y_path)
        
        n_models = X.shape[1]
        
        if args.task_type in ("binary", "multiclass"):
            model = RidgeClassifier(alpha=1.0)
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            model = Ridge(alpha=1.0)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
        scores = []
        coefs = []
        
        for train_idx, val_idx in kf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            
            if hasattr(model, "coef_"):
                # Ridge classifier coef_ is (1, n_features) or (n_classes, n_features)
                coefs.append(model.coef_.flatten().tolist())
                
            if args.task_type == "binary":
                # RidgeClassifier doesn't have predict_proba natively, uses decision_function
                preds = model.decision_function(X[val_idx])
                scores.append(roc_auc_score(y[val_idx], preds))
            elif args.task_type == "multiclass":
                preds = model.decision_function(X[val_idx])
                # Using dummy accuracy for ridge multiclass
                from sklearn.metrics import accuracy_score
                scores.append(accuracy_score(y[val_idx], model.predict(X[val_idx])))
            else:
                preds = model.predict(X[val_idx])
                scores.append(-np.sqrt(mean_squared_error(y[val_idx], preds)))
                
        result["meta_cv_score"] = round(float(np.mean(scores)), 4)
        result["meta_cv_std"] = round(float(np.std(scores)), 4)
        result["meta_coefficients"] = coefs
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        
    finally:
        result["runtime_seconds"] = round(time.time() - t_start, 1)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
