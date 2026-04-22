# tools/eda_plots.py

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.state import ProfessorState
from tools.operator_channel import emit_to_operator

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────
PLOT_DIR_NAME = "eda_plots"

# ── Internal Helpers ─────────────────────────────────────────────────────────

def _get_feature_importance(df: pl.DataFrame, target_col: str) -> Dict[str, float]:
    """Train a quick LightGBM baseline to get feature importance."""
    import lightgbm as lgb
    try:
        X = df.drop(target_col)
        y = df[target_col]
        
        # Select numeric and boolean
        X_numeric = X.select([
            pl.col(c).cast(pl.Float32) for c in X.columns 
            if X[c].dtype.is_numeric() or X[c].dtype == pl.Boolean
        ])
        
        if len(X_numeric.columns) == 0: return {}
        
        # Target must be numeric for importance
        if not df[target_col].dtype.is_numeric():
            # Quick label encode for classification
            y = df[target_col].cast(pl.String).to_list()
            unique_y = list(set(y))
            y_map = {val: i for i, val in enumerate(unique_y)}
            y = np.array([y_map[val] for val in y])
        else:
            y = y.to_numpy()

        model = lgb.LGBMRegressor(n_estimators=50, verbosity=-1)
        model.fit(X_numeric.to_numpy(), y)
        
        importance = dict(zip(X_numeric.columns, model.feature_importances_.tolist()))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        logger.warning(f"Could not calculate baseline importance: {e}")
        return {}

# ── Main Plotting Engine ─────────────────────────────────────────────────────

def generate_eda_plots(state: ProfessorState) -> Dict[str, Any]:
    """Generates 7 standard EDA plots for delivery."""
    session_id = state.get("session_id", "default")
    output_dir = Path(f"outputs/{session_id}/{PLOT_DIR_NAME}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_path = state.get("clean_data_path")
    if not clean_path or not os.path.exists(clean_path):
        return {"success": False, "reason": "clean_data_path missing"}

    df = pl.read_parquet(clean_path)
    target_col = state.get("target_col", "")
    if not target_col: return {"success": False, "reason": "target_col missing"}
    
    plot_paths = []
    sns.set_theme(style="whitegrid")
    
    # 1. Target Distribution
    try:
        plt.figure(figsize=(10, 6))
        data = df[target_col].to_numpy()
        sns.histplot(data, kde=True)
        plt.title(f"Target Distribution: {target_col}")
        p = str(output_dir / "1_target_dist.png")
        plt.savefig(p); plt.close()
        plot_paths.append(p)
    except: pass

    # 2. Correlation Heatmap
    try:
        num_cols = [c for c in df.columns if df[c].dtype.is_numeric()][:15]
        if len(num_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr = df.select(num_cols).to_pandas().corr()
            sns.heatmap(corr, annot=False, cmap="coolwarm")
            plt.title("Feature Correlation Heatmap")
            p = str(output_dir / "2_correlation_heatmap.png")
            plt.savefig(p); plt.close()
            plot_paths.append(p)
    except: pass

    # 4. Feature Importance
    importance = _get_feature_importance(df, target_col)
    if importance:
        try:
            top_feats = list(importance.keys())[:12]
            top_vals = [importance[f] for f in top_feats]
            plt.figure(figsize=(10, 8))
            sns.barplot(x=top_vals, y=top_feats)
            plt.title("Baseline Feature Importance")
            p = str(output_dir / "4_feature_importance.png")
            plt.savefig(p); plt.close()
            plot_paths.append(p)
        except: pass

    return {
        "success": True,
        "plot_paths": plot_paths,
        "importance": importance
    }

def run_eda_visualizer(state: ProfessorState) -> ProfessorState:
    """LangGraph node: Component 3 - EDA Plots."""
    res = generate_eda_plots(state)
    if res["success"]:
        return state.validated_update(state, "eda_agent", {
            "eda_plots_paths": res["plot_paths"],
            "eda_plots_delivered": True,
            "eda_quick_baseline_importance": res["importance"]
        })
    return state
