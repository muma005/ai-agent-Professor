def get_gate_config(n_rows: int) -> dict:
    """
    Adaptive gate thresholds based on dataset size.
    
    Small data: relaxed thresholds (low statistical power, high noise)
    Medium data: default thresholds
    Large data: strict thresholds (high power, genuine signal only)
    
    Returns dict with: wilcoxon_p, null_importance_percentile,
    null_importance_shuffles, cv_folds, regime
    """
    if n_rows < 1500:
        return {
            "wilcoxon_p": 0.10,
            "null_importance_percentile": 90,
            "null_importance_shuffles": 30,
            "cv_folds": 5,           # Capped — 1500/7 = 214 rows per fold is too thin
            "regime": "very_small",
        }
    elif n_rows < 5000:
        return {
            "wilcoxon_p": 0.10,
            "null_importance_percentile": 90,
            "null_importance_shuffles": 30,
            "cv_folds": 7,           # More folds = more paired observations for Wilcoxon
            "regime": "small",
        }
    elif n_rows <= 50000:
        return {
            "wilcoxon_p": 0.05,
            "null_importance_percentile": 95,
            "null_importance_shuffles": 50,
            "cv_folds": 5,
            "regime": "medium",
        }
    else:
        return {
            "wilcoxon_p": 0.02,
            "null_importance_percentile": 97,
            "null_importance_shuffles": 50,
            "cv_folds": 5,
            "regime": "large",
        }
