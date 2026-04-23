# guards/leakage_precheck.py

import re
from typing import Dict, Optional, List

# ── Danger patterns — regex list with descriptions ──────────────────────────

DANGER_PATTERNS = [
    (r"\.fit_transform\(X\)", "fit_transform on variable named X (likely full dataset)"),
    (r"\.fit_transform\(df\)", "fit_transform on variable named df (likely full dataset)"),
    (r"\.fit_transform\(data\)", "fit_transform on variable named data (likely full dataset)"),
    (r"\.fit\(X\).*\.transform", "fit on X then transform — X may be full dataset"),
    (r"StandardScaler\(\)\.fit\((?!.*train)", "StandardScaler fit on non-train variable"),
    (r"MinMaxScaler\(\)\.fit\((?!.*train)", "MinMaxScaler fit on non-train variable"),
    (r"SimpleImputer\(\)\.fit\((?!.*train)", "SimpleImputer fit on non-train variable"),
    (r"TargetEncoder\(\)\.fit\((?!.*train)", "TargetEncoder fit outside CV fold"),
    (r"LabelEncoder\(\)\.fit\(.*(?:concat|vstack)", "LabelEncoder fit on combined train+test"),
    (r"\.fit\(.*(?:concat|vstack|rbind)", "fit on concatenated train+test data"),
]

# ── Safe pattern whitelist — lines matching these are NOT flagged ────────────

SAFE_PATTERNS = [
    r"Pipeline\(",               # sklearn Pipeline handles fit/transform correctly
    r"ColumnTransformer\(",      # same
    r"cross_val_score\(",        # sklearn cross-validation handles internally
    r"cross_val_predict\(",      # same
    r"make_pipeline\(",          # same
    r"\.fit\(X_train",           # Explicit train-only fitting
    r"\.fit\(train_",            # Explicit train prefix
    r"\.fit_transform\(X_train", # Explicit train-only fit_transform
]

# ── Core Logic ──────────────────────────────────────────────────────────────

def check_code_for_leakage(code: str) -> Dict:
    """
    Scan generated code for data leakage patterns BEFORE execution.
    """
    lines = code.split("\n")
    
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line or clean_line.startswith("#"):
            continue

        # 1. Check for danger patterns
        match_found = False
        description = ""
        for pattern, desc in DANGER_PATTERNS:
            if re.search(pattern, clean_line):
                match_found = True
                description = desc
                break
        
        if match_found:
            # 2. Check surrounding context (4 lines above, 3 lines below) for safe patterns
            start = max(0, i - 4)
            end = min(len(lines), i + 4)
            context = "\n".join(lines[start:end])
            
            is_safe = False
            for safe_pattern in SAFE_PATTERNS:
                if re.search(safe_pattern, context):
                    is_safe = True
                    break
            
            if not is_safe:
                return {
                    "leakage_detected": True,
                    "line": i + 1,
                    "code_line": clean_line,
                    "description": description,
                    "fix_suggestion": "Use .fit() on training data only (X_train), or use sklearn Pipeline which handles fold-correct fitting internally.",
                }

    return {
        "leakage_detected": False,
        "line": None,
        "code_line": None,
        "description": None,
        "fix_suggestion": None,
    }
