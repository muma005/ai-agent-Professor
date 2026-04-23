# guards/data_usage_checker.py

import os
from typing import List, Dict

DATA_EXTENSIONS = {".csv", ".parquet", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".feather"}
IGNORE_FILES = {"sample_submission.csv", "samplesubmission.csv", "sample_sub.csv"}

def check_data_usage(
    data_dir: str,
    generated_code: str,
) -> Dict:
    """
    Compare available data files against what the generated code references.
    """
    if not os.path.exists(data_dir):
        return {
            "total_data_files": 0,
            "used_files": [],
            "unused_files": [],
            "all_data_used": True,
        }

    data_files = []
    for filename in os.listdir(data_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in DATA_EXTENSIONS and filename.lower() not in IGNORE_FILES:
            data_files.append(filename)

    used_files = []
    unused_files = []
    
    for fname in data_files:
        stem = os.path.splitext(fname)[0]
        # Check exact filename, stem, or quoted variants
        is_used = (
            fname in generated_code or
            f'"{fname}"' in generated_code or
            f"'{fname}'" in generated_code or
            f'"{stem}"' in generated_code or
            f"'{stem}'" in generated_code
        )
        
        if is_used:
            used_files.append(fname)
        else:
            unused_files.append(fname)

    return {
        "total_data_files": len(data_files),
        "used_files": used_files,
        "unused_files": unused_files,
        "all_data_used": len(unused_files) == 0,
    }
