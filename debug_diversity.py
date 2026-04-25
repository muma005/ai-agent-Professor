import polars as pl
import numpy as np
from scipy.stats import spearmanr
import os

def calculate_diversity(path_a: str, path_b: str) -> float:
    if not (path_a and path_b and os.path.exists(path_a) and os.path.exists(path_b)):
        print("PATHS MISSING")
        return 0.0
    try:
        df_a = pl.read_csv(path_a)
        df_b = pl.read_csv(path_b)
        col_a = df_a.columns[1]
        col_b = df_b.columns[1]
        s_a = df_a[col_a].to_numpy()
        s_b = df_b[col_b].to_numpy()
        print(f"S_A: {s_a}")
        print(f"S_B: {s_b}")
        if len(s_a) < 2: return 1.0
        if np.std(s_a) == 0 or np.std(s_b) == 0: return 1.0
        corr, _ = spearmanr(s_a, s_b)
        return float(corr)
    except Exception as e:
        print(f"ERROR: {e}")
        return 0.0

# Mock files
pl.DataFrame({"id": [1, 2, 3], "target": [0.1, 0.9, 0.4]}).write_csv("sub_a.csv")
pl.DataFrame({"id": [1, 2, 3], "target": [0.15, 0.85, 0.45]}).write_csv("sub_b.csv")

res = calculate_diversity("sub_a.csv", "sub_b.csv")
print(f"RESULT: {res}")

os.remove("sub_a.csv")
os.remove("sub_b.csv")
