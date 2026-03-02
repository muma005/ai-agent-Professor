# tools/e2b_sandbox.py
# Day 1: scaffold with preamble. Full implementation Day 2.

from RestrictedPython import compile_restricted, safe_globals

SANDBOX_PREAMBLE = """\
import polars as pl
import polars.selectors as cs
import numpy as np
# ── Library standard: Polars not Pandas ──────────────────────────
# CORRECT:   pl.read_csv()  df.write_parquet()  df.fill_null()
# INCORRECT: pd.read_csv()  df.to_parquet()     df.fillna()
# If pandas required: convert back with pl.from_pandas(df)
# ─────────────────────────────────────────────────────────────────
"""

def execute_code(code: str, session_id: str) -> dict:
    """
    Execute Python code in RestrictedPython sandbox.
    Full implementation Day 2.
    For today: just confirm the scaffold runs.
    """
    full_code = SANDBOX_PREAMBLE + code

    try:
        compiled = compile_restricted(full_code, "<sandbox>", "exec")
        glb = dict(safe_globals)
        glb["__builtins__"] = safe_globals["__builtins__"]
        exec(compiled, glb)
        return {"status": "success", "output": glb}
    except Exception as e:
        return {"status": "error", "error": str(e)}
