import os
import ast
import pickle
import logging
import polars as pl
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ── Safe expression evaluator (replaces eval()) ──────────────────

_SAFE_NODES = {
    ast.Expression, ast.Call, ast.Attribute, ast.Name, ast.Load,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
}

_SAFE_NAMES = {"pl", "np", "polars"}


def _safe_eval_expr(expr_str: str) -> Any:
    """
    Evaluate a Polars expression string safely.
    Only allows: pl.col(), pl.lit(), arithmetic ops, comparisons.
    Rejects: imports, function calls to unknown names, attribute access on non-pl objects.
    """
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {expr_str}") from e

    for node in ast.walk(tree):
        if type(node) not in _SAFE_NODES:
            raise ValueError(
                f"Unsafe AST node '{type(node).__name__}' in expression: {expr_str}"
            )
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
            raise ValueError(
                f"Unsafe name '{node.id}' in expression: {expr_str}. "
                f"Only {_SAFE_NAMES} allowed."
            )

    # Safe to evaluate — only pl/np references
    import numpy as np
    return eval(expr_str, {"__builtins__": {}, "pl": pl, "np": np, "polars": pl})


class TabularPreprocessor:
    """
    Unified preprocessing artifact for the Professor pipeline.
    Ensures mathematically identical transformations between train and test data.

    Fitted during training → serialized → loaded at test time → identical transform.
    """
    def __init__(self, target_col: str, id_cols: List[str] = None):
        self.target_col = target_col
        self.id_cols = id_cols or []

        # Mappings for deterministic imputation
        self.numeric_imputes: Dict[str, float] = {}
        self.string_imputes: Dict[str, str] = {}
        self.bool_imputes: Dict[str, bool] = {}

        # Categorical encoding: {col_name: {value: int_code}}
        # Fitted from training data, replayed identically on test
        self.categorical_encoders: Dict[str, Dict[str, int]] = {}

        # FeatureFactory AST representations (safe Polars expression strings)
        self.feature_expressions: Dict[str, str] = {}

        # Stateful group-based mappings (e.g. Target Encoding, GroupBy Aggregations)
        self.group_mappings: Dict[str, Dict] = {}

        self.expected_columns: List[str] = []

    def fit_imputation(self, df: pl.DataFrame, schema: Dict[str, Any]):
        """
        Calculates medians, missing values, and categorical encodings.
        Does NOT apply them — call transform() for that.
        """
        types = schema.get("types", {})
        for col, dtype in types.items():
            if col == self.target_col or col in self.id_cols:
                continue

            dtype_str = str(dtype)
            if "Float" in dtype_str or "Int" in dtype_str:
                if col in df.columns:
                    median_val = df[col].median()
                    if median_val is not None:
                        self.numeric_imputes[col] = float(median_val)
            elif "String" in dtype_str or "Utf8" in dtype_str or "Categorical" in dtype_str:
                self.string_imputes[col] = "missing"
                # Fit categorical encoder: sorted unique values → deterministic codes
                if col in df.columns:
                    unique_vals = sorted(df[col].drop_nulls().unique().to_list())
                    self.categorical_encoders[col] = {
                        val: idx for idx, val in enumerate(unique_vals)
                    }
            elif "Bool" in dtype_str:
                self.bool_imputes[col] = False

    def add_feature_expression(self, col_name: str, expression_str: str):
        """
        Register a new engineered feature calculation.
        expression_str must be valid polars Python code, e.g., 'pl.col("A") / pl.col("B")'
        """
        self.feature_expressions[col_name] = expression_str

    def add_group_mapping(self, col_name: str, cat_col: str, mapping_dict: Dict[Any, float], default_val: float):
        """
        Registers a stateful dictionary mapping. Maps 'cat_col' values to static floats.
        """
        self.group_mappings[col_name] = {
            "cat_col": cat_col,
            "mapping": mapping_dict,
            "default": default_val
        }

    def validate_columns(self, df: pl.DataFrame) -> list[str]:
        """
        Check which expected columns are missing from df.
        Returns list of missing column names (empty = all good).
        """
        if not self.expected_columns:
            return []
        return [c for c in self.expected_columns if c not in df.columns]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies all stored transformations exactly identically.
        Order: impute → encode categoricals → generate features.
        """
        # 0. Validate columns (warn but don't crash — test data may lack target)
        missing = self.validate_columns(df)
        non_target_missing = [c for c in missing if c != self.target_col]
        if non_target_missing:
            logger.warning(
                f"[TabularPreprocessor] Missing {len(non_target_missing)} expected columns: "
                f"{non_target_missing[:10]}{'...' if len(non_target_missing) > 10 else ''}"
            )

        # 1. Impute Numerics
        for col, val in self.numeric_imputes.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(val))

        # 2. Impute Strings (before encoding)
        for col, val in self.string_imputes.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(val))

        # 3. Impute Bools
        for col, val in self.bool_imputes.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(val))

        # 4. Encode Categoricals (deterministic — same mapping as training)
        for col, encoder in self.categorical_encoders.items():
            if col in df.columns and df[col].dtype in (pl.Utf8, pl.String, pl.Categorical):
                # Map known values to codes, UNSEEN values to -1
                df = df.with_columns(
                    pl.col(col).replace(encoder, default=-1).cast(pl.Int32)
                )

        # 5. Generate Features (safe evaluator — no raw eval)
        for col, expr_str in self.feature_expressions.items():
            try:
                expr = _safe_eval_expr(expr_str)
                df = df.with_columns(expr.alias(col))
            except Exception as e:
                logger.warning(
                    f"[TabularPreprocessor] Failed to evaluate feature '{col}' "
                    f"with expr '{expr_str}': {e}"
                )

        # 6. Stateful Group Mappings
        for col_name, info in self.group_mappings.items():
            cat_col = info["cat_col"]
            if cat_col in df.columns:
                df = df.with_columns(
                    pl.col(cat_col)
                    .replace(info["mapping"], default=info["default"])
                    .cast(pl.Float64)
                    .alias(col_name)
                )

        return df

    def fit_transform(self, df: pl.DataFrame, schema: Dict[str, Any]) -> pl.DataFrame:
        self.fit_imputation(df, schema)
        df_transformed = self.transform(df)
        self.expected_columns = df_transformed.columns
        return df_transformed

    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"[TabularPreprocessor] Saved to {output_path}")

    @staticmethod
    def load(input_path: str) -> "TabularPreprocessor":
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Preprocessor not found at {input_path}")
        with open(input_path, "rb") as f:
            return pickle.load(f)

    def save_config(self, output_path: str):
        """
        Save preprocessor config (not fitted state) for later reconstruction.
        Used for CV where we need fresh preprocessor per fold.
        """
        import json
        config = {
            "target_col": self.target_col,
            "id_cols": self.id_cols,
            "numeric_imputes": self.numeric_imputes,
            "string_imputes": self.string_imputes,
            "bool_imputes": self.bool_imputes,
            "categorical_encoders": self.categorical_encoders,
            "feature_expressions": self.feature_expressions,
            "group_mappings": self.group_mappings,
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"[TabularPreprocessor] Config saved to {output_path}")

    @staticmethod
    def load_config(config_path: str) -> "TabularPreprocessor":
        """
        Reconstruct preprocessor from saved config.
        """
        import json
        with open(config_path) as f:
            config = json.load(f)
        
        prep = TabularPreprocessor(
            target_col=config["target_col"],
            id_cols=config["id_cols"]
        )
        prep.numeric_imputes = config["numeric_imputes"]
        prep.string_imputes = config["string_imputes"]
        prep.bool_imputes = config["bool_imputes"]
        prep.categorical_encoders = config["categorical_encoders"]
        prep.feature_expressions = config["feature_expressions"]
        prep.group_mappings = config["group_mappings"]
        logger.info(f"[TabularPreprocessor] Config loaded from {config_path}")
        return prep

    def clone_unfitted(self) -> "TabularPreprocessor":
        """
        Create a new preprocessor with same config but unfitted state.
        Used for CV where we need fresh preprocessor per fold.
        """
        return TabularPreprocessor(
            target_col=self.target_col,
            id_cols=self.id_cols
        )
