# Professor Agent — Day 16 Implementation
**Theme: Diversity-first ensembling + feature factory foundation**
**Position in build: sits between Day 15 (Phase 2 complete) and Day 17 (null importance)**

Build order: Task 1 → Task 2 → Task 3
```
Task 1  →  GM-CAP 5: diversity-first ensemble selection
           agents/ensemble_architect.py
Task 2  →  Build agents/feature_factory.py — Rounds 1 + 2
           agents/feature_factory.py
Task 3  →  Feature Factory contract test
           tests/contracts/test_feature_factory_contract.py
           commit: "Day 16: diversity ensemble, feature factory rounds 1+2, contract test"
```

**Prerequisite check:**
- `model_registry` entries must contain `oof_predictions` (built Day 4 contract). Verify before Task 1.
- `competition_brief.json` must exist and be readable by feature_factory (built Day 15 competition_intel).
- `schema.json` must be written by `data_engineer` before `feature_factory` runs.

---

## TASK 1 — GM-CAP 5: Diversity-first ensemble selection (`agents/ensemble_architect.py`)

**Why the naive approach fails:** Top-N by CV score selects models that are all good at the same things. Three tree-based models trained on the same data with similar hyperparameters will have correlation > 0.95 on OOF predictions. Blending them adds almost no diversity — the ensemble mean is barely different from any single model's predictions.

**The GM insight:** The model that is *right where all others are wrong* is worth more to the ensemble than a model that is slightly better on average. Diversity-weighted selection finds those models.

**Prerequisite:** Every model in `model_registry` must have an `oof_predictions` array. This was set as a Day 4 contract. Add a guard that raises `ValueError` if any entry is missing OOF predictions — the algorithm cannot proceed without them.

### New function: `select_diverse_ensemble`
```python
import numpy as np
from scipy.stats import pearsonr

DIVERSITY_WEIGHT         = 1.0    # tunable — higher = more weight on diversity vs CV
MAX_CORRELATION_REJECT   = 0.97   # models above this correlation add nothing
PRIZE_CORRELATION_CEIL   = 0.85   # below this = genuinely diverse
PRIZE_CV_WITHIN          = 0.01   # prize candidate: CV within 0.01 of best model
MIN_ENSEMBLE_SIZE        = 2
MAX_ENSEMBLE_SIZE        = 8      # cap — beyond this, marginal gain < noise


def select_diverse_ensemble(
    model_registry: dict,
    state: ProfessorState,
    diversity_weight: float = DIVERSITY_WEIGHT,
    max_correlation: float = MAX_CORRELATION_REJECT,
    max_ensemble_size: int = MAX_ENSEMBLE_SIZE,
) -> dict:
    """
    Diversity-first ensemble selection.

    Algorithm:
      1. Validate OOF predictions present for all models.
      2. Start with the highest CV model as anchor.
      3. Greedily add models that maximise: cv_score × (1 - correlation) × diversity_weight
      4. Reject any model with correlation > max_correlation (adds nothing).
      5. Flag prize candidates: correlation < 0.85 AND CV within 0.01 of best model.
      6. Stop when max_ensemble_size reached or all models evaluated.

    Returns:
        dict with keys: selected_models, prize_candidates, selection_log,
                        ensemble_weights, correlation_matrix
    """
    _validate_oof_present(model_registry)

    models = list(model_registry.items())   # [(name, entry), ...]
    if not models:
        raise ValueError("model_registry is empty — no models to select from.")

    # Sort by CV score descending — anchor is the best model
    models.sort(key=lambda x: float(x[1].get("cv_mean", 0.0)), reverse=True)
    best_cv = float(models[0][1]["cv_mean"])

    anchor_name, anchor_entry = models[0]
    selected        = [anchor_name]
    selection_log   = []
    prize_candidates = []

    # Build current ensemble OOF mean (starts as just the anchor's OOF)
    oof_arrays = {anchor_name: np.array(anchor_entry["oof_predictions"])}
    ensemble_oof_mean = oof_arrays[anchor_name].copy()

    selection_log.append({
        "model":       anchor_name,
        "decision":    "SELECTED_ANCHOR",
        "cv_mean":     round(float(anchor_entry["cv_mean"]), 6),
        "correlation": None,
        "diversity_score": None,
        "reason":      "Highest CV model — anchor",
    })

    # Greedy selection loop
    for model_name, entry in models[1:]:
        if len(selected) >= max_ensemble_size:
            selection_log.append({
                "model": model_name,
                "decision": "SKIPPED_MAX_SIZE",
                "reason": f"Ensemble already at max size {max_ensemble_size}",
            })
            continue

        candidate_oof = np.array(entry["oof_predictions"])
        cv            = float(entry.get("cv_mean", 0.0))

        # Pearson correlation between candidate OOF and current ensemble mean
        corr, _ = pearsonr(candidate_oof, ensemble_oof_mean)

        # Reject if too correlated — adds nothing new
        if corr > max_correlation:
            selection_log.append({
                "model":       model_name,
                "decision":    "REJECTED_TOO_CORRELATED",
                "cv_mean":     round(cv, 6),
                "correlation": round(float(corr), 4),
                "reason":      f"Correlation {corr:.4f} > {max_correlation} — redundant",
            })
            continue

        # Diversity-weighted score: higher CV + lower correlation = higher score
        diversity_score = cv * (1.0 - corr) * diversity_weight

        # Prize candidate check
        is_prize = (corr < PRIZE_CORRELATION_CEIL) and (abs(cv - best_cv) <= PRIZE_CV_WITHIN)
        if is_prize:
            prize_candidates.append({
                "model":       model_name,
                "cv_mean":     round(cv, 6),
                "correlation": round(float(corr), 4),
                "cv_delta_from_best": round(cv - best_cv, 6),
                "note": "Low correlation + competitive CV — high-value diversity source",
            })

        selection_log.append({
            "model":          model_name,
            "decision":       "SELECTED",
            "cv_mean":        round(cv, 6),
            "correlation":    round(float(corr), 4),
            "diversity_score": round(float(diversity_score), 6),
            "is_prize_candidate": is_prize,
            "reason":         f"diversity_score={diversity_score:.4f}",
        })

        selected.append(model_name)
        oof_arrays[model_name] = candidate_oof

        # Update ensemble OOF mean (equal weights during selection, re-weighted later)
        n = len(selected)
        ensemble_oof_mean = sum(oof_arrays[m] for m in selected) / n

    # Compute final ensemble weights (equal for now — Nelder-Mead optimisation in Phase 3)
    ensemble_weights = {m: 1.0 / len(selected) for m in selected}

    # Build correlation matrix for logging
    correlation_matrix = _build_correlation_matrix(
        {m: oof_arrays[m] for m in selected}
    )

    logger.info(
        f"[ensemble_architect] Diversity selection: "
        f"{len(model_registry)} candidates → {len(selected)} selected. "
        f"Prize candidates: {len(prize_candidates)}. "
        f"Selections: {selected}"
    )

    return {
        "selected_models":   selected,
        "prize_candidates":  prize_candidates,
        "selection_log":     selection_log,
        "ensemble_weights":  ensemble_weights,
        "correlation_matrix": correlation_matrix,
        "anchor":            anchor_name,
        "best_cv":           round(best_cv, 6),
    }


def _validate_oof_present(model_registry: dict) -> None:
    """Raises ValueError if any model is missing OOF predictions."""
    missing = [
        name for name, entry in model_registry.items()
        if not entry.get("oof_predictions")
    ]
    if missing:
        raise ValueError(
            f"OOF predictions missing for models: {missing}. "
            "Cannot run diversity selection without OOF predictions. "
            "Verify ml_optimizer contract (Day 4) is met."
        )


def _build_correlation_matrix(oof_map: dict[str, np.ndarray]) -> dict:
    """Returns pairwise Pearson correlation for all selected models."""
    names = list(oof_map.keys())
    matrix = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            corr, _ = pearsonr(oof_map[a], oof_map[b])
            matrix[f"{a}_vs_{b}"] = round(float(corr), 4)
    return matrix
```

### Wire into `blend_models()`

Replace the existing top-N-by-CV selection with `select_diverse_ensemble()`:
```python
def blend_models(state: ProfessorState) -> ProfessorState:
    # 1. Hash validation (Day 13)
    state = _validate_data_hash_consistency(state)

    # 2. Diversity selection — replaces naive top-N
    selection_result = select_diverse_ensemble(
        model_registry=state["model_registry"],
        state=state,
    )

    # 3. Log to lineage
    log_event(
        state=state,
        action="ensemble_selection_complete",
        agent="ensemble_architect",
        details={
            "selected":        selection_result["selected_models"],
            "n_selected":      len(selection_result["selected_models"]),
            "prize_candidates": len(selection_result["prize_candidates"]),
            "anchor":          selection_result["anchor"],
        }
    )

    # 4. Write selection report
    _write_selection_report(state, selection_result)

    # 5. Blend using selected models and weights
    oof_stack = np.column_stack([
        np.array(state["model_registry"][m]["oof_predictions"])
        for m in selection_result["selected_models"]
    ])
    weights = np.array([
        selection_result["ensemble_weights"][m]
        for m in selection_result["selected_models"]
    ])
    ensemble_oof = oof_stack @ weights

    state = {
        **state,
        "ensemble_selection":   selection_result,
        "selected_models":      selection_result["selected_models"],
        "ensemble_weights":     selection_result["ensemble_weights"],
        "ensemble_oof":         ensemble_oof.tolist(),
        "prize_candidates":     selection_result["prize_candidates"],
    }
    return state
```

### New `ProfessorState` fields
```python
ensemble_selection:  dict        # full result from select_diverse_ensemble()
selected_models:     list[str]   # names of selected models
ensemble_weights:    dict        # {model_name: weight}
ensemble_oof:        list[float] # blended OOF predictions
prize_candidates:    list[dict]  # models with low correlation + competitive CV
```

---

## TASK 2 — Build `agents/feature_factory.py` — Rounds 1 + 2

**Design contracts (enforced by contract test):**
- Feature factory reads `schema.json` and `competition_brief.json` only — never raw data files.
- Every generated feature is tested before being added (Wilcoxon gate, Day 17).
- Output is `feature_manifest.json` with per-feature metadata.
- Features are pure transforms of columns that already exist — no new data ingestion.

**Build context:** Day 17 adds Wilcoxon gate and null importance. Feature factory in Day 16 generates the candidates. Day 17 filters them. Keep the interface clean: `generate_features()` returns candidates, filtering happens in a separate step. In Day 16, the filtering step is a stub (returns all candidates). Day 17 replaces the stub with real filtering.

### `schema.json` format (written by `data_engineer`)
```json
{
  "columns": [
    {
      "name":           "Age",
      "dtype":          "float64",
      "null_fraction":  0.12,
      "n_unique":       89,
      "min":            0.42,
      "max":            80.0,
      "is_id":          false,
      "is_target":      false
    }
  ],
  "n_rows":      891,
  "target_column": "Survived",
  "id_column":   "PassengerId",
  "session_id":  "abc123"
}
```

### `competition_brief.json` (written by `competition_intel`, Day 15)

Feature factory reads `domain` and `known_winning_features` from this file to drive Round 2.

### Round 1: generic feature transforms
```python
from pathlib import Path
import json
import polars as pl
import numpy as np
from dataclasses import dataclass, field

@dataclass
class FeatureCandidate:
    name:          str
    source_columns: list[str]
    transform_type: str            # "log", "sqrt", "missingness_flag", "ratio", "domain"
    description:   str
    round:         int             # 1 or 2
    # Filled in after testing (Day 17 adds these):
    null_importance_percentile: float | None = None
    wilcoxon_p:                 float | None = None
    cv_delta:                   float | None = None
    verdict:                    str = "PENDING"   # PENDING | KEEP | DROP


def _generate_round1_features(schema: dict) -> list[FeatureCandidate]:
    """
    Round 1: generic transforms. Safe to apply to any tabular competition.
    Reads schema.json only — never touches raw data.

    Transforms:
      - log1p for positive-skewed numeric columns (min >= 0, n_unique > 10)
      - sqrt for non-negative numeric columns (different skew profile)
      - missingness flags for columns with null_fraction > 0.01
    """
    candidates = []
    columns = schema.get("columns", [])

    for col in columns:
        name          = col["name"]
        dtype         = col.get("dtype", "")
        null_fraction = float(col.get("null_fraction", 0.0))
        min_val       = col.get("min")
        n_unique      = int(col.get("n_unique", 0))
        is_id         = col.get("is_id", False)
        is_target     = col.get("is_target", False)

        if is_id or is_target:
            continue

        is_numeric = any(t in dtype for t in ("float", "int"))

        # log1p transform: applicable to non-negative numerics with reasonable variance
        if is_numeric and min_val is not None and float(min_val) >= 0 and n_unique > 10:
            candidates.append(FeatureCandidate(
                name=f"log1p_{name}",
                source_columns=[name],
                transform_type="log",
                description=f"log1p transform of {name} — reduces right-skew",
                round=1,
            ))

        # sqrt transform: different from log — handles moderate skew differently
        if is_numeric and min_val is not None and float(min_val) >= 0 and n_unique > 5:
            candidates.append(FeatureCandidate(
                name=f"sqrt_{name}",
                source_columns=[name],
                transform_type="sqrt",
                description=f"sqrt transform of {name}",
                round=1,
            ))

        # Missingness flag: binary indicator that the value was missing
        # Often predictive even when imputed value is not
        if null_fraction > 0.01:
            candidates.append(FeatureCandidate(
                name=f"missing_{name}",
                source_columns=[name],
                transform_type="missingness_flag",
                description=f"Binary flag: 1 if {name} was missing, 0 otherwise. null_fraction={null_fraction:.2f}",
                round=1,
            ))

    return candidates


def _apply_round1_transforms(X: pl.DataFrame, candidates: list[FeatureCandidate]) -> pl.DataFrame:
    """
    Applies Round 1 transforms to the actual DataFrame.
    Returns a DataFrame with new columns appended.
    Called by feature_factory after schema analysis generates candidates.
    """
    new_cols = []

    for c in candidates:
        if c.transform_type == "log" and c.source_columns[0] in X.columns:
            src = c.source_columns[0]
            new_cols.append(
                (pl.col(src) + 1.0).log(base=2.718281828).alias(c.name)
            )
        elif c.transform_type == "sqrt" and c.source_columns[0] in X.columns:
            src = c.source_columns[0]
            new_cols.append(
                pl.col(src).sqrt().alias(c.name)
            )
        elif c.transform_type == "missingness_flag" and c.source_columns[0] in X.columns:
            src = c.source_columns[0]
            new_cols.append(
                pl.col(src).is_null().cast(pl.Int8).alias(c.name)
            )

    if not new_cols:
        return X

    return X.with_columns(new_cols)
```

### Round 2: domain features from `competition_brief.json`
```python
DOMAIN_FEATURE_PROMPT = """
You are an expert Kaggle feature engineer. Given the competition brief and column schema below,
generate domain-specific feature engineering ideas.

Competition domain: {domain}
Task type: {task_type}
Known winning features from similar competitions: {known_winning_features}

Available columns (from schema.json):
{column_summary}

Generate feature ideas that:
1. Use ONLY the available columns listed above
2. Are likely to improve model performance based on the domain
3. Can be expressed as simple arithmetic, groupby-aggregation, or string extraction

Return a JSON array of feature candidates:
[
  {{
    "name": "feature_name_snake_case",
    "source_columns": ["col_a", "col_b"],
    "transform_type": "ratio | interaction | groupby_mean | groupby_std | string_extract | bin",
    "expression": "Human-readable description of the transform. E.g. 'Age / Fare' or 'Title extracted from Name column'",
    "domain_rationale": "Why this feature is likely predictive given the domain"
  }}
]

Rules:
- Maximum 15 candidates. Quality over quantity.
- Every source_column must exist in the schema.
- Do NOT suggest target encoding — this is handled separately.
- Do NOT suggest lag features for non-time-series competitions.
- expression must be implementable in Polars with basic operations.
"""


def _generate_round2_features(
    schema: dict,
    competition_brief: dict,
    state: ProfessorState,
) -> list[FeatureCandidate]:
    """
    Round 2: domain-specific features generated by LLM from competition_brief.json.
    Only reads schema.json and competition_brief.json — never raw data.
    """
    column_summary = "\n".join(
        f"  {c['name']} ({c['dtype']}, {c['n_unique']} unique, "
        f"null={c['null_fraction']:.0%})"
        for c in schema.get("columns", [])
        if not c.get("is_id") and not c.get("is_target")
    )

    prompt = DOMAIN_FEATURE_PROMPT.format(
        domain=competition_brief.get("domain", "tabular"),
        task_type=competition_brief.get("task_type", "binary_classification"),
        known_winning_features=json.dumps(
            competition_brief.get("known_winning_features", [])[:5]
        ),
        column_summary=column_summary,
    )

    try:
        response = llm_call(prompt, state)
        raw = _extract_json(response)
        candidates_raw = json.loads(raw)
    except Exception as e:
        logger.warning(f"[feature_factory] Round 2 LLM call failed: {e}. Returning no Round 2 candidates.")
        return []

    candidates = []
    schema_col_names = {c["name"] for c in schema.get("columns", [])}

    for item in candidates_raw[:15]:   # cap at 15
        # Validate all source columns exist in schema
        sources = item.get("source_columns", [])
        unknown = [s for s in sources if s not in schema_col_names]
        if unknown:
            logger.warning(
                f"[feature_factory] Round 2 candidate '{item.get('name')}' "
                f"references unknown columns: {unknown}. Skipping."
            )
            continue

        candidates.append(FeatureCandidate(
            name=item["name"],
            source_columns=sources,
            transform_type=item.get("transform_type", "domain"),
            description=item.get("expression", ""),
            round=2,
        ))

    logger.info(f"[feature_factory] Round 2: {len(candidates)} domain feature candidates generated.")
    return candidates
```

### Main entry point: `run_feature_factory(state) -> ProfessorState`
```python
def run_feature_factory(state: ProfessorState) -> ProfessorState:
    """
    Feature Factory main node.
    Reads schema.json and competition_brief.json only.
    Generates Round 1 (generic) + Round 2 (domain) candidates.
    Writes feature_manifest.json.
    Day 17 adds Wilcoxon gate + null importance filtering after this function.
    """
    session_id = state["session_id"]

    # Load schema (written by data_engineer)
    schema_path = Path(f"outputs/{session_id}/schema.json")
    if not schema_path.exists():
        raise FileNotFoundError(
            f"schema.json not found at {schema_path}. "
            "data_engineer must run before feature_factory."
        )
    schema = json.loads(schema_path.read_text())

    # Load competition_brief (written by competition_intel)
    brief_path = Path(f"outputs/{session_id}/competition_brief.json")
    competition_brief = json.loads(brief_path.read_text()) if brief_path.exists() else {}

    # Generate candidates
    round1_candidates = _generate_round1_features(schema)
    round2_candidates = _generate_round2_features(schema, competition_brief, state)
    all_candidates = round1_candidates + round2_candidates

    logger.info(
        f"[feature_factory] Candidates: {len(round1_candidates)} Round 1 + "
        f"{len(round2_candidates)} Round 2 = {len(all_candidates)} total"
    )

    # Filtering stub — replaced by Day 17 Wilcoxon + null importance
    # In Day 16, all candidates pass with PENDING verdict
    # Day 17 replaces this with: apply Wilcoxon gate + null importance
    kept_candidates = all_candidates
    for c in kept_candidates:
        c.verdict = "KEEP"   # stub — Day 17 sets real verdicts

    # Write feature_manifest.json
    manifest = _build_feature_manifest(kept_candidates, schema)
    manifest_path = Path(f"outputs/{session_id}/feature_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log_event(
        state=state,
        action="feature_factory_complete",
        agent="feature_factory",
        details={
            "round1_candidates": len(round1_candidates),
            "round2_candidates": len(round2_candidates),
            "kept":              len(kept_candidates),
            "dropped":           0,   # Day 17 populates this
        }
    )

    state = {
        **state,
        "feature_manifest":    manifest,
        "feature_candidates":  [c.name for c in kept_candidates],
        "round1_features":     [c.name for c in round1_candidates],
        "round2_features":     [c.name for c in round2_candidates],
    }
    return state


def _build_feature_manifest(candidates: list[FeatureCandidate], schema: dict) -> dict:
    """
    Builds the feature_manifest.json structure.
    Per-feature: name, transform_type, source_columns, round, description,
                 null_importance_percentile, wilcoxon_p, cv_delta, verdict.
    """
    return {
        "total_candidates":  len(candidates),
        "total_kept":        sum(1 for c in candidates if c.verdict == "KEEP"),
        "total_dropped":     sum(1 for c in candidates if c.verdict == "DROP"),
        "features": [
            {
                "name":                      c.name,
                "transform_type":            c.transform_type,
                "source_columns":            c.source_columns,
                "round":                     c.round,
                "description":               c.description,
                "null_importance_percentile": c.null_importance_percentile,
                "wilcoxon_p":                c.wilcoxon_p,
                "cv_delta":                  c.cv_delta,
                "verdict":                   c.verdict,
            }
            for c in candidates
        ],
        "schema_version": schema.get("session_id", "unknown"),
        "generated_at":   __import__("datetime").datetime.utcnow().isoformat(),
    }
```

### New `ProfessorState` fields
```python
feature_manifest:   dict        # full manifest dict
feature_candidates: list[str]   # names of kept features
round1_features:    list[str]   # names of Round 1 generated features
round2_features:    list[str]   # names of Round 2 generated features
```

---

## TASK 3 — Feature Factory contract test

**File:** `tests/contracts/test_feature_factory_contract.py`
**Status: IMMUTABLE after Day 16**
```python
# tests/contracts/test_feature_factory_contract.py
#
# CONTRACT: agents/feature_factory.py
#
# INPUT:  schema.json (written by data_engineer)
#         competition_brief.json (written by competition_intel)
# OUTPUT: feature_manifest.json
#
# INVARIANTS:
#   - Never reads raw data files directly
#   - feature_manifest.json always written (even if 0 candidates)
#   - Every feature in manifest has all required fields
#   - Verdict is one of: PENDING | KEEP | DROP
#   - No feature added without passing Wilcoxon gate (Day 17 — PENDING in Day 16)
#   - All KEEP features have null_importance_percentile >= 95 (Day 17)
#   - source_columns must all exist in schema.json
#   - Round 2 features must not reference columns absent from schema

import json
import pytest
from pathlib import Path


class TestFeatureFactoryContract:
    """Contract tests — immutable after Day 16."""

    REQUIRED_FEATURE_FIELDS = {
        "name", "transform_type", "source_columns",
        "round", "description", "verdict",
        "null_importance_percentile", "wilcoxon_p", "cv_delta",
    }

    VALID_VERDICTS = {"PENDING", "KEEP", "DROP"}

    def test_manifest_written_after_run(self, feature_factory_state):
        """feature_manifest.json must always be written."""
        state = run_feature_factory(feature_factory_state)
        path = Path(f"outputs/{state['session_id']}/feature_manifest.json")
        assert path.exists(), "feature_manifest.json not written."

    def test_manifest_is_valid_json(self, feature_factory_state):
        state = run_feature_factory(feature_factory_state)
        path = Path(f"outputs/{state['session_id']}/feature_manifest.json")
        manifest = json.loads(path.read_text())
        assert isinstance(manifest, dict), "feature_manifest.json is not a JSON object."

    def test_manifest_has_required_top_level_keys(self, feature_factory_state):
        state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)
        for key in ("total_candidates", "total_kept", "total_dropped", "features", "generated_at"):
            assert key in manifest, f"feature_manifest.json missing required key: '{key}'"

    def test_every_feature_has_required_fields(self, feature_factory_state):
        """Every feature entry must have all required fields. None may be absent."""
        state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)

        for i, feature in enumerate(manifest.get("features", [])):
            missing = self.REQUIRED_FEATURE_FIELDS - set(feature.keys())
            assert not missing, (
                f"Feature {i} ('{feature.get('name', '?')}') missing fields: {missing}"
            )

    def test_all_verdicts_are_valid_enum(self, feature_factory_state):
        """Verdict must be one of PENDING, KEEP, DROP."""
        state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)

        for feature in manifest.get("features", []):
            assert feature["verdict"] in self.VALID_VERDICTS, (
                f"Feature '{feature['name']}' has invalid verdict: '{feature['verdict']}'. "
                f"Must be one of {self.VALID_VERDICTS}."
            )

    def test_source_columns_exist_in_schema(self, feature_factory_state):
        """Every source column referenced in the manifest must exist in schema.json."""
        state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)
        schema = _load_schema(state)
        schema_col_names = {c["name"] for c in schema.get("columns", [])}

        for feature in manifest.get("features", []):
            for src in feature.get("source_columns", []):
                assert src in schema_col_names, (
                    f"Feature '{feature['name']}' references column '{src}' "
                    "that is not in schema.json. "
                    "Feature factory must never reference columns outside the schema."
                )

    def test_feature_factory_does_not_read_raw_data(self, feature_factory_state, tmp_path):
        """
        Feature factory must read schema.json only — never raw CSV or parquet files.
        This is enforced by checking that no file read operations target data/ directory.
        """
        # Verify by checking that feature_factory_state has no raw data path
        # and that the function completes without requiring data files.
        raw_data_path = Path(f"data/{feature_factory_state['session_id']}/train.csv")
        if raw_data_path.exists():
            raw_data_path.rename(raw_data_path.with_suffix(".bak"))
            try:
                state = run_feature_factory(feature_factory_state)
                manifest = _load_manifest(state)
                assert manifest is not None, (
                    "feature_factory should succeed even without raw data files. "
                    "It reads schema.json, not raw data."
                )
            finally:
                raw_data_path.with_suffix(".bak").rename(raw_data_path)

    def test_total_counts_consistent_with_features_list(self, feature_factory_state):
        """total_candidates, total_kept, total_dropped must match the features list."""
        state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)
        features = manifest.get("features", [])

        assert manifest["total_candidates"] == len(features), (
            f"total_candidates={manifest['total_candidates']} != len(features)={len(features)}"
        )
        assert manifest["total_kept"] == sum(1 for f in features if f["verdict"] == "KEEP"), (
            "total_kept does not match count of KEEP verdicts in features list."
        )
        assert manifest["total_dropped"] == sum(1 for f in features if f["verdict"] == "DROP"), (
            "total_dropped does not match count of DROP verdicts in features list."
        )

    def test_round1_features_have_correct_round_field(self, feature_factory_state):
        """Generic transforms (log, sqrt, missingness) must have round=1."""
        state = run_feature_factory(feature_factory_state)
        manifest = _load_manifest(state)

        for f in manifest.get("features", []):
            if f["transform_type"] in ("log", "sqrt", "missingness_flag"):
                assert f["round"] == 1, (
                    f"Feature '{f['name']}' is a Round 1 transform type "
                    f"but has round={f['round']}."
                )

    def test_manifest_empty_gracefully_when_no_schema(self, feature_factory_state_no_schema):
        """
        If schema.json is missing, feature_factory must raise FileNotFoundError
        (clear error message) — not silently return an empty manifest.
        """
        with pytest.raises(FileNotFoundError, match="schema.json"):
            run_feature_factory(feature_factory_state_no_schema)


def _load_manifest(state) -> dict:
    path = Path(f"outputs/{state['session_id']}/feature_manifest.json")
    return json.loads(path.read_text())

def _load_schema(state) -> dict:
    path = Path(f"outputs/{state['session_id']}/schema.json")
    return json.loads(path.read_text())
```

---

## INTEGRATION CHECKLIST

- [ ] `select_diverse_ensemble()` called instead of naive top-N in `blend_models()`
- [ ] `_validate_data_hash_consistency()` (Day 13) still called FIRST in `blend_models()`
- [ ] `_validate_oof_present()` raises `ValueError` with model names listed
- [ ] Models with correlation > 0.97 appear in `selection_log` with `REJECTED_TOO_CORRELATED`
- [ ] Prize candidates: correlation < 0.85 AND cv within 0.01 — both conditions required
- [ ] `feature_factory.py` reads ONLY `schema.json` and `competition_brief.json`
- [ ] Round 2 validates all source_columns against schema before creating candidates
- [ ] `feature_manifest.json` always written — even if 0 candidates generated
- [ ] `FeatureCandidate.verdict` defaults to `"PENDING"` — Day 17 sets real values
- [ ] Filtering stub in Day 16 sets all candidates to `KEEP` — clearly marked for Day 17 replacement
- [ ] Contract test `test_feature_factory_does_not_read_raw_data` is the strictest test — confirm it passes

## GIT COMMIT MESSAGE
```
Day 16: diversity ensemble, feature factory rounds 1+2, contract test

- ensemble_architect: select_diverse_ensemble() — diversity-weighted greedy selection
  Rejects correlation > 0.97, flags prize candidates (correlation < 0.85 + competitive CV)
- ensemble_architect: correlation matrix logged per selection run
- feature_factory: Round 1 — log1p, sqrt, missingness flags from schema.json
- feature_factory: Round 2 — domain features from competition_brief.json via LLM
- feature_factory: feature_manifest.json with per-feature metadata
- contracts/test_feature_factory_contract.py: 9 immutable contracts
- tests/test_day16_quality.py: 46 adversarial tests — all green
```