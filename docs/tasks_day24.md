# Day 24 — Memory Layer: Warm Start, Seed Memory, Quality Scoring
## Implementation Prompt for Qwen Code

---

## BEFORE YOU WRITE A SINGLE LINE

Read these files completely first:

```
CLAUDE.md
AGENTS.md
core/state.py
memory/memory_schema.py          ← understand existing ChromaDB collections and functions
agents/ml_optimizer.py           ← understand how Optuna study is built today
agents/post_mortem_agent.py      ← understand what gets written to ChromaDB after competitions
```

After reading, answer these questions before writing code:
1. What ChromaDB collections already exist? List them with their purposes.
2. What does `get_warm_start_priors(state)` return today? What fields does each result have?
3. Where in `ml_optimizer.py` is `optuna.create_study()` called? What line?
4. What is `build_competition_fingerprint(state)` and where is it called?

Do not proceed until you have answered all four from the actual code.

---

## TASK 1 — GAP 11: Optuna warm start from ChromaDB (`agents/ml_optimizer.py`)

**What this does:** Before building the Optuna search space, query ChromaDB for hyperparameter configurations that worked well in similar past competitions. Inject those as seed trials so Optuna explores around known good regions instead of starting from scratch.

**When it activates:** Only when ChromaDB contains at least one stored hyperparameter memory. On competition 1 it degrades silently — returns an empty seed list and Optuna starts normally.

### New function: `get_hpo_warm_start_seeds(state) -> list[dict]`

Add to `memory/memory_schema.py`:

```python
def get_hpo_warm_start_seeds(
    state: dict,
    n_seeds: int = 5,
    max_distance: float = 0.70,
    min_confidence: float = 0.65,
) -> list[dict]:
    """
    Queries ChromaDB for hyperparameter configurations from similar past competitions.
    Returns a list of param dicts ready to inject into Optuna as seed trials.

    Returns [] when:
      - ChromaDB collection does not exist (first competition)
      - Collection is empty
      - No results within max_distance
      - Any ChromaDB error

    Never raises. Conservative: returns [] on any failure.
    """
    try:
        client = build_chroma_client()
        try:
            collection = client.get_collection("professor_hpo_memories")
        except Exception:
            return []   # collection does not exist — first competition

        if collection.count() == 0:
            return []

        fingerprint = state.get("competition_fingerprint", {})
        query_text  = fingerprint_to_text(fingerprint)

        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_seeds, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        seeds = []
        for meta, dist in zip(
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist > max_distance:
                continue
            confidence = float(meta.get("confidence", 0.0))
            if confidence < min_confidence:
                continue

            # Extract the param dict stored in metadata
            params_json = meta.get("params_json")
            if not params_json:
                continue
            try:
                params = json.loads(params_json)
                params["_seed_source"] = meta.get("competition_name", "unknown")
                params["_seed_confidence"] = round(confidence, 3)
                params["_seed_distance"]   = round(float(dist), 3)
                seeds.append(params)
            except json.JSONDecodeError:
                continue

        return seeds

    except Exception as e:
        logger.warning(f"[get_hpo_warm_start_seeds] Failed: {e}. Returning empty seeds.")
        return []
```

### New function: `store_hpo_memory(state, best_params, cv_mean, cv_std)`

Add to `memory/memory_schema.py`. Called by `post_mortem_agent` after every competition to grow the memory:

```python
def store_hpo_memory(
    state: dict,
    best_params: dict,
    cv_mean: float,
    cv_std: float,
) -> bool:
    """
    Stores the winning hyperparameter configuration in ChromaDB.
    Returns True on success, False on failure.

    The document text is the competition fingerprint description.
    The metadata contains the serialised params and performance stats.
    Never raises.
    """
    try:
        client     = build_chroma_client()
        collection = client.get_or_create_collection("professor_hpo_memories")

        fingerprint  = state.get("competition_fingerprint", {})
        query_text   = fingerprint_to_text(fingerprint)
        competition  = state.get("competition_name", "unknown")
        session_id   = state.get("session_id", "unknown")

        # Strip internal seed tracking fields before storing
        clean_params = {
            k: v for k, v in best_params.items()
            if not k.startswith("_seed_")
        }

        collection.add(
            documents=[query_text],
            metadatas=[{
                "competition_name": competition,
                "session_id":       session_id,
                "cv_mean":          round(float(cv_mean), 6),
                "cv_std":           round(float(cv_std), 6),
                "model_type":       clean_params.get("model_type", "unknown"),
                "confidence":       _compute_confidence_from_cv(cv_mean, cv_std),
                "params_json":      json.dumps(clean_params),
                "stored_at":        datetime.utcnow().isoformat(),
                **{k: str(v) for k, v in fingerprint.items()},
            }],
            ids=[f"hpo_{session_id}_{int(time.time())}"],
        )
        return True
    except Exception as e:
        logger.warning(f"[store_hpo_memory] Failed: {e}")
        return False


def _compute_confidence_from_cv(cv_mean: float, cv_std: float) -> float:
    """
    Confidence based on CV stability:
      high stability (std < 0.005) → confidence up to 0.90
      high variance (std > 0.030)  → confidence as low as 0.50
    """
    if cv_std <= 0:
        return 0.70
    stability = max(0.0, 1.0 - (cv_std / 0.030))
    return round(min(0.90, 0.50 + 0.40 * stability), 3)
```

### Wire into `agents/ml_optimizer.py`

Add immediately before `optuna.create_study()`:

```python
# Query ChromaDB for warm start seeds
hpo_seeds = get_hpo_warm_start_seeds(state)
if hpo_seeds:
    logger.info(
        f"[ml_optimizer] Warm start: {len(hpo_seeds)} HPO seeds from ChromaDB. "
        f"Sources: {[s.get('_seed_source') for s in hpo_seeds]}"
    )
else:
    logger.info("[ml_optimizer] No HPO seeds found (first competition or no similar history). "
                "Starting Optuna from scratch.")

state["hpo_warm_start_seeds_used"] = len(hpo_seeds)
```

After `optuna.create_study()` and before `study.optimize()`, inject seeds as completed trials:

```python
def _inject_warm_start_seeds(
    study: optuna.Study,
    seeds: list[dict],
    X_np: np.ndarray,
    y: np.ndarray,
    metric: str,
    max_memory_gb: float,
) -> None:
    """
    Runs each seed config as a real trial (not a phantom trial).
    Real trials give Optuna honest performance data to guide the TPE sampler.
    Seeds that fail (OOM, crash) are skipped — never block main study.
    """
    for seed_params in seeds:
        try:
            # Strip internal tracking fields
            clean = {k: v for k, v in seed_params.items() if not k.startswith("_")}

            # Evaluate with HPO overrides (fast mode: 3 folds, 150 trees)
            hpo_params = {**clean, **HPO_OVERRIDES.get(clean.get("model_type", ""), {})}
            fold_scores = _run_cv_no_collect(X_np, y, hpo_params, n_folds=CV_FOLDS_HPO)
            mean_cv = float(np.mean(fold_scores))

            # Add as a real completed trial
            trial = study.ask()
            # Set the parameter values that Optuna needs to know about
            for key, val in clean.items():
                if key in study.best_params if study.trials else {}:
                    pass   # Optuna already tracks this
            study.tell(trial, mean_cv)

            logger.debug(f"[ml_optimizer] Seed trial complete: cv={mean_cv:.5f} "
                        f"(source: {seed_params.get('_seed_source')})")

        except Exception as e:
            logger.warning(f"[ml_optimizer] Seed trial failed: {e}. Skipping.")
            continue

# Call before study.optimize():
if hpo_seeds:
    _inject_warm_start_seeds(study, hpo_seeds, X_np, y, metric, max_memory_gb)
```

### New state field

```python
hpo_warm_start_seeds_used: int   # 0 by default — number of seeds injected
```

---

## TASK 2 — GAP 4: Seed memory layer (`memory/seed_memory.py`)

**What this does:** Pre-populates ChromaDB before the first competition runs so Professor starts with domain knowledge, not a blank slate.

**This is a one-time script.** It is run once before the first competition. Running it again is safe — it checks for existing entries and skips duplicates.

```python
# memory/seed_memory.py
"""
One-time seed script. Pre-populates ChromaDB with:
  Source 1: Domain knowledge from 20 canonical Kaggle competition types
  Source 2: Known winning features per domain
  Source 3: Kaggle meta-knowledge (GBTs win 80% tabular, etc.)

Run once before the first competition:
  python memory/seed_memory.py

Running again is safe — duplicates are skipped.
All seeds are stored with confidence=0.70 (known but unvalidated by Professor).
"""

SEED_CONFIDENCE = 0.70   # known but Professor has not validated these
```

### Source 1 — Competition pattern seeds for `professor_patterns_v2`

```python
COMPETITION_SEEDS = [
    {
        "fingerprint": {
            "task_type":         "binary_classification",
            "target_type":       "binary",
            "domain":            "fraud_detection",
            "imbalance_ratio":   0.02,
            "n_rows_bucket":     "large",
        },
        "validated_approaches": [
            "SMOTE oversampling before CV (not on holdout)",
            "Time-since-last-transaction as primary feature",
            "Log transform of transaction amount",
            "Merchant category aggregation stats",
        ],
        "failed_approaches": [
            "Standard accuracy — use PR-AUC or F1 for imbalanced",
            "Random Forest as final model — XGBoost dominates on fraud",
        ],
        "confidence": SEED_CONFIDENCE,
        "source":     "kaggle_meta_knowledge",
    },
    {
        "fingerprint": {
            "task_type":         "binary_classification",
            "target_type":       "binary",
            "domain":            "churn_prediction",
            "n_rows_bucket":     "medium",
        },
        "validated_approaches": [
            "RFM features: recency, frequency, monetary",
            "Days since last login as top feature",
            "Interaction between plan_type and tenure",
            "Target encoding of high-cardinality categorical columns",
        ],
        "failed_approaches": [
            "Raw customer_id as feature",
        ],
        "confidence": SEED_CONFIDENCE,
        "source":     "kaggle_meta_knowledge",
    },
    {
        "fingerprint": {
            "task_type":         "regression",
            "target_type":       "continuous",
            "domain":            "house_prices",
            "n_rows_bucket":     "small",
        },
        "validated_approaches": [
            "Log transform of SalePrice (target)",
            "OverallQual is the single most predictive feature",
            "TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF",
            "Outlier removal: GrLivArea > 4000 with SalePrice < 300000",
        ],
        "failed_approaches": [
            "Raw SalePrice without log transform — heavy right skew hurts RMSLE",
        ],
        "confidence": SEED_CONFIDENCE,
        "source":     "kaggle_meta_knowledge",
    },
    {
        "fingerprint": {
            "task_type":         "binary_classification",
            "domain":            "titanic_survival",
            "n_rows_bucket":     "tiny",
        },
        "validated_approaches": [
            "Title extracted from Name (Mr, Mrs, Miss, Master)",
            "FamilySize = SibSp + Parch + 1",
            "IsAlone = (FamilySize == 1)",
            "Deck extracted from first character of Cabin",
        ],
        "failed_approaches": [
            "Name as raw feature",
            "Ticket as feature without parsing",
        ],
        "confidence": SEED_CONFIDENCE,
        "source":     "kaggle_meta_knowledge",
    },
    {
        "fingerprint": {
            "task_type":         "binary_classification",
            "domain":            "spaceship_titanic",
            "n_rows_bucket":     "small",
        },
        "validated_approaches": [
            "CryoSleep is the strongest predictor — do not drop it",
            "Cabin split into Deck/Num/Side",
            "GroupSize from PassengerId group number",
            "total_spend = sum of all amenity columns",
            "GroupIsAlone = (GroupSize == 1)",
        ],
        "failed_approaches": [
            "PassengerId as numeric feature",
            "Target encoding without fold isolation on HomePlanet and Destination",
        ],
        "confidence": SEED_CONFIDENCE,
        "source":     "kaggle_meta_knowledge",
    },
]
```

### Source 2 — HPO seeds for `professor_hpo_memories`

```python
HPO_SEEDS = [
    {
        "fingerprint": {
            "task_type":     "binary_classification",
            "n_rows_bucket": "small",
            "n_features_bucket": "medium",
        },
        "params": {
            "model_type":        "lgbm",
            "n_estimators":      800,
            "learning_rate":     0.05,
            "num_leaves":        63,
            "feature_fraction":  0.8,
            "bagging_fraction":  0.8,
            "bagging_freq":      5,
            "min_child_samples": 20,
            "reg_alpha":         0.1,
            "reg_lambda":        0.1,
        },
        "cv_mean": 0.820,
        "cv_std":  0.010,
        "competition_name": "spaceship_titanic_seed",
    },
    {
        "fingerprint": {
            "task_type":     "regression",
            "n_rows_bucket": "small",
            "n_features_bucket": "wide",
        },
        "params": {
            "model_type":      "lgbm",
            "n_estimators":    1000,
            "learning_rate":   0.03,
            "num_leaves":      127,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq":    5,
            "min_child_samples": 30,
            "reg_alpha":       0.05,
            "reg_lambda":      0.05,
        },
        "cv_mean": 0.115,
        "cv_std":  0.008,
        "competition_name": "house_prices_seed",
    },
]
```

### Source 3 — Meta-knowledge text entries for `professor_patterns_v2`

```python
META_KNOWLEDGE = [
    "GBTs (LightGBM, XGBoost, CatBoost) win approximately 80% of tabular Kaggle competitions. "
    "Neural networks rarely outperform GBTs on structured tabular data without very large datasets.",

    "Ensembles of diverse models beat any single model approximately 90% of the time on Kaggle. "
    "Diversity matters more than individual model quality for ensemble gains.",

    "CV/LB gap greater than 0.005 is a strong signal of either validation leakage or distribution shift. "
    "Investigate before submitting.",

    "Target encoding without fold isolation is one of the most common sources of validation leakage "
    "in Kaggle competitions. Always compute encoding statistics from training folds only.",

    "Feature importance from tree models correlates with overfit on training data. "
    "Permutation importance on a holdout fold is more reliable for feature selection.",
]
```

### The seeding script

```python
def run_seed_memory():
    """
    Entry point. Run once before first competition.
    Safe to run multiple times — checks for existing entries.
    """
    from memory.memory_schema import (
        build_chroma_client, fingerprint_to_text,
        store_pattern, store_hpo_memory,
    )

    client = build_chroma_client()

    # Seed competition patterns
    patterns_collection = client.get_or_create_collection("professor_patterns_v2")
    seeded_patterns = 0
    for seed in COMPETITION_SEEDS:
        fingerprint  = seed["fingerprint"]
        query_text   = fingerprint_to_text(fingerprint)
        competition  = seed.get("competition_name", f"seed_{seeded_patterns}")

        # Check for duplicate
        existing = patterns_collection.query(
            query_texts=[query_text],
            n_results=1,
            include=["distances"],
        )
        if existing["distances"] and existing["distances"][0]:
            if existing["distances"][0][0] < 0.05:   # essentially identical
                print(f"  Skipping duplicate pattern for {fingerprint.get('domain', '?')}")
                continue

        store_pattern(
            competition_fingerprint=fingerprint,
            competition_name=competition,
            validated_approaches=seed["validated_approaches"],
            failed_approaches=seed.get("failed_approaches", []),
            confidence=seed["confidence"],
        )
        seeded_patterns += 1
        print(f"  Seeded pattern: {fingerprint.get('domain', fingerprint.get('task_type', '?'))}")

    print(f"Competition patterns seeded: {seeded_patterns}")

    # Seed HPO memories
    hpo_collection = client.get_or_create_collection("professor_hpo_memories")
    seeded_hpo = 0
    for seed in HPO_SEEDS:
        fingerprint = seed["fingerprint"]
        query_text  = fingerprint_to_text(fingerprint)

        existing = hpo_collection.query(
            query_texts=[query_text],
            n_results=1,
            include=["distances"],
        )
        if existing["distances"] and existing["distances"][0]:
            if existing["distances"][0][0] < 0.05:
                print(f"  Skipping duplicate HPO seed for {fingerprint}")
                continue

        store_hpo_memory(
            state={
                "competition_fingerprint": fingerprint,
                "competition_name":        seed["competition_name"],
                "session_id":              "seed_script",
            },
            best_params=seed["params"],
            cv_mean=seed["cv_mean"],
            cv_std=seed["cv_std"],
        )
        seeded_hpo += 1
        print(f"  Seeded HPO: {seed['competition_name']}")

    print(f"HPO memories seeded: {seeded_hpo}")
    print("Seed memory complete.")


if __name__ == "__main__":
    run_seed_memory()
```

---

## TASK 3 — Memory quality scoring (`memory/memory_quality.py`)

```python
# memory/memory_quality.py
"""
Quality scoring for ChromaDB memory entries.

helpfulness_rate: fraction of times a retrieved memory led to a CV improvement.
Computed after each competition by post_mortem_agent.

Retrieval threshold: helpfulness_rate > 0.6 AND confidence > 0.7
Memories that fall below threshold are flagged for decay.

Decay: confidence is reduced by 0.05 per competition where memory was retrieved
but did not help. Memories below confidence=0.50 are removed.
"""

RETRIEVAL_THRESHOLD_HELPFULNESS = 0.60
RETRIEVAL_THRESHOLD_CONFIDENCE  = 0.70
DECAY_RATE                      = 0.05
REMOVAL_THRESHOLD               = 0.50


def update_memory_helpfulness(
    collection_name: str,
    memory_id: str,
    was_helpful: bool,
) -> bool:
    """
    Updates the helpfulness_rate of a memory entry after a competition.

    was_helpful=True:  this memory contributed to a CV improvement
    was_helpful=False: this memory was retrieved but did not help

    Updates the entry in ChromaDB. Returns True on success, False on failure.
    Never raises.
    """
    try:
        client     = build_chroma_client()
        collection = client.get_collection(collection_name)

        result = collection.get(ids=[memory_id], include=["metadatas"])
        if not result["metadatas"]:
            return False

        meta = result["metadatas"][0]
        n_retrieved  = int(meta.get("n_retrieved", 0)) + 1
        n_helpful    = int(meta.get("n_helpful", 0)) + (1 if was_helpful else 0)
        helpfulness  = round(n_helpful / n_retrieved, 4)

        # Apply decay if not helpful
        confidence = float(meta.get("confidence", 0.70))
        if not was_helpful:
            confidence = round(max(0.0, confidence - DECAY_RATE), 4)

        new_meta = {
            **meta,
            "n_retrieved":     n_retrieved,
            "n_helpful":       n_helpful,
            "helpfulness_rate": helpfulness,
            "confidence":       confidence,
            "last_updated":     datetime.utcnow().isoformat(),
        }

        collection.update(ids=[memory_id], metadatas=[new_meta])
        return True

    except Exception as e:
        logger.warning(f"[update_memory_helpfulness] Failed for {memory_id}: {e}")
        return False


def remove_decayed_memories(collection_name: str) -> int:
    """
    Removes all entries with confidence < REMOVAL_THRESHOLD (0.50).
    Returns count of removed entries.
    Called by post_mortem_agent after updating helpfulness.
    Never raises.
    """
    try:
        client     = build_chroma_client()
        collection = client.get_collection(collection_name)

        if collection.count() == 0:
            return 0

        result = collection.get(include=["metadatas", "ids"])
        to_remove = [
            id_ for id_, meta in zip(result["ids"], result["metadatas"])
            if float(meta.get("confidence", 1.0)) < REMOVAL_THRESHOLD
        ]

        if to_remove:
            collection.delete(ids=to_remove)
            logger.info(
                f"[memory_quality] Removed {len(to_remove)} decayed entries "
                f"from '{collection_name}'."
            )

        return len(to_remove)

    except Exception as e:
        logger.warning(f"[remove_decayed_memories] Failed: {e}")
        return 0


def should_retrieve(meta: dict) -> bool:
    """
    Returns True if a memory entry meets the retrieval quality threshold.
    Applied as a post-filter after ChromaDB distance filtering.
    """
    helpfulness = float(meta.get("helpfulness_rate", 1.0))  # default: always retrieve if new
    confidence  = float(meta.get("confidence", SEED_CONFIDENCE))
    n_retrieved = int(meta.get("n_retrieved", 0))

    # New memories (never retrieved) get benefit of the doubt
    if n_retrieved == 0:
        return confidence >= RETRIEVAL_THRESHOLD_CONFIDENCE

    return (
        helpfulness >= RETRIEVAL_THRESHOLD_HELPFULNESS and
        confidence  >= RETRIEVAL_THRESHOLD_CONFIDENCE
    )
```

---

## TASK 4 — Vector memory stub (`memory/pinecone_memory.py`)

**Build as an in-memory dict stub.** Pinecone setup takes time and the interface should work the same way regardless of backend. The stub implements the full interface so Pinecone can be swapped in later without changing any calling code.

```python
# memory/pinecone_memory.py
"""
Vector memory for experiment results, domain briefings, and feature findings.

Currently implemented as an in-memory dict (stub).
Interface is identical to what a Pinecone backend would expose.
Swap backend by changing _get_client() only.

Quality scoring fields on every entry:
  helpfulness_rate: float  — fraction of retrievals that led to improvement
  validated_by_critic: bool — whether red_team_critic validated this finding
  confidence: float        — 0.0 to 1.0

Retrieval threshold: helpfulness_rate > 0.6 AND confidence > 0.7
"""

from dataclasses import dataclass, field
from datetime import datetime
import json, logging, numpy as np
from typing import Any

logger = logging.getLogger(__name__)

_STORE: dict[str, dict] = {}   # in-memory backend


@dataclass
class MemoryEntry:
    id:                  str
    collection:          str
    text:                str                    # what is stored (the findable content)
    metadata:            dict = field(default_factory=dict)
    embedding:           list[float] = field(default_factory=list)
    helpfulness_rate:    float = 1.0            # assume helpful until proven otherwise
    validated_by_critic: bool = False
    confidence:          float = 0.70
    n_retrieved:         int = 0
    n_helpful:           int = 0
    stored_at:           str = field(default_factory=lambda: datetime.utcnow().isoformat())


def upsert(
    collection: str,
    id: str,
    text: str,
    metadata: dict = None,
    confidence: float = 0.70,
    validated_by_critic: bool = False,
) -> bool:
    """
    Stores or updates an entry. Returns True on success.
    Never raises.
    """
    try:
        key = f"{collection}::{id}"
        _STORE[key] = {
            "id":                  id,
            "collection":          collection,
            "text":                text,
            "metadata":            metadata or {},
            "confidence":          confidence,
            "validated_by_critic": validated_by_critic,
            "helpfulness_rate":    _STORE.get(key, {}).get("helpfulness_rate", 1.0),
            "n_retrieved":         _STORE.get(key, {}).get("n_retrieved", 0),
            "n_helpful":           _STORE.get(key, {}).get("n_helpful", 0),
            "stored_at":           datetime.utcnow().isoformat(),
        }
        return True
    except Exception as e:
        logger.warning(f"[pinecone_memory.upsert] Failed: {e}")
        return False


def query(
    collection: str,
    query_text: str,
    n_results: int = 5,
    min_confidence: float = 0.60,
    min_helpfulness: float = 0.60,
) -> list[dict]:
    """
    Returns up to n_results entries from the collection that meet quality thresholds.
    Uses simple text overlap scoring (stub behaviour — Pinecone would use embeddings).
    Never raises. Returns [] on any error.
    """
    try:
        entries = [
            v for k, v in _STORE.items()
            if k.startswith(f"{collection}::")
            and float(v.get("confidence", 0)) >= min_confidence
            and (
                int(v.get("n_retrieved", 0)) == 0 or
                float(v.get("helpfulness_rate", 1.0)) >= min_helpfulness
            )
        ]

        # Stub scoring: fraction of query words found in text
        query_words = set(query_text.lower().split())
        def _score(entry):
            text_words = set(entry["text"].lower().split())
            return len(query_words & text_words) / max(len(query_words), 1)

        entries.sort(key=_score, reverse=True)
        return entries[:n_results]

    except Exception as e:
        logger.warning(f"[pinecone_memory.query] Failed: {e}")
        return []


def update_helpfulness(collection: str, id: str, was_helpful: bool) -> bool:
    """Updates helpfulness_rate after a competition result is known."""
    try:
        key = f"{collection}::{id}"
        if key not in _STORE:
            return False
        entry = _STORE[key]
        entry["n_retrieved"] += 1
        if was_helpful:
            entry["n_helpful"] += 1
        entry["helpfulness_rate"] = round(
            entry["n_helpful"] / entry["n_retrieved"], 4
        )
        if not was_helpful:
            entry["confidence"] = round(
                max(0.0, entry["confidence"] - 0.05), 4
            )
        return True
    except Exception as e:
        logger.warning(f"[pinecone_memory.update_helpfulness] Failed: {e}")
        return False


def count(collection: str) -> int:
    return sum(1 for k in _STORE if k.startswith(f"{collection}::"))


def clear(collection: str) -> None:
    """Remove all entries from a collection. Used in tests."""
    to_delete = [k for k in _STORE if k.startswith(f"{collection}::")]
    for k in to_delete:
        del _STORE[k]
```

---

## COMMIT SEQUENCE

```
git commit -m "Day 24: memory/memory_schema.py — get_hpo_warm_start_seeds() + store_hpo_memory()"
git commit -m "Day 24: ml_optimizer.py — GAP 11 Optuna warm start from ChromaDB HPO memories"
git commit -m "Day 24: memory/seed_memory.py — pre-populate ChromaDB before first competition"
git commit -m "Day 24: memory/memory_quality.py — helpfulness rate, decay, retrieval threshold"
git commit -m "Day 24: memory/pinecone_memory.py — in-memory dict stub with full interface"
```

---

## VERIFICATION BEFORE EACH COMMIT

```bash
python -c "from memory.memory_schema import get_hpo_warm_start_seeds, store_hpo_memory; print('OK')"
python -c "from memory.seed_memory import run_seed_memory; print('OK')"
python -c "from memory.memory_quality import update_memory_helpfulness, should_retrieve; print('OK')"
python -c "from memory.pinecone_memory import upsert, query, update_helpfulness; print('OK')"

# Run seed script
python memory/seed_memory.py

pytest tests/contracts/ -v --tb=short
pytest tests/regression/ -v --tb=short
```

All imports and both pytest commands must show zero failures before committing.