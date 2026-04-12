# memory/seed_memory.py
"""
One-time seed script. Pre-populates ChromaDB with:
  Source 1: Domain knowledge from canonical Kaggle competition types
  Source 2: Known winning features per domain
  Source 3: Kaggle meta-knowledge (GBTs win 80% tabular, etc.)

Run once before the first competition:
  python memory/seed_memory.py

Running again is safe — duplicates are skipped.
All seeds are stored with confidence=0.70 (known but unvalidated by Professor).
"""

import logging

SEED_CONFIDENCE = 0.70   # known but Professor has not validated these

logger = logging.getLogger(__name__)

# =========================================================================
# Source 1 — Competition pattern seeds for professor_patterns_v2
# =========================================================================

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

# =========================================================================
# Source 2 — HPO seeds for professor_hpo_memories
# =========================================================================

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

# =========================================================================
# Source 3 — Meta-knowledge text entries for professor_patterns_v2
# =========================================================================

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

# =========================================================================
# Seeding logic
# =========================================================================


def run_seed_memory():
    """
    Entry point. Run once before first competition.
    Safe to run multiple times — checks for existing entries.
    """
    from memory.memory_schema import (
        build_chroma_client, fingerprint_to_text,
        store_pattern, store_hpo_memory,
        PATTERNS_COLLECTION,
    )

    client = build_chroma_client()

    # ── Seed competition patterns ──────────────────────────────
    patterns_collection = client.get_or_create_collection(PATTERNS_COLLECTION)
    seeded_patterns = 0
    for seed in COMPETITION_SEEDS:
        fingerprint = seed["fingerprint"]
        query_text = fingerprint_to_text(fingerprint)
        competition = seed.get("domain", f"seed_{seeded_patterns}")

        # Check for duplicate
        existing = patterns_collection.query(
            query_texts=[query_text],
            n_results=1,
            include=["distances"],
        )
        if existing["distances"] and existing["distances"][0]:
            if existing["distances"][0][0] < 0.05:
                logger.info(f"  Skipping duplicate pattern for {fingerprint.get('domain', '?')}")
                continue

        store_pattern(
            fingerprint=fingerprint,
            validated_approaches=seed["validated_approaches"],
            failed_approaches=seed.get("failed_approaches", []),
            competition_name=competition,
            confidence=seed["confidence"],
        )
        seeded_patterns += 1
        logger.info(f"  Seeded pattern: {fingerprint.get('domain', fingerprint.get('task_type', '?'))}")

    # ── Seed meta-knowledge text entries ───────────────────────
    for i, text in enumerate(META_KNOWLEDGE):
        patterns_collection.add(
            documents=[text],
            metadatas=[{
                "competition_name": f"meta_knowledge_{i}",
                "pattern_json": '{"meta": true}',
                "confidence": str(SEED_CONFIDENCE),
            }],
            ids=[f"meta_{i}"],
        )

    logger.info(f"Competition patterns seeded: {seeded_patterns}")

    # ── Seed HPO memories ──────────────────────────────────────
    seeded_hpo = 0
    for seed in HPO_SEEDS:
        fingerprint = seed["fingerprint"]
        query_text = fingerprint_to_text(fingerprint)

        existing = patterns_collection.query(
            query_texts=[query_text],
            n_results=1,
            include=["distances"],
        )
        if existing["distances"] and existing["distances"][0]:
            if existing["distances"][0][0] < 0.05:
                logger.info(f"  Skipping duplicate HPO seed for {fingerprint}")
                continue

        store_hpo_memory(
            state={
                "competition_fingerprint": fingerprint,
                "competition_name": seed["competition_name"],
                "session_id": "seed_script",
            },
            best_params=seed["params"],
            cv_mean=seed["cv_mean"],
            cv_std=seed["cv_std"],
        )
        seeded_hpo += 1
        logger.info(f"  Seeded HPO: {seed['competition_name']}")

    logger.info(f"HPO memories seeded: {seeded_hpo}")
    logger.info("Seed memory complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_seed_memory()
