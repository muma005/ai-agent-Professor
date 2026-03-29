"""
Competition Registry — metadata for all benchmark competitions.

Each entry contains everything needed to simulate a competition:
- Task type, target column, evaluation metric
- LB percentile curves for medal calculation
- Split strategy matching the real competition's structure
- Domain classification for Domain Research Engine testing

LB percentile curves are scraped from actual Kaggle leaderboards
after competitions close. These are public data.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompetitionEntry:
    # ── Identity ──
    slug: str                    # Kaggle competition slug
    title: str                   # Human-readable name

    # ── Task Definition ──
    task_type: str               # binary | multiclass | regression
    target_column: str           # Column name of the target
    id_column: str               # Row identifier column
    metric: str                  # Evaluation metric name
    metric_direction: str        # maximize | minimize

    # ── LB Calibration ──
    # Keys = percentile (lower = better rank), Values = score threshold
    # Example: {10: 0.810} means top 10% scored >= 0.810
    lb_percentiles: dict = field(default_factory=dict)
    gold_threshold: float = 0.0   # Score for top ~10% (gold)
    silver_threshold: float = 0.0 # Score for top ~5%
    bronze_threshold: float = 0.0 # Score for top ~25%
    total_teams: int = 0

    # ── Split Configuration ──
    split_strategy: str = "stratified"  # stratified | temporal | group
    split_column: Optional[str] = None  # Column for temporal/group splits
    test_ratio: float = 0.40            # Fraction held out as test
    public_ratio: float = 0.30          # Fraction of test → public LB
    random_seed: int = 42               # Deterministic, reproducible

    # ── Data Source ──
    download_method: str = "kaggle_api"  # kaggle_api | manual | cached
    cached_path: Optional[str] = None
    requires_join: bool = False
    join_instructions: Optional[str] = None

    # ── Domain ──
    primary_domain: str = "general"
    sub_domain: str = ""


# ════════════════════════════════════════════════════════════
# REGISTRY — Start with 3 competitions, expand over time
# ════════════════════════════════════════════════════════════

REGISTRY = [
    # ── Binary Classification ─────────────────────────────
    CompetitionEntry(
        slug="spaceship-titanic",
        title="Spaceship Titanic",
        task_type="binary",
        target_column="Transported",
        id_column="PassengerId",
        metric="accuracy",
        metric_direction="maximize",
        lb_percentiles={5: 0.815, 10: 0.810, 25: 0.795, 50: 0.780, 75: 0.760, 90: 0.740},
        gold_threshold=0.810,
        silver_threshold=0.815,
        bronze_threshold=0.795,
        total_teams=2500,
        split_strategy="stratified",
        primary_domain="transport",
        sub_domain="passenger survival prediction",
    ),

    CompetitionEntry(
        slug="titanic",
        title="Titanic - Machine Learning from Disaster",
        task_type="binary",
        target_column="Survived",
        id_column="PassengerId",
        metric="accuracy",
        metric_direction="maximize",
        lb_percentiles={5: 0.810, 10: 0.800, 25: 0.790, 50: 0.775, 75: 0.755, 90: 0.730},
        gold_threshold=0.800,
        silver_threshold=0.810,
        bronze_threshold=0.790,
        total_teams=15000,
        split_strategy="stratified",
        primary_domain="transport",
        sub_domain="passenger survival prediction",
    ),

    # ── Regression ────────────────────────────────────────
    CompetitionEntry(
        slug="house-prices-advanced-regression-techniques",
        title="House Prices: Advanced Regression",
        task_type="regression",
        target_column="SalePrice",
        id_column="Id",
        metric="rmsle",
        metric_direction="minimize",
        lb_percentiles={5: 0.115, 10: 0.120, 25: 0.130, 50: 0.145, 75: 0.165, 90: 0.200},
        gold_threshold=0.120,
        silver_threshold=0.115,
        bronze_threshold=0.130,
        total_teams=5000,
        split_strategy="stratified",
        primary_domain="real_estate",
        sub_domain="residential house price prediction",
    ),
]


def get_competition(slug: str) -> CompetitionEntry:
    """Retrieve a competition entry by slug. Raises ValueError if not found."""
    for entry in REGISTRY:
        if entry.slug == slug:
            return entry
    available = [e.slug for e in REGISTRY]
    raise ValueError(
        f"Competition '{slug}' not in registry. "
        f"Available: {available}"
    )


def list_competitions() -> list:
    """Return list of all registered competition slugs."""
    return [e.slug for e in REGISTRY]
