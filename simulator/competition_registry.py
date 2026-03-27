"""
Competition Registry — Structured database of historical competitions.

Each entry contains all metadata needed to simulate a competition faithfully:
- Task type, target column, evaluation metric
- LB percentile curve (scraped from actual Kaggle leaderboards)
- Medal thresholds (gold/silver/bronze)
- Split configuration (stratified/temporal/group, test ratio, public ratio)
- Data source information

Registry growth strategy:
- Start with 10 competitions covering binary, multiclass, regression, imbalanced
- Add 5 per month as Professor runs benchmarks
- After 50+: statistical confidence in aggregate metrics
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from pathlib import Path


@dataclass
class CompetitionEntry:
    """
    Complete metadata for simulating a single competition.
    
    Attributes:
        slug: Unique identifier (e.g., "spaceship-titanic")
        title: Human-readable name
        kaggle_id: Competition ID for Kaggle API download
        task_type: "binary" | "multiclass" | "regression" | "multilabel"
        target_column: Name of target column in dataset
        id_column: Name of ID column for submissions
        metric: Evaluation metric name
        metric_direction: "maximize" | "minimize"
        lb_percentiles: Dict mapping percentile -> score threshold
                       e.g., {10: 0.810, 25: 0.795, 50: 0.780, 75: 0.760, 90: 0.740}
        gold_threshold: Score needed for top ~10% (gold medal)
        silver_threshold: Score needed for top ~5% (silver medal)
        bronze_threshold: Score needed for top ~25% (bronze medal)
        total_teams: Total teams in original competition (for percentile calc)
        split_strategy: "stratified" | "temporal" | "group"
        split_column: Column used for temporal/group splits (None if stratified)
        test_ratio: Fraction held out as simulated test (0.40 = 40%)
        public_ratio: Fraction of test that forms public LB (0.30 = 30%)
        random_seed: Seed for deterministic, reproducible splits (always 42)
        download_method: "kaggle_api" | "manual" | "cached"
        cached_path: Local path if already downloaded
        requires_join: True if competition has multiple data files
        join_instructions: SQL or description of how to join files
        primary_domain: Domain category (transport, healthcare, finance, etc.)
        sub_domain: Specific sub-domain
    """
    
    # Identity
    slug: str
    title: str
    kaggle_id: str
    
    # Task
    task_type: str
    target_column: str
    id_column: str
    metric: str
    metric_direction: str
    
    # LB Percentile Curve (calibration data)
    lb_percentiles: Dict[int, float]
    gold_threshold: float
    silver_threshold: float
    bronze_threshold: float
    total_teams: int
    
    # Split Configuration
    split_strategy: str = "stratified"
    split_column: Optional[str] = None
    test_ratio: float = 0.40
    public_ratio: float = 0.30
    random_seed: int = 42
    
    # Data Source
    download_method: str = "kaggle_api"
    cached_path: Optional[str] = None
    requires_join: bool = False
    join_instructions: Optional[str] = None
    
    # Domain
    primary_domain: str = "general"
    sub_domain: Optional[str] = None
    
    # Additional metadata
    notes: Optional[str] = None
    year: Optional[int] = None
    
    def get_data_dir(self) -> Path:
        """Returns the data directory for this competition."""
        return Path("simulator/data") / self.slug
    
    def get_train_path(self) -> Path:
        """Returns path to train.csv (what Professor sees)."""
        return self.get_data_dir() / "train.csv"
    
    def get_test_path(self) -> Path:
        """Returns path to test.csv (features only, public + private rows)."""
        return self.get_data_dir() / "test.csv"
    
    def get_public_labels_path(self) -> Path:
        """Returns path to .public_labels.csv (HIDDEN - only leaderboard reads)."""
        return self.get_data_dir() / ".public_labels.csv"
    
    def get_private_labels_path(self) -> Path:
        """Returns path to .private_labels.csv (HIDDEN - only leaderboard reads)."""
        return self.get_data_dir() / ".private_labels.csv"
    
    def get_sample_submission_path(self) -> Path:
        """Returns path to sample_submission.csv."""
        return self.get_data_dir() / "sample_submission.csv"
    
    def estimate_percentile(self, score: float) -> float:
        """
        Estimate percentile rank for a given score using the LB curve.
        Returns percentage (lower = better). 5.0 = top 5%.
        """
        percentiles = self.lb_percentiles
        maximize = self.metric_direction == "maximize"
        
        # Sort percentiles by threshold
        sorted_pcts = sorted(percentiles.items(), key=lambda x: x[1], reverse=maximize)
        
        for pct, threshold in sorted_pcts:
            if maximize and score >= threshold:
                return float(pct)
            elif not maximize and score <= threshold:
                return float(pct)
        
        # Default: bottom half
        return 90.0
    
    def compute_medal(self, score: float) -> str:
        """
        Compute medal based on score and thresholds.
        Returns: "gold", "silver", "bronze", or "none"
        """
        maximize = self.metric_direction == "maximize"
        
        if maximize:
            if score >= self.gold_threshold:
                return "gold"
            elif score >= self.silver_threshold:
                return "silver"
            elif score >= self.bronze_threshold:
                return "bronze"
        else:
            if score <= self.gold_threshold:
                return "gold"
            elif score <= self.silver_threshold:
                return "silver"
            elif score <= self.bronze_threshold:
                return "bronze"
        
        return "none"


# =============================================================================
# INITIAL REGISTRY — 10 competitions covering major types
# =============================================================================

REGISTRY: list[CompetitionEntry] = [
    # ── Binary Classification ──
    
    CompetitionEntry(
        slug="spaceship-titanic",
        title="Spaceship Titanic",
        kaggle_id="spaceship-titanic",
        task_type="binary",
        target_column="Transported",
        id_column="PassengerId",
        metric="accuracy",
        metric_direction="maximize",
        lb_percentiles={10: 0.810, 25: 0.795, 50: 0.780, 75: 0.760, 90: 0.740},
        gold_threshold=0.810,
        silver_threshold=0.795,
        bronze_threshold=0.780,
        total_teams=2500,
        split_strategy="stratified",
        primary_domain="transport",
        sub_domain="passenger survival prediction",
        year=2022,
    ),
    
    CompetitionEntry(
        slug="titanic",
        title="Titanic - Machine Learning from Disaster",
        kaggle_id="titanic",
        task_type="binary",
        target_column="Survived",
        id_column="PassengerId",
        metric="accuracy",
        metric_direction="maximize",
        lb_percentiles={10: 0.800, 25: 0.790, 50: 0.775, 75: 0.755, 90: 0.740},
        gold_threshold=0.800,
        silver_threshold=0.790,
        bronze_threshold=0.775,
        total_teams=15000,
        split_strategy="stratified",
        primary_domain="transport",
        sub_domain="passenger survival prediction",
        year=2009,
        notes="Classic binary classification. Watch for title encoding leakage.",
    ),
    
    CompetitionEntry(
        slug="playground-series-s4e8",
        title="Playground Series Season 4 Episode 8",
        kaggle_id="playground-series-s4e8",
        task_type="binary",
        target_column="class",
        id_column="id",
        metric="auc",
        metric_direction="maximize",
        lb_percentiles={10: 0.890, 25: 0.875, 50: 0.860, 75: 0.840, 90: 0.820},
        gold_threshold=0.890,
        silver_threshold=0.875,
        bronze_threshold=0.860,
        total_teams=1800,
        split_strategy="stratified",
        primary_domain="general",
        sub_domain="synthetic binary classification",
        year=2024,
    ),
    
    # ── Imbalanced Binary ──
    
    CompetitionEntry(
        slug="icr-identify-age-related-conditions",
        title="ICR - Identify Age-Related Conditions",
        kaggle_id="icr-identify-age-related-conditions",
        task_type="binary",
        target_column="Class",
        id_column="Id",
        metric="balanced_log_loss",
        metric_direction="minimize",
        lb_percentiles={10: 0.20, 25: 0.28, 50: 0.35, 75: 0.45, 90: 0.55},
        gold_threshold=0.20,
        silver_threshold=0.28,
        bronze_threshold=0.35,
        total_teams=6400,
        split_strategy="stratified",
        primary_domain="healthcare",
        sub_domain="medical condition prediction",
        year=2023,
        notes="Highly imbalanced. Balanced log loss requires careful calibration.",
    ),
    
    # ── Regression ──
    
    CompetitionEntry(
        slug="house-prices-advanced-regression-techniques",
        title="House Prices - Advanced Regression Techniques",
        kaggle_id="house-prices-advanced-regression-techniques",
        task_type="regression",
        target_column="SalePrice",
        id_column="Id",
        metric="rmsle",
        metric_direction="minimize",
        lb_percentiles={10: 0.120, 25: 0.130, 50: 0.145, 75: 0.165, 90: 0.180},
        gold_threshold=0.120,
        silver_threshold=0.130,
        bronze_threshold=0.145,
        total_teams=5000,
        split_strategy="stratified",
        primary_domain="real_estate",
        sub_domain="house price prediction",
        year=2016,
        notes="Log-transform SalePrice. Watch for outliers in GrLivArea.",
    ),
    
    CompetitionEntry(
        slug="playground-series-s4e7",
        title="Playground Series Season 4 Episode 7",
        kaggle_id="playground-series-s4e7",
        task_type="regression",
        target_column="strength",
        id_column="id",
        metric="rmse",
        metric_direction="minimize",
        lb_percentiles={10: 5.8, 25: 6.2, 50: 6.8, 75: 7.5, 90: 8.2},
        gold_threshold=5.8,
        silver_threshold=6.2,
        bronze_threshold=6.8,
        total_teams=2100,
        split_strategy="stratified",
        primary_domain="materials",
        sub_domain="concrete strength prediction",
        year=2024,
    ),
    
    # ── Multiclass ──
    
    CompetitionEntry(
        slug="playground-series-s4e9",
        title="Playground Series Season 4 Episode 9",
        kaggle_id="playground-series-s4e9",
        task_type="multiclass",
        target_column="Status",
        id_column="id",
        metric="macro_f1",
        metric_direction="maximize",
        lb_percentiles={10: 0.850, 25: 0.830, 50: 0.810, 75: 0.785, 90: 0.760},
        gold_threshold=0.850,
        silver_threshold=0.830,
        bronze_threshold=0.810,
        total_teams=1200,
        split_strategy="stratified",
        primary_domain="healthcare",
        sub_domain="insurance claim status",
        year=2024,
    ),
    
    CompetitionEntry(
        slug="tabular-playground-series-dec-2022",
        title="Tabular Playground Series - Dec 2022",
        kaggle_id="tabular-playground-series-dec-2022",
        task_type="multiclass",
        target_column="target",
        id_column="row_id",
        metric="macro_f1",
        metric_direction="maximize",
        lb_percentiles={10: 0.720, 25: 0.700, 50: 0.680, 75: 0.655, 90: 0.630},
        gold_threshold=0.720,
        silver_threshold=0.700,
        bronze_threshold=0.680,
        total_teams=3200,
        split_strategy="stratified",
        primary_domain="general",
        sub_domain="synthetic multiclass",
        year=2022,
    ),
    
    # ── Temporal Split ──
    
    CompetitionEntry(
        slug="store-sales-time-series-forecasting",
        title="Store Sales - Time Series Forecasting",
        kaggle_id="store-sales-time-series-forecasting",
        task_type="regression",
        target_column="sales",
        id_column="id",
        metric="rmsle",
        metric_direction="minimize",
        lb_percentiles={10: 0.380, 25: 0.420, 50: 0.480, 75: 0.550, 90: 0.620},
        gold_threshold=0.380,
        silver_threshold=0.420,
        bronze_threshold=0.480,
        total_teams=4500,
        split_strategy="temporal",
        split_column="date",
        test_ratio=0.40,
        public_ratio=0.30,
        primary_domain="retail",
        sub_domain="sales forecasting",
        year=2022,
        notes="Temporal split required. Last 40% of dates form test set.",
    ),
    
    # ── Group Split ──
    
    CompetitionEntry(
        slug="livestock-disease-prediction",
        title="Livestock Disease Prediction",
        kaggle_id="livestock-disease-prediction",
        task_type="binary",
        target_column="disease",
        id_column="animal_id",
        metric="auc",
        metric_direction="maximize",
        lb_percentiles={10: 0.880, 25: 0.860, 50: 0.835, 75: 0.805, 90: 0.775},
        gold_threshold=0.880,
        silver_threshold=0.860,
        bronze_threshold=0.835,
        total_teams=980,
        split_strategy="group",
        split_column="farm_id",
        test_ratio=0.40,
        public_ratio=0.30,
        primary_domain="agriculture",
        sub_domain="animal disease prediction",
        year=2023,
        notes="Group split by farm_id. No farm appears in both train and test.",
    ),
]

# Registry lookup utilities
REGISTRY_BY_SLUG: Dict[str, CompetitionEntry] = {entry.slug: entry for entry in REGISTRY}


def get_competition(slug: str) -> CompetitionEntry:
    """Get a competition entry by slug."""
    if slug not in REGISTRY_BY_SLUG:
        raise ValueError(
            f"Unknown competition: '{slug}'. "
            f"Available: {list(REGISTRY_BY_SLUG.keys())}"
        )
    return REGISTRY_BY_SLUG[slug]


def get_all_competitions() -> list[CompetitionEntry]:
    """Get all competition entries."""
    return REGISTRY


def add_competition(entry: CompetitionEntry) -> None:
    """Add a new competition to the registry."""
    if entry.slug in REGISTRY_BY_SLUG:
        raise ValueError(f"Competition '{entry.slug}' already exists in registry.")
    REGISTRY.append(entry)
    REGISTRY_BY_SLUG[entry.slug] = entry
