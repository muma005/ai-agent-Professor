"""
Registry of historical competitions with metadata and approximate LB percentile curves.
Percentile curve: maps a raw score -> approximate percentile rank (0-100, 100 = top of LB).
Medal thresholds (approximate):
  Gold:   top 3%  -> percentile >= 97
  Silver: top 10% -> percentile >= 90
  Bronze: top 20% -> percentile >= 80
"""

from dataclasses import dataclass, field
import numpy as np

BETTER_IS_HIGHER = {"accuracy", "auc", "f1"}
BETTER_IS_LOWER  = {"rmse", "rmsle", "mae", "logloss", "log_loss", "brier_score"}


@dataclass
class LeaderboardCurve:
    breakpoints: list        # [(score, percentile), ...] sorted by score ascending
    higher_is_better: bool

    def score_to_percentile(self, score: float) -> float:
        if not self.breakpoints:
            return 50.0
        scores = [b[0] for b in self.breakpoints]
        pcts   = [b[1] for b in self.breakpoints]
        return float(np.interp(score, scores, pcts))


@dataclass
class CompetitionSpec:
    competition_id:       str
    display_name:         str
    task_type:            str   # "binary_classification" | "regression" | "multiclass"
    target_column:        str
    id_column:            str
    evaluation_metric:    str   # "accuracy" | "rmsle" | "auc" | "logloss"
    train_file:           str
    test_file:            str
    sample_submission_file: str
    lb_curve:             LeaderboardCurve
    known_winning_features: list = field(default_factory=list)
    known_pitfalls:       list  = field(default_factory=list)
    gold_threshold:       float = 0.0
    silver_threshold:     float = 0.0
    bronze_threshold:     float = 0.0


COMPETITION_REGISTRY = {

    "spaceship-titanic": CompetitionSpec(
        competition_id="spaceship-titanic",
        display_name="Spaceship Titanic",
        task_type="binary_classification",
        target_column="Transported",
        id_column="PassengerId",
        evaluation_metric="accuracy",
        train_file="train.csv",
        test_file="test.csv",
        sample_submission_file="sample_submission.csv",
        lb_curve=LeaderboardCurve(
            breakpoints=[
                (0.770, 5.0), (0.785, 20.0), (0.795, 40.0), (0.803, 60.0),
                (0.810, 80.0), (0.815, 90.0), (0.820, 95.0), (0.825, 99.0),
            ],
            higher_is_better=True,
        ),
        known_winning_features=["CryoSleep", "Cabin_deck", "total_spend", "GroupSize"],
        known_pitfalls=["Target-encoding without fold isolation", "Ignoring group structure in PassengerId"],
        gold_threshold=0.820,
        silver_threshold=0.813,
        bronze_threshold=0.805,
    ),

    "titanic": CompetitionSpec(
        competition_id="titanic",
        display_name="Titanic — Machine Learning from Disaster",
        task_type="binary_classification",
        target_column="Survived",
        id_column="PassengerId",
        evaluation_metric="accuracy",
        train_file="train.csv",
        test_file="test.csv",
        sample_submission_file="gender_submission.csv",
        lb_curve=LeaderboardCurve(
            breakpoints=[
                (0.750, 10.0), (0.770, 30.0), (0.780, 50.0),
                (0.790, 70.0), (0.800, 85.0), (0.810, 92.0), (0.820, 96.0),
            ],
            higher_is_better=True,
        ),
        known_winning_features=["Title", "FamilySize", "IsAlone", "Deck"],
        known_pitfalls=["Overfitting on 891 training rows", "Name-based leakage via titles"],
        gold_threshold=0.820,
        silver_threshold=0.800,
        bronze_threshold=0.780,
    ),

    "house-prices-advanced-regression-techniques": CompetitionSpec(
        competition_id="house-prices-advanced-regression-techniques",
        display_name="House Prices — Advanced Regression Techniques",
        task_type="regression",
        target_column="SalePrice",
        id_column="Id",
        evaluation_metric="rmsle",
        train_file="train.csv",
        test_file="test.csv",
        sample_submission_file="sample_submission.csv",
        lb_curve=LeaderboardCurve(
            breakpoints=[
                (0.160, 5.0), (0.140, 20.0), (0.130, 40.0), (0.125, 60.0),
                (0.120, 80.0), (0.115, 90.0), (0.110, 95.0), (0.105, 99.0),
            ],
            higher_is_better=False,
        ),
        known_winning_features=["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageArea"],
        known_pitfalls=["Not log-transforming SalePrice", "Outliers in GrLivArea"],
        gold_threshold=0.110,
        silver_threshold=0.118,
        bronze_threshold=0.125,
    ),
}
