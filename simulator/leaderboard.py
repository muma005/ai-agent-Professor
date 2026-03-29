"""
Simulated Leaderboard — behaves identically to Kaggle's LB.

Professor submits → gets public score back.
Private score computed but NOT revealed until competition_end().

This forces Professor's Submission Strategist to operate under
the same constraints as a real competition.
"""

import polars as pl
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

from simulator.competition_registry import CompetitionEntry
from simulator.data_splitter import SplitResult
from simulator.scorers import get_scorer


@dataclass
class SubmissionRecord:
    submission_id: int
    path: str
    public_score: float
    private_score: float        # stored but NOT returned to Professor
    day: int
    timestamp: str


@dataclass
class SubmissionResult:
    success: bool
    public_score: Optional[float] = None
    private_score: Optional[float] = None  # ALWAYS None until competition_end
    submission_id: Optional[int] = None
    public_rank_estimate: Optional[float] = None
    submissions_today: int = 0
    submissions_remaining: int = 0
    error: Optional[str] = None


@dataclass
class CompetitionResult:
    slug: str = ""
    best_public_score: Optional[float] = None
    best_private_score: Optional[float] = None
    selected_submission_1: Optional[int] = None
    selected_submission_2: Optional[int] = None
    public_rank_pct: float = 100.0
    private_rank_pct: float = 100.0
    shakeup_positions: float = 0.0
    medal: str = "none"
    total_submissions: int = 0
    days_used: int = 0
    all_submissions: list = field(default_factory=list)
    error: Optional[str] = None


class SimulatedLeaderboard:

    def __init__(
        self,
        entry: CompetitionEntry,
        split: SplitResult,
        daily_limit: int = 5,
    ):
        self.entry = entry
        self.split = split
        self.scorer = get_scorer(entry.metric)
        self.maximize = entry.metric_direction == "maximize"

        # Load hidden labels
        self.public_labels = pl.read_csv(split.public_labels_path)
        self.private_labels = pl.read_csv(split.private_labels_path)
        self.public_ids = set(self.public_labels[entry.id_column].to_list())
        self.private_ids = set(self.private_labels[entry.id_column].to_list())

        # Submission tracking
        self.submissions: List[SubmissionRecord] = []
        self.daily_limit = daily_limit
        self.current_day = 1

    def submit(self, submission_path: str) -> SubmissionResult:
        """
        Score a submission. Returns PUBLIC score only.
        Private score is computed and stored but NOT returned.
        """
        # Validate format
        try:
            sub_df = pl.read_csv(submission_path)
        except Exception as e:
            return SubmissionResult(success=False, error=f"Cannot read submission: {e}")

        required_cols = {self.entry.id_column, self.entry.target_column}
        if not required_cols.issubset(set(sub_df.columns)):
            return SubmissionResult(
                success=False,
                error=f"Missing columns. Required: {required_cols}. Got: {set(sub_df.columns)}"
            )

        # Check daily limit
        today_count = sum(1 for s in self.submissions if s.day == self.current_day)
        if today_count >= self.daily_limit:
            return SubmissionResult(
                success=False,
                error=f"Daily limit reached ({self.daily_limit}/day). "
                      f"Call leaderboard.advance_day() to proceed."
            )

        # Validate row count
        expected_rows = len(self.public_ids) + len(self.private_ids)
        if len(sub_df) != expected_rows:
            return SubmissionResult(
                success=False,
                error=f"Expected {expected_rows} rows, got {len(sub_df)}"
            )

        # Split submission into public and private
        public_preds = sub_df.filter(
            pl.col(self.entry.id_column).is_in(list(self.public_ids))
        )
        private_preds = sub_df.filter(
            pl.col(self.entry.id_column).is_in(list(self.private_ids))
        )

        # Score
        public_score = self.scorer(
            self.public_labels, public_preds,
            self.entry.id_column, self.entry.target_column
        )
        private_score = self.scorer(
            self.private_labels, private_preds,
            self.entry.id_column, self.entry.target_column
        )

        # Record
        record = SubmissionRecord(
            submission_id=len(self.submissions) + 1,
            path=submission_path,
            public_score=public_score,
            private_score=private_score,
            day=self.current_day,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.submissions.append(record)

        return SubmissionResult(
            success=True,
            public_score=public_score,
            private_score=None,  # HIDDEN
            submission_id=record.submission_id,
            public_rank_estimate=self._estimate_rank(public_score),
            submissions_today=today_count + 1,
            submissions_remaining=self.daily_limit - today_count - 1,
        )

    def advance_day(self):
        """Simulate passage of one competition day."""
        self.current_day += 1

    def competition_end(self) -> CompetitionResult:
        """Reveal private scores. Calculate final standing."""
        if not self.submissions:
            return CompetitionResult(error="No submissions made.")

        # Select best public score submission
        if self.maximize:
            best_public = max(self.submissions, key=lambda s: s.public_score)
        else:
            best_public = min(self.submissions, key=lambda s: s.public_score)

        # Best private score from selected submissions
        best_private_score = best_public.private_score

        # Percentile ranks
        public_pct = self._estimate_rank(best_public.public_score)
        private_pct = self._estimate_rank(best_private_score)
        shakeup = private_pct - public_pct  # positive = dropped rank

        return CompetitionResult(
            slug=self.entry.slug,
            best_public_score=best_public.public_score,
            best_private_score=best_private_score,
            selected_submission_1=best_public.submission_id,
            public_rank_pct=public_pct,
            private_rank_pct=private_pct,
            shakeup=shakeup,
            medal=self._compute_medal(best_private_score),
            total_submissions=len(self.submissions),
            days_used=self.current_day,
            all_submissions=[
                {
                    "id": s.submission_id,
                    "public": round(s.public_score, 6),
                    "private": round(s.private_score, 6),
                    "day": s.day,
                }
                for s in self.submissions
            ],
        )

    def _estimate_rank(self, score: float) -> float:
        """Estimate percentile using LB curves. Lower = better rank."""
        percentiles = self.entry.lb_percentiles
        for pct in sorted(percentiles.keys()):
            threshold = percentiles[pct]
            if self.maximize and score >= threshold:
                return float(pct)
            elif not self.maximize and score <= threshold:
                return float(pct)
        return 90.0

    def _compute_medal(self, score: float) -> str:
        if self.maximize:
            if score >= self.entry.silver_threshold: return "silver"
            if score >= self.entry.gold_threshold: return "gold"
            if score >= self.entry.bronze_threshold: return "bronze"
        else:
            if score <= self.entry.silver_threshold: return "silver"
            if score <= self.entry.gold_threshold: return "gold"
            if score <= self.entry.bronze_threshold: return "bronze"
        return "none"
