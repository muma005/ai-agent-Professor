"""
Simulated Leaderboard — Behaves identically to Kaggle's leaderboard.

Professor submits → gets back public score only.
Private score computed and stored but NOT revealed until competition_end().

This forces Professor's Submission Strategist to work under the same
constraints as a real competition:
- Limited submissions per day (default: 5)
- Public score feedback only
- Private score unknown until end
- Shakeup between public and private reveals true performance

Key innovation: The shakeup between public and private LB is WHERE
the real test happens. An 80/20 split with one score tells you nothing
about shakeup resilience.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from simulator.competition_registry import CompetitionEntry
from simulator.data_splitter import SplitResult
from simulator.scorers import get_scorer


@dataclass
class SubmissionRecord:
    """Record of a single submission."""
    
    submission_id: int
    path: str
    public_score: float
    private_score: float  # Stored but not revealed until end
    day: int
    timestamp: str
    
    # Computed predictions for "most different" selection
    predictions_array: Optional[np.ndarray] = None


@dataclass
class SubmissionResult:
    """Result returned to Professor after a submission."""
    
    success: bool
    public_score: Optional[float]
    private_score: Optional[float]  # Always None until competition_end
    submission_id: Optional[int] = None
    public_rank_estimate: Optional[float] = None
    submissions_today: int = 0
    submissions_remaining: int = 0
    error: Optional[str] = None


@dataclass
class CompetitionResult:
    """Result returned when competition ends."""
    
    slug: str
    best_public_score: float
    best_private_score: float
    selected_submission_1: int  # Submission ID
    selected_submission_2: Optional[int]  # Submission ID (or None if only one submission)
    public_rank_pct: float
    private_rank_pct: float
    shakeup_positions: float  # Negative = improved, Positive = dropped
    medal: str  # "gold", "silver", "bronze", "none"
    total_submissions: int
    days_used: int
    all_submissions: List[Dict[str, Any]]
    error: Optional[str] = None


class SimulatedLeaderboard:
    """
    Simulates Kaggle's leaderboard behavior.
    
    Usage:
        lb = SimulatedLeaderboard(entry, split)
        
        # Professor submits multiple times
        result = lb.submit("submission.csv")
        print(f"Public score: {result.public_score}")
        
        # Advance day to reset submission limit
        lb.advance_day()
        
        # At end of competition, reveal private scores
        final = lb.competition_end()
        print(f"Private score: {final.best_private_score}")
        print(f"Medal: {final.medal}")
    """
    
    def __init__(
        self,
        entry: CompetitionEntry,
        split: SplitResult,
        daily_submission_limit: int = 5,
    ):
        self.entry = entry
        self.split = split
        self.scorer = get_scorer(entry.metric, entry.metric_direction)
        
        # Load hidden labels (ONLY this class ever reads these)
        self.public_labels = pl.read_csv(split.public_labels_path)
        self.private_labels = pl.read_csv(split.private_labels_path)
        self.public_ids = set(self.public_labels[entry.id_column].to_list())
        self.private_ids = set(self.private_labels[entry.id_column].to_list())
        
        # Submission tracking
        self.submissions: List[SubmissionRecord] = []
        self.daily_submission_limit = daily_submission_limit
        self.current_day = 1
        
        # Cache for prediction arrays (for "most different" selection)
        self._prediction_cache: Dict[int, np.ndarray] = {}
    
    def submit(self, submission_path: str) -> SubmissionResult:
        """
        Score a submission. Returns public score only.
        Private score is computed and stored but NOT returned.
        
        Args:
            submission_path: Path to submission CSV with columns:
                - ID column (e.g., PassengerId)
                - Target column (e.g., Transported) with predictions
        
        Returns:
            SubmissionResult with public_score, submission_id, rank.
            Private score is None (hidden until competition_end).
        """
        # Validate format
        try:
            sub_df = pl.read_csv(submission_path)
            self._validate_format(sub_df)
        except Exception as e:
            return SubmissionResult(
                success=False,
                public_score=None,
                private_score=None,
                error=f"Invalid submission format: {e}",
            )
        
        # Check daily limit
        today_count = sum(1 for s in self.submissions if s.day == self.current_day)
        if today_count >= self.daily_submission_limit:
            return SubmissionResult(
                success=False,
                error=(
                    f"Daily limit reached ({self.daily_submission_limit}/day). "
                    f"Advance day with leaderboard.advance_day()."
                ),
                public_score=None,
                private_score=None,
            )
        
        # Split submission into public and private portions
        public_preds = sub_df.filter(
            pl.col(self.entry.id_column).is_in(list(self.public_ids))
        )
        private_preds = sub_df.filter(
            pl.col(self.entry.id_column).is_in(list(self.private_ids))
        )
        
        # Validate coverage
        n_public_expected = len(self.public_ids)
        n_private_expected = len(self.private_ids)
        n_public_found = len(public_preds)
        n_private_found = len(private_preds)
        
        if n_public_found < n_public_expected:
            return SubmissionResult(
                success=False,
                error=f"Missing {n_public_expected - n_public_found} public IDs in submission",
                public_score=None,
                private_score=None,
            )
        
        if n_private_found < n_private_expected:
            return SubmissionResult(
                success=False,
                error=f"Missing {n_private_expected - n_private_found} private IDs in submission",
                public_score=None,
                private_score=None,
            )
        
        # Extract predictions
        y_true_public = self.public_labels.join(
            public_preds, on=self.entry.id_column, how="left"
        )[self.entry.target_column].to_numpy()
        y_pred_public = public_preds[self.entry.target_column].to_numpy()
        
        y_true_private = self.private_labels.join(
            private_preds, on=self.entry.id_column, how="left"
        )[self.entry.target_column].to_numpy()
        y_pred_private = private_preds[self.entry.target_column].to_numpy()
        
        # Score against hidden labels
        try:
            public_score = self.scorer(y_true_public, y_pred_public)
            private_score = self.scorer(y_true_private, y_pred_private)
        except Exception as e:
            return SubmissionResult(
                success=False,
                error=f"Scoring error: {e}",
                public_score=None,
                private_score=None,
            )
        
        # Store prediction array for "most different" selection
        pred_array = y_pred_public.copy()
        
        # Store record
        record = SubmissionRecord(
            submission_id=len(self.submissions) + 1,
            path=str(submission_path),
            public_score=public_score,
            private_score=private_score,  # Stored but not revealed
            day=self.current_day,
            timestamp=datetime.utcnow().isoformat(),
            predictions_array=pred_array,
        )
        self.submissions.append(record)
        
        # Return public score only (private hidden)
        return SubmissionResult(
            success=True,
            public_score=public_score,
            private_score=None,  # HIDDEN — same as real Kaggle
            submission_id=record.submission_id,
            public_rank_estimate=self._estimate_rank(public_score, "public"),
            submissions_today=today_count + 1,
            submissions_remaining=self.daily_submission_limit - today_count - 1,
        )
    
    def advance_day(self) -> None:
        """Simulate passage of time. Resets daily submission counter."""
        self.current_day += 1
    
    def competition_end(self) -> CompetitionResult:
        """
        Reveal private scores. Select final submissions.
        
        Kaggle rules: competitor selects 2 final submissions.
        Professor selects: (1) best public score, (2) most different model.
        Private score of THOSE selections determines final rank.
        
        This is where shakeup happens.
        """
        if not self.submissions:
            return CompetitionResult(
                slug=self.entry.slug,
                best_public_score=0.0,
                best_private_score=0.0,
                selected_submission_1=0,
                selected_submission_2=None,
                public_rank_pct=100.0,
                private_rank_pct=100.0,
                shakeup_positions=0.0,
                medal="none",
                total_submissions=0,
                days_used=0,
                all_submissions=[],
                error="No submissions made.",
            )
        
        # Professor's selection strategy (same as Submission Strategist)
        maximize = self.entry.metric_direction == "maximize"
        
        # Find best public score submission
        if maximize:
            best_public = max(self.submissions, key=lambda s: s.public_score)
        else:
            best_public = min(self.submissions, key=lambda s: s.public_score)
        
        # Find most different submission (lowest correlation to best_public)
        most_different = self._find_most_different(best_public)
        
        # Reveal private scores
        final_1 = best_public.private_score
        final_2 = most_different.private_score if most_different else None
        
        # Best private score between the two selections
        if maximize:
            best_private = max(final_1, final_2) if final_2 else final_1
        else:
            best_private = min(final_1, final_2) if final_2 else final_1
        
        # Shakeup analysis
        public_rank = self._estimate_rank(best_public.public_score, "public")
        private_rank = self._estimate_rank(best_private, "private")
        shakeup = private_rank - public_rank  # Positive = dropped in rank
        
        return CompetitionResult(
            slug=self.entry.slug,
            best_public_score=best_public.public_score,
            best_private_score=best_private,
            selected_submission_1=best_public.submission_id,
            selected_submission_2=most_different.submission_id if most_different else None,
            public_rank_pct=public_rank,
            private_rank_pct=private_rank,
            shakeup_positions=shakeup,
            medal=self.entry.compute_medal(best_private),
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
    
    def _validate_format(self, sub_df: pl.DataFrame) -> None:
        """Validate submission format matches expected schema."""
        required_cols = {self.entry.id_column, self.entry.target_column}
        actual_cols = set(sub_df.columns)
        
        missing = required_cols - actual_cols
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Check for nulls
        n_nulls = sub_df[self.entry.target_column].null_count()
        if n_nulls > 0:
            raise ValueError(f"{n_nulls} null predictions")
    
    def _estimate_rank(self, score: float, lb_type: str = "public") -> float:
        """
        Estimate percentile rank using the competition's LB curve.
        Returns percentage (lower = better). 5.0 = top 5%.
        """
        return self.entry.estimate_percentile(score)
    
    def _find_most_different(
        self,
        reference: SubmissionRecord,
    ) -> Optional[SubmissionRecord]:
        """
        Find submission with lowest prediction correlation to reference.
        
        This simulates selecting a "diverse" second submission,
        which is a common Kaggle strategy to reduce shakeup risk.
        """
        if len(self.submissions) < 2:
            return None
        
        ref_preds = reference.predictions_array
        if ref_preds is None:
            return None
        
        best_candidate = None
        best_diff = -2  # Correlation ranges from -1 to 1
        
        for sub in self.submissions:
            if sub.submission_id == reference.submission_id:
                continue
            
            if sub.predictions_array is None:
                continue
            
            # Compute correlation
            corr = np.corrcoef(ref_preds, sub.predictions_array)[0, 1]
            if np.isnan(corr):
                continue
            
            # Lower correlation = more different
            if corr < best_diff:
                best_diff = corr
                best_candidate = sub
        
        return best_candidate
    
    def get_submission_history(self) -> List[Dict[str, Any]]:
        """Get full submission history (for debugging/analysis)."""
        return [
            {
                "id": s.submission_id,
                "public_score": s.public_score,
                "private_score": s.private_score,  # Only for analysis, not shown to Professor
                "day": s.day,
                "timestamp": s.timestamp,
            }
            for s in self.submissions
        ]
