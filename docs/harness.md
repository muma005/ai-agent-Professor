# Private Leaderboard Simulator — Complete Architecture Specification

## "You Can't Improve What You Can't Measure"

---

## 1. The Problem — Why This Must Exist Before Anything Else

Professor currently has no way to know if it's actually good. The only
measurement path is: enter a live Kaggle competition, wait 2-12 weeks,
see the final result. That feedback loop is useless for development.
You can't iterate on a 6-week cycle.

The v1 Historical Harness does a basic 80/20 split and compares against
known LB curves. That's a sketch, not a measurement system. It has
three critical flaws:

**Flaw 1 — No public/private LB split.** Real competitions have a public
LB (30% of test) and a private LB (70% of test). Competitors overfit to
public. The shakeup between public and private is WHERE the real test
happens. An 80/20 split with one score tells you nothing about shakeup
resilience.

**Flaw 2 — No submission simulation.** Real competitions give you 2-5
submissions per day. You get a public score back. You adapt. This
feedback loop is a fundamental part of the competition experience. The
harness runs once and scores once. Professor never learns to use
submissions strategically.

**Flaw 3 — No percentile calibration.** Saying "Professor got 0.812
accuracy" means nothing without context. Was the gold threshold 0.815
or 0.790? The harness compares against "known LB curves" but doesn't
have a systematic percentile database for calibration.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 COMPETITION SIMULATOR                        │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │  Competition  │   │    Data      │   │   Simulated      │ │
│  │  Registry     │──→│  Splitter    │──→│   Leaderboard    │ │
│  │  (metadata)   │   │  (holdout)   │   │   (scoring)      │ │
│  └──────────────┘   └──────────────┘   └──────────────────┘ │
│         │                  │                    │             │
│         │                  │                    ▼             │
│         │                  │            ┌──────────────────┐ │
│         │                  │            │   Percentile     │ │
│         │                  │            │   Calibrator     │ │
│         │                  │            │   (medal calc)   │ │
│         │                  │            └──────────────────┘ │
│         │                  │                    │             │
│         ▼                  ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Benchmark Report Generator                  │ │
│  │    (aggregate results across N competitions)             │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Four components. Each is small, focused, and independently testable.**

---

## 3. Component 1 — Competition Registry

**Purpose:** A structured database of historical competitions with all
metadata needed to simulate them faithfully.

**File:** `simulator/competition_registry.py`

**Schema per competition:**

```python
@dataclass
class CompetitionEntry:
    # Identity
    slug:               str       # "spaceship-titanic"
    title:              str       # "Spaceship Titanic"
    kaggle_id:          str       # competition ID for API download

    # Task
    task_type:          str       # "binary"|"multiclass"|"regression"|"multilabel"
    target_column:      str       # "Transported"
    id_column:          str       # "PassengerId"
    metric:             str       # "accuracy"
    metric_direction:   str       # "maximize"|"minimize"

    # LB Percentile Curve (calibration data)
    # Scraped from actual Kaggle leaderboard after competition ends
    lb_percentiles:     dict      # {10: 0.810, 25: 0.795, 50: 0.780, 75: 0.760, 90: 0.740}
    gold_threshold:     float     # score needed for top ~10% (gold)
    silver_threshold:   float     # score needed for top ~5%
    bronze_threshold:   float     # score needed for top ~25%
    total_teams:        int       # for percentile calculation

    # Split Configuration
    split_strategy:     str       # "stratified"|"temporal"|"group"
    split_column:       str       # column used for temporal/group splits (None if stratified)
    test_ratio:         float     # 0.40 — fraction held out as simulated test
    public_ratio:       float     # 0.30 — fraction of test that forms public LB
    random_seed:        int       # 42 — deterministic, reproducible

    # Data Source
    download_method:    str       # "kaggle_api"|"manual"|"cached"
    cached_path:        str       # local path if already downloaded
    requires_join:      bool      # True if competition has multiple data files
    join_instructions:  str       # SQL or description of how to join files

    # Domain
    primary_domain:     str       # "transport", "healthcare", "finance", etc.
    sub_domain:         str       # "passenger survival prediction"
```

**Initial Registry — 10 competitions covering major types:**

```python
REGISTRY = [
    # ── Binary Classification ──
    CompetitionEntry(
        slug="spaceship-titanic",
        task_type="binary", target_column="Transported",
        metric="accuracy", metric_direction="maximize",
        lb_percentiles={10: 0.810, 25: 0.795, 50: 0.780, 75: 0.760},
        gold_threshold=0.810, total_teams=2500,
        split_strategy="stratified", test_ratio=0.40, public_ratio=0.30,
        primary_domain="transport",
    ),
    CompetitionEntry(
        slug="titanic",
        task_type="binary", target_column="Survived",
        metric="accuracy", metric_direction="maximize",
        lb_percentiles={10: 0.800, 25: 0.790, 50: 0.775, 75: 0.755},
        gold_threshold=0.800, total_teams=15000,
        split_strategy="stratified", test_ratio=0.40, public_ratio=0.30,
        primary_domain="transport",
    ),
    CompetitionEntry(
        slug="playground-series-s4e8",  # example binary with imbalance
        task_type="binary", target_column="class",
        metric="auc", metric_direction="maximize",
        lb_percentiles={10: 0.890, 25: 0.875, 50: 0.860, 75: 0.840},
        gold_threshold=0.890, total_teams=1800,
        split_strategy="stratified", test_ratio=0.40, public_ratio=0.30,
        primary_domain="general",
    ),

    # ── Regression ──
    CompetitionEntry(
        slug="house-prices-advanced-regression-techniques",
        task_type="regression", target_column="SalePrice",
        metric="rmsle", metric_direction="minimize",
        lb_percentiles={10: 0.120, 25: 0.130, 50: 0.145, 75: 0.165},
        gold_threshold=0.120, total_teams=5000,
        split_strategy="stratified", test_ratio=0.40, public_ratio=0.30,
        primary_domain="real_estate",
    ),

    # ── Multiclass ──
    CompetitionEntry(
        slug="playground-series-s4e9",  # example multiclass
        task_type="multiclass", target_column="Status",
        metric="accuracy", metric_direction="maximize",
        lb_percentiles={10: 0.850, 25: 0.830, 50: 0.810, 75: 0.785},
        gold_threshold=0.850, total_teams=1200,
        split_strategy="stratified", test_ratio=0.40, public_ratio=0.30,
        primary_domain="healthcare",
    ),

    # ── Imbalanced Binary ──
    CompetitionEntry(
        slug="icr-identify-age-related-conditions",
        task_type="binary", target_column="Class",
        metric="balanced_log_loss", metric_direction="minimize",
        lb_percentiles={10: 0.20, 25: 0.28, 50: 0.35, 75: 0.45},
        gold_threshold=0.20, total_teams=6400,
        split_strategy="stratified", test_ratio=0.40, public_ratio=0.30,
        primary_domain="healthcare",
    ),
]
```

**Registry growth strategy:**
- Start with 10 competitions covering binary, multiclass, regression,
  imbalanced, temporal, different domains
- Add 5 per month as Professor runs benchmarks
- After 50+: statistical confidence in aggregate metrics
- LB percentile curves scraped from actual Kaggle leaderboards after
  competition closes (public data)

**Why 40% holdout (not 20%):**
Real Kaggle competitions typically have test sets equal to or larger than
training sets. A 20% holdout from a dataset where the original split was
50/50 means Professor trains on 80% of the original train — MORE data
than real competitors had. That inflates simulated scores and gives false
confidence. 40% holdout means Professor trains on 60% of the available
data, which is closer to the real constraint. The 40% is then split
30/70 into public/private, matching Kaggle's typical ratio.

---

## 4. Component 2 — Data Splitter

**Purpose:** Takes competition data and produces three deterministic,
reproducible partitions: train (60%), public test (12%), private test (28%).

**File:** `simulator/data_splitter.py`

**Critical design decisions:**

### Split Strategies (matched to competition type)

```python
def split_competition_data(
    data_path: str,
    entry: CompetitionEntry,
) -> SplitResult:
    """
    Produces 3 partitions from a single dataset.
    
    IMPORTANT: The split must be DETERMINISTIC and REPRODUCIBLE.
    Same data + same seed = exact same split every time.
    This is essential for regression testing — if Professor's score
    changes between runs on the same split, the change is real,
    not split variance.
    """
    
    df = pl.read_csv(data_path)
    target = entry.target_column
    seed = entry.random_seed  # always 42
    
    # ── Step 1: Split into train (60%) and test (40%) ──
    if entry.split_strategy == "stratified":
        # Stratified by target — preserves class distribution
        train_df, test_df = stratified_split(
            df, target=target, test_size=entry.test_ratio, seed=seed
        )
    
    elif entry.split_strategy == "temporal":
        # Sort by time column, take last 40% as test
        # This simulates competitions with temporal test sets
        df_sorted = df.sort(entry.split_column)
        split_idx = int(len(df_sorted) * (1 - entry.test_ratio))
        train_df = df_sorted[:split_idx]
        test_df = df_sorted[split_idx:]
    
    elif entry.split_strategy == "group":
        # Split by group — no group appears in both train and test
        # This simulates competitions with GroupKFold requirements
        groups = df[entry.split_column].unique()
        group_train, group_test = stratified_group_split(
            groups, test_size=entry.test_ratio, seed=seed
        )
        train_df = df.filter(pl.col(entry.split_column).is_in(group_train))
        test_df = df.filter(pl.col(entry.split_column).is_in(group_test))
    
    # ── Step 2: Split test into public (30%) and private (70%) ──
    public_test, private_test = stratified_split(
        test_df, target=target, test_size=0.70, seed=seed + 1
    )
    
    # ── Step 3: Save partitions ──
    # Train: what Professor receives (NO target in test files)
    # Public test: features only (target held for scoring)
    # Private test: features only (target held for scoring)
    
    train_path = f"simulator/data/{entry.slug}/train.csv"
    test_features_path = f"simulator/data/{entry.slug}/test.csv"
    public_labels_path = f"simulator/data/{entry.slug}/.public_labels.csv"  # hidden
    private_labels_path = f"simulator/data/{entry.slug}/.private_labels.csv"  # hidden
    sample_submission_path = f"simulator/data/{entry.slug}/sample_submission.csv"
    
    # Combine public + private into one test file (features only)
    # Professor sees this as the test set — same as real Kaggle
    full_test = pl.concat([public_test, private_test])
    full_test_features = full_test.drop(target)
    
    # Save
    train_df.write_csv(train_path)
    full_test_features.write_csv(test_features_path)
    public_test[[entry.id_column, target]].write_csv(public_labels_path)
    private_test[[entry.id_column, target]].write_csv(private_labels_path)
    
    # Sample submission (all zeros or median, same as Kaggle provides)
    sample = full_test_features[[entry.id_column]].with_columns(
        pl.lit(_default_prediction(entry)).alias(target)
    )
    sample.write_csv(sample_submission_path)
    
    return SplitResult(
        train_path=train_path,
        test_path=test_features_path,
        public_labels_path=public_labels_path,
        private_labels_path=private_labels_path,
        sample_submission_path=sample_submission_path,
        n_train=len(train_df),
        n_public=len(public_test),
        n_private=len(private_test),
        split_metadata={
            "strategy": entry.split_strategy,
            "seed": seed,
            "train_ratio": 1 - entry.test_ratio,
            "public_ratio": entry.public_ratio,
            "target_distribution_train": _target_stats(train_df, target),
            "target_distribution_public": _target_stats(public_test, target),
            "target_distribution_private": _target_stats(private_test, target),
        }
    )
```

### Data Isolation Rules (non-negotiable)

```
RULE 1: Professor NEVER sees private labels.
    .private_labels.csv is dotfile (hidden).
    Only the Simulated Leaderboard component reads it.
    If ANY agent code imports or reads this file → simulation is invalid.

RULE 2: Professor NEVER sees public labels directly.
    Public labels are used only by the Simulated Leaderboard to return
    a "public LB score" after each submission. Professor receives only
    the score number, never the actual labels.

RULE 3: The test.csv Professor receives contains BOTH public and private
    rows, shuffled together. Professor cannot distinguish which rows are
    public vs private. Same as real Kaggle.

RULE 4: The split is deterministic. Same seed = same split = same rows.
    This means Professor's score on the same competition is directly
    comparable across code changes. Score change = real improvement, not
    split variance.

RULE 5: Train target distribution is validated against full-data distribution.
    If the split creates a train set with significantly different target
    distribution (KS-test p < 0.01), the split is re-done with seed + 100.
    Maximum 5 re-seeds before raising an error.
```

---

## 5. Component 3 — Simulated Leaderboard

**Purpose:** Accept submission files from Professor, score them against
hidden labels, return public LB scores, and track private LB scores
(revealed only at "competition end").

**File:** `simulator/leaderboard.py`

**This is the core innovation.** It behaves exactly like the Kaggle
leaderboard API:

```python
class SimulatedLeaderboard:
    """
    Behaves identically to Kaggle's leaderboard from Professor's perspective.
    
    Professor submits → gets back public score only.
    Private score computed and stored but NOT revealed until
    competition_end() is called.
    
    This forces Professor's Submission Strategist to work under the
    same constraints as a real competition: limited submissions,
    public score feedback only, private score unknown.
    """
    
    def __init__(self, entry: CompetitionEntry, split: SplitResult):
        self.entry = entry
        self.split = split
        self.scorer = _build_scorer(entry.metric, entry.metric_direction)
        
        # Load hidden labels (ONLY this class ever reads these)
        self.public_labels = pl.read_csv(split.public_labels_path)
        self.private_labels = pl.read_csv(split.private_labels_path)
        self.public_ids = set(self.public_labels[entry.id_column].to_list())
        self.private_ids = set(self.private_labels[entry.id_column].to_list())
        
        # Submission tracking
        self.submissions: list[SubmissionRecord] = []
        self.daily_submission_limit = 5
        self.current_day = 1
        
    def submit(self, submission_path: str) -> SubmissionResult:
        """
        Score a submission. Returns public score only.
        Private score is computed and stored but NOT returned.
        
        Returns:
            SubmissionResult with public_score, submission_id, rank.
            Private score is None (hidden until competition_end).
        """
        # Validate format
        sub_df = pl.read_csv(submission_path)
        self._validate_format(sub_df)
        
        # Check daily limit
        today_count = sum(1 for s in self.submissions if s.day == self.current_day)
        if today_count >= self.daily_submission_limit:
            return SubmissionResult(
                success=False,
                error=f"Daily limit reached ({self.daily_submission_limit}/day). "
                      f"Advance day with leaderboard.advance_day().",
                public_score=None,
                private_score=None,  # always None until end
            )
        
        # Split submission into public and private portions
        public_preds = sub_df.filter(
            pl.col(self.entry.id_column).is_in(list(self.public_ids))
        )
        private_preds = sub_df.filter(
            pl.col(self.entry.id_column).is_in(list(self.private_ids))
        )
        
        # Score against hidden labels
        public_score = self.scorer(
            self.public_labels, public_preds,
            self.entry.id_column, self.entry.target_column
        )
        private_score = self.scorer(
            self.private_labels, private_preds,
            self.entry.id_column, self.entry.target_column
        )
        
        # Store record
        record = SubmissionRecord(
            submission_id=len(self.submissions) + 1,
            path=submission_path,
            public_score=public_score,
            private_score=private_score,  # stored but not revealed
            day=self.current_day,
            timestamp=datetime.utcnow().isoformat(),
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
    
    def advance_day(self):
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
            return CompetitionResult(error="No submissions made.")
        
        # Professor's selection strategy (same as Submission Strategist)
        best_public = max(self.submissions, key=lambda s: s.public_score
                         if self.entry.metric_direction == "maximize"
                         else -s.public_score)
        
        # For "most different" — find submission with lowest prediction
        # correlation to best_public submission
        most_different = self._find_most_different(best_public)
        
        # Reveal private scores
        final_1 = best_public.private_score
        final_2 = most_different.private_score if most_different else None
        
        # Best private score between the two selections
        if self.entry.metric_direction == "maximize":
            best_private = max(final_1, final_2) if final_2 else final_1
        else:
            best_private = min(final_1, final_2) if final_2 else final_1
        
        # Shakeup analysis
        public_rank = self._estimate_rank(best_public.public_score, "public")
        private_rank = self._estimate_rank(best_private, "private")
        shakeup = private_rank - public_rank  # positive = dropped in rank
        
        return CompetitionResult(
            slug=self.entry.slug,
            best_public_score=best_public.public_score,
            best_private_score=best_private,
            selected_submission_1=best_public.submission_id,
            selected_submission_2=most_different.submission_id if most_different else None,
            public_rank_pct=public_rank,
            private_rank_pct=private_rank,
            shakeup_positions=shakeup,
            medal=self._compute_medal(best_private),
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
    
    def _estimate_rank(self, score: float, lb_type: str) -> float:
        """
        Estimate percentile rank using the competition's LB curve.
        Returns percentage (lower = better). 5.0 = top 5%.
        """
        percentiles = self.entry.lb_percentiles  # {10: 0.810, 25: 0.795, ...}
        maximize = self.entry.metric_direction == "maximize"
        
        for pct in sorted(percentiles.keys()):
            threshold = percentiles[pct]
            if maximize and score >= threshold:
                return float(pct)
            elif not maximize and score <= threshold:
                return float(pct)
        
        return 90.0  # default: bottom half
    
    def _compute_medal(self, private_score: float) -> str:
        maximize = self.entry.metric_direction == "maximize"
        if maximize:
            if private_score >= self.entry.gold_threshold:
                return "gold"
            elif private_score >= self.entry.silver_threshold:
                return "silver"
            elif private_score >= self.entry.bronze_threshold:
                return "bronze"
        else:
            if private_score <= self.entry.gold_threshold:
                return "gold"
            elif private_score <= self.entry.silver_threshold:
                return "silver"
            elif private_score <= self.entry.bronze_threshold:
                return "bronze"
        return "none"
```

### Scorer Implementation

```python
def _build_scorer(metric: str, direction: str) -> Callable:
    """
    Build a scorer function that matches the exact Kaggle evaluation.
    
    CRITICAL: The scorer must produce IDENTICAL results to Kaggle's
    evaluation. Even a rounding difference can shift percentile rank.
    Use the same implementations Kaggle uses (sklearn for standard
    metrics, custom for Kaggle-specific ones like QWK).
    """
    SCORERS = {
        "accuracy":           sklearn.metrics.accuracy_score,
        "auc":                sklearn.metrics.roc_auc_score,
        "f1":                 lambda y, p: sklearn.metrics.f1_score(y, p, average="binary"),
        "macro_f1":           lambda y, p: sklearn.metrics.f1_score(y, p, average="macro"),
        "log_loss":           sklearn.metrics.log_loss,
        "balanced_log_loss":  _balanced_log_loss,  # custom implementation
        "rmse":               lambda y, p: np.sqrt(sklearn.metrics.mean_squared_error(y, p)),
        "rmsle":              _rmsle,
        "mae":                sklearn.metrics.mean_absolute_error,
        "qwk":                _quadratic_weighted_kappa,
        "map_at_k":           _mean_average_precision_at_k,
    }
    
    if metric not in SCORERS:
        raise ValueError(
            f"Unknown metric '{metric}'. Add it to SCORERS in leaderboard.py. "
            f"Available: {list(SCORERS.keys())}"
        )
    
    return SCORERS[metric]
```

---

## 6. Component 4 — Benchmark Report Generator

**Purpose:** Aggregate results across multiple competitions into a single
performance report that shows Professor's overall capability.

**File:** `simulator/report_generator.py`

**Report schema — `benchmark_report.json`:**

```json
{
  "run_id": "benchmark_2026_04_01_v2.3",
  "professor_version": "v2.3",
  "timestamp": "2026-04-01T18:00:00Z",
  "n_competitions": 10,

  "aggregate_metrics": {
    "median_percentile": 12.5,
    "mean_percentile": 15.3,
    "gold_rate": 0.30,
    "silver_rate": 0.10,
    "bronze_rate": 0.40,
    "medal_rate": 0.80,
    "no_medal_rate": 0.20,
    "catastrophic_failure_rate": 0.00,
    "mean_shakeup": -2.1,
    "mean_cv_private_gap": 0.008
  },

  "per_competition": [
    {
      "slug": "spaceship-titanic",
      "task_type": "binary",
      "domain": "transport",
      "metric": "accuracy",
      "cv_score": 0.8123,
      "public_score": 0.8067,
      "private_score": 0.8089,
      "cv_public_gap": 0.0056,
      "cv_private_gap": 0.0034,
      "public_percentile": 12.0,
      "private_percentile": 10.0,
      "shakeup": -2.0,
      "medal": "gold",
      "total_submissions": 8,
      "runtime_seconds": 342,
      "winning_model": "lgbm",
      "n_features_final": 47,
      "domain_features_used": 3,
      "domain_features_helped": 2
    }
  ],

  "version_comparison": {
    "vs_previous": {
      "previous_version": "v2.2",
      "percentile_delta": -1.8,
      "gold_rate_delta": +0.10,
      "improved_competitions": ["spaceship-titanic", "house-prices"],
      "degraded_competitions": [],
      "unchanged_competitions": ["titanic"]
    }
  },

  "component_attribution": {
    "shift_detector_triggered": 2,
    "shift_detector_helped": 2,
    "domain_features_generated": 15,
    "domain_features_kept": 8,
    "creative_features_generated": 23,
    "creative_features_kept": 6,
    "postprocess_improved": 7,
    "postprocess_delta_mean": 0.004,
    "pseudo_label_applied": 3,
    "pseudo_label_helped": 2,
    "pseudo_label_reverted": 1,
    "critic_critical_count": 1,
    "critic_replan_count": 1
  }
}
```

### Version-Over-Version Regression Detection

```python
def compare_benchmark_runs(current: BenchmarkReport, previous: BenchmarkReport):
    """
    Compare two benchmark runs to detect regressions.
    
    REGRESSION RULE: If Professor v2.3 scores WORSE than v2.2 on ANY
    competition by more than 1 percentile point, it is flagged as a
    regression. The specific competition and score delta are reported.
    
    This is the v2 regression protection equivalent for the simulator.
    """
    regressions = []
    for comp in current.per_competition:
        prev = _find_competition(previous, comp.slug)
        if prev is None:
            continue
        
        delta = comp.private_percentile - prev.private_percentile
        if delta > 1.0:  # got worse by more than 1 percentile
            regressions.append({
                "competition": comp.slug,
                "previous_percentile": prev.private_percentile,
                "current_percentile": comp.private_percentile,
                "delta": delta,
                "severity": "CRITICAL" if delta > 5.0 else "WARNING",
            })
    
    return regressions
```

---

## 7. Integration — How Professor Interacts With the Simulator

### Mode 1: Full Benchmark Run (development)

Used during v2 build to validate that changes improve overall performance.

```bash
# Run Professor against all 10 registry competitions
professor benchmark --all

# Run against a specific competition
professor benchmark --competition spaceship-titanic

# Compare against previous benchmark
professor benchmark --all --compare-with benchmark_v2.2.json
```

**Execution flow:**
```
For each competition in registry:
    1. Download data (if not cached)
    2. Split into train / public test / private test (deterministic)
    3. Create SimulatedLeaderboard instance
    4. Give Professor: train.csv, test.csv (features only), sample_submission.csv
    5. Professor runs full pipeline (identical to real competition)
    6. Professor produces submission.csv
    7. Professor submits to SimulatedLeaderboard → gets public score
    8. Professor may iterate (advance_day, re-submit) up to configured limit
    9. Call competition_end() → reveal private scores, compute medal
    10. Record results
Generate aggregate benchmark_report.json
```

**Time budget per competition:** 
Configure via `BENCHMARK_TIME_LIMIT_SECONDS` (default: 600 = 10 minutes).
Professor runs in "fast mode" for benchmarking:
- Optuna trials capped at 30 (not 200)
- Null importance shuffles capped at 5 (not 50)
- Single submission (no multi-day iteration)
- No forum scraping (use cached intel or skip)

This means a full 10-competition benchmark runs in ~2 hours.
Good enough for daily regression checks during the build.

### Mode 2: Deep Run (performance testing)

Full-fidelity simulation with no shortcuts.

```bash
professor benchmark --competition spaceship-titanic --deep
```

- Full Optuna budget (200 trials)
- Full null importance (50 shuffles)
- Multi-day simulation (Professor submits, gets public score, iterates)
- Forum scraping enabled
- Runtime: 30-60 minutes per competition

Used weekly or before phase gates, not daily.

### Mode 3: A/B Test (feature validation)

Test whether a specific component actually helps.

```bash
# Run with and without creative hypothesis engine
professor benchmark --competition spaceship-titanic \
    --ab-test creative_hypothesis_engine
```

**Execution:**
1. Run full pipeline WITH the component → score A
2. Run full pipeline WITHOUT the component (disabled in config) → score B
3. Compare: Wilcoxon across all competitions if running --all
4. Report: component helped / hurt / neutral with statistical significance

This is how you validate that the Domain Research Engine actually improves
scores, not just adds complexity. Every v2 component should be A/B tested
against the benchmark before being accepted.

---

## 8. File Structure

```
simulator/
├── __init__.py
├── competition_registry.py      # CompetitionEntry dataclass + REGISTRY list
├── data_splitter.py             # Split logic (stratified, temporal, group)
├── leaderboard.py               # SimulatedLeaderboard class
├── scorers.py                   # All metric implementations
├── report_generator.py          # Aggregate benchmark reports
├── cli.py                       # CLI: professor benchmark [options]
├── data/                        # Downloaded + split competition data
│   ├── spaceship-titanic/
│   │   ├── train.csv            # What Professor sees
│   │   ├── test.csv             # Features only (public + private rows)
│   │   ├── sample_submission.csv
│   │   ├── .public_labels.csv   # HIDDEN — only leaderboard.py reads
│   │   └── .private_labels.csv  # HIDDEN — only leaderboard.py reads
│   └── titanic/
│       └── ...
├── results/                     # Benchmark reports
│   ├── benchmark_v2.0.json
│   ├── benchmark_v2.1.json
│   └── benchmark_v2.3.json
└── tests/
    ├── test_splitter.py          # Split determinism, distribution preservation
    ├── test_leaderboard.py       # Scoring accuracy, submission limits
    └── test_scorers.py           # Each scorer matches sklearn exactly
```

---

## 9. Failure Modes and Mitigations

### Failure 1 — Split leaks information
**Risk:** Stratified split on small datasets can create splits where
certain feature combinations only appear in train or only in test,
making the simulation unrepresentative.
**Mitigation:** After splitting, validate that no categorical level
appears only in test (would cause model failure). Validate KS-test on
all numeric features between train and test (p > 0.01 required).
Re-seed if validation fails.

### Failure 2 — Scorer implementation differs from Kaggle
**Risk:** Kaggle uses specific implementations (e.g., their balanced
log loss has a specific epsilon and clipping). A slight difference
means simulated scores don't match real scores.
**Mitigation:** For each competition in the registry, verify the scorer
against known Kaggle submission scores. If we have a previous
submission with a known Kaggle score, run the scorer against it and
verify within 1e-5 tolerance. Store these verification pairs.

### Failure 3 — LB percentile curves are inaccurate
**Risk:** The percentile thresholds in the registry are estimates.
If they're wrong, medal calculations are wrong.
**Mitigation:** Scrape actual Kaggle public leaderboards after
competitions close. Store exact score-to-rank mappings. Update
registry as better data becomes available. Flag low-confidence
percentile estimates in the benchmark report.

### Failure 4 — Benchmark overhead slows development
**Risk:** If running the benchmark takes 4 hours, developers skip
it. The measurement system becomes shelfware.
**Mitigation:** Fast mode (10 min/competition, 2 hours total for 10)
runs daily. Deep mode (1 hour/competition) runs weekly. A/B tests
run on demand for specific components. The fast mode is fast enough
to run as part of the phase gate check, not a separate ceremony.

### Failure 5 — Professor optimises for the benchmark
**Risk:** After running the same 10 competitions 50 times, Professor's
memory system may overfit to those specific competitions. Performance
on the benchmark improves but real-world performance doesn't.
**Mitigation:** Expand the registry continuously (target: 50+).
Rotate "holdout competitions" that are never used for development,
only for final validation. Split registry into "dev" (8 comps for
daily use) and "holdout" (2 comps used only at phase gates).

---

## 10. Contract Tests

**File:** `tests/contracts/test_simulator_contract.py` (IMMUTABLE)

```python
class TestDataSplitter:
    def test_split_is_deterministic(self):
        # Same data + same seed = exact same split
        split1 = split_competition_data(FIXTURE_PATH, ENTRY)
        split2 = split_competition_data(FIXTURE_PATH, ENTRY)
        assert split1.train_hash == split2.train_hash
        assert split1.public_hash == split2.public_hash
        assert split1.private_hash == split2.private_hash
    
    def test_no_row_appears_in_multiple_partitions(self):
        split = split_competition_data(FIXTURE_PATH, ENTRY)
        train_ids = set(pl.read_csv(split.train_path)[ID_COL])
        test_ids_public = set(pl.read_csv(split.public_labels_path)[ID_COL])
        test_ids_private = set(pl.read_csv(split.private_labels_path)[ID_COL])
        assert len(train_ids & test_ids_public) == 0
        assert len(train_ids & test_ids_private) == 0
        assert len(test_ids_public & test_ids_private) == 0
    
    def test_target_distribution_preserved(self):
        # KS test: train target distribution ≈ full data distribution
        ...
    
    def test_test_file_has_no_target_column(self):
        split = split_competition_data(FIXTURE_PATH, ENTRY)
        test_df = pl.read_csv(split.test_path)
        assert ENTRY.target_column not in test_df.columns
    
    def test_ratios_correct(self):
        # train ≈ 60%, public ≈ 12%, private ≈ 28%
        ...

class TestSimulatedLeaderboard:
    def test_submit_returns_public_score_only(self):
        result = lb.submit(SUBMISSION_PATH)
        assert result.public_score is not None
        assert result.private_score is None  # HIDDEN
    
    def test_daily_limit_enforced(self):
        for i in range(5):
            lb.submit(SUBMISSION_PATH)
        result = lb.submit(SUBMISSION_PATH)
        assert result.success is False
        assert "limit" in result.error.lower()
    
    def test_advance_day_resets_limit(self):
        for i in range(5):
            lb.submit(SUBMISSION_PATH)
        lb.advance_day()
        result = lb.submit(SUBMISSION_PATH)
        assert result.success is True
    
    def test_competition_end_reveals_private(self):
        lb.submit(SUBMISSION_PATH)
        result = lb.competition_end()
        assert result.best_private_score is not None
        assert result.medal in ["gold", "silver", "bronze", "none"]
    
    def test_perfect_submission_gets_perfect_score(self):
        # Submit actual labels as predictions → score = perfect
        ...
    
    def test_scorer_matches_sklearn(self):
        # Every scorer produces identical output to sklearn reference
        ...

class TestBenchmarkReport:
    def test_version_comparison_detects_regression(self):
        ...
    
    def test_component_attribution_populated(self):
        ...
```

---

## 11. Build Integration — Zero Slowdown

**The simulator is built ONCE and then used as infrastructure.**

Build effort: 2-3 days (part of v2-0 or pre-v2 setup)
- Day 1: competition_registry.py + data_splitter.py + tests
- Day 2: leaderboard.py + scorers.py + tests
- Day 3: report_generator.py + cli.py + first 5 competitions cached

**After build, usage is:**
- `professor benchmark --all` runs in 2 hours (fast mode)
- Runs nightly on CI, or manually before phase gates
- A/B test specific components on demand
- No impact on Professor's pipeline code — the simulator wraps
  Professor, Professor doesn't know it's being simulated

**The simulator does NOT modify Professor's code.**
Professor receives train.csv, test.csv, and sample_submission.csv.
It runs identically to how it would on a real competition.
The simulator wraps the outside — it provides the data and scores
the output. Professor's agents, state, pipeline are untouched.

---

## 12. When to Use Each Mode

| Situation | Mode | Time | Purpose |
|-----------|------|------|---------|
| Daily during v2 build | Fast benchmark (10 comps) | 2 hours | Catch regressions |
| Before phase gate | Deep benchmark (3 comps) | 2-3 hours | Validate gate readiness |
| After adding new component | A/B test (10 comps) | 4 hours | Prove component helps |
| Weekly progress check | Full deep benchmark | 8-10 hours | Accurate performance picture |
| Release candidate | Holdout benchmark (2 reserved comps) | 1-2 hours | Unbiased final validation |