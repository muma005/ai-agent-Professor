"""
Benchmark Report Generator — Aggregate results across multiple competitions.

Produces a comprehensive benchmark report showing Professor's overall
capability, including:
- Aggregate metrics (median percentile, medal rates, shakeup)
- Per-competition breakdown
- Version-over-version regression detection
- Component attribution (which features helped/hurt)

Report schema matches the spec in harness.md section 6.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from simulator.competition_registry import CompetitionEntry, REGISTRY


@dataclass
class CompetitionResult:
    """Result for a single competition in the benchmark."""
    
    slug: str
    task_type: str
    domain: str
    metric: str
    cv_score: Optional[float]
    public_score: float
    private_score: float
    cv_public_gap: Optional[float]
    cv_private_gap: Optional[float]
    public_percentile: float
    private_percentile: float
    shakeup: float
    medal: str
    total_submissions: int
    runtime_seconds: float
    winning_model: Optional[str]
    n_features_final: Optional[int]
    domain_features_generated: Optional[int]
    domain_features_kept: Optional[int]


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all competitions."""
    
    median_percentile: float
    mean_percentile: float
    gold_rate: float
    silver_rate: float
    bronze_rate: float
    medal_rate: float
    no_medal_rate: float
    catastrophic_failure_rate: float  # Bottom 50%
    mean_shakeup: float
    mean_cv_private_gap: float


@dataclass
class VersionComparison:
    """Comparison with previous benchmark run."""
    
    previous_version: str
    percentile_delta: float
    gold_rate_delta: float
    improved_competitions: List[str]
    degraded_competitions: List[str]
    unchanged_competitions: List[str]


@dataclass
class ComponentAttribution:
    """Attribution for which components helped/hurt."""
    
    shift_detector_triggered: int
    shift_detector_helped: int
    domain_features_generated: int
    domain_features_kept: int
    creative_features_generated: int
    creative_features_kept: int
    postprocess_improved: int
    postprocess_delta_mean: float
    pseudo_label_applied: int
    pseudo_label_helped: int
    pseudo_label_reverted: int
    critic_critical_count: int
    critic_replan_count: int


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    
    run_id: str
    professor_version: str
    timestamp: str
    n_competitions: int
    
    aggregate_metrics: Dict[str, float]
    per_competition: List[Dict[str, Any]]
    
    version_comparison: Optional[Dict[str, Any]] = None
    component_attribution: Optional[Dict[str, Any]] = None
    
    def save(self, path: str) -> None:
        """Save report to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))
    
    @classmethod
    def load(cls, path: str) -> "BenchmarkReport":
        """Load report from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**data)


def generate_benchmark_report(
    results: List[Dict[str, Any]],
    professor_version: str = "2.0.0",
    run_id: Optional[str] = None,
    component_stats: Optional[Dict[str, Any]] = None,
    previous_report_path: Optional[str] = None,
) -> BenchmarkReport:
    """
    Generate a benchmark report from individual competition results.
    
    Args:
        results: List of competition result dicts
        professor_version: Version string (e.g., "v2.3")
        run_id: Unique run identifier (auto-generated if None)
        component_stats: Component attribution stats (optional)
        previous_report_path: Path to previous report for comparison (optional)
    
    Returns:
        BenchmarkReport with aggregate metrics and per-competition breakdown
    """
    run_id = run_id or f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Compute aggregate metrics
    private_percentiles = [r["private_percentile"] for r in results]
    medals = [r["medal"] for r in results]
    shakeups = [r["shakeup"] for r in results]
    cv_private_gaps = [
        abs(r["cv_private_gap"]) for r in results 
        if r.get("cv_private_gap") is not None
    ]
    
    aggregate = AggregateMetrics(
        median_percentile=float(np.median(private_percentiles)),
        mean_percentile=float(np.mean(private_percentiles)),
        gold_rate=sum(1 for m in medals if m == "gold") / len(medals),
        silver_rate=sum(1 for m in medals if m == "silver") / len(medals),
        bronze_rate=sum(1 for m in medals if m == "bronze") / len(medals),
        medal_rate=sum(1 for m in medals if m in ("gold", "silver", "bronze")) / len(medals),
        no_medal_rate=sum(1 for m in medals if m == "none") / len(medals),
        catastrophic_failure_rate=sum(1 for p in private_percentiles if p < 50) / len(medals),
        mean_shakeup=float(np.mean(shakeups)),
        mean_cv_private_gap=float(np.mean(cv_private_gaps)) if cv_private_gaps else 0.0,
    )
    
    # Version comparison
    version_comp = None
    if previous_report_path and Path(previous_report_path).exists():
        version_comp = _compare_versions(results, previous_report_path)
    
    # Component attribution
    comp_attr = None
    if component_stats:
        comp_attr = ComponentAttribution(
            shift_detector_triggered=component_stats.get("shift_detector_triggered", 0),
            shift_detector_helped=component_stats.get("shift_detector_helped", 0),
            domain_features_generated=component_stats.get("domain_features_generated", 0),
            domain_features_kept=component_stats.get("domain_features_kept", 0),
            creative_features_generated=component_stats.get("creative_features_generated", 0),
            creative_features_kept=component_stats.get("creative_features_kept", 0),
            postprocess_improved=component_stats.get("postprocess_improved", 0),
            postprocess_delta_mean=component_stats.get("postprocess_delta_mean", 0.0),
            pseudo_label_applied=component_stats.get("pseudo_label_applied", 0),
            pseudo_label_helped=component_stats.get("pseudo_label_helped", 0),
            pseudo_label_reverted=component_stats.get("pseudo_label_reverted", 0),
            critic_critical_count=component_stats.get("critic_critical_count", 0),
            critic_replan_count=component_stats.get("critic_replan_count", 0),
        )
    
    report = BenchmarkReport(
        run_id=run_id,
        professor_version=professor_version,
        timestamp=datetime.utcnow().isoformat(),
        n_competitions=len(results),
        aggregate_metrics=asdict(aggregate),
        per_competition=results,
        version_comparison=asdict(version_comp) if version_comp else None,
        component_attribution=asdict(comp_attr) if comp_attr else None,
    )
    
    return report


def _compare_versions(
    current_results: List[Dict[str, Any]],
    previous_report_path: str,
) -> VersionComparison:
    """
    Compare current results with previous benchmark to detect regressions.
    
    REGRESSION RULE: If Professor scores WORSE than previous version on ANY
    competition by more than 1 percentile point, it is flagged as a
    regression.
    """
    import numpy as np
    
    previous = BenchmarkReport.load(previous_report_path)
    previous_by_slug = {r["slug"]: r for r in previous.per_competition}
    
    improved = []
    degraded = []
    unchanged = []
    
    percentile_deltas = []
    gold_deltas = []
    
    for result in current_results:
        slug = result["slug"]
        if slug not in previous_by_slug:
            unchanged.append(slug)
            continue
        
        prev = previous_by_slug[slug]
        # Lower percentile = better rank (top 10% is better than top 20%)
        # So if current > previous, performance got WORSE (regression)
        # If current < previous, performance got BETTER (improvement)
        delta = prev["private_percentile"] - result["private_percentile"]
        percentile_deltas.append(-delta)  # Store as negative for reporting (positive = improvement)
        
        # Gold rate delta
        curr_gold = 1 if result["medal"] == "gold" else 0
        prev_gold = 1 if prev["medal"] == "gold" else 0
        gold_deltas.append(curr_gold - prev_gold)
        
        if delta > 1.0:  # Improved (current is lower/better by more than 1 percentile)
            improved.append(slug)
        elif delta < -1.0:  # Degraded (current is higher/worse by more than 1 percentile)
            degraded.append(slug)
        else:
            unchanged.append(slug)
    
    return VersionComparison(
        previous_version=previous.professor_version,
        percentile_delta=float(np.mean(percentile_deltas)) if percentile_deltas else 0.0,
        gold_rate_delta=float(np.mean(gold_deltas)) if gold_deltas else 0.0,
        improved_competitions=improved,
        degraded_competitions=degraded,
        unchanged_competitions=unchanged,
    )


def print_benchmark_summary(report: BenchmarkReport) -> None:
    """Print a human-readable summary of the benchmark report."""
    import numpy as np
    
    print("\n" + "=" * 70)
    print(f"BENCHMARK REPORT: {report.run_id}")
    print(f"Professor Version: {report.professor_version}")
    print(f"Timestamp: {report.timestamp}")
    print("=" * 70)
    
    agg = report.aggregate_metrics
    print("\n📊 AGGREGATE METRICS")
    print("-" * 40)
    print(f"  Competitions:        {report.n_competitions}")
    print(f"  Median Percentile:   {agg['median_percentile']:.1f}%")
    print(f"  Mean Percentile:     {agg['mean_percentile']:.1f}%")
    print(f"  Mean Shakeup:        {agg['mean_shakeup']:+.1f} positions")
    print(f"  CV/Private Gap:      {agg['mean_cv_private_gap']:.4f}")
    
    print("\n🏅 MEDAL RATES")
    print("-" * 40)
    print(f"  Gold:                {agg['gold_rate']*100:.1f}%")
    print(f"  Silver:              {agg['silver_rate']*100:.1f}%")
    print(f"  Bronze:              {agg['bronze_rate']*100:.1f}%")
    print(f"  Any Medal:           {agg['medal_rate']*100:.1f}%")
    print(f"  No Medal:            {agg['no_medal_rate']*100:.1f}%")
    print(f"  Catastrophic (<50%): {agg['catastrophic_failure_rate']*100:.1f}%")
    
    if report.version_comparison:
        vc = report.version_comparison
        print("\n📈 VERSION COMPARISON")
        print("-" * 40)
        print(f"  vs {vc['previous_version']}:")
        print(f"    Percentile Delta:  {vc['percentile_delta']:+.1f}")
        print(f"    Gold Rate Delta:   {vc['gold_rate_delta']:+.1f}")
        print(f"    Improved:          {vc['improved_competitions']}")
        print(f"    Degraded:          {vc['degraded_competitions']}")
    
    print("\n📋 PER-COMPETITION RESULTS")
    print("-" * 40)
    print(f"{'Competition':<35} {'Medal':<8} {'Private %':<10} {'Shakeup':<8}")
    print("-" * 70)
    
    for comp in report.per_competition:
        medal_icon = _medal_icon(comp["medal"])
        print(
            f"{comp['slug']:<35} "
            f"{medal_icon} {comp['medal']:<6} "
            f"{comp['private_percentile']:>6.1f}%     "
            f"{comp['shakeup']:>+7.1f}"
        )
    
    print("=" * 70)


def _medal_icon(medal: str) -> str:
    """Return emoji icon for medal type."""
    icons = {
        "gold": "🥇",
        "silver": "🥈",
        "bronze": "🥉",
        "none": "  ",
    }
    return icons.get(medal, "  ")


# Import numpy at module level for aggregate calculations
import numpy as np
