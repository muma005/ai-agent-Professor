"""
Private Leaderboard Simulator — Professor Benchmark Infrastructure

This module provides a complete competition simulation environment that mirrors
real Kaggle competition dynamics:
- Public/Private leaderboard split (30/70)
- Daily submission limits
- Strategic submission feedback loop
- Percentile-calibrated medal awards
- Deterministic, reproducible splits

Usage:
    professor benchmark --all              # Run all competitions (fast mode)
    professor benchmark --competition X    # Run specific competition
    professor benchmark --deep             # Full-fidelity simulation
"""

from simulator.competition_registry import CompetitionEntry, REGISTRY, get_competition
from simulator.data_splitter import split_competition_data, SplitResult
from simulator.leaderboard import SimulatedLeaderboard, SubmissionResult, CompetitionResult
from simulator.report_generator import generate_benchmark_report, print_benchmark_summary
from simulator.data_downloader import download_competition, list_downloaded_competitions

__all__ = [
    # Core classes
    "CompetitionEntry",
    "REGISTRY",
    "split_competition_data",
    "SplitResult",
    "SimulatedLeaderboard",
    "SubmissionResult",
    "CompetitionResult",
    # Report generation
    "generate_benchmark_report",
    "print_benchmark_summary",
    # Data management
    "download_competition",
    "list_downloaded_competitions",
    # Lookup utilities
    "get_competition",
]

__version__ = "2.0.0"
