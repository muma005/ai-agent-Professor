# adapters/__init__.py

"""
Competition adapters package.

Provides adapters for different competition types:
- TabularAdapter: Standard tabular data
- TimeSeriesAdapter: Time series data
- NLPAdapter: Text/NLP data
"""

from adapters.base import (
    BaseAdapter,
    AdapterResult,
    CompetitionType,
    detect_competition_type,
)

from adapters.tabular_adapter import TabularAdapter
from adapters.timeseries_adapter import TimeSeriesAdapter
from adapters.nlp_adapter import NLPAdapter

__all__ = [
    "BaseAdapter",
    "AdapterResult",
    "CompetitionType",
    "detect_competition_type",
    "TabularAdapter",
    "TimeSeriesAdapter",
    "NLPAdapter",
]
