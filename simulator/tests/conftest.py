"""
Pytest configuration and shared fixtures for simulator tests.
"""

import pytest
from datetime import date
import polars as pl


@pytest.fixture
def date_range():
    """Helper to create date ranges in tests."""
    def _date_range(start, end, length):
        from datetime import timedelta
        total_days = (end - start).days
        step = total_days / length
        return [start + timedelta(days=int(step * i)) for i in range(length)]
    return _date_range
