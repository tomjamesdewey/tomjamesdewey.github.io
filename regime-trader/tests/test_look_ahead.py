"""Tests verifying no look-ahead bias in feature engineering, HMM fitting,
and the walk-forward backtester.
"""

from __future__ import annotations

import pytest


def test_features_use_only_past_data() -> None:
    """Feature values at time t must not depend on bars after t."""
    ...


def test_hmm_training_window_excludes_test_window() -> None:
    """The HMM must only be fit on train_window data, never on test_window data."""
    ...


def test_backtest_walk_forward_no_future_leakage() -> None:
    """Walk-forward backtest steps must never use future bars to generate signals."""
    ...
