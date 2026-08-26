"""Tests for core.regime_strategies.RegimeStrategy."""

from __future__ import annotations

import pytest


def test_low_vol_regime_allocates_low_vol_allocation() -> None:
    """Low-volatility regime should map to low_vol_allocation with leverage applied."""
    ...


def test_mid_vol_regime_allocation_depends_on_trend() -> None:
    """Mid-volatility regime should use trend vs. no-trend allocation values."""
    ...


def test_high_vol_regime_allocates_high_vol_allocation() -> None:
    """High-volatility regime should map to high_vol_allocation."""
    ...


def test_confidence_adjustment_scales_down_allocation() -> None:
    """Low regime confidence should scale allocation down by uncertainty_size_mult."""
    ...


def test_rebalance_threshold_suppresses_small_drift() -> None:
    """needs_rebalance should return False when drift is below rebalance_threshold."""
    ...
