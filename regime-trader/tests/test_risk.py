"""Tests for core.risk_manager.RiskManager."""

from __future__ import annotations

import pytest


def test_position_size_respects_max_risk_per_trade() -> None:
    """calculate_position_size should never risk more than max_risk_per_trade of equity."""
    ...


def test_exposure_limit_blocks_over_allocation() -> None:
    """check_exposure_limit should reject additions that exceed max_exposure or max_leverage."""
    ...


def test_concurrency_limit_blocks_extra_positions() -> None:
    """check_concurrency_limit should reject a new position past max_concurrent."""
    ...


def test_daily_drawdown_halts_trading() -> None:
    """evaluate_drawdown should return HALT_NEW_TRADES at daily_dd_halt."""
    ...


def test_weekly_drawdown_reduces_exposure() -> None:
    """evaluate_drawdown should return REDUCE_EXPOSURE at weekly_dd_reduce."""
    ...
