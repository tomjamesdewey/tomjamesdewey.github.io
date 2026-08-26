"""Tests for backtest.backtester."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.backtester import (
    BacktestResult,
    generate_walk_forward_windows,
    simulate_allocation_series,
)


def test_generate_walk_forward_windows_tiles_forward() -> None:
    windows = generate_walk_forward_windows(n_rows=1000, train_window=252, test_window=126, step_size=126)

    assert windows[0] == (0, 252, 252, 378)
    assert windows[1] == (126, 378, 378, 504)
    for is_start, is_end, oos_start, oos_end in windows:
        assert is_end - is_start == 252
        assert oos_start == is_end
        assert oos_end > oos_start
    # Final window's OOS segment is truncated to whatever data remains.
    assert windows[-1][3] == 1000


def test_generate_walk_forward_windows_empty_when_too_short() -> None:
    assert generate_walk_forward_windows(n_rows=200, train_window=252, test_window=126, step_size=126) == []


def test_generate_walk_forward_windows_truncates_final_oos() -> None:
    """A trailing partial OOS segment is still included, just shorter."""
    windows = generate_walk_forward_windows(n_rows=300, train_window=252, test_window=126, step_size=126)
    assert windows == [(0, 252, 252, 300)]


def _tiny_bars(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices}, index=idx)


def test_simulate_allocation_series_exact_math_no_slippage() -> None:
    """Hand-verify the exact allocation formula on a tiny, known series."""
    bars = _tiny_bars([100.0, 100.0, 110.0, 110.0])
    target = pd.Series([1.0, 1.0, 1.0, 1.0], index=bars.index)

    equity, trades = simulate_allocation_series(
        bars, target, initial_capital=1000.0, slippage_pct=0.0, rebalance_threshold=0.10
    )

    # Bar 0: decide target=1.0 (drift 1.0 > 0.10), nothing executes yet (fill delay).
    assert equity.iloc[0] == pytest.approx(1000.0)
    # Bar 1: execute at open=100 -> target_shares = int(1000*1.0/100) = 10; cash = 1000 - 10*100 = 0.
    assert equity.iloc[1] == pytest.approx(0.0 + 10 * 100.0)
    # Bar 2: no new rebalance decided (target unchanged); mark to market at close=110.
    assert equity.iloc[2] == pytest.approx(10 * 110.0)
    assert len(trades) == 1
    assert trades.iloc[0]["shares_delta"] == 10
    assert trades.iloc[0]["fill_price"] == pytest.approx(100.0)


def test_simulate_allocation_series_respects_rebalance_threshold() -> None:
    """A target drift under the threshold should never trigger a trade."""
    bars = _tiny_bars([100.0] * 6)
    # First decision establishes 0.50; the next is only 0.05 away (< 0.10 threshold).
    target = pd.Series([0.50, 0.50, 0.50, 0.55, 0.55, 0.55], index=bars.index)

    _, trades = simulate_allocation_series(
        bars, target, initial_capital=1000.0, slippage_pct=0.0, rebalance_threshold=0.10
    )

    assert len(trades) == 1  # only the initial 0.0 -> 0.50 rebalance


def test_simulate_allocation_series_leverage_produces_margin_debt() -> None:
    """target_allocation > 1.0 (leverage) makes target_shares*price > equity,
    so cash goes negative — margin — while equity = cash + shares*price still holds.
    """
    bars = _tiny_bars([100.0, 100.0, 100.0])
    target = pd.Series([1.25, 1.25, 1.25], index=bars.index)

    equity, trades = simulate_allocation_series(
        bars, target, initial_capital=1000.0, slippage_pct=0.0, rebalance_threshold=0.10
    )

    # Executed at bar 1's open=100: target_shares = int(1000*1.25/100) = 12; cash = 1000-1200 = -200.
    assert trades.iloc[0]["shares_delta"] == 12
    shares_after = 12
    expected_cash = 1000.0 - 12 * 100.0
    assert expected_cash < 0
    assert equity.iloc[1] == pytest.approx(expected_cash + shares_after * 100.0)
    assert equity.iloc[1] == pytest.approx(1000.0)  # no price move yet, so equity is unchanged


def test_simulate_allocation_series_slippage_worsens_fill() -> None:
    bars = _tiny_bars([100.0, 100.0])
    target = pd.Series([1.0, 1.0], index=bars.index)

    _, trades = simulate_allocation_series(
        bars, target, initial_capital=1000.0, slippage_pct=0.01, rebalance_threshold=0.10
    )

    assert trades.iloc[0]["fill_price"] == pytest.approx(101.0)  # buy fills worse (higher)


def test_backtester_requires_minimum_history(backtester) -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    tiny_bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000.0},
        index=idx,
    )
    with pytest.raises(ValueError):
        backtester.run({"TINY": tiny_bars})


def test_backtester_run_produces_continuous_equity_curve(walkforward_result: BacktestResult) -> None:
    equity = walkforward_result.equity_curve
    assert len(equity) > 0
    assert not equity.isna().any()
    assert equity.index.is_monotonic_increasing
    assert (equity > 0).all()

    trades = walkforward_result.trades
    for col in ("timestamp", "decided_at", "target_allocation", "shares_delta", "fill_price"):
        assert col in trades.columns

    regime_history = walkforward_result.regime_history
    for col in ("label", "probability", "is_confirmed", "consecutive_bars", "target_allocation"):
        assert col in regime_history.columns
    assert regime_history["probability"].between(0.0, 1.0).all()

    assert walkforward_result.metadata["windows_per_symbol"]["TEST"] >= 1
