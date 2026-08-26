"""Tests for backtest.performance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import BacktestResult
from backtest.performance import (
    PerformanceAnalyzer,
    compute_cagr,
    compute_drawdown_series,
    compute_longest_underwater,
    compute_max_drawdown,
    compute_returns,
    compute_trade_pnls,
)


def _equity(values: list[float], start: str = "2024-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_compute_returns_matches_pct_change() -> None:
    equity = _equity([100.0, 110.0, 99.0])
    returns = compute_returns(equity)
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(99.0 / 110.0 - 1.0)


def test_compute_drawdown_series_and_max_drawdown() -> None:
    # Peak 100 -> trough 80 (-20%) -> recovers to 120 -> dips to 108 (-10%).
    equity = _equity([100.0, 90.0, 80.0, 100.0, 120.0, 108.0, 120.0])
    drawdown = compute_drawdown_series(equity)

    assert drawdown.iloc[2] == pytest.approx(-0.20)
    assert drawdown.iloc[5] == pytest.approx(-0.10)

    max_dd, duration = compute_max_drawdown(equity)
    assert max_dd == pytest.approx(-0.20)
    assert duration == 2  # bars at index 1,2 are underwater before recovering at index 3


def test_longest_underwater_can_differ_from_deepest_episode() -> None:
    """A shallow-but-long drawdown episode should win 'longest underwater'
    even when a separate deep-but-short episode has the larger magnitude."""
    equity = _equity(
        [
            100.0,  # peak
            50.0,  # deep, short episode: -50%
            100.0,  # recovers (new peak, ties prior)
            99.0, 98.0, 97.0, 96.0, 95.0,  # shallow but long episode: 5 bars underwater
            101.0,  # recovers
        ]
    )
    max_dd, max_dd_duration = compute_max_drawdown(equity)
    assert max_dd == pytest.approx(-0.50)
    assert max_dd_duration == 1

    assert compute_longest_underwater(equity) == 5


def test_compute_cagr_known_case() -> None:
    # Doubling over exactly 252 bar-to-bar steps (1 trading year) -> CAGR = 100%.
    equity = _equity([100.0] * 252 + [200.0])
    assert compute_cagr(equity) == pytest.approx(1.0, rel=1e-6)


def test_compute_trade_pnls_segments_between_fills() -> None:
    equity = _equity([100.0, 110.0, 121.0, 108.9])
    trades = pd.DataFrame(
        {
            "timestamp": [equity.index[1], equity.index[2]],
            "target_allocation": [1.0, 0.5],
        }
    )

    pnls = compute_trade_pnls(equity, trades)

    assert len(pnls) == 2
    assert pnls.iloc[0]["pnl_pct"] == pytest.approx(121.0 / 110.0 - 1.0)
    assert pnls.iloc[0]["holding_days"] == 1
    assert pnls.iloc[1]["pnl_pct"] == pytest.approx(108.9 / 121.0 - 1.0)
    assert pnls.iloc[1]["holding_days"] == 1


def test_compute_trade_pnls_empty_trades_returns_empty_frame() -> None:
    equity = _equity([100.0, 101.0])
    result = compute_trade_pnls(equity, pd.DataFrame())
    assert result.empty
    assert list(result.columns) == ["timestamp", "exit_timestamp", "pnl_pct", "holding_days"]


def test_compute_metrics_sane_on_walkforward_result(walkforward_result: BacktestResult) -> None:
    analyzer = PerformanceAnalyzer(risk_free_rate=0.045)
    metrics = analyzer.compute_metrics(walkforward_result.equity_curve, walkforward_result.trades)

    assert metrics.total_return_pct == pytest.approx(
        walkforward_result.equity_curve.iloc[-1] / walkforward_result.equity_curve.iloc[0] - 1.0
    )
    assert metrics.max_drawdown_pct <= 0.0
    assert metrics.max_drawdown_duration_days >= 0
    assert 0.0 <= metrics.win_rate <= 1.0
    assert metrics.total_trades == len(walkforward_result.trades)


def test_worst_case_stats_are_non_positive(walkforward_result: BacktestResult) -> None:
    analyzer = PerformanceAnalyzer(risk_free_rate=0.045)
    worst = analyzer.compute_worst_case_stats(walkforward_result.equity_curve, walkforward_result.trades)

    assert worst.worst_day_pct <= 0.0
    assert worst.worst_week_pct <= 0.0
    assert worst.worst_month_pct <= 0.0
    assert worst.max_consecutive_losing_trades >= 0
    assert worst.longest_underwater_days >= 0


def test_regime_breakdown_time_in_regime_sums_to_one(walkforward_result: BacktestResult) -> None:
    analyzer = PerformanceAnalyzer(risk_free_rate=0.045)
    breakdown = analyzer.compute_regime_breakdown(
        walkforward_result.equity_curve, walkforward_result.regime_history, walkforward_result.trades
    )

    assert not breakdown.empty
    assert breakdown["pct_time_in"].sum() == pytest.approx(1.0, abs=1e-6)
    assert set(breakdown.columns) == {
        "regime",
        "pct_time_in",
        "return_contribution_pct",
        "avg_trade_pnl_pct",
        "win_rate",
        "sharpe",
    }


def test_confidence_breakdown_buckets_all_trades(walkforward_result: BacktestResult) -> None:
    analyzer = PerformanceAnalyzer(risk_free_rate=0.045)
    breakdown = analyzer.compute_confidence_breakdown(
        walkforward_result.equity_curve, walkforward_result.trades, walkforward_result.regime_history
    )

    assert len(breakdown) == 4  # <50%, 50-60%, 60-70%, 70%+
    assert breakdown["trades"].sum() == len(walkforward_result.trades)


def test_generate_buy_and_hold_matches_simple_share_math() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    bars = pd.DataFrame({"close": [100.0, 110.0, 121.0]}, index=idx)

    analyzer = PerformanceAnalyzer(risk_free_rate=0.0)
    curve = analyzer.generate_buy_and_hold(bars, initial_capital=1000.0)

    assert curve.iloc[0] == pytest.approx(1000.0)
    assert curve.iloc[-1] == pytest.approx(1000.0 * 1.21)


def test_random_benchmark_monte_carlo_returns_one_row_per_seed() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    rng = np.random.RandomState(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 200)))
    bars = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995, "close": close}, index=idx)

    analyzer = PerformanceAnalyzer(risk_free_rate=0.045)
    mc = analyzer.run_random_benchmark_monte_carlo(
        bars, initial_capital=100_000, slippage_pct=0.0005, rebalance_threshold=0.10,
        trade_frequency_bars=20, n_seeds=10,
    )
    assert len(mc) == 10
    assert "total_return_pct" in mc.columns


def test_export_csvs_writes_all_files(walkforward_result: BacktestResult, tmp_path) -> None:
    analyzer = PerformanceAnalyzer(risk_free_rate=0.045)
    analyzer.export_csvs(
        tmp_path, walkforward_result.equity_curve, walkforward_result.trades, walkforward_result.regime_history
    )

    for filename in ("equity_curve.csv", "trade_log.csv", "regime_history.csv"):
        assert (tmp_path / filename).exists()
