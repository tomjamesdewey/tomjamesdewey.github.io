"""Tests for backtest.stress_test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import BacktestResult
from backtest.stress_test import StressScenario, StressTester, _apply_level_shocks


def test_apply_level_shocks_is_permanent_not_transient() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)

    shocked = _apply_level_shocks(bars, shock_indices=[2], shock_factors=[0.90])

    assert shocked["close"].iloc[0] == pytest.approx(100.0)
    assert shocked["close"].iloc[1] == pytest.approx(100.0)
    assert shocked["close"].iloc[2] == pytest.approx(90.0)
    assert shocked["close"].iloc[3] == pytest.approx(90.0)  # shift persists forward
    assert shocked["close"].iloc[4] == pytest.approx(90.0)


def test_apply_level_shocks_compounds_multiple_shocks() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)

    shocked = _apply_level_shocks(bars, shock_indices=[1, 2], shock_factors=[0.90, 0.80])

    assert shocked["close"].iloc[0] == pytest.approx(100.0)
    assert shocked["close"].iloc[1] == pytest.approx(90.0)
    assert shocked["close"].iloc[2] == pytest.approx(90.0 * 0.80)
    assert shocked["close"].iloc[3] == pytest.approx(90.0 * 0.80)


def test_inject_crash_shocks_within_range() -> None:
    tester = StressTester(backtester=None, circuit_breaker_dd_threshold=0.10)  # backtester unused here
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    bars = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)

    scenario = StressScenario(name="crash", description="test", shock_pct=-0.10)
    shocked = tester.inject_crash({"TEST": bars}, scenario, n_points=5, rng=np.random.RandomState(1))

    final_price = shocked["TEST"]["close"].iloc[-1]
    # Five compounded -10% shocks: 0.9^5 * 100.
    assert final_price == pytest.approx(100.0 * (0.9**5))


def test_crash_injection_test_runs_and_reports_bounds(backtester, walkforward_result: BacktestResult, walkforward_bars: pd.DataFrame) -> None:
    tester = StressTester(backtester, circuit_breaker_dd_threshold=0.10)

    result = tester.crash_injection_test(
        {"TEST": walkforward_bars}, baseline_result=walkforward_result, n_simulations=15, seed=1
    )

    assert result.n_simulations == 15
    assert len(result.per_simulation_max_drawdown_pct) == 15
    assert result.worst_max_drawdown_pct <= result.mean_max_drawdown_pct <= 0.0
    assert 0.0 <= result.pct_circuit_breaker_fired <= 1.0
    # Injecting -5%..-15% crashes can only ever make the drawdown worse than baseline.
    baseline_max_dd = walkforward_result.equity_curve.pipe(
        lambda eq: (eq / eq.cummax() - 1.0).min()
    )
    assert result.mean_max_drawdown_pct <= baseline_max_dd + 1e-9


def test_gap_risk_test_runs_and_reports_expected_vs_actual(
    backtester, walkforward_result: BacktestResult, walkforward_bars: pd.DataFrame
) -> None:
    tester = StressTester(backtester, circuit_breaker_dd_threshold=0.10)

    result = tester.gap_risk_test(
        {"TEST": walkforward_bars}, baseline_result=walkforward_result, n_simulations=15, seed=2
    )

    assert result.n_simulations == 15
    assert len(result.per_simulation_loss_pct) == 15
    assert result.actual_worst_loss_pct <= result.actual_mean_loss_pct <= 0.0
    assert result.expected_loss_pct <= 0.0


def test_regime_misclassification_test_runs(
    backtester, walkforward_result: BacktestResult, walkforward_bars: pd.DataFrame
) -> None:
    tester = StressTester(backtester, circuit_breaker_dd_threshold=0.10)

    result = tester.regime_misclassification_test(
        {"TEST": walkforward_bars}, baseline_result=walkforward_result, n_shuffles=15, seed=3
    )

    assert result.n_shuffles == 15
    assert len(result.per_shuffle_max_drawdown_pct) == 15
    assert result.worst_shuffled_max_drawdown_pct <= result.mean_shuffled_max_drawdown_pct <= 0.0
    assert isinstance(result.contained, bool)
