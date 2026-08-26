"""Stress testing: crash injection, gap risk, and regime-misclassification.

Crash/gap injection perturbs the *price data* and re-simulates using the
SAME target-allocation decisions the strategy already made on the
unshocked data — this tests execution/position-sizing robustness to price
shocks, not "what regimes would the HMM have found on contaminated
history" (a different, much more expensive question this module doesn't
answer). Regime misclassification instead perturbs the *decisions*
(shuffling which allocation was in force when) and re-runs against the
real, unshocked prices, per the spec: "deliberately shuffle regime
labels... if system blows up, risk management isn't independent enough."

Because these Monte Carlo loops reuse a single already-computed baseline
walk-forward run and only re-run the cheap mechanical allocation
simulation (``backtest.backtester.simulate_allocation_series``) per
iteration, 100-simulation stress tests stay fast even though a full
walk-forward HMM refit is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest.backtester import Backtester, BacktestResult, simulate_allocation_series
from backtest.performance import compute_max_drawdown
from data.feature_engineering import ATR_WINDOW, average_true_range

DEFAULT_N_SHOCK_POINTS = 10
DEFAULT_CRASH_SHOCK_RANGE = (-0.15, -0.05)
DEFAULT_GAP_ATR_MULTIPLE_RANGE = (2.0, 5.0)
DEFAULT_N_SIMULATIONS = 100


@dataclass
class StressScenario:
    """A single, reusable price-shock scenario definition."""

    name: str
    description: str
    shock_pct: float
    duration_bars: int = 1


@dataclass
class CrashInjectionResult:
    """Result of the crash-injection Monte Carlo test."""

    n_simulations: int
    mean_max_drawdown_pct: float
    worst_max_drawdown_pct: float
    pct_circuit_breaker_fired: float
    per_simulation_max_drawdown_pct: list[float]


@dataclass
class GapRiskResult:
    """Result of the gap-risk Monte Carlo test."""

    n_simulations: int
    expected_loss_pct: float
    actual_mean_loss_pct: float
    actual_worst_loss_pct: float
    per_simulation_loss_pct: list[float]


@dataclass
class RegimeMisclassificationResult:
    """Result of the regime-misclassification (shuffle) test."""

    n_shuffles: int
    baseline_max_drawdown_pct: float
    mean_shuffled_max_drawdown_pct: float
    worst_shuffled_max_drawdown_pct: float
    contained: bool
    per_shuffle_max_drawdown_pct: list[float]


def _apply_level_shocks(
    bars: pd.DataFrame, shock_indices: list[int], shock_factors: list[float]
) -> pd.DataFrame:
    """Apply a permanent multiplicative level shift to OHLC starting at
    each shocked bar (a crash/gap changes the price level going forward,
    not just that one bar)."""
    n = len(bars)
    multiplier = np.ones(n)
    for idx, factor in zip(shock_indices, shock_factors):
        multiplier[idx:] *= factor
    shocked = bars.copy()
    for col in ("open", "high", "low", "close"):
        shocked[col] = bars[col].to_numpy() * multiplier
    return shocked


class StressTester:
    """Applies stress scenarios (crashes, gaps, regime shuffles) on top of
    an already-computed walk-forward backtest and measures the impact.
    """

    def __init__(
        self,
        backtester: Backtester,
        circuit_breaker_dd_threshold: float = 0.10,
    ) -> None:
        """``circuit_breaker_dd_threshold`` is the drawdown fraction (e.g.
        0.10 for -10%) beyond which we consider a "circuit breaker" to
        have fired. This is a stand-in threshold for stress-test reporting
        purposes only — the real halt logic belongs to
        ``core.risk_manager.RiskManager``, not yet wired into the
        backtester (see backtest/backtester.py's docstring)."""
        self.backtester = backtester
        self.circuit_breaker_dd_threshold = circuit_breaker_dd_threshold

    # ------------------------------------------------------------------
    # Generic single-shock utilities
    # ------------------------------------------------------------------

    def inject_crash(
        self,
        price_data: dict[str, pd.DataFrame],
        scenario: StressScenario,
        n_points: int = DEFAULT_N_SHOCK_POINTS,
        rng: Optional[np.random.RandomState] = None,
    ) -> dict[str, pd.DataFrame]:
        """Insert ``n_points`` single-day downward gaps of ``scenario.shock_pct``
        (a negative fraction) at random bars, each a permanent level shift."""
        rng = rng or np.random.RandomState()
        shocked = {}
        for symbol, bars in price_data.items():
            n = len(bars)
            k = min(n_points, n)
            indices = sorted(rng.choice(n, size=k, replace=False).tolist())
            factors = [1.0 + scenario.shock_pct] * k
            shocked[symbol] = _apply_level_shocks(bars, indices, factors)
        return shocked

    def inject_gap(
        self,
        price_data: dict[str, pd.DataFrame],
        scenario: StressScenario,
        n_points: int = DEFAULT_N_SHOCK_POINTS,
        rng: Optional[np.random.RandomState] = None,
    ) -> dict[str, pd.DataFrame]:
        """Insert ``n_points`` overnight gaps of ``scenario.shock_pct`` (a
        negative fraction, typically an ATR multiple) at random bars."""
        return self.inject_crash(price_data, scenario, n_points, rng)

    def run_scenario(self, price_data: dict[str, pd.DataFrame], scenario: StressScenario) -> BacktestResult:
        """Re-run the FULL walk-forward backtest (fresh HMM fits included)
        against price data with a single shock scenario applied. Slower
        than the Monte Carlo methods below, which reuse one baseline run's
        decisions; useful for a one-off "what if the HMM itself had seen
        this shocked history" check."""
        shocked = self.inject_crash(price_data, scenario, n_points=1)
        return self.backtester.run(shocked)

    def run_all_scenarios(
        self, price_data: dict[str, pd.DataFrame], scenarios: list[StressScenario]
    ) -> dict[str, BacktestResult]:
        return {scenario.name: self.run_scenario(price_data, scenario) for scenario in scenarios}

    # ------------------------------------------------------------------
    # (a) Crash injection Monte Carlo
    # ------------------------------------------------------------------

    def crash_injection_test(
        self,
        price_data: dict[str, pd.DataFrame],
        baseline_result: Optional[BacktestResult] = None,
        n_simulations: int = DEFAULT_N_SIMULATIONS,
        n_points: int = DEFAULT_N_SHOCK_POINTS,
        shock_range: tuple[float, float] = DEFAULT_CRASH_SHOCK_RANGE,
        seed: int = 0,
    ) -> CrashInjectionResult:
        """Insert ``n_points`` single-day -5% to -15% gaps at random points
        and re-simulate (reusing the baseline decisions) ``n_simulations``
        times. Reports mean/worst max drawdown and the fraction of runs
        where the circuit-breaker threshold was breached.
        """
        baseline_result = baseline_result or self.backtester.run(price_data)
        rng = np.random.RandomState(seed)

        max_drawdowns: list[float] = []
        for _ in range(n_simulations):
            shocked_curve = self._simulate_shocked_portfolio(
                price_data,
                baseline_result,
                shock_fn=lambda bars, target_index: _apply_level_shocks(
                    bars,
                    sorted(rng.choice(len(bars), size=min(n_points, len(bars)), replace=False).tolist()),
                    [1.0 + rng.uniform(*shock_range) for _ in range(min(n_points, len(bars)))],
                ),
            )
            max_dd, _ = compute_max_drawdown(shocked_curve)
            max_drawdowns.append(max_dd)

        fired = [dd <= -self.circuit_breaker_dd_threshold for dd in max_drawdowns]
        return CrashInjectionResult(
            n_simulations=n_simulations,
            mean_max_drawdown_pct=float(np.mean(max_drawdowns)),
            worst_max_drawdown_pct=float(np.min(max_drawdowns)),
            pct_circuit_breaker_fired=float(np.mean(fired)),
            per_simulation_max_drawdown_pct=max_drawdowns,
        )

    # ------------------------------------------------------------------
    # (b) Gap risk Monte Carlo
    # ------------------------------------------------------------------

    def gap_risk_test(
        self,
        price_data: dict[str, pd.DataFrame],
        baseline_result: Optional[BacktestResult] = None,
        n_simulations: int = DEFAULT_N_SIMULATIONS,
        n_points: int = DEFAULT_N_SHOCK_POINTS,
        atr_multiple_range: tuple[float, float] = DEFAULT_GAP_ATR_MULTIPLE_RANGE,
        seed: int = 0,
    ) -> GapRiskResult:
        """Insert ``n_points`` overnight gaps of 2-5x ATR(14) at random
        points. Reports expected loss (exposure at the gap times the gap
        size, using the decisions actually in force) vs. actual realized
        loss (the shocked equity curve's underperformance vs. baseline).
        """
        baseline_result = baseline_result or self.backtester.run(price_data)
        baseline_final = float(baseline_result.equity_curve.iloc[-1])
        rng = np.random.RandomState(seed)

        expected_losses: list[float] = []
        actual_losses: list[float] = []

        for _ in range(n_simulations):
            expected_loss_accum = {"value": 0.0}

            def shock_fn(bars: pd.DataFrame, target_index: pd.Index, _accum=expected_loss_accum) -> pd.DataFrame:
                atr = average_true_range(bars["high"], bars["low"], bars["close"], ATR_WINDOW)
                n = len(bars)
                k = min(n_points, n)
                indices = sorted(rng.choice(n, size=k, replace=False).tolist())
                gap_pcts = []
                for idx in indices:
                    atr_val = float(atr.iloc[idx]) if not np.isnan(atr.iloc[idx]) else 0.0
                    price = float(bars["close"].iloc[idx])
                    gap_pct = -rng.uniform(*atr_multiple_range) * (atr_val / price) if price else 0.0
                    gap_pcts.append(gap_pct)
                    ts = bars.index[idx]
                    exposure = float(target_index.get(ts, 0.0))
                    _accum["value"] += exposure * gap_pct
                return _apply_level_shocks(bars, indices, [1.0 + g for g in gap_pcts])

            shocked_curve = self._simulate_shocked_portfolio(price_data, baseline_result, shock_fn)
            actual_loss = float(shocked_curve.iloc[-1] / baseline_final - 1.0)

            expected_losses.append(expected_loss_accum["value"])
            actual_losses.append(actual_loss)

        return GapRiskResult(
            n_simulations=n_simulations,
            expected_loss_pct=float(np.mean(expected_losses)),
            actual_mean_loss_pct=float(np.mean(actual_losses)),
            actual_worst_loss_pct=float(np.min(actual_losses)),
            per_simulation_loss_pct=actual_losses,
        )

    # ------------------------------------------------------------------
    # (c) Regime misclassification
    # ------------------------------------------------------------------

    def regime_misclassification_test(
        self,
        price_data: dict[str, pd.DataFrame],
        baseline_result: Optional[BacktestResult] = None,
        n_shuffles: int = DEFAULT_N_SIMULATIONS,
        seed: int = 0,
    ) -> RegimeMisclassificationResult:
        """Deliberately shuffle (randomly permute in time) the regime-driven
        target-allocation decisions and re-run against the REAL, unshocked
        prices. If the system's downside stays bounded even when regime
        calls are scrambled, risk management is doing real, independent
        work; if a shuffle can blow the account up, risk management isn't
        independent of getting the regime call right.
        """
        baseline_result = baseline_result or self.backtester.run(price_data)
        baseline_max_dd, _ = compute_max_drawdown(baseline_result.equity_curve)
        rng = np.random.RandomState(seed)

        symbols = list(price_data.keys())
        capital_per_symbol = self.backtester.initial_capital / len(symbols)

        shuffled_dds: list[float] = []
        for _ in range(n_shuffles):
            sleeve_curves = []
            for symbol in symbols:
                target_series = self._symbol_target_series(baseline_result, symbol)
                if target_series.empty:
                    continue
                shuffled_values = rng.permutation(target_series.to_numpy())
                shuffled_series = pd.Series(shuffled_values, index=target_series.index)
                bars = price_data[symbol].loc[target_series.index]
                equity_curve, _ = simulate_allocation_series(
                    bars,
                    shuffled_series,
                    capital_per_symbol,
                    self.backtester.slippage_pct,
                    self.backtester._strategy_config.rebalance_threshold,
                )
                sleeve_curves.append(equity_curve)
            if not sleeve_curves:
                continue
            portfolio_curve = pd.concat(sleeve_curves, axis=1).sum(axis=1)
            max_dd, _ = compute_max_drawdown(portfolio_curve)
            shuffled_dds.append(max_dd)

        worst = float(np.min(shuffled_dds)) if shuffled_dds else baseline_max_dd
        contained = worst > -3.0 * self.circuit_breaker_dd_threshold

        return RegimeMisclassificationResult(
            n_shuffles=n_shuffles,
            baseline_max_drawdown_pct=float(baseline_max_dd),
            mean_shuffled_max_drawdown_pct=float(np.mean(shuffled_dds)) if shuffled_dds else 0.0,
            worst_shuffled_max_drawdown_pct=worst,
            contained=contained,
            per_shuffle_max_drawdown_pct=shuffled_dds,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _symbol_target_series(self, baseline_result: BacktestResult, symbol: str) -> pd.Series:
        history = baseline_result.regime_history
        if history.empty:
            return pd.Series(dtype=float)
        if "symbol" in history.columns:
            history = history[history["symbol"] == symbol]
        return history["target_allocation"].sort_index()

    def _simulate_shocked_portfolio(
        self,
        price_data: dict[str, pd.DataFrame],
        baseline_result: BacktestResult,
        shock_fn,
    ) -> pd.Series:
        """Re-simulate every symbol's sleeve on price data shocked by
        ``shock_fn(bars, target_series) -> shocked_bars``, reusing the
        baseline run's target-allocation decisions, and return the summed
        portfolio equity curve."""
        symbols = list(price_data.keys())
        capital_per_symbol = self.backtester.initial_capital / len(symbols)

        sleeve_curves = []
        for symbol in symbols:
            target_series = self._symbol_target_series(baseline_result, symbol)
            if target_series.empty:
                continue
            bars = price_data[symbol].loc[target_series.index]
            shocked_bars = shock_fn(bars, target_series)
            equity_curve, _ = simulate_allocation_series(
                shocked_bars,
                target_series,
                capital_per_symbol,
                self.backtester.slippage_pct,
                self.backtester._strategy_config.rebalance_threshold,
            )
            sleeve_curves.append(equity_curve)

        return pd.concat(sleeve_curves, axis=1).sum(axis=1)
