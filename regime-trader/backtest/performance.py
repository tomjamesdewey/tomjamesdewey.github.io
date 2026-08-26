"""Performance analytics: Sharpe, drawdown, regime breakdown, and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PerformanceMetrics:
    """Summary performance metrics for a backtest or live equity curve."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float


class PerformanceAnalyzer:
    """Computes performance metrics and regime-level breakdowns."""

    def __init__(self, risk_free_rate: float) -> None:
        """Store the risk-free rate used for Sharpe/Calmar calculations."""
        ...

    def compute_metrics(self, equity_curve: pd.Series) -> PerformanceMetrics:
        """Compute summary performance metrics from an equity curve."""
        ...

    def compute_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Compute the drawdown-from-peak series."""
        ...

    def compute_regime_breakdown(
        self, equity_curve: pd.Series, regime_history: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute performance metrics broken down by market regime."""
        ...

    def compare_to_benchmark(
        self, equity_curve: pd.Series, benchmark_curve: pd.Series
    ) -> pd.DataFrame:
        """Compare strategy performance against a buy-and-hold benchmark."""
        ...
