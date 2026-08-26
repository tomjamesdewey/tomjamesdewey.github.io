"""Walk-forward allocation backtester.

Runs the HMM regime engine and allocation strategy over historical data
using rolling train/test windows to avoid look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import RiskManager


@dataclass
class BacktestResult:
    """Results of a backtest run."""

    equity_curve: pd.Series
    trades: pd.DataFrame
    regime_history: pd.DataFrame
    metadata: dict = field(default_factory=dict)


class Backtester:
    """Walk-forward backtester for the regime-based allocation strategy."""

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy: StrategyOrchestrator,
        risk_manager: RiskManager,
        initial_capital: float,
        slippage_pct: float,
        train_window: int,
        test_window: int,
        step_size: int,
    ) -> None:
        """Store backtest configuration and strategy components."""
        ...

    def run(self, price_data: dict[str, pd.DataFrame]) -> BacktestResult:
        """Run the full walk-forward backtest across all symbols."""
        ...

    def _run_window(
        self,
        train_data: dict[str, pd.DataFrame],
        test_data: dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """Run a single walk-forward train/test window."""
        ...

    def _simulate_fill(self, price: float, side: str) -> float:
        """Simulate a fill price including slippage."""
        ...
