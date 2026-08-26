"""Stress testing: crash injection and gap simulation scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.backtester import Backtester, BacktestResult


@dataclass
class StressScenario:
    """Defines a single stress-test scenario."""

    name: str
    description: str
    shock_pct: float
    duration_bars: int


class StressTester:
    """Applies stress scenarios (crashes, gaps) to historical data and re-runs the backtest."""

    def __init__(self, backtester: Backtester) -> None:
        """Store the backtester used to evaluate stress scenarios."""
        ...

    def inject_crash(
        self, price_data: dict[str, pd.DataFrame], scenario: StressScenario
    ) -> dict[str, pd.DataFrame]:
        """Inject a simulated crash into historical price data."""
        ...

    def inject_gap(
        self, price_data: dict[str, pd.DataFrame], scenario: StressScenario
    ) -> dict[str, pd.DataFrame]:
        """Inject a simulated overnight gap into historical price data."""
        ...

    def run_scenario(
        self, price_data: dict[str, pd.DataFrame], scenario: StressScenario
    ) -> BacktestResult:
        """Run the backtester against price data with a stress scenario applied."""
        ...

    def run_all_scenarios(
        self, price_data: dict[str, pd.DataFrame], scenarios: list[StressScenario]
    ) -> dict[str, BacktestResult]:
        """Run all provided stress scenarios and collect results."""
        ...
