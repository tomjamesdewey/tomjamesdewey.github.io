"""Risk management: position sizing, leverage, and drawdown limits.

Enforces per-trade risk, exposure, leverage, and concurrency limits, and
halts or reduces trading based on daily/weekly drawdown thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskAction(Enum):
    """Action the risk manager recommends given current portfolio state."""

    NORMAL = "normal"
    REDUCE_EXPOSURE = "reduce_exposure"
    HALT_NEW_TRADES = "halt_new_trades"


@dataclass
class PositionSizeResult:
    """Result of a position sizing calculation."""

    symbol: str
    quantity: float
    notional: float
    risk_amount: float


class RiskManager:
    """Applies position sizing rules and portfolio-level risk limits."""

    def __init__(
        self,
        max_risk_per_trade: float,
        max_exposure: float,
        max_leverage: float,
        max_single_position: float,
        max_concurrent: int,
        max_daily_trades: int,
        daily_dd_halt: float,
        weekly_dd_reduce: float,
        weekly_dd_halt: float,
        max_dd_from_peak: float,
    ) -> None:
        """Store risk limit parameters."""
        ...

    def calculate_position_size(
        self,
        symbol: str,
        equity: float,
        entry_price: float,
        stop_price: float,
    ) -> PositionSizeResult:
        """Calculate position size respecting max_risk_per_trade and max_single_position."""
        ...

    def check_exposure_limit(self, current_exposure: float, additional_notional: float) -> bool:
        """Check whether adding a position would breach max_exposure or max_leverage."""
        ...

    def check_concurrency_limit(self, open_position_count: int) -> bool:
        """Check whether opening a new position would breach max_concurrent."""
        ...

    def check_daily_trade_limit(self, trades_today: int) -> bool:
        """Check whether another trade would breach max_daily_trades."""
        ...

    def evaluate_drawdown(
        self,
        daily_pnl_pct: float,
        weekly_pnl_pct: float,
        drawdown_from_peak: float,
    ) -> RiskAction:
        """Determine the risk action based on current drawdown metrics."""
        ...
