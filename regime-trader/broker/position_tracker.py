"""Position tracking: open positions and running P&L."""

from __future__ import annotations

from dataclasses import dataclass

from broker.alpaca_client import AlpacaClient


@dataclass
class Position:
    """A single open position and its P&L."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class PositionTracker:
    """Tracks open positions and computes P&L against the broker's state."""

    def __init__(self, client: AlpacaClient) -> None:
        """Store the Alpaca client used to fetch position state."""
        ...

    def refresh(self) -> None:
        """Refresh tracked positions from the broker."""
        ...

    def get_position(self, symbol: str) -> Position | None:
        """Get the tracked position for a symbol, if any."""
        ...

    def get_all_positions(self) -> list[Position]:
        """Get all currently tracked positions."""
        ...

    def get_total_exposure(self, equity: float) -> float:
        """Compute total exposure as a fraction of equity."""
        ...

    def get_daily_pnl(self) -> float:
        """Compute realized + unrealized P&L for the current trading day."""
        ...
