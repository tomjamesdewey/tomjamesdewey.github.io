"""Terminal-based live dashboard using rich."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table


class Dashboard:
    """Renders a live-refreshing terminal dashboard of bot state."""

    def __init__(self, refresh_seconds: int = 5) -> None:
        """Initialize the dashboard console and refresh interval."""
        ...

    def render_positions(self, positions: list[dict[str, Any]]) -> Table:
        """Render a table of current open positions."""
        ...

    def render_regime_status(self, regime_by_symbol: dict[str, Any]) -> Table:
        """Render a table of current regime state per symbol."""
        ...

    def render_account_summary(self, account: dict[str, Any]) -> Table:
        """Render a summary panel of account equity, P&L, and exposure."""
        ...

    def run(self) -> None:
        """Start the live-refreshing dashboard loop."""
        ...
