"""Alpaca API wrapper.

Thin client around the Alpaca trading and market data APIs, handling
authentication and connection setup for both paper and live trading.
"""

from __future__ import annotations

from typing import Optional


class AlpacaClient:
    """Wraps the Alpaca trading and data API clients."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        """Initialize the underlying Alpaca trading and data clients."""
        ...

    def get_account(self) -> dict:
        """Fetch current account information (equity, buying power, etc.)."""
        ...

    def get_positions(self) -> list[dict]:
        """Fetch all currently open positions."""
        ...

    def get_clock(self) -> dict:
        """Fetch current market clock (open/closed status, next open/close)."""
        ...

    def is_market_open(self) -> bool:
        """Check whether the market is currently open."""
        ...
