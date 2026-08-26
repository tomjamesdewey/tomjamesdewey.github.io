"""Market data fetching: real-time quotes/bars and historical bar data."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient


class MarketDataClient:
    """Fetches real-time and historical market data via Alpaca."""

    def __init__(self, client: AlpacaClient) -> None:
        """Store the Alpaca client used to fetch market data."""
        ...

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars for a symbol."""
        ...

    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Fetch the most recent bar for a symbol."""
        ...

    def get_latest_quote(self, symbol: str) -> dict:
        """Fetch the latest bid/ask quote for a symbol."""
        ...

    def stream_bars(self, symbols: list[str], on_bar) -> None:
        """Subscribe to a real-time bar stream for the given symbols."""
        ...
