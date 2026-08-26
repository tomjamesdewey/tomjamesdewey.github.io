"""Market data fetching: real-time and historical bar/quote data via Alpaca.

Gaps (weekends, holidays, trading halts) are handled by deliberately NOT
reindexing over a fixed calendar: whatever timestamps Alpaca actually
returns are used as-is. Fabricating or forward-filling bars for missing
sessions would corrupt every downstream indicator, so a gap in the market
simply shows up as a gap in the DataFrame's index.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Awaitable, Callable, Optional, Union

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.models.bars import Bar
from alpaca.data.models.quotes import Quote
from alpaca.data.models.snapshots import Snapshot
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestBarRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

_TIMEFRAME_UNIT_ALIASES = {
    "min": TimeFrameUnit.Minute,
    "minute": TimeFrameUnit.Minute,
    "hour": TimeFrameUnit.Hour,
    "day": TimeFrameUnit.Day,
    "week": TimeFrameUnit.Week,
    "month": TimeFrameUnit.Month,
}


def parse_timeframe(timeframe: str) -> TimeFrame:
    """Parse a settings.yaml-style timeframe string (e.g. "1Day", "5Min",
    "1Hour") into an alpaca-py ``TimeFrame``."""
    digits = "".join(c for c in timeframe if c.isdigit()) or "1"
    unit_str = "".join(c for c in timeframe if c.isalpha()).lower()
    if unit_str not in _TIMEFRAME_UNIT_ALIASES:
        raise ValueError(f"Unrecognized timeframe unit in {timeframe!r}")
    return TimeFrame(int(digits), _TIMEFRAME_UNIT_ALIASES[unit_str])


def _bars_to_dataframe(bars: list[Bar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    records = [
        {
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in bars
    ]
    return pd.DataFrame.from_records(records, index="timestamp").sort_index()


class MarketDataClient:
    """Fetches real-time and historical market data via Alpaca."""

    def __init__(self, client: AlpacaClient, feed: DataFeed = DataFeed.IEX) -> None:
        """Store the Alpaca client used to fetch market data."""
        self.client = client
        self.feed = feed
        self._stream: Optional[StockDataStream] = None

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars for a symbol.

        Returns an empty (but correctly-columned) DataFrame if Alpaca has
        no bars for the requested window (e.g. a pre-IPO date range or an
        illiquid/halted symbol) rather than raising.
        """
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=parse_timeframe(timeframe),
            start=start,
            end=end,
            feed=self.feed,
        )
        bar_set = self.client.call_with_retry(self.client.data_client.get_stock_bars, request)
        bars = bar_set.data.get(symbol, [])
        if not bars:
            logger.warning("No bars returned for %s between %s and %s", symbol, start, end)
        return _bars_to_dataframe(bars)

    def get_latest_bar(self, symbol: str) -> pd.Series:
        """Fetch the most recent bar for a symbol."""
        request = StockLatestBarRequest(symbol_or_symbols=symbol, feed=self.feed)
        result = self.client.call_with_retry(self.client.data_client.get_stock_latest_bar, request)
        bar = result[symbol]
        return pd.Series(
            {"open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume},
            name=bar.timestamp,
        )

    def get_latest_quote(self, symbol: str) -> dict:
        """Fetch the latest bid/ask quote for a symbol."""
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self.feed)
        result = self.client.call_with_retry(self.client.data_client.get_stock_latest_quote, request)
        quote: Quote = result[symbol]
        return {
            "symbol": symbol,
            "timestamp": quote.timestamp,
            "bid_price": quote.bid_price,
            "bid_size": quote.bid_size,
            "ask_price": quote.ask_price,
            "ask_size": quote.ask_size,
        }

    def get_snapshot(self, symbol: str) -> dict:
        """Fetch a full snapshot (latest trade/quote/minute bar/daily bar) for a symbol."""
        request = StockSnapshotRequest(symbol_or_symbols=symbol, feed=self.feed)
        result = self.client.call_with_retry(self.client.data_client.get_stock_snapshot, request)
        snapshot: Snapshot = result[symbol]
        return snapshot.model_dump()

    def spread_pct(self, symbol: str) -> Optional[float]:
        """Bid-ask spread as a fraction of the mid price, for order-validation checks."""
        quote = self.get_latest_quote(symbol)
        bid, ask = quote["bid_price"], quote["ask_price"]
        if not bid or not ask:
            return None
        mid = (bid + ask) / 2
        return (ask - bid) / mid if mid else None

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: str,
        callback: Callable[[pd.Series], Union[None, Awaitable[None]]],
    ) -> None:
        """Subscribe to a real-time bar stream for the given symbols.

        ``timeframe`` is accepted for interface symmetry with
        ``get_historical_bars``, but Alpaca's live bar stream only emits
        1-minute bars; requesting a coarser timeframe here means the
        caller is responsible for resampling.
        """
        stream = self._ensure_stream()

        async def _handler(bar) -> None:
            series = pd.Series(
                {"open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume},
                name=bar.timestamp,
            )
            result = callback(series)
            if hasattr(result, "__await__"):
                await result

        stream.subscribe_bars(_handler, *symbols)

    def subscribe_quotes(
        self, symbols: list[str], callback: Callable[[dict], Union[None, Awaitable[None]]]
    ) -> None:
        """Subscribe to a real-time quote stream (for spread checks) for the given symbols."""
        stream = self._ensure_stream()

        async def _handler(quote) -> None:
            payload = {
                "symbol": quote.symbol,
                "timestamp": quote.timestamp,
                "bid_price": quote.bid_price,
                "bid_size": quote.bid_size,
                "ask_price": quote.ask_price,
                "ask_size": quote.ask_size,
            }
            result = callback(payload)
            if hasattr(result, "__await__"):
                await result

        stream.subscribe_quotes(_handler, *symbols)

    def run_stream(self) -> None:
        """Start processing the subscribed streams. Blocking — run this in
        its own thread/process once all desired subscribe_* calls are made."""
        if self._stream is None:
            raise RuntimeError("No subscriptions registered; call subscribe_bars/subscribe_quotes first")
        self._stream.run()

    def stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()

    def _ensure_stream(self) -> StockDataStream:
        if self._stream is None:
            self._stream = StockDataStream(self.client.api_key, self.client.secret_key, feed=self.feed)
        return self._stream
