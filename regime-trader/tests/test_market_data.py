"""Tests for data.market_data.MarketDataClient."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import pytest
from alpaca.data.models.bars import Bar
from alpaca.data.models.quotes import Quote
from alpaca.data.timeframe import TimeFrameUnit

from data.market_data import MarketDataClient, _bars_to_dataframe, parse_timeframe
from tests.conftest import make_mocked_alpaca_client


@pytest.mark.parametrize(
    "timeframe_str,expected_amount,expected_unit",
    [
        ("1Day", 1, TimeFrameUnit.Day),
        ("5Min", 5, TimeFrameUnit.Minute),
        ("1Hour", 1, TimeFrameUnit.Hour),
        ("1Week", 1, TimeFrameUnit.Week),
    ],
)
def test_parse_timeframe(timeframe_str, expected_amount, expected_unit) -> None:
    tf = parse_timeframe(timeframe_str)
    assert tf.amount == expected_amount
    assert tf.unit == expected_unit


def test_parse_timeframe_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError):
        parse_timeframe("1Fortnight")


def _make_bar(timestamp, o=100.0, h=101.0, l=99.0, c=100.5, v=1000) -> Bar:
    raw = {"t": timestamp, "o": o, "h": h, "l": l, "c": c, "v": v, "n": 10, "vw": 100.2}
    return Bar("TEST", raw)


def test_bars_to_dataframe_empty() -> None:
    df = _bars_to_dataframe([])
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_bars_to_dataframe_populated_and_sorted() -> None:
    bars = [
        _make_bar(datetime.datetime(2024, 1, 3), c=102.0),
        _make_bar(datetime.datetime(2024, 1, 2), c=101.0),
    ]
    df = _bars_to_dataframe(bars)
    assert list(df.index) == [datetime.datetime(2024, 1, 2), datetime.datetime(2024, 1, 3)]
    assert df.loc[datetime.datetime(2024, 1, 2), "close"] == 101.0


def _make_bar_set(data: dict[str, list[Bar]]) -> MagicMock:
    """A stand-in for alpaca's BarSet exposing just the .data attribute our
    code reads — constructing a real BarSet from pre-built Bar objects
    isn't supported (it expects raw dicts and builds the Bars itself)."""
    bar_set = MagicMock()
    bar_set.data = data
    return bar_set


def test_get_historical_bars_returns_dataframe(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    md = MarketDataClient(client)
    client.data_client.get_stock_bars.return_value = _make_bar_set(
        {"AAPL": [_make_bar(datetime.datetime(2024, 1, 2))]}
    )

    df = md.get_historical_bars("AAPL", "1Day", datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 3))

    assert len(df) == 1
    assert df.iloc[0]["close"] == 100.5


def test_get_historical_bars_empty_response_is_graceful(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    md = MarketDataClient(client)
    client.data_client.get_stock_bars.return_value = _make_bar_set({"AAPL": []})

    df = md.get_historical_bars("AAPL", "1Day", datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 3))

    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_get_latest_bar(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    md = MarketDataClient(client)
    bar = _make_bar(datetime.datetime(2024, 1, 5), c=123.45)
    client.data_client.get_stock_latest_bar.return_value = {"AAPL": bar}

    series = md.get_latest_bar("AAPL")

    assert series["close"] == 123.45
    assert series.name == datetime.datetime(2024, 1, 5)


def test_get_latest_quote(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    md = MarketDataClient(client)
    raw = {"t": datetime.datetime(2024, 1, 5), "bp": 99.9, "bs": 10, "ap": 100.1, "as": 5, "bx": "N", "ax": "N", "c": []}
    quote = Quote("AAPL", raw)
    client.data_client.get_stock_latest_quote.return_value = {"AAPL": quote}

    result = md.get_latest_quote("AAPL")

    assert result["bid_price"] == 99.9
    assert result["ask_price"] == 100.1


def test_spread_pct_computes_relative_spread(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    md = MarketDataClient(client)
    raw = {"t": datetime.datetime(2024, 1, 5), "bp": 99.5, "bs": 10, "ap": 100.5, "as": 5, "bx": "N", "ax": "N", "c": []}
    quote = Quote("AAPL", raw)
    client.data_client.get_stock_latest_quote.return_value = {"AAPL": quote}

    spread = md.spread_pct("AAPL")

    assert spread == pytest.approx((100.5 - 99.5) / 100.0)


def test_spread_pct_none_when_side_missing(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    md = MarketDataClient(client)
    raw = {"t": datetime.datetime(2024, 1, 5), "bp": 0.0, "bs": 0, "ap": 100.5, "as": 5, "bx": "N", "ax": "N", "c": []}
    quote = Quote("AAPL", raw)
    client.data_client.get_stock_latest_quote.return_value = {"AAPL": quote}

    assert md.spread_pct("AAPL") is None
