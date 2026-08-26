"""Tests for broker.alpaca_client.AlpacaClient."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError

import broker.alpaca_client as alpaca_client_module
from broker.alpaca_client import (
    LIVE_BASE_URL,
    LIVE_TRADING_CONFIRMATION_PHRASE,
    PAPER_BASE_URL,
    AlpacaClient,
    LiveTradingNotConfirmedError,
)
from tests.conftest import make_mocked_alpaca_client


def test_paper_is_the_default_base_url(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    assert client.paper is True
    assert client.base_url == PAPER_BASE_URL


def test_live_trading_without_confirmation_prompts_and_rejects_wrong_answer(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_client_module, "TradingClient", MagicMock())
    monkeypatch.setattr(alpaca_client_module, "StockHistoricalDataClient", MagicMock())
    monkeypatch.setattr("builtins.input", lambda *_: "nope")

    with pytest.raises(LiveTradingNotConfirmedError):
        AlpacaClient("key", "secret", paper=False, run_health_check=False)


def test_live_trading_rejects_wrong_explicit_phrase(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_client_module, "TradingClient", MagicMock())
    monkeypatch.setattr(alpaca_client_module, "StockHistoricalDataClient", MagicMock())

    with pytest.raises(LiveTradingNotConfirmedError):
        AlpacaClient("key", "secret", paper=False, confirm_live_trading="close enough", run_health_check=False)


def test_live_trading_with_correct_confirmation_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_client_module, "TradingClient", MagicMock())
    monkeypatch.setattr(alpaca_client_module, "StockHistoricalDataClient", MagicMock())

    client = AlpacaClient(
        "key", "secret", paper=False, confirm_live_trading=LIVE_TRADING_CONFIRMATION_PHRASE, run_health_check=False
    )
    assert client.base_url == LIVE_BASE_URL


def test_get_account_returns_plain_dict(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch, equity="123456")
    account = client.get_account()
    assert account["equity"] == "123456"


def test_get_positions_returns_list_of_dicts(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    mock_position = MagicMock()
    mock_position.model_dump.return_value = {"symbol": "AAPL", "qty": "10"}
    client.trading_client.get_all_positions.return_value = [mock_position]

    positions = client.get_positions()
    assert positions == [{"symbol": "AAPL", "qty": "10"}]


def test_is_market_open_reflects_clock(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    closed_clock = MagicMock()
    closed_clock.model_dump.return_value = {"is_open": False}
    client.trading_client.get_clock.return_value = closed_clock

    assert client.is_market_open() is False


def test_get_available_margin_reads_buying_power(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch, buying_power="42000")
    assert client.get_available_margin() == pytest.approx(42000.0)


def test_call_with_retry_retries_transient_errors_then_succeeds(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    client.backoff_base_seconds = 0.0
    monkeypatch.setattr(alpaca_client_module.time, "sleep", lambda *_: None)

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RequestsConnectionError("boom")
        return "ok"

    assert client.call_with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_call_with_retry_raises_after_exhausting_attempts(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    client.max_retries = 3
    client.backoff_base_seconds = 0.0
    monkeypatch.setattr(alpaca_client_module.time, "sleep", lambda *_: None)

    def always_fails():
        raise RequestsConnectionError("still broken")

    with pytest.raises(ConnectionError):
        client.call_with_retry(always_fails)


def _make_api_error(status_code: int):
    from alpaca.common.exceptions import APIError

    http_error = MagicMock()
    http_error.response.status_code = status_code
    return APIError('{"code": 40010001, "message": "bad request"}', http_error)


def test_call_with_retry_does_not_retry_non_retryable_api_error(monkeypatch) -> None:
    from alpaca.common.exceptions import APIError

    client = make_mocked_alpaca_client(monkeypatch)
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise _make_api_error(400)

    with pytest.raises(APIError):
        client.call_with_retry(bad_request)
    assert calls["n"] == 1  # not retried


def test_call_with_retry_retries_5xx_api_errors(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    client.backoff_base_seconds = 0.0
    monkeypatch.setattr(alpaca_client_module.time, "sleep", lambda *_: None)

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _make_api_error(503)
        return "ok"

    assert client.call_with_retry(flaky) == "ok"
    assert calls["n"] == 2


def test_health_check_raises_if_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(alpaca_client_module, "TradingClient", MagicMock())
    monkeypatch.setattr(alpaca_client_module, "StockHistoricalDataClient", MagicMock())
    monkeypatch.setattr(alpaca_client_module.time, "sleep", lambda *_: None)

    unreachable_trading_client = alpaca_client_module.TradingClient.return_value
    unreachable_trading_client.get_clock.side_effect = RequestsConnectionError("no network")

    with pytest.raises(ConnectionError):
        AlpacaClient("key", "secret", paper=True, max_retries=2, backoff_base_seconds=0.0)
