"""Tests for broker.order_executor.OrderExecutor."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from alpaca.trading.enums import OrderSide, OrderStatus, OrderType

import broker.order_executor as order_executor_module
from broker.order_executor import OrderExecutor, _quantity_for_signal
from core.regime_strategies import Signal
from tests.conftest import make_mocked_alpaca_client


def make_signal(**overrides) -> Signal:
    base = dict(
        symbol="AAPL",
        direction="LONG",
        confidence=0.9,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=None,
        position_size_pct=0.5,
        leverage=1.0,
        regime_id=0,
        regime_name="BULL",
        regime_probability=0.9,
        timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        reasoning="test",
        strategy_name="TestStrategy",
        metadata={},
    )
    base.update(overrides)
    return Signal(**base)


def _mock_order(**overrides):
    order = MagicMock()
    order.id = overrides.get("id", "order-1")
    order.client_order_id = overrides.get("client_order_id")
    order.symbol = overrides.get("symbol", "AAPL")
    order.side = overrides.get("side", OrderSide.BUY)
    order.qty = overrides.get("qty", "500")
    order.order_type = overrides.get("order_type", OrderType.LIMIT)
    order.status = overrides.get("status", OrderStatus.FILLED)
    order.limit_price = overrides.get("limit_price", "100.10")
    order.stop_price = overrides.get("stop_price", None)
    order.filled_qty = overrides.get("filled_qty", "500")
    order.filled_avg_price = overrides.get("filled_avg_price", "100.05")
    order.legs = overrides.get("legs", None)
    return order


def test_quantity_for_signal_uses_position_pct_leverage_and_equity() -> None:
    signal = make_signal(position_size_pct=0.5, leverage=1.0, entry_price=100.0)
    assert _quantity_for_signal(signal, equity=100_000.0) == 500.0


def test_quantity_for_signal_floors_to_whole_shares() -> None:
    signal = make_signal(position_size_pct=0.5001, leverage=1.0, entry_price=100.0)
    assert _quantity_for_signal(signal, equity=1000.0) == 5.0  # 500.1/100 floored


def test_submit_order_rejects_flat_direction(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    with pytest.raises(ValueError):
        executor.submit_order(make_signal(direction="FLAT"))


def test_submit_order_limit_price_and_client_order_id(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    submitted = _mock_order(status=OrderStatus.NEW)
    filled = _mock_order(status=OrderStatus.FILLED)
    client.trading_client.submit_order.return_value = submitted
    client.trading_client.get_order_by_id.return_value = filled

    result = executor.submit_order(make_signal(entry_price=100.0), trade_id="trade-abc")

    assert result.status == "filled"
    assert result.trade_id == "trade-abc"
    request = client.trading_client.submit_order.call_args[0][0]
    assert request.limit_price == pytest.approx(100.10)  # +0.1% for a buy
    assert request.client_order_id == "trade-abc"


def test_submit_order_sell_limit_price_is_below_entry(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    client.trading_client.submit_order.return_value = _mock_order(status=OrderStatus.NEW)
    client.trading_client.get_order_by_id.return_value = _mock_order(status=OrderStatus.FILLED)

    executor.submit_order(make_signal(direction="LONG", entry_price=100.0))
    # LONG signals always submit a BUY to open; verify SELL pricing directly via the helper instead.
    assert executor._limit_price_for(100.0, OrderSide.SELL) == pytest.approx(99.90)


def test_submit_order_generates_trade_id_when_not_supplied(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    client.trading_client.submit_order.return_value = _mock_order(status=OrderStatus.NEW)
    client.trading_client.get_order_by_id.return_value = _mock_order(status=OrderStatus.FILLED)

    result = executor.submit_order(make_signal())

    assert result.trade_id  # non-empty
    request = client.trading_client.submit_order.call_args[0][0]
    assert request.client_order_id == result.trade_id


def test_submit_order_cancels_after_timeout(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client, unfilled_timeout_seconds=30.0, poll_interval_seconds=2.0)

    # Deterministic clock: deadline computed at 0.0 (-> 30.0), then the very
    # first expiry check reports 40.0 (already past it).
    clock = iter([0.0, 40.0])
    monkeypatch.setattr(order_executor_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(order_executor_module.time, "sleep", lambda *_: None)

    client.trading_client.submit_order.return_value = _mock_order(status=OrderStatus.NEW)
    client.trading_client.get_order_by_id.side_effect = [
        _mock_order(status=OrderStatus.NEW),  # first poll: still open
        _mock_order(status=OrderStatus.CANCELED),  # re-fetched after cancel
    ]

    result = executor.submit_order(make_signal(), retry_at_market=False)

    assert result.status == "canceled"
    client.trading_client.cancel_order_by_id.assert_called_once()


def test_submit_order_retries_remainder_at_market_after_timeout(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client, unfilled_timeout_seconds=30.0, poll_interval_seconds=2.0)

    # Limit wait: deadline at 0.0 (-> 30.0), expiry check reports 40.0 (past it).
    # Market wait: deadline recomputed at 40.0 (-> 70.0), expiry check reports
    # 45.0 (not past it) so the loop proceeds to poll and picks up the fill.
    clock = iter([0.0, 40.0, 40.0, 45.0])
    monkeypatch.setattr(order_executor_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(order_executor_module.time, "sleep", lambda *_: None)

    limit_order = _mock_order(id="limit-1", status=OrderStatus.NEW)
    partially_canceled = _mock_order(id="limit-1", status=OrderStatus.CANCELED, filled_qty="200")
    market_order = _mock_order(id="mkt-1", status=OrderStatus.NEW)
    market_filled = _mock_order(id="mkt-1", status=OrderStatus.FILLED, filled_qty="300")

    client.trading_client.submit_order.side_effect = [limit_order, market_order]
    client.trading_client.get_order_by_id.side_effect = [
        _mock_order(id="limit-1", status=OrderStatus.NEW),  # first poll
        partially_canceled,  # after cancel
        _mock_order(id="mkt-1", status=OrderStatus.NEW),  # market order poll
        market_filled,
    ]

    result = executor.submit_order(make_signal(), retry_at_market=True)

    assert result.status == "filled"
    assert client.trading_client.submit_order.call_count == 2
    market_request = client.trading_client.submit_order.call_args_list[1][0][0]
    assert market_request.qty == 300  # 500 - 200 already filled


def test_submit_bracket_order_requires_stop_loss(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    with pytest.raises(ValueError):
        executor.submit_bracket_order(make_signal(stop_loss=None))


def test_submit_bracket_order_derives_take_profit_when_missing(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    client.trading_client.submit_order.return_value = _mock_order(status=OrderStatus.NEW)

    executor.submit_bracket_order(make_signal(entry_price=100.0, stop_loss=95.0, take_profit=None))

    request = client.trading_client.submit_order.call_args[0][0]
    # default 2:1 reward:risk -> entry + 2*(entry-stop) = 100 + 2*5 = 110
    assert request.take_profit.limit_price == pytest.approx(110.0)
    assert request.stop_loss.stop_price == pytest.approx(95.0)


def test_submit_bracket_order_uses_explicit_take_profit(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    client.trading_client.submit_order.return_value = _mock_order(status=OrderStatus.NEW)

    executor.submit_bracket_order(make_signal(entry_price=100.0, stop_loss=95.0), take_profit_price=120.0)

    request = client.trading_client.submit_order.call_args[0][0]
    assert request.take_profit.limit_price == pytest.approx(120.0)


def test_modify_stop_tightens_long_position(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    position = MagicMock()
    position.model_dump.return_value = {"symbol": "AAPL", "side": "long"}
    client.trading_client.get_all_positions.return_value = [position]
    client.trading_client.get_orders.return_value = [_mock_order(stop_price=95.0)]

    assert executor.modify_stop("AAPL", 97.0) is True
    client.trading_client.replace_order_by_id.assert_called_once()


def test_modify_stop_refuses_to_widen_long_position(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    position = MagicMock()
    position.model_dump.return_value = {"symbol": "AAPL", "side": "long"}
    client.trading_client.get_all_positions.return_value = [position]
    client.trading_client.get_orders.return_value = [_mock_order(stop_price=95.0)]

    assert executor.modify_stop("AAPL", 90.0) is False
    client.trading_client.replace_order_by_id.assert_not_called()


def test_modify_stop_no_position_returns_false(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    client.trading_client.get_all_positions.return_value = []

    assert executor.modify_stop("AAPL", 97.0) is False


def test_modify_stop_no_stop_order_returns_false(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    position = MagicMock()
    position.model_dump.return_value = {"symbol": "AAPL", "side": "long"}
    client.trading_client.get_all_positions.return_value = [position]
    client.trading_client.get_orders.return_value = []

    assert executor.modify_stop("AAPL", 97.0) is False


def test_cancel_order_returns_true_on_success(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    assert executor.cancel_order("order-1") is True
    client.trading_client.cancel_order_by_id.assert_called_once_with("order-1")


def test_cancel_order_returns_false_on_failure(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    client.trading_client.cancel_order_by_id.side_effect = Exception("already filled")

    assert executor.cancel_order("order-1") is False


def test_close_position(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    client.trading_client.close_position.return_value = _mock_order(status=OrderStatus.NEW, client_order_id="c1")

    result = executor.close_position("AAPL")

    assert result.symbol == "AAPL"
    client.trading_client.close_position.assert_called_once_with("AAPL")


def test_close_all_positions(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    response = MagicMock()
    response.model_dump.return_value = {"symbol": "AAPL", "status": 200}
    client.trading_client.close_all_positions.return_value = [response]

    results = executor.close_all_positions()

    assert results == [{"symbol": "AAPL", "status": 200}]
    client.trading_client.close_all_positions.assert_called_once_with(cancel_orders=True)


def test_get_order_status(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    executor = OrderExecutor(client)
    client.trading_client.get_order_by_id.return_value = _mock_order(status=OrderStatus.PARTIALLY_FILLED)

    assert executor.get_order_status("order-1") == "partially_filled"
