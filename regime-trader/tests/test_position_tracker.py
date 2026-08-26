"""Tests for broker.position_tracker.PositionTracker."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from broker.position_tracker import PositionTracker, TrackedPosition
from core.risk_manager import CircuitBreaker
from tests.conftest import make_mocked_alpaca_client
from tests.test_risk import make_config  # reuse the same RiskConfig defaults


def _mock_alpaca_position(symbol, qty, avg_entry_price, current_price):
    p = MagicMock()
    p.model_dump.return_value = {
        "symbol": symbol,
        "qty": str(qty),
        "avg_entry_price": str(avg_entry_price),
        "current_price": str(current_price),
    }
    return p


def test_sync_adopts_untracked_alpaca_position(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    client.trading_client.get_all_positions.return_value = [_mock_alpaca_position("AAPL", 10, 150.0, 155.0)]
    tracker = PositionTracker(client)

    tracker.sync_with_alpaca()

    position = tracker.get_position("AAPL")
    assert position is not None
    assert position.quantity == 10.0
    assert position.entry_price == 150.0
    assert position.regime_at_entry is None  # unknown context, honestly recorded as such


def test_sync_drops_locally_tracked_position_alpaca_no_longer_shows(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    client.trading_client.get_all_positions.return_value = []
    tracker = PositionTracker(client)
    tracker.register_entry("AAPL", 10, 150.0, regime_at_entry="BULL")

    tracker.sync_with_alpaca()

    assert tracker.get_position("AAPL") is None


def test_sync_updates_price_and_qty_for_already_tracked_position(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    client.trading_client.get_all_positions.return_value = [_mock_alpaca_position("AAPL", 12, 150.0, 158.0)]
    tracker = PositionTracker(client)
    tracker.register_entry("AAPL", 10, 150.0, regime_at_entry="BULL")

    tracker.sync_with_alpaca()

    position = tracker.get_position("AAPL")
    assert position.quantity == 12.0
    assert position.current_price == 158.0
    assert position.regime_at_entry == "BULL"  # locally-known context is preserved


def test_register_entry_creates_tracked_position(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    tracker = PositionTracker(client)

    tracker.register_entry("MSFT", 10, 300.0, stop_level=290.0, regime_at_entry="BULL", sector="Tech")

    position = tracker.get_position("MSFT")
    assert position == TrackedPosition(
        symbol="MSFT",
        quantity=10,
        entry_price=300.0,
        entry_time=position.entry_time,
        current_price=300.0,
        stop_level=290.0,
        regime_at_entry="BULL",
        regime_current="BULL",
        sector="Tech",
    )


def test_buy_fill_opens_new_position(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    tracker = PositionTracker(client)

    tracker.handle_trade_update({"event": "fill", "order": {"symbol": "AAPL", "side": "buy"}, "qty": "10", "price": "150.0"})

    position = tracker.get_position("AAPL")
    assert position.quantity == 10.0
    assert position.entry_price == 150.0
    assert tracker.trades_today == 1


def test_buy_fill_updates_weighted_average_entry(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    tracker = PositionTracker(client)
    tracker.register_entry("AAPL", 10, 150.0)

    tracker.handle_trade_update({"event": "fill", "order": {"symbol": "AAPL", "side": "buy"}, "qty": "5", "price": "160.0"})

    position = tracker.get_position("AAPL")
    assert position.quantity == 15.0
    assert position.entry_price == pytest.approx((150.0 * 10 + 160.0 * 5) / 15)


def test_sell_fill_reduces_position(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    tracker = PositionTracker(client)
    tracker.register_entry("AAPL", 15, 150.0)

    tracker.handle_trade_update({"event": "fill", "order": {"symbol": "AAPL", "side": "sell"}, "qty": "5", "price": "165.0"})

    position = tracker.get_position("AAPL")
    assert position.quantity == 10.0
    assert position.current_price == 165.0


def test_sell_fill_closing_full_quantity_removes_position(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    tracker = PositionTracker(client)
    tracker.register_entry("AAPL", 15, 150.0)

    tracker.handle_trade_update({"event": "fill", "order": {"symbol": "AAPL", "side": "sell"}, "qty": "15", "price": "165.0"})

    assert tracker.get_position("AAPL") is None


def test_sell_fill_on_untracked_symbol_is_ignored(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    tracker = PositionTracker(client)

    tracker.handle_trade_update({"event": "fill", "order": {"symbol": "AAPL", "side": "sell"}, "qty": "5", "price": "165.0"})

    assert tracker.get_position("AAPL") is None
    assert tracker.trades_today == 0


def test_non_fill_event_is_ignored(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    tracker = PositionTracker(client)

    tracker.handle_trade_update({"event": "new", "order": {"symbol": "AAPL", "side": "buy"}})

    assert tracker.get_position("AAPL") is None
    assert tracker.trades_today == 0


def test_get_total_exposure(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch)
    tracker = PositionTracker(client)
    tracker.register_entry("AAPL", 10, 100.0)  # market_value = 1000
    tracker.get_position("AAPL").current_price = 100.0

    assert tracker.get_total_exposure(equity=10_000.0) == pytest.approx(0.10)


def test_daily_and_weekly_pnl(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch, equity="105000")
    tracker = PositionTracker(client)
    tracker._daily_start_equity = 100_000.0
    tracker._weekly_start_equity = 95_000.0

    assert tracker.get_daily_pnl() == pytest.approx(0.05)
    assert tracker.get_weekly_pnl() == pytest.approx(105_000.0 / 95_000.0 - 1.0)


def test_reset_daily_clears_trade_count_and_recent_orders(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch, equity="101000")
    tracker = PositionTracker(client)
    tracker.trades_today = 5
    tracker.register_order_submitted("AAPL", "LONG")

    tracker.reset_daily()

    assert tracker.trades_today == 0
    assert tracker.recent_orders == []
    assert tracker._daily_start_equity == 101_000.0


def test_reset_daily_and_weekly_reset_their_circuit_breaker(monkeypatch, tmp_path) -> None:
    client = make_mocked_alpaca_client(monkeypatch, equity="100000")
    cb = CircuitBreaker(make_config(), lock_file_path=tmp_path / "trading_halted.lock")
    tracker = PositionTracker(client, circuit_breaker=cb)
    cb.state.daily_halt_active = True
    cb.state.weekly_halt_active = True

    tracker.reset_daily()
    assert cb.state.daily_halt_active is False
    assert cb.state.weekly_halt_active is True  # untouched by reset_daily

    tracker.reset_weekly()
    assert cb.state.weekly_halt_active is False


def test_get_portfolio_state_reflects_tracked_positions(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch, equity="100000", cash="50000", buying_power="80000")
    tracker = PositionTracker(client)
    tracker.register_entry("MSFT", 10, 300.0)

    state = tracker.get_portfolio_state()

    assert state.equity == 100_000.0
    assert state.cash == 50_000.0
    assert state.buying_power == 80_000.0
    assert "MSFT" in state.positions
    assert state.positions["MSFT"].market_value == pytest.approx(3000.0)
    assert state.peak_equity == 100_000.0
    assert state.drawdown_from_peak_pct == 0.0


def test_get_portfolio_state_tracks_peak_equity_and_drawdown(monkeypatch) -> None:
    client = make_mocked_alpaca_client(monkeypatch, equity="100000")
    tracker = PositionTracker(client)
    tracker.get_portfolio_state()  # peak = 100000

    account = client.trading_client.get_account.return_value
    account.model_dump.return_value = {"equity": "90000", "cash": "90000", "buying_power": "90000"}

    state = tracker.get_portfolio_state()

    assert state.peak_equity == 100_000.0
    assert state.drawdown_from_peak_pct == pytest.approx(90_000.0 / 100_000.0 - 1.0)


def test_circuit_breaker_updated_on_fill(monkeypatch, tmp_path) -> None:
    client = make_mocked_alpaca_client(monkeypatch, equity="100000")
    cb = CircuitBreaker(make_config(), lock_file_path=tmp_path / "trading_halted.lock")
    tracker = PositionTracker(client, circuit_breaker=cb)
    tracker._daily_start_equity = 100_000.0

    account = client.trading_client.get_account.return_value
    account.model_dump.return_value = {"equity": "96000", "cash": "96000", "buying_power": "96000"}  # -4% daily DD

    tracker.handle_trade_update({"event": "fill", "order": {"symbol": "AAPL", "side": "buy"}, "qty": "1", "price": "100.0"})

    assert cb.state.daily_halt_active is True
