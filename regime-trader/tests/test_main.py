"""Tests for main.py's TradingSession orchestration (Phase 7).

Every collaborator (AlpacaClient, MarketDataClient, OrderExecutor,
PositionTracker, RiskManager) is either the mocked AlpacaClient helper
from conftest or a MagicMock, so these tests exercise the actual
orchestration logic without any network access.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import main as main_module
from broker.position_tracker import PositionTracker
from core.regime_strategies import Signal
from tests.conftest import make_mocked_alpaca_client

FAST_HMM_OVERRIDES = {"n_candidates": [3, 4], "n_init": 2, "min_train_bars": 100}


def make_config(tmp_path, **overrides) -> dict:
    config = main_module.load_config("config/settings.yaml")
    config["hmm"].update(FAST_HMM_OVERRIDES)
    config["live"]["model_dir"] = str(tmp_path / "models")
    config["live"]["state_snapshot_path"] = str(tmp_path / "state_snapshot.json")
    config["live"]["lock_file_path"] = str(tmp_path / "trading_halted.lock")
    config["live"]["max_market_wait_seconds"] = 60
    for section, values in overrides.items():
        config[section].update(values)
    return config


def make_daily_bars(n: int = 900, seed: int = 3, end: datetime | None = None) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range(end=(end or datetime.now(timezone.utc)).date(), periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    volume = rng.randint(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def make_session(monkeypatch, tmp_path, symbols=("TEST",), dry_run=False, config=None):
    config = config or make_config(tmp_path)
    client = make_mocked_alpaca_client(monkeypatch)
    open_clock = MagicMock()
    open_clock.model_dump.return_value = {"is_open": True, "next_open": None}
    client.trading_client.get_clock.return_value = open_clock
    client.trading_client.get_all_positions.return_value = []

    market_data = MagicMock()
    market_data.get_historical_bars.return_value = make_daily_bars()
    order_executor = MagicMock()
    risk_manager = main_module.build_risk_manager(config, lock_file_path=config["live"]["lock_file_path"])
    position_tracker = PositionTracker(client, circuit_breaker=risk_manager.circuit_breaker)

    session = main_module.TradingSession(
        config, client, market_data, order_executor, position_tracker, risk_manager, list(symbols), dry_run=dry_run
    )
    return session


def make_signal(**overrides) -> Signal:
    base = dict(
        symbol="TEST",
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
        timestamp=datetime.now(timezone.utc),
        reasoning="test",
        strategy_name="TestStrategy",
        metadata={},
    )
    base.update(overrides)
    return Signal(**base)


# ----------------------------------------------------------------------
# Market hours
# ----------------------------------------------------------------------


def test_check_market_hours_returns_true_when_open(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    assert session.check_market_hours() is True


def test_check_market_hours_waits_then_rechecks(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    next_open = datetime.now(timezone.utc) + timedelta(seconds=30)
    closed_clock = MagicMock()
    closed_clock.model_dump.return_value = {"is_open": False, "next_open": next_open}
    session.client.trading_client.get_clock.return_value = closed_clock

    sleep_calls = []
    monkeypatch.setattr(main_module.time, "sleep", lambda s: sleep_calls.append(s))

    open_clock = MagicMock()
    open_clock.model_dump.return_value = {"is_open": True}
    session.client.trading_client.get_account.return_value.model_dump.return_value = {
        "equity": "100000", "cash": "50000", "buying_power": "80000",
    }
    # After the wait, is_market_open() re-checks the clock — simulate it now being open.
    call_count = {"n": 0}

    def get_clock_side_effect():
        call_count["n"] += 1
        return closed_clock if call_count["n"] == 1 else open_clock

    session.client.trading_client.get_clock.side_effect = get_clock_side_effect

    assert session.check_market_hours() is True
    assert sleep_calls == [pytest.approx(30, abs=1)]


def test_check_market_hours_returns_false_with_no_wait_window(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    closed_clock = MagicMock()
    closed_clock.model_dump.return_value = {"is_open": False, "next_open": None}
    session.client.trading_client.get_clock.return_value = closed_clock

    assert session.check_market_hours() is False


# ----------------------------------------------------------------------
# HMM load/train/retrain
# ----------------------------------------------------------------------


def test_load_or_train_hmm_trains_fresh_when_no_saved_model(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    bars = make_daily_bars()

    engine = session.load_or_train_hmm("TEST", bars)

    assert engine.n_regimes in (3, 4)
    assert (session.model_dir / "TEST_hmm.pkl").exists()
    assert "TEST" in session.hmm_last_trained


def test_load_or_train_hmm_loads_cached_fresh_model(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    bars = make_daily_bars()
    session.load_or_train_hmm("TEST", bars)
    saved_mtime = (session.model_dir / "TEST_hmm.pkl").stat().st_mtime

    session.load_or_train_hmm("TEST", bars)  # should load, not retrain

    assert (session.model_dir / "TEST_hmm.pkl").stat().st_mtime == saved_mtime


def test_load_or_train_hmm_retrains_when_stale(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    bars = make_daily_bars()
    session.load_or_train_hmm("TEST", bars)
    path = session.model_dir / "TEST_hmm.pkl"

    import pickle

    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload["training_metadata"]["training_date"] = datetime.now(timezone.utc) - timedelta(days=10)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    stale_mtime = path.stat().st_mtime

    session.load_or_train_hmm("TEST", bars)

    assert path.stat().st_mtime != stale_mtime  # retrained and re-saved


def test_retrain_if_due_skips_when_recently_trained(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    bars = make_daily_bars()
    session.engines["TEST"] = session.load_or_train_hmm("TEST", bars)
    original_engine = session.engines["TEST"]

    session.retrain_if_due("TEST", bars)

    assert session.engines["TEST"] is original_engine


def test_retrain_if_due_retrains_when_never_trained(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    bars = make_daily_bars()

    session.retrain_if_due("TEST", bars)

    assert "TEST" in session.engines
    assert "TEST" in session.orchestrators


# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------


def test_startup_returns_false_when_market_closed(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    closed_clock = MagicMock()
    closed_clock.model_dump.return_value = {"is_open": False, "next_open": None}
    session.client.trading_client.get_clock.return_value = closed_clock

    assert session.startup() is False
    assert session.engines == {}


def test_startup_trains_engines_and_syncs_positions(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)

    assert session.startup() is True
    assert "TEST" in session.engines
    assert "TEST" in session.orchestrators
    session.client.trading_client.get_all_positions.assert_called()


# ----------------------------------------------------------------------
# on_bar / signal pipeline
# ----------------------------------------------------------------------


def test_on_bar_new_trading_day_generates_and_submits_signal(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    assert session.startup() is True

    session.order_executor.submit_bracket_order.return_value = MagicMock(status="new")
    bar = pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000}, name=datetime.now(timezone.utc))

    session.on_bar("TEST", bar)

    assert session._last_regime_state.get("TEST") is not None
    assert session.order_executor.submit_bracket_order.called


def test_on_bar_dry_run_never_submits_orders(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path, dry_run=True)
    assert session.startup() is True
    bar = pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000}, name=datetime.now(timezone.utc))

    session.on_bar("TEST", bar)

    assert not session.order_executor.submit_bracket_order.called


def test_on_bar_only_processes_regime_once_per_day(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    assert session.startup() is True
    now = datetime.now(timezone.utc)
    bar1 = pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000}, name=now)
    bar2 = pd.Series(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000}, name=now + timedelta(minutes=5)
    )

    session.on_bar("TEST", bar1)
    session.market_data.get_historical_bars.reset_mock()
    session.on_bar("TEST", bar2)

    session.market_data.get_historical_bars.assert_not_called()  # no new daily fetch on the same day


def test_on_bar_hmm_error_holds_current_regime(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    assert session.startup() is True
    session.engines["TEST"].predict_regime_filtered = MagicMock(side_effect=RuntimeError("boom"))
    bar = pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000}, name=datetime.now(timezone.utc))

    session.on_bar("TEST", bar)  # must not raise

    assert "TEST" not in session._last_regime_date  # regime day was not advanced
    assert not session.order_executor.submit_bracket_order.called


def test_on_bar_unhandled_error_is_caught_and_state_saved(monkeypatch, tmp_path, caplog) -> None:
    session = make_session(monkeypatch, tmp_path)
    assert session.startup() is True
    session._last_regime_date["TEST"] = datetime.now(timezone.utc).date()  # skip regime path
    session.position_tracker.get_portfolio_state = MagicMock(side_effect=RuntimeError("catastrophic"))
    bar = pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000}, name=datetime.now(timezone.utc))

    session.on_bar("TEST", bar)  # must not raise

    assert session.state_snapshot_path.exists()
    assert "ALERT" in caplog.text


def test_on_bar_stale_feed_pauses_signals_but_housekeeping_runs(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    assert session.startup() is True
    # A bar timestamped well in the past looks like a stale/dropped feed.
    stale_bar_time = datetime.now(timezone.utc) - timedelta(hours=1)
    bar = pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000}, name=stale_bar_time)

    session.on_bar("TEST", bar)

    assert not session.order_executor.submit_bracket_order.called
    assert session.state_snapshot_path.exists()  # housekeeping still ran


# ----------------------------------------------------------------------
# process_signal
# ----------------------------------------------------------------------


def test_process_signal_approved_submits_with_trade_id(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    session.order_executor.submit_bracket_order.return_value = MagicMock(status="new")

    session.process_signal("TEST", make_signal())

    session.order_executor.submit_bracket_order.assert_called_once()
    submitted_signal, kwargs = session.order_executor.submit_bracket_order.call_args
    assert kwargs["trade_id"]  # non-empty


def test_process_signal_rejected_does_not_submit(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    from core.risk_manager import RiskDecision

    session.risk_manager.validate_signal = MagicMock(
        return_value=RiskDecision(approved=False, modified_signal=None, rejection_reason="nope")
    )

    session.process_signal("TEST", make_signal())

    assert not session.order_executor.submit_bracket_order.called


def test_process_signal_dry_run_does_not_submit(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path, dry_run=True)

    session.process_signal("TEST", make_signal())

    assert not session.order_executor.submit_bracket_order.called


# ----------------------------------------------------------------------
# Trailing stops
# ----------------------------------------------------------------------


def test_update_trailing_stop_calls_modify_stop(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    session.position_tracker.register_entry("TEST", 10, 100.0, stop_level=90.0)
    session.order_executor.modify_stop.return_value = True
    bars = make_daily_bars()

    session.update_trailing_stop("TEST", bars)

    session.order_executor.modify_stop.assert_called_once()
    symbol_arg, new_stop_arg = session.order_executor.modify_stop.call_args[0]
    assert symbol_arg == "TEST"


def test_update_trailing_stop_dry_run_skips_modify_stop(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path, dry_run=True)
    session.position_tracker.register_entry("TEST", 10, 100.0, stop_level=90.0)
    bars = make_daily_bars()

    session.update_trailing_stop("TEST", bars)

    assert not session.order_executor.modify_stop.called


def test_update_trailing_stop_no_position_is_a_noop(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    bars = make_daily_bars()

    session.update_trailing_stop("TEST", bars)  # no position registered

    assert not session.order_executor.modify_stop.called


# ----------------------------------------------------------------------
# Circuit breakers
# ----------------------------------------------------------------------


def test_check_circuit_breakers_closes_all_on_new_halt(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    session.client.trading_client.get_account.return_value.model_dump.return_value = {
        "equity": "96000", "cash": "96000", "buying_power": "96000",
    }
    session.position_tracker._daily_start_equity = 100_000.0

    session.check_circuit_breakers("TEST")

    assert session.order_executor.close_all_positions.called


def test_check_circuit_breakers_does_not_reclose_when_already_halted(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    session.client.trading_client.get_account.return_value.model_dump.return_value = {
        "equity": "96000", "cash": "96000", "buying_power": "96000",
    }
    session.position_tracker._daily_start_equity = 100_000.0
    session.check_circuit_breakers("TEST")
    session.order_executor.close_all_positions.reset_mock()

    session.check_circuit_breakers("TEST")  # still halted, but not newly so

    assert not session.order_executor.close_all_positions.called


def test_check_circuit_breakers_dry_run_does_not_close_positions(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path, dry_run=True)
    session.client.trading_client.get_account.return_value.model_dump.return_value = {
        "equity": "96000", "cash": "96000", "buying_power": "96000",
    }
    session.position_tracker._daily_start_equity = 100_000.0

    session.check_circuit_breakers("TEST")

    assert not session.order_executor.close_all_positions.called


# ----------------------------------------------------------------------
# State snapshot
# ----------------------------------------------------------------------


def test_state_snapshot_save_and_restore_round_trip(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    session.position_tracker.trades_today = 3
    session.position_tracker._daily_start_equity = 99_000.0
    session._last_regime_date["TEST"] = datetime(2024, 6, 1, tzinfo=timezone.utc).date()

    session._save_state_snapshot()
    raw = json.loads(session.state_snapshot_path.read_text())
    assert raw["trades_today"] == 3
    assert raw["last_regime_date"]["TEST"] == "2024-06-01"

    fresh = make_session(monkeypatch, tmp_path)
    fresh.state_snapshot_path = session.state_snapshot_path
    fresh._restore_state_snapshot()

    assert fresh.position_tracker.trades_today == 3
    assert fresh.position_tracker._daily_start_equity == 99_000.0
    assert fresh._last_regime_date["TEST"] == datetime(2024, 6, 1).date()


def test_restore_state_snapshot_missing_file_is_a_noop(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    session.state_snapshot_path = tmp_path / "does_not_exist.json"

    session._restore_state_snapshot()  # must not raise


# ----------------------------------------------------------------------
# Shutdown / dashboard
# ----------------------------------------------------------------------


def test_shutdown_stops_streams_and_saves_state(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)

    session.shutdown()

    session.market_data.stop_stream.assert_called_once()
    assert session.state_snapshot_path.exists()


def test_shutdown_never_closes_positions(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)

    session.shutdown()

    assert not session.order_executor.close_position.called
    assert not session.order_executor.close_all_positions.called


def test_render_dashboard_does_not_raise(monkeypatch, tmp_path) -> None:
    from rich.console import Console

    session = make_session(monkeypatch, tmp_path)
    session.position_tracker.register_entry("TEST", 10, 100.0, regime_at_entry="BULL")
    session.render_dashboard(console=Console(file=open("/dev/null", "w")))


# ----------------------------------------------------------------------
# run() wiring
# ----------------------------------------------------------------------


def test_run_subscribes_and_runs_stream_then_shuts_down(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)

    session.run()

    session.market_data.subscribe_bars.assert_called_once()
    call_args = session.market_data.subscribe_bars.call_args[0]
    assert call_args[0] == ["TEST"]
    session.market_data.run_stream.assert_called_once()
    assert session.state_snapshot_path.exists()  # shutdown ran


def test_run_does_nothing_further_when_market_closed(monkeypatch, tmp_path) -> None:
    session = make_session(monkeypatch, tmp_path)
    closed_clock = MagicMock()
    closed_clock.model_dump.return_value = {"is_open": False, "next_open": None}
    session.client.trading_client.get_clock.return_value = closed_clock

    session.run()

    assert not session.market_data.subscribe_bars.called
    assert not session.market_data.run_stream.called


# ----------------------------------------------------------------------
# build_risk_manager / build_trading_session
# ----------------------------------------------------------------------


def test_build_risk_manager_reads_all_config_fields(tmp_path) -> None:
    config = main_module.load_config("config/settings.yaml")
    risk_manager = main_module.build_risk_manager(config, lock_file_path=str(tmp_path / "lock"))

    assert risk_manager.config.max_risk_per_trade == config["risk"]["max_risk_per_trade"]
    assert risk_manager.config.max_correlation_reject == config["risk"]["max_correlation_reject"]


def test_build_trading_session_requires_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    config = make_config(tmp_path)

    with pytest.raises(RuntimeError):
        main_module.build_trading_session(config, ["TEST"])
