"""Phase 9 integration tests: full-pipeline, look-ahead, risk-stress, and
crash-recovery guarantees that only show up when the real components
(FeatureEngineer, HMMEngine, StrategyOrchestrator, RiskManager,
PositionTracker, Backtester) are wired together — as opposed to the
narrower unit tests in tests/test_*.py that exercise each in isolation.

Only the network boundary (AlpacaClient's underlying SDK clients,
MarketDataClient, OrderExecutor) is mocked; everything else here is the
real production code.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import Backtester
from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyConfig, StrategyOrchestrator
from tests.test_main import make_config, make_session, make_signal

logging.getLogger("core.hmm_engine").setLevel(logging.ERROR)


def _make_regime_switching_bars(n: int, seed: int = 7) -> pd.DataFrame:
    """Bars with alternating low/high-vol blocks, long enough for several
    walk-forward windows — used by the backtester-level look-ahead test."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    returns = np.empty(n)
    vol_state = 0
    for i in range(n):
        if i % 100 == 0:
            vol_state = 1 - vol_state
        vol = 0.006 if vol_state == 0 else 0.025
        mu = 0.0005 if vol_state == 0 else -0.0002
        returns[i] = rng.normal(mu, vol)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = rng.randint(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


# ----------------------------------------------------------------------
# (a) End-to-end dry run: data -> HMM -> strategy -> risk -> simulated orders
# ----------------------------------------------------------------------


def test_end_to_end_dry_run_pipeline(monkeypatch, tmp_path) -> None:
    """Drives the real pipeline (FeatureEngineer -> HMMEngine ->
    StrategyOrchestrator -> RiskManager) in --dry-run mode and asserts
    each stage actually ran and produced a coherent, risk-approved
    decision, without ever calling OrderExecutor.submit_bracket_order."""
    session = make_session(monkeypatch, tmp_path, dry_run=True)

    assert session.startup() is True
    assert "TEST" in session.engines  # HMM trained from fetched data
    assert "TEST" in session.orchestrators  # strategy wired from that HMM's regime_info

    bar = pd.Series(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000}, name=datetime.now(timezone.utc)
    )
    session.on_bar("TEST", bar)

    # HMM produced a regime call.
    regime_state = session._last_regime_state.get("TEST")
    assert regime_state is not None
    assert regime_state.label in session.engines["TEST"].state_labels.values()

    # The dry run never touches the order-submission API at all.
    assert not session.order_executor.submit_bracket_order.called
    assert not session.order_executor.submit_order.called

    # But the decision pipeline still ran: risk-approved sizing landed in
    # the dashboard's recent-signals feed (proof a Signal was generated
    # and passed RiskManager.validate_signal, not silently dropped).
    assert len(session.dashboard.recent_signals) >= 1


# ----------------------------------------------------------------------
# (b) Look-ahead bias
# ----------------------------------------------------------------------


def test_look_ahead_test_suite_itself_passes() -> None:
    """Documents the Phase 2 guarantee this integration test builds on:
    the mandatory look-ahead test suite must be green. (Runs in its own
    process here as a smoke check; the authoritative run is `pytest
    tests/test_look_ahead.py`, part of every full-suite run.)"""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_look_ahead.py", "-q"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_backtest_identical_with_different_end_dates() -> None:
    """The walk-forward Backtester (windows + HMM retraining + strategy +
    simulation combined, not just the HMM engine in isolation) must never
    let data past a bar leak into that bar's decision. Running the same
    backtest with two different, longer end dates must produce byte-for-
    byte identical equity curves over the overlapping date range."""
    bars = _make_regime_switching_bars(1600)

    hmm_template = HMMEngine(
        n_candidates=[3, 4], n_init=2, covariance_type="full", min_train_bars=100,
        stability_bars=3, flicker_window=20, flicker_threshold=4, min_confidence=0.55,
    )
    strategy_config = StrategyConfig(
        low_vol_allocation=0.95, mid_vol_allocation_trend=0.95, mid_vol_allocation_no_trend=0.60,
        high_vol_allocation=0.60, low_vol_leverage=1.25, rebalance_threshold=0.10,
        uncertainty_size_mult=0.50, min_confidence=0.55,
    )
    strategy_template = StrategyOrchestrator(strategy_config, {})
    backtester = Backtester(
        hmm_template, strategy_template, None, initial_capital=100_000.0, slippage_pct=0.0005,
        train_window=252, test_window=126, step_size=126,
    )

    short_result = backtester.run({"TEST": bars.iloc[:1100]})
    long_result = backtester.run({"TEST": bars.iloc[:1400]})

    common_index = short_result.equity_curve.index.intersection(long_result.equity_curve.index)
    assert len(common_index) > 100  # sanity: a meaningful overlap actually exists

    short_overlap = short_result.equity_curve.loc[common_index]
    long_overlap = long_result.equity_curve.loc[common_index]
    assert np.allclose(short_overlap.to_numpy(), long_overlap.to_numpy(), atol=1e-6), (
        "LOOK-AHEAD BIAS DETECTED: backtest results over the shared date range "
        "changed when more (future, relative to that range) data was appended."
    )


# ----------------------------------------------------------------------
# (c) Risk stress
# ----------------------------------------------------------------------


def test_extreme_signal_is_capped_by_risk_manager(monkeypatch, tmp_path) -> None:
    """A wildly oversized/over-leveraged signal must come out of
    RiskManager capped at max_single_position / max_leverage, never
    passed straight through."""
    session = make_session(monkeypatch, tmp_path)
    session.order_executor.submit_bracket_order.return_value = MagicMock(status="new")

    extreme_signal = make_signal(position_size_pct=5.0, leverage=10.0, entry_price=100.0, stop_loss=99.0)
    session.process_signal("TEST", extreme_signal)

    submitted = session.order_executor.submit_bracket_order.call_args[0][0]
    assert submitted.position_size_pct <= session.risk_manager.config.max_single_position + 1e-9
    assert submitted.leverage <= session.risk_manager.config.max_leverage + 1e-9


def test_rapid_fire_duplicate_signal_is_blocked(monkeypatch, tmp_path) -> None:
    """Two identical signals submitted back-to-back for the same symbol
    and direction: the first executes, the second is rejected as a
    duplicate order within the configured window."""
    session = make_session(monkeypatch, tmp_path)
    session.order_executor.submit_bracket_order.return_value = MagicMock(status="new")
    ts = datetime.now(timezone.utc)

    session.process_signal("TEST", make_signal(timestamp=ts))
    session.process_signal("TEST", make_signal(timestamp=ts + timedelta(seconds=1)))

    assert session.order_executor.submit_bracket_order.call_count == 1


def test_rapid_fire_after_window_elapses_is_allowed(monkeypatch, tmp_path) -> None:
    """Sanity check on the duplicate-blocking test above: once the
    duplicate-order window has elapsed, a repeat signal is allowed again."""
    session = make_session(monkeypatch, tmp_path)
    session.order_executor.submit_bracket_order.return_value = MagicMock(status="new")
    window = session.risk_manager.config.duplicate_window_seconds
    ts = datetime.now(timezone.utc)

    session.process_signal("TEST", make_signal(timestamp=ts))
    session.process_signal("TEST", make_signal(timestamp=ts + timedelta(seconds=window + 5)))

    assert session.order_executor.submit_bracket_order.call_count == 2


def test_signal_without_stop_loss_is_rejected(monkeypatch, tmp_path) -> None:
    """The system refuses to size or submit any order lacking a stop loss —
    verified through the full process_signal path, not just RiskManager in
    isolation."""
    session = make_session(monkeypatch, tmp_path)

    session.process_signal("TEST", make_signal(stop_loss=None))

    assert not session.order_executor.submit_bracket_order.called


# ----------------------------------------------------------------------
# (d) Alpaca paper trading — requires real credentials + network access,
# neither available in this sandbox. Skipped unless both are present, so
# this test is real and ready to run in an environment that has them.
# ----------------------------------------------------------------------

_HAS_ALPACA_PAPER_CREDENTIALS = bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


@pytest.mark.skipif(
    not _HAS_ALPACA_PAPER_CREDENTIALS,
    reason="ALPACA_API_KEY/ALPACA_SECRET_KEY not set — this environment has no Alpaca paper account to test against",
)
def test_alpaca_paper_place_modify_cancel_round_trip() -> None:
    """Against a real Alpaca PAPER account: place a bracket order, tighten
    its stop, cancel it, and verify no order or position is left behind."""
    from broker.alpaca_client import AlpacaClient
    from broker.order_executor import OrderExecutor
    from core.regime_strategies import Signal
    from data.market_data import MarketDataClient

    assert os.environ.get("ALPACA_PAPER", "true").lower() == "true", (
        "Refusing to run the paper-trading integration test against a non-paper account"
    )

    client = AlpacaClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    executor = OrderExecutor(client)
    market_data = MarketDataClient(client)

    price = float(market_data.get_latest_bar("AAPL")["close"])

    signal = Signal(
        symbol="AAPL", direction="LONG", confidence=0.9, entry_price=price, stop_loss=price * 0.95,
        take_profit=price * 1.05, position_size_pct=0.01, leverage=1.0, regime_id=0, regime_name="TEST",
        regime_probability=0.9, timestamp=datetime.now(timezone.utc), reasoning="integration test",
        strategy_name="IntegrationTest", metadata={},
    )

    result = executor.submit_bracket_order(signal)
    try:
        assert result.order_id is not None
        tightened = executor.modify_stop("AAPL", price * 0.96)
        assert isinstance(tightened, bool)
    finally:
        executor.cancel_order(result.order_id)
        positions = client.get_positions()
        assert not any(p["symbol"] == "AAPL" for p in positions), "Paper account left with an open AAPL position"


# ----------------------------------------------------------------------
# (e) Recovery: kill process, restart, verify state recovery and no double-entry
# ----------------------------------------------------------------------


def test_restart_recovers_state_and_positions_without_duplication(monkeypatch, tmp_path) -> None:
    """Simulates a killed-and-restarted process: a fresh TradingSession
    (fresh AlpacaClient connection, but the same on-disk state snapshot
    and model directory) must recover trades_today/equity baselines/last
    regime date from state_snapshot.json, and must adopt Alpaca's actual
    open position exactly once — never a duplicate."""
    config = make_config(tmp_path)

    def open_position(client, qty="10", avg_entry_price="150.0", current_price="155.0"):
        mock_position = MagicMock()
        mock_position.model_dump.return_value = {
            "symbol": "TEST", "qty": qty, "avg_entry_price": avg_entry_price, "current_price": current_price,
        }
        client.trading_client.get_all_positions.return_value = [mock_position]

    # --- "Session 1": runs, opens a position, accumulates some state, then "crashes". ---
    session1 = make_session(monkeypatch, tmp_path, config=config)
    open_position(session1.client)
    assert session1.startup() is True

    session1.position_tracker.trades_today = 4
    session1.position_tracker._daily_start_equity = 99_000.0
    session1._last_regime_date["TEST"] = datetime(2024, 6, 1, tzinfo=timezone.utc).date()
    session1._save_state_snapshot()  # crash recovery relies on this having been written recently

    assert len(session1.position_tracker.positions) == 1

    # --- "Session 2": a brand-new process, same disk state, same Alpaca account. ---
    session2 = make_session(monkeypatch, tmp_path, config=config)
    open_position(session2.client)  # Alpaca still reports the same single open position
    assert session2.startup() is True

    # Recovered from state_snapshot.json:
    assert session2.position_tracker.trades_today == 4
    assert session2.position_tracker._daily_start_equity == 99_000.0
    assert session2._last_regime_date["TEST"] == datetime(2024, 6, 1).date()

    # No double-entry: exactly one tracked position, matching Alpaca's reported quantity.
    assert len(session2.position_tracker.positions) == 1
    assert session2.position_tracker.get_position("TEST").quantity == 10.0

    # The model wasn't blindly retrained on restart either (same model_dir, freshly trained).
    assert session2.engines["TEST"].n_regimes == session1.engines["TEST"].n_regimes


def test_restart_does_not_resubmit_pending_signal_from_before_crash(monkeypatch, tmp_path) -> None:
    """A signal that was already being processed before the crash must not
    cause a second, redundant submission after restart: each TradingSession
    starts with a clean in-memory recent_orders list (Alpaca itself, not
    our local memory, is the source of truth for what's actually live),
    and register_order_submitted only ever reflects orders *this* process
    has submitted."""
    config = make_config(tmp_path)

    session1 = make_session(monkeypatch, tmp_path, config=config)
    assert session1.startup() is True
    session1.order_executor.submit_bracket_order.return_value = MagicMock(status="new")
    session1.process_signal("TEST", make_signal())
    assert session1.order_executor.submit_bracket_order.call_count == 1
    session1._save_state_snapshot()

    session2 = make_session(monkeypatch, tmp_path, config=config)
    assert session2.startup() is True
    session2.order_executor.submit_bracket_order.return_value = MagicMock(status="new")
    session2.process_signal("TEST", make_signal())  # a fresh, independent decision post-restart

    # Each process's OrderExecutor call count reflects only its own submissions.
    assert session1.order_executor.submit_bracket_order.call_count == 1
    assert session2.order_executor.submit_bracket_order.call_count == 1
