"""Tests for core.risk_manager."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.regime_strategies import Signal
from core.risk_manager import (
    CircuitBreaker,
    CircuitBreakerState,
    Position,
    PortfolioState,
    RecentOrder,
    RiskAction,
    RiskConfig,
    RiskManager,
)

BASE_CONFIG_KWARGS = dict(
    max_risk_per_trade=0.01,
    max_exposure=0.80,
    max_leverage=1.25,
    max_single_position=0.15,
    max_concurrent=5,
    max_daily_trades=20,
    daily_dd_reduce=0.02,
    daily_dd_halt=0.03,
    weekly_dd_reduce=0.05,
    weekly_dd_halt=0.07,
    max_dd_from_peak=0.10,
    gap_stop_multiple=3.0,
    overnight_gap_risk_pct=0.02,
    min_position_usd=100.0,
    max_correlation_reduce=0.70,
    max_correlation_reject=0.85,
    correlation_window_days=60,
    max_sector_exposure=0.30,
    max_spread_pct=0.005,
    duplicate_window_seconds=60,
    flicker_rate_threshold=4,
)


def make_config(**overrides) -> RiskConfig:
    kwargs = {**BASE_CONFIG_KWARGS, **overrides}
    return RiskConfig(**kwargs)


def make_manager(tmp_path, **config_overrides) -> RiskManager:
    return RiskManager(make_config(**config_overrides), lock_file_path=tmp_path / "trading_halted.lock")


def make_signal(**overrides) -> Signal:
    base = dict(
        symbol="AAPL",
        direction="LONG",
        confidence=0.9,
        entry_price=100.0,
        stop_loss=95.0,  # $5 stop distance
        take_profit=None,
        position_size_pct=0.90,
        leverage=1.25,
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


def make_portfolio(**overrides) -> PortfolioState:
    base = dict(
        equity=100_000.0,
        cash=100_000.0,
        buying_power=100_000.0,
        positions={},
        daily_pnl_pct=0.0,
        weekly_pnl_pct=0.0,
        peak_equity=100_000.0,
        drawdown_from_peak_pct=0.0,
        circuit_breaker_status=CircuitBreakerState(),
        flicker_rate=0,
        trades_today=0,
        recent_orders=[],
    )
    base.update(overrides)
    return PortfolioState(**base)


# ----------------------------------------------------------------------
# Position sizing
# ----------------------------------------------------------------------


def test_position_size_respects_max_risk_per_trade(tmp_path) -> None:
    """1%-risk formula: shares * stop_distance == equity * max_risk_per_trade,
    when the overnight gap-risk cap isn't the binding constraint."""
    # Loosen the overnight gap cap and the single-position cap so neither binds here.
    rm = make_manager(tmp_path, overnight_gap_risk_pct=0.10, max_single_position=0.50)
    signal = make_signal(entry_price=100.0, stop_loss=95.0)  # $5 stop distance
    portfolio = make_portfolio(equity=100_000.0)

    decision = rm.validate_signal(signal, portfolio)

    assert decision.approved
    notional = decision.modified_signal.position_size_pct * portfolio.equity
    shares = notional / signal.entry_price
    assert shares * 5.0 == pytest.approx(100_000.0 * 0.01)  # risked $ == 1% of equity


def test_gap_risk_caps_overnight_size(tmp_path) -> None:
    """When the overnight (3x gap) cap is tighter than the normal 1% formula,
    it wins and is recorded in modifications."""
    rm = make_manager(tmp_path)  # default overnight_gap_risk_pct=0.02, gap_stop_multiple=3.0
    signal = make_signal(entry_price=100.0, stop_loss=95.0)
    portfolio = make_portfolio(equity=100_000.0)

    decision = rm.validate_signal(signal, portfolio)

    assert decision.approved
    # overnight_shares = (100000*0.02)/(3*5) = 133.33 < risk_shares = (100000*0.01)/5 = 200
    expected_pct = (100_000.0 * 0.02 / (3.0 * 5.0) * 100.0) / 100_000.0
    assert decision.modified_signal.position_size_pct == pytest.approx(expected_pct)
    assert any("gap risk" in m for m in decision.modifications)


def test_missing_stop_loss_is_rejected(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal(stop_loss=None)
    decision = rm.validate_signal(signal, make_portfolio())

    assert not decision.approved
    assert "stop loss" in decision.rejection_reason
    assert decision.modified_signal is None


def test_stop_loss_on_wrong_side_is_rejected(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal(direction="LONG", entry_price=100.0, stop_loss=105.0)
    decision = rm.validate_signal(signal, make_portfolio())

    assert not decision.approved
    assert "stop loss" in decision.rejection_reason.lower()


def test_position_below_minimum_is_rejected(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal()
    portfolio = make_portfolio(equity=500.0, buying_power=500.0)

    decision = rm.validate_signal(signal, portfolio)

    assert not decision.approved
    assert "minimum" in decision.rejection_reason


def test_flat_direction_is_approved_with_zero_size(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal(direction="FLAT")
    decision = rm.validate_signal(signal, make_portfolio())

    assert decision.approved
    assert decision.modified_signal.position_size_pct == 0.0


# ----------------------------------------------------------------------
# Portfolio-level limits
# ----------------------------------------------------------------------


def test_exposure_limit_caps_size_to_remaining_budget(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal()
    # 79% already deployed; only 1% of max_exposure (80%) remains.
    portfolio = make_portfolio(
        positions={"X": Position("X", 1, 10, 10, market_value=79_000.0)}
    )

    decision = rm.validate_signal(signal, portfolio)

    assert decision.approved
    assert decision.modified_signal.position_size_pct == pytest.approx(0.01)


def test_exposure_limit_rejects_when_already_at_max(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal()
    portfolio = make_portfolio(positions={"X": Position("X", 1, 10, 10, market_value=80_000.0)})

    decision = rm.validate_signal(signal, portfolio)

    assert not decision.approved
    assert "exposure" in decision.rejection_reason


def test_concurrency_limit_blocks_extra_positions(tmp_path) -> None:
    rm = make_manager(tmp_path, max_concurrent=5)
    signal = make_signal(symbol="NEWSYM")
    positions = {f"SYM{i}": Position(f"SYM{i}", 1, 10, 10, 1000.0) for i in range(5)}
    portfolio = make_portfolio(positions=positions)

    decision = rm.validate_signal(signal, portfolio)

    assert not decision.approved
    assert "concurrent" in decision.rejection_reason


def test_concurrency_limit_allows_adding_to_existing_position(tmp_path) -> None:
    """The concurrency cap only blocks *new* symbols, not adding to one already held."""
    rm = make_manager(tmp_path, max_concurrent=5)
    signal = make_signal(symbol="SYM0")
    positions = {f"SYM{i}": Position(f"SYM{i}", 1, 10, 10, 1000.0) for i in range(5)}
    portfolio = make_portfolio(positions=positions)

    decision = rm.validate_signal(signal, portfolio)

    assert decision.approved


def test_max_daily_trades_blocks_further_trades(tmp_path) -> None:
    rm = make_manager(tmp_path, max_daily_trades=20)
    signal = make_signal()
    portfolio = make_portfolio(trades_today=20)

    decision = rm.validate_signal(signal, portfolio)

    assert not decision.approved
    assert "daily trades" in decision.rejection_reason


def test_duplicate_order_within_window_is_blocked(tmp_path) -> None:
    rm = make_manager(tmp_path, duplicate_window_seconds=60)
    signal = make_signal(timestamp=datetime(2024, 1, 2, 9, 30, 30, tzinfo=timezone.utc))
    portfolio = make_portfolio(
        recent_orders=[
            RecentOrder(symbol="AAPL", direction="LONG", timestamp=datetime(2024, 1, 2, 9, 30, 0, tzinfo=timezone.utc))
        ]
    )

    decision = rm.validate_signal(signal, portfolio)

    assert not decision.approved
    assert "duplicate" in decision.rejection_reason


def test_duplicate_order_outside_window_is_allowed(tmp_path) -> None:
    rm = make_manager(tmp_path, duplicate_window_seconds=60)
    signal = make_signal(timestamp=datetime(2024, 1, 2, 9, 32, 0, tzinfo=timezone.utc))
    portfolio = make_portfolio(
        recent_orders=[
            RecentOrder(symbol="AAPL", direction="LONG", timestamp=datetime(2024, 1, 2, 9, 30, 0, tzinfo=timezone.utc))
        ]
    )

    decision = rm.validate_signal(signal, portfolio)

    assert decision.approved


def test_regime_max_position_size_is_applied(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal(metadata={"regime_max_position_size_pct": 0.05})

    decision = rm.validate_signal(signal, make_portfolio())

    assert decision.approved
    assert decision.modified_signal.position_size_pct == pytest.approx(0.05)
    assert any("regime max" in m for m in decision.modifications)


# ----------------------------------------------------------------------
# Leverage rules
# ----------------------------------------------------------------------


def test_leverage_forced_to_1x_with_3_or_more_open_positions(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal(leverage=1.25)
    positions = {f"SYM{i}": Position(f"SYM{i}", 1, 10, 10, 1000.0) for i in range(3)}
    portfolio = make_portfolio(positions=positions)

    decision = rm.validate_signal(signal, portfolio)

    assert decision.approved
    assert decision.modified_signal.leverage == 1.0
    assert "leverage forced to 1.0x" in decision.modifications


def test_leverage_forced_to_1x_under_regime_uncertainty(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal(leverage=1.25, metadata={"uncertainty_mode": True})

    decision = rm.validate_signal(signal, make_portfolio())

    assert decision.modified_signal.leverage == 1.0


def test_leverage_forced_to_1x_under_high_flicker_rate(tmp_path) -> None:
    rm = make_manager(tmp_path, flicker_rate_threshold=4)
    signal = make_signal(leverage=1.25)
    portfolio = make_portfolio(flicker_rate=5)

    decision = rm.validate_signal(signal, portfolio)

    assert decision.modified_signal.leverage == 1.0


def test_leverage_forced_to_1x_when_circuit_breaker_active(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal(leverage=1.25)
    portfolio = make_portfolio(circuit_breaker_status=CircuitBreakerState(daily_reduce_active=True))

    decision = rm.validate_signal(signal, portfolio)

    assert decision.approved  # reduce-level breaker doesn't halt, just de-risks
    assert decision.modified_signal.leverage == 1.0


def test_leverage_capped_at_max_leverage(tmp_path) -> None:
    rm = make_manager(tmp_path, max_leverage=1.25)
    signal = make_signal(leverage=2.0)

    decision = rm.validate_signal(signal, make_portfolio())

    assert decision.modified_signal.leverage == 1.25


def test_low_vol_leverage_allowed_when_no_risk_flags(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal(leverage=1.25)

    decision = rm.validate_signal(signal, make_portfolio())

    assert decision.modified_signal.leverage == 1.25


# ----------------------------------------------------------------------
# Circuit breakers
# ----------------------------------------------------------------------


def test_circuit_breaker_halt_rejects_all_signals(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal()
    portfolio = make_portfolio(circuit_breaker_status=CircuitBreakerState(daily_halt_active=True))

    decision = rm.validate_signal(signal, portfolio)

    assert not decision.approved
    assert "circuit breaker" in decision.rejection_reason


def test_circuit_breaker_reduce_halves_position_size(tmp_path) -> None:
    rm = make_manager(tmp_path, overnight_gap_risk_pct=0.10)  # avoid the gap cap masking this
    signal = make_signal()
    normal_decision = rm.validate_signal(signal, make_portfolio())

    reduced_portfolio = make_portfolio(circuit_breaker_status=CircuitBreakerState(weekly_reduce_active=True))
    reduced_decision = rm.validate_signal(signal, reduced_portfolio)

    assert reduced_decision.approved
    assert reduced_decision.modified_signal.position_size_pct == pytest.approx(
        normal_decision.modified_signal.position_size_pct * 0.5
    )


def test_circuit_breaker_daily_thresholds(tmp_path) -> None:
    cb = CircuitBreaker(make_config(), lock_file_path=tmp_path / "trading_halted.lock")
    portfolio = make_portfolio(daily_pnl_pct=-0.025)

    state = cb.update(portfolio)

    assert state.daily_reduce_active
    assert not state.daily_halt_active

    state = cb.update(make_portfolio(daily_pnl_pct=-0.035))
    assert state.daily_halt_active


def test_circuit_breaker_weekly_thresholds(tmp_path) -> None:
    cb = CircuitBreaker(make_config(), lock_file_path=tmp_path / "trading_halted.lock")

    state = cb.update(make_portfolio(weekly_pnl_pct=-0.06))
    assert state.weekly_reduce_active
    assert not state.weekly_halt_active

    state = cb.update(make_portfolio(weekly_pnl_pct=-0.08))
    assert state.weekly_halt_active


def test_circuit_breaker_peak_drawdown_writes_lock_file(tmp_path) -> None:
    lock_path = tmp_path / "trading_halted.lock"
    cb = CircuitBreaker(make_config(), lock_file_path=lock_path)

    state = cb.update(make_portfolio(drawdown_from_peak_pct=-0.12), regime_label="CRASH")

    assert state.peak_halt_active
    assert lock_path.exists()
    assert "CRASH" in lock_path.read_text()


def test_circuit_breaker_peak_halt_persists_across_instances(tmp_path) -> None:
    lock_path = tmp_path / "trading_halted.lock"
    cb1 = CircuitBreaker(make_config(), lock_file_path=lock_path)
    cb1.update(make_portfolio(drawdown_from_peak_pct=-0.12))
    assert lock_path.exists()

    cb2 = CircuitBreaker(make_config(), lock_file_path=lock_path)
    assert cb2.check().peak_halt_active

    lock_path.unlink()
    cb3 = CircuitBreaker(make_config(), lock_file_path=lock_path)
    assert not cb3.check().peak_halt_active


def test_reset_daily_only_clears_daily_breakers(tmp_path) -> None:
    cb = CircuitBreaker(make_config(), lock_file_path=tmp_path / "trading_halted.lock")
    cb.update(make_portfolio(daily_pnl_pct=-0.035, weekly_pnl_pct=-0.08))
    assert cb.state.daily_halt_active and cb.state.weekly_halt_active

    cb.reset_daily()

    assert not cb.state.daily_halt_active
    assert cb.state.weekly_halt_active  # untouched


def test_reset_weekly_only_clears_weekly_breakers(tmp_path) -> None:
    cb = CircuitBreaker(make_config(), lock_file_path=tmp_path / "trading_halted.lock")
    cb.update(make_portfolio(daily_pnl_pct=-0.035, weekly_pnl_pct=-0.08))

    cb.reset_weekly()

    assert cb.state.daily_halt_active  # untouched
    assert not cb.state.weekly_halt_active


def test_get_history_records_every_trigger(tmp_path) -> None:
    cb = CircuitBreaker(make_config(), lock_file_path=tmp_path / "trading_halted.lock")
    cb.update(make_portfolio(daily_pnl_pct=-0.025))
    cb.update(make_portfolio(daily_pnl_pct=-0.035))

    history = cb.get_history()

    assert [e.breaker_type for e in history] == ["daily_reduce", "daily_halt"]
    assert all(e.equity == 100_000.0 for e in history)


def test_evaluate_drawdown_returns_normal_reduce_halt(tmp_path) -> None:
    rm = make_manager(tmp_path)

    assert rm.evaluate_drawdown(0.0, 0.0, 0.0) == RiskAction.NORMAL
    assert rm.evaluate_drawdown(-0.025, 0.0, 0.0) == RiskAction.REDUCE_EXPOSURE
    assert rm.evaluate_drawdown(-0.035, 0.0, 0.0) == RiskAction.HALT_NEW_TRADES
    assert rm.evaluate_drawdown(0.0, 0.0, -0.11) == RiskAction.HALT_NEW_TRADES


# ----------------------------------------------------------------------
# Correlation check
# ----------------------------------------------------------------------


def test_correlation_check_skipped_without_price_history(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal()
    portfolio = make_portfolio(positions={"MSFT": Position("MSFT", 1, 10, 10, 1000.0)})

    decision = rm.validate_signal(signal, portfolio)  # no price_history passed

    assert decision.approved


def test_correlation_above_reject_threshold_rejects_trade(tmp_path) -> None:
    rm = make_manager(tmp_path)
    signal = make_signal()
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    base = np.random.RandomState(0).normal(0, 0.01, 120)
    price_history = {
        "AAPL": pd.Series(base, index=idx),
        "MSFT": pd.Series(base, index=idx),  # identical -> correlation 1.0
    }
    portfolio = make_portfolio(positions={"MSFT": Position("MSFT", 1, 10, 10, 1000.0)})

    decision = rm.validate_signal(signal, portfolio, price_history=price_history)

    assert not decision.approved
    assert "correlation" in decision.rejection_reason


def test_correlation_above_reduce_threshold_halves_size(tmp_path) -> None:
    rm = make_manager(tmp_path, overnight_gap_risk_pct=0.10)
    signal = make_signal()
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    rng = np.random.RandomState(1)
    base = rng.normal(0, 0.01, 120)
    correlated = 0.55 * base + 0.45 * rng.normal(0, 0.01, 120)  # corr ~0.74, between reduce/reject thresholds
    price_history = {"AAPL": pd.Series(base, index=idx), "MSFT": pd.Series(correlated, index=idx)}
    portfolio = make_portfolio(positions={"MSFT": Position("MSFT", 1, 10, 10, 1000.0)})

    baseline = rm.validate_signal(signal, make_portfolio())
    decision = rm.validate_signal(signal, portfolio, price_history=price_history)

    corr = pd.Series(base).corr(pd.Series(correlated))
    assert rm.config.max_correlation_reduce < corr < rm.config.max_correlation_reject
    assert decision.approved
    assert decision.modified_signal.position_size_pct == pytest.approx(
        baseline.modified_signal.position_size_pct * 0.5
    )


def test_correlation_uncorrelated_symbol_has_no_effect(tmp_path) -> None:
    rm = make_manager(tmp_path, overnight_gap_risk_pct=0.10)
    signal = make_signal()
    idx = pd.date_range("2023-01-01", periods=120, freq="B")
    rng = np.random.RandomState(2)
    price_history = {
        "AAPL": pd.Series(rng.normal(0, 0.01, 120), index=idx),
        "MSFT": pd.Series(rng.normal(0, 0.01, 120), index=idx),
    }
    portfolio = make_portfolio(positions={"MSFT": Position("MSFT", 1, 10, 10, 1000.0)})

    baseline = rm.validate_signal(signal, make_portfolio())
    decision = rm.validate_signal(signal, portfolio, price_history=price_history)

    assert decision.approved
    assert decision.modified_signal.position_size_pct == pytest.approx(baseline.modified_signal.position_size_pct)


# ----------------------------------------------------------------------
# Buying power
# ----------------------------------------------------------------------


def test_buying_power_shrinks_position_when_insufficient(tmp_path) -> None:
    rm = make_manager(tmp_path, overnight_gap_risk_pct=0.10)
    signal = make_signal(leverage=1.0)
    portfolio = make_portfolio(equity=100_000.0, buying_power=5_000.0)

    decision = rm.validate_signal(signal, portfolio)

    assert decision.approved
    notional = decision.modified_signal.position_size_pct * portfolio.equity * decision.modified_signal.leverage
    assert notional <= 5_000.0 + 1e-6
    assert "buying power" in " ".join(decision.modifications)
