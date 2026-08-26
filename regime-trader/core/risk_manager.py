"""Risk management: position sizing, leverage, and drawdown limits.

The risk manager operates INDEPENDENTLY of the HMM. Even if regime
detection fails completely, circuit breakers catch drawdowns based on
actual P&L — defense in depth. ``RiskManager.validate_signal`` has
ABSOLUTE VETO POWER over any signal: it may reject it outright, or
silently resize/de-leverage it before approving.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.regime_strategies import DIRECTION_FLAT, DIRECTION_LONG, Signal

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Action the risk manager recommends given current portfolio state."""

    NORMAL = "normal"
    REDUCE_EXPOSURE = "reduce_exposure"
    HALT_NEW_TRADES = "halt_new_trades"


@dataclass
class RiskConfig:
    """Mirrors the ``risk`` section of settings.yaml."""

    max_risk_per_trade: float
    max_exposure: float
    max_leverage: float
    max_single_position: float
    max_concurrent: int
    max_daily_trades: int
    daily_dd_reduce: float
    daily_dd_halt: float
    weekly_dd_reduce: float
    weekly_dd_halt: float
    max_dd_from_peak: float
    gap_stop_multiple: float = 3.0
    overnight_gap_risk_pct: float = 0.02
    min_position_usd: float = 100.0
    max_correlation_reduce: float = 0.70
    max_correlation_reject: float = 0.85
    correlation_window_days: int = 60
    max_sector_exposure: float = 0.30
    max_spread_pct: float = 0.005
    duplicate_window_seconds: int = 60
    flicker_rate_threshold: int = 4


@dataclass
class Position:
    """A single currently-open position, as the risk manager sees it."""

    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    sector: Optional[str] = None
    opened_at: Optional[datetime] = None


@dataclass
class RecentOrder:
    """A recently-submitted order, kept for duplicate-order detection."""

    symbol: str
    direction: str
    timestamp: datetime


@dataclass
class CircuitBreakerState:
    """Which circuit breakers are currently active."""

    daily_reduce_active: bool = False
    daily_halt_active: bool = False
    weekly_reduce_active: bool = False
    weekly_halt_active: bool = False
    peak_halt_active: bool = False

    @property
    def any_active(self) -> bool:
        return any(
            [
                self.daily_reduce_active,
                self.daily_halt_active,
                self.weekly_reduce_active,
                self.weekly_halt_active,
                self.peak_halt_active,
            ]
        )

    @property
    def trading_halted(self) -> bool:
        """Halt-level breakers stop ALL new trading (existing positions
        should be closed by the caller — see ``CircuitBreakerEvent``)."""
        return self.daily_halt_active or self.weekly_halt_active or self.peak_halt_active

    @property
    def size_reduction_factor(self) -> float:
        """0.5 while a reduce-level breaker is active (and no halt is in
        effect — a halt rejects new trades outright, so sizing is moot)."""
        if self.trading_halted:
            return 0.0
        if self.daily_reduce_active or self.weekly_reduce_active:
            return 0.5
        return 1.0


@dataclass
class CircuitBreakerEvent:
    """An audit-log entry for one circuit-breaker trigger."""

    timestamp: datetime
    breaker_type: str
    actual_drawdown_pct: float
    equity: float
    positions_closed: list[str]
    regime_at_time: Optional[str] = None


@dataclass
class PortfolioState:
    """A snapshot of portfolio/account state, as required to validate a signal."""

    equity: float
    cash: float
    buying_power: float
    positions: dict[str, Position]
    daily_pnl_pct: float
    weekly_pnl_pct: float
    peak_equity: float
    drawdown_from_peak_pct: float  # non-positive fraction, e.g. -0.12 for -12%
    circuit_breaker_status: CircuitBreakerState
    flicker_rate: int
    trades_today: int = 0
    recent_orders: list[RecentOrder] = field(default_factory=list)


@dataclass
class RiskDecision:
    """The outcome of ``RiskManager.validate_signal``."""

    approved: bool
    modified_signal: Optional[Signal]
    rejection_reason: Optional[str]
    modifications: list[str] = field(default_factory=list)


class CircuitBreaker:
    """Tracks drawdown-based circuit breakers, independent of the HMM."""

    def __init__(self, config: RiskConfig, lock_file_path: str | Path = "trading_halted.lock") -> None:
        self.config = config
        self.lock_file_path = Path(lock_file_path)
        self.state = CircuitBreakerState(peak_halt_active=self.lock_file_path.exists())
        self.history: list[CircuitBreakerEvent] = []

    def update(self, portfolio_state: PortfolioState, regime_label: Optional[str] = None) -> CircuitBreakerState:
        """Re-evaluate every threshold against actual P&L and update state,
        logging (and, for peak-drawdown, persisting to disk) any breaker
        that just transitioned from inactive to active."""
        now = datetime.now(timezone.utc)
        daily_dd = max(0.0, -portfolio_state.daily_pnl_pct)
        weekly_dd = max(0.0, -portfolio_state.weekly_pnl_pct)
        peak_dd = abs(min(0.0, portfolio_state.drawdown_from_peak_pct))

        new_state = CircuitBreakerState(
            daily_reduce_active=daily_dd > self.config.daily_dd_reduce,
            daily_halt_active=daily_dd > self.config.daily_dd_halt,
            weekly_reduce_active=weekly_dd > self.config.weekly_dd_reduce,
            weekly_halt_active=weekly_dd > self.config.weekly_dd_halt,
            peak_halt_active=self.lock_file_path.exists() or peak_dd > self.config.max_dd_from_peak,
        )

        transitions = [
            ("daily_reduce", self.state.daily_reduce_active, new_state.daily_reduce_active, daily_dd),
            ("daily_halt", self.state.daily_halt_active, new_state.daily_halt_active, daily_dd),
            ("weekly_reduce", self.state.weekly_reduce_active, new_state.weekly_reduce_active, weekly_dd),
            ("weekly_halt", self.state.weekly_halt_active, new_state.weekly_halt_active, weekly_dd),
            ("peak_halt", self.state.peak_halt_active, new_state.peak_halt_active, peak_dd),
        ]
        for breaker_type, was_active, is_active, dd in transitions:
            if is_active and not was_active:
                positions_closed = (
                    list(portfolio_state.positions)
                    if breaker_type in ("daily_halt", "weekly_halt", "peak_halt")
                    else []
                )
                event = CircuitBreakerEvent(
                    timestamp=now,
                    breaker_type=breaker_type,
                    actual_drawdown_pct=dd,
                    equity=portfolio_state.equity,
                    positions_closed=positions_closed,
                    regime_at_time=regime_label,
                )
                logger.warning(
                    "Circuit breaker fired: %s dd=%.2f%% equity=%.2f regime=%s positions_closed=%s",
                    breaker_type,
                    dd * 100,
                    portfolio_state.equity,
                    regime_label,
                    positions_closed,
                )
                self.history.append(event)
                if breaker_type == "peak_halt":
                    self._write_lock_file(event)

        self.state = new_state
        return self.state

    def check(self) -> CircuitBreakerState:
        """Query current status without recomputing from P&L (still checks
        whether the peak-halt lock file is present, since that persists
        across process restarts until manually deleted)."""
        self.state.peak_halt_active = self.state.peak_halt_active or self.lock_file_path.exists()
        return self.state

    def reset_daily(self) -> None:
        """Clear day-scoped breakers at the start of a new trading day.
        Peak-halt is NOT day-scoped and is untouched here."""
        self.state.daily_reduce_active = False
        self.state.daily_halt_active = False

    def reset_weekly(self) -> None:
        """Clear week-scoped breakers at the start of a new week.
        Peak-halt is NOT week-scoped and is untouched here."""
        self.state.weekly_reduce_active = False
        self.state.weekly_halt_active = False

    def get_history(self) -> list[CircuitBreakerEvent]:
        return list(self.history)

    def _write_lock_file(self, event: CircuitBreakerEvent) -> None:
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file_path.write_text(
            f"TRADING HALTED at {event.timestamp.isoformat()}\n"
            f"Reason: peak drawdown {event.actual_drawdown_pct:.2%} exceeded max_dd_from_peak "
            f"({self.config.max_dd_from_peak:.2%})\n"
            f"Equity at halt: {event.equity:.2f}\n"
            f"Regime at time: {event.regime_at_time}\n"
            "Delete this file manually to resume trading.\n"
        )


class RiskManager:
    """Applies position sizing rules and portfolio-level risk limits.

    Has absolute veto power over signals from the strategy layer: it may
    reject a signal outright (``RiskDecision.approved=False``), or approve
    a resized/de-leveraged copy of it.
    """

    def __init__(self, config: RiskConfig, lock_file_path: str | Path = "trading_halted.lock") -> None:
        self.config = config
        self.circuit_breaker = CircuitBreaker(config, lock_file_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        price_history: Optional[dict[str, pd.Series]] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        is_tradeable: bool = True,
    ) -> RiskDecision:
        """Validate (and, if needed, resize/de-leverage) a strategy signal.

        ``price_history`` (a symbol -> daily-return Series map) is used
        for the correlation check when supplied; the check is skipped
        gracefully when it isn't. ``bid``/``ask`` enable the spread check;
        ``is_tradeable`` reflects the broker's asset-tradeable flag.
        """
        if signal.direction == DIRECTION_FLAT:
            return RiskDecision(
                approved=True,
                modified_signal=dataclasses.replace(signal, position_size_pct=0.0),
                rejection_reason=None,
            )

        veto = self._check_hard_vetoes(signal, portfolio_state, is_tradeable, bid, ask)
        if veto is not None:
            return self._reject(signal, veto)

        modifications: list[str] = []

        position_pct, size_modifications = self._compute_risk_based_position_pct(signal, portfolio_state)
        modifications.extend(size_modifications)
        if position_pct is None:
            return self._reject(signal, size_modifications[-1] if size_modifications else "position sizing failed")

        position_pct, cap_modifications = self._apply_exposure_caps(signal, portfolio_state, position_pct)
        modifications.extend(cap_modifications)
        if position_pct is None:
            return self._reject(signal, cap_modifications[-1] if cap_modifications else "exposure caps failed")

        if position_pct * portfolio_state.equity < self.config.min_position_usd:
            return self._reject(
                signal,
                f"position size ${position_pct * portfolio_state.equity:.2f} is below the "
                f"${self.config.min_position_usd:.0f} minimum",
            )

        position_pct, corr_modifications = self._apply_correlation_check(
            signal, portfolio_state, position_pct, price_history
        )
        modifications.extend(corr_modifications)
        if position_pct is None:
            return self._reject(signal, corr_modifications[-1] if corr_modifications else "correlation check failed")

        leverage, leverage_modifications = self._apply_leverage_rules(signal, portfolio_state)
        modifications.extend(leverage_modifications)

        reduction = portfolio_state.circuit_breaker_status.size_reduction_factor
        if reduction < 1.0:
            position_pct *= reduction
            modifications.append(f"size reduced {int((1 - reduction) * 100)}% by an active circuit breaker")

        position_pct = self._fit_to_buying_power(portfolio_state, position_pct, leverage, modifications)
        if position_pct * portfolio_state.equity < self.config.min_position_usd:
            return self._reject(signal, "insufficient buying power for even the minimum position size")

        modified_signal = dataclasses.replace(signal, position_size_pct=position_pct, leverage=leverage)
        return RiskDecision(approved=True, modified_signal=modified_signal, rejection_reason=None, modifications=modifications)

    def evaluate_drawdown(
        self,
        daily_pnl_pct: float,
        weekly_pnl_pct: float,
        drawdown_from_peak: float,
    ) -> RiskAction:
        """Coarse-grained recommended action from raw drawdown numbers,
        without needing a full ``PortfolioState`` (e.g. for quick checks
        or reporting)."""
        peak_dd = abs(min(0.0, drawdown_from_peak))
        daily_dd = max(0.0, -daily_pnl_pct)
        weekly_dd = max(0.0, -weekly_pnl_pct)

        if (
            peak_dd > self.config.max_dd_from_peak
            or daily_dd > self.config.daily_dd_halt
            or weekly_dd > self.config.weekly_dd_halt
        ):
            return RiskAction.HALT_NEW_TRADES
        if daily_dd > self.config.daily_dd_reduce or weekly_dd > self.config.weekly_dd_reduce:
            return RiskAction.REDUCE_EXPOSURE
        return RiskAction.NORMAL

    # ------------------------------------------------------------------
    # Internal steps (each returns (value_or_None, modifications_or_reason))
    # ------------------------------------------------------------------

    def _reject(self, signal: Signal, reason: str) -> RiskDecision:
        logger.warning("Signal rejected: symbol=%s reason=%s", signal.symbol, reason)
        return RiskDecision(approved=False, modified_signal=None, rejection_reason=reason)

    def _check_hard_vetoes(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        is_tradeable: bool,
        bid: Optional[float],
        ask: Optional[float],
    ) -> Optional[str]:
        if portfolio_state.circuit_breaker_status.trading_halted:
            return "circuit breaker active: trading halted"

        if signal.stop_loss is None:
            return "every position must have a stop loss"
        if signal.direction == DIRECTION_LONG and signal.stop_loss >= signal.entry_price:
            return "invalid stop loss: must be below entry price for a long position"

        if not is_tradeable:
            return f"{signal.symbol} is not currently tradeable"

        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid if mid else float("inf")
            if spread_pct > self.config.max_spread_pct:
                return f"bid-ask spread {spread_pct:.2%} exceeds max {self.config.max_spread_pct:.2%}"

        if self._is_duplicate_order(signal, portfolio_state):
            return (
                f"duplicate order: {signal.symbol} {signal.direction} within "
                f"{self.config.duplicate_window_seconds}s of a recent order"
            )

        is_new_position = signal.symbol not in portfolio_state.positions
        if is_new_position and len(portfolio_state.positions) >= self.config.max_concurrent:
            return f"max concurrent positions ({self.config.max_concurrent}) reached"

        if portfolio_state.trades_today >= self.config.max_daily_trades:
            return f"max daily trades ({self.config.max_daily_trades}) reached"

        return None

    def _is_duplicate_order(self, signal: Signal, portfolio_state: PortfolioState) -> bool:
        window = self.config.duplicate_window_seconds
        for order in portfolio_state.recent_orders:
            if order.symbol != signal.symbol or order.direction != signal.direction:
                continue
            if abs((signal.timestamp - order.timestamp).total_seconds()) <= window:
                return True
        return False

    def _compute_risk_based_position_pct(
        self, signal: Signal, portfolio_state: PortfolioState
    ) -> tuple[Optional[float], list[str]]:
        """Position size = (equity * max_risk_per_trade) / abs(entry - stop),
        further capped so a 3x gap-through the stop overnight would still
        only cost ``overnight_gap_risk_pct`` of the portfolio."""
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        if stop_distance <= 0:
            return None, ["stop loss must differ from entry price"]

        risk_shares = (portfolio_state.equity * self.config.max_risk_per_trade) / stop_distance
        overnight_shares = (portfolio_state.equity * self.config.overnight_gap_risk_pct) / (
            self.config.gap_stop_multiple * stop_distance
        )

        modifications: list[str] = []
        shares = risk_shares
        if overnight_shares < risk_shares:
            shares = overnight_shares
            modifications.append(
                f"size reduced for overnight gap risk ({self.config.gap_stop_multiple:.0f}x stop)"
            )

        notional = shares * signal.entry_price
        return notional / portfolio_state.equity, modifications

    def _apply_exposure_caps(
        self, signal: Signal, portfolio_state: PortfolioState, position_pct: float
    ) -> tuple[Optional[float], list[str]]:
        modifications: list[str] = []

        regime_cap = signal.metadata.get("regime_max_position_size_pct")
        if regime_cap is not None and position_pct > regime_cap:
            position_pct = regime_cap
            modifications.append(f"size capped at regime max ({regime_cap:.1%})")

        if position_pct > self.config.max_single_position:
            position_pct = self.config.max_single_position
            modifications.append(f"size capped at max_single_position ({self.config.max_single_position:.1%})")

        current_exposure_pct = (
            sum(p.market_value for p in portfolio_state.positions.values()) / portfolio_state.equity
            if portfolio_state.equity
            else 0.0
        )
        available_exposure = self.config.max_exposure - current_exposure_pct
        if available_exposure <= 0:
            return None, [f"max total exposure ({self.config.max_exposure:.0%}) already reached"]
        if position_pct > available_exposure:
            position_pct = available_exposure
            modifications.append(f"size capped to stay within max_exposure ({self.config.max_exposure:.0%})")

        sector = signal.metadata.get("sector")
        if sector:
            sector_exposure_pct = (
                sum(p.market_value for p in portfolio_state.positions.values() if p.sector == sector)
                / portfolio_state.equity
                if portfolio_state.equity
                else 0.0
            )
            available_sector = self.config.max_sector_exposure - sector_exposure_pct
            if available_sector <= 0:
                return None, [f"max sector exposure ({self.config.max_sector_exposure:.0%}) already reached for {sector}"]
            if position_pct > available_sector:
                position_pct = available_sector
                modifications.append(
                    f"size capped to stay within max_sector_exposure ({self.config.max_sector_exposure:.0%})"
                )

        return position_pct, modifications

    def _apply_correlation_check(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        position_pct: float,
        price_history: Optional[dict[str, pd.Series]],
    ) -> tuple[Optional[float], list[str]]:
        if not price_history:
            return position_pct, []

        max_corr = self._max_correlation(signal.symbol, portfolio_state, price_history)
        if max_corr is None:
            return position_pct, []

        if max_corr > self.config.max_correlation_reject:
            return None, [f"correlation {max_corr:.2f} with an existing position exceeds {self.config.max_correlation_reject:.2f}"]
        if max_corr > self.config.max_correlation_reduce:
            return position_pct * 0.5, [f"size halved: correlation {max_corr:.2f} with an existing position"]
        return position_pct, []

    def _max_correlation(
        self, symbol: str, portfolio_state: PortfolioState, price_history: dict[str, pd.Series]
    ) -> Optional[float]:
        if symbol not in price_history:
            return None
        target_returns = price_history[symbol].tail(self.config.correlation_window_days)

        correlations = []
        for existing_symbol in portfolio_state.positions:
            if existing_symbol == symbol or existing_symbol not in price_history:
                continue
            other_returns = price_history[existing_symbol].tail(self.config.correlation_window_days)
            aligned = pd.concat([target_returns, other_returns], axis=1).dropna()
            if len(aligned) < 2:
                continue
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if corr is not None and not np.isnan(corr):
                correlations.append(abs(float(corr)))

        return max(correlations) if correlations else None

    def _apply_leverage_rules(self, signal: Signal, portfolio_state: PortfolioState) -> tuple[float, list[str]]:
        modifications: list[str] = []
        leverage = signal.leverage

        force_1x = (
            bool(signal.metadata.get("uncertainty_mode"))
            or portfolio_state.circuit_breaker_status.any_active
            or len(portfolio_state.positions) >= 3
            or portfolio_state.flicker_rate > self.config.flicker_rate_threshold
        )
        if force_1x and leverage > 1.0:
            leverage = 1.0
            modifications.append("leverage forced to 1.0x")

        if leverage > self.config.max_leverage:
            leverage = self.config.max_leverage
            modifications.append(f"leverage capped at max_leverage ({self.config.max_leverage:.2f}x)")

        return leverage, modifications

    def _fit_to_buying_power(
        self, portfolio_state: PortfolioState, position_pct: float, leverage: float, modifications: list[str]
    ) -> float:
        required = position_pct * portfolio_state.equity * leverage
        if required <= portfolio_state.buying_power or portfolio_state.equity <= 0 or leverage <= 0:
            return position_pct
        fitted_pct = portfolio_state.buying_power / (portfolio_state.equity * leverage)
        modifications.append("size reduced to fit available buying power")
        return max(0.0, fitted_pct)
