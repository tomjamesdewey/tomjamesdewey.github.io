"""Position tracking: open positions and running P&L.

Subscribes to Alpaca's trade-update WebSocket for instant fill
notifications and updates tracked positions — and, if a
``core.risk_manager.CircuitBreaker`` is supplied, re-evaluates it against
actual P&L on every fill — independent of whatever the HMM/strategy layer
currently believes is happening.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from alpaca.trading.stream import TradingStream

from broker.alpaca_client import AlpacaClient
from core.risk_manager import CircuitBreaker, CircuitBreakerState, PortfolioState, RecentOrder
from core.risk_manager import Position as RiskPosition

logger = logging.getLogger(__name__)

FILL_EVENTS = ("fill", "partial_fill")
_TERMINAL_QTY_EPSILON = 1e-9


@dataclass
class TrackedPosition:
    """Rich per-position bookkeeping: entry context, current mark, and the
    regime active at entry vs. now (so drift can be audited)."""

    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_level: Optional[float] = None
    regime_at_entry: Optional[str] = None
    regime_current: Optional[str] = None
    sector: Optional[str] = None

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        return (self.current_price / self.entry_price - 1.0) if self.entry_price else 0.0

    @property
    def market_value(self) -> float:
        return self.current_price * self.quantity

    def holding_period(self, as_of: Optional[datetime] = None) -> timedelta:
        return (as_of or datetime.now(timezone.utc)) - self.entry_time

    def to_risk_position(self) -> RiskPosition:
        return RiskPosition(
            symbol=self.symbol,
            quantity=self.quantity,
            entry_price=self.entry_price,
            current_price=self.current_price,
            market_value=self.market_value,
            sector=self.sector,
            opened_at=self.entry_time,
        )


class PositionTracker:
    """Tracks open positions and computes P&L against the broker's state."""

    def __init__(self, client: AlpacaClient, circuit_breaker: Optional[CircuitBreaker] = None) -> None:
        """Store the Alpaca client used to fetch/reconcile position state,
        and (optionally) a CircuitBreaker to keep updated on every fill."""
        self.client = client
        self.circuit_breaker = circuit_breaker
        self.positions: dict[str, TrackedPosition] = {}
        self.recent_orders: list[RecentOrder] = []
        self.trades_today: int = 0
        self._peak_equity: Optional[float] = None
        self._daily_start_equity: Optional[float] = None
        self._weekly_start_equity: Optional[float] = None
        self._stream: Optional[TradingStream] = None

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def sync_with_alpaca(self) -> None:
        """Reconcile locally-tracked positions with Alpaca's actual open
        positions on startup. Positions Alpaca no longer shows are
        dropped; positions Alpaca has that weren't tracked locally are
        adopted (with unknown entry/regime context — logged loudly, since
        that context can never be recovered)."""
        actual_by_symbol = {p["symbol"]: p for p in self.client.get_positions()}

        for symbol in list(self.positions):
            if symbol not in actual_by_symbol:
                logger.warning("Reconcile: %s tracked locally but not open at Alpaca; dropping.", symbol)
                del self.positions[symbol]

        for symbol, position in actual_by_symbol.items():
            qty = float(position["qty"])
            price = float(position["current_price"])
            if symbol in self.positions:
                self.positions[symbol].quantity = qty
                self.positions[symbol].current_price = price
            else:
                logger.warning(
                    "Reconcile: %s open at Alpaca (qty=%s) but not tracked locally; "
                    "adopting with unknown entry/regime context.",
                    symbol,
                    qty,
                )
                self.positions[symbol] = TrackedPosition(
                    symbol=symbol,
                    quantity=qty,
                    entry_price=float(position["avg_entry_price"]),
                    entry_time=datetime.now(timezone.utc),
                    current_price=price,
                )

        equity = float(self.client.get_account()["equity"])
        self._peak_equity = max(self._peak_equity or equity, equity)
        if self._daily_start_equity is None:
            self._daily_start_equity = equity
        if self._weekly_start_equity is None:
            self._weekly_start_equity = equity

    # ------------------------------------------------------------------
    # Recording our own order flow
    # ------------------------------------------------------------------

    def register_entry(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        stop_level: Optional[float] = None,
        regime_at_entry: Optional[str] = None,
        sector: Optional[str] = None,
        entry_time: Optional[datetime] = None,
    ) -> None:
        """Record a position opened by our own order flow. Prefer letting
        ``handle_trade_update`` react to the actual fill event; call this
        directly only when a synchronous confirmation is needed (e.g. a
        market order awaited via OrderExecutor rather than the stream)."""
        self.positions[symbol] = TrackedPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time or datetime.now(timezone.utc),
            current_price=entry_price,
            stop_level=stop_level,
            regime_at_entry=regime_at_entry,
            regime_current=regime_at_entry,
            sector=sector,
        )

    def register_order_submitted(self, symbol: str, direction: str, timestamp: Optional[datetime] = None) -> None:
        """Record a just-submitted order for duplicate-order detection
        (feeds ``RiskManager``'s ``PortfolioState.recent_orders``)."""
        self.recent_orders.append(
            RecentOrder(symbol=symbol, direction=direction, timestamp=timestamp or datetime.now(timezone.utc))
        )

    def update_current_price(self, symbol: str, price: float) -> None:
        if symbol in self.positions:
            self.positions[symbol].current_price = price

    def update_current_regime(self, symbol: str, regime_label: str) -> None:
        if symbol in self.positions:
            self.positions[symbol].regime_current = regime_label

    def update_stop_level(self, symbol: str, stop_level: float) -> None:
        if symbol in self.positions:
            self.positions[symbol].stop_level = stop_level

    # ------------------------------------------------------------------
    # WebSocket fill notifications
    # ------------------------------------------------------------------

    def handle_trade_update(self, event: dict) -> None:
        """Process one trade-update event. Updates tracked positions and,
        if a CircuitBreaker was supplied, re-evaluates it against the new
        actual P&L — independent of the HMM/strategy layer."""
        order = event.get("order") or {}
        symbol = order.get("symbol")
        if event.get("event") not in FILL_EVENTS or not symbol:
            return

        fill_qty = float(event.get("qty") or order.get("filled_qty") or 0.0)
        fill_price = float(event.get("price") or order.get("filled_avg_price") or 0.0)
        if fill_qty <= 0 or fill_price <= 0:
            return

        if order.get("side") == "buy":
            applied = self._apply_buy_fill(symbol, fill_qty, fill_price)
        else:
            applied = self._apply_sell_fill(symbol, fill_qty, fill_price)
        if not applied:
            return
        self.trades_today += 1

        if self.circuit_breaker is not None:
            self._update_circuit_breaker()

    def _update_circuit_breaker(self) -> None:
        assert self.circuit_breaker is not None
        try:
            equity = float(self.client.get_account()["equity"])
        except Exception:  # noqa: BLE001 - never let a stale equity lookup crash the fill handler
            logger.warning("Could not refresh equity after fill; circuit breaker not re-evaluated this tick.")
            return
        self.circuit_breaker.update(self.get_portfolio_state(equity))

    def _apply_buy_fill(self, symbol: str, fill_qty: float, fill_price: float) -> bool:
        existing = self.positions.get(symbol)
        if existing is None:
            self.positions[symbol] = TrackedPosition(
                symbol=symbol,
                quantity=fill_qty,
                entry_price=fill_price,
                entry_time=datetime.now(timezone.utc),
                current_price=fill_price,
            )
            return True
        total_qty = existing.quantity + fill_qty
        existing.entry_price = (existing.entry_price * existing.quantity + fill_price * fill_qty) / total_qty
        existing.quantity = total_qty
        existing.current_price = fill_price
        return True

    def _apply_sell_fill(self, symbol: str, fill_qty: float, fill_price: float) -> bool:
        existing = self.positions.get(symbol)
        if existing is None:
            logger.warning("Sell fill for untracked position %s; ignoring.", symbol)
            return False
        existing.quantity -= fill_qty
        existing.current_price = fill_price
        if existing.quantity <= _TERMINAL_QTY_EPSILON:
            del self.positions[symbol]
        return True

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def start_streaming(self) -> None:
        """Subscribe to Alpaca's trade-update WebSocket for instant fill
        notifications. Blocking — run in its own thread/process."""
        self._stream = TradingStream(self.client.api_key, self.client.secret_key, paper=self.client.paper)

        async def _handler(update) -> None:
            payload = update.model_dump() if hasattr(update, "model_dump") else update
            self.handle_trade_update(payload)

        self._stream.subscribe_trade_updates(_handler)
        self._stream.run()

    def stop_streaming(self) -> None:
        if self._stream is not None:
            self._stream.stop()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[TrackedPosition]:
        """Get the tracked position for a symbol, if any."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> list[TrackedPosition]:
        """Get all currently tracked positions."""
        return list(self.positions.values())

    def get_total_exposure(self, equity: float) -> float:
        """Compute total exposure as a fraction of equity."""
        if not equity:
            return 0.0
        return sum(p.market_value for p in self.positions.values()) / equity

    def get_daily_pnl(self) -> float:
        """Compute equity P&L since the start of the current trading day."""
        if not self._daily_start_equity:
            return 0.0
        equity = float(self.client.get_account()["equity"])
        return equity / self._daily_start_equity - 1.0

    def get_weekly_pnl(self) -> float:
        """Compute equity P&L since the start of the current trading week."""
        if not self._weekly_start_equity:
            return 0.0
        equity = float(self.client.get_account()["equity"])
        return equity / self._weekly_start_equity - 1.0

    def reset_daily(self) -> None:
        """Mark the start of a new trading day (call once at market open)."""
        self._daily_start_equity = float(self.client.get_account()["equity"])
        self.trades_today = 0
        self.recent_orders.clear()
        if self.circuit_breaker is not None:
            self.circuit_breaker.reset_daily()

    def reset_weekly(self) -> None:
        """Mark the start of a new trading week."""
        self._weekly_start_equity = float(self.client.get_account()["equity"])
        if self.circuit_breaker is not None:
            self.circuit_breaker.reset_weekly()

    def get_portfolio_state(self, equity: Optional[float] = None) -> PortfolioState:
        """Build a ``core.risk_manager.PortfolioState`` snapshot from
        tracked positions and account info, for ``RiskManager.validate_signal``."""
        account = self.client.get_account()
        equity = equity if equity is not None else float(account["equity"])
        self._peak_equity = max(self._peak_equity or equity, equity)
        daily_start = self._daily_start_equity or equity
        weekly_start = self._weekly_start_equity or equity

        return PortfolioState(
            equity=equity,
            cash=float(account["cash"]),
            buying_power=float(account["buying_power"]),
            positions={symbol: p.to_risk_position() for symbol, p in self.positions.items()},
            daily_pnl_pct=(equity / daily_start - 1.0) if daily_start else 0.0,
            weekly_pnl_pct=(equity / weekly_start - 1.0) if weekly_start else 0.0,
            peak_equity=self._peak_equity,
            drawdown_from_peak_pct=(equity / self._peak_equity - 1.0) if self._peak_equity else 0.0,
            circuit_breaker_status=self.circuit_breaker.check() if self.circuit_breaker else CircuitBreakerState(),
            flicker_rate=0,
            trades_today=self.trades_today,
            recent_orders=list(self.recent_orders),
        )
