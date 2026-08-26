"""Order execution: placement, modification, and cancellation of orders.

Every order submitted through here carries a unique ``trade_id`` (Alpaca's
``client_order_id``), so a fill arriving later on the trade-update
WebSocket (see ``broker.position_tracker``) can always be traced back to
the ``Signal``/``RiskDecision`` that produced it.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from broker.alpaca_client import AlpacaClient
from core.regime_strategies import DIRECTION_FLAT, DIRECTION_LONG, Signal

logger = logging.getLogger(__name__)

#: Limit orders are priced this fraction away from the current price —
#: above it for buys, below it for sells — to trade off fill probability
#: against paying/giving up more than a small premium.
DEFAULT_LIMIT_OFFSET_PCT = 0.001

#: Cancel an unfilled limit order after this many seconds.
DEFAULT_UNFILLED_TIMEOUT_SECONDS = 30.0

#: How often to poll order status while waiting for a fill.
DEFAULT_POLL_INTERVAL_SECONDS = 2.0

#: Default reward:risk ratio used to derive a take-profit when a Signal
#: doesn't specify one.
DEFAULT_REWARD_RISK_RATIO = 2.0


@dataclass
class OrderResult:
    """Result of an order submission."""

    trade_id: str
    order_id: Optional[str]
    symbol: str
    side: str
    quantity: float
    order_type: str
    status: str
    limit_price: Optional[float] = None
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None


def _quantity_for_signal(signal: Signal, equity: float) -> float:
    """Whole-share quantity implied by a signal's sizing, given account equity."""
    notional = signal.position_size_pct * signal.leverage * equity
    return float(int(notional / signal.entry_price))


class OrderExecutor:
    """Places, modifies, and cancels orders through the Alpaca client."""

    def __init__(
        self,
        client: AlpacaClient,
        limit_offset_pct: float = DEFAULT_LIMIT_OFFSET_PCT,
        unfilled_timeout_seconds: float = DEFAULT_UNFILLED_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        """Store the Alpaca client used to execute orders."""
        self.client = client
        self.limit_offset_pct = limit_offset_pct
        self.unfilled_timeout_seconds = unfilled_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        signal: Signal,
        trade_id: Optional[str] = None,
        retry_at_market: bool = False,
    ) -> OrderResult:
        """Submit a LIMIT order sized from ``signal`` (position_size_pct *
        leverage * account equity), priced ``limit_offset_pct`` away from
        the signal's entry price. If still unfilled after
        ``unfilled_timeout_seconds``, the order is cancelled and, if
        ``retry_at_market`` is set, resubmitted as a market order for
        whatever quantity remains unfilled.
        """
        if signal.direction == DIRECTION_FLAT:
            raise ValueError("submit_order requires a directional signal, not FLAT")

        trade_id = trade_id or str(uuid.uuid4())
        side = OrderSide.BUY if signal.direction == DIRECTION_LONG else OrderSide.SELL
        equity = float(self.client.get_account()["equity"])
        qty = _quantity_for_signal(signal, equity)
        if qty <= 0:
            raise ValueError(f"Computed non-positive quantity for {signal.symbol}: {qty}")

        limit_price = self._limit_price_for(signal.entry_price, side)
        request = LimitOrderRequest(
            symbol=signal.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
            client_order_id=trade_id,
        )
        order = self.client.call_with_retry(self.client.trading_client.submit_order, request)
        logger.info("Submitted LIMIT order trade_id=%s symbol=%s qty=%s limit=%.2f", trade_id, signal.symbol, qty, limit_price)

        result = self._wait_for_terminal_status(order.id, trade_id, signal.symbol, side, qty, limit_price)

        if result.status not in ("filled",) and retry_at_market:
            remaining = qty - result.filled_qty
            if remaining > 0:
                result = self._submit_market_remainder(signal.symbol, side, remaining, trade_id)

        return result

    def submit_bracket_order(
        self,
        signal: Signal,
        trade_id: Optional[str] = None,
        take_profit_price: Optional[float] = None,
    ) -> OrderResult:
        """Submit entry + stop-loss + take-profit as an Alpaca bracket order
        (the stop and take-profit legs form an OCO pair that activates
        once the entry fills).

        ``signal.stop_loss`` is required. If ``signal.take_profit`` (or
        ``take_profit_price``) isn't set, one is derived at
        ``DEFAULT_REWARD_RISK_RATIO`` times the stop distance.
        """
        if signal.direction == DIRECTION_FLAT:
            raise ValueError("submit_bracket_order requires a directional signal, not FLAT")
        if signal.stop_loss is None:
            raise ValueError("submit_bracket_order requires signal.stop_loss")

        trade_id = trade_id or str(uuid.uuid4())
        side = OrderSide.BUY if signal.direction == DIRECTION_LONG else OrderSide.SELL
        equity = float(self.client.get_account()["equity"])
        qty = _quantity_for_signal(signal, equity)
        if qty <= 0:
            raise ValueError(f"Computed non-positive quantity for {signal.symbol}: {qty}")

        stop_distance = abs(signal.entry_price - signal.stop_loss)
        take_profit = take_profit_price or signal.take_profit
        if take_profit is None:
            take_profit = (
                signal.entry_price + DEFAULT_REWARD_RISK_RATIO * stop_distance
                if side == OrderSide.BUY
                else signal.entry_price - DEFAULT_REWARD_RISK_RATIO * stop_distance
            )

        limit_price = self._limit_price_for(signal.entry_price, side)
        request = LimitOrderRequest(
            symbol=signal.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
            order_class=OrderClass.BRACKET,
            client_order_id=trade_id,
            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(signal.stop_loss, 2)),
        )
        order = self.client.call_with_retry(self.client.trading_client.submit_order, request)
        logger.info(
            "Submitted BRACKET order trade_id=%s symbol=%s qty=%s entry=%.2f stop=%.2f take_profit=%.2f",
            trade_id, signal.symbol, qty, limit_price, signal.stop_loss, take_profit,
        )
        return self._order_result_from(order, trade_id)

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def modify_stop(self, symbol: str, new_stop: float) -> bool:
        """Tighten an open position's protective stop. Refuses (returns
        False, no-ops) to widen it — a stop only ever moves to reduce
        risk, never to increase it."""
        position = self._get_position(symbol)
        if position is None:
            logger.warning("modify_stop: no open position for %s", symbol)
            return False

        stop_order = self._find_stop_order(symbol)
        if stop_order is None:
            logger.warning("modify_stop: no open stop order for %s", symbol)
            return False

        current_stop = float(stop_order.stop_price)
        is_long = position["side"] == "long"
        tightened = (new_stop > current_stop) if is_long else (new_stop < current_stop)
        if not tightened:
            logger.warning(
                "modify_stop refused: %s new_stop=%.2f does not tighten current_stop=%.2f (side=%s)",
                symbol, new_stop, current_stop, position["side"],
            )
            return False

        self.client.call_with_retry(
            self.client.trading_client.replace_order_by_id,
            stop_order.id,
            ReplaceOrderRequest(stop_price=round(new_stop, 2)),
        )
        logger.info("Tightened stop for %s: %.2f -> %.2f", symbol, current_stop, new_stop)
        return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing open order."""
        try:
            self.client.call_with_retry(self.client.trading_client.cancel_order_by_id, order_id)
        except Exception as exc:  # noqa: BLE001 - cancellation of an already-closed order shouldn't crash callers
            logger.warning("cancel_order failed for %s: %s", order_id, exc)
            return False
        return True

    def close_position(self, symbol: str) -> Optional[OrderResult]:
        """Close a single open position at market (the entire position —
        pass no close_options, since an empty ClosePositionRequest is
        itself invalid: Alpaca requires qty or percentage only for a
        partial close)."""
        order = self.client.call_with_retry(self.client.trading_client.close_position, symbol)
        return self._order_result_from(order, order.client_order_id or str(uuid.uuid4()))

    def close_all_positions(self, cancel_open_orders: bool = True) -> list[dict]:
        """Close every open position at market, cancelling open orders first."""
        responses = self.client.call_with_retry(
            self.client.trading_client.close_all_positions, cancel_orders=cancel_open_orders
        )
        return [r.model_dump() for r in responses]

    def get_order_status(self, order_id: str) -> str:
        """Fetch the current status of an order."""
        order = self.client.call_with_retry(self.client.trading_client.get_order_by_id, order_id)
        return order.status.value if hasattr(order.status, "value") else str(order.status)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _limit_price_for(self, current_price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return current_price * (1 + self.limit_offset_pct)
        return current_price * (1 - self.limit_offset_pct)

    def _order_result_from(self, order, trade_id: str) -> OrderResult:
        return OrderResult(
            trade_id=trade_id,
            order_id=str(order.id) if getattr(order, "id", None) else None,
            symbol=order.symbol,
            side=order.side.value if hasattr(order.side, "value") else str(order.side),
            quantity=float(order.qty) if order.qty is not None else 0.0,
            order_type=order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type),
            status=order.status.value if hasattr(order.status, "value") else str(order.status),
            limit_price=float(order.limit_price) if order.limit_price is not None else None,
            filled_qty=float(order.filled_qty) if order.filled_qty is not None else 0.0,
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price is not None else None,
        )

    def _wait_for_terminal_status(
        self, order_id, trade_id: str, symbol: str, side: OrderSide, qty: float, limit_price: float
    ) -> OrderResult:
        deadline = time.monotonic() + self.unfilled_timeout_seconds
        order = self.client.call_with_retry(self.client.trading_client.get_order_by_id, order_id)

        while order.status not in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
            if time.monotonic() >= deadline:
                logger.info("Order %s unfilled after %.0fs; cancelling", order_id, self.unfilled_timeout_seconds)
                self.cancel_order(order_id)
                order = self.client.call_with_retry(self.client.trading_client.get_order_by_id, order_id)
                break
            time.sleep(self.poll_interval_seconds)
            order = self.client.call_with_retry(self.client.trading_client.get_order_by_id, order_id)

        return self._order_result_from(order, trade_id)

    def _submit_market_remainder(self, symbol: str, side: OrderSide, remaining_qty: float, trade_id: str) -> OrderResult:
        market_trade_id = f"{trade_id}-mkt"
        request = MarketOrderRequest(
            symbol=symbol, qty=remaining_qty, side=side, time_in_force=TimeInForce.DAY, client_order_id=market_trade_id
        )
        order = self.client.call_with_retry(self.client.trading_client.submit_order, request)
        logger.info("Retrying %s remainder (%s shares) at MARKET, trade_id=%s", symbol, remaining_qty, market_trade_id)
        result = self._wait_for_terminal_status(order.id, trade_id, symbol, side, remaining_qty, 0.0)
        return result

    def _get_position(self, symbol: str) -> Optional[dict]:
        for position in self.client.get_positions():
            if position["symbol"] == symbol:
                return position
        return None

    def _find_stop_order(self, symbol: str):
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], nested=True)
        orders = self.client.call_with_retry(self.client.trading_client.get_orders, filter=request)
        for order in orders:
            if order.stop_price is not None:
                return order
            for leg in order.legs or []:
                if leg.stop_price is not None:
                    return leg
        return None
