"""Order execution: placement, modification, and cancellation of orders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from broker.alpaca_client import AlpacaClient


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"


@dataclass
class OrderResult:
    """Result of an order submission."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    status: str


class OrderExecutor:
    """Places, modifies, and cancels orders through the Alpaca client."""

    def __init__(self, client: AlpacaClient) -> None:
        """Store the Alpaca client used to execute orders."""
        ...

    def submit_order(
        self,
        symbol: str,
        quantity: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> OrderResult:
        """Submit a new order."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing open order."""
        ...

    def modify_order(self, order_id: str, **kwargs) -> OrderResult:
        """Modify an existing open order (price, quantity, etc.)."""
        ...

    def get_order_status(self, order_id: str) -> str:
        """Fetch the current status of an order."""
        ...
