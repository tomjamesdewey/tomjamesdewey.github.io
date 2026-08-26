"""Signal generation: combines HMM regime detection and strategy allocation
into concrete per-symbol trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import RegimeStrategy
from core.risk_manager import RiskManager


class SignalAction(Enum):
    """Directional action for a generated signal."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"


@dataclass
class TradeSignal:
    """A concrete trading signal for a single symbol."""

    symbol: str
    action: SignalAction
    target_allocation: float
    confidence: float
    timestamp: pd.Timestamp


class SignalGenerator:
    """Combines HMM regime detection and strategy allocation into signals."""

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy: RegimeStrategy,
        risk_manager: RiskManager,
    ) -> None:
        """Store references to the HMM engine, strategy, and risk manager."""
        ...

    def generate_signals(
        self, symbols: list[str], features_by_symbol: dict[str, pd.DataFrame]
    ) -> list[TradeSignal]:
        """Generate trade signals for all symbols given current features."""
        ...

    def generate_signal_for_symbol(
        self, symbol: str, features: pd.DataFrame
    ) -> TradeSignal:
        """Generate a single symbol's trade signal from regime and strategy state."""
        ...
