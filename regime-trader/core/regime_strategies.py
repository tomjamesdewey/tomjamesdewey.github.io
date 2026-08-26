"""Volatility-regime-based allocation strategies.

Design insight: the HMM excels at detecting VOLATILITY ENVIRONMENTS, not
market direction. Stocks trend upward roughly 70% of the time in
low-volatility periods; the worst drawdowns cluster in high-volatility
spikes. So the strategy is simple:

- Low vol  -> be fully invested (calm markets trend up)
- Mid vol  -> stay invested if trend intact, reduce if not
- High vol -> defensive allocation

Regimes are bucketed into LOW/MID/HIGH purely by ``expected_volatility``
(ascending), independently of the HMM's return-sorted *label*. A "BULL"
label does NOT mean low volatility — ``StrategyOrchestrator`` never
inspects labels when choosing a strategy; it only uses the vol-rank
mapping built from ``RegimeInfo.expected_volatility``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core.hmm_engine import REGIME_LABELS_BY_COUNT, RegimeInfo, RegimeState
from data.feature_engineering import (
    ADX_WINDOW,
    ATR_WINDOW,
    PRICE_TREND_SMA_WINDOW,
    average_directional_index,
    average_true_range,
)

DIRECTION_LONG = "LONG"
DIRECTION_FLAT = "FLAT"

#: Classic Wilder ADX trend-strength threshold used to confirm "trend intact".
TREND_ADX_THRESHOLD = 20.0

#: ATR multiple used to place the default stop below entry.
STOP_LOSS_ATR_MULTIPLE = 2.0

#: Minimum bars needed to safely compute the SMA/ADX/ATR indicators below.
MIN_BARS_FOR_SIGNAL = max(PRICE_TREND_SMA_WINDOW, ADX_WINDOW, ATR_WINDOW) + 1


@dataclass
class StrategyConfig:
    """Mirrors the ``strategy`` section of settings.yaml."""

    low_vol_allocation: float
    mid_vol_allocation_trend: float
    mid_vol_allocation_no_trend: float
    high_vol_allocation: float
    low_vol_leverage: float
    rebalance_threshold: float
    uncertainty_size_mult: float
    min_confidence: float = 0.55


@dataclass
class Signal:
    """A concrete, sizeable trading signal for one symbol."""

    symbol: str
    direction: str  # DIRECTION_LONG or DIRECTION_FLAT
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    position_size_pct: float  # 0.60 - 0.95 (pre-uncertainty-adjustment range)
    leverage: float  # 1.0 or 1.25
    regime_id: int
    regime_name: str
    regime_probability: float
    timestamp: pd.Timestamp
    reasoning: str
    strategy_name: str
    metadata: dict = field(default_factory=dict)


def _is_trend_intact(bars: pd.DataFrame) -> bool:
    """Trend is "intact" when price is above its 50-SMA and ADX(14) confirms
    trend strength. Caller must ensure ``len(bars) >= MIN_BARS_FOR_SIGNAL``.
    """
    sma50 = bars["close"].rolling(window=PRICE_TREND_SMA_WINDOW).mean().iloc[-1]
    adx = average_directional_index(
        bars["high"], bars["low"], bars["close"], ADX_WINDOW
    ).iloc[-1]
    price = bars["close"].iloc[-1]
    return bool(price > sma50 and adx >= TREND_ADX_THRESHOLD)


def _atr_stop_loss(bars: pd.DataFrame, direction: str, entry_price: float) -> float:
    """Default stop: entry -/+ STOP_LOSS_ATR_MULTIPLE * ATR(14)."""
    atr = average_true_range(bars["high"], bars["low"], bars["close"], ATR_WINDOW).iloc[-1]
    if direction == DIRECTION_LONG:
        return entry_price - STOP_LOSS_ATR_MULTIPLE * atr
    return entry_price + STOP_LOSS_ATR_MULTIPLE * atr


class BaseStrategy(ABC):
    """Base class for regime-conditioned allocation strategies.

    A concrete strategy owns exactly one volatility bucket (LOW/MID/HIGH,
    assigned by ``StrategyOrchestrator``) and decides target allocation,
    leverage, and a stop from the *regime's characteristics* — never from
    its label.
    """

    def __init__(self, regime_info: RegimeInfo, config: StrategyConfig) -> None:
        self.regime_info = regime_info
        self.config = config

    @property
    def strategy_name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Optional[Signal]:
        """Generate a signal for ``symbol`` given recent ``bars`` and the
        current filtered ``regime_state``. Returns None when there isn't
        enough data to safely size a position."""
        raise NotImplementedError

    def _build_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
        direction: str,
        position_size_pct: float,
        leverage: float,
        reasoning: str,
    ) -> Signal:
        entry_price = float(bars["close"].iloc[-1])
        stop_loss = _atr_stop_loss(bars, direction, entry_price)
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=regime_state.probability,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=None,
            position_size_pct=position_size_pct,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=regime_state.timestamp,
            reasoning=reasoning,
            strategy_name=self.strategy_name,
            metadata={"regime_max_position_size_pct": self.regime_info.max_position_size_pct},
        )


class LowVolBullStrategy(BaseStrategy):
    """Low-volatility regime: be fully invested — calm markets trend up."""

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Optional[Signal]:
        if len(bars) < MIN_BARS_FOR_SIGNAL:
            return None
        reasoning = (
            f"{regime_state.label} regime (vol=LOW, prob={regime_state.probability:.2f}): "
            "low-volatility environment historically trends upward — fully invested."
        )
        return self._build_signal(
            symbol,
            bars,
            regime_state,
            direction=DIRECTION_LONG,
            position_size_pct=self.config.low_vol_allocation,
            leverage=self.config.low_vol_leverage,
            reasoning=reasoning,
        )


class MidVolCautiousStrategy(BaseStrategy):
    """Mid-volatility regime: stay invested if trend intact, reduce if not."""

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Optional[Signal]:
        if len(bars) < MIN_BARS_FOR_SIGNAL:
            return None

        trend_intact = _is_trend_intact(bars)
        if trend_intact:
            position_size_pct = self.config.mid_vol_allocation_trend
            reasoning = (
                f"{regime_state.label} regime (vol=MID, prob={regime_state.probability:.2f}): "
                "trend intact (price > 50-SMA, ADX confirms) — staying invested."
            )
        else:
            position_size_pct = self.config.mid_vol_allocation_no_trend
            reasoning = (
                f"{regime_state.label} regime (vol=MID, prob={regime_state.probability:.2f}): "
                "trend not confirmed — reducing exposure."
            )

        return self._build_signal(
            symbol,
            bars,
            regime_state,
            direction=DIRECTION_LONG,
            position_size_pct=position_size_pct,
            leverage=1.0,
            reasoning=reasoning,
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """High-volatility regime: defensive allocation — worst drawdowns cluster here."""

    def generate_signal(
        self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState
    ) -> Optional[Signal]:
        if len(bars) < MIN_BARS_FOR_SIGNAL:
            return None
        reasoning = (
            f"{regime_state.label} regime (vol=HIGH, prob={regime_state.probability:.2f}): "
            "high-volatility environment — defensive allocation."
        )
        return self._build_signal(
            symbol,
            bars,
            regime_state,
            direction=DIRECTION_LONG,
            position_size_pct=self.config.high_vol_allocation,
            leverage=1.0,
            reasoning=reasoning,
        )


# Backward-compatible aliases: several legacy/label-oriented names for the
# same three volatility-bucketed strategies above.
CrashDefenseStrategy = HighVolDefensiveStrategy
StrongBearStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
WeakBearStrategy = HighVolDefensiveStrategy
NeutralStrategy = MidVolCautiousStrategy
MeanReversionStrategy = MidVolCautiousStrategy
WeakBullStrategy = LowVolBullStrategy
BullTrendStrategy = LowVolBullStrategy
StrongBullStrategy = LowVolBullStrategy
EuphoriaStrategy = LowVolBullStrategy

#: Naive label -> strategy mapping, kept only for backward compatibility
#: with older label-driven callers. StrategyOrchestrator never uses this —
#: it always buckets by expected_volatility instead (see the module
#: docstring: a "BULL" label does NOT mean low volatility).
LABEL_TO_STRATEGY: dict[str, type[BaseStrategy]] = {
    "CRASH": CrashDefenseStrategy,
    "STRONG_BEAR": StrongBearStrategy,
    "BEAR": BearTrendStrategy,
    "WEAK_BEAR": WeakBearStrategy,
    "NEUTRAL": NeutralStrategy,
    "WEAK_BULL": WeakBullStrategy,
    "BULL": BullTrendStrategy,
    "STRONG_BULL": StrongBullStrategy,
    "EUPHORIA": EuphoriaStrategy,
}
assert set(LABEL_TO_STRATEGY) == {
    label for labels in REGIME_LABELS_BY_COUNT.values() for label in labels
}

_VOL_BUCKET_LOW = "LOW"
_VOL_BUCKET_MID = "MID"
_VOL_BUCKET_HIGH = "HIGH"

_STRATEGY_BY_BUCKET: dict[str, type[BaseStrategy]] = {
    _VOL_BUCKET_LOW: LowVolBullStrategy,
    _VOL_BUCKET_MID: MidVolCautiousStrategy,
    _VOL_BUCKET_HIGH: HighVolDefensiveStrategy,
}


def _vol_bucket(vol_rank: int, n_regimes: int) -> str:
    """Partition a 0-indexed volatility rank into LOW/MID/HIGH thirds."""
    if n_regimes <= 1:
        return _VOL_BUCKET_MID
    rank_pct = vol_rank / (n_regimes - 1)
    if rank_pct < 1 / 3:
        return _VOL_BUCKET_LOW
    if rank_pct < 2 / 3:
        return _VOL_BUCKET_MID
    return _VOL_BUCKET_HIGH


class StrategyOrchestrator:
    """Routes each fitted HMM regime to a volatility-bucketed strategy.

    Regimes are ranked by ``RegimeInfo.expected_volatility`` (ascending)
    to compute a ``vol_rank`` per ``regime_id``, independently of the
    HMM's return-based label sort. That rank is partitioned into
    LOW/MID/HIGH thirds, each delegated to its own ``BaseStrategy``.
    """

    def __init__(self, config: StrategyConfig, regime_infos: dict[int, RegimeInfo]) -> None:
        self.config = config
        self.regime_infos: dict[int, RegimeInfo] = {}
        self.vol_rank: dict[int, int] = {}
        self.vol_bucket: dict[int, str] = {}
        self.strategies: dict[int, BaseStrategy] = {}
        self.update_regime_infos(regime_infos)

    def update_regime_infos(self, regime_infos: dict[int, RegimeInfo]) -> None:
        """Rebuild the regime_id -> vol_rank -> strategy mapping.

        Call this after every HMM retrain, since state indices and their
        volatility ordering can change between fits.
        """
        self.regime_infos = dict(regime_infos)
        ordered = sorted(regime_infos.items(), key=lambda kv: kv[1].expected_volatility)
        n_regimes = len(ordered)

        self.vol_rank = {regime_id: rank for rank, (regime_id, _) in enumerate(ordered)}
        self.vol_bucket = {
            regime_id: _vol_bucket(rank, n_regimes) for regime_id, rank in self.vol_rank.items()
        }
        self.strategies = {
            regime_id: _STRATEGY_BY_BUCKET[self.vol_bucket[regime_id]](info, self.config)
            for regime_id, info in regime_infos.items()
        }

    def generate_signals(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool = False,
    ) -> list[Signal]:
        """Generate one signal per symbol for the current filtered regime state."""
        strategy = self.strategies.get(regime_state.state_id)
        if strategy is None:
            return []

        signals: list[Signal] = []
        for symbol in symbols:
            bars = bars_by_symbol.get(symbol)
            if bars is None or len(bars) == 0:
                continue
            signal = strategy.generate_signal(symbol, bars, regime_state)
            if signal is None:
                continue
            signals.append(self._apply_uncertainty(signal, regime_state, is_flickering))
        return signals

    def _apply_uncertainty(
        self, signal: Signal, regime_state: RegimeState, is_flickering: bool
    ) -> Signal:
        """Halve position size and cap leverage at 1.0x when confidence is
        below threshold or the regime is flickering."""
        low_confidence = regime_state.probability < self.config.min_confidence
        if not (low_confidence or is_flickering):
            return signal

        signal.position_size_pct = signal.position_size_pct * self.config.uncertainty_size_mult
        signal.leverage = 1.0
        signal.reasoning = f"{signal.reasoning} [UNCERTAINTY — size halved]"
        signal.metadata["uncertainty_mode"] = True
        signal.metadata["uncertainty_low_confidence"] = low_confidence
        signal.metadata["uncertainty_flickering"] = is_flickering
        return signal

    def needs_rebalance(self, current_allocation_pct: float, target_allocation_pct: float) -> bool:
        """True only when target and current allocation differ by more than
        ``rebalance_threshold`` — avoids churn from minor probability drift."""
        return abs(target_allocation_pct - current_allocation_pct) > self.config.rebalance_threshold
