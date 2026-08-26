"""Tests for core.regime_strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    BearTrendStrategy,
    BullTrendStrategy,
    CrashDefenseStrategy,
    EuphoriaStrategy,
    HighVolDefensiveStrategy,
    LABEL_TO_STRATEGY,
    LowVolBullStrategy,
    MeanReversionStrategy,
    MidVolCautiousStrategy,
    StrategyConfig,
    StrategyOrchestrator,
)

CONFIG = StrategyConfig(
    low_vol_allocation=0.95,
    mid_vol_allocation_trend=0.95,
    mid_vol_allocation_no_trend=0.60,
    high_vol_allocation=0.60,
    low_vol_leverage=1.25,
    rebalance_threshold=0.10,
    uncertainty_size_mult=0.50,
    min_confidence=0.55,
)


def _regime_info(regime_id: int, label: str, expected_volatility: float) -> RegimeInfo:
    return RegimeInfo(
        regime_id=regime_id,
        regime_name=label,
        expected_return=0.0,
        expected_volatility=expected_volatility,
        recommended_strategy_type="",
        max_leverage_allowed=1.25,
        max_position_size_pct=0.15,
        min_confidence_to_act=0.55,
    )


def _regime_state(state_id: int, label: str, probability: float) -> RegimeState:
    return RegimeState(
        label=label,
        state_id=state_id,
        probability=probability,
        state_probabilities={label: probability},
        timestamp=pd.Timestamp("2024-01-01"),
        is_confirmed=True,
        consecutive_bars=10,
    )


def _make_bars(n: int, trend: str, seed: int = 0) -> pd.DataFrame:
    """Build synthetic OHLCV bars that are either clearly trending up,
    clearly trending down, or flat/choppy (no confirmed trend)."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")

    if trend == "up":
        drift = 0.006
        noise = 0.003
    elif trend == "down":
        drift = -0.006
        noise = 0.003
    else:  # flat / choppy
        drift = 0.0
        noise = 0.012

    returns = rng.normal(drift, noise, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = rng.randint(1_000_000, 5_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_low_vol_regime_allocates_low_vol_allocation() -> None:
    """Low-volatility regime should allocate low_vol_allocation with leverage applied."""
    strategy = LowVolBullStrategy(_regime_info(0, "BULL", 0.005), CONFIG)
    bars = _make_bars(80, "up")
    state = _regime_state(0, "BULL", 0.90)

    signal = strategy.generate_signal("SPY", bars, state)

    assert signal is not None
    assert signal.position_size_pct == CONFIG.low_vol_allocation
    assert signal.leverage == CONFIG.low_vol_leverage
    assert signal.direction == "LONG"


def test_mid_vol_regime_allocation_depends_on_trend() -> None:
    """Mid-volatility regime should use the trend vs. no-trend allocation value."""
    strategy = MidVolCautiousStrategy(_regime_info(1, "NEUTRAL", 0.015), CONFIG)
    state = _regime_state(1, "NEUTRAL", 0.80)

    trending_bars = _make_bars(80, "up", seed=1)
    trending_signal = strategy.generate_signal("SPY", trending_bars, state)
    assert trending_signal is not None
    assert trending_signal.position_size_pct == CONFIG.mid_vol_allocation_trend

    choppy_bars = _make_bars(80, "flat", seed=2)
    choppy_signal = strategy.generate_signal("SPY", choppy_bars, state)
    assert choppy_signal is not None
    assert choppy_signal.position_size_pct == CONFIG.mid_vol_allocation_no_trend


def test_high_vol_regime_allocates_high_vol_allocation() -> None:
    """High-volatility regime should allocate high_vol_allocation at 1.0x leverage."""
    strategy = HighVolDefensiveStrategy(_regime_info(2, "CRASH", 0.05), CONFIG)
    bars = _make_bars(80, "down")
    state = _regime_state(2, "CRASH", 0.85)

    signal = strategy.generate_signal("SPY", bars, state)

    assert signal is not None
    assert signal.position_size_pct == CONFIG.high_vol_allocation
    assert signal.leverage == 1.0


def test_orchestrator_buckets_by_volatility_not_label() -> None:
    """A 'BULL'-labeled regime with the highest volatility must be treated as
    high-vol (defensive); a 'CRASH'-labeled regime with the lowest volatility
    must be treated as low-vol (fully invested). Buckets never key off labels.
    """
    regime_infos = {
        0: _regime_info(0, "BULL", expected_volatility=0.08),  # mislabeled: actually HIGH vol
        1: _regime_info(1, "NEUTRAL", expected_volatility=0.02),  # actually MID vol
        2: _regime_info(2, "CRASH", expected_volatility=0.003),  # mislabeled: actually LOW vol
    }
    orchestrator = StrategyOrchestrator(CONFIG, regime_infos)

    assert isinstance(orchestrator.strategies[2], LowVolBullStrategy)
    assert isinstance(orchestrator.strategies[1], MidVolCautiousStrategy)
    assert isinstance(orchestrator.strategies[0], HighVolDefensiveStrategy)


def test_orchestrator_generate_signals_routes_by_state_id() -> None:
    regime_infos = {
        0: _regime_info(0, "CRASH", expected_volatility=0.05),
        1: _regime_info(1, "NEUTRAL", expected_volatility=0.02),
        2: _regime_info(2, "EUPHORIA", expected_volatility=0.005),
    }
    orchestrator = StrategyOrchestrator(CONFIG, regime_infos)
    bars = _make_bars(80, "up", seed=3)

    state = _regime_state(2, "EUPHORIA", 0.90)  # lowest-vol regime -> LowVolBullStrategy
    signals = orchestrator.generate_signals(["SPY"], {"SPY": bars}, state)

    assert len(signals) == 1
    assert signals[0].strategy_name == "LowVolBullStrategy"
    assert signals[0].position_size_pct == CONFIG.low_vol_allocation


def test_confidence_adjustment_scales_down_allocation() -> None:
    """Low regime confidence should halve position size and force leverage to 1.0x."""
    regime_infos = {0: _regime_info(0, "EUPHORIA", expected_volatility=0.005)}
    orchestrator = StrategyOrchestrator(CONFIG, regime_infos)
    bars = _make_bars(80, "up", seed=4)

    low_confidence_state = _regime_state(0, "EUPHORIA", probability=0.40)  # < min_confidence
    signals = orchestrator.generate_signals(["SPY"], {"SPY": bars}, low_confidence_state)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.position_size_pct == pytest.approx(
        CONFIG.low_vol_allocation * CONFIG.uncertainty_size_mult
    )
    assert signal.leverage == 1.0
    assert "[UNCERTAINTY — size halved]" in signal.reasoning
    assert signal.metadata["uncertainty_mode"] is True


def test_flickering_triggers_uncertainty_even_with_high_confidence() -> None:
    """is_flickering=True should force uncertainty mode regardless of confidence."""
    regime_infos = {0: _regime_info(0, "EUPHORIA", expected_volatility=0.005)}
    orchestrator = StrategyOrchestrator(CONFIG, regime_infos)
    bars = _make_bars(80, "up", seed=5)

    high_confidence_state = _regime_state(0, "EUPHORIA", probability=0.95)
    signals = orchestrator.generate_signals(
        ["SPY"], {"SPY": bars}, high_confidence_state, is_flickering=True
    )

    assert len(signals) == 1
    signal = signals[0]
    assert signal.position_size_pct == pytest.approx(
        CONFIG.low_vol_allocation * CONFIG.uncertainty_size_mult
    )
    assert signal.leverage == 1.0
    assert signal.metadata["uncertainty_flickering"] is True
    assert signal.metadata["uncertainty_low_confidence"] is False


def test_no_uncertainty_when_confident_and_stable() -> None:
    regime_infos = {0: _regime_info(0, "EUPHORIA", expected_volatility=0.005)}
    orchestrator = StrategyOrchestrator(CONFIG, regime_infos)
    bars = _make_bars(80, "up", seed=6)

    state = _regime_state(0, "EUPHORIA", probability=0.90)
    signals = orchestrator.generate_signals(["SPY"], {"SPY": bars}, state, is_flickering=False)

    assert len(signals) == 1
    assert signals[0].position_size_pct == CONFIG.low_vol_allocation
    assert "[UNCERTAINTY" not in signals[0].reasoning
    assert signals[0].metadata == {}


def test_rebalance_threshold_suppresses_small_drift() -> None:
    """needs_rebalance should return False when drift is below rebalance_threshold."""
    orchestrator = StrategyOrchestrator(CONFIG, {0: _regime_info(0, "NEUTRAL", 0.02)})

    assert orchestrator.needs_rebalance(0.90, 0.95) is False  # 5% drift, under 10%
    assert orchestrator.needs_rebalance(0.90, 0.95 + 0.11) is True  # >10% drift


def test_backward_compatible_aliases_point_to_expected_classes() -> None:
    assert CrashDefenseStrategy is HighVolDefensiveStrategy
    assert BearTrendStrategy is HighVolDefensiveStrategy
    assert MeanReversionStrategy is MidVolCautiousStrategy
    assert BullTrendStrategy is LowVolBullStrategy
    assert EuphoriaStrategy is LowVolBullStrategy


def test_label_to_strategy_covers_all_regime_labels() -> None:
    from core.hmm_engine import REGIME_LABELS_BY_COUNT

    all_labels = {label for labels in REGIME_LABELS_BY_COUNT.values() for label in labels}
    assert set(LABEL_TO_STRATEGY) == all_labels
    for strategy_cls in LABEL_TO_STRATEGY.values():
        assert issubclass(strategy_cls, (LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy))


def test_signal_not_generated_with_insufficient_bars() -> None:
    strategy = LowVolBullStrategy(_regime_info(0, "BULL", 0.005), CONFIG)
    tiny_bars = _make_bars(5, "up")
    state = _regime_state(0, "BULL", 0.9)

    assert strategy.generate_signal("SPY", tiny_bars, state) is None
