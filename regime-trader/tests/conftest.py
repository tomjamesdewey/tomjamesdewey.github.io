"""Shared pytest fixtures for regime-trader tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMEngine
from data.feature_engineering import FeatureEngineer

# Kept small so HMM fitting stays fast in tests; still >= a full
# standardization window (252) + SMA-200 warm-up so build_feature_set
# yields plenty of valid rows.
N_BARS = 900

#: Fast-fitting HMM config shared by tests that don't care about the exact
#: production settings.yaml values.
TEST_HMM_KWARGS = dict(
    n_candidates=[3, 4],
    n_init=2,
    covariance_type="full",
    min_train_bars=100,
    stability_bars=3,
    flicker_window=20,
    flicker_threshold=4,
    min_confidence=0.55,
)


def make_synthetic_bars(n_bars: int = N_BARS, seed: int = 42) -> pd.DataFrame:
    """Build deterministic synthetic OHLCV bars with two alternating vol regimes."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="B")

    returns = np.empty(n_bars)
    vol_state = 0
    block = 150
    for i in range(n_bars):
        if i % block == 0:
            vol_state = 1 - vol_state
        vol = 0.005 if vol_state == 0 else 0.03
        mu = 0.0006 if vol_state == 0 else -0.0004
        returns[i] = rng.normal(mu, vol)

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n_bars)))
    open_ = close * (1 + rng.normal(0, 0.001, n_bars))
    volume = rng.randint(1_000_000, 5_000_000, n_bars).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture(scope="session")
def synthetic_bars() -> pd.DataFrame:
    return make_synthetic_bars()


@pytest.fixture(scope="session")
def synthetic_features(synthetic_bars: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer().build_feature_set(synthetic_bars)


@pytest.fixture(scope="session")
def trained_engine(synthetic_features: pd.DataFrame) -> HMMEngine:
    engine = HMMEngine(**TEST_HMM_KWARGS)
    engine.fit(synthetic_features)
    return engine


def fresh_inference_copy(engine: HMMEngine) -> HMMEngine:
    """A new HMMEngine sharing ``engine``'s fitted model but with fresh,
    independent filtering/stability state (no shared cache or history).
    """
    clone = HMMEngine(**TEST_HMM_KWARGS)
    clone.model = engine.model
    clone.feature_columns = engine.feature_columns
    clone.n_regimes = engine.n_regimes
    clone.state_labels = engine.state_labels
    clone.regime_info = engine.regime_info
    return clone
