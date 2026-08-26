"""Shared pytest fixtures for regime-trader tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyConfig, StrategyOrchestrator
from data.feature_engineering import FeatureEngineer


def make_mocked_alpaca_client(monkeypatch, equity: str = "100000", cash: str = "50000", buying_power: str = "80000"):
    """An AlpacaClient with TradingClient/StockHistoricalDataClient replaced
    by MagicMocks (so no network call ever happens), pre-wired with a
    healthy get_clock response (needed for the constructor's health
    check) and a default get_account response.
    """
    import broker.alpaca_client as alpaca_client_module

    monkeypatch.setattr(alpaca_client_module, "TradingClient", MagicMock())
    monkeypatch.setattr(alpaca_client_module, "StockHistoricalDataClient", MagicMock())

    client = alpaca_client_module.AlpacaClient("test-key", "test-secret", paper=True)

    healthy_clock = MagicMock()
    healthy_clock.model_dump.return_value = {"is_open": True}
    client.trading_client.get_clock.return_value = healthy_clock

    account = MagicMock()
    account.model_dump.return_value = {"equity": equity, "cash": cash, "buying_power": buying_power}
    client.trading_client.get_account.return_value = account

    return client

# Kept small so HMM fitting stays fast in tests; still >= a full
# standardization window (252) + SMA-200 warm-up so build_feature_set
# yields plenty of valid rows.
N_BARS = 900

# A full walk-forward run needs >= warm-up (~452 raw bars) + train_window
# (252) + test_window (126) valid feature rows for even one window; this
# gives a few windows' worth of walk-forward coverage.
N_WALKFORWARD_BARS = 1300

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

#: Fast-fitting walk-forward window sizes, smaller than settings.yaml's
#: production 252/126/126 so tests stay quick.
TEST_WALKFORWARD_KWARGS = dict(train_window=252, test_window=126, step_size=126)

#: Mirrors the ``strategy`` section of settings.yaml.
TEST_STRATEGY_CONFIG = StrategyConfig(
    low_vol_allocation=0.95,
    mid_vol_allocation_trend=0.95,
    mid_vol_allocation_no_trend=0.60,
    high_vol_allocation=0.60,
    low_vol_leverage=1.25,
    rebalance_threshold=0.10,
    uncertainty_size_mult=0.50,
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


@pytest.fixture(scope="session")
def walkforward_bars() -> pd.DataFrame:
    return make_synthetic_bars(n_bars=N_WALKFORWARD_BARS, seed=7)


def make_backtester(hmm_kwargs: dict | None = None, walkforward_kwargs: dict | None = None):
    from backtest.backtester import Backtester

    hmm_template = HMMEngine(**(hmm_kwargs or TEST_HMM_KWARGS))
    strategy_template = StrategyOrchestrator(TEST_STRATEGY_CONFIG, {})
    windows = walkforward_kwargs or TEST_WALKFORWARD_KWARGS
    return Backtester(
        hmm_template,
        strategy_template,
        None,
        initial_capital=100_000.0,
        slippage_pct=0.0005,
        **windows,
    )


@pytest.fixture(scope="session")
def backtester():
    return make_backtester()


@pytest.fixture(scope="session")
def walkforward_result(backtester, walkforward_bars: pd.DataFrame):
    return backtester.run({"TEST": walkforward_bars})
