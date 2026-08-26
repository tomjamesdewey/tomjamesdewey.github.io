"""Feature engineering: technical indicators and HMM feature computation."""

from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    """Computes technical indicators and feature sets used by the HMM engine."""

    def __init__(self) -> None:
        """Initialize the feature engineer."""
        ...

    def compute_returns(self, bars: pd.DataFrame) -> pd.Series:
        """Compute log or simple returns from OHLCV bars."""
        ...

    def compute_realized_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """Compute rolling realized volatility from returns."""
        ...

    def compute_trend_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute trend indicators (e.g. moving averages, ADX) from OHLCV bars."""
        ...

    def compute_hmm_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Assemble the feature matrix used as HMM model input."""
        ...

    def build_feature_set(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Build the full feature set (returns, volatility, trend) for a symbol."""
        ...
