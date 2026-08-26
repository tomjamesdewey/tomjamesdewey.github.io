"""Feature engineering: technical indicators and HMM feature computation.

All indicator computations are implemented as pure, stateless module-level
functions of their inputs (no I/O, no mutation of arguments). The
``FeatureEngineer`` class is a thin orchestrator that assembles them into
the feature matrix consumed by ``core.hmm_engine.HMMEngine``.

Every feature is causal: the value at index ``t`` depends only on data at
or before ``t``, which is required for the HMM engine's no-look-ahead
guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

RETURN_PERIODS = (1, 5, 20)
REALIZED_VOL_SHORT_WINDOW = 5
REALIZED_VOL_LONG_WINDOW = 20
VOLUME_ZSCORE_WINDOW = 50
VOLUME_TREND_SMA_WINDOW = 10
VOLUME_TREND_SLOPE_WINDOW = 10
ADX_WINDOW = 14
PRICE_TREND_SMA_WINDOW = 50
PRICE_TREND_SLOPE_WINDOW = 10
RSI_WINDOW = 14
MEAN_REVERSION_SMA_WINDOW = 200
ROC_PERIODS = (10, 20)
ATR_WINDOW = 14
STANDARDIZATION_WINDOW = 252


def _validate_ohlcv(bars: pd.DataFrame) -> None:
    """Raise if ``bars`` is missing any required OHLCV column."""
    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing required OHLCV columns: {missing}")


def log_return(close: pd.Series, period: int) -> pd.Series:
    """Log return of ``close`` over ``period`` bars: log(C_t / C_{t-period})."""
    return np.log(close / close.shift(period))


def realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of ``returns`` over ``window`` bars."""
    return returns.rolling(window=window, min_periods=window).std()


def volatility_ratio(vol_short: pd.Series, vol_long: pd.Series) -> pd.Series:
    """Ratio of a short-window volatility series to a long-window one."""
    return vol_short / vol_long


def volume_zscore(volume: pd.Series, window: int = VOLUME_ZSCORE_WINDOW) -> pd.Series:
    """Rolling z-score of volume vs. its own trailing ``window``-bar mean/std."""
    rolling_mean = volume.rolling(window=window, min_periods=window).mean()
    rolling_std = volume.rolling(window=window, min_periods=window).std()
    return (volume - rolling_mean) / rolling_std


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling linear-regression slope of ``series`` over trailing ``window`` bars.

    Fits y = a + b*x on the last ``window`` points (x = 0..window-1) at each
    step and returns b, i.e. the per-bar rate of change of the series level.
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_demeaned = x - x_mean
    denom = float((x_demeaned**2).sum())

    def _slope(y: np.ndarray) -> float:
        return float((x_demeaned * (y - y.mean())).sum() / denom)

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def average_directional_index(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = ADX_WINDOW
) -> pd.Series:
    """Average Directional Index (ADX) over ``window`` bars."""
    return ADXIndicator(high=high, low=low, close=close, window=window).adx()


def relative_strength_index(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Relative Strength Index (RSI) over ``window`` bars."""
    return RSIIndicator(close=close, window=window).rsi()


def distance_from_sma_pct(close: pd.Series, window: int) -> pd.Series:
    """Distance of ``close`` from its ``window``-bar SMA, as a % of price."""
    sma = close.rolling(window=window, min_periods=window).mean()
    return (close - sma) / sma * 100.0


def rate_of_change(close: pd.Series, period: int) -> pd.Series:
    """Rate of change of ``close`` over ``period`` bars, as a percentage."""
    return (close - close.shift(period)) / close.shift(period) * 100.0


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = ATR_WINDOW
) -> pd.Series:
    """Average True Range (ATR) over ``window`` bars."""
    return AverageTrueRange(high=high, low=low, close=close, window=window).average_true_range()


def rolling_zscore(features: pd.DataFrame, window: int = STANDARDIZATION_WINDOW) -> pd.DataFrame:
    """Standardize every column of ``features`` using a rolling z-score.

    ``z_t = (x_t - rolling_mean(x, window)) / rolling_std(x, window)``,
    computed independently per column using only data at or before ``t``.
    """
    rolling_mean = features.rolling(window=window, min_periods=window).mean()
    rolling_std = features.rolling(window=window, min_periods=window).std()
    return (features - rolling_mean) / rolling_std


class FeatureEngineer:
    """Computes technical indicators and feature sets used by the HMM engine."""

    def __init__(self, standardization_window: int = STANDARDIZATION_WINDOW) -> None:
        """Store the rolling lookback used for final feature standardization."""
        self.standardization_window = standardization_window

    def compute_returns(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns over 1, 5, and 20 periods."""
        _validate_ohlcv(bars)
        return pd.DataFrame(
            {f"return_{p}": log_return(bars["close"], p) for p in RETURN_PERIODS},
            index=bars.index,
        )

    def compute_realized_volatility(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute realized volatility (20-period) and the 5/20 volatility ratio."""
        _validate_ohlcv(bars)
        returns_1 = log_return(bars["close"], 1)
        vol_short = realized_volatility(returns_1, REALIZED_VOL_SHORT_WINDOW)
        vol_long = realized_volatility(returns_1, REALIZED_VOL_LONG_WINDOW)
        return pd.DataFrame(
            {
                "realized_vol_20": vol_long,
                "vol_ratio_5_20": volatility_ratio(vol_short, vol_long),
            },
            index=bars.index,
        )

    def compute_volume_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute normalized volume z-score and volume trend (SMA slope)."""
        _validate_ohlcv(bars)
        volume_sma = bars["volume"].rolling(
            window=VOLUME_TREND_SMA_WINDOW, min_periods=VOLUME_TREND_SMA_WINDOW
        ).mean()
        return pd.DataFrame(
            {
                "volume_zscore_50": volume_zscore(bars["volume"], VOLUME_ZSCORE_WINDOW),
                "volume_trend_10": rolling_slope(volume_sma, VOLUME_TREND_SLOPE_WINDOW),
            },
            index=bars.index,
        )

    def compute_trend_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute ADX(14) and the slope of the 50-period SMA."""
        _validate_ohlcv(bars)
        price_sma = bars["close"].rolling(
            window=PRICE_TREND_SMA_WINDOW, min_periods=PRICE_TREND_SMA_WINDOW
        ).mean()
        return pd.DataFrame(
            {
                "adx_14": average_directional_index(
                    bars["high"], bars["low"], bars["close"], ADX_WINDOW
                ),
                "sma_slope_50": rolling_slope(price_sma, PRICE_TREND_SLOPE_WINDOW),
            },
            index=bars.index,
        )

    def compute_mean_reversion_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute RSI(14) and distance from the 200-period SMA (% of price)."""
        _validate_ohlcv(bars)
        return pd.DataFrame(
            {
                "rsi_14": relative_strength_index(bars["close"], RSI_WINDOW),
                "dist_from_sma200_pct": distance_from_sma_pct(
                    bars["close"], MEAN_REVERSION_SMA_WINDOW
                ),
            },
            index=bars.index,
        )

    def compute_momentum_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute rate of change over 10 and 20 periods."""
        _validate_ohlcv(bars)
        return pd.DataFrame(
            {f"roc_{p}": rate_of_change(bars["close"], p) for p in ROC_PERIODS},
            index=bars.index,
        )

    def compute_range_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute normalized ATR (ATR(14) / close)."""
        _validate_ohlcv(bars)
        atr = average_true_range(bars["high"], bars["low"], bars["close"], ATR_WINDOW)
        return pd.DataFrame({"atr_norm_14": atr / bars["close"]}, index=bars.index)

    def compute_hmm_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Assemble the full raw (pre-standardization) HMM feature matrix."""
        _validate_ohlcv(bars)
        return pd.concat(
            [
                self.compute_returns(bars),
                self.compute_realized_volatility(bars),
                self.compute_volume_features(bars),
                self.compute_trend_indicators(bars),
                self.compute_mean_reversion_features(bars),
                self.compute_momentum_features(bars),
                self.compute_range_features(bars),
            ],
            axis=1,
        )

    def build_feature_set(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Build the final, standardized HMM feature matrix for a symbol.

        Computes all raw features, then standardizes every column with a
        rolling z-score (``standardization_window``-period lookback) and
        drops warm-up rows containing NaNs.
        """
        raw_features = self.compute_hmm_features(bars)
        standardized = rolling_zscore(raw_features, self.standardization_window)
        return standardized.dropna()
