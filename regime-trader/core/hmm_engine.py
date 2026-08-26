"""HMM regime detection engine.

Fits Gaussian Hidden Markov Models to market features and classifies the
current market regime (e.g. low/mid/high volatility) with a confidence
score, applying stability and flicker filtering to avoid noisy switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class RegimeState:
    """Represents a single detected regime state at a point in time."""

    regime_id: int
    label: str
    confidence: float
    timestamp: pd.Timestamp


class HMMEngine:
    """Fits and applies a Gaussian HMM for market regime detection."""

    def __init__(
        self,
        n_candidates: list[int],
        n_init: int,
        covariance_type: str,
        min_train_bars: int,
        stability_bars: int,
        flicker_window: int,
        flicker_threshold: int,
        min_confidence: float,
    ) -> None:
        """Store HMM configuration parameters."""
        ...

    def select_model(self, features: pd.DataFrame) -> GaussianHMM:
        """Select the best-fitting HMM across n_candidates via BIC/AIC."""
        ...

    def fit(self, features: pd.DataFrame) -> None:
        """Fit the HMM to historical feature data."""
        ...

    def predict_regime(self, features: pd.DataFrame) -> RegimeState:
        """Predict the current regime and confidence from recent features."""
        ...

    def label_states(self, model: GaussianHMM) -> dict[int, str]:
        """Map raw HMM state indices to human-readable regime labels."""
        ...

    def apply_stability_filter(self, raw_states: pd.Series) -> pd.Series:
        """Smooth raw state predictions using the stability_bars threshold."""
        ...

    def detect_flicker(self, states: pd.Series) -> bool:
        """Detect excessive regime switching within the flicker_window."""
        ...
