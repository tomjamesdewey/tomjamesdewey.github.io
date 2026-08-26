"""Volatility-regime-based allocation strategies.

Translates a detected market regime (and trend context) into target
portfolio allocation and leverage, per the thresholds in settings.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.hmm_engine import RegimeState


@dataclass
class AllocationTarget:
    """Target allocation and leverage produced by a regime strategy."""

    allocation: float
    leverage: float
    regime_label: str
    confidence: float


class RegimeStrategy:
    """Maps regime states to target portfolio allocation and leverage."""

    def __init__(
        self,
        low_vol_allocation: float,
        mid_vol_allocation_trend: float,
        mid_vol_allocation_no_trend: float,
        high_vol_allocation: float,
        low_vol_leverage: float,
        rebalance_threshold: float,
        uncertainty_size_mult: float,
    ) -> None:
        """Store strategy allocation parameters."""
        ...

    def get_allocation(
        self, regime: RegimeState, trend_confirmed: bool
    ) -> AllocationTarget:
        """Compute target allocation/leverage for the current regime and trend."""
        ...

    def apply_confidence_adjustment(
        self, target: AllocationTarget, confidence: float
    ) -> AllocationTarget:
        """Scale down target allocation when regime confidence is low."""
        ...

    def needs_rebalance(self, current_allocation: float, target_allocation: float) -> bool:
        """Determine whether allocation drift exceeds rebalance_threshold."""
        ...
