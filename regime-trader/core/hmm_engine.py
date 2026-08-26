"""HMM regime detection engine.

Design philosophy: the HMM is a VOLATILITY CLASSIFIER. It detects whether
the market is in a calm, moderate, or turbulent volatility environment —
it does NOT predict price direction. The strategy layer is responsible for
turning that classification into portfolio allocation.

Fits a Gaussian HMM (with automatic model-order selection via BIC) to a
market feature matrix, and classifies the current regime using strictly
causal (forward-algorithm) filtered inference — never the Viterbi
``predict()`` method, which revises past states using future observations
and would leak look-ahead bias into a backtest.

Regime *labels* (CRASH/BEAR/.../EUPHORIA) are assigned by sorting states by
mean return purely for human readability when reporting/logging; they do
not, by themselves, drive any strategy decision — the strategy layer keys
off volatility, not the label.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

#: Human-readable regime labels, assigned by sorting fitted states by mean
#: return (ascending). Purely for readability/logging — the strategy layer
#: sorts by volatility independently and does not key off these labels.
REGIME_LABELS_BY_COUNT: dict[int, tuple[str, ...]] = {
    3: ("BEAR", "NEUTRAL", "BULL"),
    4: ("CRASH", "BEAR", "BULL", "EUPHORIA"),
    5: ("CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"),
    6: ("CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"),
    7: (
        "CRASH",
        "STRONG_BEAR",
        "WEAK_BEAR",
        "NEUTRAL",
        "WEAK_BULL",
        "STRONG_BULL",
        "EUPHORIA",
    ),
}

#: Return feature column consulted to rank states by mean return for labeling.
RETURN_FEATURE_COLUMN = "return_1"

#: Default number of EM iterations per HMM fit.
DEFAULT_N_ITER = 200

#: Regime-metadata defaults keyed by "bucket" (label with STRONG_/WEAK_ stripped).
_REGIME_INFO_DEFAULTS: dict[str, dict[str, float | str]] = {
    "CRASH": {
        "recommended_strategy_type": "RISK_OFF",
        "max_leverage_allowed": 0.50,
        "max_position_size_pct": 0.05,
        "min_confidence_to_act": 0.70,
    },
    "BEAR": {
        "recommended_strategy_type": "DEFENSIVE",
        "max_leverage_allowed": 0.75,
        "max_position_size_pct": 0.08,
        "min_confidence_to_act": 0.65,
    },
    "NEUTRAL": {
        "recommended_strategy_type": "NEUTRAL",
        "max_leverage_allowed": 1.00,
        "max_position_size_pct": 0.12,
        "min_confidence_to_act": 0.60,
    },
    "BULL": {
        "recommended_strategy_type": "AGGRESSIVE",
        "max_leverage_allowed": 1.25,
        "max_position_size_pct": 0.15,
        "min_confidence_to_act": 0.55,
    },
    "EUPHORIA": {
        "recommended_strategy_type": "AGGRESSIVE",
        "max_leverage_allowed": 1.25,
        "max_position_size_pct": 0.15,
        "min_confidence_to_act": 0.55,
    },
}


def _regime_bucket(label: str) -> str:
    """Strip STRONG_/WEAK_ qualifiers to get the base regime-info bucket."""
    return label.replace("STRONG_", "").replace("WEAK_", "")


@dataclass
class RegimeInfo:
    """Static metadata describing one fitted regime state."""

    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """A single filtered regime observation at one point in time."""

    label: str
    state_id: int
    probability: float
    state_probabilities: dict[str, float]
    timestamp: pd.Timestamp
    is_confirmed: bool
    consecutive_bars: int


def _param_count(n_components: int, n_features: int, covariance_type: str) -> int:
    """Count free parameters of a GaussianHMM, for BIC computation."""
    transmat_params = n_components * (n_components - 1)
    startprob_params = n_components - 1
    means_params = n_components * n_features

    if covariance_type == "full":
        cov_params = n_components * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        cov_params = n_components * n_features
    elif covariance_type == "tied":
        cov_params = n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        cov_params = n_components
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    return transmat_params + startprob_params + means_params + cov_params


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
        self.n_candidates = list(n_candidates)
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.min_train_bars = min_train_bars
        self.stability_bars = stability_bars
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold
        self.min_confidence = min_confidence

        self.model: Optional[GaussianHMM] = None
        self.n_regimes: Optional[int] = None
        self.feature_columns: Optional[list[str]] = None
        self.state_labels: dict[int, str] = {}
        self.regime_info: dict[int, RegimeInfo] = {}
        self.training_metadata: dict = {}
        self.candidate_results: list[dict] = []

        self._reset_filtering_state()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def select_model(self, features: pd.DataFrame) -> GaussianHMM:
        """Select the best-fitting HMM across n_candidates via BIC.

        For each candidate ``n_components`` in ``self.n_candidates``, fits
        ``self.n_init`` randomly-initialized models and keeps the one with
        the highest log-likelihood for that candidate. The candidate with
        the lowest BIC across all tested state counts is returned. All
        candidate BIC scores are logged, along with the one selected.
        """
        X = features.values
        n_samples, n_features = X.shape

        candidate_results: list[dict] = []
        best_model: Optional[GaussianHMM] = None
        best_bic = np.inf
        best_n_components: Optional[int] = None

        for n_components in self.n_candidates:
            best_ll_for_n = -np.inf
            best_model_for_n: Optional[GaussianHMM] = None

            for init_idx in range(self.n_init):
                candidate = GaussianHMM(
                    n_components=n_components,
                    covariance_type=self.covariance_type,
                    n_iter=DEFAULT_N_ITER,
                    random_state=init_idx,
                )
                try:
                    candidate.fit(X)
                    log_likelihood = candidate.score(X)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    logger.warning(
                        "HMM fit failed for n_components=%d init=%d: %s",
                        n_components,
                        init_idx,
                        exc,
                    )
                    continue

                if log_likelihood > best_ll_for_n:
                    best_ll_for_n = log_likelihood
                    best_model_for_n = candidate

            if best_model_for_n is None:
                logger.warning(
                    "All %d initializations failed for n_components=%d; skipping candidate",
                    self.n_init,
                    n_components,
                )
                continue

            n_params = _param_count(n_components, n_features, self.covariance_type)
            bic = -2.0 * best_ll_for_n + n_params * np.log(n_samples)
            converged = bool(best_model_for_n.monitor_.converged)
            iterations = int(best_model_for_n.monitor_.iter)

            candidate_results.append(
                {
                    "n_components": n_components,
                    "log_likelihood": float(best_ll_for_n),
                    "bic": float(bic),
                    "n_params": n_params,
                    "converged": converged,
                    "iterations": iterations,
                }
            )
            logger.info(
                "HMM candidate n_components=%d bic=%.2f log_likelihood=%.2f "
                "converged=%s iterations=%d",
                n_components,
                bic,
                best_ll_for_n,
                converged,
                iterations,
            )

            if bic < best_bic:
                best_bic = bic
                best_model = best_model_for_n
                best_n_components = n_components

        if best_model is None or best_n_components is None:
            raise RuntimeError(
                "HMM model selection failed: no candidate converged for any "
                f"n_components in {self.n_candidates}"
            )

        self.candidate_results = candidate_results
        logger.info(
            "Selected HMM model: n_components=%d bic=%.2f (best of %d candidates)",
            best_n_components,
            best_bic,
            len(candidate_results),
        )
        return best_model

    def fit(
        self,
        features: pd.DataFrame,
        raw_returns: Optional[pd.Series] = None,
    ) -> None:
        """Fit the HMM to historical feature data.

        ``features`` should already be the standardized HMM feature matrix
        (e.g. from ``FeatureEngineer.build_feature_set``). If ``raw_returns``
        (unstandardized simple/log returns, aligned to ``features.index``)
        is supplied, it is used — via an offline Viterbi decode of the
        *training* data only — to compute real expected-return/volatility
        statistics per regime for reporting. This offline decode never
        touches live/backtest inference, which always uses
        ``predict_regime_filtered`` (forward algorithm) instead.
        """
        clean = features.dropna()
        if len(clean) < self.min_train_bars:
            raise ValueError(
                f"Need at least {self.min_train_bars} clean bars to train "
                f"(got {len(clean)})"
            )

        model = self.select_model(clean)
        self.model = model
        self.feature_columns = list(clean.columns)
        self.n_regimes = model.n_components

        self.state_labels = self.label_states(model)
        self.regime_info = self._build_regime_info(clean, raw_returns)
        self._reset_filtering_state()

        self.training_metadata = {
            "n_regimes": self.n_regimes,
            "bic": min(c["bic"] for c in self.candidate_results),
            "training_date": datetime.now(timezone.utc),
            "labels": dict(self.state_labels),
            "candidates": self.candidate_results,
            "n_samples": len(clean),
            "feature_columns": self.feature_columns,
        }
        logger.info(
            "HMM trained: n_regimes=%d labels=%s n_samples=%d",
            self.n_regimes,
            self.state_labels,
            len(clean),
        )

    # ------------------------------------------------------------------
    # Regime labeling
    # ------------------------------------------------------------------

    def _return_feature_index(self) -> int:
        """Index of the return column used to rank states for labeling."""
        if self.feature_columns and RETURN_FEATURE_COLUMN in self.feature_columns:
            return self.feature_columns.index(RETURN_FEATURE_COLUMN)
        return 0

    def label_states(self, model: GaussianHMM) -> dict[int, str]:
        """Map raw HMM state indices to human-readable regime labels.

        States are sorted by mean return (ascending) and assigned labels
        from ``REGIME_LABELS_BY_COUNT[model.n_components]`` accordingly:
        lowest return -> CRASH/BEAR, highest -> BULL/EUPHORIA.
        """
        n_components = model.n_components
        if n_components not in REGIME_LABELS_BY_COUNT:
            raise ValueError(
                f"No regime label set defined for {n_components} states "
                f"(supported: {sorted(REGIME_LABELS_BY_COUNT)})"
            )

        return_idx = self._return_feature_index()
        mean_returns = model.means_[:, return_idx]
        ascending_order = np.argsort(mean_returns)
        labels = REGIME_LABELS_BY_COUNT[n_components]

        return {int(raw_state): labels[rank] for rank, raw_state in enumerate(ascending_order)}

    def _build_regime_info(
        self, clean_features: pd.DataFrame, raw_returns: Optional[pd.Series]
    ) -> dict[int, RegimeInfo]:
        """Build per-state RegimeInfo metadata.

        Uses a Viterbi decode of the training set purely for offline
        regime characterization (expected return/volatility per state).
        This is safe here because it only ever looks at already-observed
        training history; it is never used for live/backtest inference.
        """
        assert self.model is not None
        return_idx = self._return_feature_index()

        expected_return: dict[int, float]
        expected_volatility: dict[int, float]

        if raw_returns is not None:
            decoded = self.model.predict(clean_features.values)
            aligned_returns = raw_returns.reindex(clean_features.index)
            expected_return = {}
            expected_volatility = {}
            for state in range(self.n_regimes):
                mask = decoded == state
                state_returns = aligned_returns[mask].dropna()
                if len(state_returns) > 0:
                    expected_return[state] = float(state_returns.mean())
                    expected_volatility[state] = float(state_returns.std())
                else:
                    expected_return[state] = float(self.model.means_[state, return_idx])
                    expected_volatility[state] = float(
                        np.sqrt(self._state_covariance(state)[return_idx, return_idx])
                    )
        else:
            expected_return = {
                state: float(self.model.means_[state, return_idx])
                for state in range(self.n_regimes)
            }
            expected_volatility = {
                state: float(np.sqrt(self._state_covariance(state)[return_idx, return_idx]))
                for state in range(self.n_regimes)
            }

        regime_info: dict[int, RegimeInfo] = {}
        for state in range(self.n_regimes):
            label = self.state_labels[state]
            defaults = _REGIME_INFO_DEFAULTS[_regime_bucket(label)]
            regime_info[state] = RegimeInfo(
                regime_id=state,
                regime_name=label,
                expected_return=expected_return[state],
                expected_volatility=expected_volatility[state],
                recommended_strategy_type=str(defaults["recommended_strategy_type"]),
                max_leverage_allowed=float(defaults["max_leverage_allowed"]),
                max_position_size_pct=float(defaults["max_position_size_pct"]),
                min_confidence_to_act=float(defaults["min_confidence_to_act"]),
            )
        return regime_info

    # ------------------------------------------------------------------
    # Forward-algorithm (filtered) inference — no look-ahead bias
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("HMMEngine.fit() must be called before inference")

    def _state_covariance(self, state: int) -> np.ndarray:
        """Full covariance matrix for ``state``, regardless of covariance_type."""
        assert self.model is not None
        n_features = len(self.feature_columns or [])
        if self.covariance_type == "full":
            return self.model.covars_[state]
        if self.covariance_type == "diag":
            return np.diag(self.model.covars_[state])
        if self.covariance_type == "spherical":
            return np.eye(n_features) * self.model.covars_[state]
        if self.covariance_type == "tied":
            return self.model.covars_
        raise ValueError(f"Unsupported covariance_type: {self.covariance_type}")

    def _log_emission_matrix(self, X: np.ndarray) -> np.ndarray:
        """log N(x_t; mean_state, cov_state) for every bar t and state."""
        assert self.model is not None
        n_samples = X.shape[0]
        log_emission = np.empty((n_samples, self.n_regimes))
        for state in range(self.n_regimes):
            dist = multivariate_normal(
                mean=self.model.means_[state],
                cov=self._state_covariance(state),
                allow_singular=True,
            )
            log_emission[:, state] = dist.logpdf(X)
        return log_emission

    def _reset_filtering_state(self) -> None:
        """Clear the forward-algorithm cache and stability-filter bookkeeping."""
        self._log_alpha_cache: Optional[np.ndarray] = None
        self._cache_index: Optional[pd.Index] = None
        self._confirmed_label: Optional[str] = None
        self._confirmed_consecutive_bars = 0
        self._pending_label: Optional[str] = None
        self._pending_count = 0
        self._change_event_positions: list[int] = []
        self._last_change_confirmed = False
        self._processed_bars = 0

    def _forward_pass(self, features_up_to_now: pd.DataFrame) -> np.ndarray:
        """Compute log P(state_t, obs_1..t) for every t via the forward algorithm.

        Reuses a cached alpha table when ``features_up_to_now`` extends a
        previously seen, unchanged prefix (same index values), so that
        repeated calls in a live/backtest loop only do incremental work.
        Regardless of caching, ``log_alpha[t]`` is always a pure function
        of observations ``0..t`` only — it never depends on later rows,
        which is what guarantees no look-ahead bias.
        """
        assert self.model is not None
        X = features_up_to_now[self.feature_columns].values
        n_samples = X.shape[0]

        log_startprob = np.log(np.clip(self.model.startprob_, 1e-300, 1.0))
        log_transmat = np.log(np.clip(self.model.transmat_, 1e-300, 1.0))
        log_emission = self._log_emission_matrix(X)

        log_alpha = np.empty((n_samples, self.n_regimes))
        start_idx = 0

        if (
            self._log_alpha_cache is not None
            and self._cache_index is not None
            and n_samples >= len(self._cache_index)
            and list(features_up_to_now.index[: len(self._cache_index)]) == list(self._cache_index)
        ):
            cached_len = len(self._cache_index)
            log_alpha[:cached_len] = self._log_alpha_cache
            start_idx = cached_len

        if start_idx == 0 and n_samples > 0:
            log_alpha[0] = log_startprob + log_emission[0]
            start_idx = 1

        for t in range(start_idx, n_samples):
            prev = log_alpha[t - 1]
            log_alpha[t] = logsumexp(prev[:, None] + log_transmat, axis=0) + log_emission[t]

        if n_samples > 0:
            self._log_alpha_cache = log_alpha.copy()
            self._cache_index = features_up_to_now.index

        return log_alpha

    @staticmethod
    def _normalize_log_probs(log_probs: np.ndarray) -> np.ndarray:
        """Convert a log-probability vector into a normalized probability vector."""
        return np.exp(log_probs - logsumexp(log_probs))

    def _advance_stability_filter(self, raw_label: str) -> tuple[str, bool, int, bool]:
        """Advance the stability filter by one bar of raw (argmax) label.

        Returns (displayed_label, is_confirmed, consecutive_bars, changed_now).
        A regime change is only reflected in the displayed label once the
        new raw label has persisted for ``stability_bars`` consecutive
        bars; until then, the previous confirmed label is kept (and
        ``is_confirmed`` is False, signaling the strategy layer should
        reduce sizing during the transition).
        """
        changed_now = False

        if self._confirmed_label is None:
            self._confirmed_label = raw_label
            self._confirmed_consecutive_bars = 1
            self._pending_label = None
            self._pending_count = 0
            return self._confirmed_label, True, self._confirmed_consecutive_bars, changed_now

        if raw_label == self._confirmed_label:
            self._pending_label = None
            self._pending_count = 0
            self._confirmed_consecutive_bars += 1
            return self._confirmed_label, True, self._confirmed_consecutive_bars, changed_now

        # raw_label differs from the currently confirmed regime.
        if raw_label == self._pending_label:
            self._pending_count += 1
        else:
            self._pending_label = raw_label
            self._pending_count = 1

        if self._pending_count >= self.stability_bars:
            logger.warning(
                "Regime change confirmed: %s -> %s (after %d bars)",
                self._confirmed_label,
                self._pending_label,
                self._pending_count,
            )
            self._confirmed_label = self._pending_label
            self._confirmed_consecutive_bars = self._pending_count
            self._pending_label = None
            self._pending_count = 0
            changed_now = True
            return self._confirmed_label, True, self._confirmed_consecutive_bars, changed_now

        logger.debug(
            "Regime transition pending: %s challenged by %s (%d/%d bars)",
            self._confirmed_label,
            raw_label,
            self._pending_count,
            self.stability_bars,
        )
        return self._confirmed_label, False, self._confirmed_consecutive_bars, changed_now

    def predict_regime_filtered(self, features_up_to_now: pd.DataFrame) -> list[RegimeState]:
        """Compute the filtered (forward-algorithm) regime at every bar.

        For each bar ``t``, computes ``P(state_t | obs_1..t)`` using only
        observations up to and including ``t`` — never
        ``model.predict()`` (Viterbi), which smooths using the *entire*
        sequence and would leak future information into the classification
        of past bars. As a direct consequence, the regime computed for a
        given bar is identical whether it is the last bar supplied or an
        interior bar of a longer, later-supplied sequence (see
        ``tests/test_look_ahead.py``).

        Also applies the stability filter (confirmed only after persisting
        ``stability_bars`` bars) and updates flicker-rate bookkeeping.
        """
        self._require_fitted()
        n_samples = len(features_up_to_now)
        if n_samples == 0:
            return []

        log_alpha = self._forward_pass(features_up_to_now)

        results: list[RegimeState] = []
        for t in range(n_samples):
            probs_t = self._normalize_log_probs(log_alpha[t])
            raw_state = int(np.argmax(probs_t))
            raw_label = self.state_labels[raw_state]

            displayed_label, is_confirmed, consecutive_bars, changed_now = (
                self._advance_stability_filter(raw_label)
            )
            self._last_change_confirmed = changed_now
            self._processed_bars += 1
            if changed_now:
                self._change_event_positions.append(self._processed_bars - 1)

            state_probabilities = {
                self.state_labels[s]: float(probs_t[s]) for s in range(self.n_regimes)
            }
            results.append(
                RegimeState(
                    label=displayed_label,
                    state_id=raw_state,
                    probability=float(probs_t[raw_state]),
                    state_probabilities=state_probabilities,
                    timestamp=features_up_to_now.index[t],
                    is_confirmed=is_confirmed,
                    consecutive_bars=consecutive_bars,
                )
            )

        return results

    def predict_regime(self, features: pd.DataFrame) -> RegimeState:
        """Predict the current (most recent) regime from filtered inference."""
        results = self.predict_regime_filtered(features)
        if not results:
            raise ValueError("predict_regime requires at least one row of features")
        return results[-1]

    def predict_regime_proba(self, features_up_to_now: pd.DataFrame) -> dict[str, float]:
        """Return the filtered regime probability distribution for the latest bar."""
        return self.predict_regime(features_up_to_now).state_probabilities

    # ------------------------------------------------------------------
    # Stability / flicker introspection
    # ------------------------------------------------------------------

    def get_regime_stability(self) -> int:
        """Consecutive bars the currently confirmed regime has held."""
        return self._confirmed_consecutive_bars

    def get_transition_matrix(self) -> pd.DataFrame:
        """Learned state transition probabilities, indexed/labeled by regime."""
        self._require_fitted()
        assert self.model is not None
        labels = [self.state_labels[i] for i in range(self.n_regimes)]
        return pd.DataFrame(self.model.transmat_, index=labels, columns=labels)

    def detect_regime_change(self) -> bool:
        """True only if the most recently processed bar confirmed a regime change."""
        return self._last_change_confirmed

    def get_regime_flicker_rate(self) -> int:
        """Number of confirmed regime changes within the trailing flicker_window bars."""
        cutoff = self._processed_bars - self.flicker_window
        return sum(1 for pos in self._change_event_positions if pos >= cutoff)

    def is_flickering(self) -> bool:
        """True if the flicker rate exceeds flicker_threshold (forces uncertainty mode)."""
        return self.get_regime_flicker_rate() > self.flicker_threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str | Path) -> None:
        """Pickle the fitted model together with its labeling/metadata."""
        self._require_fitted()
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "state_labels": self.state_labels,
            "regime_info": self.regime_info,
            "training_metadata": self.training_metadata,
            "config": {
                "n_candidates": self.n_candidates,
                "n_init": self.n_init,
                "covariance_type": self.covariance_type,
                "min_train_bars": self.min_train_bars,
                "stability_bars": self.stability_bars,
                "flicker_window": self.flicker_window,
                "flicker_threshold": self.flicker_threshold,
                "min_confidence": self.min_confidence,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_model(self, path: str | Path) -> None:
        """Load a model previously saved with ``save_model``."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]
        self.state_labels = payload["state_labels"]
        self.regime_info = payload["regime_info"]
        self.training_metadata = payload["training_metadata"]
        self.n_regimes = self.model.n_components
        self._reset_filtering_state()
