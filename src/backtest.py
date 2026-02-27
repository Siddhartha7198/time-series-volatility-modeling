#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:12:20 2026

@author: poddar
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.volatility_models import GJRGARCHModel, GARCHSpec, MultiAssetVolatilityFitter
from src.correlation_models import EWMACorrelation, EWMACorrSpec
from src.var_engine import VaREngine, VaRSpec


@dataclass(frozen=True)
class BacktestSpec:
    """
    Rolling backtest specification for 1-day VaR/ES.
    """
    train_window: int = 750
    corr_window: int = 252
    ewma_lambda: float = 0.97

    # risk level
    alpha: float = 0.99

    # simulation
    n_sims: int = 20000
    random_seed: int = 42
    sim_dist: str = "student_t"   # {"gaussian","student_t"}
    nu: float = 8.0               # df for multivariate t (nu>2)

    # GARCH model spec (univariate)
    garch_spec: GARCHSpec = GARCHSpec(p=1, o=1, q=1, dist="t", mean="Zero")


class VaRBacktester:
    """
    Rolling 1-day VaR/ES backtest:
    - per-asset GJR-GARCH-t
    - EWMA correlation on standardized residuals
    - multivariate simulation (Gaussian or Student-t)
    """

    def __init__(self, spec: BacktestSpec):
        self.spec = spec
        self._engine = VaREngine(
            VaRSpec(
                alpha=spec.alpha,
                n_sims=spec.n_sims,
                random_seed=spec.random_seed,
                sim_dist=spec.sim_dist,
                nu=spec.nu,
            )
        )

    def run(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ) -> pd.DataFrame:
        self._validate_inputs(returns, weights)

        T, N = returns.shape
        w = np.asarray(weights, dtype=float)

        t0 = self.spec.train_window if start_index is None else int(start_index)
        t1 = (T - 1) if end_index is None else int(end_index)  # need t+1 realizations

        if t0 < self.spec.train_window:
            t0 = self.spec.train_window
        if t1 > T - 1:
            t1 = T - 1
        if t0 >= t1:
            raise ValueError("Invalid start/end indices for backtest.")

        rows = []

        for t in range(t0, t1):
            train = returns.iloc[t - self.spec.train_window: t].copy()
            forecast_date = returns.index[t + 1]

            # 1) Fit per-asset GJR-GARCH-t
            base_model = GJRGARCHModel(spec=self.spec.garch_spec, scale=100.0)
            mvf = MultiAssetVolatilityFitter(base_model).fit(train)

            # 2) Correlation from standardized residuals (EWMA)
            z = mvf.standardized_residual_matrix()
            corr_spec = EWMACorrSpec(lam=self.spec.ewma_lambda, window=self.spec.corr_window)
            R_t = EWMACorrelation(corr_spec).fit(z).correlation_matrix()

            # 3) One-step variance forecasts vector
            var_fc = mvf.forecast_one_step_variances().loc[returns.columns].values

            # 4) VaR + ES forecast
            VaR_t, ES_t, _ = self._engine.compute_var_es(weights=w, var_forecast=var_fc, corr=R_t)

            # 5) Realized next-day portfolio return
            realized_rp = float(returns.iloc[t + 1].values @ w)

            breach = int(realized_rp < -VaR_t)

            rows.append(
                {
                    "date": forecast_date,
                    "VaR": float(VaR_t),
                    "ES": float(ES_t),
                    "realized_portfolio_return": float(realized_rp),
                    "breach": int(breach),
                }
            )

        return pd.DataFrame(rows).set_index("date")

    @staticmethod
    def summarize(backtest_df: pd.DataFrame, alpha: float) -> Dict[str, float]:
        if backtest_df.empty:
            raise ValueError("backtest_df is empty.")
        n = len(backtest_df)
        breaches = int(backtest_df["breach"].sum())
        freq = breaches / n
        expected = 1.0 - alpha
        avg_var = float(backtest_df["VaR"].mean())
        avg_es = float(backtest_df["ES"].mean())
        return {
            "n_obs": float(n),
            "breaches": float(breaches),
            "breach_freq": float(freq),
            "expected_freq": float(expected),
            "breach_freq_minus_expected": float(freq - expected),
            "avg_VaR": float(avg_var),
            "avg_ES": float(avg_es),
        }

    @staticmethod
    def _validate_inputs(returns: pd.DataFrame, weights: np.ndarray) -> None:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        if returns.isna().any().any():
            raise ValueError("returns contains NaNs.")
        if not returns.index.is_monotonic_increasing:
            raise ValueError("returns index must be sorted increasing by date.")
        if returns.shape[1] < 2:
            raise ValueError("Need at least 2 assets for portfolio risk backtest.")

        w = np.asarray(weights, dtype=float)
        if w.shape != (returns.shape[1],):
            raise ValueError("weights shape must match number of assets.")
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("weights must sum to 1.0.")
        if not np.isfinite(w).all():
            raise ValueError("weights must be finite.")