#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:18:19 2026

@author: poddar
"""



from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2


@dataclass(frozen=True)
class LjungBoxSpec:
    """
    Ljung-Box diagnostic specification.
    """
    lags: int = 20


class Diagnostics:
    """
    Diagnostics for (standardized) residual series.
    Provides:
    - Ljung-Box on z_t (serial correlation)
    - Ljung-Box on z_t^2 (remaining ARCH effects)
    """

    def __init__(self, lb_spec: Optional[LjungBoxSpec] = None):
        self.lb_spec = lb_spec or LjungBoxSpec()

    def ljung_box(self, x: pd.Series, lags: Optional[int] = None) -> Tuple[float, float]:
        """
        Compute Ljung-Box Q statistic and chi-square p-value.

        Parameters
        ----------
        x : pd.Series
            Input series (e.g., standardized residuals)
        lags : int
            Number of lags

        Returns
        -------
        (Q, p_value)
        """
        if not isinstance(x, pd.Series):
            raise TypeError("x must be a pandas Series.")
        x = x.dropna()
        if len(x) < 50:
            raise ValueError("Too few observations for Ljung-Box (need >= 50).")

        m = int(lags if lags is not None else self.lb_spec.lags)
        if m <= 0:
            raise ValueError("lags must be positive.")
        if m >= len(x) - 1:
            raise ValueError("lags too large for sample size.")

        acf = self._acf(x.values, nlags=m)
        n = len(x)

        # Ljung-Box Q
        # Q = n(n+2) sum_{k=1..m} rho_k^2 / (n-k)
        ks = np.arange(1, m + 1)
        Q = n * (n + 2.0) * np.sum((acf[1:] ** 2) / (n - ks))
        p = 1.0 - chi2.cdf(Q, df=m)
        return float(Q), float(p)

    def residual_diagnostics(self, z: pd.Series) -> Dict[str, float]:
        """
        Run diagnostics on standardized residuals z_t and z_t^2.

        Returns
        -------
        dict with:
            lb_Q_z, lb_p_z, lb_Q_z2, lb_p_z2
        """
        Qz, pz = self.ljung_box(z)
        Qz2, pz2 = self.ljung_box(z ** 2)

        return {
            "lb_Q_z": Qz,
            "lb_p_z": pz,
            "lb_Q_z2": Qz2,
            "lb_p_z2": pz2,
        }

    @staticmethod
    def gjr_persistence_proxy(params: pd.Series) -> float:
        """
        Heuristic persistence proxy for GJR-GARCH(1,1):

        For GJR, one common approximation is:
        $
        \\alpha + \\beta + \\gamma/2
        $

        This assumes symmetric probability of negative shocks.

        Returns NaN if required params missing.
        """
        if not isinstance(params, pd.Series):
            params = pd.Series(params)

        # arch uses names like: omega, alpha[1], gamma[1], beta[1]
        try:
            a = float(params.get("alpha[1]"))
            b = float(params.get("beta[1]"))
            g = float(params.get("gamma[1]"))
        except Exception:
            return float("nan")

        return float(a + b + 0.5 * g)

    @staticmethod
    def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
        """
        Simple ACF (biased) up to nlags, returning array of length nlags+1 including lag0.
        """
        x = np.asarray(x, dtype=float)
        x = x - x.mean()
        n = len(x)
        denom = np.dot(x, x)
        if denom <= 0:
            raise ValueError("Degenerate series for ACF.")

        acf = np.empty(nlags + 1, dtype=float)
        acf[0] = 1.0
        for k in range(1, nlags + 1):
            acf[k] = np.dot(x[: n - k], x[k:]) / denom
        return acf


class MultiAssetDiagnostics:
    """
    Run diagnostics across assets for fitted volatility models.
    Expects a dict-like {asset: model}, where model provides:
    - standardized_residuals() -> pd.Series
    - params() -> pd.Series
    """

    def __init__(self, diag: Optional[Diagnostics] = None):
        self.diag = diag or Diagnostics()

    def run(self, models: Dict[str, object]) -> pd.DataFrame:
        rows = []
        for asset, model in models.items():
            z = model.standardized_residuals()
            params = model.params()

            d = self.diag.residual_diagnostics(z)
            pers = self.diag.gjr_persistence_proxy(params)

            rows.append(
                {
                    "asset": asset,
                    **d,
                    "persistence_proxy": pers,
                }
            )

        return pd.DataFrame(rows).set_index("asset")