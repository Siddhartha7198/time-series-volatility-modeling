#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:23:33 2026

@author: poddar
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EWMACorrSpec:
    """
    EWMA correlation specification.
    """
    lam: float = 0.97
    window: int = 252
    jitter: float = 1e-12  # numerical stability


class CorrelationModel:
    """
    Base interface.
    """
    def fit(self, z: pd.DataFrame) -> "CorrelationModel":
        raise NotImplementedError

    def correlation_matrix(self) -> np.ndarray:
        raise NotImplementedError


class EWMACorrelation(CorrelationModel):
    """
    EWMA correlation of standardized residuals.

    Notes:
    - z should be standardized residuals (approximately iid with unit variance),
      so the EWMA covariance of z is effectively an EWMA correlation driver.
    - We compute correlation explicitly to normalize scale.
    """

    def __init__(self, spec: Optional[EWMACorrSpec] = None):
        self.spec = spec or EWMACorrSpec()
        self._corr: Optional[np.ndarray] = None
        self._assets: Optional[list[str]] = None

    def fit(self, z: pd.DataFrame) -> "EWMACorrelation":
        self._validate_z(z)
        self._assets = list(z.columns)

        z_win = z.tail(self.spec.window)
        Z = z_win.values  # shape (T, N)
        T, N = Z.shape

        lam = float(self.spec.lam)
        if not (0.0 < lam < 1.0):
            raise ValueError("lam must be in (0, 1).")

        # EWMA weights (most recent highest weight)
        # weights: [lam^(T-1), ..., lam^1, lam^0]
        powers = np.arange(T - 1, -1, -1, dtype=float)
        w = lam ** powers
        w = w / w.sum()

        # Weighted second moment S = sum_t w_t z_t z_t'
        S = np.zeros((N, N), dtype=float)
        for t in range(T):
            zt = Z[t, :].reshape(N, 1)
            S += w[t] * (zt @ zt.T)

        # Stabilize (avoid tiny negative eigenvalues from numerics)
        S = self._make_psd(S, jitter=self.spec.jitter)

        # Convert to correlation
        d = np.sqrt(np.clip(np.diag(S), self.spec.jitter, np.inf))
        Corr = S / np.outer(d, d)

        # Clip correlations to valid open interval to avoid numerical issues downstream
        Corr = np.clip(Corr, -0.999, 0.999)
        np.fill_diagonal(Corr, 1.0)

        Corr = self._make_psd(Corr, jitter=self.spec.jitter)
        self._corr = Corr
        return self

    def correlation_matrix(self) -> np.ndarray:
        if self._corr is None:
            raise ValueError("Model not fitted.")
        return self._corr.copy()

    def assets(self) -> list[str]:
        if self._assets is None:
            raise ValueError("Model not fitted.")
        return list(self._assets)

    @staticmethod
    def _validate_z(z: pd.DataFrame) -> None:
        if not isinstance(z, pd.DataFrame):
            raise TypeError("z must be a pandas DataFrame.")
        if z.shape[1] < 2:
            raise ValueError("Need at least 2 assets.")
        if z.isna().any().any():
            raise ValueError("z contains NaNs. Ensure standardized residuals are clean/aligned.")
        if not z.index.is_monotonic_increasing:
            raise ValueError("z index must be sorted increasing by date.")
        if len(z) < 50:
            raise ValueError("Too few observations for correlation estimation (need >= 50).")

    @staticmethod
    def _make_psd(A: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
        """
        Ensure matrix is positive semidefinite by eigenvalue clipping.
        """
        A = 0.5 * (A + A.T)
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals_clipped = np.clip(eigvals, jitter, np.inf)
        return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T