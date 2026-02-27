#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 17:28:18 2026

@author: poddar
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StressSpec:
    """
    Stress overlay parameters.

    vol_multiplier:
        scalar or vector of length N. Multiplies vol (not variance).
        Example: 1.5 => vol up 50%, variance up 125%.

    corr_blend:
        kappa in [0,1]. Blends current corr toward target corr.
    """
    vol_multiplier: float = 1.5
    corr_blend: float = 0.3
    jitter: float = 1e-6


class StressOverlay:
    """
    Applies stress overlays to variance forecasts and correlation matrix.
    """

    def __init__(self, spec: Optional[StressSpec] = None):
        self.spec = spec or StressSpec()

    def apply(
        self,
        var_forecast: np.ndarray,
        corr: np.ndarray,
        target_corr: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns stressed (var_forecast, corr).

        Parameters
        ----------
        var_forecast : (N,) variance vector
        corr : (N,N) correlation matrix
        target_corr : (N,N) optional target correlation matrix for blending
        """
        var_forecast = np.asarray(var_forecast, dtype=float)
        corr = np.asarray(corr, dtype=float)

        if var_forecast.ndim != 1:
            raise ValueError("var_forecast must be 1D.")
        N = var_forecast.shape[0]
        if corr.shape != (N, N):
            raise ValueError("corr shape mismatch.")
        if np.any(var_forecast <= 0):
            raise ValueError("var_forecast must be positive.")

        # Vol shock: variance scales by multiplier^2
        m = self._as_multiplier(self.spec.vol_multiplier, N)
        var_stressed = var_forecast * (m ** 2)

        # Correlation stress: blend toward target_corr (or toward higher average corr)
        kappa = float(self.spec.corr_blend)
        if not (0.0 <= kappa <= 1.0):
            raise ValueError("corr_blend must be in [0,1].")

        if target_corr is None:
            # default target: increase off-diagonal toward 0.8 in magnitude (simple crisis proxy)
            target_corr = self._default_target_corr(corr, level=0.8)

        target_corr = np.asarray(target_corr, dtype=float)
        if target_corr.shape != (N, N):
            raise ValueError("target_corr shape mismatch.")

        corr_stressed = (1.0 - kappa) * corr + kappa * target_corr
        corr_stressed = self._make_psd(corr_stressed, jitter=self.spec.jitter)
        corr_stressed = np.clip(corr_stressed, -0.999, 0.999)
        np.fill_diagonal(corr_stressed, 1.0)

        return var_stressed, corr_stressed

    @staticmethod
    def _as_multiplier(x: float | np.ndarray, N: int) -> np.ndarray:
        m = np.asarray(x, dtype=float)
        if m.ndim == 0:
            return np.full(N, float(m))
        if m.shape != (N,):
            raise ValueError("vol_multiplier vector must have shape (N,).")
        return m

    @staticmethod
    def _default_target_corr(corr: np.ndarray, level: float = 0.8) -> np.ndarray:
        N = corr.shape[0]
        out = corr.copy()
        for i in range(N):
            for j in range(N):
                if i == j:
                    out[i, j] = 1.0
                else:
                    # push toward +level preserving sign (simple)
                    out[i, j] = np.sign(out[i, j]) * level
        return out

    @staticmethod
    def _make_psd(A: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
        A = 0.5 * (A + A.T)
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.clip(eigvals, jitter, np.inf)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T