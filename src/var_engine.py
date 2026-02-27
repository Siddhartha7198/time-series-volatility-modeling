#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:00:53 2026

@author: poddar
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VaRSpec:
    """
    VaR/ES engine specification.
    """
    alpha: float = 0.99
    n_sims: int = 20000
    random_seed: int = 42

    # Simulation distribution:
    # - "gaussian": r ~ N(0, Sigma)
    # - "student_t": multivariate t via scale-mixture of normals
    sim_dist: str = "gaussian"

    # degrees of freedom for student_t (must be > 2 for finite variance)
    nu: float = 8.0


class VaREngine:
    """
    Computes 1-day portfolio VaR and ES from conditional variance forecasts and correlation matrix.
    """

    def __init__(self, spec: Optional[VaRSpec] = None):
        self.spec = spec or VaRSpec()

        if not (0.0 < self.spec.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if self.spec.n_sims < 1000:
            raise ValueError("n_sims too small; use at least 1000.")
        if self.spec.sim_dist not in {"gaussian", "student_t"}:
            raise ValueError("sim_dist must be one of {'gaussian', 'student_t'}")

        if self.spec.sim_dist == "student_t":
            if not (self.spec.nu > 2.0):
                raise ValueError("For student_t simulation, nu must be > 2 (finite variance).")

        self._rng = np.random.default_rng(self.spec.random_seed)

    @staticmethod
    def build_covariance(var_forecast: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """
        Build covariance matrix Sigma = D * R * D.

        Parameters
        ----------
        var_forecast : np.ndarray
            Vector of one-step-ahead variances (N,)
        corr : np.ndarray
            Correlation matrix (N, N)

        Returns
        -------
        np.ndarray
            Covariance matrix (N, N)
        """
        var_forecast = np.asarray(var_forecast, dtype=float)
        corr = np.asarray(corr, dtype=float)

        if var_forecast.ndim != 1:
            raise ValueError("var_forecast must be a 1D array.")
        N = var_forecast.shape[0]
        if corr.shape != (N, N):
            raise ValueError(f"corr must have shape ({N},{N}).")
        if np.any(var_forecast <= 0.0) or not np.isfinite(var_forecast).all():
            raise ValueError("var_forecast must be positive and finite.")

        sigma = np.sqrt(var_forecast)
        D = np.diag(sigma)
        Sigma = D @ corr @ D
        Sigma = 0.5 * (Sigma + Sigma.T)  # enforce symmetry
        return Sigma

    def simulate_portfolio_returns(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate portfolio returns r_p = w' r.

        - gaussian: r ~ N(0, cov)
        - student_t: r ~ multivariate t via scale-mixture (elliptical t)

        Returns
        -------
        np.ndarray
            Simulated portfolio returns (n_sims,)
        """
        w = np.asarray(weights, dtype=float)
        cov = np.asarray(cov, dtype=float)

        if w.ndim != 1:
            raise ValueError("weights must be 1D.")
        N = w.shape[0]
        if cov.shape != (N, N):
            raise ValueError(f"cov must have shape ({N},{N}).")
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("weights must sum to 1.0.")

        if self.spec.sim_dist == "gaussian":
            R = self._simulate_mv_normal(cov)
        else:
            R = self._simulate_mv_student_t(cov, nu=self.spec.nu)

        rp = R @ w
        return rp

    def var_from_simulation(self, portfolio_returns: np.ndarray) -> float:
        """
        VaR_{alpha} = - quantile_{1-alpha}(r_p)
        """
        x = np.asarray(portfolio_returns, dtype=float)
        if x.ndim != 1:
            raise ValueError("portfolio_returns must be 1D.")
        q = np.quantile(x, 1.0 - self.spec.alpha)
        return float(-q)

    def es_from_simulation(self, portfolio_returns: np.ndarray) -> float:
        """
        Expected Shortfall (ES) at level alpha from simulated returns.

        Let q = Q_{1-alpha}(r_p). Then:
        $
        ES_{\\alpha} = - E[r_p | r_p \\le q]
        $

        Returns
        -------
        float
            ES (positive number, loss scale)
        """
        x = np.asarray(portfolio_returns, dtype=float)
        if x.ndim != 1:
            raise ValueError("portfolio_returns must be 1D.")

        q = np.quantile(x, 1.0 - self.spec.alpha)
        tail = x[x <= q]
        if tail.size == 0:
            # with finite sims and very high alpha, tail can be empty; fallback to VaR
            return float(-q)

        return float(-tail.mean())

    def compute_var_es(
        self,
        weights: np.ndarray,
        var_forecast: np.ndarray,
        corr: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Full pipeline: build covariance -> simulate -> compute VaR and ES.

        Returns
        -------
        (VaR, ES, simulated_portfolio_returns)
        """
        cov = self.build_covariance(var_forecast, corr)
        rp = self.simulate_portfolio_returns(weights, cov)
        var = self.var_from_simulation(rp)
        es = self.es_from_simulation(rp)
        return var, es, rp

    # ---------- internal simulation helpers ----------

    def _simulate_mv_normal(self, cov: np.ndarray) -> np.ndarray:
        """
        Draw R ~ N(0, cov) with shape (n_sims, N).
        """
        L = self._safe_cholesky(cov)
        Z = self._rng.standard_normal(size=(self.spec.n_sims, cov.shape[0]))
        return Z @ L.T

    def _simulate_mv_student_t(self, cov: np.ndarray, nu: float) -> np.ndarray:
        """
        Elliptical multivariate Student-t via scale mixture of normals:

        $
        r = y / sqrt(u/nu)
        $
        where:
        $
        y ~ N(0, cov),   u ~ Chi^2_{nu}
        $

        We also rescale so that Cov(r) matches cov (finite variance case nu>2).
        For elliptical t, Cov(r) = nu/(nu-2) * cov, so multiply by sqrt((nu-2)/nu)
        to match cov.
        """
        N = cov.shape[0]
        L = self._safe_cholesky(cov)

        Z = self._rng.standard_normal(size=(self.spec.n_sims, N))
        y = Z @ L.T  # N(0, cov)

        u = self._rng.chisquare(df=nu, size=self.spec.n_sims)  # shape (n_sims,)
        scale = np.sqrt(u / nu).reshape(-1, 1)

        r = y / scale

        # variance normalization to ensure Cov(r) == cov (for nu>2)
        norm = np.sqrt((nu - 2.0) / nu)
        r = norm * r
        return r



    @staticmethod
    def _safe_cholesky(A: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
        """
        Robust Cholesky for near-PSD matrices.
        Cholesky requires strictly PD. Stress overlays can easily push covariance to PSD.

        Strategy:
        1) Try standard cholesky.
        2) Add diagonal jitter adaptively (x10 each attempt).
        3) If still failing, project to SPD via eigenvalue flooring and retry.
        """
        A = np.asarray(A, dtype=float)
        A = 0.5 * (A + A.T)

        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            pass

        # Adaptive diagonal jitter
        diag_mean = float(np.mean(np.diag(A)))
        # scale jitter to matrix magnitude; avoid tiny jitter when cov entries are ~1e-4 etc.
        base = max(jitter, 1e-12 * max(1.0, diag_mean))

        for k in range(10):
            eps = base * (10.0 ** k)
            try:
                return np.linalg.cholesky(A + eps * np.eye(A.shape[0]))
            except np.linalg.LinAlgError:
                continue

        # eigenvalue floor to force SPD, then cholesky
        eigvals, eigvecs = np.linalg.eigh(A)
        floor = max(base, 1e-10 * max(1.0, np.max(eigvals)))
        eigvals = np.clip(eigvals, floor, np.inf)
        A_spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        A_spd = 0.5 * (A_spd + A_spd.T)
        return np.linalg.cholesky(A_spd)