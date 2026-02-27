#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:15:49 2026

@author: poddar
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass(frozen=True)
class GARCHSpec:
    """
    Volatility model specification.
    """
    p: int = 1                  # ARCH term
    o: int = 1                  # asymmetry term (o=1 => GJR)
    q: int = 1                  # GARCH term
    dist: str = "t"             # "t" for Student-t, "normal" for Gaussian
    mean: str = "Zero"          # "Zero" is common for daily VaR; can switch to "Constant"


class VolatilityModel:
    """
    Base interface for volatility models.
    """

    def fit(self, series: pd.Series):
        raise NotImplementedError

    def conditional_volatility(self) -> pd.Series:
        raise NotImplementedError

    def standardized_residuals(self) -> pd.Series:
        raise NotImplementedError

    def forecast_one_step_variance(self) -> float:
        raise NotImplementedError


class GJRGARCHModel(VolatilityModel):
    """
    Wrapper around arch.arch_model for GJR-GARCH with Student-t innovations.
    Designed for daily returns.
    """

    def __init__(self, spec: Optional[GARCHSpec] = None, scale: float = 100.0):
        """
        Parameters
        ----------
        spec : GARCHSpec
            Model specification
        scale : float
            Returns are multiplied by `scale` for numerical stability during MLE.
            Internally, outputs are scaled back to original return units.
        """
        self.spec = spec or GARCHSpec()
        self.scale = float(scale)

        self._res = None
        self._index = None

    def fit(self, series: pd.Series) -> "GJRGARCHModel":
        """
        Fit model to a single return series.
        """
        self._validate_series(series)
        self._index = series.index

        y = (series * self.scale).astype(float)

        am = arch_model(
            y,
            mean=self.spec.mean,
            vol="GARCH",
            p=self.spec.p,
            o=self.spec.o,
            q=self.spec.q,
            dist=self.spec.dist,
            rescale=False,
        )
        self._res = am.fit(disp="off")
        return self

    def summary_text(self) -> str:
        if self._res is None:
            raise ValueError("Model not fitted.")
        return str(self._res.summary())

    def params(self) -> pd.Series:
        if self._res is None:
            raise ValueError("Model not fitted.")
        return self._res.params.copy()

    def conditional_volatility(self) -> pd.Series:
        """
        In-sample conditional volatility sigma_t (in original return units).
        """
        if self._res is None:
            raise ValueError("Model not fitted.")
        sigma = (self._res.conditional_volatility / self.scale)
        sigma.index = self._index[-len(sigma):]
        return sigma

    def residuals(self) -> pd.Series:
        """
        In-sample residuals epsilon_t (in original return units).
        """
        if self._res is None:
            raise ValueError("Model not fitted.")
        eps = (self._res.resid / self.scale)
        eps.index = self._index[-len(eps):]
        return eps

    def standardized_residuals(self) -> pd.Series:
        """
        z_t = epsilon_t / sigma_t
        """
        eps = self.residuals()
        sig = self.conditional_volatility()
        z = eps / sig
        z.name = "std_resid"
        return z.replace([np.inf, -np.inf], np.nan).dropna()

    def forecast_one_step_variance(self) -> float:
        """
        One-step-ahead forecast of variance (in original return units^2).

        Returns
        -------
        float
            sigma_{t+1}^2 forecast
        """
        if self._res is None:
            raise ValueError("Model not fitted.")

        # arch forecast returns variance in scaled units^2
        f = self._res.forecast(horizon=1, reindex=False)
        var_scaled = float(f.variance.values[-1, 0])  # last forecast, 1-step
        var = var_scaled / (self.scale ** 2)
        return var

    def forecast_one_step_vol(self) -> float:
        """
        One-step-ahead forecast of volatility (in original return units).
        """
        return float(np.sqrt(self.forecast_one_step_variance()))

    @staticmethod
    def _validate_series(series: pd.Series) -> None:
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series.")
        if series.isna().any():
            raise ValueError("Series contains NaNs.")
        if not series.index.is_monotonic_increasing:
            raise ValueError("Series index must be sorted increasing by date.")
        if series.shape[0] < 200:
            raise ValueError("Too few observations for stable GARCH estimation (need >= 200).")


class MultiAssetVolatilityFitter:
    """
    Fits a univariate volatility model per asset and exposes aligned outputs.
    """

    def __init__(self, model: VolatilityModel):
        self.model_template = model
        self.models: Dict[str, VolatilityModel] = {}

    def fit(self, returns: pd.DataFrame) -> "MultiAssetVolatilityFitter":
        self._validate_returns(returns)
        self.models = {}

        for col in returns.columns:
            # create a fresh model instance per asset
            m = GJRGARCHModel(spec=getattr(self.model_template, "spec", None),
                              scale=getattr(self.model_template, "scale", 100.0))
            m.fit(returns[col])
            self.models[col] = m

        return self

    def conditional_vol_matrix(self) -> pd.DataFrame:
        vols = {k: v.conditional_volatility() for k, v in self.models.items()}
        df = pd.DataFrame(vols).dropna(how="any")
        return df

    def standardized_residual_matrix(self) -> pd.DataFrame:
        zs = {k: v.standardized_residuals() for k, v in self.models.items()}
        df = pd.DataFrame(zs).dropna(how="any")
        return df

    def forecast_one_step_variances(self) -> pd.Series:
        """
        Returns vector of sigma_{i,t+1}^2 forecasts indexed by asset.
        """
        out = {k: v.forecast_one_step_variance() for k, v in self.models.items()}
        return pd.Series(out, name="var_forecast")

    @staticmethod
    def _validate_returns(returns: pd.DataFrame) -> None:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        if returns.shape[1] < 2:
            raise ValueError("Need at least 2 assets for multi-asset VaR.")
        if returns.isna().any().any():
            raise ValueError("returns contains NaNs.")
        if not returns.index.is_monotonic_increasing:
            raise ValueError("returns index must be sorted increasing by date.")