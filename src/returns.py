#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 14:55:26 2026

@author: poddar
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnConfig:
    """
    Configuration for return computation.
    """
    log_returns: bool = True
    demean: bool = False  


class ReturnCalculator:
    """
    Computes asset and portfolio returns from a price DataFrame.
    Expects:
    - prices indexed by datetime-like index
    - columns as tickers/assets
    """

    def __init__(self, config: Optional[ReturnConfig] = None):
        self.config = config or ReturnConfig()

    def compute_asset_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 1-day asset returns.

        Returns
        -------
        pd.DataFrame
            Returns indexed like prices (minus first row), columns same as prices.
        """
        self._validate_prices(prices)

        if self.config.log_returns:
            rets = np.log(prices).diff()
        else:
            rets = prices.pct_change()

        rets = rets.dropna(how="any")

        if self.config.demean:
            rets = rets - rets.mean(axis=0)

        return rets

    def compute_portfolio_returns(
        self,
        asset_returns: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.Series:
        """
        Compute portfolio returns r_p,t = w' r_t.

        Parameters
        ----------
        asset_returns : pd.DataFrame
            Asset return matrix (T x N)
        weights : np.ndarray
            Portfolio weights (N,)

        Returns
        -------
        pd.Series
            Portfolio return series indexed by date.
        """
        self._validate_returns(asset_returns)
        w = self._validate_weights(weights, asset_returns.shape[1])
        rp = asset_returns.values @ w
        return pd.Series(rp, index=asset_returns.index, name="portfolio_return")

    def train_test_split_by_index(
        self,
        returns: pd.DataFrame,
        train_window: int,
        t: int
    ) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """
        Convenience helper for rolling-window workflows.

        Parameters
        ----------
        returns : pd.DataFrame
            Return matrix (T x N)
        train_window : int
            Number of observations used for training
        t : int
            End index (exclusive) for the training window

        Returns
        -------
        (train_returns, forecast_date)
            train_returns: returns[t-train_window:t]
            forecast_date: returns.index[t] (the date for which we forecast next-day risk)
        """
        self._validate_returns(returns)
        if train_window <= 10:
            raise ValueError("train_window too small for volatility modeling.")
        if t <= train_window:
            raise ValueError("t must be greater than train_window.")
        if t >= len(returns):
            raise ValueError("t must be < len(returns).")

        train = returns.iloc[t - train_window: t].copy()
        forecast_date = returns.index[t]
        return train, forecast_date

    @staticmethod
    def _validate_prices(prices: pd.DataFrame) -> None:
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame.")
        if prices.shape[1] < 2:
            # multi-asset project; enforce >=2 assets
            raise ValueError("prices must have at least 2 columns (assets).")
        if prices.isna().any().any():
            raise ValueError("prices contains NaNs. Clean/align prices before computing returns.")
        if not prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be sorted increasing by date.")

    @staticmethod
    def _validate_returns(returns: pd.DataFrame) -> None:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        if returns.shape[1] < 2:
            raise ValueError("returns must have at least 2 columns (assets).")
        if returns.isna().any().any():
            raise ValueError("returns contains NaNs.")
        if not returns.index.is_monotonic_increasing:
            raise ValueError("returns index must be sorted increasing by date.")

    @staticmethod
    def _validate_weights(weights: np.ndarray, n_assets: int) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_assets,):
            raise ValueError(f"weights must have shape ({n_assets},), got {w.shape}.")
        if not np.isfinite(w).all():
            raise ValueError("weights contains non-finite values.")
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("weights must sum to 1.0.")
        return w