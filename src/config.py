#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 14:27:55 2026
tickers, weights, windows, confidence, etc.
@author: poddar
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass(frozen=True)
class ProjectConfig:
    # Universe
    tickers: List[str]

    # Data
    start_date: str = "2014-01-01"
    end_date: Optional[str] = None  # None => most recent

    # Portfolio
    weights: Optional[np.ndarray] = None  # if None => equal weight

    # VaR
    alpha: float = 0.99
    horizon_days: int = 1

    # Estimation / backtest
    train_window: int = 750          # ~3 years of daily data
    corr_window: int = 252           # 1y window for EWMA corr inputs
    ewma_lambda: float = 0.97

    # Monte Carlo
    n_sims: int = 20000
    random_seed: int = 42

    def get_weights(self) -> np.ndarray:
        if self.weights is None:
            w = np.ones(len(self.tickers), dtype=float)
            return w / w.sum()
        w = np.asarray(self.weights, dtype=float)
        if w.shape != (len(self.tickers),):
            raise ValueError("weights must have shape (n_assets,)")
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("weights must sum to 1.0")
        return w