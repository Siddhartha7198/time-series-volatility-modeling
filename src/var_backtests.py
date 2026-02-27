#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 17:19:32 2026

@author: poddar
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2


@dataclass(frozen=True)
class VaRTestSpec:
    alpha: float = 0.99


class VaRBacktests:
    """
    Implements:
    - Kupiec Unconditional Coverage test
    - Christoffersen Independence test
    - Christoffersen Conditional Coverage test (UC + IND)
    """

    def __init__(self, spec: VaRTestSpec):
        self.spec = spec
        if not (0.0 < self.spec.alpha < 1.0):
            raise ValueError("alpha must be in (0,1).")

    @staticmethod
    def _safe_log(x: float) -> float:
        return float(np.log(max(x, 1e-16)))

    def kupiec_uc(self, breaches: pd.Series) -> Dict[str, float]:
        """
        Kupiec (1995) Unconditional Coverage test.

        Inputs
        ------
        breaches : pd.Series of 0/1

        Returns
        -------
        dict with LR_uc, p_value_uc, n, x, phat, p
        """
        b = self._to_binary(breaches)
        n = int(b.size)
        x = int(b.sum())
        p = 1.0 - float(self.spec.alpha)
        phat = x / n if n > 0 else 0.0

        # Likelihood ratio:
        # LR = -2 [ log L(p) - log L(phat) ]
        # log L(q) = x log q + (n-x) log(1-q)
        logL_p = x * self._safe_log(p) + (n - x) * self._safe_log(1.0 - p)
        logL_phat = x * self._safe_log(phat) + (n - x) * self._safe_log(1.0 - phat)

        LR_uc = -2.0 * (logL_p - logL_phat)
        pval = 1.0 - chi2.cdf(LR_uc, df=1)

        return {
            "n": float(n),
            "x": float(x),
            "p": float(p),
            "phat": float(phat),
            "LR_uc": float(LR_uc),
            "p_value_uc": float(pval),
        }

    def christoffersen_ind(self, breaches: pd.Series) -> Dict[str, float]:
        """
        Christoffersen (1998) Independence test.

        Builds transition counts:
        n00, n01, n10, n11 based on breaches at t-1 -> t.
        """
        b = self._to_binary(breaches)

        if b.size < 2:
            raise ValueError("Need at least 2 observations for independence test.")

        b_prev = b[:-1]
        b_curr = b[1:]

        n00 = int(np.sum((b_prev == 0) & (b_curr == 0)))
        n01 = int(np.sum((b_prev == 0) & (b_curr == 1)))
        n10 = int(np.sum((b_prev == 1) & (b_curr == 0)))
        n11 = int(np.sum((b_prev == 1) & (b_curr == 1)))

        # Transition probabilities
        # pi01 = P(I_t=1 | I_{t-1}=0), pi11 = P(I_t=1 | I_{t-1}=1)
        denom0 = n00 + n01
        denom1 = n10 + n11
        pi01 = n01 / denom0 if denom0 > 0 else 0.0
        pi11 = n11 / denom1 if denom1 > 0 else 0.0

        # Unconditional prob
        x = int(b.sum())
        n = int(b.size)
        pi = x / n if n > 0 else 0.0

        # log-likelihood under independence (single pi)
        logL_ind = (n01 + n11) * self._safe_log(pi) + (n00 + n10) * self._safe_log(1.0 - pi)

        # log-likelihood under Markov (pi01, pi11)
        logL_markov = (
            n01 * self._safe_log(pi01) + n00 * self._safe_log(1.0 - pi01)
            + n11 * self._safe_log(pi11) + n10 * self._safe_log(1.0 - pi11)
        )

        LR_ind = -2.0 * (logL_ind - logL_markov)
        pval = 1.0 - chi2.cdf(LR_ind, df=1)

        return {
            "n00": float(n00), "n01": float(n01), "n10": float(n10), "n11": float(n11),
            "pi": float(pi), "pi01": float(pi01), "pi11": float(pi11),
            "LR_ind": float(LR_ind),
            "p_value_ind": float(pval),
        }

    def christoffersen_cc(self, breaches: pd.Series) -> Dict[str, float]:
        """
        Conditional Coverage = UC + IND (df=2).
        """
        uc = self.kupiec_uc(breaches)
        ind = self.christoffersen_ind(breaches)

        LR_cc = float(uc["LR_uc"] + ind["LR_ind"])
        pval = 1.0 - chi2.cdf(LR_cc, df=2)

        return {
            "LR_cc": float(LR_cc),
            "p_value_cc": float(pval),
            **{f"uc_{k}": v for k, v in uc.items()},
            **{f"ind_{k}": v for k, v in ind.items()},
        }

    @staticmethod
    def _to_binary(breaches: pd.Series) -> np.ndarray:
        if isinstance(breaches, pd.Series):
            b = breaches.values
        else:
            b = np.asarray(breaches)
        b = b.astype(int)
        if not np.all((b == 0) | (b == 1)):
            raise ValueError("breaches must be binary 0/1.")
        return b