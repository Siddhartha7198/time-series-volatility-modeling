#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 14:35:34 2026
Download adjusted prices, Align calendars across assets
Return a clean DataFrame ready for return computation
@author: poddar
"""

import pandas as pd
import yfinance as yf
from typing import List, Optional


class MarketDataLoader:
    """
    Handles:
    - Downloading adjusted close prices
    - Aligning multi-asset calendar
    - Basic data validation
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        self._prices = None

    def download_data(self) -> pd.DataFrame:
        """
        Downloads adjusted close prices using yfinance.
        Returns a clean price DataFrame indexed by date.
        """

        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )

        if "Close" not in data.columns:
            raise ValueError("Close column not found in downloaded data.")

        prices = data["Close"].copy()

        # Ensure DataFrame shape even if single ticker
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        prices = prices.sort_index()

        self._prices = prices
        return self._prices

    def clean_data(self) -> pd.DataFrame:
        """
        Cleans downloaded price data:
        - Drops rows with any missing values (calendar alignment)
        - Removes duplicate index values
        - Ensures monotonic date index
        """

        if self._prices is None:
            raise ValueError("Call download_data() before clean_data().")

        prices = self._prices.copy()

        # Remove duplicate dates
        prices = prices[~prices.index.duplicated(keep="first")]

        # Sort index
        prices = prices.sort_index()

        # Drop rows with missing values (intersection calendar)
        prices = prices.dropna(how="any")

        # Final validation
        if prices.isna().any().any():
            raise ValueError("NaNs remain after cleaning.")

        if not prices.index.is_monotonic_increasing:
            raise ValueError("Date index is not sorted.")

        self._prices = prices
        return self._prices

    def get_prices(self) -> pd.DataFrame:
        """
        Returns cleaned prices.
        """
        if self._prices is None:
            raise ValueError("Data not loaded. Call download_data() first.")
        return self._prices.copy()