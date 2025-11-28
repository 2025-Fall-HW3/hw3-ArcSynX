"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        # Only allocate to sector ETFs (exclude SPY)
        sector_cols = assets

        # Momentum and volatility window 
        mom_window = max(self.lookback, 60)   # At least 60 days for momentum
        vol_window = max(self.lookback // 2, 20)  # Shorter window for volatility estimate
        top_k = 3                             # Select top 3 by momentum each time

        # Use log-return as a basis for momentum 
        sector_ret = self.returns[sector_cols]
        log_ret = np.log1p(sector_ret)

        # Rolling momentum score: sum of log(1+r)
        rolling_mom = (
            log_ret.rolling(window=mom_window, min_periods=mom_window)
                   .sum()
        )

        # Rolling volatility: standard deviation of original daily return
        rolling_vol = (
            sector_ret.rolling(window=vol_window, min_periods=vol_window)
                      .std()
        )

        for idx, date in enumerate(self.price.index):
            # Skip if not enough data at the beginning, fill by ffill later
            if idx < max(mom_window, vol_window):
                continue

            mom_today = rolling_mom.loc[date]
            vol_today = rolling_vol.loc[date]

            # Skip if today's values are all invalid
            if mom_today.isna().all() or vol_today.isna().all():
                continue

            # Only keep sectors with positive momentum, use all if none
            positive = mom_today[mom_today > 0]
            if positive.empty:
                ranked = mom_today.sort_values(ascending=False)
            else:
                ranked = positive.sort_values(ascending=False)

            # Take top_k sectors by momentum
            selected = ranked.head(top_k).index

            # Get the short-term volatility of the selected sectors
            vol_sel = vol_today.loc[selected].replace(0.0, np.nan)

            # Inverse-volatility weighted score
            inv_vol = 1.0 / vol_sel
            inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            score_sum = inv_vol.sum()
            if score_sum > 0:
                weights_sel = inv_vol / score_sum
            else:
                # fallback: if volatility is problematic, use equal weight
                weights_sel = pd.Series(1.0 / len(selected), index=selected)

            # Build the full daily weight vector: only invest in selected sectors
            today_w = pd.Series(0.0, index=self.price.columns)
            today_w.loc[selected] = weights_sel

            # Set SPY's weight to zero
            today_w.loc[self.exclude] = 0.0

            # Write today's weights
            self.portfolio_weights.loc[date, :] = today_w

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)


    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
