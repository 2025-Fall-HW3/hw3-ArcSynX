"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings
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

start = "2019-01-01"
end = "2024-04-01"

# Initialize df and df_returns
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust = False)
    df[asset] = raw['Adj Close']

df_returns = df.pct_change().fillna(0)


"""
Problem 1: 

Implement an equal weighting strategy as dataframe "eqw". Please do "not" include SPY.
"""


class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 1 Below
        """
        # Assign a fixed equal weight to all assets except the one in exclude
        n_assets = len(assets)
        # The weight of each asset = 1 / n_assets
        equal_weights = np.ones(n_assets, dtype=float) / n_assets
        self.portfolio_weights.loc[:, assets] = equal_weights
        

        """
        TODO: Complete Task 1 Above
        """
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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


"""
Problem 2:

Implement a risk parity strategy as dataframe "rp". Please do "not" include SPY.
"""


class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 2 Below
        """
        # Get the returns of all assets except SPY 
        sub_ret = df_returns[assets]

        # Calculate risk parity weights each day, starting from where we have enough lookback data
        for i in range(self.lookback + 1, len(df)):
            # Take the window of past lookback days of returns (excluding today)
            window = sub_ret.iloc[i - self.lookback : i]

            # Use numpy to manually calculate the standard deviation of each asset (do not use pandas' std())
            # Steps: first calculate mean, then mean squared difference, then take sqrt
            mean_vec = window.mean(axis=0).to_numpy()            # Mean return of each asset
            diff_sq = (window.to_numpy() - mean_vec) ** 2        # (r - μ)^2
            var_vec = diff_sq.mean(axis=0)                       # Variance of each asset
            sigma = np.sqrt(var_vec)                             # Volatility of each asset

            # Skip abnormal numerical situations (avoid division by zero)
            sigma_adj = np.maximum(sigma, 1e-12)

            # Risk parity score: The lower the volatility, the higher the score
            score = 1.0 / sigma_adj

            total = score.sum()
            if total <= 0:
                # Should not happen in theory, but for insurance use equal weight
                w = np.full(len(assets), 1.0 / len(assets))
            else:
                # Normalize to weights so the sum is 1
                w = score * (1.0 / total)

            # Save the weights for day i (only fill columns not including SPY)
            self.portfolio_weights.loc[df.index[i], assets] = w


        """
        TODO: Complete Task 2 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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


"""
Problem 3:

Implement a Markowitz strategy as dataframe "mv". Please do "not" include SPY.
"""


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(self.lookback + 1, len(df)):
            R_n = df_returns.copy()[assets].iloc[i - self.lookback : i]
            self.portfolio_weights.loc[df.index[i], assets] = self.mv_opt(
                R_n, self.gamma
            )

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        # R_n: the past lookback days' returns (DataFrame with columns = assets)
        # gamma: risk aversion coefficient

        # Estimate μ and Σ
        mu = R_n.mean().values           # shape: (n,)
        Sigma = R_n.cov().values         # shape: (n, n)
        n = len(mu)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                """
                TODO: Complete Task 3 Below
                """

                # Decision variables: asset weights w_i, long-only: w_i >= 0
                # Set ub=1 to avoid excessive weights 
                w = model.addMVar(shape=n, lb=0.0, ub=1.0, name="w")

                # Budget constraint: weights sum to 1 (fully invested)
                model.addConstr(w.sum() == 1.0, name="budget")

                # Linear part: w^T μ
                ret_term = mu @ w

                # Quadratic risk term: w^T Σ w
                risk_term = w @ Sigma @ w

                # Avoid the case where gamma is None
                if gamma is None:
                    gamma = 0.0

                # Objective: max w^T μ − (γ/2) w^T Σ w
                model.setObjective(ret_term - 0.5 * float(gamma) * risk_term,
                                   gp.GRB.MAXIMIZE)

                """
                TODO: Complete Task 3 Above
                """
                model.optimize()

                if model.status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                     
                    solution = model.getVarByName("w[0]").X  # This way will cause an error, use w.X
                    solution = w.X.tolist()
                else:
                    # If the solver fails, fallback to equal weight to avoid interruption
                    solution = [1.0 / n] * n

        return solution


    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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
    from grader import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 1"
    )
    """
    NOTE: For Assignment Judge
    """
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

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader.py
    judge.run_grading(args)
