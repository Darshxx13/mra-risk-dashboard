"""
VaR and Expected Shortfall three ways:
  1. Historical simulation  - no distribution assumption, replay actual returns
  2. Parametric (variance-covariance) - assumes normal distribution
  3. Monte Carlo simulation - simulate future paths from fitted distribution

Also includes
  - Rolling VaR for time series monitoring
  - Backtesting with Basel traffic light test
  - Expected Shortfall (CVaR) - the FRTB replacement for VaR
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── VaR: Historical Simulation 

def var_historical(returns: pd.Series, confidence: float = 0.99) -> float:
    return float(np.percentile(returns, (1 - confidence) * 100))


# ── VaR: Parametric (Normal)

def var_parametric(returns: pd.Series, confidence: float = 0.99) -> float:
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    return float(mu + z * sigma)


# ── VaR: Monte Carlo Simulation 
def var_montecarlo(
    returns: pd.Series,
    confidence: float = 0.99,
    n_simulations: int = 10000,
    horizon: int = 1,
    seed: int = 42,
) -> float:
    
    rng = np.random.default_rng(seed)
    mu = returns.mean()
    sigma = returns.std()

    simulated = rng.normal(loc=mu, scale=sigma, size=(n_simulations, horizon))
    path_returns = simulated.sum(axis=1) 

    return float(np.percentile(path_returns, (1 - confidence) * 100))


# ── Expected Shortfall (CVaR) 

def expected_shortfall(returns: pd.Series, confidence: float = 0.99) -> float:
    var = var_historical(returns, confidence)
    tail_losses = returns[returns <= var]
    if len(tail_losses) == 0:
        return var
    return float(tail_losses.mean())


# ── Rolling VaR (time series monitoring)

def rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.99,
    method: str = "historical",
) -> pd.Series:
    def _var(window_returns):
        if method == "historical":
            return var_historical(window_returns, confidence)
        elif method == "parametric":
            return var_parametric(window_returns, confidence)
        elif method == "montecarlo":
            return var_montecarlo(window_returns, confidence)
        else:
            raise ValueError(f"Unknown method: {method}")

    rolled = returns.rolling(window=window).apply(_var, raw=False)
    rolled.name = f"VaR_{method}_{int(confidence*100)}"
    return rolled


# ── Backtesting

def backtest_var(
    returns: pd.Series,
    var_series: pd.Series,
    window: int = 250,
) -> pd.DataFrame:
    aligned = pd.DataFrame({
        "return": returns,
        "var":    var_series,
    }).dropna()

    aligned["exception"] = aligned["return"] < aligned["var"]
    aligned["rolling_exceptions"] = (
        aligned["exception"]
        .rolling(window=window, min_periods=window)
        .sum()
        .astype("Int64")
    )

    def _zone(n):
        if pd.isna(n):
            return "N/A"
        if n <= 4:
            return "Green"
        elif n <= 9:
            return "Yellow"
        else:
            return "Red"

    aligned["zone"] = aligned["rolling_exceptions"].apply(_zone)
    return aligned


# ── Summary table

def risk_summary(returns: pd.Series, confidence: float = 0.99) -> pd.DataFrame:

    vh = var_historical(returns, confidence)
    vp = var_parametric(returns, confidence)
    vm = var_montecarlo(returns, confidence)
    es = expected_shortfall(returns, confidence)

    rows = [
        {"Method": "Historical Simulation", "VaR":  vh, "Notes": "No distribution assumption"},
        {"Method": "Parametric (Normal)",   "VaR":  vp, "Notes": "Assumes normal returns"},
        {"Method": "Monte Carlo",           "VaR":  vm, "Notes": "Simulated paths (10k scenarios)"},
        {"Method": "Expected Shortfall",    "VaR":  es, "Notes": "Avg loss beyond VaR (FRTB metric)"},
    ]
    df = pd.DataFrame(rows)
    df["VaR (%)"] = (df["VaR"] * 100).round(3)
    return df[["Method", "VaR (%)", "Notes"]]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    test_returns = pd.Series(rng.normal(-0.0002, 0.012, 1000))

    print("=== Risk Summary ===")
    print(risk_summary(test_returns))

    bt = backtest_var(test_returns, rolling_var(test_returns, window=252))
    total_exceptions = bt["exception"].sum()
    print(f"\nBacktest: {total_exceptions} exceptions over {len(bt)} days")
    print(f"Exception rate: {total_exceptions/len(bt)*100:.2f}%")
