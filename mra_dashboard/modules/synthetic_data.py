"""
realistic synthetic market data when yfinance is unavailable.
useful for stress testing edge cases.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta


ASSET_PARAMS = {
    "SPY": {"mu_annual": 0.10,  "vol_annual": 0.18, "label": "Equities (S&P 500)"},
    "TLT": {"mu_annual": 0.01,  "vol_annual": 0.14, "label": "Rates (20Y Treasury)"},
    "UUP": {"mu_annual": 0.01,  "vol_annual": 0.06, "label": "FX (USD Index)"},
    "GLD": {"mu_annual": 0.07,  "vol_annual": 0.13, "label": "Commodities (Gold)"},
    "HYG": {"mu_annual": 0.04,  "vol_annual": 0.08, "label": "Credit (High Yield)"},
}

CORRELATION = np.array([
    # SPY   TLT    UUP    GLD    HYG
    [ 1.00, -0.30, -0.05, -0.05,  0.75],   # SPY
    [-0.30,  1.00, -0.20,  0.20, -0.20],   # TLT
    [-0.05, -0.20,  1.00, -0.30, -0.05],   # UUP
    [-0.05,  0.20, -0.30,  1.00, -0.05],   # GLD
    [ 0.75, -0.20, -0.05, -0.05,  1.00],   # HYG
])

TICKERS_SYNTH = list(ASSET_PARAMS.keys())


def generate_synthetic_prices(
    start: str = "2018-01-01",
    end: str = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate correlated synthetic price series using geometric Brownian motion.
    Returns a DataFrame of daily prices indexed by trading dates.
    """
    rng = np.random.default_rng(seed)

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end) if end else pd.Timestamp(date.today())

    dates = pd.bdate_range(start=start_dt, end=end_dt)
    n     = len(dates)

    dt    = 1 / 252 

    mus   = np.array([p["mu_annual"]  for p in ASSET_PARAMS.values()]) * dt
    vols  = np.array([p["vol_annual"] for p in ASSET_PARAMS.values()]) * np.sqrt(dt)

    # Cholesky decomposition 
    L     = np.linalg.cholesky(CORRELATION)
    Z     = rng.standard_normal((n, len(TICKERS_SYNTH)))
    corr_Z = Z @ L.T

    log_returns = mus + vols * corr_Z

    # compound into price series starting at 100
    log_prices = np.cumsum(log_returns, axis=0)
    prices     = 100 * np.exp(log_prices)

    df = pd.DataFrame(prices, index=dates, columns=TICKERS_SYNTH)
    return df


def generate_crisis_period(
    base_prices: pd.DataFrame,
    crisis_start: str,
    crisis_end: str,
    crash_magnitudes: dict = None,
    seed: int = 99,
) -> pd.DataFrame:
    """
    For stress scenarios.
    """
    if crash_magnitudes is None:
        crash_magnitudes = {
            "SPY": -0.35, "TLT": 0.15, "UUP": 0.05,
            "GLD": 0.10,  "HYG": -0.25,
        }

    prices = base_prices.copy()
    mask   = (prices.index >= crisis_start) & (prices.index <= crisis_end)
    n_days = mask.sum()

    if n_days == 0:
        return prices

    rng = np.random.default_rng(seed)

    for ticker, total_move in crash_magnitudes.items():
        if ticker not in prices.columns:
            continue
        
        daily_drift = total_move / n_days
        noise       = rng.normal(0, abs(daily_drift) * 0.5, n_days)
        daily_moves = daily_drift + noise
        cumulative  = np.exp(np.cumsum(daily_moves))
        prices.loc[mask, ticker] = prices.loc[mask, ticker].values * (cumulative / cumulative[0])

    return prices


if __name__ == "__main__":
    prices = generate_synthetic_prices()
    print(f"Generated {len(prices)} trading days")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(prices.tail())
