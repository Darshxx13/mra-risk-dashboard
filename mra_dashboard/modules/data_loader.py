import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = {
    "SPY":  "Equities (S&P 500)",
    "TLT":  "Rates (20Y Treasury)",
    "UUP":  "FX (USD Index)",
    "GLD":  "Commodities (Gold)",
    "HYG":  "Credit (High Yield)",
}

DEFAULT_WEIGHTS = {
    "SPY": 0.30,
    "TLT": 0.25,
    "UUP": 0.15,
    "GLD": 0.15,
    "HYG": 0.15,
}


def fetch_prices(start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    try:
        raw = yf.download(
            list(TICKERS.keys()),
            start=start,
            end=end,
            auto_adjust=True,
        )
        prices = raw["Close"].dropna()
        return prices
    except Exception:
        from modules.synthetic_data import generate_synthetic_prices
        return generate_synthetic_prices(start=start, end=end)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_returns(returns: pd.DataFrame, weights: dict = None) -> pd.Series:
    if weights is None:
        weights = DEFAULT_WEIGHTS
    w = pd.Series(weights)
    w = w.reindex(returns.columns).fillna(0)
    w = w / w.sum()
    port = returns.dot(w)
    port.name = "Portfolio"
    return port


if __name__ == "__main__":
    prices = fetch_prices()
    returns = compute_returns(prices)
    port = portfolio_returns(returns)
    print(f"Loaded {len(prices)} trading days")
    print(f"Assets: {list(prices.columns)}")
    print(f"Portfolio return stats:\n{port.describe()}")
