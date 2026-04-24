"""
Implements two types of stress tests:

1. Historical scenarios - replay actual market crisis windows
   (2008 GFC, COVID March 2020, 2022 rate shock)

2. Hypothetical scenarios - apply custom market shocks
   (rates up 200bps, equities -20%, combined macro shock)
"""

import numpy as np
import pandas as pd
from typing import Dict


# ── Historical Scenario Windows 

HISTORICAL_SCENARIOS = {
    "2008 GFC":          ("2008-09-01", "2009-03-31"),
    "COVID Crash":       ("2020-02-19", "2020-03-23"),
    "2022 Rate Shock":   ("2022-01-01", "2022-12-31"),
    "2018 Q4 Selloff":   ("2018-10-01", "2018-12-31"),
    "Taper Tantrum":     ("2013-05-01", "2013-09-30"),
}


def historical_stress(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    scenario_name: str,
) -> Dict:
   
    if scenario_name not in HISTORICAL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. "
                         f"Choose from {list(HISTORICAL_SCENARIOS.keys())}")

    start, end = HISTORICAL_SCENARIOS[scenario_name]

    window = returns.loc[start:end]
    if window.empty:
        return {"error": f"No data in window {start} to {end}"}

    # weighted portfolio return
    w = pd.Series(weights).reindex(returns.columns).fillna(0)
    w = w / w.sum()
    port = window.dot(w)

    cumulative_loss = float(port.sum())
    worst_day       = float(port.min())
    max_drawdown    = float((port.cumsum().cummax() - port.cumsum()).max())
    n_days          = len(port)
    n_negative      = int((port < 0).sum())

    return {
        "scenario":        scenario_name,
        "window":          f"{start} to {end}",
        "cumulative_loss": round(cumulative_loss * 100, 2),
        "worst_day":       round(worst_day * 100, 2),
        "max_drawdown":    round(max_drawdown * 100, 2),
        "trading_days":    n_days,
        "negative_days":   n_negative,
    }


def all_historical_stress(
    returns: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.DataFrame:
    """Run all historical scenarios and return a summary DataFrame."""
    results = []
    for name in HISTORICAL_SCENARIOS:
        r = historical_stress(returns, weights, name)
        if "error" not in r:
            results.append(r)
    return pd.DataFrame(results).set_index("scenario")


# ── Hypothetical Shocks 

HYPOTHETICAL_SCENARIOS = {
    "Rates +200bps": {
        "TLT": -0.15,   # long duration bonds fall when rates rise
        "HYG": -0.08,   # credit spreads widen
        "SPY": -0.05,   # equities mildly negative
        "UUP": +0.02,   # USD slightly stronger
        "GLD": -0.03,   # gold mixed
    },
    "Equity Crash -20%": {
        "SPY": -0.20,
        "HYG": -0.12,   # risk-off hits credit
        "TLT": +0.08,   # flight to safety
        "GLD": +0.05,   # gold as safe haven
        "UUP": +0.03,
    },
    "Combined Macro Shock": {
        # stagflation: rates up + equities down + commodities up
        "SPY": -0.15,
        "TLT": -0.12,
        "HYG": -0.10,
        "GLD": +0.08,
        "UUP": +0.05,
    },
    "FX Dislocation": {
        # sharp USD move, stress on FX-linked assets
        "UUP": -0.08,
        "GLD": +0.06,
        "SPY": -0.04,
        "TLT": +0.02,
        "HYG": -0.03,
    },
    "Liquidity Crisis": {
        # everything sells off (March 2020 style)
        "SPY": -0.25,
        "TLT": -0.05,
        "HYG": -0.18,
        "GLD": -0.08,
        "UUP": +0.04,
    },
}


def hypothetical_stress(
    weights: Dict[str, float],
    scenario_name: str,
) -> Dict:
    if scenario_name not in HYPOTHETICAL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. "
                         f"Choose from {list(HYPOTHETICAL_SCENARIOS.keys())}")

    shocks = HYPOTHETICAL_SCENARIOS[scenario_name]

    w = pd.Series(weights)
    w = w / w.sum()

    shock_series = pd.Series(shocks).reindex(w.index).fillna(0)
    portfolio_impact = float(w.dot(shock_series))

    # per-asset cont to total loss
    contributions = (w * shock_series).sort_values()

    return {
        "scenario":         scenario_name,
        "portfolio_impact": round(portfolio_impact * 100, 2),
        "contributions":    (contributions * 100).round(2).to_dict(),
        "worst_asset":      contributions.idxmin(),
        "worst_asset_loss": round(contributions.min() * 100, 2),
    }


def all_hypothetical_stress(weights: Dict[str, float]) -> pd.DataFrame:
    results = []
    for name in HYPOTHETICAL_SCENARIOS:
        r = hypothetical_stress(weights, name)
        results.append({
            "Scenario":          r["scenario"],
            "Portfolio Impact %": r["portfolio_impact"],
            "Worst Asset":        r["worst_asset"],
            "Worst Asset Loss %": r["worst_asset_loss"],
        })
    df = pd.DataFrame(results).set_index("Scenario")
    return df.sort_values("Portfolio Impact %")


if __name__ == "__main__":
    from data_loader import fetch_prices, compute_returns, DEFAULT_WEIGHTS

    prices  = fetch_prices()
    returns = compute_returns(prices)

    print("=== Historical Stress Tests ===")
    hist = all_historical_stress(returns, DEFAULT_WEIGHTS)
    print(hist[["cumulative_loss", "worst_day", "max_drawdown"]].to_string())

    print("\n=== Hypothetical Stress Tests ===")
    hypo = all_hypothetical_stress(DEFAULT_WEIGHTS)
    print(hypo.to_string())
