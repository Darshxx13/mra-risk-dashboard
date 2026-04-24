"""
ai_commentary.py

The prompt is engineered to produce output in the style of an MRA risk report:
  - Factual, not alarmist
  - References specific metrics
  - Flags anything requiring attention
  - Suggests monitoring priorities
"""

import json
import requests
from typing import Dict, Optional

import os

try:
    import streamlit as st
except ImportError:
    st = None

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY and st:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

ANTHROPIC_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


def generate_risk_commentary(
    metrics: Dict,
    stress_results: Dict,
    backtest_zone: str,
    portfolio_weights: Dict,
    date: str,
) -> str:
    """
    Generate a senior-management-ready risk commentary from computed metrics.

    Parameters
    ----------
    metrics : dict
        Keys: var_historical, var_parametric, var_montecarlo, expected_shortfall
              (all as percentages, e.g. -1.23 means -1.23%)
    stress_results : dict
        Hypothetical scenario impacts as percentages
    backtest_zone : str
        Basel traffic light zone: 'Green', 'Yellow', or 'Red'
    portfolio_weights : dict
        Asset weights
    date : str
        Report date string
    """

    prompt = f"""You are a market risk analyst at a major investment bank writing a daily risk commentary 
for senior management. Write in a professional, factual tone. Be concise. Flag anything 
that needs attention. Do not use em dashes.

Today's Date: {date}

Portfolio Composition:
{json.dumps(portfolio_weights, indent=2)}

Risk Metrics (as % of portfolio value):
- Historical VaR (99%, 1-day):     {metrics.get('var_historical', 'N/A')}%
- Parametric VaR (99%, 1-day):     {metrics.get('var_parametric', 'N/A')}%
- Monte Carlo VaR (99%, 1-day):    {metrics.get('var_montecarlo', 'N/A')}%
- Expected Shortfall (99%, 1-day): {metrics.get('expected_shortfall', 'N/A')}%

Backtesting Status (Basel Traffic Light):
- Zone: {backtest_zone}
- Green = 0-4 exceptions (model accepted)
- Yellow = 5-9 exceptions (under scrutiny)
- Red = 10+ exceptions (model rejected)

Stress Test Results (hypothetical scenarios, portfolio impact %):
{json.dumps(stress_results, indent=2)}

Write a 2-paragraph risk commentary:
Paragraph 1: Summarize today's risk position, referencing the VaR and ES figures. 
Note the spread between methods and what it implies about tail risk.
Paragraph 2: Highlight the most severe stress scenario, comment on the backtesting 
status, and identify which asset class poses the largest concentration risk today. 
Flag any metrics that warrant closer monitoring.

Keep it under 200 words total. Sound like a human analyst, not a chatbot."""

    payload = {
        "model": MODEL,
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Commentary unavailable: {str(e)}]"


def format_metrics_for_api(
    var_h: float,
    var_p: float,
    var_mc: float,
    es: float,
) -> Dict:
    """Convert raw float VaR values to percentage dict for the API call."""
    return {
        "var_historical":    round(var_h * 100, 3),
        "var_parametric":    round(var_p * 100, 3),
        "var_montecarlo":    round(var_mc * 100, 3),
        "expected_shortfall": round(es * 100, 3),
    }


if __name__ == "__main__":
    # smoke test with dummy metrics
    test_metrics = {
        "var_historical":    -1.45,
        "var_parametric":    -1.32,
        "var_montecarlo":    -1.51,
        "expected_shortfall": -2.10,
    }
    test_stress = {
        "Equity Crash -20%":   -8.20,
        "Combined Macro Shock": -5.60,
        "Rates +200bps":       -3.40,
    }
    commentary = generate_risk_commentary(
        metrics=test_metrics,
        stress_results=test_stress,
        backtest_zone="Green",
        portfolio_weights={"SPY": 0.30, "TLT": 0.25, "UUP": 0.15, "GLD": 0.15, "HYG": 0.15},
        date="2026-04-20",
    )
    print(commentary)
