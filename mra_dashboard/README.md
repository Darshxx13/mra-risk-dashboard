# Market Risk Analytics Dashboard
### MRA-style risk monitoring tool built for the Morgan Stanley Market Risk Analyst role

A Python-based market risk dashboard that mirrors the core workflows of Morgan Stanley's Market Risk Analytics (MRA) group. Built using real market data, regulatory-grade risk methodologies, and gen AI-powered commentary generation.

---

## What it does

**Core risk metrics**
- Value at Risk (VaR) computed three ways: historical simulation, parametric (normal), and Monte Carlo
- Expected Shortfall (CVaR): the FRTB regulatory replacement for VaR
- Side-by-side comparison of all methods with interpretation of spread

**Backtesting**
- Rolling VaR vs realized P&L comparison
- Exception flagging with Basel III traffic light test (Green / Yellow / Red zones)
- Exception rate vs theoretical expected rate

**Stress testing**
- Historical scenarios: 2008 GFC, COVID March 2020, 2022 rate shock, 2018 Q4 selloff, Taper Tantrum
- Hypothetical shocks: rates +200bps, equity crash -20%, combined macro shock, FX dislocation, liquidity crisis
- Per-asset contribution breakdown for each scenario

**Asset-level attribution**
- Individual VaR and weighted contribution per asset class
- Coverage across equities, rates, FX, commodities, and credit (mirrors MRA desk structure)

**Gen AI commentary**
- Uses Anthropic's Claude API to generate plain-English daily risk summaries
- Output formatted for senior management review
- Directly demonstrates gen AI workflow integration capability

---

## Portfolio structure

Assets selected to mirror the five desk groups within Morgan Stanley's FRM:

| Ticker | Asset              | Desk         |
|--------|--------------------|--------------|
| SPY    | S&P 500 ETF        | Equities     |
| TLT    | 20Y Treasury ETF   | Rates/Macro  |
| UUP    | USD Index ETF      | FX/Macro     |
| GLD    | Gold ETF           | Commodities  |
| HYG    | High Yield Bond ETF | Credit/Micro |

---

## Tech stack

- `yfinance` for real market data
- `pandas` / `numpy` for data manipulation and statistical modeling
- `scipy` for distribution fitting and parametric VaR
- `plotly` for interactive charts
- `streamlit` for the dashboard front end
- Anthropic Claude API for gen AI risk commentary

---

## Run locally

```bash
git clone https://github.com/Darshxx13/mra-risk-dashboard
cd mra-risk-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## Methodology notes

**Why three VaR methods?**
Each method has tradeoffs. Historical simulation makes no distributional assumption but is limited by the history available. Parametric VaR is fast but underestimates tail risk when returns are fat-tailed. Monte Carlo is the most flexible and can extend to non-normal distributions and multi-day horizons. The spread between methods signals the degree of non-normality in the return distribution.

**Why Expected Shortfall over VaR?**
VaR tells you the threshold loss at a confidence level but says nothing about severity beyond that threshold. ES (CVaR) computes the average loss in the worst scenarios. Under the Basel III Fundamental Review of the Trading Book (FRTB), ES at 97.5% replaces VaR at 99% as the primary regulatory capital metric because it is subadditive and better captures tail shape.

**Backtesting and the Basel traffic light**
A model at 99% confidence should produce exceptions approximately 1% of the time, roughly 2-3 per year. The Basel framework evaluates model performance over 250 trading days: 0-4 exceptions (Green zone, model accepted), 5-9 exceptions (Yellow zone, capital multiplier applied), 10+ exceptions (Red zone, model rejected). This dashboard tracks rolling exceptions and flags zone changes.

---

## Author

Raja Darshini Rajamani  
B.S. Mathematics (Business concentration), Purdue University 2025  
Minors: Computer Science, Economics, Statistics  
CFA Level 1 | Financial Mathematics (SOA) | Securities Industry Essentials (FINRA)  
[darshinirajamani.com](https://darshinirajamani.com) | [LinkedIn](https://linkedin.com/in/darshini-rajamani)
