import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from modules.data_loader import (
    fetch_prices, compute_returns, portfolio_returns,
    TICKERS, DEFAULT_WEIGHTS,
)
from modules.risk_metrics import (
    var_historical, var_parametric, var_montecarlo,
    expected_shortfall, rolling_var, backtest_var, risk_summary,
)
from modules.stress_testing import (
    all_historical_stress, all_hypothetical_stress,
    hypothetical_stress, HYPOTHETICAL_SCENARIOS,
)
from modules.ai_commentary import (
    generate_risk_commentary, format_metrics_for_api,
)

# ── Page config ──

st.set_page_config(
    page_title="MRA Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #003366;
    }
    .zone-green  { color: #1a7a4a; font-weight: 600; }
    .zone-yellow { color: #b8860b; font-weight: 600; }
    .zone-red    { color: #c0392b; font-weight: 600; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #003366;
        border-bottom: 2px solid #003366;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──

with st.sidebar:
    st.markdown("## Market Risk Analytics")
    st.markdown("*MRA-style risk monitoring dashboard*")
    st.divider()

    st.markdown("### Portfolio Weights")
    weights = {}
    total = 0
    for ticker, label in TICKERS.items():
        default = DEFAULT_WEIGHTS[ticker]
        w = st.slider(
            f"{ticker} — {label.split('(')[1].rstrip(')')}",
            min_value=0.0,
            max_value=1.0,
            value=default,
            step=0.05,
            key=f"w_{ticker}",
        )
        weights[ticker] = w
        total += w

    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f}. Normalizing automatically.")
    w_sum = sum(weights.values())
    weights = {k: v / w_sum for k, v in weights.items()}

    st.divider()
    st.markdown("### Parameters")
    confidence = st.selectbox("VaR Confidence Level", [0.99, 0.975, 0.95], index=0,
                              format_func=lambda x: f"{int(x*100)}%")
    lookback = st.selectbox("Data Lookback", ["3 Years", "5 Years", "Full History"],
                            index=1)
    rolling_window = st.slider("Rolling VaR Window (days)", 60, 504, 252, step=21)

    lookback_map = {"3 Years": "2022-01-01", "5 Years": "2020-01-01",
                    "Full History": "2018-01-01"}
    start_date = lookback_map[lookback]


# ── Data loading ──

@st.cache_data(ttl=3600)
def load_data(start):
    prices  = fetch_prices(start=start)
    returns = compute_returns(prices)
    return prices, returns

with st.spinner("Loading market data..."):
    prices, returns = load_data(start_date)

port = portfolio_returns(returns, weights)

today_str = datetime.today().strftime("%Y-%m-%d")


# ── Header ───

st.title("Market Risk Analytics Dashboard")
col_date, col_update = st.columns([3, 1])
with col_date:
    st.caption(f"Report Date: {today_str}  |  Data: {prices.index[0].date()} to {prices.index[-1].date()}  |  {len(prices)} trading days")
with col_update:
    st.caption(f"Confidence: {int(confidence*100)}%  |  Window: {rolling_window}d")

st.divider()


# ── Section 1: Core Risk Metrics ───

st.markdown('<div class="section-header">1. Current Risk Metrics</div>', unsafe_allow_html=True)

recent = port.iloc[-252:] 
vh  = var_historical(recent, confidence)
vp  = var_parametric(recent, confidence)
vm  = var_montecarlo(recent, confidence)
es  = expected_shortfall(recent, confidence)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Historical VaR", f"{vh*100:.3f}%",
              help="Percentile of actual historical returns. No distribution assumption.")
with col2:
    st.metric("Parametric VaR", f"{vp*100:.3f}%",
              help="Assumes normally distributed returns. Faster but underestimates fat tails.")
with col3:
    st.metric("Monte Carlo VaR", f"{vm*100:.3f}%",
              help="10,000 simulated return paths from fitted distribution.")
with col4:
    st.metric("Expected Shortfall", f"{es*100:.3f}%",
              help="Average loss beyond VaR. FRTB regulatory metric replacing VaR.")
with col5:
    spread = abs(es - vh)
    st.metric("ES / VaR Spread", f"{spread*100:.3f}%",
              help="Larger spread = heavier tail risk. Key signal for tail shape.")

st.divider()


# ── Section 2: Return Distribution ──

st.markdown('<div class="section-header">2. Return Distribution & VaR Comparison</div>', unsafe_allow_html=True)

col_dist, col_summary = st.columns([2, 1])

with col_dist:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=recent * 100,
        nbinsx=80,
        name="Daily Returns",
        marker_color="#4a90d9",
        opacity=0.7,
    ))
    for val, label, color in [
        (vh, f"Hist VaR {int(confidence*100)}%", "#e74c3c"),
        (vp, f"Param VaR {int(confidence*100)}%", "#e67e22"),
        (vm, f"MC VaR {int(confidence*100)}%",    "#9b59b6"),
        (es, f"Expected Shortfall",               "#c0392b"),
    ]:
        fig.add_vline(
            x=val * 100,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
        )
    fig.update_layout(
        title="Portfolio Daily Return Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        plot_bgcolor="black",
        paper_bgcolor="black",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_summary:
    st.markdown("**VaR Method Comparison**")
    summary_df = risk_summary(recent, confidence)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.markdown("**Interpretation**")
    diff_hist_param = abs(vh - vp) * 100
    if diff_hist_param > 0.3:
        st.warning(f"Historical and parametric VaR differ by {diff_hist_param:.2f}%. "
                   "Suggests return distribution is non-normal (fat tails present).")
    else:
        st.success(f"Historical and parametric VaR aligned ({diff_hist_param:.2f}% spread). "
                   "Returns approximately normal.")

st.divider()


# ── Section 3: Rolling VaR & Backtesting ───

st.markdown('<div class="section-header">3. Rolling VaR & Backtesting</div>', unsafe_allow_html=True)

roll_var = rolling_var(port, window=rolling_window, confidence=confidence, method="historical")
bt = backtest_var(port, roll_var)

current_zone     = bt["zone"].iloc[-1] if not bt.empty else "N/A"
total_exceptions = int(bt["exception"].sum())
exception_rate   = total_exceptions / len(bt) * 100 if len(bt) > 0 else 0

col_bt1, col_bt2, col_bt3 = st.columns(3)
with col_bt1:
    st.metric("Total Exceptions", total_exceptions,
              help="Days where actual loss exceeded VaR forecast.")
with col_bt2:
    st.metric("Exception Rate", f"{exception_rate:.2f}%",
              help=f"Expected ~{(1-confidence)*100:.0f}% at {int(confidence*100)}% confidence.")
with col_bt3:
    zone_color = {"Green": "normal", "Yellow": "inverse", "Red": "off"}.get(current_zone, "normal")
    st.metric("Basel Zone (current)", current_zone)

# Rolling VaR chart with exceptions marked
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=port.index, y=port * 100,
    name="Daily Return", line=dict(color="#4a90d9", width=0.8), opacity=0.6,
))
fig2.add_trace(go.Scatter(
    x=roll_var.index, y=roll_var * 100,
    name=f"Rolling VaR ({rolling_window}d)", line=dict(color="#e74c3c", width=1.5, dash="dash"),
))

exceptions = bt[bt["exception"]]
fig2.add_trace(go.Scatter(
    x=exceptions.index, y=exceptions["return"] * 100,
    mode="markers", name="Exceptions",
    marker=dict(color="#c0392b", size=6, symbol="x"),
))

fig2.update_layout(
    title=f"Rolling {int(confidence*100)}% VaR vs Realized Returns",
    xaxis_title="Date", yaxis_title="Return (%)",
    height=380,plot_bgcolor="black", paper_bgcolor="black",
    legend=dict(orientation="h", y=1.02),
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()


# ── Section 4: Stress Testing ───

st.markdown('<div class="section-header">4. Stress Testing</div>', unsafe_allow_html=True)

tab_hist, tab_hypo = st.tabs(["Historical Scenarios", "Hypothetical Shocks"])

with tab_hist:
    hist_df = all_historical_stress(returns, weights)
    if not hist_df.empty:
        fig3 = px.bar(
            hist_df.reset_index(),
            x="scenario",
            y="cumulative_loss",
            color="cumulative_loss",
            color_continuous_scale=["#c0392b", "#e67e22", "#f1c40f"],
            labels={"cumulative_loss": "Cumulative Loss (%)", "scenario": ""},
            title="Portfolio Performance During Historical Stress Periods",
        )
        fig3.update_layout(height=340, plot_bgcolor="black",paper_bgcolor="black",
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(hist_df, use_container_width=True)

with tab_hypo:
    hypo_df = all_hypothetical_stress(weights)
    fig4 = px.bar(
        hypo_df.reset_index(),
        x="Scenario",
        y="Portfolio Impact %",
        color="Portfolio Impact %",
        color_continuous_scale=["#c0392b", "#e67e22", "#f1c40f"],
        title="Hypothetical Scenario Impact on Portfolio",
    )
    fig4.update_layout(height=340, plot_bgcolor="black", paper_bgcolor="black",
                       coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    selected_scenario = st.selectbox("Drill into scenario:", list(HYPOTHETICAL_SCENARIOS.keys()))
    scenario_detail = hypothetical_stress(weights, selected_scenario)
    contrib_df = pd.DataFrame.from_dict(
        scenario_detail["contributions"], orient="index", columns=["Impact %"]
    ).sort_values("Impact %")
    fig5 = px.bar(
        contrib_df.reset_index(),
        x="index", y="Impact %",
        color="Impact %",
        color_continuous_scale=["#c0392b", "#95a5a6", "#27ae60"],
        title=f"Asset Contribution: {selected_scenario}",
        labels={"index": "Asset"},
    )
    fig5.update_layout(height=300, plot_bgcolor="black",paper_bgcolor="black",
                       coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

st.divider()


# ── Section 5: Asset-level Risk Breakdown ───

st.markdown('<div class="section-header">5. Asset-Level Risk Attribution</div>', unsafe_allow_html=True)

asset_vars = {}
for ticker in returns.columns:
    asset_returns = returns[ticker].dropna().iloc[-252:]
    w_i = weights.get(ticker, 0)
    asset_vars[ticker] = {
        "Weight":       round(w_i * 100, 1),
        "Asset VaR %":  round(var_historical(asset_returns, confidence) * 100, 3),
        "Contribution %": round(var_historical(asset_returns, confidence) * w_i * 100, 3),
        "Label":        TICKERS.get(ticker, ticker),
    }

asset_df = pd.DataFrame(asset_vars).T.sort_values("Contribution %")
st.dataframe(asset_df, use_container_width=True)

fig6 = px.pie(
    asset_df.reset_index(),
    values="Weight",
    names="index",
    title="Portfolio Weight Allocation",
    hole=0.4,
)
fig6.update_layout(height=300)
st.plotly_chart(fig6, use_container_width=True)

st.divider()


# ── Section 6: AI Risk Commentary ──

st.markdown('<div class="section-header">6. AI-Generated Risk Commentary</div>', unsafe_allow_html=True)
st.caption("Powered by Grok — mirrors the gen AI workflow integration initiative within MRA")

if st.button("Generate Daily Risk Commentary", type="primary"):
    with st.spinner("Generating commentary..."):
        metrics_for_api = format_metrics_for_api(vh, vp, vm, es)
        hypo_impacts = {
            name: round(hypothetical_stress(weights, name)["portfolio_impact"], 2)
            for name in list(HYPOTHETICAL_SCENARIOS.keys())[:3]
        }
        commentary = generate_risk_commentary(
            metrics=metrics_for_api,
            stress_results=hypo_impacts,
            backtest_zone=current_zone,
            portfolio_weights={k: round(v, 3) for k, v in weights.items()},
            date=today_str,
        )
    st.markdown("**Daily Risk Report**")
    st.info(commentary)
    st.caption(f"Generated: {today_str} | Model: Claude Sonnet | Confidence: {int(confidence*100)}%")
else:
    st.caption("Click to generate a plain-English risk summary for senior management review.")

st.divider()
st.caption("Built by Raja Darshini Rajamani | Purdue Mathematics '25 | CFA L1 | FM (SOA) | SIE | darshinirajamani.com")
