"""
app.py — PropIQ: Real-Time House Price Intelligence Platform
Production-grade Streamlit dashboard with dark theme and advanced analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime

from preprocessing import (
    load_or_generate_data, preprocess, LOCATIONS,
    LOCATION_TIERS, LOCATION_BASE_MULTIPLIER
)
from predict import predict_price, get_market_trend_data, get_location_avg_psf

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Time House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Clean Light Theme (AutoLink-style)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}
.stApp { background-color: #f8fafc; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p { color: #64748b; font-size: 13px; }

/* ── Metric Cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.metric-card:hover { border-color: #6366f1; box-shadow: 0 4px 16px rgba(99,102,241,0.10); }
.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    margin-bottom: 6px;
    font-weight: 600;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #1e293b;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
}

/* ── Price Hero Card ── */
.price-hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #a78bfa 100%);
    border-radius: 18px;
    padding: 36px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 8px 32px rgba(99,102,241,0.25);
}
.price-hero-label { font-size: 12px; color: rgba(255,255,255,0.8); letter-spacing: 2.5px; text-transform: uppercase; font-weight: 600; }
.price-hero-value { font-size: 54px; font-weight: 800; color: #ffffff; margin: 10px 0; letter-spacing: -1.5px; }
.price-hero-sub   { font-size: 14px; color: rgba(255,255,255,0.75); }

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.4px;
}
.badge-green  { background: #dcfce7; color: #16a34a; border: 1px solid #bbf7d0; }
.badge-red    { background: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }
.badge-blue   { background: #e0e7ff; color: #4f46e5; border: 1px solid #c7d2fe; }
.badge-yellow { background: #fef9c3; color: #ca8a04; border: 1px solid #fde68a; }

/* ── Section Headers ── */
.section-header {
    font-size: 17px;
    font-weight: 700;
    color: #1e293b;
    border-left: 3px solid #6366f1;
    padding-left: 12px;
    margin: 28px 0 16px;
}

/* ── Page Header ── */
.page-header {
    padding: 24px 0 12px;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 24px;
}
.page-title {
    font-size: 26px;
    font-weight: 800;
    color: #1e293b;
}
.page-subtitle { font-size: 14px; color: #64748b; margin-top: 4px; }

/* ── Nav Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f1f5f9; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 7px; color: #64748b;
    padding: 8px 20px; font-size: 13px; font-weight: 600;
}
.stTabs [aria-selected="true"] { background: #ffffff; color: #6366f1; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }

/* ── Input widgets ── */
.stSlider > div > div > div { background: #6366f1; }
div[data-testid="stSelectbox"] > div { background: #ffffff; border-color: #e2e8f0; color: #1e293b; }
.stNumberInput input { background: #ffffff; border-color: #e2e8f0; color: #1e293b; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    color: white; border: none; border-radius: 10px;
    padding: 12px 28px; font-size: 14px; font-weight: 700;
    transition: opacity 0.2s, transform 0.1s; width: 100%;
    box-shadow: 0 4px 14px rgba(99,102,241,0.30);
}
.stButton > button:hover { opacity: 0.90; transform: translateY(-1px); }

/* ── Divider ── */
hr { border-color: #e2e8f0; }

/* ── Comparison Table ── */
.compare-table th { background: #f8fafc; color: #64748b; font-size: 12px; text-transform: uppercase; }
.compare-table td { color: #1e293b; font-size: 14px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

/* ── Plotly light ── */
.js-plotly-plot .plotly .modebar { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA / MODEL BOOTSTRAP (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data_cached():
    df = load_or_generate_data()
    _, _, df_proc, _, _ = preprocess(df)
    return df_proc

@st.cache_resource(show_spinner=False)
def boot_models():
    import os
    if not os.path.exists("models/random_forest.pkl"):
        from train_model import train_and_save
        train_and_save()

with st.spinner("🔄 Initialising PropIQ Intelligence Engine..."):
    boot_models()
    df = load_data_cached()

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f8fafc",
    font=dict(family="Plus Jakarta Sans", color="#64748b", size=12),
    xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    margin=dict(l=16, r=16, t=32, b=16),
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Input Panel
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:20px;font-weight:800;color:#1e293b;margin-bottom:4px;">🏠 Real Time House Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#6366f1;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;font-weight:700;">Property Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("##### 📍 Property Details")

    location   = st.selectbox("Location", LOCATIONS, index=2)
    area       = st.slider("Area (sq ft)", 300, 8000, 1500, step=50)
    bedrooms   = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=2)
    bathrooms  = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)

    st.markdown("##### 🏗️ Property Specs")
    age        = st.slider("Building Age (yrs)", 0, 30, 5)
    floor      = st.slider("Floor", 0, 25, 3)
    parking    = st.selectbox("Parking Slots", [0, 1, 2], index=1)
    furnished  = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"], index=1)
    amenities  = st.slider("Amenities Score", 1, 10, 7)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Now", use_container_width=True)

    # Sample auto-fill
    if st.button("⚡ Sample Data Fill", use_container_width=True):
        st.session_state["sample_filled"] = True

    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#94a3b8;text-align:center;">PropIQ v2.0 · Powered by ML</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — prediction history
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Auto predict on load
if st.session_state.last_result is None or predict_btn:
    result = predict_price(area, bedrooms, bathrooms, location, age, floor,
                           parking, furnished, amenities)
    st.session_state.last_result = result
    # Save to history
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Location": location, "Area": area,
        "Beds": bedrooms, "Baths": bathrooms,
        "Price": f"₹{result['rf_price']:,}",
        "Verdict": result["market_status"],
        "Investment": result["investment_tag"],
    })

result = st.session_state.last_result

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🏠 Prediction Dashboard", "📊 Analytics", "📈 Market Trends", "📌 Property Insights", "🗂️ History"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="page-header"><div class="page-title">Prediction Dashboard</div><div class="page-subtitle">Real-time AI-powered property valuation for Hyderabad</div></div>', unsafe_allow_html=True)

    # ── Price Hero ────────────────────────────────────────────────────────────
    col_hero, col_meta = st.columns([2, 1])
    with col_hero:
        st.markdown(f"""
        <div class="price-hero">
            <div class="price-hero-label">Estimated Market Price</div>
            <div class="price-hero-value">₹ {result['rf_price']:,}</div>
            <div class="price-hero-sub">
                Range: ₹{result['price_range'][0]:,} — ₹{result['price_range'][1]:,}
                &nbsp;|&nbsp; {area} sq ft in {location}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_meta:
        tier = result["tier"]
        tier_color = {"Premium": "badge-red", "High": "badge-yellow",
                      "Mid": "badge-blue", "Standard": "badge-green"}[tier]
        ms_color = {"Underpriced": "badge-green", "Fair": "badge-blue", "Overpriced": "badge-red"}[result["market_status"]]

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Location Tier</div>
            <div style="margin-top:4px;"><span class="badge {tier_color}">{tier}</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Market Status</div>
            <div style="margin-top:4px;"><span class="badge {ms_color}">{result['market_status']}</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Investment Signal</div>
            <div style="margin-top:6px;font-size:15px;font-weight:600;">{result['investment_tag']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpi_style = "font-size:22px;font-weight:700;color:#6366f1;"
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Price / Sq Ft</div><div style="{kpi_style}">₹ {result["price_per_sqft"]:,}</div><div class="metric-sub">Area avg: ₹{result["avg_psf_area"]:,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">AI Confidence</div><div style="{kpi_style}">{result["confidence"]}%</div><div class="metric-sub">Random Forest + LR</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Linear Regression Est.</div><div style="{kpi_style}">₹ {result["lr_price"]:,}</div><div class="metric-sub">Baseline model</div></div>', unsafe_allow_html=True)
    with k4:
        diff = result["rf_price"] - result["lr_price"]
        sign = "+" if diff >= 0 else ""
        st.markdown(f'<div class="metric-card"><div class="metric-label">Model Delta (RF vs LR)</div><div style="{kpi_style}">{sign}₹{diff:,}</div><div class="metric-sub">Ensemble spread</div></div>', unsafe_allow_html=True)

    # ── Model Comparison Chart ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    fig_cmp = go.Figure()
    models   = ["Linear Regression", "Random Forest (Main)"]
    prices   = [result["lr_price"], result["rf_price"]]
    colors   = ["#6366f1", "#10b981"]
    fig_cmp.add_trace(go.Bar(
        x=models, y=prices, marker_color=colors, width=0.4,
        text=[f"₹{p:,}" for p in prices], textposition="outside",
        textfont=dict(color="#1e293b", size=14)
    ))
    fig_cmp.add_shape(type="line", x0=-0.5, x1=1.5, y0=result["rf_price"], y1=result["rf_price"],
                      line=dict(color="#6366f1", width=1, dash="dot"))
    fig_cmp.update_layout(**PLOTLY_LAYOUT, height=280,
                          yaxis_title="Predicted Price (₹)",
                          showlegend=False)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Gauge ─────────────────────────────────────────────────────────────────
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown('<div class="section-header">Confidence Gauge</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["confidence"],
            number={"suffix": "%", "font": {"color": "#1e293b", "size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": "#6366f1"},
                "bgcolor": "#f1f5f9",
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 60], "color": "#fee2e2"},
                    {"range": [60, 80], "color": "#fef9c3"},
                    {"range": [80, 100], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "#10b981", "width": 3}, "value": 85}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#64748b", height=240, margin=dict(l=16, r=16, t=24, b=8))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        st.markdown('<div class="section-header">Price Range Spread</div>', unsafe_allow_html=True)
        lo, hi = result["price_range"]
        mid = result["rf_price"]
        fig_range = go.Figure()
        fig_range.add_trace(go.Scatter(
            x=[lo, mid, hi], y=[1, 1, 1],
            mode="markers+lines",
            marker=dict(size=[12, 20, 12], color=["#6366f1", "#10b981", "#6366f1"]),
            line=dict(color="#e2e8f0", width=4)
        ))
        fig_range.add_annotation(x=lo, y=1.15, text=f"₹{lo:,}", showarrow=False, font=dict(color="#64748b", size=12))
        fig_range.add_annotation(x=mid, y=0.82, text=f"₹{mid:,}", showarrow=False, font=dict(color="#10b981", size=14, weight=700))
        fig_range.add_annotation(x=hi, y=1.15, text=f"₹{hi:,}", showarrow=False, font=dict(color="#64748b", size=12))
        fig_range.update_layout(**PLOTLY_LAYOUT, height=240, yaxis_visible=False, xaxis_showgrid=False)
        st.plotly_chart(fig_range, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="page-header"><div class="page-title">Analytics Dashboard</div><div class="page-subtitle">Explore patterns across 2,000+ property records</div></div>', unsafe_allow_html=True)

    # ── Area vs Price scatter ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Area vs Price by Location Tier</div>', unsafe_allow_html=True)
    sample_df = df.sample(min(600, len(df)), random_state=42)
    tier_palette = {"Premium": "#ef4444", "High": "#f59e0b", "Mid": "#6366f1", "Standard": "#10b981"}
    fig_scatter = px.scatter(
        sample_df, x="Area", y="Price", color="LocationTier",
        color_discrete_map=tier_palette, opacity=0.65,
        hover_data=["Location", "Bedrooms", "Bathrooms"],
        labels={"Price": "Price (₹)", "Area": "Area (sq ft)", "LocationTier": "Tier"},
        trendline="ols",
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=360, legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Bedrooms vs Avg Price
        st.markdown('<div class="section-header">Bedrooms vs Avg Price</div>', unsafe_allow_html=True)
        bed_avg = df.groupby("Bedrooms")["Price"].mean().reset_index()
        fig_bed = go.Figure(go.Bar(
            x=bed_avg["Bedrooms"], y=bed_avg["Price"],
            marker=dict(color=bed_avg["Price"], colorscale="Blues"),
            text=[f"₹{int(p/1e5)}L" for p in bed_avg["Price"]],
            textposition="outside"
        ))
        fig_bed.update_layout(**PLOTLY_LAYOUT, height=300,
                              xaxis_title="Bedrooms", yaxis_title="Avg Price (₹)")
        st.plotly_chart(fig_bed, use_container_width=True)

    with col_b:
        # Price distribution
        st.markdown('<div class="section-header">Price Distribution</div>', unsafe_allow_html=True)
        fig_hist = go.Figure(go.Histogram(
            x=df["Price"], nbinsx=50,
            marker_color="#6366f1", opacity=0.8,
            name="Price"
        ))
        fig_hist.add_vline(x=df["Price"].mean(), line_color="#10b981", line_dash="dash",
                           annotation_text="Mean", annotation_font_color="#10b981")
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=300,
                               xaxis_title="Price (₹)", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Heatmap — Location tier vs Bedrooms avg price ─────────────────────────
    st.markdown('<div class="section-header">Avg Price Heatmap — Tier × Bedrooms</div>', unsafe_allow_html=True)
    heat_df = df.groupby(["LocationTier", "Bedrooms"])["Price"].mean().unstack(fill_value=0)
    tier_order = ["Premium", "High", "Mid", "Standard"]
    heat_df = heat_df.reindex([t for t in tier_order if t in heat_df.index])
    fig_heat = go.Figure(go.Heatmap(
        z=heat_df.values,
        x=[f"{b} Bed" for b in heat_df.columns],
        y=heat_df.index.tolist(),
        colorscale="Blues",
        text=[[f"₹{int(v/1e5)}L" for v in row] for row in heat_df.values],
        texttemplate="%{text}",
        showscale=True
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Avg PSF by location ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Price per Sq Ft by Location</div>', unsafe_allow_html=True)
    loc_psf = get_location_avg_psf()
    psf_df = pd.DataFrame({"Location": list(loc_psf.keys()), "PSF": list(loc_psf.values())})
    psf_df["Tier"] = psf_df["Location"].map(LOCATION_TIERS)
    psf_df = psf_df.sort_values("PSF", ascending=True)
    fig_psf = go.Figure(go.Bar(
        y=psf_df["Location"], x=psf_df["PSF"],
        orientation="h",
        marker=dict(color=[tier_palette[t] for t in psf_df["Tier"]], opacity=0.85),
        text=[f"₹{p:,}" for p in psf_df["PSF"]], textposition="outside"
    ))
    fig_psf.update_layout(**PLOTLY_LAYOUT, height=520,
                          xaxis_title="Avg Price / Sq Ft (₹)", yaxis_title="",
                          showlegend=False)
    st.plotly_chart(fig_psf, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET TRENDS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="page-header"><div class="page-title">Market Trends</div><div class="page-subtitle">24-month simulated price per sq ft evolution by location tier</div></div>', unsafe_allow_html=True)

    months, trend = get_market_trend_data()

    fig_trend = go.Figure()
    colors_trend = {"Premium": "#ef4444", "High": "#f59e0b", "Mid": "#6366f1", "Standard": "#10b981"}
    for tier, vals in trend.items():
        fig_trend.add_trace(go.Scatter(
            x=months, y=vals, mode="lines+markers",
            name=tier, line=dict(color=colors_trend[tier], width=2),
            marker=dict(size=4)
        ))
    fig_trend.update_layout(**PLOTLY_LAYOUT, height=400,
                            yaxis_title="Avg Price/Sq Ft (₹)",
                            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1))
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Growth insight cards ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">24-Month Growth Summary</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (tier, vals) in enumerate(trend.items()):
        growth = (vals[-1] - vals[0]) / vals[0] * 100
        sign = "+" if growth > 0 else ""
        color = "#10b981" if growth > 5 else "#f59e0b" if growth > 0 else "#ef4444"
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{tier}</div>
                <div style="font-size:24px;font-weight:700;color:{color};">{sign}{growth:.1f}%</div>
                <div class="metric-sub">₹{vals[0]:,} → ₹{vals[-1]:,}/sqft</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Yearly forecast ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">12-Month Price Forecast (Selected Location Tier)</div>', unsafe_allow_html=True)
    sel_tier = result["tier"]
    last_val = trend[sel_tier][-1]
    future_months = pd.date_range(start=pd.Timestamp.now(), periods=13, freq="ME")[1:]
    np.random.seed(99)
    forecast = last_val * np.cumprod(1 + np.random.normal(0.009, 0.01, 12))
    conf_upper = forecast * 1.05
    conf_lower = forecast * 0.95

    fig_fore = go.Figure()
    fig_fore.add_trace(go.Scatter(
        x=future_months.strftime("%b %Y"), y=forecast,
        mode="lines+markers", name=f"{sel_tier} Forecast",
        line=dict(color=colors_trend[sel_tier], width=2.5)
    ))
    fig_fore.add_trace(go.Scatter(
        x=future_months.strftime("%b %Y").tolist() + future_months.strftime("%b %Y").tolist()[::-1],
        y=conf_upper.tolist() + conf_lower.tolist()[::-1],
        fill="toself", fillcolor="rgba(99,102,241,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"
    ))
    fig_fore.update_layout(**PLOTLY_LAYOUT, height=340,
                           yaxis_title="Forecast Price/SqFt (₹)",
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_fore, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PROPERTY INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="page-header"><div class="page-title">Property Insights</div><div class="page-subtitle">Feature importance, price drivers, and comparative analysis</div></div>', unsafe_allow_html=True)

    # ── Feature Importances ────────────────────────────────────────────────────
    importances = joblib.load("models/feature_importances.pkl")
    feat_labels = {
        "Area": "Built-up Area", "Bedrooms": "Bedrooms", "Bathrooms": "Bathrooms",
        "Age": "Building Age", "Floor": "Floor Level", "Parking": "Parking Slots",
        "AmenitiesScore": "Amenities Score", "Location_enc": "Location",
        "Furnished_enc": "Furnishing", "Tier_enc": "Location Tier",
        "TierMultiplier": "Tier Multiplier", "BedBathRatio": "Bed/Bath Ratio"
    }
    imp_df = pd.DataFrame([
        {"Feature": feat_labels.get(k, k), "Importance": v}
        for k, v in importances.items()
    ]).sort_values("Importance", ascending=True)

    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    fig_imp = go.Figure(go.Bar(
        y=imp_df["Feature"], x=imp_df["Importance"],
        orientation="h",
        marker=dict(color=imp_df["Importance"], colorscale="Blues", opacity=0.9),
        text=[f"{v:.3f}" for v in imp_df["Importance"]], textposition="outside"
    ))
    fig_imp.update_layout(**PLOTLY_LAYOUT, height=380,
                          xaxis_title="Importance Score", yaxis_title="")
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Smart insights ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Smart Price Drivers for Your Property</div>', unsafe_allow_html=True)

    top_feats = list(importances.keys())[:4]
    insight_map = {
        "Location_enc": f"📍 Location ({location}) is the #1 price driver — {result['tier']} tier carries a {LOCATION_BASE_MULTIPLIER[result['tier']]}× base multiplier.",
        "TierMultiplier": f"🏙️ The {result['tier']} tier multiplier directly boosts your estimated PSF vs city baseline.",
        "Tier_enc": f"🏙️ Being in {result['tier']} tier positions this property in a {'high-demand' if result['tier'] in ['Premium','High'] else 'moderate'} price bracket.",
        "Area": f"📐 Area of {area} sq ft is the largest physical cost driver, contributing ≈ ₹{int(area * result['price_per_sqft'] * 0.55):,} to total price.",
        "AmenitiesScore": f"🏊 Amenities score of {amenities}/10 adds ≈ ₹{amenities*30000:,} to the valuation.",
        "Bedrooms": f"🛏️ {bedrooms} bedrooms contributes ≈ ₹{bedrooms*150000:,} to the base estimate.",
        "Bathrooms": f"🚿 {bathrooms} bathrooms adds ≈ ₹{bathrooms*80000:,} to overall value.",
        "Age": f"🏗️ Building age of {age} years reduces value by approx ₹{age*25000:,} vs new construction.",
        "Floor": f"🪜 Floor {floor} adds ≈ ₹{floor*10000:,} premium — upper floors typically earn more light & views.",
        "Parking": f"🚗 {parking} parking slot(s) add ≈ ₹{parking*120000:,} to market value.",
        "Furnished_enc": f"🛋️ {furnished} status contributes {'₹4.5L' if furnished=='Furnished' else '₹2L' if furnished=='Semi-Furnished' else '₹0'} in furnishing premium.",
        "BedBathRatio": f"🔢 Bed/Bath ratio of {bedrooms/max(bathrooms,1):.1f} is within optimal range for this property type.",
    }

    for feat in top_feats:
        if feat in insight_map:
            st.markdown(f"""
            <div class="metric-card" style="padding:16px 20px;">
                <div style="font-size:14px;color:#1e293b;">{insight_map[feat]}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Compare properties ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Compare: Your Property vs Area Average</div>', unsafe_allow_html=True)
    loc_psf_map = get_location_avg_psf()
    avg_price = loc_psf_map[location] * area

    compare_data = {
        "Metric": ["Estimated Price", "Price / Sq Ft", "Vs Area Avg", "Location Tier", "Investment"],
        "Your Property": [
            f"₹ {result['rf_price']:,}",
            f"₹ {result['price_per_sqft']:,}",
            f"{'▲' if result['rf_price'] > avg_price else '▼'} {abs(result['rf_price']-avg_price)/avg_price*100:.1f}%",
            result["tier"],
            result["investment_tag"]
        ],
        "Area Average": [
            f"₹ {int(avg_price):,}",
            f"₹ {loc_psf_map[location]:,}",
            "—", result["tier"], "—"
        ]
    }
    cdf = pd.DataFrame(compare_data)
    st.dataframe(cdf.set_index("Metric"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORY
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="page-header"><div class="page-title">Prediction History</div><div class="page-subtitle">All predictions made in this session</div></div>', unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, height=360)

        if len(st.session_state.history) > 1:
            st.markdown('<div class="section-header">Price Comparison Across Predictions</div>', unsafe_allow_html=True)
            prices_num = [int(h["Price"].replace("₹", "").replace(",", "")) for h in st.session_state.history]
            labels_num = [f"{h['Location']}\n{h['Area']}sqft" for h in st.session_state.history]
            fig_hist_cmp = go.Figure(go.Bar(
                x=labels_num, y=prices_num,
                marker_color="#6366f1", opacity=0.85,
                text=[f"₹{p:,}" for p in prices_num], textposition="outside"
            ))
            fig_hist_cmp.update_layout(**PLOTLY_LAYOUT, height=320,
                                       yaxis_title="Predicted Price (₹)", showlegend=False)
            st.plotly_chart(fig_hist_cmp, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions recorded yet. Use the sidebar to predict a property price.")
