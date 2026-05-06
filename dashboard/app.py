import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import CLEAN_FILE
from src.loader import load_raw
from src.preprocessor import preprocess, save_clean
from src.eda import market_concentration, gainers_losers
from src.analyzer import segment_analysis

# 페이지 설정
st.set_page_config(page_title="Crypto Analysis", page_icon="📊", layout="wide")

@st.cache_data
def get_data():
    if CLEAN_FILE.exists():
        return pd.read_csv(CLEAN_FILE)
    df = preprocess(load_raw())
    save_clean(df)
    return df

df = get_data()

# 사이드바 필터
st.sidebar.title("Filters")
cap_opt  = ["All"] + sorted(df["cap_tier"].dropna().unique().tolist())
sel_cap  = st.sidebar.selectbox("Cap Tier", cap_opt)
inc_stab = st.sidebar.checkbox("Include Stablecoins", False)
min_b    = st.sidebar.slider("Min Market Cap ($B)", 0.0, 100.0, 0.0, 0.1)
ret_col  = st.sidebar.selectbox("Return Period", ["30d_pct", "7d_pct", "24h_pct"])
top_n    = st.sidebar.slider("Top N (charts)", 5, 30, 15)

# 필터링 로직
mask = pd.Series([True] * len(df), index=df.index)
if sel_cap != "All":
    mask &= df["cap_tier"] == sel_cap
if not inc_stab:
    mask &= ~df["is_stablecoin"]
if min_b > 0:
    mask &= df["market_cap"] >= min_b * 1e9
filt = df[mask].copy()

# 헤더
st.title("Cryptocurrency Market Analysis")
st.caption(f"전체 {len(df):,}개 | 필터 후 {len(filt):,}개")

# KPI 섹션
conc = market_concentration(df)
med  = filt[ret_col].median()
pos  = (filt[ret_col] > 0).sum() / filt[ret_col].notna().sum() * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Market Cap",  f"${conc['total_market_cap_B']:.1f}B")
c2.metric("BTC Dominance",     f"{conc['btc_dominance_pct']}%")
c3.metric(f"{ret_col} Median", f"{'+' if med > 0 else ''}{med:.1f}%")
c4.metric("% Coins Positive",  f"{pos:.1f}%")
c5.metric("Coins Shown",       f"{len(filt):,}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Market", "Returns", "Liquidity", "Rankings", "Data"]
)

# Tab 1 — Market
with tab1:
    l, r = st.columns(2)
    with l:
        st.subheader("Market Cap Share")
        top10 = df.nlargest(10, "market_cap")[["Symbol", "market_cap"]].dropna()
        other = df["market_cap"].sum() - top10["market_cap"].sum()
        pie   = pd.concat([top10, pd.DataFrame({"Symbol": ["Others"], "market_cap": [other]})])
        fig   = px.pie(pie, values="market_cap", names="Symbol", hole=0.5)
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(showlegend=False, height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch", key="pie_mcap_share")
    with r:
        st.subheader("Coins per Segment")
        seg = segment_analysis(df).reset_index()
        fig = px.bar(seg, x="segment", y="count", text="count",
                     color="median_30d", color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=0)
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch", key="bar_segment_count")

    st.subheader("Top 15 — Market Cap vs 24h Volume ($B)")
    t15 = df.nlargest(15, "market_cap").dropna(subset=["market_cap", "volume"])
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Market Cap", x=t15["Symbol"],
                         y=t15["market_cap"] / 1e9, marker_color="#378ADD"))
    fig.add_trace(go.Bar(name="24h Volume", x=t15["Symbol"],
                         y=t15["volume"] / 1e9, marker_color="#1D9E75"))
    fig.update_layout(barmode="group", yaxis_title="$B", height=340,
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch", key="bar_mcap_vs_vol")

# Tab 2 — Returns
with tab2:
    l, r = st.columns(2)
    with l:
        st.subheader("Return Distribution")
        hist = filt[ret_col].dropna().clip(-100, 500)
        fig  = px.histogram(hist, nbins=60, color_discrete_sequence=["#378ADD"])
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=hist.median(), line_color="#1D9E75",
                      annotation_text=f"Med {hist.median():.1f}%")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, width="stretch", key="hist_returns")
    with r:
        st.subheader("Multi-Period — Top 10 Coins")
        syms = df.nlargest(10, "market_cap")["Symbol"].tolist()
        pdf  = df[df["Symbol"].isin(syms)][["Symbol","24h_pct","7d_pct","30d_pct"]]
        pdf  = pdf.melt("Symbol", var_name="Period", value_name="Return")
        pdf["Period"] = pdf["Period"].str.replace("_pct", "").str.upper()
        fig  = px.bar(pdf, x="Symbol", y="Return", color="Period", barmode="group",
                      color_discrete_sequence=["#378ADD","#1D9E75","#BA7517"])
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch", key="bar_multi_period")

    st.subheader(f"Top & Bottom by {ret_col.replace('_pct','').upper()}")
    n_side   = min(top_n, len(filt))
    top_c    = filt.nlargest(n_side, ret_col)[["Symbol", ret_col]].dropna()
    bot_c    = filt.nsmallest(n_side // 2, ret_col)[["Symbol", ret_col]].dropna()
    rank_df  = pd.concat([top_c, bot_c]).drop_duplicates("Symbol")
    rank_df  = rank_df.sort_values(ret_col, ascending=True)
    rank_df["Dir"] = rank_df[ret_col].apply(lambda x: "Up" if x >= 0 else "Down")
    fig = px.bar(rank_df, x=ret_col, y="Symbol", orientation="h", color="Dir",
                 color_discrete_map={"Up": "#1D9E75", "Down": "#D85A30"})
    fig.update_layout(height=max(380, n_side * 22), showlegend=False)
    st.plotly_chart(fig, width="stretch", key="bar_top_bottom")

# Tab 3 — Liquidity
with tab3:
    st.subheader("Market Cap vs Volume (log-log)")
    sc = df.dropna(subset=["market_cap","volume"])
    sc = sc[(sc["market_cap"] > 0) & (sc["volume"] > 0)]
    fig = px.scatter(sc, x="market_cap", y="volume", color="cap_tier",
                     hover_name="Coin Name",
                     hover_data={"Symbol": True, "30d_pct": True},
                     log_x=True, log_y=True,
                     color_discrete_map={"Large":"#378ADD","Mid":"#1D9E75",
                                         "Small":"#BA7517","Unknown":"#888"})
    fig.update_traces(marker=dict(opacity=0.5, size=5))
    fig.update_layout(height=460)
    st.plotly_chart(fig, width="stretch", key="scatter_liquidity")

    st.subheader("High Liquidity Alert (vol/mcap > 50%)")
    susp = (df[df["vol_mcap_ratio"] > 0.5]
            .sort_values("vol_mcap_ratio", ascending=False)
            [["Coin Name","Symbol","price","volume","market_cap","vol_mcap_ratio"]]
            .head(20).reset_index(drop=True))
    susp["vol_mcap_ratio"] = susp["vol_mcap_ratio"].round(2)
    st.dataframe(susp, width="stretch")

# Tab 4 — Rankings
with tab4:
    g_col, l_col = st.columns(2)
    with g_col:
        st.subheader("Top Gainers (30d, mcap>$100M)")
        gainers, _ = gainers_losers(filt, "30d_pct", min_mcap=1e8, n=15)
        gainers_disp = gainers.copy()
        gainers_disp["30d_pct"] = gainers_disp["30d_pct"].round(1)
        gainers_disp["market_cap"] = gainers_disp["market_cap"].apply(
            lambda x: f"${x/1e9:.2f}B" if x >= 1e9 else f"${x/1e6:.0f}M")
        st.dataframe(gainers_disp, width="stretch", hide_index=True)
    with l_col:
        st.subheader("Top Losers (30d, mcap>$100M)")
        _, losers = gainers_losers(filt, "30d_pct", min_mcap=1e8, n=15)
        losers_disp = losers.copy()
        losers_disp["30d_pct"] = losers_disp["30d_pct"].round(1)
        losers_disp["market_cap"] = losers_disp["market_cap"].apply(
            lambda x: f"${x/1e9:.2f}B" if x >= 1e9 else f"${x/1e6:.0f}M")
        st.dataframe(losers_disp, width="stretch", hide_index=True) 

# Tab 5 — Raw Data
with tab5:
    st.subheader("Filtered Data")
    show = ["Rank","Coin Name","Symbol","price","24h_pct",
            "7d_pct","30d_pct","volume","market_cap","cap_tier"]
    existing = [c for c in show if c in filt.columns]
    st.dataframe(filt[existing].sort_values("market_cap", ascending=False).head(500),
                 width="stretch", hide_index=True)
    csv = filt[existing].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "filtered.csv", "text/csv")
    
# streamlit run dashboard/app.py