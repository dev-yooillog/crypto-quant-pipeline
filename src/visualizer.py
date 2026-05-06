import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHART_DIR, CHART_DPI, COLOR_UP, COLOR_DOWN, PALETTE
from src.eda import (market_concentration, top_n_by_mcap, return_distribution)
from src.analyzer import (return_correlation, segment_analysis, top_performers)

plt.rcParams["font.family"]       = "DejaVu Sans"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False


def _save(fig, name):
    path = CHART_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=CHART_DPI)
    plt.close(fig)
    print(f"  [chart] {path.name}")  
    return path


def chart_donut(df):
    conc  = market_concentration(df)
    total = df["market_cap"].sum()

    def mc(sym):
        row = df[df["Symbol"] == sym]["market_cap"]
        return float(row.values[0]) if len(row) else 0.0

    btc    = mc("BTC")
    eth    = mc("ETH")
    bnb    = mc("BNB")
    xrp    = mc("XRP")
    stable = df[df["is_stablecoin"]]["market_cap"].sum()
    others = total - btc - eth - bnb - xrp - stable

    sizes  = [btc, eth, stable, bnb, xrp, others]
    labels = ["Bitcoin", "Ethereum", "Stablecoins", "BNB", "XRP", "Others"]
    colors = [PALETTE["primary"], PALETTE["success"], PALETTE["warning"],
              PALETTE["danger"], PALETTE["purple"], PALETTE["gray"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 2})
    ax.legend(wedges, [f"{l}  {s/total*100:.1f}%" for l, s in zip(labels, sizes)],
              loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, frameon=False)
    ax.text(0, 0, f"${conc['total_market_cap_B']:.0f}B\nTotal",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#333")
    ax.set_title("Market Cap Share", fontsize=14, fontweight="bold", pad=20)
    return _save(fig, "01_market_cap_donut")


def chart_volume(df):
    top10 = top_n_by_mcap(df, 10).dropna(subset=["volume"])
    top10["vol_B"] = top10["volume"] / 1e9

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(top10["Symbol"], top10["vol_B"],
                  color=PALETTE["primary"], edgecolor="none", width=0.6)
    for bar, val in zip(bars, top10["vol_B"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"${val:.1f}B", ha="center", va="bottom", fontsize=9)
    ax.set_title("24h Volume — Top 10 Coins", fontsize=13, fontweight="bold")
    ax.set_ylabel("Volume ($B)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}B"))
    fig.tight_layout()
    return _save(fig, "02_volume_top10")


def chart_return_hist(df):
    series  = df[~df["is_stablecoin"]]["30d_pct"].dropna()
    clipped = series.clip(-100, 500)
    dist    = return_distribution(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(clipped, bins=70, color=PALETTE["primary"], edgecolor="none", alpha=0.85)
    axes[0].axvline(0, color="#aaa", linestyle="--", linewidth=1)
    axes[0].axvline(series.median(), color=COLOR_UP, linewidth=2,
                    label=f"Median {series.median():.1f}%")
    axes[0].axvline(series.mean(), color=PALETTE["warning"], linewidth=1.5,
                    linestyle="-.", label=f"Mean {series.mean():.1f}%")
    axes[0].set_title("30d Return Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("30d Return (%)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=10, frameon=False)

    bar_colors = [COLOR_DOWN, "#EF9F27", "#5DCAA5", "#1D9E75", "#085041"]
    bars = axes[1].bar(dist["bins"], dist["counts"],
                       color=bar_colors, edgecolor="none", width=0.6)
    for bar, cnt in zip(bars, dist["counts"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                     f"{cnt:,}", ha="center", va="bottom", fontsize=11)
    axes[1].set_title("30d Return Buckets", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    return _save(fig, "03_return_histogram")


def chart_period_returns(df):
    symbols = ["BTC","ETH","BNB","XRP","SOL","ADA","AVAX","LINK","MATIC","DOT"]
    sub = (df[df["Symbol"].isin(symbols)]
           .drop_duplicates("Symbol")
           .set_index("Symbol"))
    available = [s for s in symbols if s in sub.index]
    sub = sub.reindex(available)

    cols   = ["24h_pct", "7d_pct", "30d_pct"]
    labels = ["24h", "7d", "30d"]
    colors = [PALETTE["primary"], PALETTE["success"], PALETTE["warning"]]
    x      = np.arange(len(available))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (col, lbl, color) in enumerate(zip(cols, labels, colors)):
        ax.bar(x + i * width, sub[col].fillna(0), width,
               label=lbl, color=color, edgecolor="none", alpha=0.9)
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(available, fontsize=10)
    ax.set_ylabel("Return (%)")
    ax.set_title("Multi-Period Returns — Major Coins", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    return _save(fig, "04_period_returns")


def chart_heatmap(df):
    corr   = return_correlation(df)
    labels = ["1h", "24h", "7d", "30d"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8})
    ax.set_title("Return Period Correlation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "05_correlation_heatmap")


def chart_segment(df):
    seg    = segment_analysis(df).reset_index()
    seg    = seg.sort_values("median_30d", ascending=True)
    colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in seg["median_30d"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(seg["segment"], seg["median_30d"],
                   color=colors, edgecolor="none", height=0.5)
    for bar, val in zip(bars, seg["median_30d"]):
        xp = val + 1 if val >= 0 else val - 1
        ha = "left"  if val >= 0 else "right"
        ax.text(xp, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha=ha, fontsize=10)
    ax.axvline(0, color="#999", linewidth=0.8)
    ax.set_title("Median 30d Return by Segment", fontsize=13, fontweight="bold")
    ax.set_xlabel("Median 30d Return (%)")
    fig.tight_layout()
    return _save(fig, "06_segment_returns")


def chart_scatter(df):
    sub = df.dropna(subset=["market_cap", "volume"])
    sub = sub[(sub["market_cap"] > 0) & (sub["volume"] > 0)]
    tier_colors = {
        "Large":   PALETTE["primary"],
        "Mid":     PALETTE["success"],
        "Small":   PALETTE["warning"],
        "Unknown": PALETTE["gray"],
    }
    c_list = sub["cap_tier"].map(tier_colors).fillna(PALETTE["gray"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(sub["market_cap"], sub["volume"],
               c=c_list, alpha=0.35, s=12, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Market Cap ($, log)")
    ax.set_ylabel("24h Volume ($, log)")
    ax.set_title("Market Cap vs Volume (log-log)", fontsize=13, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=l) for l, c in tier_colors.items()]
    ax.legend(handles=patches, fontsize=9, frameon=False)
    fig.tight_layout()
    return _save(fig, "07_mcap_vs_volume_scatter")


def chart_gainers(df):
    gainers = top_performers(df, "30d_pct", min_mcap=1e8, n=12)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(gainers["Symbol"], gainers["30d_pct"],
                  color=COLOR_UP, edgecolor="none", width=0.6)
    for bar, val in zip(bars, gainers["30d_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"+{val:.0f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=COLOR_UP)
    ax.set_title("Top 30d Gainers (Market Cap > $100M)", fontsize=13, fontweight="bold")
    ax.set_ylabel("30d Return (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    return _save(fig, "08_top_gainers")


def run_all_charts(df):
    paths = [
        chart_donut(df),
        chart_volume(df),
        chart_return_hist(df),
        chart_period_returns(df),
        chart_heatmap(df),
        chart_segment(df),
        chart_scatter(df),
        chart_gainers(df),
    ]
    print(f"  → {len(paths)}개 차트 완료 → outputs/charts")
    return paths


if __name__ == "__main__":
    from src.loader import load_raw
    from src.preprocessor import preprocess
    df = preprocess(load_raw())
    run_all_charts(df)