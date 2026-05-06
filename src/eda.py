import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

NUMERIC_COLS = ["price", "volume", "market_cap",
                "1h_pct", "24h_pct", "7d_pct", "30d_pct"]


def descriptive_stats(df):
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    stats = df[cols].describe().T
    stats["missing"]     = df[cols].isnull().sum()
    stats["missing_pct"] = (stats["missing"] / len(df) * 100).round(1)
    return stats.round(4)


def missing_report(df):
    total = df.isnull().sum()
    pct   = (total / len(df) * 100).round(2)
    return (pd.DataFrame({"missing_count": total, "missing_pct": pct})
            .query("missing_count > 0"))


def return_distribution(df, col="30d_pct"):
    series = df[col].dropna()
    bins   = [-np.inf, -50, 0, 50, 100, np.inf]
    labels = ["< -50%", "-50~0%", "0~50%", "50~100%", "> 100%"]
    counts = pd.cut(series, bins=bins, labels=labels).value_counts().reindex(labels)
    return {"bins": labels, "counts": counts.tolist()}


def top_n_by_mcap(df, n=20):
    cols = ["Rank", "Coin Name", "Symbol", "price",
            "24h_pct", "30d_pct", "volume", "market_cap",
            "cap_tier", "is_stablecoin"]
    existing = [c for c in cols if c in df.columns]
    return (df.dropna(subset=["market_cap"])
              .nlargest(n, "market_cap")[existing]
              .reset_index(drop=True))


def gainers_losers(df, col="30d_pct", min_mcap=1e8, n=10):
    filtered = df[df["market_cap"] >= min_mcap].dropna(subset=[col])
    gainers  = filtered.nlargest(n, col)[["Coin Name", "Symbol", col, "market_cap"]]
    losers   = filtered.nsmallest(n, col)[["Coin Name", "Symbol", col, "market_cap"]]
    return gainers.reset_index(drop=True), losers.reset_index(drop=True)


def market_concentration(df):
    total = df["market_cap"].sum()
    if total == 0:
        return {}

    def _dom(sym):
        row = df[df["Symbol"] == sym]["market_cap"]
        return float(row.values[0] / total * 100) if len(row) else 0.0

    top10_mc = df.nlargest(10, "market_cap")["market_cap"].sum()
    stable   = df[df["is_stablecoin"]]["market_cap"].sum()

    return {
        "total_market_cap_B":      round(total / 1e9, 1),
        "btc_dominance_pct":       round(_dom("BTC"), 1),
        "eth_dominance_pct":       round(_dom("ETH"), 1),
        "stablecoin_pct":          round(stable / total * 100, 1),
        "top10_concentration_pct": round(top10_mc / total * 100, 1),
    }


def print_eda_report(df):
    print("\n[ 1. 기술통계 ]")
    print(descriptive_stats(df).to_string())
    print("\n[ 2. 결측치 ]")
    print(missing_report(df).to_string())
    print("\n[ 3. 시장 집중도 ]")
    for k, v in market_concentration(df).items():
        print(f"  {k:<35}: {v}")
    print("\n[ 4. 30일 수익률 분포 ]")
    dist = return_distribution(df)
    for b, c in zip(dist["bins"], dist["counts"]):
        print(f"  {b:<12}: {c:>5}  {'█' * int(c / 100)}")
    print("\n[ 5. 시가총액 상위 10개 ]")
    print(top_n_by_mcap(df, 10).to_string())


if __name__ == "__main__":
    from src.loader import load_raw
    from src.preprocessor import preprocess
    df = preprocess(load_raw())
    print_eda_report(df)