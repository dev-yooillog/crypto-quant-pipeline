import pandas as pd
import numpy as np
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def return_summary(df):
    non_stable = df[~df["is_stablecoin"]]
    cols = [c for c in ["1h_pct", "24h_pct", "7d_pct", "30d_pct"]
            if c in df.columns]
    result = non_stable[cols].describe().T
    result["positive_count"] = [(non_stable[c] > 0).sum() for c in cols]
    result["negative_count"] = [(non_stable[c] < 0).sum() for c in cols]
    result["positive_rate"]  = (
        result["positive_count"] /
        (result["positive_count"] + result["negative_count"]) * 100
    ).round(1)
    return result.round(2)


def return_correlation(df):
    cols = [c for c in ["1h_pct", "24h_pct", "7d_pct", "30d_pct"]
            if c in df.columns]
    return df[cols].corr().round(3)


def price_volume_correlation(df):
    sub = df.dropna(subset=["price", "volume"])
    sub = sub[(sub["price"] > 0) & (sub["volume"] > 0)]
    r, p = stats.spearmanr(np.log(sub["price"]), np.log(sub["volume"]))
    return {"spearman_r": round(r, 3), "p_value": round(p, 6)}


def liquidity_analysis(df):
    sub = df.dropna(subset=["vol_mcap_ratio"]).copy()
    sub["liquidity_tier"] = pd.cut(
        sub["vol_mcap_ratio"],
        bins=[0, 0.05, 0.2, 0.5, np.inf],
        labels=["낮음(<5%)", "보통(5-20%)", "높음(20-50%)", "매우높음(>50%)"],
    )
    return sub.groupby("liquidity_tier", observed=True)["Coin Name"].count()


def suspicious_liquidity(df, threshold=1.0, n=15):
    cols = ["Coin Name", "Symbol", "price", "volume", "market_cap", "vol_mcap_ratio"]
    existing = [c for c in cols if c in df.columns]
    return (df[df["vol_mcap_ratio"] > threshold]
            .nlargest(n, "vol_mcap_ratio")[existing]
            .reset_index(drop=True))


def segment_analysis(df):
    df = df.copy()
    df["segment"] = df.apply(
        lambda r: "Stablecoin" if r["is_stablecoin"]
        else r.get("cap_tier", "Unknown"),
        axis=1,
    )
    return df.groupby("segment").agg(
        count=("Coin Name", "count"),
        mean_30d=("30d_pct", "mean"),
        median_30d=("30d_pct", "median"),
        mean_24h=("24h_pct", "mean"),
        avg_mcap=("market_cap", "mean"),
    ).round(2)


def top_performers(df, col="30d_pct", min_mcap=1e8, n=10):
    cols = ["Coin Name", "Symbol", "price", col, "market_cap", "cap_tier"]
    existing = [c for c in cols if c in df.columns]
    return (df[df["market_cap"] >= min_mcap]
            .dropna(subset=[col])
            .nlargest(n, col)[existing]
            .reset_index(drop=True))


def volatility_proxy(df):
    df = df.copy()
    df["volatility_1h"]  = df["1h_pct"].abs()
    df["volatility_24h"] = df["24h_pct"].abs()
    return df.groupby("cap_tier")[["volatility_1h", "volatility_24h"]].mean().round(3)


def print_analysis_report(df):
    print("  핵심 분석 리포트")
    print("\n[ 1. 기간별 수익률 요약 ]")
    print(return_summary(df).to_string())
    print("\n[ 2. 수익률 상관관계 ]")
    print(return_correlation(df).to_string())
    print("\n[ 3. 가격-거래량 스피어만 상관 ]")
    print(price_volume_correlation(df))
    print("\n[ 4. 유동성 티어 분포 ]")
    print(liquidity_analysis(df).to_string())
    print("\n[ 5. 의심 유동성 코인 ]")
    print(suspicious_liquidity(df).to_string())
    print("\n[ 6. 세그먼트별 수익률 ]")
    print(segment_analysis(df).to_string())
    print("\n[ 7. 30일 고수익 코인 ]")
    print(top_performers(df).to_string())
    print("\n[ 8. 변동성 프록시 ]")
    print(volatility_proxy(df).to_string())


if __name__ == "__main__":
    from src.loader import load_raw
    from src.preprocessor import preprocess
    df = preprocess(load_raw())
    print_analysis_report(df)