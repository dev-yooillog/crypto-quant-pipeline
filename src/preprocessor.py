import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLEAN_FILE, STABLECOIN_SYMBOLS,
    LARGE_CAP_THRESHOLD, MID_CAP_THRESHOLD,
    HIGH_LIQUIDITY_RATIO, OUTLIER_RETURN_PCT,
)


def _parse_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("$", "").replace(",", "").replace(" ", "")
    if s in ("-", "", "$-", "—", "∞"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_pct(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def preprocess(df):
    df = df.copy()

    df["price"]      = df["Price"].apply(_parse_number)
    df["1h_pct"]     = df["1h"].apply(_parse_pct)
    df["24h_pct"]    = df["24h"].apply(_parse_pct)
    df["7d_pct"]     = df["7d"].apply(_parse_pct)
    df["30d_pct"]    = df["30d"].apply(_parse_pct)
    df["volume"]     = df["24h Volume"].apply(_parse_number)
    df["market_cap"] = df["Market Cap"].apply(_parse_number)
    df["supply"]     = df["Circulating Supply"].apply(_parse_number)

    df["is_stablecoin"] = df["Symbol"].isin(STABLECOIN_SYMBOLS)

    def _cap_tier(mc):
        if pd.isna(mc):
            return "Unknown"
        if mc >= LARGE_CAP_THRESHOLD:
            return "Large"
        if mc >= MID_CAP_THRESHOLD:
            return "Mid"
        return "Small"

    df["cap_tier"]          = df["market_cap"].apply(_cap_tier)
    df["vol_mcap_ratio"]    = df["volume"] / df["market_cap"]
    df["is_high_liq_flag"]  = df["vol_mcap_ratio"] > HIGH_LIQUIDITY_RATIO
    df["is_return_outlier"] = df["30d_pct"].abs() > OUTLIER_RETURN_PCT

    return df


def save_clean(df, path=CLEAN_FILE):
    df.to_csv(path, index=False)
    print(f"[preprocessor] 저장 → {path}")


def load_clean(path=CLEAN_FILE):
    return pd.read_csv(path)


def print_preprocess_summary(df):
    cols = ["price", "volume", "market_cap", "1h_pct", "24h_pct", "7d_pct", "30d_pct"]
    existing = [c for c in cols if c in df.columns]
    print("\n[ 결과 요약 ]")
    print(df[existing].describe().round(2).to_string())
    print(f"\n  스테이블코인  : {df['is_stablecoin'].sum()}")
    print(f"  Large Cap    : {(df['cap_tier'] == 'Large').sum()}")
    print(f"  Mid Cap      : {(df['cap_tier'] == 'Mid').sum()}")
    print(f"  Small Cap    : {(df['cap_tier'] == 'Small').sum()}")
    print(f"  유동성 경고   : {df['is_high_liq_flag'].sum()}")
    print(f"  수익률 이상치 : {df['is_return_outlier'].sum()}")


if __name__ == "__main__":
    from src.loader import load_raw
    raw = load_raw()
    clean = preprocess(raw)
    print_preprocess_summary(clean)
    save_clean(clean)