import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_FILE

EXPECTED_COLS = [
    "Rank", "Coin Name", "Symbol", "Price",
    "1h", "24h", "7d", "30d",
    "24h Volume", "Circulating Supply", "Total Supply", "Market Cap",
]

def load_raw(path=RAW_FILE):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"누락: {missing}")
    return df

def print_summary(df):
    print("  데이터셋 기본 정보")
    print(f"  총 행   : {len(df):,}")
    print(f"  총 컬럼 : {df.shape[1]}")
    print(f"  컬럼 목록  : {df.columns.tolist()}")
    print("\n  결측치 현황:")
    for col, cnt in df.isnull().sum().items():
        if cnt > 0:
            print(f"    {col:<25}: {cnt:>5} ({cnt/len(df)*100:.1f}%)")
    print("=" * 55)

if __name__ == "__main__":
    df = load_raw()
    print_summary(df)
    print(df.head(3).to_string())