from pathlib import Path

ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR    = ROOT_DIR / "outputs"
CHART_DIR     = OUTPUT_DIR / "charts"
REPORT_DIR    = OUTPUT_DIR / "reports"
EXPORT_DIR    = OUTPUT_DIR / "data"

RAW_FILE   = RAW_DIR       / "CryptocurrencyData.csv"
CLEAN_FILE = PROCESSED_DIR / "crypto_clean.csv"

for _d in [PROCESSED_DIR, CHART_DIR, REPORT_DIR, EXPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

LARGE_CAP_THRESHOLD  = 1_000_000_000
MID_CAP_THRESHOLD    =   100_000_000
OUTLIER_RETURN_PCT   = 1000
HIGH_LIQUIDITY_RATIO = 0.5

STABLECOIN_SYMBOLS = ["USDT", "USDC", "BUSD", "DAI", "TUSD", "FRAX", "USDP"]

PALETTE = {
    "primary": "#378ADD",
    "success": "#1D9E75",
    "danger":  "#D85A30",
    "warning": "#BA7517",
    "purple":  "#7F77DD",
    "gray":    "#888780",
}

COLOR_UP   = PALETTE["success"]
COLOR_DOWN = PALETTE["danger"]
CHART_DPI  = 150