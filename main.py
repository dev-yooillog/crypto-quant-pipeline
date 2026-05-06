import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.loader       import load_raw, print_summary
from src.preprocessor import preprocess, save_clean, print_preprocess_summary
from src.eda          import print_eda_report
from src.analyzer     import print_analysis_report
from src.visualizer   import run_all_charts
from src.reporter     import save_report
from config           import EXPORT_DIR


def banner(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")


def run(skip_charts=False, skip_report=False):
    t0 = time.time()

    banner("Step 1 / 6  |  Load")
    df_raw = load_raw()
    print_summary(df_raw)

    banner("Step 2 / 6  |  Preprocess")
    df = preprocess(df_raw)
    print_preprocess_summary(df)
    save_clean(df)
    df.to_csv(EXPORT_DIR / "crypto_clean.csv", index=False)

    banner("Step 3 / 6  |  EDA")
    print_eda_report(df)

    banner("Step 4 / 6  |  Analysis")
    print_analysis_report(df)

    banner("Step 5 / 6  |  Visualize")
    if skip_charts:
        print("  [skipped]")
    else:
        run_all_charts(df)

    banner("Step 6 / 6  |  Report")
    if skip_report:
        print("  [skipped]")
    else:
        save_report(df)

    banner(f"DONE  |  {time.time()-t0:.1f}s")
    print("""
  outputs/data/crypto_clean.csv
  outputs/charts/*.png
  outputs/reports/crypto_analysis_report.md

  대시보드: streamlit run dashboard/app.py
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-charts", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()
    run(skip_charts=args.skip_charts, skip_report=args.skip_report)