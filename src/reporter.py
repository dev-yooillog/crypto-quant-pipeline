import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import REPORT_DIR
from src.eda import market_concentration, return_distribution, gainers_losers
from src.analyzer import return_summary, segment_analysis, top_performers


def _pct(v):
    if pd.isna(v):
        return "N/A"
    return f"{'+' if v > 0 else ''}{v:.1f}%"


def _usd(v):
    if pd.isna(v):
        return "N/A"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    if v >= 1e6:
        return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


def generate_report(df):
    now         = datetime.now().strftime("%Y-%m-%d %H:%M")
    conc        = market_concentration(df)
    ret         = return_summary(df)
    dist        = return_distribution(df)
    seg         = segment_analysis(df)
    gainers, _  = gainers_losers(df, "30d_pct", min_mcap=1e8, n=5)
    top30d      = top_performers(df, "30d_pct", min_mcap=1e8, n=10)
    median_30d  = df[~df["is_stablecoin"]]["30d_pct"].median()
    pos_rate    = (df["30d_pct"] > 0).sum() / df["30d_pct"].notna().sum() * 100

    L = []
    L.append("# Cryptocurrency Market Analysis Report")
    L.append(f"> {now}  |  분석 코인: {len(df):,}개\n")

    L.append("---\n## Executive Summary\n")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Total Market Cap        | **${conc['total_market_cap_B']:.1f}B** |")
    L.append(f"| BTC Dominance           | **{conc['btc_dominance_pct']}%** |")
    L.append(f"| ETH Dominance           | {conc['eth_dominance_pct']}% |")
    L.append(f"| Stablecoin Share        | {conc['stablecoin_pct']}% |")
    L.append(f"| Top-10 Concentration    | {conc['top10_concentration_pct']}% |")
    L.append(f"| 30d Median Return       | **{_pct(median_30d)}** |")
    L.append(f"| % Coins Positive (30d)  | {pos_rate:.1f}% |\n")

    L.append("---\n## 1. 30d Return Distribution\n")
    L.append("| Bucket | Count |")
    L.append("|--------|-------|")
    for b, c in zip(dist["bins"], dist["counts"]):
        L.append(f"| {b} | {c:,} |")

    L.append("\n---\n## 2. Return Summary (excl. Stablecoins)\n")
    L.append("| Period | Mean | Median | Positive Rate |")
    L.append("|--------|------|--------|---------------|")
    for idx in ret.index:
        period = idx.replace("_pct", "").upper()
        L.append(f"| {period} | {_pct(ret.loc[idx,'mean'])} "
                 f"| {_pct(ret.loc[idx,'50%'])} "
                 f"| {ret.loc[idx,'positive_rate']:.1f}% |")

    L.append("\n---\n## 3. Segment Analysis\n")
    L.append("| Segment | Count | Mean 30d | Median 30d |")
    L.append("|---------|-------|----------|------------|")
    for s, row in seg.iterrows():
        L.append(f"| {s} | {int(row['count'])} "
                 f"| {_pct(row['mean_30d'])} "
                 f"| {_pct(row['median_30d'])} |")

    L.append("\n---\n## 4. Top 30d Performers (mcap > $100M)\n")
    L.append("| Coin | Symbol | 30d | Market Cap |")
    L.append("|------|--------|-----|------------|")
    for _, r in top30d.iterrows():
        L.append(f"| {r['Coin Name']} | {r['Symbol']} "
                 f"| {_pct(r['30d_pct'])} | {_usd(r['market_cap'])} |")

    L.append("\n---\n## 5. Key Insights\n")
    insights = [
        ("극단적 시장 집중",
         f"상위 10개 코인이 {conc['top10_concentration_pct']}% 점유. BTC 단독 {conc['btc_dominance_pct']}%."),
        ("전반적 강세장",
         f"30d 중앙값 {_pct(median_30d)}. 코인의 {pos_rate:.1f}%가 양수 수익률."),
        ("USDT 유동성 허브",
         "USDT 24h 거래량 $47B → BTC($22B)의 2배. 스테이블코인이 시장 완충재."),
        ("단기-장기 수익률 무상관",
         "1h↔30d 상관계수 ≈ 0. 단기 모멘텀으로 장기 성과 예측 불가."),
        ("소형 코인 의심 유동성",
         "vol/mcap > 10,000배 코인 다수. wash trading 또는 시세 조종 가능성."),
    ]
    for i, (title, body) in enumerate(insights, 1):
        L.append(f"**{i}. {title}**")
        L.append(f"> {body}\n")

    L.append("---\n## 6. Recommended Actions\n")
    for title, body in [
        ("포트폴리오", "BTC/ETH 코어(50-60%) + L1 알트(30%) + 소형(≤10%) 3-레이어 구조."),
        ("기회",      "30d 50-200% + 시총 $1B↑ 코인 → RSI/온체인 검증 후 접근."),
        ("리스크",    "vol/mcap 극단 코인 진입 금지. 규제 이슈 코인 익스포저 조절."),
        ("데이터",    "시계열 OHLCV, 온체인 지표, 펀딩레이트, 소셜 센티먼트 추가 필요."),
    ]:
        L.append(f"- **{title}**: {body}")

    L.append("\n---\n## 7. Limitations\n")
    for item in [
        "단일 시점 스냅샷 — 추세 분석 불가.",
        "가격 결측치 41.6% — 소형 코인 신뢰도 저하.",
        "Total Supply 단위 불일치 — 공급량 분석 제외.",
        "거래소 거래량 중복 집계 가능성.",
    ]:
        L.append(f"- {item}")

    return "\n".join(L)


def save_report(df, path=None):
    if path is None:
        path = REPORT_DIR / "crypto_analysis_report.md"
    Path(path).write_text(generate_report(df), encoding="utf-8")
    print(f"[reporter] 저장 → outputs/reports/{Path(path).name}") 
    return Path(path)


if __name__ == "__main__":
    from src.loader import load_raw
    from src.preprocessor import preprocess
    df = preprocess(load_raw())
    save_report(df)