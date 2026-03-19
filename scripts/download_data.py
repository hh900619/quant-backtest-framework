from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm


# =========================================================
# 路徑設定
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


# =========================================================
# 🔥 完整市場清單（升級版）
# =========================================================
DEFAULT_TICKERS = [

    # =========================
    # 🇺🇸 美股指數
    # =========================
    "^GSPC",   # S&P500
    "^NDX",    # Nasdaq 100
    "^DJI",    # Dow Jones
    "^RUT",    # Russell 2000
    "^VIX",    # 恐慌指數

    # =========================
    # 🇹🇼 台股
    # =========================
    "^TWII",

    # =========================
    # 🌏 全球指數
    # =========================
    "^FTSE",
    "^GDAXI",
    "^N225",
    "^HSI",
    "000001.SS",

    # =========================
    # 📈 ETF
    # =========================
    "SPY",
    "QQQ",
    "DIA",
    "IWM",

    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLP",
    "XLY",

    # =========================
    # 🧠 科技龍頭
    # =========================
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",

    # =========================
    # 🪙 商品
    # =========================
    "GC=F",
    "SI=F",
    "CL=F",
    "HG=F",

    # =========================
    # 💱 匯率
    # =========================
    "USDJPY=X",
    "EURUSD=X",
    "GBPUSD=X",
    "USDTWD=X",

    # =========================
    # 📉 利率 / 債券
    # =========================
    "^TNX",
    "IEF",
    "TLT",

    # =========================
    # 🧪 風格 ETF
    # =========================
    "ARKK",
    "SMH",
    "XBI",
]


# =========================================================
# 工具函數
# =========================================================
def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def safe_filename(ticker: str) -> str:
    return ticker.replace("^", "").replace("/", "_").replace("=", "")


def file_exists(ticker: str) -> bool:
    filename = safe_filename(ticker)
    return (DATA_DIR / f"{filename}.parquet").exists()


def download_one_ticker(
    ticker: str,
    period: str = "15y",
    interval: str = "1d",
) -> pd.DataFrame:

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    df = flatten_yf_columns(df)
    df = df.dropna().copy()

    return df


def save_dataframe(df: pd.DataFrame, ticker: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    filename = safe_filename(ticker)

    csv_path = DATA_DIR / f"{filename}.csv"
    parquet_path = DATA_DIR / f"{filename}.parquet"

    df.to_csv(csv_path)
    df.to_parquet(parquet_path)

    return csv_path, parquet_path


# =========================================================
# 主程式
# =========================================================
def main():

    print("=" * 60)
    print("🚀 開始下載全球市場資料...")
    print(f"📁 儲存位置：{DATA_DIR}")
    print("=" * 60)

    success = []
    failed = []
    skipped = []

    for ticker in tqdm(DEFAULT_TICKERS):

        if file_exists(ticker):
            print(f"[SKIP] {ticker} 已存在")
            skipped.append(ticker)
            continue

        try:
            df = download_one_ticker(ticker)

            if df.empty:
                print(f"[FAILED] {ticker} 空資料")
                failed.append(ticker)
                continue

            csv_path, parquet_path = save_dataframe(df, ticker)

            print(f"[OK] {ticker} | rows={len(df)}")

            success.append(ticker)

        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
            failed.append(ticker)

    print("\n" + "=" * 60)
    print("🎉 完成下載")
    print(f"✅ 成功：{len(success)}")
    print(f"⏭️ 跳過：{len(skipped)}")
    print(f"❌ 失敗：{len(failed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()