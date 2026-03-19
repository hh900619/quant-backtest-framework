from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


# =========================================================
# Ticker <-> filename mapping
# =========================================================
# 這裡放你常用、且有特殊符號的 ticker
# 目的是避免：
# ^TWII -> TWII 之後無法正確還原
SPECIAL_TICKER_MAP = {
    "^TWII": "TWII",
    "^GSPC": "GSPC",
    "^NDX": "NDX",
    "^TNX": "TNX",
    "^VIX": "VIX",
    "GC=F": "GC_F",
    "CL=F": "CL_F",
    "SI=F": "SI_F",
    "HG=F": "HG_F",
    "JPY=X": "JPY_X",
    "TWD=X": "TWD_X",
    "EURUSD=X": "EURUSD_X",
    "BTC-USD": "BTC_USD",
    "ETH-USD": "ETH_USD",
}

# 反向 mapping
REVERSE_SPECIAL_TICKER_MAP = {v: k for k, v in SPECIAL_TICKER_MAP.items()}


# =========================================================
# Basic helpers
# =========================================================
def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """將 yfinance 可能回傳的 MultiIndex 欄位壓平為單層欄位。"""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            parts = [str(x) for x in col if str(x) != ""]
            new_cols.append(parts[0])
        df.columns = new_cols
    return df


def safe_filename(ticker: str) -> str:
    """
    將 ticker 轉成適合本地存檔的檔名。
    先查特殊 mapping，沒有再做一般字元替換。
    """
    if ticker in SPECIAL_TICKER_MAP:
        return SPECIAL_TICKER_MAP[ticker]

    return (
        ticker.replace("^", "")
        .replace("/", "_")
        .replace("=", "_")
        .replace("-", "_")
    )


def restore_ticker_name(filename: str) -> str:
    """
    將本地檔名轉回 ticker。
    優先查特殊 mapping，否則只回傳檔名字串本身。
    """
    if filename in REVERSE_SPECIAL_TICKER_MAP:
        return REVERSE_SPECIAL_TICKER_MAP[filename]

    return filename


# =========================================================
# Local file discovery
# =========================================================
def list_local_tickers() -> list[str]:
    """掃描 data/raw 中現有的本地資料。"""
    if not RAW_DATA_DIR.exists():
        return []

    tickers = set()
    for path in RAW_DATA_DIR.iterdir():
        if path.suffix.lower() in [".csv", ".parquet"]:
            tickers.add(restore_ticker_name(path.stem))

    return sorted(tickers)


def get_local_file_paths(ticker: str) -> tuple[Path, Path]:
    """取得本地 CSV / Parquet 路徑。"""
    filename = safe_filename(ticker)
    csv_path = RAW_DATA_DIR / f"{filename}.csv"
    parquet_path = RAW_DATA_DIR / f"{filename}.parquet"
    return csv_path, parquet_path


# =========================================================
# Download / Load
# =========================================================
def download_price_data(
    ticker: str,
    period: str = "10y",
    interval: str = "1d",
) -> pd.DataFrame:
    """從 yfinance 下載價格資料。"""
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


def load_local_price_data(ticker: str) -> pd.DataFrame:
    """從本地 data/raw 讀取價格資料，優先讀 parquet。"""
    csv_path, parquet_path = get_local_file_paths(ticker)

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(
            f"本地找不到 {ticker} 的資料。請確認 data/raw/ 裡是否有對應檔案。"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index().dropna().copy()
    return df


def load_price_data(
    ticker: str,
    source: str = "yfinance",
    period: str = "10y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    統一資料入口：
    - source='yfinance' -> 線上抓
    - source='local'    -> 本地讀
    """
    if source == "local":
        return load_local_price_data(ticker)
    elif source == "yfinance":
        return download_price_data(ticker=ticker, period=period, interval=interval)
    else:
        raise ValueError(f"不支援的資料來源：{source}")


# =========================================================
# Validation
# =========================================================
def validate_price_data(df: pd.DataFrame) -> tuple[bool, str]:
    """檢查資料是否完整可用。"""
    if df is None or df.empty:
        return False, "抓不到資料，請確認 ticker 是否正確。"

    required_cols = ["Open", "High", "Low", "Close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, f"缺少必要欄位：{missing}"

    return True, "ok"