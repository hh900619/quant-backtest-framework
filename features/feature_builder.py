from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    ret_window: int = 20
    vol_window: int = 20
    range_window: int = 20
    trend_window: int = 20
    up_ratio_window: int = 20
    short_window: int = 5
    long_window: int = 20


# =========================================================
# Feature groups
# =========================================================
BASE_FEATURES = [
    "ret_mid",
    "vol",
    "range",
    "trend_r2",
    "up_ratio",
    "accel",
    "dist_ma",
    "drawdown",
    "flip_rate",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
]

CUSTOM_FEATURES = [
    "efficiency",
    "bias",
    "trend_strength",
    "chop_score",
    "structure_purity",
    "persistence_score",
    "vol_stability",
    "trend_break_risk",
]


def get_base_feature_columns() -> list[str]:
    return BASE_FEATURES.copy()


def get_custom_feature_columns() -> list[str]:
    return CUSTOM_FEATURES.copy()


def get_feature_columns() -> list[str]:
    return BASE_FEATURES + CUSTOM_FEATURES


# =========================================================
# Internal helpers
# =========================================================
def calc_rolling_r2(series: pd.Series, window: int) -> pd.Series:
    def _calc(x):
        if len(x) < 2:
            return np.nan
        y = x.values
        x_idx = np.arange(len(y))
        slope, intercept = np.polyfit(x_idx, y, 1)
        y_pred = slope * x_idx + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return series.rolling(window).apply(_calc, raw=False)


# =========================================================
# Main builder
# =========================================================
def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()

    # -----------------------------------------------------
    # Base market data transforms
    # -----------------------------------------------------
    out["ret"] = out["Close"].pct_change()
    out["ret_mid"] = out["Close"].pct_change(cfg.ret_window)

    # volatility
    out["vol"] = out["ret"].rolling(cfg.vol_window).std()

    # daily / bar range
    out["range"] = (out["High"] - out["Low"]) / out["Close"]

    # trend fit quality
    out["trend_r2"] = calc_rolling_r2(out["Close"], cfg.trend_window)

    # bullish / bearish bias from bar direction ratio
    out["up_ratio"] = (out["ret"] > 0).rolling(cfg.up_ratio_window).mean()

    # acceleration of return
    out["accel"] = out["ret"].diff()

    # distance from long moving average
    ma = out["Close"].rolling(cfg.long_window).mean()
    out["dist_ma"] = (out["Close"] - ma) / ma

    # drawdown from running peak
    rolling_max = out["Close"].cummax()
    out["drawdown"] = (out["Close"] - rolling_max) / rolling_max

    # flip rate: how often return direction changes
    direction = np.sign(out["ret"])
    out["flip_rate"] = (direction != direction.shift()).rolling(cfg.ret_window).mean()

    # candle structure
    body = (out["Close"] - out["Open"]).abs()
    total_range = (out["High"] - out["Low"]) + 1e-6

    out["body_ratio"] = body / total_range
    out["upper_shadow_ratio"] = (out["High"] - out[["Open", "Close"]].max(axis=1)) / total_range
    out["lower_shadow_ratio"] = (out[["Open", "Close"]].min(axis=1) - out["Low"]) / total_range

    # -----------------------------------------------------
    # Your custom market-language features
    # -----------------------------------------------------

    # 1) 急 vs 緩（效率）
    out["efficiency"] = out["ret_mid"].abs() / (out["vol"] + 1e-6)

    # 2) 偏多 / 偏空
    out["bias"] = out["up_ratio"] - 0.5

    # 3) 趨勢強度
    out["trend_strength"] = out["trend_r2"] * out["ret_mid"].abs()

    # 4) 亂盤程度 / chop
    out["chop_score"] = out["flip_rate"] * out["vol"]

    # 5) 結構純度
    out["structure_purity"] = out["trend_r2"] / (out["flip_rate"] + 1e-6)

    # -----------------------------------------------------
    # New stabilizing / refining features
    # -----------------------------------------------------

    # 6) persistence_score
    # 趨勢強度不是只看當下，而是看最近一段時間是否持續存在
    out["persistence_score"] = out["trend_strength"].rolling(cfg.short_window).mean()

    # 7) vol_stability
    # 波動本身是否穩定，越高代表波動結構不穩
    out["vol_stability"] = out["vol"].rolling(cfg.short_window).std()

    # 8) trend_break_risk
    # 趨勢強度是否正在破裂，越負代表趨勢在衰退
    out["trend_break_risk"] = out["trend_strength"].diff()

    return out


# =========================================================
# Matrix prep
# =========================================================
def prepare_feature_matrix(df: pd.DataFrame, selected_cols: list[str]) -> pd.DataFrame:
    mat = df[selected_cols].copy()
    mat = mat.replace([np.inf, -np.inf], np.nan).dropna()
    return mat