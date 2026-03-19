import numpy as np


def compute_shock_score(df):
    """
    🔥 市場異常偵測分數（Shock Score）
    """

    score = (
        (df["vol"] > df["vol"].rolling(50).mean() * 1.5).astype(int)
        + (df["flip_rate"] > 0.6).astype(int)
        + (df["drawdown"] < -0.1).astype(int)
        + (abs(df["accel"]) > 0.002).astype(int)
        + (abs(df["dist_ma"]) > 0.05).astype(int)
    )

    return score


def risk_override(signal_series, shock_score, threshold=2):
    """
    🔥 風控覆蓋層（Risk Override）
    shock_score 過高 → 強制 Flat
    """

    override_signal = signal_series.copy()

    mask = shock_score >= threshold
    override_signal[mask] = 0

    return override_signal