from __future__ import annotations

import pandas as pd


def moving_average_signal(close: pd.Series, ma_window: int = 50) -> pd.Series:
    """
    最基本的順勢 baseline：
    價格在均線上方做多，在均線下方做空。
    """
    ma = close.rolling(ma_window).mean()

    signal = pd.Series(0.0, index=close.index)
    signal[close > ma] = 1.0
    signal[close < ma] = -1.0

    return signal


def breakout_signal(close: pd.Series, window: int = 20) -> pd.Series:
    """
    最基本的突破 baseline：
    突破過去 N 期高點做多，跌破過去 N 期低點做空。
    """
    rolling_high = close.rolling(window).max().shift(1)
    rolling_low = close.rolling(window).min().shift(1)

    signal = pd.Series(0.0, index=close.index)
    signal[close > rolling_high] = 1.0
    signal[close < rolling_low] = -1.0

    return signal