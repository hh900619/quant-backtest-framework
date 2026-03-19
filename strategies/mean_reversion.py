from __future__ import annotations

import pandas as pd


def mean_reversion_signal(
    close: pd.Series,
    window: int = 20,
    z_entry: float = 1.0,
) -> pd.Series:
    """
    最基本的均值回歸 baseline：
    價格偏離均值太多時，反向進場。
    """
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()

    zscore = (close - mean) / std

    signal = pd.Series(0.0, index=close.index)
    signal[zscore > z_entry] = -1.0
    signal[zscore < -z_entry] = 1.0

    return signal