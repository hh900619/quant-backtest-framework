from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_signal(close: pd.Series, signal: pd.Series) -> pd.DataFrame:
    """
    根據 signal 做最基本的回測。
    signal:
        1 = long
       -1 = short
        0 = flat
    """
    ret = close.pct_change().fillna(0.0)
    position = signal.shift(1).fillna(0.0)
    strategy_ret = position * ret
    equity = (1.0 + strategy_ret).cumprod()

    out = pd.DataFrame(
        {
            "ret": ret,
            "position": position,
            "strategy_ret": strategy_ret,
            "equity": equity,
        },
        index=close.index,
    )

    return out


def performance_stats(bt: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """
    計算基本績效指標。
    """
    r = bt["strategy_ret"].dropna()

    if len(r) == 0:
        return {
            "total_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_dd": np.nan,
        }

    total_return = bt["equity"].iloc[-1] - 1.0

    if len(r) > 1:
        ann_return = (1.0 + total_return) ** (periods_per_year / len(r)) - 1.0
    else:
        ann_return = np.nan

    ann_vol = r.std() * np.sqrt(periods_per_year)

    if r.std() != 0:
        sharpe = r.mean() / r.std() * np.sqrt(periods_per_year)
    else:
        sharpe = np.nan

    rolling_max = bt["equity"].cummax()
    drawdown = bt["equity"] / rolling_max - 1.0
    max_dd = drawdown.min()

    return {
        "total_return": float(total_return),
        "ann_return": float(ann_return) if pd.notna(ann_return) else np.nan,
        "ann_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_dd": float(max_dd) if pd.notna(max_dd) else np.nan,
    }


def regime_strategy_table(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    """
    看某個策略在不同 regime 下的表現差異。
    """
    temp = df.copy()
    temp["signal"] = signal.shift(1)
    temp["ret"] = temp["Close"].pct_change()
    temp["strategy_ret"] = temp["signal"] * temp["ret"]

    table = (
        temp.dropna(subset=["regime"])
        .groupby("regime")["strategy_ret"]
        .agg(["mean", "std", "count"])
    )

    table["sharpe_like"] = np.where(
        table["std"] != 0,
        table["mean"] / table["std"] * np.sqrt(252),
        np.nan,
    )

    return table