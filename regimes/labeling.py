from __future__ import annotations

from typing import Dict, List

import pandas as pd


# =========================================================
# Future returns
# =========================================================
def add_future_returns(
    df: pd.DataFrame,
    horizons: List[int] | None = None,
) -> pd.DataFrame:
    if horizons is None:
        horizons = [5, 10, 20]

    out = df.copy()
    for h in horizons:
        out[f"future_{h}"] = out["Close"].pct_change(h).shift(-h)

    return out


# =========================================================
# Regime summary
# =========================================================
def compute_regime_summary(
    df: pd.DataFrame,
    horizons: List[int] | None = None,
) -> pd.DataFrame:
    if horizons is None:
        horizons = [5, 10, 20]

    base_cols = [
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
        "efficiency",
        "bias",
        "trend_strength",
        "chop_score",
        "structure_purity",
        "persistence_score",
        "vol_stability",
        "trend_break_risk",
    ]
    future_cols = [f"future_{h}" for h in horizons]
    cols = [c for c in base_cols + future_cols if c in df.columns]

    grouped = df.dropna(subset=["regime"]).groupby("regime")
    summary = grouped[cols].mean()

    count = grouped.size().rename("count")
    ratio = (count / count.sum()).rename("ratio")

    summary = summary.join(count).join(ratio)
    return summary


# =========================================================
# Fixed Label Engine (Phase 1 正式版)
# =========================================================
def auto_label_regimes(regime_summary: pd.DataFrame) -> Dict[int, str]:
    """
    Phase 1 正式版 label engine。

    這版故意不用 median / quantile 動態判斷，
    也不用雙層 structure-direction 系統，
    目的是保留一套穩定、可解釋、可重現的固定規則。

    目前對應的市場語言：
    - 高效率趨勢上漲
    - 偏多盤整
    - 混亂盤整
    - 高波動混亂
    - 一般空頭趨勢
    - 極端崩盤
    """
    label_map: Dict[int, str] = {}

    for regime_id, row in regime_summary.iterrows():
        ret = row.get("ret_mid", 0.0)
        vol = row.get("vol", 0.0)
        trend = row.get("trend_r2", 0.0)
        purity = row.get("structure_purity", 0.0)
        flip = row.get("flip_rate", 0.0)

        # 1. 極端崩盤：跌幅很深 + 波動極大
        if ret < -0.06 and vol > 0.03:
            label = "極端崩盤"

        # 2. 高波動混亂：高波動 + 高切換
        elif vol > 0.035 and flip > 0.6:
            label = "高波動混亂"

        # 3. 一般空頭趨勢：中度負報酬 + 趨勢結構明顯
        elif ret < -0.015 and trend > 0.5:
            label = "一般空頭趨勢"

        # 4. 高效率趨勢上漲：正報酬明顯 + 結構純度高
        elif ret > 0.03 and purity > 1.8:
            label = "高效率趨勢上漲"

        # 5. 偏多盤整：有些偏多，但不是乾淨主升
        elif ret > 0.015 and trend > 0.5:
            label = "偏多盤整"

        # 6. 其他預設為混亂盤整
        else:
            label = "混亂盤整"

        label_map[int(regime_id)] = label

    return label_map


# =========================================================
# Apply labels to dataframe
# =========================================================
def apply_regime_labels(
    df: pd.DataFrame,
    label_map: Dict[int, str],
    auto_label_map: Dict[int, str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["regime_name"] = out["regime"].map(label_map)

    if auto_label_map is not None:
        out["regime_auto_name"] = out["regime"].map(auto_label_map)

    return out


# =========================================================
# Transition matrix
# =========================================================
def compute_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.dropna(subset=["regime"]).copy()
    temp["next_regime"] = temp["regime"].shift(-1)

    trans = pd.crosstab(
        temp["regime"],
        temp["next_regime"],
        normalize="index",
    )

    return trans