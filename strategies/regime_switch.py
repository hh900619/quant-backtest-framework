from __future__ import annotations

import pandas as pd


STRATEGY_OPTIONS = [
    "Trend MA",
    "Mean Reversion",
    "Breakout",
    "Flat",
]


def build_regime_switch_signal(
    regime_series: pd.Series,
    signal_library: dict[str, pd.Series],
    regime_to_strategy: dict[int, str],
) -> pd.Series:
    """
    根據每個 regime 對應的策略，組合成一條最終信號。
    """
    idx = regime_series.index
    out = pd.Series(0.0, index=idx, dtype=float)

    for regime_id, strategy_name in regime_to_strategy.items():
        mask = regime_series == regime_id

        if strategy_name == "Flat":
            out.loc[mask] = 0.0
            continue

        sig = signal_library[strategy_name].reindex(idx).fillna(0.0)
        out.loc[mask] = sig.loc[mask]

    return out.fillna(0.0)


def suggest_regime_strategy_map(
    score_table: pd.DataFrame,
    threshold: float = 0.0,
) -> dict[int, str]:
    """
    根據 regime × strategy compatibility 表，自動建議每個 regime 最適合的策略。
    如果最佳分數 <= threshold，則建議 Flat。
    """
    strategy_cols = [c for c in ["Trend MA", "Mean Reversion", "Breakout"] if c in score_table.columns]
    mapping: dict[int, str] = {}

    for regime_id, row in score_table[strategy_cols].iterrows():
        best_score = row.max()
        best_strategy = row.idxmax()

        if pd.isna(best_score) or best_score <= threshold:
            mapping[int(regime_id)] = "Flat"
        else:
            mapping[int(regime_id)] = best_strategy

    return mapping


def summarize_regime_strategy_map(
    regime_to_strategy: dict[int, str],
    label_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    把 regime 對應策略整理成表格，方便顯示。
    """
    rows = []
    for regime_id, strategy_name in sorted(regime_to_strategy.items()):
        rows.append(
            {
                "regime": regime_id,
                "label": label_map.get(regime_id, f"Regime {regime_id}") if label_map else f"Regime {regime_id}",
                "selected_strategy": strategy_name,
            }
        )

    return pd.DataFrame(rows)

def fixed_regime_strategy_map(label_map: dict[int, str]) -> dict[int, str]:
    """
    🔥 固定版策略 mapping（結構導向）
    """

    mapping = {}

    for regime_id, label in label_map.items():

        if label == "高效率趨勢上漲":
            mapping[regime_id] = "Trend MA"

        elif label == "偏多盤整":
            mapping[regime_id] = "Breakout"

        elif label == "混亂盤整":
            mapping[regime_id] = "Mean Reversion"

        elif label == "一般空頭趨勢":
            mapping[regime_id] = "Trend MA"  # 可做空

        elif label == "高波動混亂":
            mapping[regime_id] = "Flat"

        elif label == "極端崩盤":
            mapping[regime_id] = "Flat"

        else:
            mapping[regime_id] = "Flat"

    return mapping