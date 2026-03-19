from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.loader import list_local_tickers, load_price_data, validate_price_data
from features.feature_builder import (
    FeatureConfig,
    build_features,
    get_base_feature_columns,
    get_custom_feature_columns,
    get_feature_columns,
    prepare_feature_matrix,
)
from metrics.performance import regime_strategy_table
from regimes.cluster import (
    assign_regimes,
    apply_advanced_regime_smoothing,
    fit_kmeans,
    predict_kmeans,
    get_cluster_quality,
    get_regime_counts,
)
from regimes.labeling import (
    add_future_returns,
    apply_regime_labels,
    auto_label_regimes,
    compute_regime_summary,
    compute_transition_matrix,
)
from strategies.mean_reversion import mean_reversion_signal
from strategies.trend import breakout_signal, moving_average_signal
from utils.risk import compute_shock_score, risk_override

warnings.filterwarnings("ignore")


# =========================================================
# 中文顯示
# =========================================================
COLUMN_LABELS = {
    "Open": "開盤價 (Open)",
    "High": "最高價 (High)",
    "Low": "最低價 (Low)",
    "Close": "收盤價 (Close)",
    "Volume": "成交量 (Volume)",
    "ret": "單期報酬 (ret)",
    "ret_mid": "中期報酬 (ret_mid)",
    "vol": "波動度 (vol)",
    "range": "平均振幅 (range)",
    "trend_r2": "趨勢穩定度 (trend_r2)",
    "up_ratio": "上漲K比例 (up_ratio)",
    "accel": "動能加速度 (accel)",
    "dist_ma": "價格偏離均線 (dist_ma)",
    "drawdown": "回撤幅度 (drawdown)",
    "flip_rate": "方向切換頻率 (flip_rate)",
    "body_ratio": "K棒實體比例 (body_ratio)",
    "upper_shadow_ratio": "上影線比例 (upper_shadow_ratio)",
    "lower_shadow_ratio": "下影線比例 (lower_shadow_ratio)",
    "efficiency": "效率分數 (efficiency)",
    "bias": "偏向分數 (bias)",
    "trend_strength": "趨勢強度 (trend_strength)",
    "chop_score": "混亂分數 (chop_score)",
    "structure_purity": "結構純度 (structure_purity)",
    "persistence_score": "持續性分數 (persistence_score)",
    "vol_stability": "波動穩定性 (vol_stability)",
    "trend_break_risk": "趨勢破裂風險 (trend_break_risk)",
    "future_5": "未來5期報酬 (future_5)",
    "future_10": "未來10期報酬 (future_10)",
    "future_20": "未來20期報酬 (future_20)",
    "count": "出現次數 (count)",
    "ratio": "出現比例 (ratio)",
    "label": "手動名稱 (label)",
    "auto_label": "自動名稱 (auto_label)",
    "regime": "狀態編號 (regime)",
    "regime_raw": "原始狀態 (regime_raw)",
    "regime_smoothed": "平滑後狀態 (regime_smoothed)",
    "regime_name": "市場狀態名稱 (regime_name)",
    "regime_auto_name": "自動判定名稱 (regime_auto_name)",
    "total_return": "總報酬 (total_return)",
    "ann_return": "年化報酬 (ann_return)",
    "ann_vol": "年化波動 (ann_vol)",
    "sharpe": "夏普值 (sharpe)",
    "max_dd": "最大回撤 (max_dd)",
    "win_rate": "勝率 (win_rate)",
    "profit_factor": "獲利因子 (profit_factor)",
    "avg_trade": "平均每筆報酬 (avg_trade)",
    "trade_count": "開倉次數 (trade_count)",
    "turnover_count": "交易變動次數 (turnover_count)",
    "selected_strategy": "選用策略 (selected_strategy)",
}

COLUMN_EXPLAIN = {
    "中期報酬 (ret_mid)": "看中期方向。正值代表偏上漲，負值代表偏下跌。",
    "波動度 (vol)": "看市場劇烈程度。越高代表越亂、越危險。",
    "平均振幅 (range)": "看K棒平均大小。越高代表每天震盪越大。",
    "趨勢穩定度 (trend_r2)": "看趨勢是否乾淨。越接近1越像穩定趨勢，越接近0越亂。",
    "上漲K比例 (up_ratio)": "看多空主導程度。高代表多方主導，低代表空方主導。",
    "價格偏離均線 (dist_ma)": "看價格距離均線有多遠，可判斷過熱或過弱。",
    "回撤幅度 (drawdown)": "看目前距離近期高點跌了多少，越負代表回撤越深。",
    "方向切換頻率 (flip_rate)": "看市場是不是上下亂切，越高通常越像盤整。",
    "效率分數 (efficiency)": "同樣漲跌幅下，波動越小，效率越高。",
    "偏向分數 (bias)": "盤整或震盪時偏多還是偏空。大於0偏多，小於0偏空。",
    "趨勢強度 (trend_strength)": "方向性與趨勢穩定度的組合。越高代表越像有效趨勢。",
    "混亂分數 (chop_score)": "波動與方向切換的組合。越高通常越像亂盤、假突破環境。",
    "結構純度 (structure_purity)": "趨勢穩定度相對於亂度。越高代表結構更乾淨。",
    "持續性分數 (persistence_score)": "趨勢強度在最近一段時間是否持續存在。",
    "波動穩定性 (vol_stability)": "波動本身穩不穩。越高通常表示市場狀態更不穩。",
    "趨勢破裂風險 (trend_break_risk)": "趨勢強度是否開始衰退。越負通常代表趨勢正在破裂。",
    "總報酬 (total_return)": "整段回測最終總報酬。",
    "年化報酬 (ann_return)": "把整體報酬換算成每年平均表現。",
    "年化波動 (ann_vol)": "策略每年的波動程度。",
    "夏普值 (sharpe)": "風險調整後報酬，通常越高越好。",
    "最大回撤 (max_dd)": "從高點跌下來最深的一次幅度。",
    "勝率 (win_rate)": "獲利交易比例。低勝率不一定壞，但通常需要高盈虧比支撐。",
    "獲利因子 (profit_factor)": "總獲利 / 總虧損。大於1代表有正優勢，越高越好。",
    "平均每筆報酬 (avg_trade)": "每筆交易平均期望值。",
    "開倉次數 (trade_count)": "從空手到持倉的次數。",
    "交易變動次數 (turnover_count)": "包含開倉、平倉、反手等所有倉位變動次數。",
}


def zh_name(col: str) -> str:
    return COLUMN_LABELS.get(col, col)


def rename_columns_zh(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [zh_name(c) for c in out.columns]
    return out


def rename_index_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out.index = [f"Regime {int(x)}" for x in out.index]
    except Exception:
        pass
    return out


def explain_columns(cols: list[str]):
    lines = []
    for c in cols:
        c_zh = zh_name(c)
        if c_zh in COLUMN_EXPLAIN:
            lines.append(f"- **{c_zh}**：{COLUMN_EXPLAIN[c_zh]}")
    if lines:
        st.info("\n".join(lines))


# =========================================================
# Helpers
# =========================================================
def structural_regime_strategy_map(label_map: dict[int, str]) -> dict[int, str]:
    """
    Phase 1 固定 mapping。
    保留有結構意義的 baseline，不追求當前樣本最佳績效。
    """
    mapping: dict[int, str] = {}

    for regime_id, label in label_map.items():
        if label == "高效率趨勢上漲":
            mapping[regime_id] = "Trend MA"
        elif label == "偏多盤整":
            mapping[regime_id] = "Breakout"
        elif label == "混亂盤整":
            mapping[regime_id] = "Flat"
        elif label == "一般空頭趨勢":
            mapping[regime_id] = "Flat"
        elif label == "高波動混亂":
            mapping[regime_id] = "Flat"
        elif label == "極端崩盤":
            mapping[regime_id] = "Flat"
        else:
            mapping[regime_id] = "Flat"

    return mapping


def build_regime_switch_signal(
    regime_series: pd.Series,
    signal_library: dict[str, pd.Series],
    regime_to_strategy: dict[int, str],
) -> pd.Series:
    idx = regime_series.index
    out = pd.Series(0.0, index=idx, dtype=float)

    zero_sig = pd.Series(0.0, index=idx, dtype=float)
    local_library = {
        "Trend MA": signal_library["Trend MA"].reindex(idx).fillna(0.0),
        "Mean Reversion": signal_library["Mean Reversion"].reindex(idx).fillna(0.0),
        "Breakout": signal_library["Breakout"].reindex(idx).fillna(0.0),
        "Flat": zero_sig,
    }

    for regime_id, strategy_name in regime_to_strategy.items():
        mask = regime_series == regime_id
        sig = local_library.get(strategy_name, zero_sig)
        out.loc[mask] = sig.loc[mask]

    return out.fillna(0.0)


def backtest_with_costs(
    close: pd.Series,
    signal: pd.Series,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    close = close.astype(float)
    signal = signal.astype(float).reindex(close.index).fillna(0.0)

    asset_ret = close.pct_change().fillna(0.0)
    position = signal.shift(1).fillna(0.0)

    gross_ret = position * asset_ret

    turnover = signal.diff().abs().fillna(signal.abs())
    cost_rate = cost_bps / 10000.0
    cost = turnover * cost_rate

    net_ret = gross_ret - cost
    equity = (1.0 + net_ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    out = pd.DataFrame(
        {
            "close": close,
            "signal": signal,
            "position": position,
            "asset_ret": asset_ret,
            "gross_ret": gross_ret,
            "cost": cost,
            "returns": net_ret,
            "equity": equity,
            "drawdown": drawdown,
            "turnover": turnover,
        },
        index=close.index,
    )
    return out


def performance_stats_local(bt: pd.DataFrame) -> dict[str, float]:
    ret = bt["returns"].dropna()
    if ret.empty:
        return {
            "total_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
        }

    total_return = float(bt["equity"].iloc[-1] - 1.0)
    ann_return = float((1.0 + ret.mean()) ** 252 - 1.0)
    ann_vol = float(ret.std(ddof=0) * np.sqrt(252))
    sharpe = float(ret.mean() / ret.std(ddof=0) * np.sqrt(252)) if ret.std(ddof=0) != 0 else 0.0
    max_dd = float(bt["drawdown"].min())

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def trade_quality_stats(bt: pd.DataFrame) -> dict[str, float]:
    ret = bt["returns"].dropna()
    if ret.empty:
        return {
            "win_rate": 0.0,
            "rr": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "trade_count": 0,
            "turnover_count": 0,
        }

    win_rate = float((ret > 0).mean())
    avg_win = ret[ret > 0].mean() if (ret > 0).any() else 0.0
    avg_loss = ret[ret < 0].mean() if (ret < 0).any() else 0.0
    rr = float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0

    gross_profit = ret[ret > 0].sum()
    gross_loss = ret[ret < 0].sum()
    profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss != 0 else 0.0

    avg_trade = float(ret.mean())

    signal = bt["signal"].fillna(0.0)
    trade_count = int(((signal.shift(1) == 0) & (signal != 0)).sum())
    turnover_count = int((signal.diff().abs() > 0).sum())

    return {
        "win_rate": win_rate,
        "rr": rr,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "trade_count": trade_count,
        "turnover_count": turnover_count,
    }


def evaluation_report_text(stats: dict[str, float]) -> list[str]:
    lines = []

    if stats["win_rate"] > 0.55:
        lines.append("✔ 勝率偏高：系統穩定，但也要注意是否過度保守。")
    elif stats["win_rate"] > 0.4:
        lines.append("✔ 勝率中性：屬於健康區間。")
    else:
        lines.append("⚠ 勝率偏低：需要較高的盈虧比或更厚的 edge 支撐。")

    if stats["rr"] > 2.0:
        lines.append("🔥 盈虧比很強：少數勝利交易就能覆蓋虧損。")
    elif stats["rr"] > 1.2:
        lines.append("✔ 盈虧比尚可：具備基本趨勢策略特徵。")
    else:
        lines.append("⚠ 盈虧比偏低：代表贏的交易還不夠大。")

    if stats["profit_factor"] > 2.0:
        lines.append("🔥 Profit Factor 很強：策略優勢相對明確。")
    elif stats["profit_factor"] > 1.5:
        lines.append("✔ Profit Factor 良好：有實盤研究價值。")
    elif stats["profit_factor"] > 1.2:
        lines.append("⚠ Profit Factor 普通：有正優勢，但很薄。")
    else:
        lines.append("❌ Profit Factor 過低：加上交易成本後容易失去優勢。")

    if stats["avg_trade"] > 0:
        lines.append("✔ 平均每筆交易為正期望值。")
    else:
        lines.append("❌ 平均每筆交易為負期望值。")

    return lines


def summarize_regime_strategy_map(
    regime_to_strategy: dict[int, str],
    label_map: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for regime_id, strategy_name in sorted(regime_to_strategy.items()):
        rows.append(
            {
                "regime": regime_id,
                "label": label_map.get(regime_id, f"Regime {regime_id}"),
                "selected_strategy": strategy_name,
            }
        )
    return pd.DataFrame(rows)


def split_in_out_sample_index(index: pd.DatetimeIndex, test_years: int = 3) -> pd.Timestamp:
    end_date = index.max()
    split_date = end_date - pd.DateOffset(years=test_years)
    return pd.Timestamp(split_date)


def plot_equity_interactive(bt_map: Dict[str, pd.DataFrame], title: str):
    fig = go.Figure()
    for name, bt in bt_map.items():
        if "equity" not in bt.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=bt.index,
                y=bt["equity"],
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis_title="Equity",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_price_interactive(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=f"{ticker} Close",
            line=dict(width=1.5),
        )
    )
    fig.update_layout(
        title=f"{ticker} Price（價格走勢）",
        xaxis_title="Date",
        yaxis_title="Close",
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_price_with_regime_interactive(df: pd.DataFrame, ticker: str):
    plot_df = df.dropna(subset=["regime"]).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["Close"],
            mode="lines",
            name="Close",
            line=dict(width=1.2, color="white"),
        )
    )

    unique_regimes = sorted(plot_df["regime"].dropna().unique())
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    for i, regime in enumerate(unique_regimes):
        sub = plot_df[plot_df["regime"] == regime]
        fig.add_trace(
            go.Scatter(
                x=sub.index,
                y=sub["Close"],
                mode="markers",
                name=f"Regime {int(regime)}",
                marker=dict(size=6, color=colors[i % len(colors)]),
                text=sub["regime_name"] if "regime_name" in sub.columns else None,
                hovertemplate=(
                    "Date=%{x}<br>"
                    "Close=%{y:.2f}<br>"
                    "Regime=" + str(int(regime)) + "<br>"
                    "Name=%{text}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{ticker} Price + Regime（價格 + 市場狀態）",
        template="plotly_dark",
        height=650,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis_title="Close",
    )
    st.plotly_chart(fig, use_container_width=True)


def fit_and_assign_pure_regimes(
    feature_df: pd.DataFrame,
    selected_features: list[str],
    n_clusters: int,
    split_date: pd.Timestamp,
    use_smoothing: bool,
    majority_window: int,
    confirm_bars: int,
    min_run_length: int,
) -> tuple[pd.DataFrame, dict, dict, object, object]:
    """
    真正 Phase 1 純化版本：
    - 只用 in-sample fit scaler + kmeans
    - 用同一個 scaler/model 去 assign OOS
    - smoothing 用 causal 版本，可套在整條時間軸
    - label 預設用 in-sample summary 產生
    """
    in_feature_df = feature_df[feature_df.index < split_date].copy()
    oos_feature_df = feature_df[feature_df.index >= split_date].copy()

    in_matrix = prepare_feature_matrix(in_feature_df, selected_features)
    if in_matrix.empty:
        raise ValueError("In-sample feature matrix 為空，請調整期間或 feature。")

    labels_in, scaler, model = fit_kmeans(in_matrix, n_clusters=n_clusters)

    in_regime_df = assign_regimes(in_feature_df, in_matrix, labels_in)

    oos_matrix = prepare_feature_matrix(oos_feature_df, selected_features)
    if not oos_matrix.empty:
        labels_oos = predict_kmeans(oos_matrix, scaler, model)
        oos_regime_df = assign_regimes(oos_feature_df, oos_matrix, labels_oos)
    else:
        oos_regime_df = oos_feature_df.copy()
        oos_regime_df["regime"] = pd.Series(dtype="float64")

    regime_df = pd.concat([in_regime_df, oos_regime_df], axis=0).sort_index()
    regime_df["regime_raw"] = regime_df["regime"]

    if use_smoothing:
        regime_df = apply_advanced_regime_smoothing(
            regime_df,
            source_col="regime_raw",
            target_col="regime_smoothed",
            majority_window=majority_window,
            confirm_bars=confirm_bars,
            min_run_length=min_run_length,
        )
        active_regime_col = "regime_smoothed"
    else:
        regime_df["regime_smoothed"] = regime_df["regime_raw"]
        active_regime_col = "regime_raw"

    regime_df["regime"] = regime_df[active_regime_col]
    regime_df = add_future_returns(regime_df, horizons=[5, 10, 20])

    # 品質指標只看 in-sample fit
    quality = get_cluster_quality(in_matrix, labels_in, model)

    # in-sample summary 生成 auto labels（Phase 1 純化）
    regime_df_is = regime_df[regime_df.index < split_date].copy()
    regime_df_is["regime"] = regime_df_is[active_regime_col]
    in_summary = compute_regime_summary(regime_df_is, horizons=[5, 10, 20])
    auto_label_map = auto_label_regimes(in_summary)

    return regime_df, quality, auto_label_map, scaler, model


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Regime Research Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Regime Research Dashboard（市場狀態研究儀表板）")
st.caption("Phase 1 修正版：修正 OOS purity、保留 regime / label / shock / costs。")


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("研究設定 Research Settings")

page = st.sidebar.radio(
    "選擇頁面 Select Page",
    ["Data", "Features", "Regimes", "Analysis", "Strategy Lab"],
)

st.sidebar.subheader("資料來源 Data Source")
data_source = st.sidebar.selectbox("來源 Source", ["yfinance", "local"], index=0)

local_tickers = list_local_tickers()
if data_source == "local" and local_tickers:
    ticker = st.sidebar.selectbox("本地資料 Local Data", local_tickers, index=0)
else:
    ticker = st.sidebar.text_input("Ticker", value="SPY")

period = st.sidebar.selectbox("期間 Period", ["5y", "10y", "15y", "max"], index=1)
interval = st.sidebar.selectbox("週期 Interval", ["1d", "1wk"], index=0)

st.sidebar.subheader("Feature 視窗 Feature Windows")
ret_window = st.sidebar.slider("ret_window", 10, 120, 20)
vol_window = st.sidebar.slider("vol_window", 10, 120, 20)
range_window = st.sidebar.slider("range_window", 10, 120, 20)
trend_window = st.sidebar.slider("trend_window", 10, 120, 20)
up_ratio_window = st.sidebar.slider("up_ratio_window", 10, 120, 20)
short_window = st.sidebar.slider("short_window", 3, 30, 5)
long_window = st.sidebar.slider("long_window", 10, 120, 20)

st.sidebar.subheader("Regime 模型 Model")
n_clusters = st.sidebar.slider("群數 KMeans Clusters", 3, 8, 6)

st.sidebar.subheader("Feature 模式 Feature Mode")
feature_mode = st.sidebar.radio(
    "選擇分群模式",
    ["只用自訂特徵 Custom Only", "只用基礎特徵 Base Only", "混合特徵 Mixed"],
    index=0,
)

if feature_mode == "只用自訂特徵 Custom Only":
    default_features = get_custom_feature_columns()
elif feature_mode == "只用基礎特徵 Base Only":
    default_features = [
        "ret_mid",
        "vol",
        "range",
        "trend_r2",
        "up_ratio",
        "accel",
        "dist_ma",
        "drawdown",
        "flip_rate",
    ]
else:
    default_features = [
        "ret_mid",
        "vol",
        "trend_r2",
        "up_ratio",
        "efficiency",
        "bias",
        "trend_strength",
        "chop_score",
        "structure_purity",
        "persistence_score",
        "vol_stability",
        "trend_break_risk",
    ]

selected_features = st.sidebar.multiselect(
    "分群特徵 Selected Features",
    get_feature_columns(),
    default=default_features,
)

st.sidebar.subheader("Regime 穩定化 Smoothing")
use_smoothing = st.sidebar.checkbox("啟用高階平滑 Enable Advanced Smoothing", value=True)
majority_window = st.sidebar.slider("Majority Window", 1, 11, 5, step=2)
confirm_bars = st.sidebar.slider("Confirm Bars", 1, 5, 3, step=1)
min_run_length = st.sidebar.slider("Minimum Run Length", 1, 5, 3, step=1)

st.sidebar.subheader("驗證設定 Validation")
cost_bps = st.sidebar.slider("單位交易成本（bps）", 0.0, 20.0, 5.0, 0.5)
use_shock = st.sidebar.checkbox("啟用 Shock Filter", value=True)
shock_threshold = st.sidebar.slider("Shock Threshold", 1, 5, 2, step=1)
test_years = st.sidebar.slider("Out-of-sample 年數", 1, 5, 3, step=1)

cfg = FeatureConfig(
    ret_window=ret_window,
    vol_window=vol_window,
    range_window=range_window,
    trend_window=trend_window,
    up_ratio_window=up_ratio_window,
    short_window=short_window,
    long_window=long_window,
)


# =========================================================
# Main pipeline
# =========================================================
@st.cache_data(show_spinner=False)
def load_data_cached(
    ticker: str,
    source: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    return load_price_data(
        ticker=ticker,
        source=source,
        period=period,
        interval=interval,
    )


try:
    raw_df = load_data_cached(ticker, data_source, period, interval)
except Exception as e:
    st.error(f"下載或讀取資料失敗 Download/Load Failed：{e}")
    st.stop()

ok, msg = validate_price_data(raw_df)
if not ok:
    st.warning(msg)
    st.stop()

if len(selected_features) == 0:
    st.warning("至少要選一個 feature 才能分群。")
    st.stop()

feature_df = build_features(raw_df, cfg)
if feature_df.empty:
    st.warning("Feature dataframe 為空。")
    st.stop()

split_date = split_in_out_sample_index(feature_df.index, test_years=test_years)

try:
    regime_df, quality, auto_label_map, scaler, model = fit_and_assign_pure_regimes(
        feature_df=feature_df,
        selected_features=selected_features,
        n_clusters=n_clusters,
        split_date=split_date,
        use_smoothing=use_smoothing,
        majority_window=majority_window,
        confirm_bars=confirm_bars,
        min_run_length=min_run_length,
    )
except Exception as e:
    st.error(f"Regime 建立失敗：{e}")
    st.stop()

# 給整段 timeline 都套 label，但 label 預設來自 in-sample
st.sidebar.subheader("Regime 命名 Manual Naming")
manual_label_map = {}
existing_regimes = sorted([int(x) for x in regime_df["regime"].dropna().unique()])

for regime_id in existing_regimes:
    default_name = auto_label_map.get(regime_id, f"Regime {regime_id}")
    manual_label_map[regime_id] = st.sidebar.text_input(
        f"Regime {regime_id} 名稱",
        value=default_name,
        key=f"regime_name_{regime_id}",
    )

label_map = manual_label_map
regime_df = apply_regime_labels(regime_df, label_map, auto_label_map=auto_label_map)

# Summary：顯示 full-sample，但 label 預設來源是 IS
regime_summary = compute_regime_summary(regime_df, horizons=[5, 10, 20])
regime_summary["auto_label"] = regime_summary.index.map(auto_label_map)
regime_summary["label"] = regime_summary.index.map(label_map)

transition_matrix = compute_transition_matrix(regime_df)


# =========================================================
# Pages
# =========================================================
if page == "Data":
    st.header("📊 Data（原始市場資料）")
    st.info("這一頁只做資料檢查，不做績效解讀。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("資料筆數 Rows", len(raw_df))
    c2.metric("開始日期 Start", str(raw_df.index.min().date()))
    c3.metric("結束日期 End", str(raw_df.index.max().date()))
    c4.metric("最新收盤價 Last Close", f"{raw_df['Close'].iloc[-1]:.2f}")

    st.subheader("價格資料預覽 Raw Data Preview")
    st.dataframe(rename_columns_zh(raw_df.tail(20)), use_container_width=True)

    st.subheader("互動價格圖 Interactive Price Chart")
    plot_price_interactive(raw_df, ticker)

elif page == "Features":
    st.header("⚙️ Features（市場結構特徵）")
    st.info("這一頁用來確認你選的特徵是否有解釋力。")

    st.subheader("目前拿去分群的特徵 Selected Features")
    st.write(selected_features)
    explain_columns(selected_features)

    custom_show = [c for c in get_custom_feature_columns() if c in feature_df.columns]
    base_show = [c for c in get_base_feature_columns() if c in feature_df.columns]

    st.subheader("Custom Features（你的市場語言）")
    st.dataframe(rename_columns_zh(feature_df[custom_show].tail(30)), use_container_width=True)

    st.subheader("Base Features（基礎市場特徵）")
    st.dataframe(rename_columns_zh(feature_df[base_show].tail(30)), use_container_width=True)

    st.subheader("Custom Feature Summary")
    st.dataframe(rename_columns_zh(feature_df[custom_show].describe().T), use_container_width=True)

    st.subheader("Base Feature Summary")
    st.dataframe(rename_columns_zh(feature_df[base_show].describe().T), use_container_width=True)

elif page == "Regimes":
    st.header("🧠 Regimes（市場狀態分類）")
    st.info(
        "這版 regime 是真正 OOS-safe 的：\n"
        "只用 in-sample fit KMeans / scaler，再對 OOS assign。"
    )

    active_regime_col = "regime_smoothed" if use_smoothing else "regime_raw"

    st.caption(
        f"目前顯示欄位：{active_regime_col} | "
        f"split_date={split_date.date()} | "
        f"majority_window={majority_window}, confirm_bars={confirm_bars}, min_run_length={min_run_length}"
    )

    c0, c1, c2 = st.columns(3)
    c0.metric("Clusters", n_clusters)
    c1.metric("Inertia (IS fit)", f"{quality['inertia']:.2f}")
    c2.metric("Silhouette (IS fit)", f"{quality['silhouette']:.4f}" if pd.notna(quality["silhouette"]) else "N/A")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Regime Summary")
        display_summary = rename_columns_zh(rename_index_regime(regime_summary.copy()))
        st.dataframe(display_summary, use_container_width=True)

    with c4:
        st.subheader("Regime Counts")
        counts = get_regime_counts(regime_df, regime_col="regime")
        counts.index = [f"Regime {int(x)}" for x in counts.index]
        st.bar_chart(counts)

    st.subheader("Price + Regime")
    plot_price_with_regime_interactive(regime_df, ticker)

    st.subheader("Recent Regime Rows")
    recent_cols = ["Close", "regime_raw", "regime_smoothed", "regime_name", "regime_auto_name"]
    recent_cols = [c for c in recent_cols if c in regime_df.columns]
    recent_regime = regime_df.dropna(subset=["regime"])[recent_cols].tail(20)
    st.dataframe(rename_columns_zh(recent_regime), use_container_width=True)

elif page == "Analysis":
    st.header("📈 Analysis（狀態分析）")
    st.info("這一頁看 regime 是否有經濟意義，不追求最好看的回測。")

    st.subheader("未來報酬比較")
    analysis_table = regime_summary[
        ["label", "auto_label", "future_5", "future_10", "future_20", "count", "ratio"]
    ].copy()
    analysis_table = rename_index_regime(analysis_table)
    st.dataframe(rename_columns_zh(analysis_table), use_container_width=True)

    st.subheader("Custom Features by Regime")
    custom_cols = [c for c in get_custom_feature_columns() if c in regime_summary.columns]
    custom_table = regime_summary[["label", "auto_label"] + custom_cols].copy()
    custom_table = rename_index_regime(custom_table)
    st.dataframe(rename_columns_zh(custom_table), use_container_width=True)

    st.subheader("Base Features by Regime")
    base_cols = [c for c in get_base_feature_columns() if c in regime_summary.columns]
    base_table = regime_summary[["label", "auto_label"] + base_cols].copy()
    base_table = rename_index_regime(base_table)
    st.dataframe(rename_columns_zh(base_table), use_container_width=True)

    st.subheader("Regime Transition Matrix")
    trans_df = transition_matrix.copy()
    trans_df.index = [f"Regime {int(x)}" for x in trans_df.index]
    try:
        trans_df.columns = [f"Regime {int(x)}" for x in trans_df.columns]
    except Exception:
        pass
    st.dataframe(trans_df, use_container_width=True)

elif page == "Strategy Lab":
    st.header("💰 Strategy Lab（Phase 1 驗證版）")
    st.info(
        "目標不是做漂亮績效，而是驗證：\n"
        "1. regime framework 是否有意義\n"
        "2. 固定 mapping + shock + 成本後是否還有 edge\n"
        "3. OOS 是否能站住"
    )

    strategy_regime_df = regime_df.copy()

    # baseline signals
    ma_sig = moving_average_signal(strategy_regime_df["Close"], ma_window=50)
    mr_sig = mean_reversion_signal(strategy_regime_df["Close"], window=20, z_entry=1.0)
    bo_sig = breakout_signal(strategy_regime_df["Close"], window=20)

    signal_library = {
        "Trend MA": ma_sig,
        "Mean Reversion": mr_sig,
        "Breakout": bo_sig,
    }

    # baseline backtest with costs
    ma_bt = backtest_with_costs(strategy_regime_df["Close"], ma_sig, cost_bps=cost_bps)
    mr_bt = backtest_with_costs(strategy_regime_df["Close"], mr_sig, cost_bps=cost_bps)
    bo_bt = backtest_with_costs(strategy_regime_df["Close"], bo_sig, cost_bps=cost_bps)

    stats_df = pd.DataFrame(
        {
            "Trend MA": performance_stats_local(ma_bt),
            "Mean Reversion": performance_stats_local(mr_bt),
            "Breakout": performance_stats_local(bo_bt),
        }
    ).T

    st.subheader("整體策略績效 Overall Performance（含成本）")
    explain_columns(["total_return", "ann_return", "ann_vol", "sharpe", "max_dd"])
    st.dataframe(rename_columns_zh(stats_df), use_container_width=True)

    st.subheader("互動資金曲線 Interactive Equity Curves（含成本）")
    plot_equity_interactive(
        {
            "Trend MA": ma_bt,
            "Mean Reversion": mr_bt,
            "Breakout": bo_bt,
        },
        title="Baseline Strategy Equity Curves（含成本）",
    )

    # compatibility table（研究用途）
    st.subheader("市場狀態 × 策略適配性 Regime × Strategy Compatibility（研究用途）")
    regime_ma = regime_strategy_table(strategy_regime_df, ma_sig)
    regime_mr = regime_strategy_table(strategy_regime_df, mr_sig)
    regime_bo = regime_strategy_table(strategy_regime_df, bo_sig)

    combo = pd.concat(
        {
            "Trend MA": regime_ma["sharpe_like"],
            "Mean Reversion": regime_mr["sharpe_like"],
            "Breakout": regime_bo["sharpe_like"],
        },
        axis=1,
    )
    combo["label"] = combo.index.map(label_map)
    combo["auto_label"] = combo.index.map(auto_label_map)
    combo = rename_index_regime(combo)
    st.dataframe(rename_columns_zh(combo), use_container_width=True)

    # fixed mapping
    st.subheader("固定版 Regime → Strategy Mapping（結構導向）")
    regime_to_strategy = structural_regime_strategy_map(label_map)
    mapping_df = summarize_regime_strategy_map(regime_to_strategy, label_map=label_map)
    st.dataframe(rename_columns_zh(mapping_df), use_container_width=True)

    # auto switch
    switch_sig = build_regime_switch_signal(
        regime_series=strategy_regime_df["regime"],
        signal_library=signal_library,
        regime_to_strategy=regime_to_strategy,
    )

    if use_shock:
        shock_score = compute_shock_score(strategy_regime_df)
        switch_sig = risk_override(switch_sig, shock_score, threshold=shock_threshold)

        st.subheader("Shock Filter 觸發概況")
        st.write(f"Shock 觸發筆數：{int((shock_score >= shock_threshold).sum())}")

    switch_bt = backtest_with_costs(
        strategy_regime_df["Close"],
        switch_sig,
        cost_bps=cost_bps,
    )

    switch_perf = performance_stats_local(switch_bt)
    switch_stats = pd.DataFrame({"Auto Switch": switch_perf}).T
    compare_stats = pd.concat([stats_df, switch_stats], axis=0)

    st.subheader("加入自動切換後的績效 Performance with Auto Switch（含成本）")
    st.dataframe(rename_columns_zh(compare_stats), use_container_width=True)

    st.subheader("包含自動切換的資金曲線 Equity Curves with Auto Switch（含成本）")
    plot_equity_interactive(
        {
            "Trend MA": ma_bt,
            "Mean Reversion": mr_bt,
            "Breakout": bo_bt,
            "Auto Switch": switch_bt,
        },
        title="Equity Curves with Auto Switch（含成本）",
    )

    # trade quality
    tq = trade_quality_stats(switch_bt)

    st.subheader("📊 交易品質分析（含成本）")
    explain_columns(["win_rate", "profit_factor", "avg_trade", "trade_count", "turnover_count"])
    st.write(f"勝率 (Win Rate)：{tq['win_rate']:.2%}")
    st.write(f"盈虧比 (RR)：{tq['rr']:.2f}")
    st.write(f"Profit Factor：{tq['profit_factor']:.2f}")
    st.write(f"平均每筆報酬：{tq['avg_trade']:.5f}")
    st.write(f"開倉次數：{tq['trade_count']}")
    st.write(f"交易變動次數：{tq['turnover_count']}")

    st.subheader("🧠 系統解讀報告")
    for line in evaluation_report_text(tq):
        st.write(line)

    st.markdown("---")

    # -----------------------------------------------------
    # Out-of-sample validation（真正 OOS）
    # -----------------------------------------------------
    st.header("🧪 Out-of-Sample 驗證")
    st.info(
        "這裡的 regime 不是全資料先 fit 再切樣本，\n"
        "而是只用 in-sample fit 後，再 assign 到 OOS。"
    )

    is_bt = switch_bt[switch_bt.index < split_date].copy()
    oos_bt = switch_bt[switch_bt.index >= split_date].copy()

    if len(is_bt) < 50 or len(oos_bt) < 20:
        st.warning("資料不足以做穩定的 in-sample / out-of-sample 驗證。請拉長期間。")
    else:
        is_stats = performance_stats_local(is_bt)
        oos_stats = performance_stats_local(oos_bt)

        split_stats = pd.DataFrame(
            {
                "In-Sample": is_stats,
                "Out-of-Sample": oos_stats,
            }
        )
        st.subheader("In-Sample vs Out-of-Sample")
        st.dataframe(rename_columns_zh(split_stats), use_container_width=True)

        st.subheader("In-Sample / Out-of-Sample Equity")
        plot_equity_interactive(
            {
                "In-Sample": is_bt,
                "Out-of-Sample": oos_bt,
            },
            title="In-Sample / Out-of-Sample Equity（含成本）",
        )

        oos_tq = trade_quality_stats(oos_bt)
        st.subheader("Out-of-Sample 交易品質")
        st.write(f"勝率：{oos_tq['win_rate']:.2%}")
        st.write(f"盈虧比：{oos_tq['rr']:.2f}")
        st.write(f"Profit Factor：{oos_tq['profit_factor']:.2f}")
        st.write(f"平均每筆報酬：{oos_tq['avg_trade']:.5f}")
        st.write(f"開倉次數：{oos_tq['trade_count']}")
        st.write(f"交易變動次數：{oos_tq['turnover_count']}")

        st.subheader("樣本外判讀")
        if oos_tq["profit_factor"] > 1.2 and oos_stats["total_return"] > 0:
            st.success("樣本外仍保有正優勢。這代表 Phase 1 框架有進一步研究價值。")
        elif oos_stats["total_return"] > 0:
            st.info("樣本外勉強為正，但 edge 很薄。這比較像研究框架成立，不代表已可實盤。")
        else:
            st.warning("樣本外已失去優勢。這代表目前 regime framework 有研究價值，但 alpha 還不夠厚。")