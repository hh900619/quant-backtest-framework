from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# =========================================================
# Scaling / KMeans
# =========================================================
def scale_features_fit(matrix: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    只用 matrix 本身 fit scaler。
    給 in-sample 用。
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)
    return X, scaler


def scale_features_transform(matrix: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """
    使用既有 scaler transform。
    給 out-of-sample 用。
    """
    return scaler.transform(matrix)


def fit_kmeans(
    matrix: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, StandardScaler, KMeans]:
    """
    只用當前 matrix fit scaler + kmeans。
    適合 in-sample。
    """
    X, scaler = scale_features_fit(matrix)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    )
    labels = model.fit_predict(X)

    return labels, scaler, model


def predict_kmeans(
    matrix: pd.DataFrame,
    scaler: StandardScaler,
    model: KMeans,
) -> np.ndarray:
    """
    用既有 scaler / model 對新資料做 regime assign。
    適合 out-of-sample。
    """
    X = scale_features_transform(matrix, scaler)
    labels = model.predict(X)
    return labels


def run_kmeans(
    matrix: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, StandardScaler, KMeans]:
    """
    相容舊版 app.py 的 wrapper。
    現在內部其實就是 fit_kmeans。
    """
    return fit_kmeans(
        matrix=matrix,
        n_clusters=n_clusters,
        random_state=random_state,
    )


def assign_regimes(
    feature_df: pd.DataFrame,
    matrix: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    out = feature_df.copy()
    regime_series = pd.Series(labels, index=matrix.index, name="regime")
    out = out.join(regime_series, how="left")
    return out


def get_regime_counts(df: pd.DataFrame, regime_col: str = "regime") -> pd.Series:
    return df[regime_col].value_counts(dropna=True).sort_index()


def get_cluster_quality(matrix: pd.DataFrame, labels: np.ndarray, model: KMeans) -> dict:
    """
    這裡只做研究品質評估，不參與交易決策。
    """
    X, _ = scale_features_fit(matrix)

    result = {
        "inertia": float(model.inertia_),
        "silhouette": np.nan,
    }

    if len(set(labels)) > 1 and len(matrix) > len(set(labels)):
        result["silhouette"] = float(silhouette_score(X, labels))

    return result


# =========================================================
# Causal smoothing（只用過去，不看未來）
# =========================================================
def _majority_vote(x: pd.Series) -> float:
    s = pd.Series(x).dropna()
    if s.empty:
        return np.nan
    mode_vals = s.mode()
    if len(mode_vals) == 0:
        return np.nan
    return float(mode_vals.iloc[0])


def smooth_regimes_majority(
    regime_series: pd.Series,
    window: int = 5,
) -> pd.Series:
    """
    第一層：causal rolling majority vote
    只看「當下與過去」，不看未來。
    """
    if window <= 1:
        return regime_series.copy().astype("Int64")

    valid = regime_series.dropna().astype(int)

    # 關鍵：不再 center=True
    smoothed = valid.rolling(
        window=window,
        min_periods=1,
    ).apply(_majority_vote, raw=False)

    out = regime_series.copy()
    out.loc[valid.index] = smoothed.round().astype("Int64")
    return out.astype("Int64")


def apply_confirmation_filter(
    regime_series: pd.Series,
    confirm_bars: int = 3,
) -> pd.Series:
    """
    第二層：新 regime 必須連續出現 confirm_bars 根，才承認切換。
    純 causal，沒有未來資料。
    """
    if confirm_bars <= 1:
        return regime_series.copy().astype("Int64")

    valid = regime_series.dropna().astype(int)
    if valid.empty:
        return regime_series.copy().astype("Int64")

    result = []
    current_regime = valid.iloc[0]
    candidate_regime = None
    candidate_count = 0

    for val in valid:
        if val == current_regime:
            candidate_regime = None
            candidate_count = 0
            result.append(current_regime)
            continue

        if candidate_regime is None or val != candidate_regime:
            candidate_regime = val
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count >= confirm_bars:
            current_regime = candidate_regime
            candidate_regime = None
            candidate_count = 0

        result.append(current_regime)

    out_valid = pd.Series(result, index=valid.index, dtype="Int64")
    out = regime_series.copy()
    out.loc[out_valid.index] = out_valid
    return out.astype("Int64")


def enforce_min_run_length_causal(
    regime_series: pd.Series,
    min_run_length: int = 3,
) -> pd.Series:
    """
    第三層：因果版 minimum run filter

    舊版會看前後 run 長度來合併，會用到未來資料。
    這版不做前後比較，只做「確認延後」：
    - 一個新 regime 即使已被確認，
      若目前 run 長度還沒達到 min_run_length，就暫時維持前 regime
    - 一旦 run 長度 >= min_run_length，從該點開始正式採用新 regime

    這樣的代價是：
    - regime 會更 lag
    但好處是：
    - 沒有未來資料洩漏
    """
    if min_run_length <= 1:
        return regime_series.copy().astype("Int64")

    valid = regime_series.dropna().astype(int)
    if valid.empty:
        return regime_series.copy().astype("Int64")

    out_vals: list[int] = []

    confirmed_regime = valid.iloc[0]
    current_candidate = valid.iloc[0]
    candidate_count = 0

    for val in valid:
        if val == current_candidate:
            candidate_count += 1
        else:
            current_candidate = val
            candidate_count = 1

        # run 長度還不夠，就維持舊 regime
        if current_candidate != confirmed_regime and candidate_count < min_run_length:
            out_vals.append(confirmed_regime)
            continue

        # run 長度夠了，從現在開始才正式切換
        if current_candidate != confirmed_regime and candidate_count >= min_run_length:
            confirmed_regime = current_candidate

        out_vals.append(confirmed_regime)

    out_valid = pd.Series(out_vals, index=valid.index, dtype="Int64")
    out = regime_series.copy()
    out.loc[out_valid.index] = out_valid
    return out.astype("Int64")


def build_final_regime_series(
    regime_series: pd.Series,
    majority_window: int = 5,
    confirm_bars: int = 3,
    min_run_length: int = 3,
) -> pd.Series:
    """
    高階平滑（Phase 1 修正版）
    1. causal majority smoothing
    2. confirmation filter
    3. causal minimum run filter
    """
    out = regime_series.copy()
    out = smooth_regimes_majority(out, window=majority_window)
    out = apply_confirmation_filter(out, confirm_bars=confirm_bars)
    out = enforce_min_run_length_causal(out, min_run_length=min_run_length)
    return out.astype("Int64")


def apply_advanced_regime_smoothing(
    df: pd.DataFrame,
    source_col: str = "regime",
    target_col: str = "regime_smoothed",
    majority_window: int = 5,
    confirm_bars: int = 3,
    min_run_length: int = 3,
) -> pd.DataFrame:
    out = df.copy()
    out[target_col] = build_final_regime_series(
        out[source_col],
        majority_window=majority_window,
        confirm_bars=confirm_bars,
        min_run_length=min_run_length,
    )
    return out