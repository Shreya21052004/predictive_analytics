"""anomaly.py — Unsupervised anomaly detection for Compute and Storage.

Replaces the rule-label → Random Forest approach with three complementary
unsupervised signals that work without any ground-truth incident data:

Signal 1 — Per-resource rolling z-score
    For each metric, compute a rolling 7-day mean and std from that resource's
    own history. Flag hours where (value - rolling_mean) / rolling_std > 3.
    This catches "a metric that's normally flat suddenly spikes" without any
    global threshold. A CPU that lives at 5% looks very different from one
    that lives at 70% — they get their own baselines.

Signal 2 — Max/Avg ratio (intra-hour burst detection)
    Maximum / Average within the same hour. A ratio of 1.0 = perfectly flat
    hour. A ratio of 16× = one sample hit 80% while the average was 5%.
    Captures bursts that averages hide and that the normaliser previously
    discarded by always picking Average.

Signal 3 — Per-resource IsolationForest on the multi-metric feature vector
    Trained only on that resource's own history (not the global population).
    Learns the normal joint distribution of all available metrics for this
    specific resource. Anomaly = a point that doesn't fit the resource's own
    pattern, regardless of absolute values.

Composite score:
    anomaly_score = weighted average of the three normalised signals.
    Weights are adaptive: signals with more data and lower base rates get
    higher weight. Resources with < 24 hours of history fall back to
    signal 1 only.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .metric_registry import canonical_features

# ── Constants ──────────────────────────────────────────────────────────────
ROLLING_WINDOW_H   = 7 * 24   # 7 days of hourly data for rolling baseline
MIN_HISTORY_FOR_IF = 48        # minimum hours before IsolationForest is trusted
MIN_HISTORY_FOR_ROLL = 24      # minimum hours for rolling z-score
Z_SCORE_THRESHOLD  = 3.0       # standard deviations for per-resource baseline
MAX_AVG_RATIO_THRESHOLD = 5.0  # flag if max > 5× avg within same hour
IF_CONTAMINATION   = 0.05      # expected anomaly fraction for IsolationForest

# Weights for composite score
W_ZSCORE   = 0.40
W_MAXAVG   = 0.25
W_IF       = 0.35


# ── Feature extraction ──────────────────────────────────────────────────────

def _primary_features(category: str) -> List[str]:
    """Return the canonical Average-value features for a category."""
    return canonical_features(category)


def _max_features(history: pd.DataFrame, category: str) -> List[str]:
    """Return available _max columns for burst detection."""
    return [
        f"{feat}_max"
        for feat in _primary_features(category)
        if f"{feat}_max" in history.columns and history[f"{feat}_max"].notna().any()
    ]


def _build_feature_matrix(history: pd.DataFrame, category: str) -> pd.DataFrame:
    """Assemble the numeric feature matrix for IsolationForest.

    Includes: canonical averages + max/avg ratios for available metrics.
    Fills NaN with column median (per-resource, not global population).
    """
    feats = [f for f in _primary_features(category) if f in history.columns]
    if not feats:
        return pd.DataFrame()

    X = history[feats].copy()

    # Add max/avg ratio features
    for feat in feats:
        max_col = f"{feat}_max"
        if max_col in history.columns:
            avg_vals = history[feat].replace(0, np.nan)
            ratio = history[max_col] / avg_vals
            X[f"{feat}_burst_ratio"] = ratio.clip(upper=50)  # cap extreme ratios

    # Fill NaN with per-resource column median
    for col in X.columns:
        median = X[col].median()
        if pd.isna(median):
            X[col] = X[col].fillna(0.0)
        else:
            X[col] = X[col].fillna(median)

    return X.astype(float)


# ── Signal 1: Per-resource rolling z-score ─────────────────────────────────

def rolling_zscore_scores(history: pd.DataFrame, category: str) -> pd.Series:
    """Return a per-row anomaly score [0, 1] based on rolling z-score.

    For each canonical feature, compute rolling mean/std over the past
    ROLLING_WINDOW_H hours (excluding the current point).  The z-score
    for that point is |value - rolling_mean| / rolling_std.  Take the max
    z-score across all features as the row score, then normalise to [0, 1]
    using a sigmoid-like mapping capped at Z_SCORE_THRESHOLD × 2.
    """
    feats = [f for f in _primary_features(category) if f in history.columns]
    if not feats or len(history) < MIN_HISTORY_FOR_ROLL:
        return pd.Series(0.0, index=history.index)

    h = history.sort_values("timestamp").copy()
    max_z = pd.Series(0.0, index=h.index)

    for feat in feats:
        col = h[feat].astype(float)
        if col.notna().sum() < MIN_HISTORY_FOR_ROLL:
            continue
        # shift(1) so the current point is not in its own rolling window
        roll_mean = col.shift(1).rolling(ROLLING_WINDOW_H, min_periods=12).mean()
        roll_std  = col.shift(1).rolling(ROLLING_WINDOW_H, min_periods=12).std()
        roll_std  = roll_std.replace(0, np.nan)
        z = ((col - roll_mean) / roll_std).abs().fillna(0.0)
        max_z = np.maximum(max_z, z)

    # Normalise: z=0 → 0.0, z=3 → 0.75, z=6 → 1.0
    score = (max_z / (Z_SCORE_THRESHOLD * 2)).clip(0.0, 1.0)
    return score.reindex(history.index).fillna(0.0)


# ── Signal 2: Max/Avg burst ratio ──────────────────────────────────────────

def burst_ratio_scores(history: pd.DataFrame, category: str) -> pd.Series:
    """Return a per-row anomaly score [0, 1] based on intra-hour burst ratio.

    For each feature that has a _max column, compute max/avg ratio.
    A ratio > MAX_AVG_RATIO_THRESHOLD is anomalous.  Score is the maximum
    ratio across all features, normalised to [0, 1].
    """
    feats = [f for f in _primary_features(category) if f in history.columns]
    if not feats:
        return pd.Series(0.0, index=history.index)

    max_ratio = pd.Series(0.0, index=history.index)

    for feat in feats:
        max_col = f"{feat}_max"
        if max_col not in history.columns:
            continue
        avg_vals = history[feat].astype(float).replace(0, np.nan)
        max_vals = history[max_col].astype(float)
        ratio = (max_vals / avg_vals).fillna(1.0).clip(1.0, 50.0)
        max_ratio = np.maximum(max_ratio, ratio)

    # Normalise: ratio=1 → 0.0, ratio=5 → 0.5, ratio=10 → 1.0
    score = ((max_ratio - 1.0) / (MAX_AVG_RATIO_THRESHOLD * 2 - 1.0)).clip(0.0, 1.0)
    return score.reindex(history.index).fillna(0.0)


# ── Signal 3: Per-resource IsolationForest ─────────────────────────────────

def isolation_forest_scores(history: pd.DataFrame, category: str) -> pd.Series:
    """Return a per-row anomaly score [0, 1] using per-resource IsolationForest.

    Trained on the resource's own history — learns what's normal for THIS
    resource, not the global population. Resources with insufficient history
    return 0.0 (no opinion).
    """
    if len(history) < MIN_HISTORY_FOR_IF:
        return pd.Series(0.0, index=history.index)

    X = _build_feature_matrix(history, category)
    if X.empty or X.shape[1] == 0:
        return pd.Series(0.0, index=history.index)

    # Scale features (IsolationForest is not scale-invariant unlike tree models)
    scaler = RobustScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except Exception:
        return pd.Series(0.0, index=history.index)

    contamination = min(IF_CONTAMINATION, max(0.01, 1.0 / len(history)))
    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        max_samples=min(256, len(history)),
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_scaled)

    # decision_function: lower = more anomalous. Map to [0, 1].
    raw_scores = clf.decision_function(X_scaled)  # typically in [-0.5, 0.5]
    # Invert and normalise: -0.5 → 1.0 (very anomalous), +0.5 → 0.0 (normal)
    norm = ((-raw_scores) - (-raw_scores).min()) / (
        ((-raw_scores).max() - (-raw_scores).min()) + 1e-9
    )
    return pd.Series(norm, index=X.index).reindex(history.index).fillna(0.0)


# ── Composite score ────────────────────────────────────────────────────────

def compute_anomaly_score(
    resource_history: pd.DataFrame,
    category: str,
) -> Tuple[float, Dict]:
    """Compute a composite anomaly score [0, 100] for the latest row.

    Returns (score_0_to_100, detail_dict) where detail_dict breaks down
    the contribution from each signal for explainability.
    """
    if resource_history.empty:
        return 0.0, {"error": "no_history"}

    h = resource_history.sort_values("timestamp").copy()

    z_scores   = rolling_zscore_scores(h, category)
    br_scores  = burst_ratio_scores(h, category)
    if_scores  = isolation_forest_scores(h, category)

    n = len(h)

    # Adaptive weights: reduce IF weight if not enough history
    w_if_actual = W_IF if n >= MIN_HISTORY_FOR_IF else 0.0
    w_z_actual  = W_ZSCORE + (W_IF - w_if_actual) * 0.6
    w_br_actual = W_MAXAVG + (W_IF - w_if_actual) * 0.4
    total_w = w_z_actual + w_br_actual + w_if_actual

    composite = (
        w_z_actual  * z_scores  +
        w_br_actual * br_scores +
        w_if_actual * if_scores
    ) / total_w

    # Latest point
    latest_idx = h.index[-1]
    latest_composite = float(composite.loc[latest_idx]) if latest_idx in composite.index else 0.0
    latest_z  = float(z_scores.loc[latest_idx])  if latest_idx in z_scores.index  else 0.0
    latest_br = float(br_scores.loc[latest_idx]) if latest_idx in br_scores.index else 0.0
    latest_if = float(if_scores.loc[latest_idx]) if latest_idx in if_scores.index else 0.0

    score_100 = round(latest_composite * 100, 2)

    # Which signal drove it — for explainability
    drivers = []
    if latest_z  > 0.4: drivers.append(f"rolling_zscore={latest_z:.2f}")
    if latest_br > 0.4: drivers.append(f"burst_ratio={latest_br:.2f}")
    if latest_if > 0.4: drivers.append(f"isolation_forest={latest_if:.2f}")

    detail = {
        "rolling_zscore_signal":  round(latest_z,  3),
        "burst_ratio_signal":     round(latest_br, 3),
        "isolation_forest_signal": round(latest_if, 3),
        "composite_anomaly_score": score_100,
        "history_points_used":    n,
        "drivers":                drivers if drivers else ["none_dominant"],
        "method":                 "unsupervised_composite",
        "weights_used": {
            "rolling_zscore":    round(w_z_actual / total_w, 3),
            "burst_ratio":       round(w_br_actual / total_w, 3),
            "isolation_forest":  round(w_if_actual / total_w, 3),
        },
    }

    return score_100, detail


# ── Batch scoring (called from generate_predictions) ───────────────────────

def score_resource(resource_history: pd.DataFrame, category: str) -> Tuple[float, Dict]:
    """Public entry point. Returns (risk_score_0_to_100, anomaly_detail)."""
    try:
        return compute_anomaly_score(resource_history, category)
    except Exception as e:
        return 0.0, {"error": str(e), "method": "unsupervised_composite"}
