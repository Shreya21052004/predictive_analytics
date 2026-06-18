from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from .models import MODEL_VERSION
from .anomaly import score_resource
from .metric_registry import PRIMARY_FORECAST_FEATURE, canonical_features, metric_name_for_feature


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean(value):
    if isinstance(value, (dict, list)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def is_real(value):
    return value is not None and not pd.isna(value)


def is_true_flag(value):
    return value is True or value == 1


def positive(value):
    return is_real(value) and value > 0


def severity(score):
    if score >= 90:
        return "CRITICAL"
    if score >= 75:
        return "HIGH"
    if score >= 50:
        return "WARNING"
    return "INFO"


def badge(score):
    if score >= 75:
        return "red"
    if score >= 50:
        return "amber"
    if score >= 25:
        return "blue"
    return "green"


def channel_for(sev):
    return {
        "CRITICAL": "pagerduty",
        "HIGH": "pagerduty",
        "WARNING": "slack",
        "INFO": "suppressed",
    }[sev]


def slope_forecast(group, feature, thresholds):
    series = group[["timestamp", feature]].dropna().sort_values("timestamp")
    if len(series) < 2:
        return {}
    x = (series["timestamp"] - series["timestamp"].min()).dt.total_seconds() / 3600.0
    y = series[feature].astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    current_y = y.iloc[-1]
    out = {
        "current": float(current_y),
        "hourly_slope": float(slope),
    }
    for threshold in thresholds:
        key = f"breach_{int(threshold)}"
        if current_y >= threshold:
            out[key] = series["timestamp"].iloc[-1].isoformat()
        elif slope > 0:
            hours = (threshold - current_y) / slope
            MAX_FORECAST_HOURS = 87_600  # cap at ~10y to avoid Timestamp overflow
            if hours > MAX_FORECAST_HOURS:
                out[key] = None
            else:
                try:
                    out[key] = (series["timestamp"].iloc[-1] + pd.Timedelta(hours=float(hours))).isoformat()
                except (OverflowError, pd.errors.OutOfBoundsDatetime):
                    out[key] = None
        else:
            out[key] = None
    return out


# Network metrics that are bursty/spiky — exponential smoothing fits better than linear
BURSTY_METRICS = {
    "canonical_net_in_bytes",
    "canonical_net_out_bytes",
    "canonical_active_connections",
    "canonical_dns_queries",
    "canonical_throughput",
}

MIN_NONZERO_FRAC = 0.2
MIN_POINTS_FOR_FORECAST = 5
MIN_STD_NORM = 0.01

# >= this many hourly points -> aggregate to daily means before fitting trend
LONG_SERIES_THRESHOLD = 72


def _exp_smooth_forecast(y_vals, alpha=0.3):
    s = float(y_vals[0])
    for v in y_vals[1:]:
        s = alpha * float(v) + (1 - alpha) * s
    return s


def _clean_series(series, feature):
    """Strip likely missing-reported-as-zero tails.
    >80% zero = genuinely idle (keep as-is). 20-80% zero = strip trailing zeros.
    <20% zero = genuine zeros, keep all."""
    y = series[feature].values.astype(float)
    zero_frac = (y == 0).sum() / len(y)

    if zero_frac > 0.8:
        return series, "idle"

    if zero_frac > MIN_NONZERO_FRAC:
        nonzero_idx = np.where(y != 0)[0]
        if len(nonzero_idx) >= MIN_POINTS_FOR_FORECAST:
            last_nonzero = nonzero_idx[-1]
            series = series.iloc[:last_nonzero + 1]
        return series, "cleaned"

    return series, "ok"


def _detect_mixed_stats(resource_history, feature):
    stat_col = f"{feature}_stat"
    if stat_col not in resource_history.columns:
        return False
    stats_used = resource_history[stat_col].dropna().unique()
    return len(stats_used) > 1


def _long_series_forecast(x, y, is_pct, current_y_val):
    """For series >= LONG_SERIES_THRESHOLD points, aggregate to daily means first
    to remove intra-day diurnal noise before fitting the trend line."""
    day_bins = (x // 24).astype(int)
    daily_y, daily_x = [], []
    for d in sorted(set(day_bins)):
        mask = day_bins == d
        if mask.sum() >= 6:
            daily_y.append(float(y[mask].mean()))
            daily_x.append(float(x[mask].mean()))

    if len(daily_y) < 3:
        return None, None, None, None

    daily_x_arr = np.array(daily_x)
    daily_y_arr = np.array(daily_y)
    slope_raw, intercept_raw = np.polyfit(daily_x_arr, daily_y_arr, 1)

    yhat = slope_raw * daily_x_arr + intercept_raw
    ss_res = float(np.sum((daily_y_arr - yhat) ** 2))
    ss_tot = float(np.sum((daily_y_arr - daily_y_arr.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

    return slope_raw, r2, daily_x_arr, daily_y_arr


def behavioral_forecast(category, resource_history, row):
    feature = choose_forecast_feature(category, resource_history)
    provider = row.get("service_name")
    metric_name = metric_name_for_feature(category, provider, feature)

    null_result = {
        "metric_name": metric_name,
        "canonical_feature": feature,
        "current_value": None,
        "forecast_1h": None,
        "forecast_6h": None,
        "forecast_24h": None,
        "forecast_7d": None,
        "forecast_30d": None,
        "trend_direction": "stable",
        "trend_fit_r2": 0.0,
        "forecast_reliability": "low",
        "forecast_method": "none",
        "data_note": None,
    }

    if not feature or feature not in resource_history.columns:
        return {**null_result, "data_note": "feature_not_available"}

    series = resource_history[["timestamp", feature]].dropna().sort_values("timestamp")
    if series.empty:
        return {**null_result, "data_note": "no_data_points"}

    current = float(series[feature].iloc[-1])

    if len(series) < MIN_POINTS_FOR_FORECAST:
        return {
            **null_result,
            "current_value": clean(current),
            "forecast_1h": clean(current),
            "forecast_6h": clean(current),
            "forecast_24h": clean(current),
            "forecast_7d": clean(current),
            "forecast_30d": clean(current),
            "data_note": f"insufficient_points_{len(series)}",
        }

    mixed_stats = _detect_mixed_stats(resource_history, feature)

    series, clean_status = _clean_series(series, feature)
    if len(series) < 2:
        return {
            **null_result,
            "current_value": clean(current),
            "forecast_1h": clean(current),
            "forecast_6h": clean(current),
            "forecast_24h": clean(current),
            "forecast_7d": clean(current),
            "forecast_30d": clean(current),
            "data_note": "all_zeros_after_cleaning",
        }

    x = (series["timestamp"] - series["timestamp"].min()).dt.total_seconds() / 3600.0
    y = series[feature].astype(float)
    is_pct = feature and (feature.endswith("_pct") or feature == "canonical_health_pct")

    y_min, y_max = float(y.min()), float(y.max())
    y_range = y_max - y_min
    y_norm = (y - y_min) / y_range if y_range > 0 else pd.Series(np.zeros(len(y)), index=y.index)

    if y_norm.std() < MIN_STD_NORM:
        return {
            **null_result,
            "current_value": clean(current),
            "forecast_1h": clean(current),
            "forecast_6h": clean(current),
            "forecast_24h": clean(current),
            "forecast_7d": clean(current),
            "forecast_30d": clean(current),
            "trend_direction": "stable",
            "trend_fit_r2": 1.0 if clean_status == "idle" else 0.0,
            "forecast_reliability": "high" if clean_status == "idle" else "low",
            "forecast_method": "flat_signal",
            "data_note": "mixed_stat_fields" if mixed_stats else clean_status,
        }

    if feature in BURSTY_METRICS and not mixed_stats:
        alpha = 0.3
        smoothed_level = _exp_smooth_forecast(y.values, alpha=alpha)
        recent_n = max(3, len(y) // 4)
        y_recent = y.values[-recent_n:]
        x_recent = np.arange(len(y_recent), dtype=float)
        slope_recent, _ = np.polyfit(x_recent, y_recent, 1)
        if len(x) > 1:
            interval_h = float((x.iloc[-1] - x.iloc[-2]))
        else:
            interval_h = 1.0

        def forecast(hours):
            steps = hours / max(interval_h, 0.001)
            val = smoothed_level + slope_recent * steps
            return max(0.0, min(100.0, val)) if is_pct else max(0.0, val)

        y_norm_vals = y_norm.values
        smoothed_series = [y_norm_vals[0]]
        for v in y_norm_vals[1:]:
            smoothed_series.append(alpha * v + (1 - alpha) * smoothed_series[-1])
        smoothed_arr = np.array(smoothed_series)
        ss_res = float(np.sum((y_norm_vals - smoothed_arr) ** 2))
        ss_tot = float(np.sum((y_norm_vals - y_norm_vals.mean()) ** 2))
        confidence = 0.0 if ss_tot == 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        method = "exponential_smoothing"

        if slope_recent > 0.05 * smoothed_level:
            trend = "increasing"
        elif slope_recent < -0.05 * smoothed_level:
            trend = "decreasing"
        else:
            trend = "stable"

    else:
        x_arr = x.values
        y_arr = y.values

        if len(series) >= LONG_SERIES_THRESHOLD:
            slope_raw, confidence, daily_x, daily_y = _long_series_forecast(
                x_arr, y_arr, is_pct, float(y.iloc[-1])
            )
            if slope_raw is None:
                slope_raw, _ = np.polyfit(x_arr, y_arr, 1)
                slope_norm, intercept_norm = np.polyfit(x_arr, y_norm.values, 1)
                yhat_norm = slope_norm * x_arr + intercept_norm
                ss_res = float(np.sum((y_norm.values - yhat_norm) ** 2))
                ss_tot = float(np.sum((y_norm.values - y_norm.values.mean()) ** 2))
                confidence = 0.0 if ss_tot == 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        else:
            slope_norm, intercept_norm = np.polyfit(x_arr, y_norm.values, 1)
            slope_raw, _ = np.polyfit(x_arr, y_arr, 1)
            yhat_norm = slope_norm * x_arr + intercept_norm
            ss_res = float(np.sum((y_norm.values - yhat_norm) ** 2))
            ss_tot = float(np.sum((y_norm.values - y_norm.values.mean()) ** 2))
            confidence = 0.0 if ss_tot == 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

        signal_range = y_arr.max() - y_arr.min() if y_arr.max() != y_arr.min() else 1.0
        slope_norm_equiv = slope_raw / signal_range
        if slope_norm_equiv > 0.05:
            trend = "increasing"
        elif slope_norm_equiv < -0.05:
            trend = "decreasing"
        else:
            trend = "stable"

        observed_min = float(y.min())
        current_y_val = float(y.iloc[-1])

        def forecast(hours):
            val = current_y_val + slope_raw * hours
            if is_pct:
                floor = max(0.0, observed_min * 0.9)
                return max(floor, min(100.0, val))
            return max(0.0, val)

        projected_24h = current_y_val + slope_raw * 24
        if is_pct and projected_24h < observed_min * 0.75:
            trend = "stable"

        method = "linear_regression"

    data_note = "mixed_stat_fields" if mixed_stats else clean_status

    if confidence >= 0.7:
        forecast_reliability = "high"
    elif confidence >= 0.3:
        forecast_reliability = "moderate"
    else:
        forecast_reliability = "low"

    if confidence < 0.3:
        obs_lo = float(y.min())
        obs_hi = float(y.max())
        def _clamp_to_history(val):
            if is_pct:
                return max(obs_lo * 0.9, min(obs_hi * 1.1, val))
            return max(0.0, val)
        f1h   = round(_clamp_to_history(forecast(1)),      3)
        f6h   = round(_clamp_to_history(forecast(6)),      3)
        f24h  = round(_clamp_to_history(forecast(24)),     3)
        f7d   = round(_clamp_to_history(forecast(7 * 24)), 3)
        f30d  = round(_clamp_to_history(forecast(30 * 24)),3)
    else:
        f1h   = round(forecast(1),       3)
        f6h   = round(forecast(6),       3)
        f24h  = round(forecast(24),      3)
        f7d   = round(forecast(7 * 24),  3)
        f30d  = round(forecast(30 * 24), 3)

    return {
        "metric_name": metric_name,
        "canonical_feature": feature,
        "current_value": clean(current),
        "forecast_1h": f1h,
        "forecast_6h": f6h,
        "forecast_24h": f24h,
        "forecast_7d": f7d,
        "forecast_30d": f30d,
        "trend_direction": trend,
        "trend_fit_r2": round(confidence, 3),
        "forecast_reliability": forecast_reliability,
        "forecast_method": method,
        "data_note": data_note,
    }


def choose_forecast_feature(category, resource_history):
    preferred = PRIMARY_FORECAST_FEATURE.get(category)
    if preferred in resource_history.columns and resource_history[preferred].notna().any():
        return preferred
    for feature in canonical_features(category):
        if feature in resource_history.columns and resource_history[feature].notna().any():
            return feature
    return preferred


def latest_rows(df):
    if df.empty:
        return df
    idx = df.sort_values("timestamp").groupby(["resource_id", "category"], dropna=False).tail(1).index
    return df.loc[idx].copy()


def consolidated_row(category, resource_history, base_row):
    row = base_row.copy()
    for feature in canonical_features(category):
        if feature not in resource_history.columns:
            row[feature] = None
            row[f"{feature}_available"] = False
            continue
        series = resource_history[["timestamp", feature]].dropna().sort_values("timestamp")
        row[feature] = None if series.empty else series[feature].iloc[-1]
        row[f"{feature}_available"] = not series.empty
    return row


def category_payload(category, resource_history, row, risk_score):
    subnet = slope_forecast(resource_history, "canonical_subnet_util_pct", [80, 90]) if "canonical_subnet_util_pct" in resource_history else {}
    return {
        "risk_score": round(risk_score, 2),
        "ddos_probability": round(min(1.0, positive(row.get("canonical_ddos_signal")) * 0.8 + risk_score / 500), 3),
        "connection_saturation_forecast": slope_forecast(resource_history, "canonical_active_connections", [10000, 25000]) if "canonical_active_connections" in resource_history else {},
        "subnet_capacity_forecast": subnet,
        "health_pct": clean(row.get("canonical_health_pct")),
        "available_signals": availability(category, row),
    }


def availability(category, row):
    return {
        feature: is_true_flag(row.get(f"{feature}_available"))
        for feature in canonical_features(category)
    }


def data_completeness_score(category, row):
    features = canonical_features(category)
    if not features:
        return 0.0
    available = sum(1 for feature in features if is_true_flag(row.get(f"{feature}_available")))
    return round(available / len(features), 3)


def failure_predictions(category, forecast, row):
    failures = []

    def add(metric, threshold, window, value, failure_type):
        if value is not None and not pd.isna(value) and value >= threshold:
            failures.append(
                {
                    "failure_type": failure_type,
                    "metric_name": metric,
                    "threshold": threshold,
                    "window": window,
                    "predicted_value": round(float(value), 3),
                }
            )

    metric = forecast.get("metric_name")
    if forecast.get("canonical_feature") == "canonical_active_connections":
        add(metric, 10000, "1h", forecast.get("forecast_1h"), "CONNECTION_SATURATION")
    subnet = row.get("canonical_subnet_util_pct")
    if subnet is not None and not pd.isna(subnet) and subnet >= 80:
        add("subnet_utilization_percent", 80, "current", subnet, "SUBNET_CAPACITY_RISK")
    if positive(row.get("canonical_ddos_signal")):
        add("ddos_signal", 1, "current", row.get("canonical_ddos_signal"), "DDOS_SIGNAL_DETECTED")
    health = row.get("canonical_health_pct")
    if health is not None and not pd.isna(health) and health < 90:
        add("health_pct", 90, "current", 100 - health, "BACKEND_HEALTH_DEGRADED")
    return failures


def recommendations(category, row, forecast, failures, anomaly_score):
    recs = []
    if any(item["failure_type"] == "DDOS_SIGNAL_DETECTED" for item in failures):
        recs.append("Enable or review DDoS protection and traffic filtering")
    if row.get("canonical_health_pct") is not None and not pd.isna(row.get("canonical_health_pct")) and row.get("canonical_health_pct") < 90:
        recs.append("Check unhealthy backend targets")
    if row.get("canonical_subnet_util_pct") is not None and not pd.isna(row.get("canonical_subnet_util_pct")) and row.get("canonical_subnet_util_pct") >= 80:
        recs.append("Expand subnet capacity or clean unused IP allocations")
    if anomaly_score >= 0.6:  # unsupervised composite — no labels, lower threshold
        recs.append("Investigate recent behavior change against baseline")
    return recs


def summary_for(category, row, risk_score):
    parts = []
    if positive(row.get("canonical_ddos_signal")):
        parts.append("DDoS signal present")
    if row.get("canonical_health_pct") is not None and not pd.isna(row.get("canonical_health_pct")):
        parts.append(f"health {row.get('canonical_health_pct'):.1f}%")
    if row.get("canonical_active_connections") is not None and not pd.isna(row.get("canonical_active_connections")):
        parts.append(f"connections {row.get('canonical_active_connections'):.0f}")
    if not parts:
        parts.append(f"Network risk score {risk_score:.0f}")
    return " + ".join(parts)


def prediction_type_for(category, payload):
    if payload.get("ddos_probability", 0) >= 0.5:
        return "anomaly"
    return "risk_score"


def generate_predictions(df, models, target_categories=("Network",)):
    predictions = []
    for _, row in latest_rows(df).iterrows():
        category = row["category"]
        if category not in target_categories:
            continue
        if not canonical_features(category):
            continue
        history = df[(df["resource_id"] == row["resource_id"]) & (df["category"] == category)]
        signal_row = consolidated_row(category, history, row)
        artifact = models.get(category)
        model_version = artifact["model_version"] if artifact else MODEL_VERSION.get(category, "unsupervised-v2.0")

        completeness = data_completeness_score(category, signal_row)
        if completeness == 0.0:
            risk_score = 0.0
            sev = "INFO"
            anomaly_score = 0.0
            is_anomalous = False
            alert_trigger = False
            anomaly_detail = {"error": "no_telemetry"}
            recs_override = ["No telemetry available for this resource at this timestamp — verify monitoring agent/collector is reporting"]
        else:
            # Unsupervised composite: rolling z-score + max/avg burst ratio + per-resource IsolationForest
            risk_score, anomaly_detail = score_resource(history, category)
            anomaly_score = round(max(0.0, min(1.0, risk_score / 100.0)), 3)
            is_anomalous = anomaly_score >= 0.6
            sev = severity(risk_score)
            alert_trigger = risk_score >= 40
            recs_override = None

        payload = category_payload(category, history, signal_row, risk_score)
        forecast = behavioral_forecast(category, history, row)
        failures = failure_predictions(category, forecast, signal_row)
        recs = recs_override if recs_override is not None else recommendations(category, signal_row, forecast, failures, anomaly_score)
        prediction = {
            "prediction_id": f"pred_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid4().hex[:6]}",
            "prediction_type": prediction_type_for(category, payload),
            "model_version": model_version,
            "generated_at": now_iso(),
            "resource_id": clean(row.get("resource_id")),
            "resource_name": clean(row.get("resource_name")),
            "account_id": clean(row.get("account_id")),
            "category": clean(category),
            "service_family": clean(category),
            "provider": clean(row.get("service_name")),
            "component": clean(row.get("component")),
            "location": clean(row.get("location")),
            "tags": clean(row.get("tags")),
            "prediction_timestamp": now_iso(),
            "data_completeness_score": completeness,
            "behavioral_forecast": forecast,
            "failure_predictions": failures,
            "anomaly_score": anomaly_score,
            "is_anomalous": is_anomalous,
            "recommendations": recs,
            "payload": payload,
            "anomaly_detail": anomaly_detail,
            "alert": {
                "trigger": alert_trigger,
                "severity": sev,
                "channel": channel_for(sev),
                "group_key": f"{row.get('account_id')}::{category}",
                "suppressed": sev == "INFO",
                "ttl_minutes": 60,
            },
            "dashboard": {
                "display_score": int(round(risk_score)),
                "display_label": "Risk score",
                "trend": "STABLE",
                "badge_color": badge(risk_score),
                "summary_text": summary_for(category, signal_row, risk_score),
            },
            "feedback": {
                "outcome_recorded": False,
                "outcome_label": None,
                "outcome_at": None,
            },
        }
        predictions.append(prediction)
    return predictions
