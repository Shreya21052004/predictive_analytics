from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

from .metric_registry import canonical_features


MODEL_VERSION = {
    "Network": "network-risk-v1.0",
}


def is_real(value):
    return value is not None and not pd.isna(value)


def ge(row, key, threshold):
    value = row.get(key)
    return is_real(value) and value >= threshold


def gt(row, key, threshold):
    value = row.get(key)
    return is_real(value) and value > threshold


def lt(row, key, threshold):
    value = row.get(key)
    return is_real(value) and value < threshold


def feature_columns(df, category):
    bases = canonical_features(category)
    numeric_cols = []
    categorical_cols = []
    for col in bases:
        if col in df.columns:
            numeric_cols.append(col)
        available = f"{col}_available"
        if available in df.columns:
            numeric_cols.append(available)
    for col in ("service_name", "component", "location"):
        if col in df.columns:
            categorical_cols.append(col)
    return numeric_cols + categorical_cols, numeric_cols, categorical_cols


def build_preprocessor(df, numeric_cols, categorical_cols, for_xgboost=False):
    """XGBoost path: passthrough numerics (handles NaN, scale-invariant).
    IsolationForest/RandomForest path: median imputation + RobustScaler."""
    transformers = []
    if numeric_cols:
        if for_xgboost:
            from sklearn.preprocessing import FunctionTransformer
            transformers.append(("num", FunctionTransformer(), numeric_cols))
        else:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ])
            transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop")


def weak_risk_label(row, category, col_stats=None):
    """Threshold-based risk labeller for Network, with z-score fallback."""
    score = 0
    score += 45 if gt(row, "canonical_ddos_signal", 0) else 0
    score += 35 if lt(row, "canonical_health_pct", 90) else 0
    score += 30 if ge(row, "canonical_subnet_util_pct", 80) else 0
    score += 15 if gt(row, "canonical_active_connections", 10000) else 0

    if score >= 50:
        return 1

    if col_stats:
        for col, (mean, std) in col_stats.items():
            val = row.get(col)
            if val is not None and not pd.isna(val) and std > 0:
                if abs(float(val) - mean) > 2 * std:
                    return 1
    return 0


def add_missingness_features(df, numeric_cols):
    df = df.copy()
    for col in numeric_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def time_aware_split(df, test_frac=0.2):
    """Per-resource split: oldest (1-test_frac) rows train, newest test_frac test."""
    train_idx, test_idx = [], []
    for _, grp in df.groupby("resource_id", sort=False):
        grp_sorted = grp.sort_values("timestamp")
        n = len(grp_sorted)
        split_point = max(1, int(n * (1 - test_frac)))
        train_idx.extend(grp_sorted.index[:split_point].tolist())
        test_idx.extend(grp_sorted.index[split_point:].tolist())
    return df.loc[train_idx], df.loc[test_idx]


def evaluate_classifier(model, X_test, y_test):
    """Labels come from weak_risk_label() (rules + z-score), not real incidents.
    Read accuracy/ROC-AUC with that in mind — use F1/recall as the real signal,
    and check label_positive_rate before trusting the numbers."""
    if len(X_test) == 0:
        return {}
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred).astype(int)
    y_test_arr = np.array(y_test).astype(int)

    label_positive_rate = round(float(y_test_arr.mean()), 4)
    majority_class = int(y_test_arr.mean() >= 0.5)
    baseline_acc = round(float((y_test_arr == majority_class).mean()), 4)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test_arr, y_pred)), 4),
        "baseline_accuracy": baseline_acc,
        "precision": round(float(precision_score(y_test_arr, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test_arr, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test_arr, y_pred, zero_division=0)), 4),
        "label_positive_rate": label_positive_rate,
        "model_type": "xgboost" if HAS_XGBOOST else "random_forest",
        "label_source": "weak_rule_labels",
    }
    estimator = model.named_steps["estimator"]
    if hasattr(estimator, "predict_proba") and y_test.nunique() > 1:
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test_arr, y_prob)), 4)
        except Exception:
            pass
    metrics["test_size"] = len(X_test)
    return metrics


def evaluate_anomaly(model, X_test, y_test):
    if len(X_test) == 0:
        return {}
    scores = model.decision_function(X_test)
    predicted_anomaly_rate = float((scores < 0).sum() / len(scores))
    actual_risk_rate = float(y_test.mean()) if len(y_test) > 0 else 0.0
    return {
        "predicted_anomaly_rate": round(predicted_anomaly_rate, 4),
        "actual_risk_rate": round(actual_risk_rate, 4),
        "test_size": len(X_test),
    }


def train_category(df, category):
    category_df = df[df["category"] == category].copy()
    if category_df.empty:
        return None

    cols, numeric_cols, categorical_cols = feature_columns(category_df, category)
    if not cols:
        return None

    for col in numeric_cols:
        category_df[col] = pd.to_numeric(category_df[col], errors="coerce")

    col_stats = {
        col: (float(category_df[col].mean()), float(category_df[col].std()))
        for col in numeric_cols
        if col in category_df.columns and category_df[col].std() > 0
    }
    y_all = category_df.apply(lambda row: weak_risk_label(row, category, col_stats), axis=1)

    category_df = add_missingness_features(category_df, numeric_cols)
    missing_cols = [f"{c}_missing" for c in numeric_cols if f"{c}_missing" in category_df.columns]
    all_cols = cols + missing_cols

    train_df, test_df = time_aware_split(category_df)
    X_train = train_df[all_cols].copy()
    y_train = y_all.loc[train_df.index]
    X_test = test_df[all_cols].copy()
    y_test = y_all.loc[test_df.index]

    kind = "classifier" if (y_train.nunique() > 1 and len(train_df) >= 10) else "anomaly"
    preprocessor = build_preprocessor(X_train, numeric_cols + missing_cols, categorical_cols, for_xgboost=(kind == "classifier" and HAS_XGBOOST))

    if kind == "classifier":
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        scale_pos = neg / pos if pos > 0 else 1.0
        if HAS_XGBOOST:
            estimator = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=42,
                missing=np.nan,
                verbosity=0,
            )
        else:
            estimator = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
    else:
        estimator = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            max_samples="auto",
            random_state=42,
        )

    model = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    if kind == "classifier":
        model.fit(X_train, y_train)
        eval_metrics = evaluate_classifier(model, X_test, y_test)
    else:
        model.fit(X_train)
        eval_metrics = evaluate_anomaly(model, X_test, y_test)

    return {
        "category": category,
        "kind": kind,
        "model": model,
        "features": all_cols,
        "model_version": MODEL_VERSION[category],
        "eval_metrics": eval_metrics,
    }


def train_models(df, models_dir):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    trained = {}
    for category in ("Network",):
        artifact = train_category(df, category)
        if artifact is None:
            continue
        path = Path(models_dir) / f"{category.lower()}_model.joblib"
        joblib.dump(artifact, path)
        trained[category] = artifact
    return trained


def load_models(models_dir):
    loaded = {}
    for category in ("Network",):
        path = Path(models_dir) / f"{category.lower()}_model.joblib"
        if path.exists():
            loaded[category] = joblib.load(path)
    return loaded


def risk_from_model(artifact, row):
    X = pd.DataFrame([{col: row.get(col) for col in artifact["features"]}])
    numeric_cols = [c for c in artifact["features"] if c not in ("service_name", "component", "location")]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    estimator = artifact["model"].named_steps["estimator"]
    if artifact["kind"] == "classifier" and hasattr(estimator, "predict_proba"):
        return float(artifact["model"].predict_proba(X)[0, 1] * 100.0)

    score = float(artifact["model"].decision_function(X)[0])
    return float(np.clip((0.12 - score) / 0.24 * 100.0, 0, 100))
