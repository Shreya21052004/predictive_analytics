from collections import defaultdict
from datetime import datetime, timezone
import math
import pandas as pd

from .config import CATEGORY_ALIASES, PROVIDER_ALIASES
from .metric_registry import REGISTRY


def normalize_category(value):
    if value is None:
        return None
    return CATEGORY_ALIASES.get(str(value).strip().lower(), str(value).strip())


def normalize_provider(value):
    if value is None:
        return "unknown"
    return PROVIDER_ALIASES.get(str(value).strip().lower(), str(value).strip())


def get_ci(d, *keys):
    """Look up the first present key in d, case-insensitively.

    Source documents mix casing conventions across providers/ingestion paths
    (e.g. CloudWatch-style {"average": ...} vs GCP-derived
    {"Average": ..., "Sum": ..., "SampleCount": ...}). This checks each
    candidate key against a lowercased map of d's keys so either casing works.
    """
    if not isinstance(d, dict):
        return None
    lowered = {str(k).lower(): v for k, v in d.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None:
            return value
    return None


# Stat fields in priority order. Average first for utilisation-style metrics,
# but we also record WHICH stat was used so mixed-stat series can be detected.
STAT_PRIORITY = ["average", "maximum", "sum", "total", "minimum", "count", "samplecount", "value"]


def extract_metric_value(doc):
    
    """
    Extract (numeric_value, stat_field) from a doc.

    Handles:
    - metric_value.Average / metric_value.average
    - metric_value.Maximum / metric_value.maximum
    - metric_value.Minimum, Sum, Total, etc.
    - Top-level doc fields as fallback
    - Filters NaN values
    """

    metric_value = doc.get("metric_value")

    if isinstance(metric_value, dict):
        lowered = {str(k).lower(): (k, v) for k, v in metric_value.items()}

        for stat in STAT_PRIORITY:
            entry = lowered.get(stat)

            if entry is None or entry[1] is None:
                continue

            try:
                value = float(entry[1])

                if math.isnan(value):
                    continue

                return value, stat

            except (TypeError, ValueError):
                continue

    # fallback: top-level doc keys
    lowered = {str(k).lower(): (k, v) for k, v in doc.items()}

    for stat in STAT_PRIORITY:
        entry = lowered.get(stat)

        if entry is None or entry[1] is None:
            continue

        try:
            value = float(entry[1])

            if math.isnan(value):
                continue

            return value, stat

        except (TypeError, ValueError):
            continue

    return None, None


def extract_stat_bundle(item):
    """Extract Average, Maximum, Minimum, SampleCount from a single metric_value item.

    The normaliser picks Average as the canonical value but discards Maximum/Minimum.
    This function returns the full bundle so anomaly features (max/avg ratio,
    intra-hour range) can be computed without re-reading the raw documents.

    Returns dict with keys: average, maximum, minimum, sample_count (all float or None).
    """
    if not isinstance(item, dict):
        return {"average": None, "maximum": None, "minimum": None, "sample_count": None}
    lowered = {str(k).lower(): v for k, v in item.items()}
    def _f(key):
        v = lowered.get(key)

        if v is None:
            return None

        try:
            value = float(v)

            if math.isnan(value):
                return None

            return value

        except (TypeError, ValueError):
            return None
    return {
        "average":      _f("average"),
        "maximum":      _f("maximum"),
        "minimum":      _f("minimum"),
        "sample_count": _f("samplecount"),
    }


def parse_datetime(raw):
    if raw is None:
        return datetime.now(timezone.utc)
    if isinstance(raw, dict) and "$date" in raw:
        raw = raw["$date"]
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(parsed):
        return datetime.now(timezone.utc)
    return parsed.to_pydatetime()


def extract_timestamp(doc):
    metric_value = doc.get("metric_value")
    raw = None
    if isinstance(metric_value, dict):
        raw = get_ci(metric_value, "timestamp")
    raw = raw or doc.get("from_date") or doc.get("created_at")
    return parse_datetime(raw)


def metric_points(doc):
    """Return list of (timestamp, value, stat_field, bundle) tuples for a doc.

    bundle: dict with average, maximum, minimum, sample_count — used for
    anomaly features (max/avg ratio, intra-hour range).
    stat_field: which stat field the canonical value came from.
    """
    metric_value = doc.get("metric_value")
    if isinstance(metric_value, list):
        points = []
        for item in metric_value:
            if not isinstance(item, dict):
                continue
            ts_raw = get_ci(item, "timestamp") or doc.get("from_date") or doc.get("created_at")
            value, stat = extract_metric_value({"metric_value": item})
            bundle = extract_stat_bundle(item)
            points.append((parse_datetime(ts_raw), value, stat, bundle))
        return points
    value, stat = extract_metric_value(doc)
    bundle = extract_stat_bundle(doc.get("metric_value") or doc)
    return [(extract_timestamp(doc), value, stat, bundle)]


def resource_id(doc):
    return doc.get("element_id") or doc.get("resourceId") or doc.get("resource_id") or doc.get("inventory_id") or doc.get("_id")


def resource_name(doc):
    return doc.get("resource_name") or doc.get("display_name") or doc.get("name") or str(resource_id(doc))


def provider(doc):
    return normalize_provider(doc.get("service_name") or doc.get("provider") or "unknown")


def base_context(doc):
    return {
        "resource_id": str(resource_id(doc)),
        "resource_name": resource_name(doc),
        "account_id": doc.get("account_id"),
        "account_name": doc.get("account_name"),
        "category": normalize_category(doc.get("category")),
        "component": doc.get("component") or doc.get("service_type") or doc.get("resource_type"),
        "location": doc.get("location"),
        "service_name": provider(doc),
        "tags": doc.get("tags"),
    }


def normalize_documents(documents):
    grouped = {}
    raw_metrics = defaultdict(dict)
    resource_tags = {}  # last-seen tags per resource_id (tags are static metadata)

    for doc in documents:
        category = normalize_category(doc.get("category"))
        metric = doc.get("metric")
        if not category or not metric:
            continue
        rid = str(resource_id(doc))
        # Track last-seen tags per resource (tags are typically static per resource)
        if doc.get("tags") is not None:
            resource_tags[rid] = doc.get("tags")
        for ts, value, stat, bundle in metric_points(doc):
            key = (rid, category, ts)
            if key not in grouped:
                grouped[key] = {
                    **base_context(doc),
                    "timestamp": ts,
                }
            if value is not None:
                # Track (value, stat_field, bundle) so we can detect mixed-stat series
                # and derive anomaly features (max/avg ratio, intra-hour range)
                existing = raw_metrics[key].get(metric)
                if existing is None:
                    raw_metrics[key][metric] = (value, stat, bundle)
                else:
                    existing_val, existing_stat, _ = existing
                    if stat == "average" and existing_stat != "average":
                        raw_metrics[key][metric] = (value, stat, bundle)
                    # else keep existing

    rows = []
    for key, row in grouped.items():
        category = row["category"]
        service_name = row["service_name"]
        rid = key[0]
        # Propagate last-seen tags for this resource (may be None if none found)
        row["tags"] = resource_tags.get(rid, row.get("tags"))
        metrics = raw_metrics[key]
        for feature, provider_map in REGISTRY.get(category, {}).items():
            value = None
            stat_used = None
            available = False
            f_max = None
            f_min = None
            f_sample_count = None
            transforms = provider_map.get(service_name, {})
            for raw_name, transform in transforms.items():
                if raw_name in metrics and metrics[raw_name] is not None:
                    try:
                        raw_val, stat_used, bundle = metrics[raw_name]
                        value = transform(raw_val)
                        available = True
                        # Also store Maximum/Minimum/SampleCount from the same item
                        # so anomaly features can be derived without re-reading docs
                        if bundle:
                            bmax = bundle.get("maximum")
                            bmin = bundle.get("minimum")
                            bsc  = bundle.get("sample_count")
                            f_max          = transform(bmax) if bmax is not None else None
                            f_min          = transform(bmin) if bmin is not None else None
                            f_sample_count = bsc
                        break
                    except (TypeError, ValueError):
                        value = None
            row[feature] = value
            row[f"{feature}_available"] = available
            row[f"{feature}_stat"] = stat_used
            row[f"{feature}_max"] = f_max        # Maximum within this hour-bucket
            row[f"{feature}_min"] = f_min        # Minimum within this hour-bucket
            row[f"{feature}_sample_count"] = f_sample_count  # data density
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["category", "resource_id", "timestamp"]).reset_index(drop=True)
