from pymongo import MongoClient


def get_collection(uri, db_name, collection_name):
    client = MongoClient(uri)
    return client[db_name][collection_name]


def load_documents(uri, db_name, collection_name, limit=50000, categories=None):
    collection = get_collection(uri, db_name, collection_name)
    query = {}
    if categories:
        query["category"] = {"$in": list(categories)}
    projection = {
        "resourceId": 1,
        "resource_id": 1,
        "element_id": 1,
        "inventory_id": 1,
        "resource_name": 1,
        "display_name": 1,
        "name": 1,
        "account_id": 1,
        "account_name": 1,
        "category": 1,
        "component": 1,
        "service_type": 1,
        "resource_type": 1,
        "location": 1,
        "service_name": 1,
        "provider": 1,
        "metric": 1,
        "metric_value": 1,
        "from_date": 1,
        "created_at": 1,
        "tags": 1,
    }
    cursor = collection.find(query, projection).sort([("from_date", 1), ("created_at", 1)]).batch_size(2000)
    if limit and limit > 0:
        cursor = cursor.limit(limit)
    try:
        return list(cursor)
    finally:
        cursor.close()


def _bson_safe(obj):
    """Recursively convert a prediction dict to BSON-safe types.

    MongoDB rejects: numpy scalars, numpy arrays, NaN/Infinity floats,
    pandas Timestamps, and any non-serializable nested objects.
    This sanitizer walks the entire document tree and converts everything
    to plain Python int/float/str/list/dict/None.
    """
    import math
    import numpy as np
    import pandas as pd

    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [_bson_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _bson_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_bson_safe(v) for v in obj]
    # Last resort: stringify unknown types
    try:
        return str(obj)
    except Exception:
        return None


def write_predictions(uri, db_name, collection_name, predictions):
    if not predictions:
        return 0
    collection = get_collection(uri, db_name, collection_name)
    # Sanitize every document before sending to MongoDB.
    # insert_many fails with a low-level BSON error if any value is a numpy
    # scalar, NaN, Infinity, or other non-serializable type.
    safe = [_bson_safe(p) for p in predictions]
    # Insert in batches of 500 to avoid hitting the 16 MB BSON document size limit
    batch_size = 500
    inserted = 0
    for i in range(0, len(safe), batch_size):
        batch = safe[i:i + batch_size]
        collection.insert_many(batch)
        inserted += len(batch)
    return inserted
