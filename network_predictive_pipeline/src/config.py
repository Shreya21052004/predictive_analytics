from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    mongo_uri: str = "mongodb://localhost:27017"
    db_name: str = "mydb"
    source_collection: str = "network_data"
    output_collection: str = "22prediction_logs_network"
    models_dir: Path = Path("models")
    min_points_for_model: int = 20
    limit: int = 0  # 0 = load all documents


# Network-only pipeline. Kept separate from Compute/Storage on purpose so it
# can be run independently without the extra memory/CPU load of the other
# categories. Wire-up to the unified pipeline comes later.
TARGET_CATEGORIES = ("Network",)

CATEGORY_ALIASES = {
    "network": "Network",
    "networking": "Network",
}

SUPPORTED_CATEGORIES = ("Network",)

PROVIDER_ALIASES = {
    "aws": "AWS",
    "amazon": "AWS",
    "amazon web services": "AWS",
    "azure": "Azure",
    "microsoft azure": "Azure",
    "gcp": "GCP",
    "google": "GCP",
    "google cloud": "GCP",
    "google cloud platform": "GCP",
    "oci": "OCI",
    "oracle": "OCI",
    "oracle cloud": "OCI",
}

RISK_THRESHOLDS = {
    "INFO": 25,
    "WARNING": 50,
    "HIGH": 75,
    "CRITICAL": 90,
}
