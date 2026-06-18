from .metric_registry import canonical_features


PCT_FEATURES = {
    "canonical_cpu_pct",
    "canonical_cpu_idle_pct",
    "canonical_iowait_pct",
    "canonical_mem_used_pct",
    "canonical_swap_used_pct",
    "canonical_disk_used_pct",
    "canonical_health_pct",
    "canonical_subnet_util_pct",
    "canonical_storage_used_pct",
    "canonical_burst_balance_pct",
    "availability_pct",
}


def validate_normalized(df):
    issues = []
    if df.empty:
        return ["No normalized rows were produced. Check category, metric, and metric_value fields."]

    for category in sorted(df["category"].dropna().unique()):
        cat_df = df[df["category"] == category]
        for feature in canonical_features(category):
            if feature not in cat_df.columns:
                continue
            if feature not in PCT_FEATURES:
                continue
            for provider, provider_df in cat_df.groupby("service_name", dropna=False):
                series = provider_df[feature].dropna()
                if series.empty:
                    continue
                if series.max() > 150 or series.min() < -5:
                    issues.append(
                        f"{category}.{feature} ({provider}) has suspicious percent range "
                        f"min={series.min():.2f}, max={series.max():.2f}"
                    )
    return issues
