def pct(value):
    if value is None:
        return None
    value = float(value)
    if 0 <= value <= 1:
        return value * 100.0
    return value


def identity(value):
    if value is None:
        return None
    return float(value)


REGISTRY = {
    "Network": {
        "canonical_net_in_bytes": {
            "AWS": {"BytesIn": identity, "PacketsIn": identity},
            "Azure": {"ByteCount": identity, "PEBytesIn": identity, "PacketCount": identity},
        },
        "canonical_net_out_bytes": {
            "AWS": {"BytesOut": identity, "PacketsOut": identity},
            "Azure": {"PEBytesOut": identity},
        },
        "canonical_active_connections": {
            "AWS": {"ActiveConnectionCount": identity},
            "Azure": {"TotalConnectionCount": identity},
        },
        "canonical_health_pct": {
            "AWS": {"HealthCheckPercentageHealthy": pct},
            "Azure": {"VipAvailability": pct, "DipAvailability": pct},
        },
        "canonical_health_status": {
            "AWS": {"HealthCheckStatus": identity},
        },
        "canonical_ddos_signal": {
            "AWS": {"PacketDropCountBlackhole": identity, "PacketDropCountNoRoute": identity},
            "Azure": {
                "IfUnderDDoSAttack": identity,
                "PacketsInDDoS": identity,
                "BytesInDDoS": identity,
                "PacketsDroppedDDoS": identity,
                "BytesDroppedDDoS": identity,
                "PacketsForwardedDDoS": identity,
                "BytesForwardedDDoS": identity,
                "TCPPacketsInDDoS": identity,
                "TCPBytesInDDoS": identity,
                "TCPPacketsDroppedDDoS": identity,
                "TCPBytesDroppedDDoS": identity,
                "TCPPacketsForwardedDDoS": identity,
                "TCPBytesForwardedDDoS": identity,
                "UDPPacketsInDDoS": identity,
                "UDPBytesInDDoS": identity,
                "SynCount": identity,
            },
        },
        "canonical_subnet_util_pct": {
            "Azure": {
                "VirtualNetworkLinkCapacityUtilization": pct,
                "RecordSetCapacityUtilization": pct,
                "VirtualNetworkWithRegistrationCapacityUtilization": pct,
            },
        },
        "canonical_dns_queries": {
            "AWS": {"InboundQueryVolume": identity},
            "Azure": {"QueryVolume": identity, "RecordSetCount": identity},
        },
        "canonical_throughput": {
            "Azure": {"Throughput": identity, "TotalRequests": identity},
        },
        "canonical_vnet_link_count": {
            "Azure": {"VirtualNetworkLinkCount": identity, "VirtualNetworkWithRegistrationLinkCount": identity},
        },
    },
}


def canonical_features(category):
    return list(REGISTRY.get(category, {}).keys())


PRIMARY_FORECAST_FEATURE = {
    "Network": "canonical_active_connections",
}


def primary_metric_name(category, provider):
    feature = PRIMARY_FORECAST_FEATURE.get(category)
    return metric_name_for_feature(category, provider, feature)


def metric_name_for_feature(category, provider, feature):
    if not feature:
        return None
    provider_map = REGISTRY.get(category, {}).get(feature, {})
    raw_metrics = provider_map.get(provider, {})
    if raw_metrics:
        return next(iter(raw_metrics.keys()))
    return feature
