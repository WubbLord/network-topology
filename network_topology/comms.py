from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .common import coerce_float, deep_find_first
from .topology import compute_average_hops, node_count


def infer_remote_traffic(mapping_stats: Mapping[str, Any]) -> dict[str, Any]:
    raw = mapping_stats.get("raw", {})
    notes: list[str] = []

    remote_bytes = _find_numeric(
        mapping_stats,
        raw,
        keys=(
            "remote_bytes",
            "inter_device_bytes",
            "network_bytes",
            "byte_traffic",
            "traffic_bytes",
        ),
    )
    message_count = _find_numeric(
        mapping_stats,
        raw,
        keys=("message_count", "messages", "num_messages", "network_messages"),
    )
    num_accesses = _find_numeric(
        mapping_stats,
        raw,
        keys=("remote_accesses", "num_accesses", "access_count", "network_accesses"),
    )
    byte_hops = _find_numeric(
        mapping_stats,
        raw,
        keys=("byte_hops", "network_byte_hops", "remote_byte_hops"),
    )
    src_dst_pairs = _extract_src_dst_pairs(mapping_stats, raw)

    if remote_bytes is not None:
        notes.append("Used direct remote-byte signal from mapping output.")
    else:
        remote_bytes, approx_accesses, approx_pairs = _infer_from_component_actions(
            mapping_stats.get("per_component_actions", [])
        )
        if remote_bytes:
            notes.append("Approximated remote traffic from interconnect-like component actions.")
            if num_accesses is None:
                num_accesses = approx_accesses
            if not src_dst_pairs:
                src_dst_pairs = approx_pairs
        else:
            remote_bytes = 0.0

    if remote_bytes == 0.0:
        tensor_bytes, tensor_accesses = _infer_from_tensor_stats(mapping_stats.get("per_tensor_stats", []))
        if tensor_bytes > 0:
            remote_bytes = tensor_bytes
            if num_accesses is None:
                num_accesses = tensor_accesses
            notes.append(
                "Approximated remote traffic from tensor partitioning metadata because direct traffic was unavailable."
            )

    if num_accesses is None:
        num_accesses = message_count or 0.0

    if remote_bytes == 0.0:
        notes.append("No remote traffic signal was found; returning zero-byte traffic.")

    return {
        "remote_bytes": float(remote_bytes),
        "message_count": int(math.ceil(message_count)) if message_count is not None else None,
        "num_accesses": float(num_accesses),
        "src_dst_pairs": src_dst_pairs,
        "byte_hops": byte_hops,
        "notes": notes,
    }


def estimate_comm_from_mapping(
    mapping_stats: Mapping[str, Any],
    topology_config: Mapping[str, Any],
) -> dict[str, Any]:
    traffic = infer_remote_traffic(mapping_stats)
    notes = list(traffic.get("notes", []))
    remote_bytes = float(traffic["remote_bytes"])

    default_message_size = float(topology_config.get("default_message_size_bytes", 262_144))
    message_count = traffic.get("message_count")
    if message_count is None:
        message_count = int(math.ceil(remote_bytes / default_message_size)) if remote_bytes > 0 else 0
        notes.append("Estimated message count from remote bytes and default_message_size_bytes.")

    average_hops = compute_average_hops(topology_config, traffic.get("src_dst_pairs"))
    byte_hops = traffic.get("byte_hops")
    if byte_hops is None:
        byte_hops = remote_bytes * average_hops

    remote_bytes, message_count, average_hops, byte_hops, collective_notes = _apply_collective_adjustment(
        remote_bytes=remote_bytes,
        message_count=message_count,
        average_hops=average_hops,
        byte_hops=byte_hops,
        topology_config=topology_config,
        has_explicit_pairs=bool(traffic.get("src_dst_pairs")),
        has_direct_byte_hops=traffic.get("byte_hops") is not None,
    )
    notes.extend(collective_notes)

    energy_per_byte_hop = float(topology_config.get("energy_per_byte_hop_j", 0.0))
    per_hop_latency = float(topology_config.get("per_hop_latency_s", 0.0))
    link_bandwidth = topology_config.get("link_bandwidth_bytes_per_s")
    bandwidth_value = coerce_float(link_bandwidth)
    if bandwidth_value is None or bandwidth_value <= 0:
        serialization_latency = 0.0
        notes.append("link_bandwidth_bytes_per_s is missing or non-positive; serialization latency is treated as zero.")
    else:
        serialization_latency = byte_hops / bandwidth_value

    propagation_latency = message_count * average_hops * per_hop_latency
    total_network_energy = byte_hops * energy_per_byte_hop
    total_network_latency = serialization_latency + propagation_latency

    return {
        "total_network_energy_j": total_network_energy,
        "total_network_latency_s": total_network_latency,
        "total_remote_bytes": remote_bytes,
        "total_byte_hops": byte_hops,
        "message_count": int(message_count),
        "average_hops": average_hops,
        "num_accesses": float(traffic.get("num_accesses", 0.0)),
        "effective_cost_mode": str(topology_config.get("effective_cost_mode", "per_byte")).lower(),
        "notes": notes,
    }


def compute_effective_comm_cost(comm_estimate: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(comm_estimate.get("effective_cost_mode", "per_byte")).lower()
    total_energy = float(comm_estimate.get("total_network_energy_j", 0.0))
    total_latency = float(comm_estimate.get("total_network_latency_s", 0.0))

    if mode == "per_access":
        denominator = float(comm_estimate.get("num_accesses", 0.0))
        if denominator <= 0:
            return {
                "mode": mode,
                "energy_per_access": 0.0,
                "latency_per_access": 0.0,
                "denominator": denominator,
                "notes": ["num_accesses was zero; effective per-access cost was clamped to zero."],
            }
        return {
            "mode": mode,
            "energy_per_access": total_energy / denominator,
            "latency_per_access": total_latency / denominator,
            "denominator": denominator,
            "notes": [],
        }

    denominator = float(comm_estimate.get("total_remote_bytes", 0.0))
    if denominator <= 0:
        return {
            "mode": "per_byte",
            "energy_per_byte": 0.0,
            "latency_per_byte": 0.0,
            "denominator": denominator,
            "notes": ["total_remote_bytes was zero; effective per-byte cost was clamped to zero."],
        }
    return {
        "mode": "per_byte",
        "energy_per_byte": total_energy / denominator,
        "latency_per_byte": total_latency / denominator,
        "denominator": denominator,
        "notes": [],
    }


def _find_numeric(*sources: Any, keys: tuple[str, ...]) -> float | None:
    for source in sources:
        value = deep_find_first(source, keys)
        coerced = coerce_float(value)
        if coerced is not None:
            return coerced
    return None


def _extract_src_dst_pairs(*sources: Any) -> list[dict[str, Any]]:
    for source in sources:
        candidate = deep_find_first(source, ("src_dst_pairs", "traffic_pairs", "flows", "flow_pairs"))
        if not isinstance(candidate, list):
            continue
        normalized: list[dict[str, Any]] = []
        for pair in candidate:
            if not isinstance(pair, Mapping):
                continue
            src = pair.get("src", pair.get("source"))
            dst = pair.get("dst", pair.get("destination"))
            if src is None or dst is None:
                continue
            normalized_pair = {"src": src, "dst": dst}
            bytes_value = (
                coerce_float(pair.get("bytes"))
                or coerce_float(pair.get("traffic_bytes"))
                or coerce_float(pair.get("weight"))
            )
            if bytes_value is not None:
                normalized_pair["bytes"] = bytes_value
            normalized.append(normalized_pair)
        if normalized:
            return normalized
    return []


def _infer_from_component_actions(component_actions: Any) -> tuple[float, float, list[dict[str, Any]]]:
    if not isinstance(component_actions, list):
        return 0.0, 0.0, []

    remote_keywords = ("remote", "interconnect", "network", "ici", "link", "collective", "logical")
    remote_bytes = 0.0
    remote_accesses = 0.0
    pairs: list[dict[str, Any]] = []

    for entry in component_actions:
        if not isinstance(entry, Mapping):
            continue
        text_fields = " ".join(
            str(entry.get(key, ""))
            for key in ("component", "name", "level", "target", "path")
        ).lower()
        is_remote = bool(entry.get("is_remote")) or bool(entry.get("above_hbm"))
        if not is_remote and not any(keyword in text_fields for keyword in remote_keywords):
            continue

        bytes_value = (
            coerce_float(entry.get("bytes"))
            or coerce_float(entry.get("total_bytes"))
            or coerce_float(entry.get("traffic_bytes"))
        )
        count_value = (
            coerce_float(entry.get("count"))
            or coerce_float(entry.get("accesses"))
            or coerce_float(entry.get("messages"))
        )
        bytes_per_action = (
            coerce_float(entry.get("bytes_per_action"))
            or coerce_float(entry.get("bytes_per_access"))
            or coerce_float(entry.get("bytes_per_message"))
        )
        if bytes_value is None and count_value is not None and bytes_per_action is not None:
            bytes_value = count_value * bytes_per_action
        if bytes_value is None:
            continue

        remote_bytes += bytes_value
        remote_accesses += count_value or 0.0
        src = entry.get("src", entry.get("source"))
        dst = entry.get("dst", entry.get("destination"))
        if src is not None and dst is not None:
            pairs.append({"src": src, "dst": dst, "bytes": bytes_value})

    return remote_bytes, remote_accesses, pairs


def _infer_from_tensor_stats(tensor_stats: Any) -> tuple[float, float]:
    if not isinstance(tensor_stats, list):
        return 0.0, 0.0

    remote_bytes = 0.0
    remote_accesses = 0.0
    remote_keywords = ("remote", "interconnect", "network", "logical", "device")

    for entry in tensor_stats:
        if not isinstance(entry, Mapping):
            continue
        base_bytes = (
            coerce_float(entry.get("bytes"))
            or coerce_float(entry.get("size_bytes"))
            or coerce_float(entry.get("payload_bytes"))
        )
        if base_bytes is None:
            continue

        accesses = (
            coerce_float(entry.get("accesses"))
            or coerce_float(entry.get("count"))
            or coerce_float(entry.get("uses"))
            or 1.0
        )
        remote_fraction = (
            coerce_float(entry.get("remote_fraction"))
            or coerce_float(entry.get("nonlocal_fraction"))
            or coerce_float(entry.get("offchip_fraction"))
        )
        shard_count = (
            coerce_float(entry.get("shard_count"))
            or coerce_float(entry.get("partitions"))
            or coerce_float(entry.get("replica_count"))
            or coerce_float(entry.get("device_count"))
            or 1.0
        )

        placement = str(entry.get("placement", entry.get("level", ""))).lower()
        if remote_fraction is None:
            if shard_count > 1:
                remote_fraction = (shard_count - 1.0) / shard_count
            elif any(keyword in placement for keyword in remote_keywords):
                remote_fraction = 1.0
            else:
                continue

        remote_bytes += base_bytes * accesses * remote_fraction
        remote_accesses += accesses

    return remote_bytes, remote_accesses


def _apply_collective_adjustment(
    *,
    remote_bytes: float,
    message_count: int,
    average_hops: float,
    byte_hops: float,
    topology_config: Mapping[str, Any],
    has_explicit_pairs: bool,
    has_direct_byte_hops: bool,
) -> tuple[float, int, float, float, list[str]]:
    collective = topology_config.get("collective")
    if not isinstance(collective, Mapping):
        return remote_bytes, message_count, average_hops, byte_hops, []
    if has_explicit_pairs or has_direct_byte_hops:
        return remote_bytes, message_count, average_hops, byte_hops, []

    collective_type = str(collective.get("type", "")).lower()
    algorithm = str(collective.get("algorithm", "")).lower()
    participants = int(collective.get("participants", node_count(topology_config)))
    notes: list[str] = []

    if collective_type == "allreduce" and algorithm == "ring" and participants > 1 and remote_bytes > 0:
        multiplier = 2.0 * (participants - 1.0) / participants
        adjusted_remote_bytes = remote_bytes * multiplier
        adjusted_message_count = max(message_count, 2 * (participants - 1))
        adjusted_average_hops = 1.0
        adjusted_byte_hops = adjusted_remote_bytes * adjusted_average_hops
        notes.append("Applied ring allreduce expansion to remote bytes before cost writeback.")
        return (
            adjusted_remote_bytes,
            adjusted_message_count,
            adjusted_average_hops,
            adjusted_byte_hops,
            notes,
        )

    notes.append("Collective configuration was present but no specialized model was applied.")
    return remote_bytes, message_count, average_hops, byte_hops, notes
