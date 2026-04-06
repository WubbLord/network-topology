from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import coerce_float, deep_find_first, normalize_key, stable_signature, to_builtin


def extract_mapping_stats(mapping_result: Any) -> dict[str, Any]:
    raw = to_builtin(mapping_result)
    best_container = _select_best_container(raw)
    mapping_structure = _extract_mapping_structure(best_container, raw)
    total_energy = _extract_numeric(best_container, raw, keys=("total_energy", "energy", "mapper_energy"))
    total_latency = _extract_numeric(best_container, raw, keys=("total_latency", "latency", "mapper_latency"))
    per_component_actions = _extract_named_entries(
        best_container,
        raw,
        candidate_keys=("per_component_actions", "component_actions", "actions_by_component", "components"),
        label_key="component",
        fallback_label="component",
    )
    per_tensor_stats = _extract_named_entries(
        best_container,
        raw,
        candidate_keys=("per_tensor_stats", "tensor_stats", "tensors", "tensor_accesses"),
        label_key="tensor",
        fallback_label="tensor",
    )

    notes: list[str] = []
    if total_energy is None:
        notes.append("Mapper energy was not found; total_energy is None.")
    if total_latency is None:
        notes.append("Mapper latency was not found; total_latency is None.")
    if not per_component_actions:
        notes.append("Per-component actions were not found; communication inference may fall back to tensor heuristics.")
    if not per_tensor_stats:
        notes.append("Per-tensor stats were not found; tensor-based communication inference may be unavailable.")

    return {
        "total_energy": total_energy,
        "total_latency": total_latency,
        "per_component_actions": per_component_actions,
        "per_tensor_stats": per_tensor_stats,
        "mapping_structure": mapping_structure,
        "mapping_signature": stable_signature(mapping_structure if mapping_structure is not None else best_container),
        "raw": raw,
        "notes": notes,
    }


def _select_best_container(raw: Any) -> Any:
    if not isinstance(raw, Mapping):
        return raw

    preferred_keys = (
        "best_result",
        "best_mapping",
        "best",
        "mapping_result",
        "mapping",
        "result",
    )
    for candidate_key in preferred_keys:
        for key, value in raw.items():
            if normalize_key(key) == normalize_key(candidate_key):
                return value
    return raw


def _extract_mapping_structure(best_container: Any, raw: Any) -> Any:
    for source in (best_container, raw):
        candidate = deep_find_first(
            source,
            ("mapping_structure", "mapping", "schedule", "mapping_tree", "placement"),
        )
        if candidate is not None:
            return candidate
    return best_container


def _extract_numeric(*sources: Any, keys: tuple[str, ...]) -> float | None:
    for source in sources:
        value = deep_find_first(source, keys)
        coerced = coerce_float(value)
        if coerced is not None:
            return coerced
    return None


def _extract_named_entries(
    *sources: Any,
    candidate_keys: tuple[str, ...],
    label_key: str,
    fallback_label: str,
) -> list[dict[str, Any]]:
    for source in sources:
        candidate = deep_find_first(source, candidate_keys)
        normalized = _normalize_named_entries(candidate, label_key=label_key, fallback_label=fallback_label)
        if normalized:
            return normalized
    return []


def _normalize_named_entries(candidate: Any, *, label_key: str, fallback_label: str) -> list[dict[str, Any]]:
    if candidate is None:
        return []

    if isinstance(candidate, Mapping):
        if _looks_like_single_entry(candidate):
            normalized = _normalize_single_entry(None, candidate, label_key=label_key, fallback_label=fallback_label)
            return [normalized] if normalized else []

        entries: list[dict[str, Any]] = []
        for key, value in candidate.items():
            normalized = _normalize_single_entry(str(key), value, label_key=label_key, fallback_label=fallback_label)
            if normalized is not None:
                entries.append(normalized)
        return entries

    if isinstance(candidate, list):
        entries = []
        for item in candidate:
            normalized = _normalize_single_entry(None, item, label_key=label_key, fallback_label=fallback_label)
            if normalized is not None:
                entries.append(normalized)
        return entries

    return []


def _looks_like_single_entry(candidate: Mapping[str, Any]) -> bool:
    normalized_keys = {normalize_key(key) for key in candidate.keys()}
    signal_keys = {
        "component",
        "name",
        "tensor",
        "count",
        "accesses",
        "bytes",
        "sizebytes",
        "totalbytes",
    }
    return bool(normalized_keys & signal_keys)


def _normalize_single_entry(
    provided_name: str | None,
    candidate: Any,
    *,
    label_key: str,
    fallback_label: str,
) -> dict[str, Any] | None:
    if candidate is None:
        return None

    if isinstance(candidate, Mapping):
        normalized = dict(candidate)
    else:
        scalar = coerce_float(candidate)
        if scalar is None:
            return None
        normalized = {"value": scalar}

    if not any(key in normalized for key in (label_key, "name", "component", "tensor")):
        normalized[label_key] = provided_name or fallback_label
    elif provided_name and label_key not in normalized:
        normalized[label_key] = provided_name

    total_bytes = _value_from_keys(
        normalized,
        ("bytes", "total_bytes", "traffic_bytes", "size_bytes", "payload_bytes", "remote_bytes"),
    )
    count = _value_from_keys(
        normalized,
        ("count", "accesses", "num_accesses", "messages", "action_count", "uses"),
    )
    bytes_per_action = _value_from_keys(
        normalized,
        ("bytes_per_action", "bytes_per_access", "bytes_per_message"),
    )

    if total_bytes is None and count is not None and bytes_per_action is not None:
        normalized["bytes"] = count * bytes_per_action
    elif total_bytes is not None and "bytes" not in normalized:
        normalized["bytes"] = total_bytes

    if count is not None and "count" not in normalized:
        normalized["count"] = count

    return normalized


def _value_from_keys(candidate: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    normalized_keys = {normalize_key(key): key for key in candidate.keys()}
    for key in keys:
        existing_key = normalized_keys.get(normalize_key(key))
        if existing_key is None:
            continue
        value = coerce_float(candidate.get(existing_key))
        if value is not None:
            return value
    return None
