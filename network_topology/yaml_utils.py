from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML document at {path} to load into a dictionary.")
    return data


def save_yaml(data: Mapping[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def find_node_by_name(tree: Any, name: str) -> dict[str, Any] | None:
    if isinstance(tree, dict):
        if tree.get("name") == name:
            return tree
        for value in tree.values():
            match = find_node_by_name(value, name)
            if match is not None:
                return match
        return None
    if isinstance(tree, list):
        for item in tree:
            match = find_node_by_name(item, name)
            if match is not None:
                return match
    return None


def get_nested_value(node: Mapping[str, Any], path: str) -> Any:
    cursor: Any = node
    for fragment in path.split("."):
        if not isinstance(cursor, Mapping) or fragment not in cursor:
            return None
        cursor = cursor[fragment]
    return cursor


def set_nested_value(node: dict[str, Any], path: str, value: Any) -> None:
    cursor = node
    fragments = path.split(".")
    for fragment in fragments[:-1]:
        next_value = cursor.get(fragment)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[fragment] = next_value
        cursor = next_value
    cursor[fragments[-1]] = value


def resolve_patch_paths(mode: str, style: str, patch_config: Mapping[str, Any]) -> tuple[str, str]:
    energy_path = patch_config.get("energy_path")
    latency_path = patch_config.get("latency_path")
    if isinstance(energy_path, str) and isinstance(latency_path, str):
        return energy_path, latency_path

    if mode == "per_access" and style == "toll_component":
        return "attributes.energy_per_traversal", "attributes.latency_per_traversal"
    if mode == "per_access":
        return "attributes.energy_per_access", "attributes.latency_per_access"
    return "attributes.energy_per_byte", "attributes.latency_per_byte"


def patch_logical_comm_cost(
    arch_spec: Mapping[str, Any],
    effective_cost: Mapping[str, Any],
    patch_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    patch_config = patch_config or {}
    mode = str(effective_cost.get("mode", "per_byte")).lower()
    style = str(patch_config.get("style", "toll_component")).lower()
    target_name = str(patch_config.get("target_name", "logical_interconnect"))

    patched = copy.deepcopy(dict(arch_spec))
    target = find_node_by_name(patched, target_name)
    if target is None:
        raise KeyError(f"Unable to find architecture node named '{target_name}'.")

    energy_path, latency_path = resolve_patch_paths(mode, style, patch_config)
    if mode == "per_access":
        energy_value = float(effective_cost.get("energy_per_access", 0.0))
        latency_value = float(effective_cost.get("latency_per_access", 0.0))
    else:
        energy_value = float(effective_cost.get("energy_per_byte", 0.0))
        latency_value = float(effective_cost.get("latency_per_byte", 0.0))

    set_nested_value(target, energy_path, energy_value)
    set_nested_value(target, latency_path, latency_value)
    return patched
