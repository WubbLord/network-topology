from __future__ import annotations

import copy
import logging
import os
import tempfile
from collections.abc import Mapping
from typing import Any

from .accelforge_adapter import run_mapping
from .comms import compute_effective_comm_cost, estimate_comm_from_mapping
from .common import coerce_float
from .mapping import extract_mapping_stats
from .yaml_utils import (
    find_node_by_name,
    get_nested_value,
    load_yaml,
    patch_logical_comm_cost,
    resolve_patch_paths,
    save_yaml,
)

LOGGER = logging.getLogger(__name__)


def run_comm_fixed_point_loop(
    arch_yaml_path: str,
    workload_yaml_path: str,
    topology_config: dict[str, Any],
    max_iters: int = 10,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-9,
) -> dict[str, Any]:
    original_arch = load_yaml(arch_yaml_path)
    patch_config = topology_config.get("architecture_patch") or topology_config.get("patch") or {}
    cost_mode = str(topology_config.get("effective_cost_mode", "per_byte")).lower()
    damping = float(topology_config.get("damping", 1.0))
    current_cost = _initial_effective_cost(original_arch, patch_config, cost_mode, topology_config)

    iterations: list[dict[str, Any]] = []
    previous_signature: str | None = None
    converged = False
    final_mapping_stats: dict[str, Any] | None = None
    final_comm_estimate: dict[str, Any] | None = None
    final_architecture: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix="comm-fixed-point-") as tmpdir:
        for iteration in range(1, max_iters + 1):
            patched_arch = patch_logical_comm_cost(original_arch, current_cost, patch_config)
            patched_arch_path = os.path.join(tmpdir, f"arch_iter_{iteration}.yaml")
            save_yaml(patched_arch, patched_arch_path)

            mapping_result = run_mapping(patched_arch_path, workload_yaml_path)
            mapping_stats = extract_mapping_stats(mapping_result)
            comm_estimate = estimate_comm_from_mapping(mapping_stats, topology_config)
            effective_cost = compute_effective_comm_cost(comm_estimate)
            next_cost = _blend_effective_cost(current_cost, effective_cost, damping)
            deltas = _cost_deltas(current_cost, next_cost)

            mapping_signature = mapping_stats["mapping_signature"]
            mapping_stable = previous_signature is not None and mapping_signature == previous_signature
            cost_converged = _is_cost_converged(current_cost, next_cost, rel_tol=rel_tol, abs_tol=abs_tol)
            iteration_converged = mapping_stable and cost_converged

            iteration_record = {
                "iteration": iteration,
                "assumed_effective_cost": copy.deepcopy(current_cost),
                "mapping_signature": mapping_signature,
                "mapper_energy": mapping_stats.get("total_energy"),
                "mapper_latency": mapping_stats.get("total_latency"),
                "remote_bytes": comm_estimate["total_remote_bytes"],
                "byte_hops": comm_estimate["total_byte_hops"],
                "network_energy": comm_estimate["total_network_energy_j"],
                "network_latency": comm_estimate["total_network_latency_s"],
                "computed_effective_cost": effective_cost,
                "next_effective_cost": copy.deepcopy(next_cost),
                "convergence_deltas": deltas,
                "mapping_stable": mapping_stable,
                "cost_converged": cost_converged,
                "converged": iteration_converged,
                "notes": list(mapping_stats.get("notes", [])) + list(comm_estimate.get("notes", [])),
            }
            iterations.append(iteration_record)

            LOGGER.info(
                "iter=%d sig=%s mapper_energy=%s mapper_latency=%s remote_bytes=%.3f byte_hops=%.3f "
                "network_energy=%.6e network_latency=%.6e next_cost=%s deltas=%s converged=%s",
                iteration,
                mapping_signature,
                mapping_stats.get("total_energy"),
                mapping_stats.get("total_latency"),
                comm_estimate["total_remote_bytes"],
                comm_estimate["total_byte_hops"],
                comm_estimate["total_network_energy_j"],
                comm_estimate["total_network_latency_s"],
                next_cost,
                deltas,
                iteration_converged,
            )

            final_mapping_stats = mapping_stats
            final_comm_estimate = comm_estimate
            final_architecture = patched_arch
            current_cost = next_cost
            previous_signature = mapping_signature

            if iteration_converged:
                converged = True
                break

    return {
        "final_mapping": (final_mapping_stats or {}).get("mapping_structure"),
        "final_mapping_stats": final_mapping_stats,
        "iterations": iterations,
        "effective_cost": current_cost,
        "comm_estimate": final_comm_estimate,
        "final_patched_architecture": final_architecture,
        "converged": converged,
    }


def _initial_effective_cost(
    original_arch: Mapping[str, Any],
    patch_config: Mapping[str, Any],
    cost_mode: str,
    topology_config: Mapping[str, Any],
) -> dict[str, Any]:
    configured = topology_config.get("initial_effective_cost")
    if isinstance(configured, Mapping):
        return _canonicalize_effective_cost(configured, cost_mode)

    target_name = patch_config.get("target_name", "logical_interconnect")
    target = find_node_by_name(original_arch, str(target_name))
    if target is not None:
        style = str(patch_config.get("style", "toll_component")).lower()
        energy_path, latency_path = resolve_patch_paths(cost_mode, style, patch_config)
        current_energy = coerce_float(get_nested_value(target, energy_path))
        current_latency = coerce_float(get_nested_value(target, latency_path))
        if current_energy is not None or current_latency is not None:
            if cost_mode == "per_access":
                return {
                    "mode": "per_access",
                    "energy_per_access": current_energy or 0.0,
                    "latency_per_access": current_latency or 0.0,
                }
            return {
                "mode": "per_byte",
                "energy_per_byte": current_energy or 0.0,
                "latency_per_byte": current_latency or 0.0,
            }

    if cost_mode == "per_access":
        return {"mode": "per_access", "energy_per_access": 0.0, "latency_per_access": 0.0}
    return {"mode": "per_byte", "energy_per_byte": 0.0, "latency_per_byte": 0.0}


def _canonicalize_effective_cost(cost: Mapping[str, Any], default_mode: str) -> dict[str, Any]:
    mode = str(cost.get("mode", cost.get("effective_cost_mode", default_mode))).lower()
    if mode == "per_access":
        return {
            "mode": "per_access",
            "energy_per_access": float(cost.get("energy_per_access", 0.0)),
            "latency_per_access": float(cost.get("latency_per_access", 0.0)),
        }
    return {
        "mode": "per_byte",
        "energy_per_byte": float(cost.get("energy_per_byte", 0.0)),
        "latency_per_byte": float(cost.get("latency_per_byte", 0.0)),
    }


def _blend_effective_cost(
    current_cost: Mapping[str, Any],
    new_cost: Mapping[str, Any],
    damping: float,
) -> dict[str, Any]:
    damping = min(max(damping, 0.0), 1.0)
    current = _canonicalize_effective_cost(current_cost, str(current_cost.get("mode", "per_byte")))
    target = _canonicalize_effective_cost(new_cost, current["mode"])

    if current["mode"] == "per_access":
        return {
            "mode": "per_access",
            "energy_per_access": ((1.0 - damping) * current["energy_per_access"]) + (damping * target["energy_per_access"]),
            "latency_per_access": ((1.0 - damping) * current["latency_per_access"]) + (damping * target["latency_per_access"]),
        }
    return {
        "mode": "per_byte",
        "energy_per_byte": ((1.0 - damping) * current["energy_per_byte"]) + (damping * target["energy_per_byte"]),
        "latency_per_byte": ((1.0 - damping) * current["latency_per_byte"]) + (damping * target["latency_per_byte"]),
    }


def _cost_deltas(current_cost: Mapping[str, Any], next_cost: Mapping[str, Any]) -> dict[str, float]:
    current = _canonicalize_effective_cost(current_cost, str(current_cost.get("mode", "per_byte")))
    next_value = _canonicalize_effective_cost(next_cost, current["mode"])

    deltas: dict[str, float] = {}
    if current["mode"] == "per_access":
        for key in ("energy_per_access", "latency_per_access"):
            deltas[f"{key}_abs"] = abs(next_value[key] - current[key])
            baseline = max(abs(current[key]), 1e-12)
            deltas[f"{key}_rel"] = abs(next_value[key] - current[key]) / baseline
    else:
        for key in ("energy_per_byte", "latency_per_byte"):
            deltas[f"{key}_abs"] = abs(next_value[key] - current[key])
            baseline = max(abs(current[key]), 1e-12)
            deltas[f"{key}_rel"] = abs(next_value[key] - current[key]) / baseline
    return deltas


def _is_cost_converged(
    current_cost: Mapping[str, Any],
    next_cost: Mapping[str, Any],
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    current = _canonicalize_effective_cost(current_cost, str(current_cost.get("mode", "per_byte")))
    next_value = _canonicalize_effective_cost(next_cost, current["mode"])

    if current["mode"] == "per_access":
        keys = ("energy_per_access", "latency_per_access")
    else:
        keys = ("energy_per_byte", "latency_per_byte")

    for key in keys:
        delta = abs(next_value[key] - current[key])
        baseline = max(abs(current[key]), abs(next_value[key]), 1.0)
        if delta > abs_tol and (delta / baseline) > rel_tol:
            return False
    return True
