#!/usr/bin/env python3
"""
Topology sweep: evaluate workloads on multiple topologies with congestion modeling.

Maps on 2 chips (fast), scales data movement to 64 chips, evaluates on 5 topologies.

Usage:
    ACCELFORGE_ROOT=/path/to/accelforge .venv/bin/python sweep_gpt3.py [--damping]
"""

import argparse
import contextlib
import copy
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.disable(logging.WARNING)

from network_topology import compute_network_cost, make_tpu_v4_topology
from network_topology.cost_model import NetworkTransfer, CollectiveType
from network_topology.topology import Torus3D, Mesh3D, Ring
from network_topology.tpu_v4 import ICI_LINK_BW_UNIDIR, ICI_ENERGY_PER_BIT_PER_HOP, ICI_PER_HOP_LATENCY

SCRIPT_DIR = Path(__file__).resolve().parent
ACCELFORGE = Path(os.environ.get("ACCELFORGE_ROOT", SCRIPT_DIR.parent / "accelforge")).expanduser()
ARCH = SCRIPT_DIR / "accelforge_configs" / "tpu_v4_distributed_1d.yaml"
MAP_CHIPS = 2
EVAL_CHIPS = 64
SCALE = EVAL_CHIPS / MAP_CHIPS
DEFAULT_NETWORK_READ_ENERGY = 7.03e-12
DEFAULT_NETWORK_WRITE_ENERGY = 7.03e-12
DEFAULT_NETWORK_READ_LATENCY = 1.0 / (8.0 * 614e9)
DEFAULT_NETWORK_WRITE_LATENCY = 1.0 / (8.0 * 614e9)
MAX_PROXY_ITERS = int(os.environ.get("MAX_PROXY_ITERS", 6))
PROXY_DAMPING = float(os.environ.get("PROXY_DAMPING", 0.5))
PROXY_REL_TOL = float(os.environ.get("PROXY_REL_TOL", 0.05))

hw = dict(link_bandwidth=ICI_LINK_BW_UNIDIR, energy_per_bit_per_hop=ICI_ENERGY_PER_BIT_PER_HOP,
          per_hop_latency=ICI_PER_HOP_LATENCY)

TOPOLOGIES = {
    # "Torus 4x4x4": Torus3D(dims=(4, 4, 4), **hw),
    # "Mesh 4x4x4": Mesh3D(dims=(4, 4, 4), **hw),
    # "Torus 8x2x4": Torus3D(dims=(8, 2, 4), **hw),
    # "Torus 16x2x2": Torus3D(dims=(16, 2, 2), **hw),
    "Ring 64": Ring(num_chips=64, **hw),
}


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _serialize_transfer(transfer: NetworkTransfer):
    return {
        "tensor_name": transfer.tensor_name,
        "data_bytes": float(transfer.data_bytes),
        "collective_type": transfer.collective_type.name,
        "src_chip": transfer.src_chip,
        "dst_chips": transfer.dst_chips,
        "participating_chips": transfer.participating_chips,
    }


def save_results(payload: dict) -> Path:
    out_dir = make_run_dir()
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
    return out_path


def make_run_dir() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir = SCRIPT_DIR / "logs" / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def save_results_in_dir(out_dir: Path, payload: dict) -> Path:
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
    return out_path


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "unnamed"


def _mapping_value_to_data(value):
    if value is None:
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _mapping_value_to_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mapping_value_to_data(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_mapping_value_to_data(v) for v in sorted(value, key=str)]

    if value.__class__.__name__ == "InvertibleSet":
        return {
            "type": value.__class__.__name__,
            "instance": _mapping_value_to_data(value.instance),
            "full_space": _mapping_value_to_data(value.full_space),
            "space_type": getattr(value.space_type, "__name__", str(value.space_type)),
            "element_to_child_space": _mapping_value_to_data(value.element_to_child_space),
            "_bits_per_value": getattr(value, "_bits_per_value", None),
        }

    model_fields = getattr(value.__class__, "model_fields", None)
    if model_fields:
        data = {"type": value.__class__.__name__}
        for field_name in model_fields:
            field_value = getattr(value, field_name)
            if field_value is None:
                continue
            data[field_name] = _mapping_value_to_data(field_value)
        return data

    if isinstance(value, type):
        return value.__name__

    return str(value)


def _mapping_to_yaml(mapping) -> str:
    from accelforge.util._yaml import to_yaml_string

    return to_yaml_string({"mapping": _mapping_value_to_data(mapping)})


def parse_args():
    parser = argparse.ArgumentParser(description="Run the topology sweep.")
    parser.add_argument(
        "--damping",
        action="store_true",
        help=(
            "Enable damped proxy updates. When omitted, each iteration applies the "
            "new proxy estimate directly."
        ),
    )
    return parser.parse_args()


def _tensor_rank_vars(einsum, tensor_name):
    for tensor_access in einsum.tensor_accesses:
        if tensor_access.name == tensor_name:
            return set(dict(tensor_access.projection).values())
    raise KeyError(f"Tensor {tensor_name} not found in Einsum {einsum.name}")


def _chip_sharded_rank_vars(mapping, flattened_arch, tensor_name, tensor_rank_vars):
    from accelforge.frontend.mapping.mapping import (
        MappingNodeWithChildren,
        Reservation,
        Spatial,
        TensorHolder,
    )

    def _normalize_single_tensor_nodes(node):
        if isinstance(node, MappingNodeWithChildren):
            normalized = []
            for child in node.nodes:
                if isinstance(child, TensorHolder) and len(child.tensors) > 1:
                    for tensor in child.tensors:
                        new_child = copy.copy(child)
                        new_child.tensors = [tensor]
                        normalized.append(new_child)
                elif isinstance(child, Reservation) and len(child.purposes) > 1:
                    for purpose in child.purposes:
                        new_child = copy.copy(child)
                        new_child.purposes = [purpose]
                        normalized.append(new_child)
                else:
                    _normalize_single_tensor_nodes(child)
                    normalized.append(child)
            node.nodes = normalized

    mapping_copy = copy.deepcopy(mapping)
    _normalize_single_tensor_nodes(mapping_copy)
    tensor_mapping = mapping_copy._get_single_tensor_mapping(
        tensor_name, flattened_arch, tensor_rank_vars
    )
    return {
        s.rank_variable
        for s in tensor_mapping.get_nodes_of_type(Spatial)
        if s.component == "ChipArray"
    }


def _effective_transfer_bytes(raw_bytes, tensor_size_bytes):
    return float(max(raw_bytes, tensor_size_bytes * SCALE))


def _comparison_transfer_bytes(raw_bytes, tensor_size_bytes):
    if raw_bytes > 0:
        return float(raw_bytes)
    return float(tensor_size_bytes * SCALE)


def _initial_network_proxies():
    return {
        "NETWORK_READ_ENERGY": DEFAULT_NETWORK_READ_ENERGY,
        "NETWORK_WRITE_ENERGY": DEFAULT_NETWORK_WRITE_ENERGY,
        "NETWORK_READ_LATENCY": DEFAULT_NETWORK_READ_LATENCY,
        "NETWORK_WRITE_LATENCY": DEFAULT_NETWORK_WRITE_LATENCY,
    }


def _proxy_keys(action):
    if action == "read":
        return ("NETWORK_READ_ENERGY", "NETWORK_READ_LATENCY")
    if action == "write":
        return ("NETWORK_WRITE_ENERGY", "NETWORK_WRITE_LATENCY")
    raise ValueError(f"Unsupported proxy action: {action}")


def _max_relative_change(old_values, new_values):
    max_change = 0.0
    for key, old_value in old_values.items():
        new_value = new_values[key]
        baseline = max(abs(old_value), 1e-30)
        max_change = max(max_change, abs(new_value - old_value) / baseline)
    return float(max_change)


def _damp_network_proxies(old_values, new_values, use_damping=False):
    if not use_damping:
        return dict(new_values)
    return {
        key: (1.0 - PROXY_DAMPING) * old_values[key] + PROXY_DAMPING * new_values[key]
        for key in old_values
    }


def _collective_mix(transfers):
    total_bytes = float(sum(t.data_bytes for t in transfers))
    allgather_bytes = float(
        sum(t.data_bytes for t in transfers if t.collective_type == CollectiveType.ALLGATHER)
    )
    allgather_pct = 100.0 * allgather_bytes / total_bytes if total_bytes > 0 else 0.0
    return total_bytes, allgather_pct


def _annotate_collective_decisions(decisions, network_result):
    per_transfer = {
        transfer["tensor"]: transfer for transfer in network_result.per_transfer
    }
    max_individual_latency = max(
        (float(transfer["latency"]) for transfer in network_result.per_transfer),
        default=0.0,
    )
    congestion_scale = (
        float(network_result.total_latency) / max_individual_latency
        if max_individual_latency > 0
        else 1.0
    )

    annotated = []
    for decision in decisions:
        annotated_decision = dict(decision)
        transfer_name = (
            f"{decision['einsum']}:{decision['tensor_name']}"
            if decision["collective_type"] is not None and decision["data_bytes"] > 0
            else None
        )
        transfer = per_transfer.get(transfer_name)
        if transfer is None:
            annotated_decision["estimated_network_energy"] = 0.0
            annotated_decision["estimated_network_latency"] = 0.0
            annotated_decision["estimated_network_energy_per_byte"] = 0.0
            annotated_decision["estimated_network_latency_per_byte"] = 0.0
        else:
            estimated_latency = float(transfer["latency"]) * congestion_scale
            annotated_decision["estimated_network_energy"] = float(transfer["energy"])
            annotated_decision["estimated_network_latency"] = estimated_latency
            annotated_decision["estimated_network_energy_per_byte"] = (
                float(transfer["energy"]) / decision["data_bytes"]
                if decision["data_bytes"] > 0
                else 0.0
            )
            annotated_decision["estimated_network_latency_per_byte"] = (
                estimated_latency / decision["data_bytes"]
                if decision["data_bytes"] > 0
                else 0.0
            )
        annotated.append(annotated_decision)
    return annotated


def _updated_network_proxies(old_proxies, decisions, network_result):
    updated = dict(old_proxies)
    buckets = {
        "read": {"energy": 0.0, "latency": 0.0, "bytes": 0.0},
        "write": {"energy": 0.0, "latency": 0.0, "bytes": 0.0},
    }

    for decision in decisions:
        proxy_action = decision["proxy_action"]
        if proxy_action is None or decision["data_bytes"] <= 0:
            continue
        bucket = buckets[proxy_action]
        bucket["energy"] += float(decision["estimated_network_energy"])
        bucket["latency"] += float(decision["estimated_network_latency"])
        bucket["bytes"] += float(decision["data_bytes"])

    for action, bucket in buckets.items():
        energy_key, latency_key = _proxy_keys(action)
        if bucket["bytes"] <= 0:
            continue
        updated[energy_key] = bucket["energy"] / bucket["bytes"]
        updated[latency_key] = bucket["latency"] / bucket["bytes"]

    return updated


def _make_collective_decision(
    einsum_name,
    tensor_name,
    collective_type,
    proxy_action,
    data_bytes,
    reason,
    tensor_ranks,
    chip_sharded_ranks,
):
    return {
        "einsum": einsum_name,
        "tensor_name": tensor_name,
        "collective_type": None if collective_type is None else collective_type.name,
        "proxy_action": proxy_action,
        "data_bytes": float(data_bytes),
        "reason": reason,
        "tensor_ranks": sorted(tensor_ranks),
        "chip_sharded_ranks": sorted(chip_sharded_ranks),
    }


def _infer_matmul_collectives(einsum_name, tensor_infos, read_bytes, write_bytes, tensor_size_bytes):
    input_infos = [info for info in tensor_infos.values() if not info["is_output"]]
    output_infos = [info for info in tensor_infos.values() if info["is_output"]]
    if len(input_infos) != 2 or len(output_infos) != 1:
        raise ValueError(f"{einsum_name} is not a 2-input matmul")

    left, right = input_infos
    output = output_infos[0]

    left_ranks = left["tensor_ranks"]
    right_ranks = right["tensor_ranks"]
    output_ranks = output["tensor_ranks"]
    contracting_ranks = (left_ranks & right_ranks) - output_ranks
    left_output_ranks = (left_ranks & output_ranks) - right_ranks
    right_output_ranks = (right_ranks & output_ranks) - left_ranks

    left_contracting = left["chip_sharded_ranks"] & contracting_ranks
    right_contracting = right["chip_sharded_ranks"] & contracting_ranks
    left_noncontracting = left["chip_sharded_ranks"] & left_output_ranks
    right_noncontracting = right["chip_sharded_ranks"] & right_output_ranks

    decisions = []

    if left_contracting and right_contracting:
        collective_type = (
            CollectiveType.REDUCE_SCATTER
            if output["chip_sharded_ranks"] & output_ranks
            else CollectiveType.ALLREDUCE
        )
        data_bytes = _effective_transfer_bytes(
            write_bytes[output["name"]],
            tensor_size_bytes[output["name"]],
        )
        decisions.append(
            _make_collective_decision(
                einsum_name,
                output["name"],
                collective_type,
                "write",
                data_bytes,
                "Both inputs shard the same contracting dimension, so local partial "
                "products must be reduced across chips.",
                output_ranks,
                output["chip_sharded_ranks"],
            )
        )
        return decisions

    if bool(left_contracting) ^ bool(right_contracting):
        gathered = left if left_contracting else right
        data_bytes = _effective_transfer_bytes(
            read_bytes[gathered["name"]],
            tensor_size_bytes[gathered["name"]],
        )
        decisions.append(
            _make_collective_decision(
                einsum_name,
                gathered["name"],
                CollectiveType.ALLGATHER,
                "read",
                data_bytes,
                "Exactly one input shards a contracting dimension, so that input "
                "must be all-gathered before the local matmul.",
                gathered["tensor_ranks"],
                gathered["chip_sharded_ranks"],
            )
        )
        return decisions

    if left_noncontracting and right_noncontracting:
        gathered = min(
            (left, right),
            key=lambda info: _comparison_transfer_bytes(
                read_bytes[info["name"]],
                tensor_size_bytes[info["name"]],
            ),
        )
        data_bytes = _effective_transfer_bytes(
            read_bytes[gathered["name"]],
            tensor_size_bytes[gathered["name"]],
        )
        decisions.append(
            _make_collective_decision(
                einsum_name,
                gathered["name"],
                CollectiveType.ALLGATHER,
                "read",
                data_bytes,
                "Both inputs shard different output dimensions on the same chip axis, "
                "so one input is all-gathered first to remove the layout conflict.",
                gathered["tensor_ranks"],
                gathered["chip_sharded_ranks"],
            )
        )
        return decisions

    decisions.append(
        _make_collective_decision(
            einsum_name,
            output["name"],
            None,
            None,
            0.0,
            "No input shards a contracting dimension and there is no output-layout "
            "conflict, so the matmul is local on each chip.",
            output_ranks,
            output["chip_sharded_ranks"],
        )
    )
    return decisions


def load_accelforge(accelforge_root: Path):
    accelforge_root = accelforge_root.expanduser().resolve()
    if not accelforge_root.exists():
        raise SystemExit(
            f"AccelForge repo not found at {accelforge_root}. "
            "Set ACCELFORGE_ROOT to a local AccelForge checkout."
        )

    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(accelforge_root))

    try:
        import accelforge as af
        from accelforge.mapper.FFM import map_workload_to_arch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Unable to import accelforge. Install it in the active environment or "
            "point ACCELFORGE_ROOT at a local checkout."
        ) from exc

    return af, map_workload_to_arch


def make_workloads(workloads_dir: Path):
    return [
        # ("Tiny 256x256", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 256, "KN": 256}),
        ("Small 1Kx1K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 1024, "KN": 1024}),
        # ("Medium 4Kx4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 4096, "KN": 4096}),
        # ("Wide 4Kx16K (FFN-like)", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 4096, "KN": 16384}),
        # ("Tall 16Kx4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 16384, "KN": 4096}),
        # ("2-layer chain 4Kx4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 2, "M": 4096, "KN": 4096}),
        # ("3-layer chain 4Kx4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 3, "M": 4096, "KN": 4096}),
        ("Attn-like 128x128", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 128, "KN": 128}),
        # ("Attn-like 512x512", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 512, "KN": 512}),
        # ("Attn-like 2Kx2K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 2048, "KN": 2048}),
        # ("Decode 1x4096", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 1, "KN": 4096}),
        # ("Decode 1x16384", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 1, "KN": 16384}),
        # ("Batch 64tok x 4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 64, "KN": 4096}),
        # ("Batch 1024tok x 4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 1024, "KN": 4096}),
    ]


def map_and_extract(path, params, network_proxies, af, map_workload_to_arch):
    from accelforge.frontend.mapping.mapping import Compute

    spec = af.Spec.from_yaml(
        str(ARCH),
        str(path),
        jinja_parse_data={"NUM_CHIPS": MAP_CHIPS, **network_proxies, **params},
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        mappings = map_workload_to_arch(spec)
    energy_bk = mappings.energy(per_component=True)
    compute_e = float(sum(v for k, v in energy_bk.items() if k != "NetworkMemory")) * SCALE
    compute_l = float(mappings.latency())
    actions = mappings.actions(per_einsum=True, per_component=True, per_tensor=True)
    energy_details = mappings.energy(
        per_einsum=True, per_component=True, per_tensor=True, per_action=True
    )
    latency_details = mappings.latency(per_einsum=True, per_component=True)
    mapping = mappings.mapping(0)
    mapping_yaml = _mapping_to_yaml(mapping)
    compute_components = {
        compute.einsum: compute.component for compute in mapping.get_nodes_of_type(Compute)
    }
    tensor_size_bytes = {
        tensor: float(bits) / 8.0 for tensor, bits in mappings.per_tensor_size().items()
    }

    network_reads = defaultdict(float)
    network_writes = defaultdict(float)
    for (einsum_name, comp, tensor, action), count in actions.items():
        if comp != "NetworkMemory" or count <= 0:
            continue
        if action == "read":
            network_reads[(einsum_name, tensor)] += float(count) * SCALE
        elif action == "write":
            network_writes[(einsum_name, tensor)] += float(count) * SCALE

    transfers = []
    collective_decisions = []

    for einsum_name in mappings.einsum_names:
        einsum = spec.workload.einsums[einsum_name]
        flat_arch = mappings.flattened_arches[(einsum_name, compute_components[einsum_name])]
        tensor_infos = {}
        for tensor_access in einsum.tensor_accesses:
            tensor_ranks = _tensor_rank_vars(einsum, tensor_access.name)
            tensor_infos[tensor_access.name] = {
                "name": tensor_access.name,
                "tensor_ranks": tensor_ranks,
                "chip_sharded_ranks": _chip_sharded_rank_vars(
                    mapping, flat_arch, tensor_access.name, tensor_ranks
                ),
                "is_output": tensor_access.output,
            }

        decisions = _infer_matmul_collectives(
            einsum_name,
            tensor_infos,
            {name: network_reads[(einsum_name, name)] for name in tensor_infos},
            {name: network_writes[(einsum_name, name)] for name in tensor_infos},
            tensor_size_bytes,
        )
        collective_decisions.extend(decisions)

        for decision in decisions:
            if decision["collective_type"] is None or decision["data_bytes"] <= 0:
                continue
            transfers.append(
                NetworkTransfer(
                    tensor_name=f"{decision['einsum']}:{decision['tensor_name']}",
                    data_bytes=decision["data_bytes"],
                    collective_type=CollectiveType[decision["collective_type"]],
                )
            )

    return {
        "compute_energy": compute_e,
        "compute_latency": compute_l,
        "transfers": transfers,
        "collective_decisions": collective_decisions,
        "network_proxies": dict(network_proxies),
        "mapping_yaml": mapping_yaml,
        "mapping_energy": _json_ready(energy_details),
        "mapping_latency": _json_ready(latency_details),
        "mapping_actions": _json_ready(actions),
    }


def run_feedback_loop(
    path,
    params,
    topology,
    af,
    map_workload_to_arch,
    mapping_dir: Path | None = None,
    use_damping=False,
):
    network_proxies = _initial_network_proxies()
    iterations = []
    final_mapping_result = None
    final_network_result = None
    converged = False

    if mapping_dir is not None:
        mapping_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, MAX_PROXY_ITERS + 1):
        mapping_result = map_and_extract(path, params, network_proxies, af, map_workload_to_arch)
        mapping_yaml_relpath = None
        if mapping_dir is not None:
            mapping_yaml_path = mapping_dir / f"iter_{iteration:03d}.yaml"
            mapping_yaml_path.write_text(mapping_result["mapping_yaml"], encoding="utf-8")
            mapping_yaml_relpath = str(
                Path("mappings") / mapping_yaml_path.relative_to(mapping_dir.parents[1])
            )
        network_result = compute_network_cost(topology, mapping_result["transfers"])
        collective_decisions = _annotate_collective_decisions(
            mapping_result["collective_decisions"], network_result
        )
        proposed_proxies = _updated_network_proxies(
            network_proxies, collective_decisions, network_result
        )
        updated_proxies = _damp_network_proxies(
            network_proxies, proposed_proxies, use_damping=use_damping
        )
        raw_relative_change = _max_relative_change(network_proxies, proposed_proxies)
        applied_relative_change = _max_relative_change(network_proxies, updated_proxies)
        total_bytes, allgather_pct = _collective_mix(mapping_result["transfers"])

        iterations.append(
            {
                "iteration": iteration,
                "input_network_proxies": dict(network_proxies),
                "proposed_network_proxies": dict(proposed_proxies),
                "updated_network_proxies": dict(updated_proxies),
                "raw_relative_change": raw_relative_change,
                "applied_relative_change": applied_relative_change,
                "mapping": {
                    "compute_energy": float(mapping_result["compute_energy"]),
                    "compute_latency": float(mapping_result["compute_latency"]),
                    "mapping_yaml_path": mapping_yaml_relpath,
                    "mapping_energy": mapping_result["mapping_energy"],
                    "mapping_latency": mapping_result["mapping_latency"],
                    "mapping_actions": mapping_result["mapping_actions"],
                },
                "network": {
                    "total_energy": float(network_result.total_energy),
                    "total_latency": float(network_result.total_latency),
                    "energy_per_network_access": float(network_result.energy_per_network_access),
                    "latency_per_network_access": float(network_result.latency_per_network_access),
                    "total_network_bytes": float(network_result.total_network_bytes),
                    "per_transfer": _json_ready(network_result.per_transfer),
                },
                "transfer_count": len(mapping_result["transfers"]),
                "total_bytes": total_bytes,
                "allgather_pct": allgather_pct,
                "collective_decisions": collective_decisions,
            }
        )

        final_mapping_result = dict(mapping_result)
        final_mapping_result["collective_decisions"] = collective_decisions
        final_mapping_result["network_proxies"] = dict(mapping_result["network_proxies"])
        final_mapping_result["mapping_yaml_path"] = mapping_yaml_relpath
        final_network_result = network_result
        network_proxies = updated_proxies

        if applied_relative_change < PROXY_REL_TOL:
            converged = True
            break

    total_bytes, allgather_pct = _collective_mix(final_mapping_result["transfers"])
    return {
        "converged": converged,
        "iterations": iterations,
        "final_network_proxies": dict(network_proxies),
        "final_mapping_result": final_mapping_result,
        "final_network_result": final_network_result,
        "final_transfers": final_mapping_result["transfers"],
        "final_collective_decisions": final_mapping_result["collective_decisions"],
        "total_bytes": total_bytes,
        "allgather_pct": allgather_pct,
    }


def main():
    args = parse_args()
    af, map_workload_to_arch = load_accelforge(ACCELFORGE)
    workloads_dir = ACCELFORGE.resolve() / "examples" / "workloads"
    if not workloads_dir.exists():
        raise SystemExit(
            f"Expected workload specs under {workloads_dir}, but the directory does not exist."
        )
    workloads = make_workloads(workloads_dir)
    run_dir = make_run_dir()
    mappings_root = run_dir / "mappings"

    print("=" * 120)
    print(
        f"Topology Sweep: {len(workloads)} workloads x {len(TOPOLOGIES)} topologies "
        f"({EVAL_CHIPS} chips, congestion-aware, iterative feedback, "
        f"damping={'on' if args.damping else 'off'})"
    )
    print("=" * 120)

    print(f"\n{'Topology':<18s} {'Diam':>5s} {'AvgHop':>7s} {'Degree':>7s} {'BisectBW':>10s}")
    print("-" * 50)
    topology_summaries = {}
    for tn in TOPOLOGIES:
        s = TOPOLOGIES[tn].summary()
        topology_summaries[tn] = _json_ready(s)
        print(f"{tn:<18s} {s['diameter']:>5d} {s['avg_hops']:>7.2f} {s['min_degree']:>3d}-{s['max_degree']:<3d} "
              f"{s['bisection_bandwidth_TB_s']:>8.2f} TB/s")

    results = []
    for desc, path, params in workloads:
        print(f"\nWorkload: {desc}")
        lats = {}
        bytes_by_topology = {}
        allgather_pct_by_topology = {}
        compute_e_by_topology = {}
        compute_l_by_topology = {}
        mapping_wall_time_s_by_topology = {}
        topology_results = {}
        for topology_name, topo in TOPOLOGIES.items():
            t0 = time.time()
            print(f"  {topology_name:<18s} mapping+feedback...", end=" ", flush=True)
            try:
                loop_result = run_feedback_loop(
                    path,
                    params,
                    topo,
                    af,
                    map_workload_to_arch,
                    mappings_root / _safe_path_part(desc) / _safe_path_part(topology_name),
                    use_damping=args.damping,
                )
            except Exception as exc:
                print(f"FAILED: {exc}")
                continue

            elapsed = time.time() - t0
            iteration_count = len(loop_result["iterations"])
            convergence_note = "converged" if loop_result["converged"] else "max-iters"
            print(f"done ({elapsed:.0f}s, {iteration_count} iters, {convergence_note})")

            final_mapping = loop_result["final_mapping_result"]
            final_network = loop_result["final_network_result"]
            lats[topology_name] = float(final_network.total_latency)
            bytes_by_topology[topology_name] = float(loop_result["total_bytes"])
            allgather_pct_by_topology[topology_name] = float(loop_result["allgather_pct"])
            compute_e_by_topology[topology_name] = float(final_mapping["compute_energy"])
            compute_l_by_topology[topology_name] = float(final_mapping["compute_latency"])
            mapping_wall_time_s_by_topology[topology_name] = float(elapsed)

            topology_results[topology_name] = {
                "topology_summary": topology_summaries[topology_name],
                "mapping_wall_time_s": float(elapsed),
                "feedback_converged": loop_result["converged"],
                "feedback_iterations": loop_result["iterations"],
                "final_network_proxies": dict(loop_result["final_network_proxies"]),
                "compute_energy": float(final_mapping["compute_energy"]),
                "compute_latency": float(final_mapping["compute_latency"]),
                "final_mapping_yaml_path": final_mapping["mapping_yaml_path"],
                "bytes": float(loop_result["total_bytes"]),
                "allgather_pct": float(loop_result["allgather_pct"]),
                "transfers": [_serialize_transfer(t) for t in loop_result["final_transfers"]],
                "collective_decisions": loop_result["final_collective_decisions"],
                "mapping_energy": final_mapping["mapping_energy"],
                "mapping_latency": final_mapping["mapping_latency"],
                "mapping_actions": final_mapping["mapping_actions"],
                "total_energy": float(final_network.total_energy),
                "total_latency": float(final_network.total_latency),
                "energy_per_network_access": float(final_network.energy_per_network_access),
                "latency_per_network_access": float(final_network.latency_per_network_access),
                "total_network_bytes": float(final_network.total_network_bytes),
                "per_transfer": _json_ready(final_network.per_transfer),
            }

        if not topology_results:
            continue

        results.append(
            {
                "desc": desc,
                "workload_path": str(path),
                "params": params,
                "compute_e": sum(compute_e_by_topology.values()) / len(compute_e_by_topology),
                "compute_l": sum(compute_l_by_topology.values()) / len(compute_l_by_topology),
                "compute_e_by_topology": compute_e_by_topology,
                "compute_l_by_topology": compute_l_by_topology,
                "mapping_wall_time_s": (
                    sum(mapping_wall_time_s_by_topology.values())
                    / len(mapping_wall_time_s_by_topology)
                ),
                "mapping_wall_time_s_by_topology": mapping_wall_time_s_by_topology,
                "bytes": sum(bytes_by_topology.values()) / len(bytes_by_topology),
                "bytes_by_topology": bytes_by_topology,
                "allgather_pct": (
                    sum(allgather_pct_by_topology.values()) / len(allgather_pct_by_topology)
                ),
                "allgather_pct_by_topology": allgather_pct_by_topology,
                "lats": lats,
                "topologies": topology_results,
            }
        )

    print(f"\n\n{'=' * 120}")
    print("RESULTS (latency in ms, with link congestion)")
    print(f"{'=' * 120}")
    print(f"{'Workload':>28s} {'Bytes':>10s} {'AG%':>7s}", end="")
    for tn in TOPOLOGIES:
        print(f" {tn:>14s}", end="")
    print(f" {'Best':>10s} {'Worst':>10s} {'Speedup':>8s}")
    print("-" * 130)

    for r in results:
        available_topologies = [name for name in TOPOLOGIES if name in r["lats"]]
        if not available_topologies:
            continue
        best = min(available_topologies, key=r["lats"].get)
        worst = max(available_topologies, key=r["lats"].get)
        speedup = r["lats"][worst] / r["lats"][best] if r["lats"][best] > 0 else 0
        print(f"{r['desc']:>28s} {r['bytes']:>10.2e} {r['allgather_pct']:>6.0f}%", end="")
        for topology_name in TOPOLOGIES:
            latency = r["lats"].get(topology_name)
            if latency is None:
                print(f" {'n/a':>14s}", end="")
            else:
                print(f" {latency * 1e3:>12.1f}ms", end="")
        print(f" {best.split()[0]:>10s} {worst.split()[0]:>10s} {speedup:>7.2f}x")

    print(f"\n{'=' * 120}")
    print("INSIGHTS")
    print(f"{'=' * 120}")
    torus_wins = sum(
        1 for r in results if r["lats"] and min(r["lats"], key=r["lats"].get) == "Torus 4x4x4"
    )
    print(f"Torus 4x4x4 wins: {torus_wins}/{len(results)} workloads")
    insights = {"torus_4x4x4_wins": torus_wins, "num_workloads": len(results)}

    allgather_heavy = [
        r
        for r in results
        if r["allgather_pct"] > 80
        and "Mesh 4x4x4" in r["lats"]
        and "Torus 4x4x4" in r["lats"]
    ]
    if allgather_heavy:
        avg_speedup = (
            sum(
                r["lats"]["Mesh 4x4x4"] / r["lats"]["Torus 4x4x4"]
                for r in allgather_heavy
            )
            / len(allgather_heavy)
        )
        print(f"AllGather-heavy (>80%): Torus avg {avg_speedup:.2f}x faster than Mesh")
        insights["allgather_heavy_avg_mesh_over_torus_speedup"] = avg_speedup

    reduction_heavy = [
        r
        for r in results
        if r["allgather_pct"] < 20
        and "Mesh 4x4x4" in r["lats"]
        and "Torus 4x4x4" in r["lats"]
    ]
    if reduction_heavy:
        avg_speedup = (
            sum(
                r["lats"]["Mesh 4x4x4"] / r["lats"]["Torus 4x4x4"]
                for r in reduction_heavy
            )
            / len(reduction_heavy)
        )
        print(f"Reduction-heavy (<20% all-gather): Torus avg {avg_speedup:.2f}x faster than Mesh")
        insights["reduction_heavy_avg_mesh_over_torus_speedup"] = avg_speedup

    results_path = save_results_in_dir(run_dir, {
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "accelforge_root": str(ACCELFORGE.resolve()),
        "workloads_dir": str(workloads_dir),
        "architecture_yaml": str(ARCH),
        "mappings_dir": str(mappings_root.relative_to(run_dir)),
        "map_chips": MAP_CHIPS,
        "eval_chips": EVAL_CHIPS,
        "scale": SCALE,
        "feedback_loop": {
            "max_proxy_iters": MAX_PROXY_ITERS,
            "damping_enabled": args.damping,
            "proxy_damping": PROXY_DAMPING,
            "proxy_rel_tol": PROXY_REL_TOL,
            "initial_network_proxies": _initial_network_proxies(),
        },
        "topology_summaries": topology_summaries,
        "results": results,
        "insights": insights,
    })

    print(f"\nSaved results to {results_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
