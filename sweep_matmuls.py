#!/usr/bin/env python3
"""
Topology sweep: evaluate workloads on multiple topologies.

Maps on 8 chips, scales data movement to 64 chips, and evaluates candidate topologies.

Usage:
    ACCELFORGE_ROOT=/path/to/accelforge .venv/bin/python sweep_matmuls.py [--damping]
"""

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.disable(logging.WARNING)

from network_topology import compute_network_cost, make_tpu_v4_topology
from network_topology.cost_model import NetworkTransfer, CollectiveType
from network_topology.topology import Torus3D, Mesh3D, Ring, TorusND, CirculantHD
from network_topology.tpu_v4 import ICI_LINK_BW_UNIDIR, ICI_ENERGY_PER_BIT_PER_HOP, ICI_PER_HOP_LATENCY

SCRIPT_DIR = Path(__file__).resolve().parent
ACCELFORGE = Path(os.environ.get("ACCELFORGE_ROOT", SCRIPT_DIR.parent / "accelforge")).expanduser()
ARCH = SCRIPT_DIR / "accelforge_configs" / "tpu_v4_distributed_1d.yaml"
MAP_CHIPS = 8
EVAL_CHIPS = 64
SCALE = EVAL_CHIPS / MAP_CHIPS
DEFAULT_NETWORK_READ_ENERGY = 7.03e-12
DEFAULT_NETWORK_WRITE_ENERGY = 7.03e-12
DEFAULT_NETWORK_READ_LATENCY = 1.0 / (8.0 * 614e9)
DEFAULT_NETWORK_WRITE_LATENCY = 1.0 / (8.0 * 614e9)
MAX_PROXY_ITERS = int(os.environ.get("MAX_PROXY_ITERS", 6))
PROXY_DAMPING = float(os.environ.get("PROXY_DAMPING", 0.5))
PROXY_REL_TOL = float(os.environ.get("PROXY_REL_TOL", 0.05))
DIRECT_GPT3_MODEL = "gpt3_175b_tensor_parallel"

hw = dict(link_bandwidth=ICI_LINK_BW_UNIDIR, energy_per_bit_per_hop=ICI_ENERGY_PER_BIT_PER_HOP,
          per_hop_latency=ICI_PER_HOP_LATENCY)

TOPOLOGIES = {
    "Torus 4x4x4": Torus3D(dims=(4, 4, 4), **hw),
    "Mesh 4x4x4": Mesh3D(dims=(4, 4, 4), **hw),
    "Torus 8x2x4": Torus3D(dims=(8, 2, 4), **hw),
    "Ring 64": Ring(num_chips=64, **hw),
    # Higher-dimensional tori (proposed topologies for Milestone 3)
    "4D Torus 4x4x2x2": TorusND(dims=(4, 4, 2, 2), **hw),     # 6 links/chip
    "5D Torus 4x2x2x2x2": TorusND(dims=(4, 2, 2, 2, 2), **hw), # 6 links/chip
    "6D Hypercube": TorusND(dims=(2, 2, 2, 2, 2, 2), **hw),     # 6 links/chip
    # Optimal circulant (Milestone 3 proposal — same 6 links/chip as Torus)
    "Circulant {1,5,17}": CirculantHD(64, (1, 5, 17), **hw),   # 6 links/chip, 2.29x faster AR
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
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    # Fallback for AccelForge's custom types (Float, etc.) — stringify
    return str(value)


def _serialize_transfer(transfer: NetworkTransfer):
    return {
        "tensor_name": transfer.tensor_name,
        "data_bytes": float(transfer.data_bytes),
        "collective_type": transfer.collective_type.name,
        "src_chip": transfer.src_chip,
        "dst_chips": transfer.dst_chips,
        "participating_chips": transfer.participating_chips,
    }


def _all_eval_chips() -> list[int]:
    return list(range(EVAL_CHIPS))


def _network_transfer_from_decision(decision) -> NetworkTransfer:
    collective_type = CollectiveType[decision["collective_type"]]
    kwargs = {}
    if collective_type == CollectiveType.BROADCAST:
        kwargs["src_chip"] = 0
        kwargs["dst_chips"] = [chip for chip in _all_eval_chips() if chip != 0]
    elif collective_type in (
        CollectiveType.ALLREDUCE,
        CollectiveType.REDUCE_SCATTER,
        CollectiveType.ALLGATHER,
    ):
        kwargs["participating_chips"] = _all_eval_chips()
    return NetworkTransfer(
        tensor_name=f"{decision['einsum']}:{decision['tensor_name']}",
        data_bytes=decision["data_bytes"],
        collective_type=collective_type,
        **kwargs,
    )


def _is_direct_gpt3_workload(params) -> bool:
    return params.get("__direct_model") == DIRECT_GPT3_MODEL


def _direct_gpt3_config(params):
    batch_size = int(params.get("BATCH_SIZE", 1))
    n_tokens = int(params.get("N_TOKENS", 8192))
    n_layers = int(params.get("N_LAYERS", 96))
    num_heads = int(params.get("NUM_HEADS", 96))
    head_dim = int(params.get("HEAD_DIM", 128))
    bytes_per_value = int(params.get("BYTES_PER_VALUE", 1))
    hidden_dim = num_heads * head_dim
    activation_bytes = float(batch_size * n_tokens * hidden_dim * bytes_per_value)
    return {
        "model": DIRECT_GPT3_MODEL,
        "batch_size": batch_size,
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_dim": hidden_dim,
        "bytes_per_value": bytes_per_value,
        "activation_bytes": activation_bytes,
        "collectives_per_layer": 2,
        "collective_pattern": "Megatron-style tensor-parallel forward pass",
    }


def _direct_gpt3_network_transfers(params):
    config = _direct_gpt3_config(params)
    participants = _all_eval_chips()
    transfers = []
    decisions = []

    for layer_idx in range(config["n_layers"]):
        for suffix, reason in (
            (
                "attention_output_allreduce",
                "Tensor-parallel attention output projection is reduced across chips.",
            ),
            (
                "ffn_output_allreduce",
                "Tensor-parallel FFN down projection is reduced across chips.",
            ),
        ):
            einsum_name = f"layer{layer_idx}:{suffix}"
            tensor_name = "activation"
            data_bytes = config["activation_bytes"]
            transfers.append(
                NetworkTransfer(
                    tensor_name=f"{einsum_name}:{tensor_name}",
                    data_bytes=data_bytes,
                    collective_type=CollectiveType.ALLREDUCE,
                    participating_chips=participants,
                )
            )
            decisions.append(
                _make_collective_decision(
                    einsum_name,
                    tensor_name,
                    CollectiveType.ALLREDUCE,
                    "write",
                    data_bytes,
                    reason,
                    {"B", "M", "D"},
                    {"D"},
                )
            )

    return config, transfers, decisions


def _direct_gpt3_mapping_yaml(config):
    return "\n".join(
        [
            "direct_gpt3_175b:",
            f"  model: {config['model']}",
            f"  batch_size: {config['batch_size']}",
            f"  n_tokens: {config['n_tokens']}",
            f"  n_layers: {config['n_layers']}",
            f"  hidden_dim: {config['hidden_dim']}",
            f"  bytes_per_value: {config['bytes_per_value']}",
            f"  activation_bytes: {config['activation_bytes']}",
            "  collectives_per_layer: 2",
            "  collective_type: ALLREDUCE",
            "  note: AccelForge GPT3 QK mapping is bypassed for this direct network model.",
            "",
        ]
    )


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


def save_named_json_in_dir(out_dir: Path, filename: str, payload: dict) -> Path:
    out_path = out_dir / filename
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
    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        help="Run only this topology (e.g. 'Torus 4x4x4'). For parallel Slurm jobs.",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Run only this workload (substring match on name). For parallel Slurm jobs.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Shared output directory. If set, results are saved as <run-dir>/<topology>.json.",
    )
    return parser.parse_args()


def _tensor_rank_vars(einsum, tensor_name):
    for tensor_access in einsum.tensor_accesses:
        if tensor_access.name == tensor_name:
            return set(dict(tensor_access.projection).values())
    raise KeyError(f"Tensor {tensor_name} not found in Einsum {einsum.name}")


def _rank_var_set(value):
    if isinstance(value, (set, frozenset, list, tuple)):
        return {str(v) for v in value}
    return {str(value)}


def _spatial_rank_vars_for_einsum(spatial, einsum_name):
    einsum_to_rank_variables = getattr(spatial, "_einsum_to_rank_variables", None) or {}
    if einsum_to_rank_variables:
        for key, value in einsum_to_rank_variables.items():
            if str(key) == str(einsum_name):
                return _rank_var_set(value)
        return set()
    return _rank_var_set(spatial.rank_variable)


def _chip_sharded_rank_vars(einsum_name, mapping, tensor_rank_vars):
    from accelforge.frontend.mapping.mapping import Spatial

    tensor_rank_vars = {str(rank_var) for rank_var in tensor_rank_vars}
    sharded = set()
    for spatial in mapping.get_nodes_of_type(Spatial):
        if spatial.component != "ChipArray":
            continue
        if getattr(spatial, "_constrained_to_one", False):
            continue
        if getattr(spatial, "calculated_n_iterations", None) == 1:
            continue
        sharded.update(
            rank_var
            for rank_var in _spatial_rank_vars_for_einsum(spatial, einsum_name)
            if rank_var in tensor_rank_vars
        )
    return sharded


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


def _mapping_sha256(mapping_yaml: str) -> str:
    return hashlib.sha256(mapping_yaml.encode("utf-8")).hexdigest()


def _delta_summary(before: float, after: float):
    delta = float(after - before)
    if abs(before) <= 1e-30:
        relative_change = 0.0 if abs(after) <= 1e-30 else None
    else:
        relative_change = delta / float(before)
    return {
        "before": float(before),
        "after": float(after),
        "delta": delta,
        "relative_change": relative_change,
        "percent_change": (
            None if relative_change is None else float(relative_change * 100.0)
        ),
    }


def _summarize_mapping_costs(total_latency, energy_details, latency_details):
    energy_by_component = defaultdict(float)
    per_einsum = defaultdict(
        lambda: {
            "energy_by_component_scaled": defaultdict(float),
            "latency_by_component": {},
            "non_network_energy_scaled": 0.0,
            "proxy_network_energy_scaled": 0.0,
        }
    )

    for (einsum_name, component, _tensor, _action), value in energy_details.items():
        scaled_value = float(value) * SCALE
        energy_by_component[component] += scaled_value
        bucket = per_einsum[einsum_name]
        bucket["energy_by_component_scaled"][component] += scaled_value
        if component == "NetworkMemory":
            bucket["proxy_network_energy_scaled"] += scaled_value
        else:
            bucket["non_network_energy_scaled"] += scaled_value

    for (einsum_name, component), value in latency_details.items():
        per_einsum[einsum_name]["latency_by_component"][component] = float(value)

    proxy_total_latency_reconstructed = 0.0
    non_network_total_latency_reconstructed = 0.0
    normalized_per_einsum = {}
    for einsum_name, bucket in sorted(per_einsum.items()):
        latency_by_component = dict(sorted(bucket["latency_by_component"].items()))
        proxy_network_latency = float(latency_by_component.get("NetworkMemory", 0.0))
        non_network_bottleneck_latency = max(
            (
                latency
                for component, latency in latency_by_component.items()
                if component != "NetworkMemory"
            ),
            default=0.0,
        )
        proxy_total_latency = max(latency_by_component.values(), default=0.0)
        proxy_total_latency_reconstructed += proxy_total_latency
        non_network_total_latency_reconstructed += non_network_bottleneck_latency
        normalized_per_einsum[einsum_name] = {
            "energy_by_component_scaled": dict(
                sorted(bucket["energy_by_component_scaled"].items())
            ),
            "latency_by_component": latency_by_component,
            "non_network_energy_scaled": float(bucket["non_network_energy_scaled"]),
            "proxy_network_energy_scaled": float(bucket["proxy_network_energy_scaled"]),
            "non_network_bottleneck_latency": non_network_bottleneck_latency,
            "proxy_network_latency": proxy_network_latency,
            "proxy_total_latency": proxy_total_latency,
        }

    proxy_total_energy_scaled = float(sum(energy_by_component.values()))
    proxy_network_energy_scaled = float(energy_by_component.get("NetworkMemory", 0.0))
    return {
        "proxy_total_energy_scaled": proxy_total_energy_scaled,
        "proxy_non_network_energy_scaled": (
            proxy_total_energy_scaled - proxy_network_energy_scaled
        ),
        "proxy_network_energy_scaled": proxy_network_energy_scaled,
        "proxy_total_latency": float(total_latency),
        "proxy_total_latency_reconstructed": float(proxy_total_latency_reconstructed),
        "non_network_total_latency_reconstructed": float(
            non_network_total_latency_reconstructed
        ),
        "energy_by_component_scaled": dict(sorted(energy_by_component.items())),
        "per_einsum": normalized_per_einsum,
    }


def _summarize_actual_network(decisions, network_result):
    per_collective_type = defaultdict(
        lambda: {
            "count": 0,
            "data_bytes": 0.0,
            "estimated_network_energy": 0.0,
            "assigned_network_latency": 0.0,
        }
    )
    per_einsum = defaultdict(
        lambda: {
            "data_bytes": 0.0,
            "estimated_network_energy": 0.0,
            "assigned_network_latency": 0.0,
            "collective_counts": defaultdict(int),
            "collective_bytes": defaultdict(float),
            "collectives": [],
        }
    )

    for decision in decisions:
        collective_type = decision["collective_type"]
        data_bytes = float(decision["data_bytes"])
        if collective_type is None or data_bytes <= 0:
            continue

        energy = float(decision.get("estimated_network_energy", 0.0))
        latency = float(decision.get("estimated_network_latency", 0.0))
        per_collective_type[collective_type]["count"] += 1
        per_collective_type[collective_type]["data_bytes"] += data_bytes
        per_collective_type[collective_type]["estimated_network_energy"] += energy
        per_collective_type[collective_type]["assigned_network_latency"] += latency

        bucket = per_einsum[decision["einsum"]]
        bucket["data_bytes"] += data_bytes
        bucket["estimated_network_energy"] += energy
        bucket["assigned_network_latency"] += latency
        bucket["collective_counts"][collective_type] += 1
        bucket["collective_bytes"][collective_type] += data_bytes
        bucket["collectives"].append(
            {
                "tensor_name": decision["tensor_name"],
                "collective_type": collective_type,
                "proxy_action": decision.get("proxy_action"),
                "data_bytes": data_bytes,
                "estimated_network_energy": energy,
                "estimated_network_latency": latency,
                "reason": decision.get("reason"),
            }
        )

    normalized_per_einsum = {}
    for einsum_name, bucket in sorted(per_einsum.items()):
        normalized_per_einsum[einsum_name] = {
            "data_bytes": float(bucket["data_bytes"]),
            "estimated_network_energy": float(bucket["estimated_network_energy"]),
            "assigned_network_latency": float(bucket["assigned_network_latency"]),
            "collective_counts": dict(sorted(bucket["collective_counts"].items())),
            "collective_bytes": dict(sorted(bucket["collective_bytes"].items())),
            "collectives": sorted(
                bucket["collectives"], key=lambda item: item["tensor_name"]
            ),
        }

    return {
        "total_energy": float(network_result.total_energy),
        "total_latency": float(network_result.total_latency),
        "total_bytes": float(network_result.total_network_bytes),
        "energy_per_byte": float(network_result.energy_per_network_access),
        "latency_per_byte": float(network_result.latency_per_network_access),
        "per_transfer": _json_ready(network_result.per_transfer),
        "per_collective_type": dict(sorted(per_collective_type.items())),
        "per_einsum": normalized_per_einsum,
    }


def _estimate_actual_system_cost(mapping_cost_summary, network_summary):
    per_einsum = {}
    total_latency = 0.0
    einsum_names = sorted(
        set(mapping_cost_summary["per_einsum"]) | set(network_summary["per_einsum"])
    )
    for einsum_name in einsum_names:
        mapping_einsum = mapping_cost_summary["per_einsum"].get(einsum_name, {})
        network_einsum = network_summary["per_einsum"].get(einsum_name, {})

        non_network_energy_scaled = float(
            mapping_einsum.get("non_network_energy_scaled", 0.0)
        )
        proxy_network_energy_scaled = float(
            mapping_einsum.get("proxy_network_energy_scaled", 0.0)
        )
        actual_network_energy = float(
            network_einsum.get("estimated_network_energy", 0.0)
        )
        non_network_bottleneck_latency = float(
            mapping_einsum.get("non_network_bottleneck_latency", 0.0)
        )
        proxy_network_latency = float(mapping_einsum.get("proxy_network_latency", 0.0))
        proxy_total_latency = float(mapping_einsum.get("proxy_total_latency", 0.0))
        assigned_network_latency = float(
            network_einsum.get("assigned_network_latency", 0.0)
        )
        estimated_actual_total_latency = max(
            non_network_bottleneck_latency, assigned_network_latency
        )
        total_latency += estimated_actual_total_latency

        per_einsum[einsum_name] = {
            "non_network_energy_scaled": non_network_energy_scaled,
            "proxy_network_energy_scaled": proxy_network_energy_scaled,
            "actual_network_energy": actual_network_energy,
            "non_network_bottleneck_latency": non_network_bottleneck_latency,
            "proxy_network_latency": proxy_network_latency,
            "proxy_total_latency": proxy_total_latency,
            "assigned_network_latency": assigned_network_latency,
            "estimated_actual_total_energy_scaled": (
                non_network_energy_scaled + actual_network_energy
            ),
            "estimated_actual_total_latency": estimated_actual_total_latency,
        }

    return {
        "estimated_total_energy_scaled": (
            float(mapping_cost_summary["proxy_non_network_energy_scaled"])
            + float(network_summary["total_energy"])
        ),
        "estimated_total_latency": float(total_latency),
        "network_only_total_energy": float(network_summary["total_energy"]),
        "network_only_total_latency": float(network_summary["total_latency"]),
        "network_only_total_bytes": float(network_summary["total_bytes"]),
        "per_einsum": per_einsum,
    }


def _collective_signature(decisions):
    return sorted(
        (
            decision["einsum"],
            decision["tensor_name"],
            decision["collective_type"],
            decision["proxy_action"],
            tuple(decision.get("chip_sharded_ranks", [])),
        )
        for decision in decisions
    )


def _build_milestone2_topology_summary(topology_result):
    iterations = topology_result["feedback_iterations"]
    if not iterations:
        return {}

    trace = []
    previous_mapping_hash = None
    previous_collective_signature = None
    for iteration in iterations:
        mapping_hash = iteration["mapping"]["mapping_sha256"]
        collective_signature = _collective_signature(iteration["collective_decisions"])
        trace.append(
            {
                "iteration": iteration["iteration"],
                "mapping_yaml_path": iteration["mapping"]["mapping_yaml_path"],
                "mapping_sha256": mapping_hash,
                "mapping_changed_from_previous": (
                    None
                    if previous_mapping_hash is None
                    else mapping_hash != previous_mapping_hash
                ),
                "collective_plan_changed_from_previous": (
                    None
                    if previous_collective_signature is None
                    else collective_signature != previous_collective_signature
                ),
                "input_network_proxies": iteration["input_network_proxies"],
                "updated_network_proxies": iteration["updated_network_proxies"],
                "raw_relative_change": iteration["raw_relative_change"],
                "applied_relative_change": iteration["applied_relative_change"],
                "proxy_model_total_energy_scaled": iteration["mapping"]["cost_summary"][
                    "proxy_total_energy_scaled"
                ],
                "proxy_model_total_latency": iteration["mapping"]["cost_summary"][
                    "proxy_total_latency"
                ],
                "actual_network_total_energy": iteration["network"]["actual_summary"][
                    "total_energy"
                ],
                "actual_network_total_latency": iteration["network"]["actual_summary"][
                    "total_latency"
                ],
                "actual_network_total_bytes": iteration["network"]["actual_summary"][
                    "total_bytes"
                ],
                "estimated_actual_total_energy_scaled": iteration[
                    "estimated_actual_system"
                ]["estimated_total_energy_scaled"],
                "estimated_actual_total_latency": iteration[
                    "estimated_actual_system"
                ]["estimated_total_latency"],
                "allgather_pct": iteration["allgather_pct"],
            }
        )
        previous_mapping_hash = mapping_hash
        previous_collective_signature = collective_signature

    first_iteration = iterations[0]
    final_iteration = iterations[-1]

    return {
        "feedback_converged": topology_result["feedback_converged"],
        "iteration_count": len(iterations),
        "baseline_without_network_model": {
            "iteration": first_iteration["iteration"],
            "mapping_yaml_path": first_iteration["mapping"]["mapping_yaml_path"],
            "mapping_sha256": first_iteration["mapping"]["mapping_sha256"],
            "network_proxies_used": first_iteration["input_network_proxies"],
            "proxy_model": first_iteration["mapping"]["cost_summary"],
        },
        "first_run_with_actual_network_applied": {
            "iteration": first_iteration["iteration"],
            "mapping_yaml_path": first_iteration["mapping"]["mapping_yaml_path"],
            "mapping_sha256": first_iteration["mapping"]["mapping_sha256"],
            "proxy_model": first_iteration["mapping"]["cost_summary"],
            "actual_network": first_iteration["network"]["actual_summary"],
            "estimated_actual_system": first_iteration["estimated_actual_system"],
            "collective_decisions": first_iteration["collective_decisions"],
        },
        "final_stabilized_mapping": {
            "iteration": final_iteration["iteration"],
            "mapping_yaml_path": final_iteration["mapping"]["mapping_yaml_path"],
            "mapping_sha256": final_iteration["mapping"]["mapping_sha256"],
            "network_proxies_used": final_iteration["input_network_proxies"],
            "final_network_proxies": topology_result["final_network_proxies"],
            "proxy_model": final_iteration["mapping"]["cost_summary"],
            "actual_network": final_iteration["network"]["actual_summary"],
            "estimated_actual_system": final_iteration["estimated_actual_system"],
            "collective_decisions": final_iteration["collective_decisions"],
        },
        "changes": {
            "mapping_hash_changed_first_to_final": (
                first_iteration["mapping"]["mapping_sha256"]
                != final_iteration["mapping"]["mapping_sha256"]
            ),
            "collective_plan_changed_first_to_final": (
                _collective_signature(first_iteration["collective_decisions"])
                != _collective_signature(final_iteration["collective_decisions"])
            ),
            "without_network_proxy_to_first_actual": {
                "energy": _delta_summary(
                    first_iteration["mapping"]["cost_summary"][
                        "proxy_total_energy_scaled"
                    ],
                    first_iteration["estimated_actual_system"][
                        "estimated_total_energy_scaled"
                    ],
                ),
                "latency": _delta_summary(
                    first_iteration["mapping"]["cost_summary"]["proxy_total_latency"],
                    first_iteration["estimated_actual_system"]["estimated_total_latency"],
                ),
            },
            "first_actual_to_final_actual": {
                "energy": _delta_summary(
                    first_iteration["estimated_actual_system"][
                        "estimated_total_energy_scaled"
                    ],
                    final_iteration["estimated_actual_system"][
                        "estimated_total_energy_scaled"
                    ],
                ),
                "latency": _delta_summary(
                    first_iteration["estimated_actual_system"]["estimated_total_latency"],
                    final_iteration["estimated_actual_system"]["estimated_total_latency"],
                ),
            },
            "without_network_proxy_to_final_actual": {
                "energy": _delta_summary(
                    first_iteration["mapping"]["cost_summary"][
                        "proxy_total_energy_scaled"
                    ],
                    final_iteration["estimated_actual_system"][
                        "estimated_total_energy_scaled"
                    ],
                ),
                "latency": _delta_summary(
                    first_iteration["mapping"]["cost_summary"]["proxy_total_latency"],
                    final_iteration["estimated_actual_system"]["estimated_total_latency"],
                ),
            },
        },
        "convergence_trace": trace,
    }


def _build_milestone2_payload(
    run_dir: Path,
    results,
    topology_summaries,
    args,
    workloads_dir: Path,
):
    workload_entries = []
    aggregate = defaultdict(
        lambda: {
            "count": 0,
            "final_estimated_total_energy_scaled_sum": 0.0,
            "final_estimated_total_latency_sum": 0.0,
        }
    )
    best_latency_topology_counts = defaultdict(int)
    best_energy_topology_counts = defaultdict(int)

    for workload_result in results:
        topology_entries = {}
        final_latency_by_topology = {}
        final_energy_by_topology = {}

        for topology_name, topology_result in workload_result["topologies"].items():
            summary = _build_milestone2_topology_summary(topology_result)
            topology_entries[topology_name] = summary
            final_entry = summary["final_stabilized_mapping"]["estimated_actual_system"]
            final_energy = float(final_entry["estimated_total_energy_scaled"])
            final_latency = float(final_entry["estimated_total_latency"])
            final_energy_by_topology[topology_name] = final_energy
            final_latency_by_topology[topology_name] = final_latency
            aggregate[topology_name]["count"] += 1
            aggregate[topology_name]["final_estimated_total_energy_scaled_sum"] += final_energy
            aggregate[topology_name]["final_estimated_total_latency_sum"] += final_latency

        best_latency_topology = min(final_latency_by_topology, key=final_latency_by_topology.get)
        best_energy_topology = min(final_energy_by_topology, key=final_energy_by_topology.get)
        best_latency_topology_counts[best_latency_topology] += 1
        best_energy_topology_counts[best_energy_topology] += 1

        workload_entries.append(
            {
                "desc": workload_result["desc"],
                "workload_path": workload_result["workload_path"],
                "params": workload_result["params"],
                "topologies": topology_entries,
                "cross_topology": {
                    "best_final_latency_topology": best_latency_topology,
                    "best_final_energy_topology": best_energy_topology,
                    "final_latency_rankings": [
                        {"topology": name, "estimated_total_latency": value}
                        for name, value in sorted(
                            final_latency_by_topology.items(), key=lambda item: item[1]
                        )
                    ],
                    "final_energy_rankings": [
                        {"topology": name, "estimated_total_energy_scaled": value}
                        for name, value in sorted(
                            final_energy_by_topology.items(), key=lambda item: item[1]
                        )
                    ],
                },
            }
        )

    return {
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "source_results_json": str(Path(run_dir.name) / "results.json"),
        "milestone": "Milestone 2",
        "questions": [
            "How much is the difference between the energy with and without the network model?",
            "How much is the difference between the first run of the mapper, but including the network energy and latency costs, and the final mapper run?",
        ],
        "methodology": {
            "baseline_without_network_model": (
                "Iteration 1 mapper cost with the initial logical NetworkMemory "
                "proxy values before topology-aware feedback."
            ),
            "first_run_with_actual_network_applied": (
                "Iteration 1 mapping, with the actual topology-model network "
                "energy and latency substituted for the proxy NetworkMemory cost."
            ),
            "final_stabilized_mapping": (
                "Last feedback-loop iteration after the network proxy values stop "
                "changing or the max-iteration limit is reached."
            ),
            "energy_note": (
                "AccelForge energies come from the 8-chip mapping run and are scaled "
                f"by {SCALE:.0f}x to estimate the 64-chip system."
            ),
            "latency_note": (
                "Estimated actual total latency is reconstructed per einsum as "
                "max(non-network bottleneck latency, assigned network latency), "
                "then summed across einsums. The network-only latency is the sum "
                "of independently costed collective latencies."
            ),
        },
        "accelforge_root": str(ACCELFORGE.resolve()),
        "workloads_dir": str(workloads_dir),
        "architecture_yaml": str(ARCH),
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
        "aggregate_topology_summary": {
            topology_name: {
                "num_workloads": values["count"],
                "avg_final_estimated_total_energy_scaled": (
                    values["final_estimated_total_energy_scaled_sum"] / values["count"]
                    if values["count"] > 0
                    else 0.0
                ),
                "avg_final_estimated_total_latency": (
                    values["final_estimated_total_latency_sum"] / values["count"]
                    if values["count"] > 0
                    else 0.0
                ),
                "best_final_latency_wins": best_latency_topology_counts.get(
                    topology_name, 0
                ),
                "best_final_energy_wins": best_energy_topology_counts.get(
                    topology_name, 0
                ),
            }
            for topology_name, values in sorted(aggregate.items())
        },
        "workloads": workload_entries,
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
            estimated_latency = float(transfer["latency"])
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


def _fallback_network_access_decisions(
    einsum_name,
    tensor_infos,
    read_bytes,
    write_bytes,
    read_collective_type=CollectiveType.ALLGATHER,
    reason_prefix="Fallback NetworkMemory model",
):
    decisions = []
    for name, info in tensor_infos.items():
        rbytes = float(read_bytes.get(name, 0.0))
        wbytes = float(write_bytes.get(name, 0.0))
        if rbytes > 0 and not info["is_output"]:
            decisions.append(_make_collective_decision(
                einsum_name,
                name,
                read_collective_type,
                "read",
                rbytes,
                f"{reason_prefix}: treating read as {read_collective_type.name}",
                info["tensor_ranks"],
                info["chip_sharded_ranks"],
            ))
        if wbytes > 0 and info["is_output"]:
            decisions.append(_make_collective_decision(
                einsum_name,
                name,
                CollectiveType.ALLREDUCE,
                "write",
                wbytes,
                f"{reason_prefix}: treating write as ALLREDUCE",
                info["tensor_ranks"],
                info["chip_sharded_ranks"],
            ))
    return decisions


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
        from accelforge.util import set_n_parallel_jobs
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Unable to import accelforge. Install it in the active environment or "
            "point ACCELFORGE_ROOT at a local checkout."
        ) from exc

    n_jobs = os.environ.get("ACCELFORGE_N_JOBS") or os.environ.get("SLURM_CPUS_PER_TASK")
    if n_jobs:
        try:
            parsed_n_jobs = int(n_jobs)
        except ValueError:
            print(f"WARNING: ignoring invalid AccelForge parallelism value: {n_jobs!r}")
        else:
            if parsed_n_jobs > 0:
                set_n_parallel_jobs(parsed_n_jobs, print_message=True)

    return af, map_workload_to_arch


def make_workloads(workloads_dir: Path):
    return [
        # === AccelForge transformer workload templates ===
        (
            "GPT3 175B",
            workloads_dir / "gpt3_175B.yaml",
            {
                "__direct_model": DIRECT_GPT3_MODEL,
                "BATCH_SIZE": 1,
                "N_TOKENS": 8192,
                "N_LAYERS": 96,
                "BYTES_PER_VALUE": 1,
            },
        ),

        # === Smaller matmuls for faster debugging/sanity sweeps ===
        ("Small 2Kx8K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 2048, "KN": 8192}),
        ("Small 4Kx16K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 4096, "KN": 16384}),
        ("Medium 8Kx32K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 8192, "KN": 32768}),
        ("Medium 16Kx32K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 16384, "KN": 32768}),

        # === Giant matmuls that force chip sharding (total working set > 32 GiB) ===
        # Wide (FFN-like): T0=2GB, W0=69GB, T1=2GB → W0 alone forces sharding
        ("Wide 8Kx256K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 8192, "KN": 262144}),
        # Very wide (decode-like): T0=0.5GB, W0=69GB, T1=0.5GB → weight-dominated
        ("VeryWide 2Kx256K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 2048, "KN": 262144}),
        # Rectangular: T0=8GB, W0=32GB, T1=8GB → mixed
        ("Rect 64Kx128K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 65536, "KN": 131072}),
        # Square: T0=16GB, W0=16GB, T1=16GB → 48 GB total
        ("Square 128Kx128K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 131072, "KN": 131072}),
        # Tall (activation-heavy): T0=17GB, W0=4GB, T1=17GB → 38 GB total
        ("Tall 256Kx64K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 262144, "KN": 65536}),
        # Large square (2x bigger): T0=64GB, W0=64GB, T1=64GB → heavily sharded
        ("LargeSquare 256Kx256K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 262144, "KN": 262144}),
        # Very tall: T0=69GB, W0=0.5GB, T1=69GB → activation-dominated
        ("VeryTall 256Kx8K", workloads_dir / "matmuls.yaml",
         {"N_EINSUMS": 1, "M": 262144, "KN": 8192}),
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
    total_mapping_latency = float(mappings.latency())
    actions = mappings.actions(per_einsum=True, per_component=True, per_tensor=True)
    energy_details = mappings.energy(
        per_einsum=True, per_component=True, per_tensor=True, per_action=True
    )
    latency_details = mappings.latency(per_einsum=True, per_component=True)
    mapping = mappings.mapping(0)
    mapping_yaml = _mapping_to_yaml(mapping)
    mapping_cost_summary = _summarize_mapping_costs(
        total_mapping_latency, energy_details, latency_details
    )
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
    raw_network_access_decisions = []

    for einsum_name in mappings.einsum_names:
        einsum = spec.workload.einsums[einsum_name]
        tensor_infos = {}
        for tensor_access in einsum.tensor_accesses:
            tensor_ranks = _tensor_rank_vars(einsum, tensor_access.name)
            tensor_infos[tensor_access.name] = {
                "name": tensor_access.name,
                "tensor_ranks": tensor_ranks,
                "chip_sharded_ranks": _chip_sharded_rank_vars(
                    einsum_name, mapping, tensor_ranks
                ),
                "is_output": tensor_access.output,
            }

        per_tensor_read_bytes = {
            name: network_reads[(einsum_name, name)] for name in tensor_infos
        }
        per_tensor_write_bytes = {
            name: network_writes[(einsum_name, name)] for name in tensor_infos
        }
        raw_network_access_decisions.extend(
            _fallback_network_access_decisions(
                einsum_name,
                tensor_infos,
                per_tensor_read_bytes,
                per_tensor_write_bytes,
                read_collective_type=CollectiveType.BROADCAST,
                reason_prefix="Raw NetworkMemory fallback",
            )
        )

        try:
            decisions = _infer_matmul_collectives(
                einsum_name,
                tensor_infos,
                per_tensor_read_bytes,
                per_tensor_write_bytes,
                tensor_size_bytes,
            )
        except ValueError:
            # Not a 2-input matmul (e.g. copy op, softmax, elementwise).
            # Use per-einsum fallback: reads → ALLGATHER, writes → ALLREDUCE.
            decisions = _fallback_network_access_decisions(
                einsum_name,
                tensor_infos,
                per_tensor_read_bytes,
                per_tensor_write_bytes,
                reason_prefix="Non-matmul fallback",
            )
        collective_decisions.extend(decisions)

        for decision in decisions:
            if decision["collective_type"] is None or decision["data_bytes"] <= 0:
                continue
            transfers.append(_network_transfer_from_decision(decision))

    # Fallback: if sharding-based inference produced nothing, emit raw decisions
    # and transfers for every NetworkMemory access.
    # This is less precise than the sharding analysis but ensures non-zero traffic
    # when AccelForge picks a 1-chip mapping that bypasses chip parallelism.
    if not transfers:
        for decision in raw_network_access_decisions:
            if decision["collective_type"] is None or decision["data_bytes"] <= 0:
                continue
            collective_decisions.append(decision)
            transfers.append(_network_transfer_from_decision(decision))

    return {
        "compute_energy": compute_e,
        "compute_latency": total_mapping_latency,
        "transfers": transfers,
        "collective_decisions": collective_decisions,
        "network_proxies": dict(network_proxies),
        "mapping_yaml": mapping_yaml,
        "mapping_sha256": _mapping_sha256(mapping_yaml),
        "mapping_cost_summary": mapping_cost_summary,
        "mapping_energy": _json_ready(energy_details),
        "mapping_latency": _json_ready(latency_details),
        "mapping_actions": _json_ready(actions),
    }


def run_direct_gpt3_workload(
    params,
    topology,
    mapping_dir: Path | None = None,
):
    config, transfers, decisions = _direct_gpt3_network_transfers(params)
    mapping_yaml = _direct_gpt3_mapping_yaml(config)
    mapping_yaml_relpath = None
    if mapping_dir is not None:
        mapping_dir.mkdir(parents=True, exist_ok=True)
        mapping_yaml_path = mapping_dir / "iter_001.yaml"
        mapping_yaml_path.write_text(mapping_yaml, encoding="utf-8")
        mapping_yaml_relpath = str(
            Path("mappings") / mapping_yaml_path.relative_to(mapping_dir.parents[1])
        )

    network_result = compute_network_cost(topology, transfers)
    collective_decisions = _annotate_collective_decisions(decisions, network_result)
    network_summary = _summarize_actual_network(collective_decisions, network_result)
    mapping_cost_summary = {
        "proxy_total_energy_scaled": 0.0,
        "proxy_non_network_energy_scaled": 0.0,
        "proxy_network_energy_scaled": 0.0,
        "proxy_total_latency": 0.0,
        "proxy_total_latency_reconstructed": 0.0,
        "non_network_total_latency_reconstructed": 0.0,
        "energy_by_component_scaled": {},
        "per_einsum": {},
        "direct_model": config,
    }
    estimated_actual_system = _estimate_actual_system_cost(
        mapping_cost_summary, network_summary
    )
    total_bytes, allgather_pct = _collective_mix(transfers)
    network_proxies = _initial_network_proxies()
    mapping_sha256 = _mapping_sha256(mapping_yaml)
    mapping_result = {
        "compute_energy": 0.0,
        "compute_latency": 0.0,
        "transfers": transfers,
        "collective_decisions": collective_decisions,
        "network_proxies": dict(network_proxies),
        "mapping_yaml": mapping_yaml,
        "mapping_yaml_path": mapping_yaml_relpath,
        "mapping_sha256": mapping_sha256,
        "mapping_cost_summary": mapping_cost_summary,
        "mapping_energy": {},
        "mapping_latency": {},
        "mapping_actions": {},
        "direct_model": config,
    }

    iterations = [
        {
            "iteration": 1,
            "input_network_proxies": dict(network_proxies),
            "proposed_network_proxies": dict(network_proxies),
            "updated_network_proxies": dict(network_proxies),
            "raw_relative_change": 0.0,
            "applied_relative_change": 0.0,
            "mapping": {
                "compute_energy": 0.0,
                "compute_latency": 0.0,
                "mapping_yaml_path": mapping_yaml_relpath,
                "mapping_sha256": mapping_sha256,
                "cost_summary": mapping_cost_summary,
                "mapping_energy": {},
                "mapping_latency": {},
                "mapping_actions": {},
            },
            "network": {
                "total_energy": float(network_result.total_energy),
                "total_latency": float(network_result.total_latency),
                "energy_per_network_access": float(
                    network_result.energy_per_network_access
                ),
                "latency_per_network_access": float(
                    network_result.latency_per_network_access
                ),
                "total_network_bytes": float(network_result.total_network_bytes),
                "per_transfer": _json_ready(network_result.per_transfer),
                "actual_summary": network_summary,
            },
            "transfer_count": len(transfers),
            "total_bytes": total_bytes,
            "allgather_pct": allgather_pct,
            "collective_decisions": collective_decisions,
            "estimated_actual_system": estimated_actual_system,
            "direct_model": config,
        }
    ]

    return {
        "converged": True,
        "iterations": iterations,
        "final_network_proxies": dict(network_proxies),
        "final_mapping_result": mapping_result,
        "final_network_result": network_result,
        "final_transfers": transfers,
        "final_collective_decisions": collective_decisions,
        "total_bytes": total_bytes,
        "allgather_pct": allgather_pct,
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

    if _is_direct_gpt3_workload(params):
        return run_direct_gpt3_workload(params, topology, mapping_dir)

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
        network_summary = _summarize_actual_network(collective_decisions, network_result)
        estimated_actual_system = _estimate_actual_system_cost(
            mapping_result["mapping_cost_summary"], network_summary
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
                    "mapping_sha256": mapping_result["mapping_sha256"],
                    "cost_summary": mapping_result["mapping_cost_summary"],
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
                    "actual_summary": network_summary,
                },
                "transfer_count": len(mapping_result["transfers"]),
                "total_bytes": total_bytes,
                "allgather_pct": allgather_pct,
                "collective_decisions": collective_decisions,
                "estimated_actual_system": estimated_actual_system,
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

    # Filter workloads if --workload is specified
    if args.workload:
        workloads = [(d, p, prm) for d, p, prm in workloads if args.workload in d]
        if not workloads:
            raise SystemExit(
                f"No workload matching '{args.workload}'. "
                f"Available: {', '.join(d for d, _, _ in make_workloads(workloads_dir))}"
            )

    # Filter topologies if --topology is specified (for parallel Slurm jobs)
    active_topologies = TOPOLOGIES
    if args.topology:
        if args.topology not in TOPOLOGIES:
            raise SystemExit(
                f"Unknown topology '{args.topology}'. "
                f"Available: {', '.join(TOPOLOGIES.keys())}"
            )
        active_topologies = {args.topology: TOPOLOGIES[args.topology]}

    # Use shared run-dir if specified, otherwise create a new one
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir()
    mappings_root = run_dir / "mappings"

    print("=" * 120)
    print(
        f"Topology Sweep: {len(workloads)} workloads x {len(active_topologies)} topologies "
        f"({EVAL_CHIPS} chips, independent collective costs, iterative feedback, "
        f"damping={'on' if args.damping else 'off'})"
    )
    print("=" * 120)

    print(f"\n{'Topology':<18s} {'Diam':>5s} {'AvgHop':>7s} {'Degree':>7s} {'BisectBW':>10s}")
    print("-" * 50)
    topology_summaries = {}
    for tn in active_topologies:
        s = active_topologies[tn].summary()
        topology_summaries[tn] = _json_ready(s)
        print(f"{tn:<18s} {s['diameter']:>5d} {s['avg_hops']:>7.2f} {s['min_degree']:>3d}-{s['max_degree']:<3d} "
              f"{s['bisection_bandwidth_TB_s']:>8.2f} TB/s")

    results = []
    failures = []
    for desc, path, params in workloads:
        print(f"\nWorkload: {desc}")
        lats = {}
        bytes_by_topology = {}
        allgather_pct_by_topology = {}
        compute_e_by_topology = {}
        compute_l_by_topology = {}
        mapping_wall_time_s_by_topology = {}
        topology_results = {}
        for topology_name, topo in active_topologies.items():
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
                elapsed = time.time() - t0
                print(f"FAILED after {elapsed:.0f}s: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                failures.append(
                    {
                        "workload": desc,
                        "workload_path": str(path),
                        "params": dict(params),
                        "topology": topology_name,
                        "elapsed_s": float(elapsed),
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                )
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
    print("RESULTS (latency in ms, summed independent collective costs)")
    print(f"{'=' * 120}")
    print(f"{'Workload':>28s} {'Bytes':>10s} {'AG%':>7s}", end="")
    for tn in active_topologies:
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

    payload = {
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
        "failures": failures,
        "insights": insights,
    }

    # When running a subset (parallel mode), save per-combination JSON
    if args.topology or args.workload:
        parts = []
        if args.workload:
            parts.append(_safe_path_part(args.workload))
        if args.topology:
            parts.append(_safe_path_part(args.topology))
        combo_filename = "__".join(parts) + ".json"
        combo_path = run_dir / combo_filename
        with combo_path.open("w", encoding="utf-8") as f:
            json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
        print(f"\nSaved results to {combo_path}")
    else:
        results_path = save_results_in_dir(run_dir, payload)
        milestone2_path = save_named_json_in_dir(
            run_dir,
            "milestone2_analysis.json",
            _build_milestone2_payload(run_dir, results, topology_summaries, args, workloads_dir),
        )
        print(f"\nSaved results to {results_path}")
        print(f"Saved milestone 2 analysis to {milestone2_path}")

    print("\nDone!")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
