#!/usr/bin/env python3
"""
Direct network comparison for realistic transformer workloads.

This bypasses AccelForge and evaluates the network traffic patterns that show
up in tensor-parallel inference and data-parallel training. The goal is to
compare topology-specific collective schedules directly:

  - 3D torus: per-dimension ring collectives
  - 3D mesh: per-dimension linear pipeline collectives
  - ring: single ring collectives
  - 6D hypercube: per-dimension ring collectives
  - circulant: three edge-disjoint Hamiltonian ring collectives

Usage:
    python analyze_gpt3_stress.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_topology import compute_network_cost
from network_topology.cost_model import CollectiveType, NetworkTransfer
from network_topology.topology import CirculantHD, Mesh3D, Ring, Topology, Torus3D, TorusND
from network_topology.tpu_v4 import (
    ICI_ENERGY_PER_BIT_PER_HOP,
    ICI_LINK_BW_UNIDIR,
    ICI_PER_HOP_LATENCY,
)

NUM_CHIPS = 64
BYTES_PER_VALUE = 2

hw = dict(
    link_bandwidth=ICI_LINK_BW_UNIDIR,
    energy_per_bit_per_hop=ICI_ENERGY_PER_BIT_PER_HOP,
    per_hop_latency=ICI_PER_HOP_LATENCY,
)

TOPOLOGIES = {
    "Circulant {1,5,17}": CirculantHD(64, (1, 5, 17), **hw),
    "6D Hypercube": TorusND(dims=(2, 2, 2, 2, 2, 2), **hw),
    "Torus 4x4x4": Torus3D(dims=(4, 4, 4), **hw),
    "Mesh 4x4x4": Mesh3D(dims=(4, 4, 4), **hw),
    "Ring 64": Ring(num_chips=64, **hw),
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_layers: int
    hidden_dim: int
    ffn_dim: int
    num_heads: int
    parameter_count_b: float
    bytes_per_value: int = BYTES_PER_VALUE


@dataclass(frozen=True)
class Scenario:
    name: str
    batch_size: int
    seq_len: int
    category: str


@dataclass(frozen=True)
class WorkloadCase:
    name: str
    model: ModelConfig
    scenario: Scenario
    strategy: str
    transfers: list[NetworkTransfer]


MODELS = {
    "Llama-3-8B": ModelConfig(
        "Llama-3-8B", num_layers=32, hidden_dim=4096, ffn_dim=14336,
        num_heads=32, parameter_count_b=8.0,
    ),
    "Llama-2-13B": ModelConfig(
        "Llama-2-13B", num_layers=40, hidden_dim=5120, ffn_dim=13824,
        num_heads=40, parameter_count_b=13.0,
    ),
    "Llama-2-70B": ModelConfig(
        "Llama-2-70B", num_layers=80, hidden_dim=8192, ffn_dim=28672,
        num_heads=64, parameter_count_b=70.0,
    ),
    "GPT3-175B": ModelConfig(
        "GPT3-175B", num_layers=96, hidden_dim=12288, ffn_dim=49152,
        num_heads=96, parameter_count_b=175.0,
    ),
}


SCENARIOS = [
    Scenario("interactive decode 1 token", batch_size=1, seq_len=1, category="decode"),
    Scenario("batched decode 64 tokens", batch_size=64, seq_len=1, category="decode"),
    Scenario("high-throughput decode 256 tokens", batch_size=256, seq_len=1, category="decode"),
    Scenario("short prefill 1x512", batch_size=1, seq_len=512, category="prefill"),
    Scenario("prefill 1x2K", batch_size=1, seq_len=2048, category="prefill"),
    Scenario("long prefill 1x8K", batch_size=1, seq_len=8192, category="prefill"),
    Scenario("batch prefill 8x2K", batch_size=8, seq_len=2048, category="prefill"),
]


def _all_chips() -> list[int]:
    return list(range(NUM_CHIPS))


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
    return str(value)


def activation_bytes(model: ModelConfig, scenario: Scenario) -> float:
    return (
        scenario.batch_size
        * scenario.seq_len
        * model.hidden_dim
        * model.bytes_per_value
    )


def tensor_parallel_forward_transfers(
    model: ModelConfig,
    scenario: Scenario,
) -> list[NetworkTransfer]:
    """
    Megatron-style tensor-parallel forward pass.

    Each transformer layer contributes two activation AllReduces:
      1. attention output projection
      2. FFN down projection

    This intentionally models realistic small-message decode cases as well as
    bandwidth-heavy prefill cases.
    """
    nbytes = activation_bytes(model, scenario)
    participants = _all_chips()
    transfers = []
    for layer_idx in range(model.num_layers):
        transfers.append(
            NetworkTransfer(
                tensor_name=f"layer{layer_idx}:attention_output_allreduce",
                data_bytes=nbytes,
                collective_type=CollectiveType.ALLREDUCE,
                participating_chips=participants,
            )
        )
        transfers.append(
            NetworkTransfer(
                tensor_name=f"layer{layer_idx}:ffn_output_allreduce",
                data_bytes=nbytes,
                collective_type=CollectiveType.ALLREDUCE,
                participating_chips=participants,
            )
        )
    return transfers


def data_parallel_gradient_transfers(model: ModelConfig) -> list[NetworkTransfer]:
    grad_bytes = model.parameter_count_b * 1e9 * model.bytes_per_value
    return [
        NetworkTransfer(
            tensor_name="gradient_allreduce",
            data_bytes=grad_bytes,
            collective_type=CollectiveType.ALLREDUCE,
            participating_chips=_all_chips(),
        )
    ]


def pipeline_parallel_transfers(
    model: ModelConfig,
    scenario: Scenario,
    num_stages: int = 8,
) -> list[NetworkTransfer]:
    nbytes = activation_bytes(model, scenario)
    chips_per_stage = NUM_CHIPS // num_stages
    transfers = []
    for stage in range(num_stages - 1):
        src = (stage + 1) * chips_per_stage - 1
        dst = (stage + 1) * chips_per_stage
        transfers.append(
            NetworkTransfer(
                tensor_name=f"pipeline_stage_{stage}_to_{stage + 1}",
                data_bytes=nbytes,
                collective_type=CollectiveType.POINT_TO_POINT,
                src_chip=src,
                dst_chips=[dst],
            )
        )
    return transfers


def build_workloads() -> list[WorkloadCase]:
    workloads = []
    selected = [
        ("Llama-3-8B", "interactive decode 1 token"),
        ("Llama-3-8B", "batched decode 64 tokens"),
        ("Llama-3-8B", "short prefill 1x512"),
        ("Llama-3-8B", "prefill 1x2K"),
        ("Llama-2-70B", "interactive decode 1 token"),
        ("Llama-2-70B", "high-throughput decode 256 tokens"),
        ("Llama-2-70B", "prefill 1x2K"),
        ("Llama-2-70B", "long prefill 1x8K"),
        ("GPT3-175B", "batched decode 64 tokens"),
        ("GPT3-175B", "long prefill 1x8K"),
        ("GPT3-175B", "batch prefill 8x2K"),
    ]
    scenarios_by_name = {scenario.name: scenario for scenario in SCENARIOS}

    for model_name, scenario_name in selected:
        model = MODELS[model_name]
        scenario = scenarios_by_name[scenario_name]
        workloads.append(
            WorkloadCase(
                name=f"{model.name} {scenario.name}",
                model=model,
                scenario=scenario,
                strategy="tensor_parallel_forward",
                transfers=tensor_parallel_forward_transfers(model, scenario),
            )
        )

    # Include two non-forward patterns so the output is not only AllReduce.
    for model_name in ("Llama-3-8B", "Llama-2-70B"):
        model = MODELS[model_name]
        scenario = scenarios_by_name["prefill 1x2K"]
        workloads.append(
            WorkloadCase(
                name=f"{model.name} data-parallel gradient sync",
                model=model,
                scenario=scenario,
                strategy="data_parallel_training",
                transfers=data_parallel_gradient_transfers(model),
            )
        )
        workloads.append(
            WorkloadCase(
                name=f"{model.name} pipeline activations 8 stages",
                model=model,
                scenario=scenario,
                strategy="pipeline_parallel_forward",
                transfers=pipeline_parallel_transfers(model, scenario),
            )
        )

    return workloads


def routing_strategy(topology: Topology) -> str:
    if isinstance(topology, CirculantHD):
        return "3 edge-disjoint Hamiltonian rings for full AllReduce; reverse-edge tree broadcast"
    if isinstance(topology, Torus3D):
        return "3D torus per-dimension ring collectives"
    if isinstance(topology, TorusND):
        return "N-dimensional torus per-dimension ring collectives"
    if isinstance(topology, Mesh3D):
        return "3D mesh per-dimension linear pipeline collectives"
    if isinstance(topology, Ring):
        return "single bidirectional ring collective"
    return "shortest-path routing"


def transfer_mix(transfers: list[NetworkTransfer]) -> dict:
    by_type = defaultdict(float)
    for transfer in transfers:
        by_type[transfer.collective_type.name] += transfer.data_bytes
    total = sum(by_type.values())
    return {
        "total_bytes": total,
        "by_collective_bytes": dict(sorted(by_type.items())),
        "by_collective_pct": {
            name: (100.0 * value / total if total > 0 else 0.0)
            for name, value in sorted(by_type.items())
        },
    }


def analyze_link_stress(topology: Topology, transfers: list[NetworkTransfer]) -> dict:
    from network_topology.cost_model import _get_transfer_link_loads

    merged_loads = defaultdict(float)
    for transfer in transfers:
        loads = _get_transfer_link_loads(topology, transfer)
        for link, nbytes in loads.items():
            merged_loads[link] += nbytes

    if not merged_loads:
        return {
            "max_link_load_bytes": 0.0,
            "avg_active_link_load_bytes": 0.0,
            "load_imbalance": 0.0,
            "num_active_links": 0,
            "total_links": topology.total_links,
            "link_utilization_pct": 0.0,
            "bottleneck_links": [],
            "bottleneck_latency_ms": 0.0,
        }

    loads_list = list(merged_loads.values())
    max_load = max(loads_list)
    avg_active = sum(loads_list) / len(loads_list)
    bottleneck_links = [
        {
            "link": f"{src}->{dst}",
            "bytes": nbytes,
            "relative_load": nbytes / max_load if max_load > 0 else 0.0,
        }
        for (src, dst), nbytes in sorted(
            merged_loads.items(), key=lambda item: item[1], reverse=True
        )[:10]
    ]
    return {
        "max_link_load_bytes": max_load,
        "avg_active_link_load_bytes": avg_active,
        "load_imbalance": max_load / avg_active if avg_active > 0 else 0.0,
        "num_active_links": len(merged_loads),
        "total_links": topology.total_links,
        "link_utilization_pct": (
            100.0 * len(merged_loads) / topology.total_links
            if topology.total_links > 0
            else 0.0
        ),
        "bottleneck_links": bottleneck_links,
        "bottleneck_latency_ms": max_load / topology.link_bandwidth * 1e3,
    }


def evaluate_workload(workload: WorkloadCase) -> dict:
    mix = transfer_mix(workload.transfers)
    topology_results = {}
    for topology_name, topology in TOPOLOGIES.items():
        cost = compute_network_cost(topology, workload.transfers)
        stress = analyze_link_stress(topology, workload.transfers)
        topology_results[topology_name] = {
            "routing_strategy": routing_strategy(topology),
            "latency_s": float(cost.total_latency),
            "energy_j": float(cost.total_energy),
            "total_network_bytes": float(cost.total_network_bytes),
            "energy_per_network_byte": float(cost.energy_per_network_access),
            "latency_per_network_byte": float(cost.latency_per_network_access),
            "stress": stress,
            "per_transfer_sample": cost.per_transfer[:4],
        }

    latencies = {name: result["latency_s"] for name, result in topology_results.items()}
    energies = {name: result["energy_j"] for name, result in topology_results.items()}
    min_latency = min(latencies.values())
    min_energy = min(energies.values())
    latency_tol = max(1e-15, min_latency * 1e-9)
    energy_tol = max(1e-15, min_energy * 1e-9)
    best_latency_topologies = [
        name for name, value in latencies.items() if value <= min_latency + latency_tol
    ]
    best_energy_topologies = [
        name for name, value in energies.items() if value <= min_energy + energy_tol
    ]
    torus_latency = latencies["Torus 4x4x4"]
    torus_energy = energies["Torus 4x4x4"]
    circ_latency = latencies["Circulant {1,5,17}"]
    circ_energy = energies["Circulant {1,5,17}"]

    return {
        "name": workload.name,
        "model": workload.model.name,
        "scenario": workload.scenario.name,
        "strategy": workload.strategy,
        "batch_size": workload.scenario.batch_size,
        "seq_len": workload.scenario.seq_len,
        "activation_bytes": activation_bytes(workload.model, workload.scenario),
        "num_transfers": len(workload.transfers),
        "transfer_mix": mix,
        "best_latency_topology": best_latency_topologies[0],
        "best_latency_topologies": best_latency_topologies,
        "best_energy_topology": best_energy_topologies[0],
        "best_energy_topologies": best_energy_topologies,
        "circulant_vs_torus_latency_speedup": (
            torus_latency / circ_latency if circ_latency > 0 else 0.0
        ),
        "circulant_vs_torus_energy_ratio": (
            circ_energy / torus_energy if torus_energy > 0 else 0.0
        ),
        "topologies": topology_results,
    }


def synthetic_message_sweep() -> list[dict]:
    sizes = [
        1 << 10,
        4 << 10,
        16 << 10,
        64 << 10,
        256 << 10,
        1 << 20,
        4 << 20,
        16 << 20,
        64 << 20,
        256 << 20,
        1 << 30,
    ]
    rows = []
    for nbytes in sizes:
        transfer = NetworkTransfer(
            tensor_name="single_allreduce",
            data_bytes=float(nbytes),
            collective_type=CollectiveType.ALLREDUCE,
            participating_chips=_all_chips(),
        )
        results = {
            name: compute_network_cost(topology, [transfer])
            for name, topology in TOPOLOGIES.items()
        }
        circ = results["Circulant {1,5,17}"]
        torus = results["Torus 4x4x4"]
        rows.append(
            {
                "message_bytes": float(nbytes),
                "circulant_latency_ms": circ.total_latency * 1e3,
                "torus_latency_ms": torus.total_latency * 1e3,
                "circulant_vs_torus_speedup": (
                    torus.total_latency / circ.total_latency
                    if circ.total_latency > 0
                    else 0.0
                ),
                "circulant_energy_mj": circ.total_energy * 1e3,
                "torus_energy_mj": torus.total_energy * 1e3,
            }
        )
    return rows


def _short_label(result: dict) -> str:
    label = result["name"]
    replacements = {
        "interactive decode 1 token": "decode\n1",
        "batched decode 64 tokens": "decode\n64",
        "high-throughput decode 256 tokens": "decode\n256",
        "short prefill 1x512": "prefill\n1x512",
        "prefill 1x2K": "prefill\n1x2K",
        "long prefill 1x8K": "prefill\n1x8K",
        "batch prefill 8x2K": "prefill\n8x2K",
        "data-parallel gradient sync": "grad\nsync",
        "pipeline activations 8 stages": "pipeline\n8-stage",
    }
    model = result["model"].replace("Llama-", "L").replace("GPT3-", "G")
    for old, new in replacements.items():
        if old in label:
            return f"{model}\n{new}"
    return label.replace(" ", "\n")


def _style_axis(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_figures(workload_results: list[dict], message_rows: list[dict], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    topology_colors = {
        "Circulant {1,5,17}": "#7b3294",
        "6D Hypercube": "#008837",
        "Torus 4x4x4": "#2166ac",
        "Mesh 4x4x4": "#d6604d",
        "Ring 64": "#bf812d",
    }
    labels = [_short_label(result) for result in workload_results]
    x = list(range(len(workload_results)))

    # 1. Absolute latency by realistic workload.
    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.15
    offsets = [-2 * width, -width, 0, width, 2 * width]
    for offset, topology_name in zip(offsets, TOPOLOGIES):
        values = [
            result["topologies"][topology_name]["latency_s"] * 1e3
            for result in workload_results
        ]
        ax.bar(
            [i + offset for i in x],
            values,
            width,
            label=topology_name,
            color=topology_colors[topology_name],
            zorder=3,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    _style_axis(ax, "Network Latency Across Realistic Workloads", "Latency (ms, log scale)")
    ax.legend(ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "realistic_latency_by_topology.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. Circulant speedup and energy ratio versus 3D torus.
    fig, ax1 = plt.subplots(figsize=(15, 5.8))
    speedups = [result["circulant_vs_torus_latency_speedup"] for result in workload_results]
    energy_ratios = [result["circulant_vs_torus_energy_ratio"] for result in workload_results]
    ax1.bar(x, speedups, color="#7b3294", alpha=0.85, zorder=3, label="Latency speedup")
    ax1.axhline(1.0, color="#333333", linewidth=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Circulant / Torus latency speedup")
    _style_axis(ax1, "Circulant Versus 3D Torus", None)
    ax2 = ax1.twinx()
    ax2.plot(x, energy_ratios, color="#008837", marker="o", linewidth=2.0, label="Energy ratio")
    ax2.set_ylabel("Circulant energy / Torus energy")
    ax2.set_ylim(0, max(1.0, max(energy_ratios) * 1.2))
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "circulant_vs_torus_realistic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3. Message-size sweep, showing the fixed-latency regime and bandwidth regime.
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    sizes = [row["message_bytes"] for row in message_rows]
    speedups = [row["circulant_vs_torus_speedup"] for row in message_rows]
    ax1.plot(sizes, speedups, color="#7b3294", marker="o", linewidth=2.0)
    ax1.set_xscale("log")
    ax1.set_ylim(1.0, max(speedups) * 1.1)
    _style_axis(ax1, "AllReduce Message Size Sweep", "Torus latency / Circulant latency")
    ax1.set_xlabel("Message bytes")
    fig.tight_layout()
    fig.savefig(fig_dir / "allreduce_message_size_speedup.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4. Bottleneck-link load reduction versus torus.
    fig, ax = plt.subplots(figsize=(15, 5.8))
    maxlink_ratios = []
    for result in workload_results:
        circ = result["topologies"]["Circulant {1,5,17}"]["stress"]["max_link_load_bytes"]
        torus = result["topologies"]["Torus 4x4x4"]["stress"]["max_link_load_bytes"]
        maxlink_ratios.append(torus / circ if circ > 0 else 0.0)
    ax.bar(x, maxlink_ratios, color="#542788", zorder=3)
    ax.axhline(1.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    _style_axis(ax, "Bottleneck Link Load Reduction", "Torus max-link load / Circulant max-link load")
    fig.tight_layout()
    fig.savefig(fig_dir / "bottleneck_link_reduction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_topology_summary() -> None:
    print("=" * 132)
    print("REALISTIC TRANSFORMER NETWORK COMPARISON")
    print("Direct model: topology-specific collective schedules, 64 TPU-v4-like chips")
    print("=" * 132)
    print(
        f"\n{'Topology':<22s} {'Diam':>5s} {'AvgHop':>7s} {'Degree':>7s} "
        f"{'Bisect':>10s} {'Routing schedule'}"
    )
    print("-" * 132)
    for name, topology in TOPOLOGIES.items():
        summary = topology.summary()
        print(
            f"{name:<22s} {summary['diameter']:>5d} {summary['avg_hops']:>7.2f} "
            f"{summary['min_degree']:>3d}-{summary['max_degree']:<3d} "
            f"{summary['bisection_bandwidth_TB_s']:>7.2f}TB/s  "
            f"{routing_strategy(topology)}"
        )


def print_workload_result(result: dict) -> None:
    mix = result["transfer_mix"]
    print(
        f"\n{result['name']} [{result['strategy']}]\n"
        f"  batch={result['batch_size']} seq={result['seq_len']} "
        f"activation={result['activation_bytes'] / 1e6:.3f} MB "
        f"transfers={result['num_transfers']} total={mix['total_bytes'] / 1e9:.3f} GB"
    )
    print(
        f"  {'Topology':<22s} {'Latency':>11s} {'Energy':>11s} "
        f"{'MaxLink':>11s} {'Active':>9s} {'Imbal':>7s}"
    )
    print("  " + "-" * 78)
    for topology_name in TOPOLOGIES:
        item = result["topologies"][topology_name]
        stress = item["stress"]
        print(
            f"  {topology_name:<22s} {item['latency_s'] * 1e3:>9.3f}ms "
            f"{item['energy_j'] * 1e3:>9.3f}mJ "
            f"{stress['max_link_load_bytes'] / 1e6:>9.3f}MB "
            f"{stress['num_active_links']:>4d}/{stress['total_links']:<4d} "
            f"{stress['load_imbalance']:>6.2f}x"
        )
    print(
        f"  -> best latency: {', '.join(result['best_latency_topologies'])}; "
        f"best energy: {', '.join(result['best_energy_topologies'])}; "
        f"circulant vs torus latency speedup: "
        f"{result['circulant_vs_torus_latency_speedup']:.2f}x; "
        f"circulant energy / torus energy: "
        f"{result['circulant_vs_torus_energy_ratio']:.2f}"
    )


def main() -> None:
    print_topology_summary()

    workload_results = []
    for workload in build_workloads():
        result = evaluate_workload(workload)
        print_workload_result(result)
        workload_results.append(result)

    message_rows = synthetic_message_sweep()

    print("\n" + "=" * 132)
    print("SINGLE ALLREDUCE MESSAGE-SIZE SWEEP")
    print("=" * 132)
    print(
        f"{'Bytes':>12s} {'Circ Lat':>12s} {'Torus Lat':>12s} "
        f"{'Speedup':>8s} {'Circ E':>12s} {'Torus E':>12s}"
    )
    print("-" * 76)
    for row in message_rows:
        print(
            f"{row['message_bytes']:>12.0f} "
            f"{row['circulant_latency_ms']:>10.4f}ms "
            f"{row['torus_latency_ms']:>10.4f}ms "
            f"{row['circulant_vs_torus_speedup']:>7.2f}x "
            f"{row['circulant_energy_mj']:>10.4f}mJ "
            f"{row['torus_energy_mj']:>10.4f}mJ"
        )

    summary = {
        "num_workloads": len(workload_results),
        "circulant_min_latency_count": sum(
            1
            for result in workload_results
            if "Circulant {1,5,17}" in result["best_latency_topologies"]
        ),
        "circulant_unique_min_latency_count": sum(
            1
            for result in workload_results
            if result["best_latency_topologies"] == ["Circulant {1,5,17}"]
        ),
        "circulant_min_energy_count": sum(
            1
            for result in workload_results
            if "Circulant {1,5,17}" in result["best_energy_topologies"]
        ),
        "circulant_unique_min_energy_count": sum(
            1
            for result in workload_results
            if result["best_energy_topologies"] == ["Circulant {1,5,17}"]
        ),
        "avg_circulant_vs_torus_latency_speedup": (
            sum(r["circulant_vs_torus_latency_speedup"] for r in workload_results)
            / len(workload_results)
        ),
        "avg_circulant_energy_over_torus": (
            sum(r["circulant_vs_torus_energy_ratio"] for r in workload_results)
            / len(workload_results)
        ),
    }
    print("\n" + "=" * 132)
    print("SUMMARY")
    print("=" * 132)
    print(
        f"Circulant minimum latency: {summary['circulant_min_latency_count']}/"
        f"{summary['num_workloads']} workloads "
        f"({summary['circulant_unique_min_latency_count']} unique)"
    )
    print(
        f"Circulant minimum energy:  {summary['circulant_min_energy_count']}/"
        f"{summary['num_workloads']} workloads "
        f"({summary['circulant_unique_min_energy_count']} unique)"
    )
    print(
        f"Average circulant vs torus latency speedup: "
        f"{summary['avg_circulant_vs_torus_latency_speedup']:.2f}x"
    )
    print(
        f"Average circulant energy / torus energy: "
        f"{summary['avg_circulant_energy_over_torus']:.2f}"
    )

    out_dir = Path("logs") / f"realistic_network_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "realistic_workload_results.json"
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "num_chips": NUM_CHIPS,
        "topologies": {
            name: {
                "summary": topology.summary(),
                "routing_strategy": routing_strategy(topology),
            }
            for name, topology in TOPOLOGIES.items()
        },
        "models": {name: model.__dict__ for name, model in MODELS.items()},
        "summary": summary,
        "message_size_sweep": message_rows,
        "results": workload_results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
    generate_figures(workload_results, message_rows, out_dir)
    print(f"\nSaved detailed results to {out_path}")
    print(f"Saved figures to {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
