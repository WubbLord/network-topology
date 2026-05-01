#!/usr/bin/env python3
"""
Synthetic MoE token-dispatch sweep.

The model treats a MoE layer as two phase-synchronous sparse all-to-all steps:
token dispatch from the token-owning chip to each selected expert chip, followed
by output combine from the expert chip back to the token owner. Expert placement
is varied to show how topology-aware placement changes congestion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from network_topology.cost_model import (
    CollectiveType,
    NetworkPhase,
    NetworkTransfer,
    compute_network_phased_cost,
)
from network_topology.topology import CirculantHD, Mesh3D, Ring, Topology, Torus3D, TorusND
from network_topology.tpu_v4 import (
    ICI_ENERGY_PER_BIT_PER_HOP,
    ICI_LINK_BW_UNIDIR,
    ICI_PER_HOP_LATENCY,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PLACEMENT_STRATEGIES = ("clustered", "spread", "topology_aware")

HW = {
    "link_bandwidth": ICI_LINK_BW_UNIDIR,
    "energy_per_bit_per_hop": ICI_ENERGY_PER_BIT_PER_HOP,
    "per_hop_latency": ICI_PER_HOP_LATENCY,
}

TOPOLOGIES = {
    "Torus 4x4x4": Torus3D(dims=(4, 4, 4), **HW),
    "Mesh 4x4x4": Mesh3D(dims=(4, 4, 4), **HW),
    "Torus 8x2x4": Torus3D(dims=(8, 2, 4), **HW),
    "Ring 64": Ring(num_chips=64, **HW),
    "4D Torus 4x4x2x2": TorusND(dims=(4, 4, 2, 2), **HW),
    "5D Torus 4x2x2x2x2": TorusND(dims=(4, 2, 2, 2, 2), **HW),
    "6D Hypercube": TorusND(dims=(2, 2, 2, 2, 2, 2), **HW),
    "Circulant {1,5,17}": CirculantHD(64, (1, 5, 17), **HW),
}

CORE_TOPOLOGIES = (
    "Torus 4x4x4",
    "Mesh 4x4x4",
    "Ring 64",
    "Circulant {1,5,17}",
)


@dataclass(frozen=True)
class MoeWorkload:
    name: str
    num_chips: int = 64
    num_experts: int = 16
    tokens_per_chip: int = 1024
    hidden_size: int = 12288
    bytes_per_value: int = 2
    top_k: int = 2
    local_fraction: float = 0.75
    output_scale: float = 1.0
    experiment: str = "baseline"

    @property
    def token_bytes(self) -> int:
        return self.hidden_size * self.bytes_per_value

    @property
    def logical_dispatch_bytes(self) -> float:
        return (
            self.num_chips
            * self.tokens_per_chip
            * self.top_k
            * self.token_bytes
        )


DEFAULT_WORKLOADS = (
    MoeWorkload(
        name="MoE local E16 top2",
        num_experts=16,
        tokens_per_chip=1024,
        hidden_size=12288,
        top_k=2,
        local_fraction=0.75,
    ),
    MoeWorkload(
        name="MoE local E64 top2",
        num_experts=64,
        tokens_per_chip=1024,
        hidden_size=12288,
        top_k=2,
        local_fraction=0.90,
    ),
    MoeWorkload(
        name="MoE uniform E16 top2",
        num_experts=16,
        tokens_per_chip=1024,
        hidden_size=12288,
        top_k=2,
        local_fraction=0.0,
    ),
    MoeWorkload(
        name="MoE heavy E8 top1",
        num_experts=8,
        tokens_per_chip=4096,
        hidden_size=4096,
        top_k=1,
        local_fraction=0.85,
    ),
)

BATCH_SCALING_TOKENS_PER_CHIP = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


def make_batch_scaling_workloads() -> tuple[MoeWorkload, ...]:
    return tuple(
        MoeWorkload(
            name=f"MoE batch T{tokens_per_chip} E16 top2",
            num_experts=16,
            tokens_per_chip=tokens_per_chip,
            hidden_size=12288,
            top_k=2,
            local_fraction=0.75,
            experiment="batch_scaling",
        )
        for tokens_per_chip in BATCH_SCALING_TOKENS_PER_CHIP
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return str(value)


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-")


def make_run_dir() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir = SCRIPT_DIR / "logs" / f"moe-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _validate_workload(workload: MoeWorkload) -> None:
    if workload.num_chips <= 0:
        raise ValueError("num_chips must be positive.")
    if workload.num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    if workload.tokens_per_chip <= 0:
        raise ValueError("tokens_per_chip must be positive.")
    if workload.hidden_size <= 0 or workload.bytes_per_value <= 0:
        raise ValueError("hidden_size and bytes_per_value must be positive.")
    if workload.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not 0.0 <= workload.local_fraction <= 1.0:
        raise ValueError("local_fraction must be in [0, 1].")


def _local_expert_for_source(workload: MoeWorkload, source_chip: int) -> int:
    expert = source_chip * workload.num_experts // workload.num_chips
    return min(workload.num_experts - 1, expert)


def expert_source_bytes(workload: MoeWorkload) -> dict[int, dict[int, float]]:
    """
    Return expected dispatch bytes by expert and source chip.

    The distribution is over expert slots, not necessarily unique top-k experts:
    local_fraction of each token's top-k slots goes to its source-region expert,
    while the remaining slots are spread uniformly across all experts.
    """
    _validate_workload(workload)
    bytes_by_expert: dict[int, dict[int, float]] = {
        expert: defaultdict(float) for expert in range(workload.num_experts)
    }
    token_slot_bytes = float(workload.tokens_per_chip * workload.token_bytes)
    uniform_slots = workload.top_k * (1.0 - workload.local_fraction)
    local_slots = workload.top_k * workload.local_fraction

    for source_chip in range(workload.num_chips):
        if uniform_slots > 0:
            uniform_bytes = token_slot_bytes * uniform_slots / workload.num_experts
            for expert in range(workload.num_experts):
                bytes_by_expert[expert][source_chip] += uniform_bytes

        if local_slots > 0:
            local_expert = _local_expert_for_source(workload, source_chip)
            bytes_by_expert[local_expert][source_chip] += token_slot_bytes * local_slots

    return {
        expert: dict(sorted(source_bytes.items()))
        for expert, source_bytes in bytes_by_expert.items()
    }


def placement_hop_objective(
    topology: Topology,
    expert_sources: dict[int, dict[int, float]],
    placement: dict[int, int],
) -> float:
    objective = 0.0
    for expert, source_bytes in expert_sources.items():
        expert_chip = placement[expert]
        for source_chip, data_bytes in source_bytes.items():
            objective += float(data_bytes) * float(topology.hop_count[source_chip, expert_chip])
    return objective


def _max_experts_per_chip(workload: MoeWorkload) -> int:
    return max(1, math.ceil(workload.num_experts / workload.num_chips))


def _respects_expert_capacity(workload: MoeWorkload, placement: dict[int, int]) -> bool:
    max_experts_per_chip = _max_experts_per_chip(workload)
    used_by_chip: dict[int, int] = defaultdict(int)
    for expert_chip in placement.values():
        used_by_chip[expert_chip] += 1
        if used_by_chip[expert_chip] > max_experts_per_chip:
            return False
    return True


def _hop_greedy_placement(
    workload: MoeWorkload,
    topology: Topology,
    expert_sources: dict[int, dict[int, float]],
) -> dict[int, int]:
    max_experts_per_chip = _max_experts_per_chip(workload)
    used_by_chip: dict[int, int] = defaultdict(int)
    placement: dict[int, int] = {}
    expert_order = sorted(
        range(workload.num_experts),
        key=lambda expert: sum(expert_sources[expert].values()),
        reverse=True,
    )

    for expert in expert_order:
        candidates = [
            chip for chip in range(topology.num_chips)
            if used_by_chip[chip] < max_experts_per_chip
        ]
        if not candidates:
            candidates = list(range(topology.num_chips))

        def candidate_key(chip: int) -> tuple[float, int, int]:
            weighted_hops = sum(
                float(data_bytes) * float(topology.hop_count[source_chip, chip])
                for source_chip, data_bytes in expert_sources[expert].items()
            )
            return weighted_hops, used_by_chip[chip], chip

        expert_chip = min(candidates, key=candidate_key)
        placement[expert] = expert_chip
        used_by_chip[expert_chip] += 1

    return dict(sorted(placement.items()))


def _placement_latency(
    workload: MoeWorkload,
    topology: Topology,
    expert_sources: dict[int, dict[int, float]],
    placement: dict[int, int],
) -> float:
    phases, _traffic = build_moe_phases(workload, expert_sources, placement)
    return float(compute_network_phased_cost(topology, phases).total_latency)


def _latency_local_search_placement(
    workload: MoeWorkload,
    topology: Topology,
    expert_sources: dict[int, dict[int, float]],
    initial_placement: dict[int, int],
    max_passes: int = 2,
) -> dict[int, int]:
    current = dict(initial_placement)
    current_latency = _placement_latency(workload, topology, expert_sources, current)
    expert_order = sorted(
        range(workload.num_experts),
        key=lambda expert: sum(expert_sources[expert].values()),
        reverse=True,
    )

    for _pass_idx in range(max_passes):
        improved = False
        for expert in expert_order:
            best_chip = current[expert]
            best_latency = current_latency

            for candidate_chip in range(topology.num_chips):
                if candidate_chip == current[expert]:
                    continue
                trial = dict(current)
                trial[expert] = candidate_chip
                if not _respects_expert_capacity(workload, trial):
                    continue

                trial_latency = _placement_latency(
                    workload, topology, expert_sources, trial
                )
                if trial_latency < best_latency:
                    best_latency = trial_latency
                    best_chip = candidate_chip

            if best_chip != current[expert]:
                current[expert] = best_chip
                current_latency = best_latency
                improved = True

        if not improved:
            break

    return dict(sorted(current.items()))


def place_experts(
    workload: MoeWorkload,
    topology: Topology,
    strategy: str,
    expert_sources: dict[int, dict[int, float]] | None = None,
) -> dict[int, int]:
    if topology.num_chips != workload.num_chips:
        raise ValueError(
            f"Workload uses {workload.num_chips} chips but topology has {topology.num_chips}."
        )
    if strategy not in PLACEMENT_STRATEGIES:
        raise ValueError(f"Unknown expert placement strategy: {strategy}")

    if strategy == "clustered":
        return {expert: expert % topology.num_chips for expert in range(workload.num_experts)}

    if strategy == "spread":
        if workload.num_experts == 1:
            return {0: 0}
        return {
            expert: round(expert * (topology.num_chips - 1) / (workload.num_experts - 1))
            for expert in range(workload.num_experts)
        }

    if expert_sources is None:
        expert_sources = expert_source_bytes(workload)

    seed_placements = [
        place_experts(workload, topology, "clustered", expert_sources),
        place_experts(workload, topology, "spread", expert_sources),
        _hop_greedy_placement(workload, topology, expert_sources),
    ]
    best_seed = min(
        seed_placements,
        key=lambda placement: _placement_latency(
            workload, topology, expert_sources, placement
        ),
    )
    return _latency_local_search_placement(
        workload, topology, expert_sources, best_seed
    )


def build_moe_phases(
    workload: MoeWorkload,
    expert_sources: dict[int, dict[int, float]],
    placement: dict[int, int],
) -> tuple[list[NetworkPhase], dict[str, float]]:
    dispatch_flows: dict[tuple[int, int], float] = defaultdict(float)
    combine_flows: dict[tuple[int, int], float] = defaultdict(float)
    local_dispatch_bytes = 0.0
    logical_dispatch_bytes = 0.0

    for expert, source_bytes in expert_sources.items():
        expert_chip = placement[expert]
        for source_chip, data_bytes in source_bytes.items():
            data_bytes = float(data_bytes)
            logical_dispatch_bytes += data_bytes
            if data_bytes <= 0:
                continue
            if source_chip == expert_chip:
                local_dispatch_bytes += data_bytes
                continue
            dispatch_flows[(source_chip, expert_chip)] += data_bytes
            combine_flows[(expert_chip, source_chip)] += data_bytes * workload.output_scale

    dispatch_transfers = [
        NetworkTransfer(
            tensor_name=f"dispatch:{src}->{dst}",
            data_bytes=data_bytes,
            collective_type=CollectiveType.POINT_TO_POINT,
            src_chip=src,
            dst_chips=[dst],
        )
        for (src, dst), data_bytes in sorted(dispatch_flows.items())
    ]
    combine_transfers = [
        NetworkTransfer(
            tensor_name=f"combine:{src}->{dst}",
            data_bytes=data_bytes,
            collective_type=CollectiveType.POINT_TO_POINT,
            src_chip=src,
            dst_chips=[dst],
        )
        for (src, dst), data_bytes in sorted(combine_flows.items())
    ]
    phases = [
        NetworkPhase("dispatch", dispatch_transfers),
        NetworkPhase("combine", combine_transfers),
    ]
    remote_dispatch_bytes = float(sum(dispatch_flows.values()))
    remote_combine_bytes = float(sum(combine_flows.values()))
    return phases, {
        "logical_dispatch_bytes": float(logical_dispatch_bytes),
        "local_dispatch_bytes": float(local_dispatch_bytes),
        "remote_dispatch_bytes": remote_dispatch_bytes,
        "remote_combine_bytes": remote_combine_bytes,
        "remote_dispatch_fraction": (
            remote_dispatch_bytes / logical_dispatch_bytes
            if logical_dispatch_bytes > 0
            else 0.0
        ),
    }


def evaluate_moe_workload(
    workload: MoeWorkload,
    topology_name: str,
    topology: Topology,
    placement_strategy: str,
) -> dict:
    expert_sources = expert_source_bytes(workload)
    placement = place_experts(workload, topology, placement_strategy, expert_sources)
    phases, traffic_summary = build_moe_phases(workload, expert_sources, placement)
    network_result = compute_network_phased_cost(topology, phases)

    expert_load_bytes = {
        expert: float(sum(source_bytes.values()))
        for expert, source_bytes in expert_sources.items()
    }
    busiest_phase = max(network_result.per_phase, key=lambda phase: phase["latency"])
    remote_payload_bytes = (
        traffic_summary["remote_dispatch_bytes"] + traffic_summary["remote_combine_bytes"]
    )
    logical_payload_bytes = traffic_summary["logical_dispatch_bytes"] * (
        1.0 + workload.output_scale
    )
    payload_throughput_bytes_per_s = (
        remote_payload_bytes / network_result.total_latency
        if network_result.total_latency > 0
        else 0.0
    )
    logical_throughput_bytes_per_s = (
        logical_payload_bytes / network_result.total_latency
        if network_result.total_latency > 0
        else 0.0
    )
    token_throughput_per_s = (
        workload.num_chips * workload.tokens_per_chip / network_result.total_latency
        if network_result.total_latency > 0
        else 0.0
    )
    aggregate_directed_link_bandwidth = topology.total_links * topology.link_bandwidth
    payload_link_efficiency = (
        payload_throughput_bytes_per_s / aggregate_directed_link_bandwidth
        if aggregate_directed_link_bandwidth > 0
        else 0.0
    )

    return {
        "workload": asdict(workload),
        "topology": topology_name,
        "topology_summary": topology.summary(),
        "placement_strategy": placement_strategy,
        "expert_placement": placement,
        "expert_load_bytes": expert_load_bytes,
        "placement_hop_objective": placement_hop_objective(
            topology, expert_sources, placement
        ),
        "traffic_summary": traffic_summary,
        "total_energy": float(network_result.total_energy),
        "total_latency": float(network_result.total_latency),
        "total_network_bytes": float(network_result.total_network_bytes),
        "energy_per_network_access": float(network_result.energy_per_network_access),
        "latency_per_network_access": float(network_result.latency_per_network_access),
        "remote_payload_bytes": float(remote_payload_bytes),
        "logical_payload_bytes": float(logical_payload_bytes),
        "payload_throughput_bytes_per_s": float(payload_throughput_bytes_per_s),
        "logical_throughput_bytes_per_s": float(logical_throughput_bytes_per_s),
        "token_throughput_per_s": float(token_throughput_per_s),
        "aggregate_directed_link_bandwidth": float(aggregate_directed_link_bandwidth),
        "payload_link_efficiency": float(payload_link_efficiency),
        "per_phase": network_result.per_phase,
        "busiest_phase": busiest_phase["phase"],
        "busiest_phase_latency": float(busiest_phase["latency"]),
    }


def _evaluate_task(task: tuple[MoeWorkload, str, str]) -> dict:
    workload, topology_name, placement_strategy = task
    return evaluate_moe_workload(
        workload,
        topology_name,
        TOPOLOGIES[topology_name],
        placement_strategy,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic MoE topology sweep.")
    parser.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Run only workloads whose names contain this substring.",
    )
    parser.add_argument(
        "--topology",
        action="append",
        default=None,
        help="Run only this topology. May be passed multiple times.",
    )
    parser.add_argument(
        "--placement",
        action="append",
        choices=PLACEMENT_STRATEGIES,
        default=None,
        help="Run only this placement strategy. May be passed multiple times.",
    )
    parser.add_argument(
        "--all-topologies",
        action="store_true",
        help="Include every topology instead of the core four.",
    )
    parser.add_argument(
        "--experiment",
        choices=("baseline", "batch_scaling", "all"),
        default="baseline",
        help="Select the baseline MoE cases, batch-scaling cases, or both.",
    )
    parser.add_argument(
        "--batch-sweep",
        action="store_true",
        help="Shortcut for --experiment batch_scaling.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help="Parallel worker processes. Defaults to SLURM_CPUS_PER_TASK or 1.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to logs/moe-<timestamp>.",
    )
    return parser.parse_args()


def _select_workloads(workload_filter: str | None, experiment: str) -> list[MoeWorkload]:
    workloads = []
    if experiment in ("baseline", "all"):
        workloads.extend(DEFAULT_WORKLOADS)
    if experiment in ("batch_scaling", "all"):
        workloads.extend(make_batch_scaling_workloads())

    if workload_filter:
        workloads = [
            workload for workload in workloads
            if workload_filter.lower() in workload.name.lower()
        ]
        if not workloads:
            raise SystemExit(
                f"No MoE workload matching {workload_filter!r}. "
                f"Available: {', '.join(workload.name for workload in DEFAULT_WORKLOADS)}"
            )
    return workloads


def _select_topologies(topology_filters: list[str] | None, all_topologies: bool):
    if all_topologies:
        selected = dict(TOPOLOGIES)
    else:
        selected = {name: TOPOLOGIES[name] for name in CORE_TOPOLOGIES}

    if topology_filters:
        explicit = {}
        for requested in topology_filters:
            if requested in TOPOLOGIES:
                explicit[requested] = TOPOLOGIES[requested]
                continue
            matches = [
                name for name in TOPOLOGIES
                if requested.lower() in name.lower()
            ]
            if len(matches) != 1:
                raise SystemExit(
                    f"Topology filter {requested!r} matched {matches or 'nothing'}. "
                    f"Use one of: {', '.join(TOPOLOGIES)}"
                )
            explicit[matches[0]] = TOPOLOGIES[matches[0]]
        selected = explicit
    return selected


def main() -> None:
    args = parse_args()
    experiment = "batch_scaling" if args.batch_sweep else args.experiment
    workloads = _select_workloads(args.workload, experiment)
    topologies = _select_topologies(args.topology, args.all_topologies)
    placements = args.placement or list(PLACEMENT_STRATEGIES)
    jobs = max(1, args.jobs)
    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print(
        f"MoE Sweep: {len(workloads)} workloads x {len(topologies)} topologies "
        f"x {len(placements)} placements, experiment={experiment}, jobs={jobs}"
    )
    print("=" * 110)
    print(
        f"{'Workload':>22s} {'Topology':>20s} {'Placement':>15s} "
        f"{'Bytes':>10s} {'Remote':>8s} {'Latency':>12s} {'Eff':>8s}"
    )
    print("-" * 110)

    results = []
    t0 = time.time()
    tasks = [
        (workload, topology_name, placement)
        for workload in workloads
        for topology_name in topologies
        for placement in placements
    ]

    if jobs == 1:
        result_iter = map(_evaluate_task, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        result_iter = executor.map(_evaluate_task, tasks, chunksize=1)

    try:
        for result in result_iter:
            results.append(result)
            traffic = result["traffic_summary"]
            print(
                f"{result['workload']['name']:>22s} {result['topology']:>20s} "
                f"{result['placement_strategy']:>15s} "
                f"{result['total_network_bytes']:>10.2e} "
                f"{100.0 * traffic['remote_dispatch_fraction']:>7.1f}% "
                f"{result['total_latency'] * 1e3:>10.2f}ms "
                f"{100.0 * result['payload_link_efficiency']:>7.3f}%"
            )
    finally:
        if jobs != 1:
            executor.shutdown()

    best_by_workload_topology = {}
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["workload"]["name"], result["topology"])].append(result)

    print("\nBest placement by workload/topology:")
    for key, entries in sorted(grouped.items()):
        best = min(entries, key=lambda item: item["total_latency"])
        worst = max(entries, key=lambda item: item["total_latency"])
        speedup = (
            worst["total_latency"] / best["total_latency"]
            if best["total_latency"] > 0
            else 0.0
        )
        best_by_workload_topology["::".join(key)] = {
            "best_placement": best["placement_strategy"],
            "best_latency": best["total_latency"],
            "worst_placement": worst["placement_strategy"],
            "worst_latency": worst["total_latency"],
            "placement_speedup": speedup,
        }
        print(
            f"  {key[0]} on {key[1]}: {best['placement_strategy']} "
            f"({best['total_latency'] * 1e3:.2f}ms, {speedup:.2f}x vs worst)"
        )

    payload = {
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "elapsed_s": float(time.time() - t0),
        "experiment": experiment,
        "jobs": jobs,
        "chip_model": {
            "num_chips": 64,
            "note": (
                "Synthetic MoE evaluates explicit 64-chip topologies directly. "
                "It does not use the AccelForge MAP_CHIPS=8 scaling shortcut."
            ),
        },
        "methodology": {
            "model": "Synthetic MoE token dispatch/combine",
            "traffic": (
                "Each token contributes top_k expert slots. Dispatch sends the "
                "activation to the expert chip, and combine returns the expert "
                "output to the token owner."
            ),
            "placement_strategies": {
                "clustered": "Experts are placed on low-numbered chips.",
                "spread": "Experts are evenly spread over the chip id range.",
                "topology_aware": (
                    "Starts from clustered, spread, and hop-greedy seeds, then "
                    "runs a small exact-latency local search under a simple "
                    "per-chip expert capacity."
                ),
            },
            "phase_costing": (
                "Point-to-point transfers within dispatch or combine are merged "
                "as concurrent link loads; dispatch and combine phase latencies "
                "are summed."
            ),
        },
        "topologies": {name: topology.summary() for name, topology in topologies.items()},
        "placements": placements,
        "results": results,
        "best_by_workload_topology": best_by_workload_topology,
    }
    output_path = run_dir / "moe_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
    latest_path = run_dir / f"{_safe_path_part('moe_results')}.json"
    if latest_path != output_path:
        latest_path.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nSaved MoE results to {output_path}")


if __name__ == "__main__":
    main()
