#!/usr/bin/env python3
"""
Topology sweep: evaluate workloads on multiple topologies with congestion modeling.

Maps on 2 chips (fast), scales data movement to 64 chips, evaluates on 5 topologies.

Usage:
    ACCELFORGE_ROOT=/path/to/accelforge .venv/bin/python sweep_gpt3.py
"""

import contextlib
import json
import logging
import os
import sys
import time
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

hw = dict(link_bandwidth=ICI_LINK_BW_UNIDIR, energy_per_bit_per_hop=ICI_ENERGY_PER_BIT_PER_HOP,
          per_hop_latency=ICI_PER_HOP_LATENCY)

TOPOLOGIES = {
    "Torus 4x4x4": Torus3D(dims=(4, 4, 4), **hw),
    "Mesh 4x4x4": Mesh3D(dims=(4, 4, 4), **hw),
    "Torus 8x2x4": Torus3D(dims=(8, 2, 4), **hw),
    "Torus 16x2x2": Torus3D(dims=(16, 2, 2), **hw),
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
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir = SCRIPT_DIR / "logs" / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
    return out_path

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
        ("Tiny 256x256", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 256, "KN": 256}),
        ("Small 1Kx1K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 1024, "KN": 1024}),
        ("Medium 4Kx4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 4096, "KN": 4096}),
        ("Wide 4Kx16K (FFN-like)", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 4096, "KN": 16384}),
        ("Tall 16Kx4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 16384, "KN": 4096}),
        ("2-layer chain 4Kx4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 2, "M": 4096, "KN": 4096}),
        ("3-layer chain 4Kx4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 3, "M": 4096, "KN": 4096}),
        ("Attn-like 128x128", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 128, "KN": 128}),
        ("Attn-like 512x512", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 512, "KN": 512}),
        ("Attn-like 2Kx2K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 2048, "KN": 2048}),
        ("Decode 1x4096", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 1, "KN": 4096}),
        ("Decode 1x16384", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 1, "KN": 16384}),
        ("Batch 64tok x 4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 64, "KN": 4096}),
        ("Batch 1024tok x 4K", workloads_dir / "matmuls.yaml", {"N_EINSUMS": 1, "M": 1024, "KN": 4096}),
    ]


def map_and_extract(path, params, af, map_workload_to_arch):
    spec = af.Spec.from_yaml(str(ARCH), str(path), jinja_parse_data={"NUM_CHIPS": MAP_CHIPS, **params})
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        mappings = map_workload_to_arch(spec)
    energy_bk = mappings.energy(per_component=True)
    compute_e = float(sum(v for k, v in energy_bk.items() if k != "NetworkMemory")) * SCALE
    compute_l = float(mappings.latency())
    actions = mappings.actions(per_component=True, per_tensor=True)
    transfers = []
    for (comp, tensor, action), count in actions.items():
        if comp != "NetworkMemory" or count <= 0:
            continue
        data_bytes = float(count) * SCALE
        ct = CollectiveType.BROADCAST if action == "read" else CollectiveType.ALLREDUCE
        transfers.append(NetworkTransfer(tensor, data_bytes, ct, src_chip=0 if action == "read" else None))
    return compute_e, compute_l, transfers


def main():
    af, map_workload_to_arch = load_accelforge(ACCELFORGE)
    workloads_dir = ACCELFORGE.resolve() / "examples" / "workloads"
    if not workloads_dir.exists():
        raise SystemExit(
            f"Expected workload specs under {workloads_dir}, but the directory does not exist."
        )
    workloads = make_workloads(workloads_dir)

    print("=" * 120)
    print(f"Topology Sweep: {len(workloads)} workloads x {len(TOPOLOGIES)} topologies ({EVAL_CHIPS} chips, congestion-aware)")
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
        t0 = time.time()
        print(f"\nMapping {desc}...", end=" ", flush=True)
        try:
            ce, cl, transfers = map_and_extract(path, params, af, map_workload_to_arch)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        elapsed = time.time() - t0
        print(f"done ({elapsed:.0f}s)")

        total_bytes = sum(t.data_bytes for t in transfers)
        bcast_pct = (
            sum(t.data_bytes for t in transfers if t.collective_type == CollectiveType.BROADCAST)
            / total_bytes * 100
            if total_bytes > 0 else 0
        )

        lats = {}
        topology_results = {}
        for tn, topo in TOPOLOGIES.items():
            r = compute_network_cost(topo, transfers)
            lats[tn] = r.total_latency
            topology_results[tn] = {
                "topology_summary": topology_summaries[tn],
                "total_energy": float(r.total_energy),
                "total_latency": float(r.total_latency),
                "energy_per_network_access": float(r.energy_per_network_access),
                "latency_per_network_access": float(r.latency_per_network_access),
                "total_network_bytes": float(r.total_network_bytes),
                "per_transfer": _json_ready(r.per_transfer),
            }
        results.append({
            "desc": desc,
            "workload_path": str(path),
            "params": params,
            "compute_e": ce,
            "compute_l": cl,
            "mapping_wall_time_s": elapsed,
            "bytes": total_bytes,
            "bcast_pct": bcast_pct,
            "lats": lats,
            "transfers": [_serialize_transfer(t) for t in transfers],
            "topologies": topology_results,
        })

    print(f"\n\n{'=' * 120}")
    print("RESULTS (latency in ms, with link congestion)")
    print(f"{'=' * 120}")
    print(f"{'Workload':>28s} {'Bytes':>10s} {'Bcast%':>7s}", end="")
    for tn in TOPOLOGIES:
        print(f" {tn:>14s}", end="")
    print(f" {'Best':>10s} {'Worst':>10s} {'Speedup':>8s}")
    print("-" * 130)

    for r in results:
        best = min(r["lats"], key=r["lats"].get)
        worst = max(r["lats"], key=r["lats"].get)
        speedup = r["lats"][worst] / r["lats"][best] if r["lats"][best] > 0 else 0
        print(f"{r['desc']:>28s} {r['bytes']:>10.2e} {r['bcast_pct']:>6.0f}%", end="")
        for tn in TOPOLOGIES:
            print(f" {r['lats'][tn] * 1e3:>12.1f}ms", end="")
        print(f" {best.split()[0]:>10s} {worst.split()[0]:>10s} {speedup:>7.2f}x")

    print(f"\n{'=' * 120}")
    print("INSIGHTS")
    print(f"{'=' * 120}")
    torus_wins = sum(1 for r in results if min(r["lats"], key=r["lats"].get) == "Torus 4x4x4")
    print(f"Torus 4x4x4 wins: {torus_wins}/{len(results)} workloads")
    insights = {"torus_4x4x4_wins": torus_wins, "num_workloads": len(results)}

    bcast_heavy = [r for r in results if r["bcast_pct"] > 80]
    if bcast_heavy:
        avg_speedup = sum(r["lats"]["Mesh 4x4x4"] / r["lats"]["Torus 4x4x4"] for r in bcast_heavy) / len(bcast_heavy)
        print(f"Broadcast-heavy (>80%): Torus avg {avg_speedup:.2f}x faster than Mesh")
        insights["broadcast_heavy_avg_mesh_over_torus_speedup"] = avg_speedup

    ar_heavy = [r for r in results if r["bcast_pct"] < 60]
    if ar_heavy:
        avg_speedup = sum(r["lats"]["Mesh 4x4x4"] / r["lats"]["Torus 4x4x4"] for r in ar_heavy) / len(ar_heavy)
        print(f"AllReduce-heavy (<60% bcast): Torus avg {avg_speedup:.2f}x faster than Mesh")
        insights["allreduce_heavy_avg_mesh_over_torus_speedup"] = avg_speedup

    results_path = save_results({
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "accelforge_root": str(ACCELFORGE.resolve()),
        "workloads_dir": str(workloads_dir),
        "architecture_yaml": str(ARCH),
        "map_chips": MAP_CHIPS,
        "eval_chips": EVAL_CHIPS,
        "scale": SCALE,
        "topology_summaries": topology_summaries,
        "results": results,
        "insights": insights,
    })

    print(f"\nSaved results to {results_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
