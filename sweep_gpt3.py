#!/usr/bin/env python3
"""
Topology sweep: evaluate workloads on multiple topologies with congestion modeling.

Maps on 2 chips (fast), scales data movement to 64 chips, evaluates on 5 topologies.

Usage:
    /Users/nikhil/Documents/StudioProjects/accelforge/.venv/bin/python sweep_gpt3.py
"""

import sys, os, math, contextlib, logging, time
from pathlib import Path

sys.path.insert(0, ".")
logging.disable(logging.WARNING)

import accelforge as af
from accelforge.mapper.FFM import map_workload_to_arch
from network_topology import compute_network_cost, make_tpu_v4_topology
from network_topology.cost_model import NetworkTransfer, CollectiveType
from network_topology.topology import Torus3D, Mesh3D, Ring
from network_topology.tpu_v4 import ICI_LINK_BW_UNIDIR, ICI_ENERGY_PER_BIT_PER_HOP, ICI_PER_HOP_LATENCY

ACCELFORGE = Path("/Users/nikhil/Documents/StudioProjects/accelforge")
ARCH = "accelforge_configs/tpu_v4_distributed_1d.yaml"
MAP_CHIPS = 2
EVAL_CHIPS = 64
SCALE = EVAL_CHIPS / MAP_CHIPS

hw = dict(link_bandwidth=ICI_LINK_BW_UNIDIR, energy_per_bit_per_hop=ICI_ENERGY_PER_BIT_PER_HOP,
          per_hop_latency=ICI_PER_HOP_LATENCY)

TOPOLOGIES = {
    "Torus 4x4x4": Torus3D(dims=(4,4,4), **hw),
    "Mesh 4x4x4":  Mesh3D(dims=(4,4,4), **hw),
    "Torus 8x2x4": Torus3D(dims=(8,2,4), **hw),
    "Torus 16x2x2":Torus3D(dims=(16,2,2), **hw),
    "Ring 64":      Ring(num_chips=64, **hw),
}

WL = ACCELFORGE / "examples/workloads"
WORKLOADS = [
    ("Tiny 256x256",           WL/"matmuls.yaml", {"N_EINSUMS":1, "M":256,   "KN":256}),
    ("Small 1Kx1K",            WL/"matmuls.yaml", {"N_EINSUMS":1, "M":1024,  "KN":1024}),
    ("Medium 4Kx4K",           WL/"matmuls.yaml", {"N_EINSUMS":1, "M":4096,  "KN":4096}),
    ("Wide 4Kx16K (FFN-like)", WL/"matmuls.yaml", {"N_EINSUMS":1, "M":4096,  "KN":16384}),
    ("Tall 16Kx4K",            WL/"matmuls.yaml", {"N_EINSUMS":1, "M":16384, "KN":4096}),
    ("2-layer chain 4Kx4K",    WL/"matmuls.yaml", {"N_EINSUMS":2, "M":4096,  "KN":4096}),
    ("3-layer chain 4Kx4K",    WL/"matmuls.yaml", {"N_EINSUMS":3, "M":4096,  "KN":4096}),
    ("Attn-like 128x128",      WL/"matmuls.yaml", {"N_EINSUMS":1, "M":128,   "KN":128}),
    ("Attn-like 512x512",      WL/"matmuls.yaml", {"N_EINSUMS":1, "M":512,   "KN":512}),
    ("Attn-like 2Kx2K",        WL/"matmuls.yaml", {"N_EINSUMS":1, "M":2048,  "KN":2048}),
    ("Decode 1x4096",          WL/"matmuls.yaml", {"N_EINSUMS":1, "M":1,     "KN":4096}),
    ("Decode 1x16384",         WL/"matmuls.yaml", {"N_EINSUMS":1, "M":1,     "KN":16384}),
    ("Batch 64tok x 4K",       WL/"matmuls.yaml", {"N_EINSUMS":1, "M":64,    "KN":4096}),
    ("Batch 1024tok x 4K",     WL/"matmuls.yaml", {"N_EINSUMS":1, "M":1024,  "KN":4096}),
]


def map_and_extract(path, params):
    spec = af.Spec.from_yaml(ARCH, str(path), jinja_parse_data={"NUM_CHIPS": MAP_CHIPS, **params})
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        mappings = map_workload_to_arch(spec)
    energy_bk = mappings.energy(per_component=True)
    compute_e = float(sum(v for k,v in energy_bk.items() if k != "NetworkMemory")) * SCALE
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


# --- Main ---
print("=" * 120)
print(f"Topology Sweep: {len(WORKLOADS)} workloads x {len(TOPOLOGIES)} topologies ({EVAL_CHIPS} chips, congestion-aware)")
print("=" * 120)

print(f"\n{'Topology':<18s} {'Diam':>5s} {'AvgHop':>7s} {'Degree':>7s} {'BisectBW':>10s}")
print("-" * 50)
for name, topo in TOPOLOGIES.items():
    s = topo.summary()
    print(f"{name:<18s} {s['diameter']:>5d} {s['avg_hops']:>7.2f} {s['min_degree']:>3d}-{s['max_degree']:<3d} "
          f"{s['bisection_bandwidth_TB_s']:>8.2f} TB/s")

results = []
for desc, path, params in WORKLOADS:
    t0 = time.time()
    print(f"\nMapping {desc}...", end=" ", flush=True)
    try:
        ce, cl, transfers = map_and_extract(path, params)
    except Exception as e:
        print(f"FAILED: {e}")
        continue
    print(f"done ({time.time()-t0:.0f}s)")

    total_bytes = sum(t.data_bytes for t in transfers)
    bcast_pct = sum(t.data_bytes for t in transfers if t.collective_type == CollectiveType.BROADCAST) / total_bytes * 100 if total_bytes > 0 else 0

    lats = {}
    for tn, topo in TOPOLOGIES.items():
        r = compute_network_cost(topo, transfers)
        lats[tn] = r.total_latency
    results.append({"desc": desc, "compute_e": ce, "compute_l": cl, "bytes": total_bytes,
                     "bcast_pct": bcast_pct, "lats": lats})

# --- Results table ---
print(f"\n\n{'='*120}")
print("RESULTS (latency in ms, with link congestion)")
print(f"{'='*120}")
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
        print(f" {r['lats'][tn]*1e3:>12.1f}ms", end="")
    print(f" {best.split()[0]:>10s} {worst.split()[0]:>10s} {speedup:>7.2f}x")

# --- Insights ---
print(f"\n{'='*120}")
print("INSIGHTS")
print(f"{'='*120}")
torus_wins = sum(1 for r in results if min(r["lats"], key=r["lats"].get) == "Torus 4x4x4")
print(f"Torus 4x4x4 wins: {torus_wins}/{len(results)} workloads")

bcast_heavy = [r for r in results if r["bcast_pct"] > 80]
if bcast_heavy:
    avg_speedup = sum(r["lats"]["Mesh 4x4x4"] / r["lats"]["Torus 4x4x4"] for r in bcast_heavy) / len(bcast_heavy)
    print(f"Broadcast-heavy (>80%): Torus avg {avg_speedup:.2f}x faster than Mesh")

ar_heavy = [r for r in results if r["bcast_pct"] < 60]
if ar_heavy:
    avg_speedup = sum(r["lats"]["Mesh 4x4x4"] / r["lats"]["Torus 4x4x4"] for r in ar_heavy) / len(ar_heavy)
    print(f"AllReduce-heavy (<60% bcast): Torus avg {avg_speedup:.2f}x faster than Mesh")

print("\nDone!")
