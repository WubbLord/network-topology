#!/usr/bin/env python3
"""
Exhaustive search for optimal 6-regular topologies on 64 chips.

Physical constraint: each chip has exactly 6 bidirectional ICI links (TPU v4).
Goal: find the graph that minimizes AllReduce latency for a given data size.

Approach:
  Phase 1: Exhaustive search over Hamiltonian-decomposable circulant graphs
           C(64, {g1, g2, g3}) where each generator is odd and coprime to 64.
           These admit a provably optimal 3-parallel-ring AllReduce.
           ~560 candidates, evaluates in seconds.

  Phase 2: Random 6-regular graph search + simulated annealing.
           Searches beyond circulant graphs for potentially better structures.

  Phase 3: Known graph families (Petersen, Cayley, etc.)

Usage:
    python search_optimal_topology.py
    python search_optimal_topology.py --data-bytes 1e11
"""

import argparse
import itertools
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from network_topology.topology import (
    Topology, Torus3D, Mesh3D, Ring, TorusND, CirculantHD, Custom,
    _make_circulant_adj,
)
from network_topology.tpu_v4 import ICI_LINK_BW_UNIDIR, ICI_ENERGY_PER_BIT_PER_HOP, ICI_PER_HOP_LATENCY

N = 64          # number of chips
DEGREE = 6      # links per chip
NUM_GENS = 3    # circulant generators (3 generators x 2 directions = 6 links)

hw = dict(
    link_bandwidth=ICI_LINK_BW_UNIDIR,
    energy_per_bit_per_hop=ICI_ENERGY_PER_BIT_PER_HOP,
    per_hop_latency=ICI_PER_HOP_LATENCY,
)


def evaluate_topology(topo: Topology, data_bytes: float) -> dict:
    """Evaluate a topology on AllReduce and broadcast for a given data size."""
    s = topo.summary()
    ar_energy, ar_latency = topo.allreduce_cost(data_bytes)
    bc_energy, bc_latency = topo.broadcast_cost(data_bytes, src=0)

    # Congested: concurrent AllReduce + broadcast (worst case for training)
    ar_loads = topo._allreduce_link_loads(data_bytes, list(range(topo.num_chips)))
    bc_loads = topo._broadcast_link_loads(data_bytes, 0,
                                          list(range(1, topo.num_chips)))
    _, congested_latency = topo.compute_congested_cost([ar_loads, bc_loads])

    return {
        "diameter": s["diameter"],
        "avg_hops": s["avg_hops"],
        "bisection_bw_TB_s": s["bisection_bandwidth_TB_s"],
        "degree": s["min_degree"],
        "ar_latency_ms": ar_latency * 1e3,
        "ar_energy_J": ar_energy,
        "bc_latency_ms": bc_latency * 1e3,
        "bc_energy_J": bc_energy,
        "congested_latency_ms": congested_latency * 1e3,
    }


def phase1_circulant_search(data_bytes: float) -> list[dict]:
    """Exhaustive search over Hamiltonian-decomposable circulant graphs."""
    print(f"\n{'='*80}")
    print("PHASE 1: Hamiltonian-Decomposable Circulant Search")
    print(f"{'='*80}")

    # Odd numbers 1..31 that are coprime to 64
    # Since 64 = 2^6, any odd number is coprime to 64
    odd_generators = [g for g in range(1, N // 2) if g % 2 == 1]
    print(f"  Odd generators coprime to {N}: {odd_generators}")
    print(f"  C({len(odd_generators)}, {NUM_GENS}) = "
          f"{math.comb(len(odd_generators), NUM_GENS)} candidate triples")

    results = []
    t0 = time.time()

    for gens in itertools.combinations(odd_generators, NUM_GENS):
        try:
            topo = CirculantHD(N, gens, **hw)
        except ValueError:
            continue

        metrics = evaluate_topology(topo, data_bytes)
        metrics["name"] = f"C(64, {{{','.join(map(str, gens))}}})"
        metrics["generators"] = gens
        metrics["type"] = "circulant_hd"
        results.append(metrics)

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["ar_latency_ms"])

    print(f"  Evaluated {len(results)} circulant topologies in {elapsed:.1f}s")
    if results:
        print(f"  Best AllReduce: {results[0]['name']} "
              f"({results[0]['ar_latency_ms']:.2f} ms)")
        print(f"  Worst AllReduce: {results[-1]['name']} "
              f"({results[-1]['ar_latency_ms']:.2f} ms)")

    return results


def phase2_random_search(data_bytes: float, num_random: int = 2000,
                         sa_steps: int = 5000) -> list[dict]:
    """Random 6-regular graphs + simulated annealing."""
    print(f"\n{'='*80}")
    print("PHASE 2: Random 6-Regular Graph Search + Simulated Annealing")
    print(f"{'='*80}")

    try:
        import networkx as nx
    except ImportError:
        print("  networkx not installed, skipping random search")
        return []

    results = []
    t0 = time.time()

    # Generate random 6-regular graphs
    print(f"  Generating {num_random} random 6-regular graphs...")
    best_random_latency = float("inf")
    best_random_adj = None

    for seed in range(num_random):
        try:
            G = nx.random_regular_graph(DEGREE, N, seed=seed)
        except nx.NetworkXError:
            continue

        adj = nx.to_numpy_array(G, dtype=int)
        topo = Custom(adj, **hw)
        metrics = evaluate_topology(topo, data_bytes)
        metrics["name"] = f"Random6Reg(seed={seed})"
        metrics["type"] = "random"
        results.append(metrics)

        if metrics["ar_latency_ms"] < best_random_latency:
            best_random_latency = metrics["ar_latency_ms"]
            best_random_adj = adj.copy()

    elapsed_random = time.time() - t0
    print(f"  Evaluated {len(results)} random graphs in {elapsed_random:.1f}s")
    if results:
        best = min(results, key=lambda r: r["ar_latency_ms"])
        print(f"  Best random: {best['name']} ({best['ar_latency_ms']:.2f} ms)")

    # Simulated annealing from best random graph
    if best_random_adj is not None and sa_steps > 0:
        print(f"  Running simulated annealing ({sa_steps} steps)...")
        t1 = time.time()
        sa_adj = best_random_adj.copy()
        sa_topo = Custom(sa_adj, **hw)
        sa_lat = evaluate_topology(sa_topo, data_bytes)["ar_latency_ms"]
        improvements = 0

        for step in range(sa_steps):
            # Random edge swap: remove edge (a,b), (c,d), add (a,c), (b,d)
            # preserving 6-regularity
            new_adj = sa_adj.copy()
            edges = list(zip(*np.where(np.triu(new_adj) > 0)))
            if len(edges) < 2:
                continue
            e1, e2 = random.sample(edges, 2)
            a, b = e1
            c, d = e2
            # Try swap: remove (a,b),(c,d), add (a,c),(b,d)
            if new_adj[a, c] == 0 and new_adj[b, d] == 0 and a != c and b != d:
                new_adj[a, b] = new_adj[b, a] = 0
                new_adj[c, d] = new_adj[d, c] = 0
                new_adj[a, c] = new_adj[c, a] = 1
                new_adj[b, d] = new_adj[d, b] = 1

                # Check connectivity
                try:
                    new_topo = Custom(new_adj, **hw)
                    if new_topo.diameter < 100:  # connected
                        new_lat = evaluate_topology(new_topo, data_bytes)["ar_latency_ms"]
                        # Accept if better or with SA probability
                        temp = max(0.01, 1.0 - step / sa_steps)
                        if new_lat < sa_lat or random.random() < math.exp(-(new_lat - sa_lat) / (sa_lat * temp * 0.1)):
                            sa_adj = new_adj
                            sa_lat = new_lat
                            if new_lat < best_random_latency:
                                improvements += 1
                except Exception:
                    pass

        elapsed_sa = time.time() - t1
        sa_topo = Custom(sa_adj, **hw)
        sa_metrics = evaluate_topology(sa_topo, data_bytes)
        sa_metrics["name"] = f"SA_optimized"
        sa_metrics["type"] = "sa_optimized"
        results.append(sa_metrics)
        print(f"  SA finished in {elapsed_sa:.1f}s, {improvements} improvements")
        print(f"  SA best: {sa_lat:.2f} ms")

    return results


def phase3_known_families(data_bytes: float) -> list[dict]:
    """Evaluate known graph families."""
    print(f"\n{'='*80}")
    print("PHASE 3: Known Topology Baselines")
    print(f"{'='*80}")

    results = []

    baselines = {
        "Torus 4x4x4": Torus3D(dims=(4, 4, 4), **hw),
        "Mesh 4x4x4": Mesh3D(dims=(4, 4, 4), **hw),
        "Ring 64": Ring(num_chips=64, **hw),
        "6D Hypercube": TorusND(dims=(2, 2, 2, 2, 2, 2), **hw),
        "4D Torus 4x4x2x2": TorusND(dims=(4, 4, 2, 2), **hw),
    }

    for name, topo in baselines.items():
        metrics = evaluate_topology(topo, data_bytes)
        metrics["name"] = name
        metrics["type"] = "baseline"
        metrics["degree"] = topo.summary()["min_degree"]
        results.append(metrics)
        print(f"  {name:25s}  deg={metrics['degree']}  "
              f"diam={metrics['diameter']}  AR={metrics['ar_latency_ms']:.2f}ms  "
              f"BC={metrics['bc_latency_ms']:.2f}ms")

    return results


def print_comparison(all_results: list[dict], data_bytes: float):
    """Print comparison table sorted by AllReduce latency."""
    # Get Torus 4x4x4 as baseline
    torus = next((r for r in all_results if r["name"] == "Torus 4x4x4"), None)
    torus_ar = torus["ar_latency_ms"] if torus else 1.0

    # Only show degree-6 topologies (fair comparison)
    degree6 = [r for r in all_results if r.get("degree", 6) == 6]
    degree6.sort(key=lambda r: r["ar_latency_ms"])

    print(f"\n{'='*120}")
    print(f"COMPARISON TABLE — 6-regular topologies on {N} chips, "
          f"data={data_bytes/1e9:.0f} GB")
    print(f"{'='*120}")
    print(f"{'Rank':>4s}  {'Topology':>30s}  {'Type':>12s}  {'Diam':>4s}  "
          f"{'AvgHop':>6s}  {'BisecBW':>8s}  {'AR (ms)':>10s}  "
          f"{'BC (ms)':>10s}  {'Cong (ms)':>10s}  {'vs Torus':>9s}")
    print("-" * 120)

    for rank, r in enumerate(degree6[:30], 1):
        speedup = torus_ar / r["ar_latency_ms"] if r["ar_latency_ms"] > 0 else 0
        print(f"{rank:>4d}  {r['name']:>30s}  {r['type']:>12s}  "
              f"{r['diameter']:>4d}  {r['avg_hops']:>6.2f}  "
              f"{r['bisection_bw_TB_s']:>7.2f}T  "
              f"{r['ar_latency_ms']:>10.2f}  {r['bc_latency_ms']:>10.2f}  "
              f"{r['congested_latency_ms']:>10.2f}  "
              f"{speedup:>8.2f}x")

    # Also show top non-degree-6 for reference
    others = [r for r in all_results if r.get("degree", 6) != 6]
    if others:
        print(f"\n--- Non-degree-6 (for reference) ---")
        for r in sorted(others, key=lambda r: r["ar_latency_ms"])[:5]:
            speedup = torus_ar / r["ar_latency_ms"] if r["ar_latency_ms"] > 0 else 0
            print(f"      {r['name']:>30s}  {r['type']:>12s}  "
                  f"{r['diameter']:>4d}  {r['avg_hops']:>6.2f}  "
                  f"{r['bisection_bw_TB_s']:>7.2f}T  "
                  f"{r['ar_latency_ms']:>10.2f}  {r['bc_latency_ms']:>10.2f}  "
                  f"{r['congested_latency_ms']:>10.2f}  "
                  f"{speedup:>8.2f}x")


def print_top_analysis(top: dict, torus: dict, data_bytes: float):
    """Detailed analysis of the best topology found."""
    print(f"\n{'='*80}")
    print("BEST TOPOLOGY ANALYSIS")
    print(f"{'='*80}")
    print(f"  Topology: {top['name']}")
    if "generators" in top:
        print(f"  Generators: {top['generators']}")
    print(f"  Type: {top['type']}")
    print()
    print(f"  {'Property':<30s} {'Best':>12s} {'Torus 4x4x4':>12s} {'Ratio':>8s}")
    print(f"  {'-'*70}")

    comparisons = [
        ("Diameter (hops)", "diameter", False),
        ("Avg hops", "avg_hops", False),
        ("Bisection BW (TB/s)", "bisection_bw_TB_s", True),
        ("AllReduce latency (ms)", "ar_latency_ms", False),
        ("AllReduce energy (J)", "ar_energy_J", False),
        ("Broadcast latency (ms)", "bc_latency_ms", False),
        ("Congested latency (ms)", "congested_latency_ms", False),
    ]

    for label, key, higher_better in comparisons:
        tv = torus[key]
        bv = top[key]
        if isinstance(bv, float):
            ratio = tv / bv if bv > 0 else 0
            if not higher_better:
                ratio_str = f"{ratio:.2f}x" if ratio >= 1 else f"{1/ratio:.2f}x worse"
            else:
                ratio = bv / tv if tv > 0 else 0
                ratio_str = f"{ratio:.2f}x"
            print(f"  {label:<30s} {bv:>12.2f} {tv:>12.2f} {ratio_str:>8s}")
        else:
            print(f"  {label:<30s} {bv:>12d} {tv:>12d}")

    ar_speedup = torus["ar_latency_ms"] / top["ar_latency_ms"]
    cong_speedup = torus["congested_latency_ms"] / top["congested_latency_ms"]
    energy_saving = (1 - top["ar_energy_J"] / torus["ar_energy_J"]) * 100

    print(f"\n  AllReduce speedup: {ar_speedup:.2f}x")
    print(f"  Congested speedup: {cong_speedup:.2f}x")
    print(f"  Energy saving: {energy_saving:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Search for optimal 6-regular topologies")
    parser.add_argument("--data-bytes", type=float, default=1.37e+11,
                        help="Data size for evaluation (default: 137 GB, matching Wide 8Kx256K)")
    parser.add_argument("--num-random", type=int, default=2000,
                        help="Number of random 6-regular graphs to evaluate")
    parser.add_argument("--sa-steps", type=int, default=3000,
                        help="Simulated annealing steps")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip phase 2 (random search)")
    args = parser.parse_args()

    data_bytes = args.data_bytes
    print(f"Searching for optimal 6-regular topology on {N} chips")
    print(f"Data size: {data_bytes/1e9:.1f} GB")
    print(f"Link bandwidth: {hw['link_bandwidth']/1e9:.0f} GB/s")

    all_results = []

    # Phase 1: Circulant search (fast, exhaustive)
    circulant_results = phase1_circulant_search(data_bytes)
    all_results.extend(circulant_results)

    # Phase 2: Random search (slower, broader)
    if not args.skip_random:
        random_results = phase2_random_search(data_bytes, args.num_random, args.sa_steps)
        all_results.extend(random_results)

    # Phase 3: Known baselines
    baseline_results = phase3_known_families(data_bytes)
    all_results.extend(baseline_results)

    # Comparison
    print_comparison(all_results, data_bytes)

    # Detailed analysis of best degree-6
    degree6 = [r for r in all_results if r.get("degree", 6) == 6]
    if degree6:
        best = min(degree6, key=lambda r: r["ar_latency_ms"])
        torus = next((r for r in all_results if r["name"] == "Torus 4x4x4"), None)
        if torus:
            print_top_analysis(best, torus, data_bytes)

    # Save results
    if args.output:
        out_path = Path(args.output)
        serializable = []
        for r in all_results:
            sr = {k: v for k, v in r.items()}
            if "generators" in sr:
                sr["generators"] = list(sr["generators"])
            serializable.append(sr)
        with out_path.open("w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
