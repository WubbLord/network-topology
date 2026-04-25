#!/usr/bin/env python3
"""
Aggregate per-job JSON results from parallel Slurm jobs into a single results.json.

Usage:
    python aggregate_results.py <run-dir>

Reads all partial JSON files in <run-dir>, merges them by workload and topology,
prints the summary table with stress analysis, and writes a combined results.json.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

TOPOLOGY_ORDER = ["Circulant {1,5,17}", "6D Hypercube", "Torus 4x4x4", "4D Torus 4x4x2x2", "Mesh 4x4x4", "Ring 64", "Torus 8x2x4", "5D Torus 4x2x2x2x2"]


def load_partial_results(run_dir: Path):
    """Load all partial JSON files, returning (workload_desc, topo_name) -> data."""
    entries = {}
    for json_file in sorted(run_dir.glob("*.json")):
        if json_file.name == "results.json":
            continue
        with json_file.open() as f:
            data = json.load(f)
        for r in data.get("results", []):
            desc = r.get("desc", "Unknown")
            for topo_name, topo_data in r.get("topologies", {}).items():
                entries[(desc, topo_name)] = {
                    "file": json_file.name,
                    "metadata": data,
                    "workload_result": r,
                    "topology_result": topo_data,
                }
    return entries


def merge_results(run_dir: Path):
    entries = load_partial_results(run_dir)
    if not entries:
        print(f"No partial JSON files found in {run_dir}")
        sys.exit(1)

    # Group by workload
    workload_descs = []
    seen = set()
    for (desc, _) in entries:
        if desc not in seen:
            workload_descs.append(desc)
            seen.add(desc)

    # Use first entry's metadata as base
    first = next(iter(entries.values()))["metadata"]

    # Collect topology summaries
    topology_summaries = {}
    for (_, topo_name), info in entries.items():
        if topo_name not in topology_summaries:
            topology_summaries[topo_name] = info["topology_result"].get("topology_summary", {})

    available_topos = [t for t in TOPOLOGY_ORDER if t in topology_summaries]

    # Build merged results per workload
    merged_results = []
    for desc in workload_descs:
        lats = {}
        bytes_by_topo = {}
        ag_pct_by_topo = {}
        compute_e_by_topo = {}
        compute_l_by_topo = {}
        wall_time_by_topo = {}
        topo_results = {}

        base_wr = None
        for topo_name in TOPOLOGY_ORDER:
            key = (desc, topo_name)
            if key not in entries:
                continue
            info = entries[key]
            tr = info["topology_result"]
            wr = info["workload_result"]
            if base_wr is None:
                base_wr = wr

            lats[topo_name] = tr.get("total_latency", 0)
            bytes_by_topo[topo_name] = tr.get("bytes", wr.get("bytes", 0))
            ag_pct_by_topo[topo_name] = tr.get("allgather_pct", wr.get("allgather_pct", 0))
            compute_e_by_topo[topo_name] = tr.get("compute_energy", wr.get("compute_e", 0))
            compute_l_by_topo[topo_name] = tr.get("compute_latency", wr.get("compute_l", 0))
            wall_time_by_topo[topo_name] = tr.get("mapping_wall_time_s", 0)
            topo_results[topo_name] = tr

        if not lats or base_wr is None:
            continue

        n = len(lats)
        merged_results.append({
            "desc": desc,
            "workload_path": base_wr.get("workload_path", ""),
            "params": base_wr.get("params", {}),
            "compute_e": sum(compute_e_by_topo.values()) / n,
            "compute_l": sum(compute_l_by_topo.values()) / n,
            "compute_e_by_topology": compute_e_by_topo,
            "compute_l_by_topology": compute_l_by_topo,
            "mapping_wall_time_s": sum(wall_time_by_topo.values()) / n,
            "mapping_wall_time_s_by_topology": wall_time_by_topo,
            "bytes": sum(bytes_by_topo.values()) / n,
            "bytes_by_topology": bytes_by_topo,
            "allgather_pct": sum(ag_pct_by_topo.values()) / n,
            "allgather_pct_by_topology": ag_pct_by_topo,
            "lats": lats,
            "topologies": topo_results,
        })

    # ---- Print summary table ----
    print(f"\n{'=' * 140}")
    print("MERGED RESULTS (latency in ms, with link congestion)")
    print(f"{'=' * 140}")
    print(f"{'Workload':>30s} {'Bytes':>10s} {'AG%':>5s}", end="")
    for tn in available_topos:
        print(f" {tn:>14s}", end="")
    print(f" {'Best':>10s} {'Speedup':>8s}")
    print("-" * 140)

    torus_wins = 0
    for r in merged_results:
        avail = [t for t in TOPOLOGY_ORDER if t in r["lats"]]
        if not avail:
            continue
        best = min(avail, key=r["lats"].get)
        worst = max(avail, key=r["lats"].get)
        speedup = r["lats"][worst] / r["lats"][best] if r["lats"][best] > 0 else 0
        if best == "Torus 4x4x4":
            torus_wins += 1
        print(f"{r['desc']:>30s} {r['bytes']:>10.2e} {r['allgather_pct']:>4.0f}%", end="")
        for tn in available_topos:
            lat = r["lats"].get(tn)
            if lat is None:
                print(f" {'n/a':>14s}", end="")
            elif lat == 0:
                print(f" {'0.0ms':>14s}", end="")
            else:
                print(f" {lat * 1e3:>12.1f}ms", end="")
        print(f" {best.split()[0]:>10s} {speedup:>7.2f}x")

    # ---- Per-workload stress analysis ----
    print(f"\n{'=' * 140}")
    print("NETWORK STRESS ANALYSIS (per-transfer breakdown)")
    print(f"{'=' * 140}")

    for r in merged_results:
        print(f"\n--- {r['desc']} ---")
        for topo_name in available_topos:
            if topo_name not in r["topologies"]:
                continue
            tr = r["topologies"][topo_name]
            transfers = tr.get("per_transfer", [])
            if not transfers:
                print(f"  {topo_name}: no transfers")
                continue

            total_lat = tr.get("total_latency", 0)
            total_bytes = tr.get("total_network_bytes", 0)
            print(f"  {topo_name}: {len(transfers)} transfers, "
                  f"{total_bytes:.2e} bytes, {total_lat*1e3:.1f}ms total")

            # Sort transfers by latency contribution
            sorted_xfers = sorted(transfers, key=lambda x: x.get("latency", 0), reverse=True)
            for i, xf in enumerate(sorted_xfers[:5]):  # top 5 hottest transfers
                tensor = xf.get("tensor", "?")
                coll = xf.get("collective", "?")
                lat = xf.get("latency", 0)
                dbytes = xf.get("data_bytes", 0)
                pct = (lat / total_lat * 100) if total_lat > 0 else 0
                print(f"    [{i+1}] {tensor:>40s}  {coll:<15s}  "
                      f"{dbytes:.2e}B  {lat*1e3:>8.1f}ms  ({pct:>5.1f}%)")

    # ---- Collective mix ----
    print(f"\n{'=' * 140}")
    print("COLLECTIVE MIX (bytes by type per workload)")
    print(f"{'=' * 140}")

    for r in merged_results:
        # Use first available topology for collective mix
        for topo_name in available_topos:
            if topo_name not in r["topologies"]:
                continue
            tr = r["topologies"][topo_name]
            coll_decisions = tr.get("collective_decisions", [])
            if not coll_decisions:
                continue

            mix = defaultdict(float)
            for d in coll_decisions:
                ctype = d.get("collective_type") or "LOCAL"
                mix[ctype] += d.get("data_bytes", 0)

            total = sum(mix.values()) or 1
            parts = ", ".join(f"{k}: {v/total*100:.0f}%" for k, v in sorted(mix.items()))
            print(f"  {r['desc']:>30s} ({topo_name}): {parts}")
            break  # One topology is enough to show the mix

    # ---- Insights ----
    print(f"\n{'=' * 140}")
    print("INSIGHTS")
    print(f"{'=' * 140}")
    print(f"Torus 4x4x4 wins: {torus_wins}/{len(merged_results)} workloads")
    insights = {"torus_4x4x4_wins": torus_wins, "num_workloads": len(merged_results)}

    # ---- Wall times ----
    print(f"\nPer-job wall times:")
    total_wall = 0
    max_wall = 0
    for r in merged_results:
        for tn, wt in r.get("mapping_wall_time_s_by_topology", {}).items():
            total_wall += wt
            max_wall = max(max_wall, wt)
            print(f"  {r['desc']:>30s} x {tn:<14s}: {wt:.0f}s ({wt/60:.1f}m)")

    if max_wall > 0:
        print(f"\n  Parallel wall time:  {max_wall:.0f}s ({max_wall/60:.1f}m)")
        print(f"  Serial would be:     {total_wall:.0f}s ({total_wall/60:.1f}m)")
        print(f"  Parallelization speedup: {total_wall/max_wall:.1f}x")

    # ---- Save combined results ----
    combined = {
        "run_timestamp": first.get("run_timestamp", ""),
        "accelforge_root": first.get("accelforge_root", ""),
        "workloads_dir": first.get("workloads_dir", ""),
        "architecture_yaml": first.get("architecture_yaml", ""),
        "map_chips": first.get("map_chips", 8),
        "eval_chips": first.get("eval_chips", 64),
        "scale": first.get("scale", 8.0),
        "feedback_loop": first.get("feedback_loop", {}),
        "topology_summaries": topology_summaries,
        "results": merged_results,
        "insights": insights,
    }

    out_path = run_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, sort_keys=True)
    print(f"\nSaved combined results to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <run-dir>")
        sys.exit(1)
    merge_results(Path(sys.argv[1]))
