#!/usr/bin/env python3
"""Explore circulant replacements for public TPU single-slice configurations.

This script is intentionally lightweight:

* It searches Hamiltonian-decomposable circulant graphs C(N, generators) without
  constructing an NxN all-pairs shortest-path matrix.
* It replays existing AccelForge-derived GPT-3/MoE MAP64 transfer traces with
  the same link-level congestion formulas used by the rest of this repo.

The replay is a topology/hardware-generation what-if, not a fresh AccelForge
mapping for each public TPU generation. To stay aligned with the repo's 64-chip
AccelForge runs, the default systems below are supported 64-chip single slices:
8x8 for 2D-torus generations and 4x4x4 for 3D-torus generations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from network_topology.tpu_v4 import (
    ICI_ENERGY_PER_BIT_PER_HOP,
    ICI_PER_HOP_LATENCY,
)


DEFAULT_GPT3_TRACE = Path("logs/slurm-gpt3-map64-12971957/results.json")
DEFAULT_MOE_TRACE = Path("logs/slurm-moe-map64-rerun-20260501-160133/results.json")


@dataclass(frozen=True)
class SystemSpec:
    name: str
    chips: int
    degree: int
    baseline: str
    baseline_dims: tuple[int, ...]
    link_bandwidth_Bps: float
    slice_shape: str
    source_note: str
    preferred_generators: tuple[int, ...] | None = None

    @property
    def generators(self) -> int:
        return self.degree // 2


SYSTEMS = [
    SystemSpec(
        name="TPU v5e 64-chip slice",
        chips=64,
        degree=4,
        baseline="2D torus 8x8",
        baseline_dims=(8, 8),
        link_bandwidth_Bps=100e9,
        slice_shape="8x8",
        source_note=(
            "Google Cloud TPU v5e docs: 8x8 is a supported 64-chip 2D slice; "
            "v5e has 4 ICI ports/chip and 400 GB/s bidirectional ICI per chip."
        ),
    ),
    SystemSpec(
        name="TPU v6e 64-chip slice",
        chips=64,
        degree=4,
        baseline="2D torus 8x8",
        baseline_dims=(8, 8),
        link_bandwidth_Bps=200e9,
        slice_shape="8x8",
        source_note=(
            "Google Cloud TPU v6e docs: 8x8 is a supported 64-chip 2D slice; "
            "v6e has 4 ICI ports/chip and 800 GB/s bidirectional ICI per chip."
        ),
    ),
    SystemSpec(
        name="TPU v4 one-cube slice",
        chips=64,
        degree=6,
        baseline="3D torus 4x4x4",
        baseline_dims=(4, 4, 4),
        link_bandwidth_Bps=45e9,
        slice_shape="4x4x4",
        source_note=(
            "Google Cloud TPU v4 docs: v4-128 is a supported 64-chip 4x4x4 "
            "large topology; the full v4 Pod is 4096 chips but is not used here."
        ),
        preferred_generators=(1, 5, 17),
    ),
    SystemSpec(
        name="TPU v5p one-cube slice",
        chips=64,
        degree=6,
        baseline="3D torus 4x4x4",
        baseline_dims=(4, 4, 4),
        link_bandwidth_Bps=200e9,
        slice_shape="4x4x4",
        source_note=(
            "Google Cloud TPU v5p docs: 4x4x4 is a supported 64-chip one-cube "
            "slice with full 3D torus connectivity; v5p has 1200 GB/s "
            "bidirectional ICI per chip."
        ),
        preferred_generators=(1, 5, 17),
    ),
    SystemSpec(
        name="TPU7x Ironwood one-cube slice",
        chips=64,
        degree=6,
        baseline="3D torus 4x4x4",
        baseline_dims=(4, 4, 4),
        link_bandwidth_Bps=200e9,
        slice_shape="4x4x4",
        source_note=(
            "Google Cloud TPU7x docs: 4x4x4 is a supported 64-chip one-cube "
            "slice; TPU7x has 3D torus ICI, 1200 GB/s bidirectional ICI per "
            "chip, and 200 GB/s per axis."
        ),
        preferred_generators=(1, 5, 17),
    ),
]


def coprime_generators(n: int) -> list[int]:
    return [g for g in range(1, n // 2 + 1) if math.gcd(g, n) == 1]


def nearest_coprime(n: int, target: float, *, used: set[int] | None = None) -> int:
    used = used or set()
    candidates = coprime_generators(n)
    return min(candidates, key=lambda g: (abs(g - target), g in used, g))


def l1_ball_size(radius: int, generators: int) -> int:
    """Size of a radius-r L1 ball in an infinite degree-2k lattice."""
    total = 0
    for i in range(generators + 1):
        total += (2**i) * math.comb(generators, i) * math.comb(radius, i)
    return total


def circulant_lower_bound_diameter(n: int, generators: int) -> int:
    radius = 0
    while l1_ball_size(radius, generators) < n:
        radius += 1
    return radius


def circulant_distances(n: int, generators: tuple[int, ...]) -> list[int]:
    """Distances from node 0 in C(n, generators). Circulants are vertex transitive."""
    steps = tuple(sorted(set(generators) | {n - g for g in generators}))
    dist = [-1] * n
    dist[0] = 0
    queue = deque([0])
    while queue:
        node = queue.popleft()
        next_dist = dist[node] + 1
        for step in steps:
            child = (node + step) % n
            if dist[child] < 0:
                dist[child] = next_dist
                queue.append(child)
    if any(value < 0 for value in dist):
        raise ValueError(f"C({n},{generators}) is disconnected")
    return dist


def circulant_metrics(n: int, generators: tuple[int, ...]) -> dict:
    distances = circulant_distances(n, generators)
    nonzero = distances[1:]
    return {
        "diameter": max(nonzero),
        "avg_hops": sum(nonzero) / len(nonzero),
        "degree": 2 * len(generators),
    }


def geometric_seed(n: int, k: int) -> tuple[int, ...]:
    used: set[int] = set()
    gens = []
    for i in range(k):
        target = n ** (i / k)
        if i == 0:
            target = 1
        g = nearest_coprime(n, target, used=used)
        while g in used:
            target += 1
            g = nearest_coprime(n, target, used=used)
        used.add(g)
        gens.append(g)
    return tuple(sorted(gens))


def random_generator_sets(n: int, k: int, samples: int, rng: random.Random) -> Iterable[tuple[int, ...]]:
    candidates = coprime_generators(n)
    for _ in range(samples):
        yield tuple(sorted(rng.sample(candidates, k)))


def local_search_generators(
    n: int,
    k: int,
    initial: tuple[int, ...],
    *,
    max_passes: int = 3,
) -> tuple[tuple[int, ...], dict]:
    candidates = coprime_generators(n)
    current = tuple(sorted(initial))
    current_metrics = circulant_metrics(n, current)

    def score(metrics: dict) -> tuple[float, float]:
        return metrics["diameter"], metrics["avg_hops"]

    for _pass in range(max_passes):
        improved = False
        for idx in range(k):
            best = current
            best_metrics = current_metrics
            for candidate in candidates:
                if candidate in current:
                    continue
                trial = list(current)
                trial[idx] = candidate
                trial = tuple(sorted(trial))
                if len(set(trial)) != k:
                    continue
                metrics = circulant_metrics(n, trial)
                if score(metrics) < score(best_metrics):
                    best = trial
                    best_metrics = metrics
            if best != current:
                current = best
                current_metrics = best_metrics
                improved = True
        if not improved:
            break
    return current, current_metrics


def search_circulant(n: int, k: int, *, samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    starts = {geometric_seed(n, k)}
    starts.update(random_generator_sets(n, k, samples, rng))

    best_gens: tuple[int, ...] | None = None
    best_metrics: dict | None = None
    candidates_seen = 0

    def score(metrics: dict) -> tuple[float, float]:
        return metrics["diameter"], metrics["avg_hops"]

    for gens in sorted(starts):
        candidates_seen += 1
        final_gens, metrics = local_search_generators(n, k, gens)
        if best_metrics is None or score(metrics) < score(best_metrics):
            best_gens = final_gens
            best_metrics = metrics

    assert best_gens is not None and best_metrics is not None
    lower_bound = circulant_lower_bound_diameter(n, k)
    return {
        "generators": best_gens,
        "diameter": best_metrics["diameter"],
        "avg_hops": best_metrics["avg_hops"],
        "lower_bound": lower_bound,
        "diameter_gap": best_metrics["diameter"] - lower_bound,
        "starts": candidates_seen,
    }


def prefer_known_tie(n: int, result: dict, preferred: tuple[int, ...] | None) -> dict:
    """Use the paper's C(64,{1,5,17}) when it ties the searched optimum."""
    if preferred is None:
        return result
    preferred_metrics = circulant_metrics(n, preferred)
    current_score = (result["diameter"], result["avg_hops"])
    preferred_score = (preferred_metrics["diameter"], preferred_metrics["avg_hops"])
    if preferred_score == current_score:
        updated = dict(result)
        updated["generators"] = preferred
        return updated
    return result


def torus_diameter(dims: tuple[int, ...]) -> int:
    return sum(dim // 2 for dim in dims)


def torus_avg_hops(dims: tuple[int, ...]) -> float:
    # Average ring distance over ordered pairs on a cycle, summed by dimension.
    total = 0.0
    for dim in dims:
        distances = [min(delta, dim - delta) for delta in range(dim)]
        total += sum(distances) / dim
    return total


def allreduce_max_load(data_bytes: float, n: int, k: int, kind: str, dims: tuple[int, ...] | None = None) -> float:
    if kind == "circulant":
        return 2.0 * (n - 1) * data_bytes / (n * k)
    if kind == "torus":
        assert dims is not None
        return max(2.0 * (dim - 1) * data_bytes / dim for dim in dims if dim > 1)
    raise ValueError(kind)


def collective_cost(
    collective: str,
    data_bytes: float,
    *,
    n: int,
    k: int,
    kind: str,
    diameter: int,
    link_bandwidth_Bps: float,
    dims: tuple[int, ...] | None = None,
) -> tuple[float, float]:
    """Return (energy_J, latency_s) for the repo's independent collective model."""
    if data_bytes <= 0:
        return 0.0, 0.0

    if collective == "ALLREDUCE":
        max_load = allreduce_max_load(data_bytes, n, k, kind, dims)
        total_link_bytes = 2.0 * (n - 1) * data_bytes
    elif collective in {"REDUCE_SCATTER", "ALLGATHER"}:
        max_load = allreduce_max_load(data_bytes, n, k, kind, dims) / 2.0
        total_link_bytes = (n - 1) * data_bytes
    elif collective == "BROADCAST":
        max_load = data_bytes
        total_link_bytes = (n - 1) * data_bytes
    else:
        # The AccelForge traces used here are collective-only. Keep this explicit.
        raise ValueError(f"Unsupported collective for large-system replay: {collective}")

    energy = total_link_bytes * 8.0 * ICI_ENERGY_PER_BIT_PER_HOP
    latency = max_load / link_bandwidth_Bps + diameter * ICI_PER_HOP_LATENCY
    return energy, latency


def load_transfers(path: Path, workload_substring: str) -> list[dict]:
    data = json.loads(path.read_text())
    for result in data.get("results", []):
        if workload_substring in result.get("desc", ""):
            topo_result = result["topologies"].get("Circulant {1,5,17}")
            if topo_result is None:
                topo_result = next(iter(result["topologies"].values()))
            return list(topo_result["per_transfer"])
    raise ValueError(f"No workload matching {workload_substring!r} in {path}")


def replay_trace(transfers: list[dict], spec: SystemSpec, circulant: dict) -> list[dict]:
    rows = []
    for kind in ("torus", "circulant"):
        if kind == "torus":
            diameter = torus_diameter(spec.baseline_dims)
            avg_hops = torus_avg_hops(spec.baseline_dims)
            k = len(spec.baseline_dims)
            dims = spec.baseline_dims
            name = spec.baseline
            generators = ""
        else:
            diameter = int(circulant["diameter"])
            avg_hops = float(circulant["avg_hops"])
            k = spec.generators
            dims = None
            name = f"C({spec.chips},{{{','.join(map(str, circulant['generators']))}}})"
            generators = " ".join(map(str, circulant["generators"]))

        total_energy = 0.0
        total_latency = 0.0
        total_bytes = 0.0
        mix = Counter()
        for transfer in transfers:
            collective = transfer.get("collective") or transfer.get("collective_type")
            data_bytes = float(transfer.get("data_bytes", transfer.get("bytes", 0.0)) or 0.0)
            energy, latency = collective_cost(
                collective,
                data_bytes,
                n=spec.chips,
                k=k,
                kind=kind,
                diameter=diameter,
                link_bandwidth_Bps=spec.link_bandwidth_Bps,
                dims=dims,
            )
            total_energy += energy
            total_latency += latency
            total_bytes += data_bytes
            mix[collective] += 1

        rows.append(
            {
                "system": spec.name,
                "topology": name,
                "kind": kind,
                "generators": generators,
                "diameter": diameter,
                "avg_hops": avg_hops,
                "latency_s": total_latency,
                "energy_j": total_energy,
                "trace_data_bytes": total_bytes,
                "transfer_count": len(transfers),
                "collective_mix": dict(sorted(mix.items())),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_replay(rows: list[dict], trace_name: str, out_dir: Path) -> Path:
    systems = [spec.name for spec in SYSTEMS]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    fig.patch.set_facecolor("white")
    x = np.arange(len(systems))
    width = 0.36
    for offset, kind, color in [(-width / 2, "torus", "#3D8CCB"), (width / 2, "circulant", "#6F4EAE")]:
        values = [
            next(row["latency_s"] for row in rows if row["system"] == system and row["kind"] == kind)
            for system in systems
        ]
        ax.bar(x + offset, values, width * 0.92, label=kind.title(), color=color, edgecolor="white", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Serialized network replay latency (s, log scale)")
    ax.set_title(f"Higher-TPU Single-Slice What-If Replay: {trace_name}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=20, ha="right")
    ax.grid(axis="y", which="both", color="#E6E6E6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / f"{trace_name.lower()}_higher_tpu_replay.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/higher_tpu_single_slice_circulants"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpt3-trace", type=Path, default=DEFAULT_GPT3_TRACE)
    parser.add_argument("--moe-trace", type=Path, default=DEFAULT_MOE_TRACE)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    search_rows = []
    search_by_system = {}
    for spec in SYSTEMS:
        result = search_circulant(spec.chips, spec.generators, samples=args.samples, seed=args.seed)
        result = prefer_known_tie(spec.chips, result, spec.preferred_generators)
        search_by_system[spec.name] = result
        search_rows.append(
            {
                "system": spec.name,
                "chips": spec.chips,
                "degree": spec.degree,
                "baseline": spec.baseline,
                "slice_shape": spec.slice_shape,
                "baseline_dims": "x".join(map(str, spec.baseline_dims)),
                "baseline_diameter": torus_diameter(spec.baseline_dims),
                "baseline_avg_hops": torus_avg_hops(spec.baseline_dims),
                "circulant_generators": " ".join(map(str, result["generators"])),
                "circulant_diameter": result["diameter"],
                "circulant_avg_hops": result["avg_hops"],
                "circulant_lower_bound": result["lower_bound"],
                "circulant_diameter_gap": result["diameter_gap"],
                "starts": result["starts"],
                "source_note": spec.source_note,
            }
        )

    write_csv(args.out_dir / "circulant_search_summary.csv", search_rows)

    gpt3_transfers = load_transfers(args.gpt3_trace, "GPT3 175B")
    moe_transfers = load_transfers(args.moe_trace, "MoE ExpertFFN E16 T1024")

    gpt3_rows = []
    moe_rows = []
    for spec in SYSTEMS:
        gpt3_rows.extend(replay_trace(gpt3_transfers, spec, search_by_system[spec.name]))
        moe_rows.extend(replay_trace(moe_transfers, spec, search_by_system[spec.name]))
    for row in gpt3_rows:
        row["trace"] = "gpt3_map64"
    for row in moe_rows:
        row["trace"] = "moe_expertffn_t1024_map64"
    replay_rows = gpt3_rows + moe_rows
    write_csv(args.out_dir / "trace_replay_summary.csv", replay_rows)
    plot_replay(gpt3_rows, "GPT3", args.out_dir)
    plot_replay(moe_rows, "MoE", args.out_dir)

    lines = [
        "# Higher-TPU Single-Slice Circulant Analysis",
        "",
        "This is a topology what-if over public TPU single-slice shapes and connectivities.",
        "The AccelForge portion reuses existing MAP64 GPT-3 and MoE transfer traces,",
        "then replays those collective traces under same-size torus and circulant systems",
        "with the repo's independent link-congestion cost model.",
        "This deliberately uses 64-chip slices, not full-pod chip counts.",
        "",
        "## Public TPU Single-Slice Baselines",
        "",
    ]
    for spec in SYSTEMS:
        lines.append(f"- {spec.name}: {spec.source_note}")
    lines.extend(["", "## Generator Search Results", ""])
    lines.append(
        "| System | Baseline | Best searched circulant | Diameter | Avg hops | L1 lower bound |"
    )
    lines.append("|---|---|---|---:|---:|---:|")
    for row in search_rows:
        lines.append(
            f"| {row['system']} | {row['baseline']} ({row['slice_shape']}) | "
            f"C({row['chips']},{{{row['circulant_generators'].replace(' ', ',')}}}) | "
            f"{row['circulant_diameter']} | {row['circulant_avg_hops']:.2f} | "
            f"{row['circulant_lower_bound']} |"
        )
    lines.extend(["", "## AccelForge-Trace Replay", ""])
    lines.append("| Trace | System | Torus latency | Circulant latency | Torus/Circulant |")
    lines.append("|---|---|---:|---:|---:|")
    for trace, rows in [("GPT-3 MAP64", gpt3_rows), ("MoE ExpertFFN T1024 MAP64", moe_rows)]:
        for spec in SYSTEMS:
            torus = next(row for row in rows if row["system"] == spec.name and row["kind"] == "torus")
            circ = next(row for row in rows if row["system"] == spec.name and row["kind"] == "circulant")
            lines.append(
                f"| {trace} | {spec.name} | {torus['latency_s']:.3g}s | "
                f"{circ['latency_s']:.3g}s | {torus['latency_s'] / circ['latency_s']:.2f}x |"
            )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This does not prove global graph optimality; it searches Hamiltonian-decomposable circulants.",
            "- The replay is not a fresh per-generation AccelForge mapping. It re-costs existing AccelForge-derived MAP64 traces.",
            "- Physical feasibility depends on packaging, cable/OCS constraints, routing firmware, and serviceability.",
            "- v4/v5p/TPU7x pod sizes are larger than 64 chips; this run intentionally models one 4x4x4 cube slice.",
            "- Per-link bandwidth is modeled as bidirectional per-chip ICI bandwidth divided by the modeled link budget.",
        ]
    )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_dir}")
    for path in sorted(args.out_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
