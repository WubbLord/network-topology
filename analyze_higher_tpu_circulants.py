#!/usr/bin/env python3
"""Explore circulant replacements for public TPU slice configurations.

This script is intentionally lightweight:

* It searches Hamiltonian-decomposable circulant graphs C(N, generators) without
  constructing an NxN all-pairs shortest-path matrix.
* It replays existing AccelForge-derived GPT-3/MoE MAP64 transfer traces with
  the same link-level congestion formulas used by the rest of this repo.

The replay is a topology/hardware-generation what-if, not a fresh AccelForge
mapping for each public TPU generation. The 64-chip scenario stays aligned with
the repo's MAP64 AccelForge runs; the realistic-large-slices scenario asks how
the circulant idea extends to larger supported TPU slice shapes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
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


SYSTEMS_64 = [
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


REALISTIC_LARGE_SYSTEMS = [
    SystemSpec(
        name="TPU v5e 128-chip slice",
        chips=128,
        degree=4,
        baseline="2D torus 8x16",
        baseline_dims=(8, 16),
        link_bandwidth_Bps=100e9,
        slice_shape="8x16",
        source_note="Google Cloud TPU v5e docs: 8x16 is a supported 128-chip 2D slice.",
    ),
    SystemSpec(
        name="TPU v5e full-pod slice",
        chips=256,
        degree=4,
        baseline="2D torus 16x16",
        baseline_dims=(16, 16),
        link_bandwidth_Bps=100e9,
        slice_shape="16x16",
        source_note="Google Cloud TPU v5e docs: 16x16 is the supported 256-chip full-pod 2D slice.",
    ),
    SystemSpec(
        name="TPU v6e 128-chip slice",
        chips=128,
        degree=4,
        baseline="2D torus 8x16",
        baseline_dims=(8, 16),
        link_bandwidth_Bps=200e9,
        slice_shape="8x16",
        source_note="Google Cloud TPU v6e docs: 8x16 is a supported 128-chip 2D slice.",
    ),
    SystemSpec(
        name="TPU v6e full-pod slice",
        chips=256,
        degree=4,
        baseline="2D torus 16x16",
        baseline_dims=(16, 16),
        link_bandwidth_Bps=200e9,
        slice_shape="16x16",
        source_note="Google Cloud TPU v6e docs: 16x16 is the supported 256-chip full-pod 2D slice.",
    ),
    SystemSpec(
        name="TPU v4 128-chip slice",
        chips=128,
        degree=6,
        baseline="3D torus 4x4x8",
        baseline_dims=(4, 4, 8),
        link_bandwidth_Bps=45e9,
        slice_shape="4x4x8",
        source_note="Google Cloud TPU v4 docs: v4-256 is a supported 128-chip 4x4x8 topology.",
    ),
    SystemSpec(
        name="TPU v4 256-chip slice",
        chips=256,
        degree=6,
        baseline="3D torus 4x8x8",
        baseline_dims=(4, 8, 8),
        link_bandwidth_Bps=45e9,
        slice_shape="4x8x8",
        source_note="Google Cloud TPU v4 docs: v4-512 is a supported 256-chip 4x8x8 topology.",
    ),
    SystemSpec(
        name="TPU v4 512-chip slice",
        chips=512,
        degree=6,
        baseline="3D torus 8x8x8",
        baseline_dims=(8, 8, 8),
        link_bandwidth_Bps=45e9,
        slice_shape="8x8x8",
        source_note="Google Cloud TPU v4 docs: v4-1024 is a supported 512-chip 8x8x8 topology.",
    ),
    SystemSpec(
        name="TPU v4 1024-chip slice",
        chips=1024,
        degree=6,
        baseline="3D torus 8x8x16",
        baseline_dims=(8, 8, 16),
        link_bandwidth_Bps=45e9,
        slice_shape="8x8x16",
        source_note="Google Cloud TPU v4 docs: v4-2048 is a supported 1024-chip 8x8x16 topology.",
    ),
    SystemSpec(
        name="TPU v4 2048-chip slice",
        chips=2048,
        degree=6,
        baseline="3D torus 8x16x16",
        baseline_dims=(8, 16, 16),
        link_bandwidth_Bps=45e9,
        slice_shape="8x16x16",
        source_note="Google Cloud TPU v4 docs: v4-4096 is a supported 2048-chip 8x16x16 topology.",
    ),
    SystemSpec(
        name="TPU v5p 128-chip slice",
        chips=128,
        degree=6,
        baseline="3D torus 4x4x8",
        baseline_dims=(4, 4, 8),
        link_bandwidth_Bps=200e9,
        slice_shape="4x4x8",
        source_note="Google Cloud TPU v5p docs: 4x4x8 is a supported 128-chip two-cube slice.",
    ),
    SystemSpec(
        name="TPU v5p 256-chip slice",
        chips=256,
        degree=6,
        baseline="3D torus 4x8x8",
        baseline_dims=(4, 8, 8),
        link_bandwidth_Bps=200e9,
        slice_shape="4x8x8",
        source_note="Google Cloud TPU v5p docs: 4x8x8 is a supported 256-chip four-cube slice.",
    ),
    SystemSpec(
        name="TPU v5p 512-chip slice",
        chips=512,
        degree=6,
        baseline="3D torus 8x8x8",
        baseline_dims=(8, 8, 8),
        link_bandwidth_Bps=200e9,
        slice_shape="8x8x8",
        source_note="Google Cloud TPU v5p docs: 8x8x8 is a supported 512-chip eight-cube slice.",
    ),
    SystemSpec(
        name="TPU v5p 1024-chip slice",
        chips=1024,
        degree=6,
        baseline="3D torus 8x8x16",
        baseline_dims=(8, 8, 16),
        link_bandwidth_Bps=200e9,
        slice_shape="8x8x16",
        source_note="Google Cloud TPU v5p docs: 8x8x16 is a supported 1024-chip 16-cube slice.",
    ),
    SystemSpec(
        name="TPU v5p 2048-chip slice",
        chips=2048,
        degree=6,
        baseline="3D torus 8x16x16",
        baseline_dims=(8, 16, 16),
        link_bandwidth_Bps=200e9,
        slice_shape="8x16x16",
        source_note="Google Cloud TPU v5p docs: 8x16x16 is a supported 2048-chip 32-cube slice.",
    ),
    SystemSpec(
        name="TPU v5p 4096-chip slice",
        chips=4096,
        degree=6,
        baseline="3D torus 16x16x16",
        baseline_dims=(16, 16, 16),
        link_bandwidth_Bps=200e9,
        slice_shape="16x16x16",
        source_note="Google Cloud TPU v5p docs: 16x16x16 is a supported 4096-chip 64-cube slice.",
    ),
    SystemSpec(
        name="TPU v5p 6144-chip max slice",
        chips=6144,
        degree=6,
        baseline="3D torus 16x16x24",
        baseline_dims=(16, 16, 24),
        link_bandwidth_Bps=200e9,
        slice_shape="16x16x24",
        source_note="Google Cloud TPU v5p docs: 16x16x24 is the maximum listed 6144-chip single-slice training shape.",
    ),
    SystemSpec(
        name="TPU7x 128-chip slice",
        chips=128,
        degree=6,
        baseline="3D torus 4x4x8",
        baseline_dims=(4, 4, 8),
        link_bandwidth_Bps=200e9,
        slice_shape="4x4x8",
        source_note="Google Cloud TPU7x docs: 4x4x8 is a supported 128-chip two-cube slice.",
    ),
    SystemSpec(
        name="TPU7x 256-chip slice",
        chips=256,
        degree=6,
        baseline="3D torus 4x8x8",
        baseline_dims=(4, 8, 8),
        link_bandwidth_Bps=200e9,
        slice_shape="4x8x8",
        source_note="Google Cloud TPU7x docs: 4x8x8 is a supported 256-chip four-cube slice.",
    ),
    SystemSpec(
        name="TPU7x 512-chip slice",
        chips=512,
        degree=6,
        baseline="3D torus 8x8x8",
        baseline_dims=(8, 8, 8),
        link_bandwidth_Bps=200e9,
        slice_shape="8x8x8",
        source_note="Google Cloud TPU7x docs: 8x8x8 is a supported 512-chip eight-cube slice.",
    ),
    SystemSpec(
        name="TPU7x 1024-chip slice",
        chips=1024,
        degree=6,
        baseline="3D torus 8x8x16",
        baseline_dims=(8, 8, 16),
        link_bandwidth_Bps=200e9,
        slice_shape="8x8x16",
        source_note="Google Cloud TPU7x docs: 8x8x16 is a supported 1024-chip 16-cube slice.",
    ),
    SystemSpec(
        name="TPU7x 2048-chip slice",
        chips=2048,
        degree=6,
        baseline="3D torus 8x16x16",
        baseline_dims=(8, 16, 16),
        link_bandwidth_Bps=200e9,
        slice_shape="8x16x16",
        source_note="Google Cloud TPU7x docs: 8x16x16 is a supported 2048-chip 32-cube slice.",
    ),
]


SCENARIOS = {
    "single-slice64": SYSTEMS_64,
    "realistic-large-slices": REALISTIC_LARGE_SYSTEMS,
    "all": SYSTEMS_64 + REALISTIC_LARGE_SYSTEMS,
}


@lru_cache(maxsize=None)
def coprime_generators(n: int) -> tuple[int, ...]:
    return tuple(g for g in range(1, n // 2 + 1) if math.gcd(g, n) == 1)


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


def normalize_generator_set(n: int, values: Iterable[float], k: int) -> tuple[int, ...] | None:
    used: set[int] = set()
    gens: list[int] = []
    for value in values:
        target = min(max(1.0, value), n / 2)
        gen = nearest_coprime(n, target, used=used)
        offset = 0
        while gen in used and offset < n:
            offset += 1
            gen = nearest_coprime(n, target + offset, used=used)
        if gen in used:
            return None
        used.add(gen)
        gens.append(gen)
    if len(gens) != k:
        return None
    return tuple(sorted(gens))


def structured_generator_sets(n: int, k: int, rng: random.Random, samples: int) -> set[tuple[int, ...]]:
    """Candidate pool for larger circulants.

    The exhaustive coordinate search is fine at N=64, but gets unnecessarily
    slow for 1K-6K chip slices. These seeds cover common lattice-like choices,
    scaled versions of the paper's 64-chip design, and random Hamiltonian
    decompositions.
    """
    starts: set[tuple[int, ...]] = {geometric_seed(n, k)}
    if k == 2:
        root = math.sqrt(n)
        for delta in range(-8, 9):
            candidate = normalize_generator_set(n, (root + delta, n / (root + delta + 1)), k)
            if candidate is not None:
                starts.add(candidate)
        for frac in (1 / 5, 1 / 4, 1 / 3, 2 / 5):
            candidate = normalize_generator_set(n, (n * frac, n * (0.5 - frac / 2)), k)
            if candidate is not None:
                starts.add(candidate)
    elif k == 3:
        root = n ** (1 / 3)
        scaled_paper = normalize_generator_set(n, (1, 5 * n / 64, 17 * n / 64), k)
        if scaled_paper is not None:
            starts.add(scaled_paper)
        for d1 in range(-4, 5):
            for d2 in range(-6, 7, 2):
                candidate = normalize_generator_set(n, (1, root + d1, root * root + d2 * root), k)
                if candidate is not None:
                    starts.add(candidate)
        for f1, f2 in ((1 / 9, 1 / 3), (1 / 11, 3 / 11), (1 / 13, 5 / 13), (1 / 17, 5 / 17)):
            candidate = normalize_generator_set(n, (1, n * f1, n * f2), k)
            if candidate is not None:
                starts.add(candidate)
    starts.update(random_generator_sets(n, k, samples, rng))
    return starts


def mutate_generator_sets(
    n: int,
    seeds: Iterable[tuple[int, ...]],
    *,
    window: int,
) -> set[tuple[int, ...]]:
    mutations: set[tuple[int, ...]] = set()
    deltas = sorted(set(range(-window, window + 1)) | {-2 * window, 2 * window})
    for seed in seeds:
        for idx in range(len(seed)):
            for delta in deltas:
                if delta == 0:
                    continue
                trial = list(seed)
                trial[idx] = trial[idx] + delta
                candidate = normalize_generator_set(n, trial, len(seed))
                if candidate is not None:
                    mutations.add(candidate)
    return mutations


def heuristic_search_generators(n: int, k: int, *, samples: int, seed: int) -> tuple[tuple[int, ...], dict, int]:
    rng = random.Random(seed)
    candidates = structured_generator_sets(n, k, rng, samples)

    def score(metrics: dict) -> tuple[float, float]:
        return metrics["diameter"], metrics["avg_hops"]

    evaluated: dict[tuple[int, ...], dict] = {}
    best_gens: tuple[int, ...] | None = None
    best_metrics: dict | None = None

    for round_idx in range(3):
        for gens in sorted(candidates):
            if gens in evaluated:
                continue
            metrics = circulant_metrics(n, gens)
            evaluated[gens] = metrics
            if best_metrics is None or score(metrics) < score(best_metrics):
                best_gens = gens
                best_metrics = metrics
        top = sorted(evaluated.items(), key=lambda item: score(item[1]))[:12]
        window = max(2, int(round(n ** (1 / k))) // (2 + round_idx))
        candidates = mutate_generator_sets(n, (gens for gens, _ in top), window=window)

    assert best_gens is not None and best_metrics is not None
    return best_gens, best_metrics, len(evaluated)


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
    search_mode = "coordinate"
    if n <= 512:
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
    else:
        search_mode = "heuristic"
        best_gens, best_metrics, candidates_seen = heuristic_search_generators(
            n,
            k,
            samples=samples,
            seed=seed,
        )
    lower_bound = circulant_lower_bound_diameter(n, k)
    return {
        "generators": best_gens,
        "diameter": best_metrics["diameter"],
        "avg_hops": best_metrics["avg_hops"],
        "lower_bound": lower_bound,
        "diameter_gap": best_metrics["diameter"] - lower_bound,
        "starts": candidates_seen,
        "search_mode": search_mode,
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


def plot_replay(rows: list[dict], trace_name: str, out_dir: Path, systems: list[SystemSpec]) -> Path:
    system_names = [spec.name for spec in systems]
    fig, ax = plt.subplots(figsize=(max(13, len(system_names) * 0.62), 6.2))
    fig.patch.set_facecolor("white")
    x = np.arange(len(system_names))
    width = 0.36
    for offset, kind, color in [(-width / 2, "torus", "#3D8CCB"), (width / 2, "circulant", "#6F4EAE")]:
        values = [
            next(row["latency_s"] for row in rows if row["system"] == system and row["kind"] == kind)
            for system in system_names
        ]
        ax.bar(x + offset, values, width * 0.92, label=kind.title(), color=color, edgecolor="white", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Serialized network replay latency (s, log scale)")
    ax.set_title(f"Higher-TPU Slice What-If Replay: {trace_name}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(system_names, rotation=35, ha="right")
    ax.grid(axis="y", which="both", color="#E6E6E6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / f"{trace_name.lower()}_higher_tpu_replay.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_speedups(rows: list[dict], trace_name: str, out_dir: Path, systems: list[SystemSpec]) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    fig.patch.set_facecolor("white")
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for spec in systems:
        torus = next(row for row in rows if row["system"] == spec.name and row["kind"] == "torus")
        circ = next(row for row in rows if row["system"] == spec.name and row["kind"] == "circulant")
        generation = spec.name.split()[1] if spec.name.startswith("TPU ") else spec.name.split()[0]
        groups[generation].append((spec.chips, torus["latency_s"] / circ["latency_s"]))
    colors = ["#3D8CCB", "#6F4EAE", "#2E8B57", "#C06B3E", "#555555"]
    for color, (generation, values) in zip(colors, sorted(groups.items())):
        values = sorted(values)
        ax.plot(
            [chips for chips, _ in values],
            [speedup for _, speedup in values],
            marker="o",
            linewidth=2,
            label=generation,
            color=color,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("TPU chips in supported slice")
    ax.set_ylabel("Torus latency / circulant latency")
    ax.set_title(f"Circulant Speedup Across Realistic Slices: {trace_name}", fontweight="bold")
    ax.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = out_dir / f"{trace_name.lower()}_slice_speedups.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("logs/higher_tpu_circulants"))
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="realistic-large-slices",
        help="Which public TPU slice set to evaluate.",
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpt3-trace", type=Path, default=DEFAULT_GPT3_TRACE)
    parser.add_argument("--moe-trace", type=Path, default=DEFAULT_MOE_TRACE)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    systems = SCENARIOS[args.scenario]

    search_rows = []
    search_by_system = {}
    search_cache = {}
    for spec in systems:
        cache_key = (spec.chips, spec.generators)
        if cache_key not in search_cache:
            search_cache[cache_key] = search_circulant(
                spec.chips,
                spec.generators,
                samples=args.samples,
                seed=args.seed,
            )
        result = prefer_known_tie(spec.chips, search_cache[cache_key], spec.preferred_generators)
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
                "search_mode": result["search_mode"],
                "source_note": spec.source_note,
            }
        )

    write_csv(args.out_dir / "circulant_search_summary.csv", search_rows)

    gpt3_transfers = load_transfers(args.gpt3_trace, "GPT3 175B")
    moe_transfers = load_transfers(args.moe_trace, "MoE ExpertFFN E16 T1024")

    gpt3_rows = []
    moe_rows = []
    for spec in systems:
        gpt3_rows.extend(replay_trace(gpt3_transfers, spec, search_by_system[spec.name]))
        moe_rows.extend(replay_trace(moe_transfers, spec, search_by_system[spec.name]))
    for row in gpt3_rows:
        row["trace"] = "gpt3_map64"
    for row in moe_rows:
        row["trace"] = "moe_expertffn_t1024_map64"
    replay_rows = gpt3_rows + moe_rows
    write_csv(args.out_dir / "trace_replay_summary.csv", replay_rows)
    plot_replay(gpt3_rows, "GPT3", args.out_dir, systems)
    plot_replay(moe_rows, "MoE", args.out_dir, systems)
    if args.scenario != "single-slice64":
        plot_speedups(gpt3_rows, "GPT3", args.out_dir, systems)
        plot_speedups(moe_rows, "MoE", args.out_dir, systems)

    title = {
        "single-slice64": "Higher-TPU 64-Chip Single-Slice Circulant Analysis",
        "realistic-large-slices": "Higher-TPU Realistic Large-Slice Circulant Analysis",
        "all": "Higher-TPU Circulant Analysis",
    }[args.scenario]
    intent = {
        "single-slice64": "This deliberately uses 64-chip slices, not full-pod chip counts.",
        "realistic-large-slices": (
            "This deliberately uses supported slices larger than 64 chips to test "
            "whether the circulant idea extends beyond the paper's MAP64 design."
        ),
        "all": "This includes both the 64-chip baseline slices and larger supported slices.",
    }[args.scenario]
    lines = [
        f"# {title}",
        "",
        "This is a topology what-if over public TPU slice shapes and connectivities.",
        "The AccelForge portion reuses existing MAP64 GPT-3 and MoE transfer traces,",
        "then replays those collective traces under torus and searched-circulant systems",
        "with the repo's independent link-congestion cost model.",
        intent,
        "",
        "## Public TPU Slice Baselines",
        "",
    ]
    for spec in systems:
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
        for spec in systems:
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
            "- Larger-slice results are theoretical extensions of the AccelForge MAP64 communication trace, not new large-N mapper runs.",
            "- Per-link bandwidth is modeled as bidirectional per-chip ICI bandwidth divided by the modeled link budget.",
        ]
    )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_dir}")
    for path in sorted(args.out_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
