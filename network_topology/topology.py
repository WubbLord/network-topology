"""
Network topology definitions with link-level congestion modeling.

Congestion model:
    1. Each transfer is routed onto physical links with per-link byte loads.
    2. Link loads from all concurrent transfers are merged.
    3. Latency = max_link_load / link_bandwidth (bottleneck link).
    4. Energy = total bit-hops (independent of congestion).

Known topologies (Ring, Mesh3D, Torus3D) override routing with optimal
algorithms. Custom topologies use generic shortest-path routing.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse.csgraph import shortest_path

LinkLoads = dict[tuple[int, int], float]


# --- Adjacency matrix generators ---

def _chip_index_nd(coords: tuple[int, ...], dims: tuple[int, ...]) -> int:
    idx = 0
    for c, d in zip(coords, dims):
        idx = idx * d + c
    return idx


def _make_ring_adj(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[i, (i - 1) % n] = 1
    return adj


def _make_mesh_adj(dims: tuple[int, ...]) -> np.ndarray:
    total = math.prod(dims)
    adj = np.zeros((total, total), dtype=int)
    for flat_idx in range(total):
        coords = []
        remaining = flat_idx
        for d in reversed(dims):
            coords.append(remaining % d)
            remaining //= d
        coords.reverse()
        for dim in range(len(dims)):
            if coords[dim] + 1 < dims[dim]:
                neighbor = list(coords)
                neighbor[dim] += 1
                j = _chip_index_nd(tuple(neighbor), dims)
                adj[flat_idx, j] = 1
                adj[j, flat_idx] = 1
    return adj


def _make_torus_adj(dims: tuple[int, ...]) -> np.ndarray:
    total = math.prod(dims)
    adj = np.zeros((total, total), dtype=int)
    for flat_idx in range(total):
        coords = []
        remaining = flat_idx
        for d in reversed(dims):
            coords.append(remaining % d)
            remaining //= d
        coords.reverse()
        for dim in range(len(dims)):
            neighbor = list(coords)
            neighbor[dim] = (coords[dim] + 1) % dims[dim]
            j = _chip_index_nd(tuple(neighbor), dims)
            if j != flat_idx:
                adj[flat_idx, j] = 1
                adj[j, flat_idx] = 1
    return adj


def _flat_to_coords(flat_idx: int, dims: tuple) -> list[int]:
    coords = []
    for d in reversed(dims):
        coords.append(flat_idx % d)
        flat_idx //= d
    coords.reverse()
    return coords


# --- Base Topology ---

@dataclass
class Topology:
    """
    Base topology defined by an adjacency matrix.

    All link parameters default to TPU v4 ICI values:
        link_bandwidth: 45 GB/s unidirectional per link [JAX Scaling Book]
        energy_per_bit_per_hop: ~4 pJ/bit [estimated from Jouppi et al. ISCA 2023]
        per_hop_latency: ~500 ns [estimated]
    """
    adj_matrix: np.ndarray
    link_bandwidth: float
    energy_per_bit_per_hop: float
    per_hop_latency: float = 500e-9
    full_duplex: bool = True

    num_chips: int = field(init=False)
    hop_count: np.ndarray = field(init=False, repr=False)
    predecessors: np.ndarray = field(init=False, repr=False)
    diameter: int = field(init=False)
    avg_hops: float = field(init=False)
    bisection_bandwidth: float = field(init=False)
    total_links: int = field(init=False)

    def __post_init__(self):
        assert self.adj_matrix.ndim == 2
        assert np.allclose(self.adj_matrix, self.adj_matrix.T), "Adjacency must be symmetric"
        self.num_chips = self.adj_matrix.shape[0]
        self.hop_count, self.predecessors = shortest_path(
            self.adj_matrix, method="D", unweighted=True, return_predecessors=True,
        )
        self.diameter = int(self.hop_count.max())
        mask = ~np.eye(self.num_chips, dtype=bool)
        self.avg_hops = float(self.hop_count[mask].mean()) if self.num_chips > 1 else 0.0
        self.total_links = int(self.adj_matrix.sum())
        self.bisection_bandwidth = self._compute_bisection_bandwidth()

    def _compute_bisection_bandwidth(self) -> float:
        n = self.num_chips
        if n <= 1:
            return 0.0
        order = np.argsort(self.hop_count.mean(axis=1))
        half = n // 2
        set_a, set_b = set(order[:half]), set(order[half:])
        crossing = sum(self.adj_matrix[i, j] for i in set_a for j in set_b)
        return crossing * self.link_bandwidth

    # --- Path reconstruction ---

    def _get_path(self, src: int, dst: int) -> list[int]:
        if src == dst:
            return [src]
        path = []
        current = dst
        while current != src:
            path.append(current)
            current = int(self.predecessors[src, current])
        path.append(src)
        path.reverse()
        return path

    def _get_path_links(self, src: int, dst: int) -> list[tuple[int, int]]:
        path = self._get_path(src, dst)
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    # --- Generic link-load routing (overridden by subclasses) ---

    def _broadcast_link_loads(self, data_bytes: float, src: int, dst_chips: list[int]) -> LinkLoads:
        """BFS spanning tree broadcast. Interior links carry data_bytes * subtree_size."""
        if not dst_chips:
            return {}
        dst_set = set(dst_chips)
        loads: LinkLoads = defaultdict(float)
        visited = {src}
        queue = [src]
        parent = {}
        while queue and dst_set:
            next_queue = []
            for node in queue:
                for neighbor in range(self.num_chips):
                    if self.adj_matrix[node, neighbor] > 0 and neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = node
                        next_queue.append(neighbor)
                        dst_set.discard(neighbor)
            queue = next_queue
        subtree_count = defaultdict(int)
        for dst in dst_chips:
            node = dst
            while node in parent:
                p = parent[node]
                subtree_count[(p, node)] += 1
                node = p
        for (p, c), count in subtree_count.items():
            loads[(p, c)] += data_bytes * count
        return dict(loads)

    def _allreduce_link_loads(self, data_bytes: float, participating_chips: list[int]) -> LinkLoads:
        """Reduce-scatter + all-gather via shortest-path pairwise routing."""
        n = len(participating_chips)
        if n <= 1:
            return {}
        loads: LinkLoads = defaultdict(float)
        chunk = data_bytes / n
        for i in participating_chips:
            for j in participating_chips:
                if i == j:
                    continue
                for link in self._get_path_links(i, j):
                    loads[link] += 2 * chunk
        return dict(loads)

    def _point_to_point_link_loads(self, data_bytes: float, src: int, dst: int) -> LinkLoads:
        if src == dst:
            return {}
        loads: LinkLoads = {}
        for link in self._get_path_links(src, dst):
            loads[link] = loads.get(link, 0) + data_bytes
        return loads

    # --- Cost from link loads ---

    def _cost_from_link_loads(self, loads: LinkLoads) -> tuple[float, float]:
        """
        Energy = sum(bytes_on_link * 8 * energy_per_bit_per_hop) for all links.
        Latency = max(bytes_on_link) / link_bandwidth + diameter * per_hop_latency.
        """
        if not loads:
            return (0.0, 0.0)
        total_bytes = sum(loads.values())
        energy = total_bytes * 8 * self.energy_per_bit_per_hop
        max_load = max(loads.values())
        latency = max_load / self.link_bandwidth + self.diameter * self.per_hop_latency
        return (energy, latency)

    # --- Public API ---

    def point_to_point_cost(self, data_bytes, src, dst):
        return self._cost_from_link_loads(self._point_to_point_link_loads(data_bytes, src, dst))

    def broadcast_cost(self, data_bytes, src, dst_chips=None):
        if dst_chips is None:
            dst_chips = [i for i in range(self.num_chips) if i != src]
        return self._cost_from_link_loads(self._broadcast_link_loads(data_bytes, src, dst_chips))

    def allreduce_cost(self, data_bytes, participating_chips=None):
        if participating_chips is None:
            participating_chips = list(range(self.num_chips))
        return self._cost_from_link_loads(self._allreduce_link_loads(data_bytes, participating_chips))

    def reduce_scatter_cost(self, data_bytes, participating_chips=None):
        if participating_chips is None:
            participating_chips = list(range(self.num_chips))
        loads = self._allreduce_link_loads(data_bytes, participating_chips)
        return self._cost_from_link_loads({k: v / 2 for k, v in loads.items()})

    def allgather_cost(self, data_bytes, participating_chips=None):
        return self.reduce_scatter_cost(data_bytes, participating_chips)

    def compute_congested_cost(self, transfer_loads: list[LinkLoads]) -> tuple[float, float]:
        """Merge link loads from multiple concurrent transfers, return congested cost."""
        merged: LinkLoads = defaultdict(float)
        for loads in transfer_loads:
            for link, b in loads.items():
                merged[link] += b
        return self._cost_from_link_loads(dict(merged))

    def summary(self) -> dict:
        degree = self.adj_matrix.sum(axis=1)
        return {
            "num_chips": self.num_chips, "diameter": self.diameter,
            "avg_hops": round(self.avg_hops, 2),
            "bisection_bandwidth_TB_s": round(self.bisection_bandwidth / 1e12, 2),
            "link_bandwidth_GB_s": round(self.link_bandwidth / 1e9, 2),
            "energy_per_bit_per_hop_pJ": round(self.energy_per_bit_per_hop * 1e12, 2),
            "topology_type": type(self).__name__,
            "min_degree": int(degree.min()), "max_degree": int(degree.max()),
            "total_links": self.total_links,
        }


# --- Ring ---

@dataclass
class Ring(Topology):
    """Ring topology (degree 2). Optimal ring all-reduce, but broadcast bottlenecks on 2 links."""

    def __init__(self, num_chips, link_bandwidth, energy_per_bit_per_hop,
                 per_hop_latency=500e-9, full_duplex=True):
        super().__init__(_make_ring_adj(num_chips), link_bandwidth,
                         energy_per_bit_per_hop, per_hop_latency, full_duplex)

    def _allreduce_link_loads(self, data_bytes, participating_chips=None):
        n = self.num_chips if participating_chips is None else len(participating_chips)
        if n <= 1:
            return {}
        bpl = 2 * (n - 1) * data_bytes / n
        return {(i, (i + 1) % self.num_chips): bpl for i in range(self.num_chips)}

    def _broadcast_link_loads(self, data_bytes, src, dst_chips):
        if not dst_chips:
            return {}
        n = self.num_chips
        loads: LinkLoads = defaultdict(float)
        half = n // 2
        for k in range(half):
            loads[((src + k) % n, (src + k + 1) % n)] += data_bytes
        for k in range(n - 1 - half):
            loads[((src - k) % n, (src - k - 1) % n)] += data_bytes
        return dict(loads)


# --- 2D Mesh ---

@dataclass
class Mesh2D(Topology):
    dims: tuple[int, int] = (1, 1)

    def __init__(self, dims, link_bandwidth, energy_per_bit_per_hop,
                 per_hop_latency=500e-9, full_duplex=True):
        object.__setattr__(self, "dims", dims)
        super().__init__(_make_mesh_adj(dims), link_bandwidth,
                         energy_per_bit_per_hop, per_hop_latency, full_duplex)


# --- 3D Mesh ---

@dataclass
class Mesh3D(Topology):
    """3D mesh with dimension-ordered routing. Used by small TPU v4 slices (dims < 4)."""
    dims: tuple[int, int, int] = (1, 1, 1)

    def __init__(self, dims, link_bandwidth, energy_per_bit_per_hop,
                 per_hop_latency=500e-9, full_duplex=True):
        object.__setattr__(self, "dims", dims)
        super().__init__(_make_mesh_adj(dims), link_bandwidth,
                         energy_per_bit_per_hop, per_hop_latency, full_duplex)

    def _allreduce_link_loads(self, data_bytes, participating_chips=None):
        dx, dy, dz = self.dims
        n = dx * dy * dz
        if n <= 1:
            return {}
        loads: LinkLoads = defaultdict(float)
        for flat_idx in range(n):
            coords = _flat_to_coords(flat_idx, self.dims)
            for dim, d_size in enumerate([dx, dy, dz]):
                if coords[dim] + 1 < d_size:
                    neighbor = list(coords)
                    neighbor[dim] += 1
                    j = _chip_index_nd(tuple(neighbor), self.dims)
                    bpl = 2 * (d_size - 1) * data_bytes / d_size
                    loads[(flat_idx, j)] += bpl
                    loads[(j, flat_idx)] += bpl
        return dict(loads)

    def _broadcast_link_loads(self, data_bytes, src, dst_chips):
        dx, dy, dz = self.dims
        loads: dict[tuple[int,int], float] = {}
        sc = _flat_to_coords(src, self.dims)

        # X from src
        for x in range(dx):
            if x == sc[0]: continue
            cx = sc[0]
            while cx != x:
                nx = cx + (1 if x > cx else -1)
                f, t = list(sc), list(sc); f[0], t[0] = cx, nx
                loads[(_chip_index_nd(tuple(f), self.dims), _chip_index_nd(tuple(t), self.dims))] = \
                    loads.get((_chip_index_nd(tuple(f), self.dims), _chip_index_nd(tuple(t), self.dims)), 0) + data_bytes
                cx = nx
        # Y from each X-row chip
        for x in range(dx):
            for y in range(dy):
                if y == sc[1]: continue
                cy = sc[1]
                while cy != y:
                    ny = cy + (1 if y > cy else -1)
                    f, t = [x, cy, sc[2]], [x, ny, sc[2]]
                    lk = (_chip_index_nd(tuple(f), self.dims), _chip_index_nd(tuple(t), self.dims))
                    loads[lk] = loads.get(lk, 0) + data_bytes
                    cy = ny
        # Z from each XY-plane chip
        for x in range(dx):
            for y in range(dy):
                for z in range(dz):
                    if z == sc[2]: continue
                    cz = sc[2]
                    while cz != z:
                        nz = cz + (1 if z > cz else -1)
                        f, t = [x, y, cz], [x, y, nz]
                        lk = (_chip_index_nd(tuple(f), self.dims), _chip_index_nd(tuple(t), self.dims))
                        loads[lk] = loads.get(lk, 0) + data_bytes
                        cz = nz
        return loads


# --- 3D Torus (primary TPU v4 topology) ---

@dataclass
class Torus3D(Topology):
    """
    3D torus with per-dimension ring collectives.
    TPU v4 uses this for slices where all dims >= 4.
    6 links/chip spread broadcast and all-reduce across 3 independent dimensions.
    """
    dims: tuple[int, int, int] = (1, 1, 1)

    def __init__(self, dims, link_bandwidth, energy_per_bit_per_hop,
                 per_hop_latency=500e-9, full_duplex=True):
        object.__setattr__(self, "dims", dims)
        super().__init__(_make_torus_adj(dims), link_bandwidth,
                         energy_per_bit_per_hop, per_hop_latency, full_duplex)

    def _allreduce_link_loads(self, data_bytes, participating_chips=None):
        dx, dy, dz = self.dims
        n = dx * dy * dz
        if n <= 1:
            return {}
        loads: LinkLoads = defaultdict(float)
        for flat_idx in range(n):
            coords = _flat_to_coords(flat_idx, self.dims)
            for dim, d_size in enumerate([dx, dy, dz]):
                neighbor = list(coords)
                neighbor[dim] = (coords[dim] + 1) % d_size
                j = _chip_index_nd(tuple(neighbor), self.dims)
                if j != flat_idx:
                    loads[(flat_idx, j)] += 2 * (d_size - 1) * data_bytes / d_size
        return dict(loads)

    def _broadcast_link_loads(self, data_bytes, src, dst_chips):
        dx, dy, dz = self.dims
        loads: dict[tuple[int,int], float] = {}
        sc = _flat_to_coords(src, self.dims)

        def _ring_bcast_1d(dim_idx, dim_size, sources):
            for base in sources:
                center = base[dim_idx]
                half = dim_size // 2
                for step in range(1, half + 1):
                    f, t = list(base), list(base)
                    f[dim_idx] = (center + step - 1) % dim_size
                    t[dim_idx] = (center + step) % dim_size
                    lk = (_chip_index_nd(tuple(f), self.dims), _chip_index_nd(tuple(t), self.dims))
                    loads[lk] = loads.get(lk, 0) + data_bytes
                for step in range(1, dim_size - 1 - half + 1):
                    f, t = list(base), list(base)
                    f[dim_idx] = (center - step + 1) % dim_size
                    t[dim_idx] = (center - step) % dim_size
                    lk = (_chip_index_nd(tuple(f), self.dims), _chip_index_nd(tuple(t), self.dims))
                    loads[lk] = loads.get(lk, 0) + data_bytes

        _ring_bcast_1d(0, dx, [list(sc)])
        _ring_bcast_1d(1, dy, [[x, sc[1], sc[2]] for x in range(dx)])
        _ring_bcast_1d(2, dz, [[x, y, sc[2]] for x in range(dx) for y in range(dy)])
        return loads


# --- Custom topology ---

@dataclass
class Custom(Topology):
    """Arbitrary adjacency matrix. Uses generic shortest-path routing."""
    def __init__(self, adj_matrix, link_bandwidth, energy_per_bit_per_hop,
                 per_hop_latency=500e-9, full_duplex=True):
        super().__init__(adj_matrix, link_bandwidth, energy_per_bit_per_hop,
                         per_hop_latency, full_duplex)
