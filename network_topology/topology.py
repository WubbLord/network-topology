from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from math import prod
from typing import Any

from .common import coerce_float


def node_count(topology_config: Mapping[str, Any]) -> int:
    dims = _dims_for_topology(topology_config)
    if dims:
        return prod(dims)
    count = topology_config.get("num_nodes", topology_config.get("nodes"))
    if count is None:
        raise ValueError("Topology configuration must include either dims or num_nodes.")
    return int(count)


def shortest_path_hops(src: Any, dst: Any, topology_config: Mapping[str, Any]) -> float:
    if src == dst:
        return 0.0

    topology_type = str(topology_config.get("type", "fully_connected")).lower()
    if topology_type == "fully_connected":
        return 1.0

    if topology_type == "ring":
        size = node_count(topology_config)
        src_index = int(_node_to_linear_index(src, topology_config))
        dst_index = int(_node_to_linear_index(dst, topology_config))
        delta = abs(src_index - dst_index) % size
        return float(min(delta, size - delta))

    dims = _dims_for_topology(topology_config)
    src_coords = _node_to_coords(src, dims, topology_config)
    dst_coords = _node_to_coords(dst, dims, topology_config)

    if topology_type in {"2d_mesh", "3d_mesh"}:
        return float(sum(abs(a - b) for a, b in zip(src_coords, dst_coords)))
    if topology_type == "torus":
        return float(sum(min(abs(a - b), dim - abs(a - b)) for a, b, dim in zip(src_coords, dst_coords, dims)))

    raise ValueError(f"Unsupported topology type '{topology_type}'.")


def compute_average_hops(
    topology_config: Mapping[str, Any],
    src_dst_pairs: Sequence[Mapping[str, Any]] | None = None,
) -> float:
    if src_dst_pairs:
        weighted_hops = 0.0
        total_weight = 0.0
        for pair in src_dst_pairs:
            src = pair.get("src", pair.get("source"))
            dst = pair.get("dst", pair.get("destination"))
            if src is None or dst is None:
                continue
            weight = (
                coerce_float(pair.get("bytes"))
                or coerce_float(pair.get("weight"))
                or coerce_float(pair.get("count"))
                or 1.0
            )
            weighted_hops += shortest_path_hops(src, dst, topology_config) * weight
            total_weight += weight
        if total_weight > 0:
            return weighted_hops / total_weight

    nodes = list(_enumerate_nodes(topology_config))
    if len(nodes) <= 1:
        return 0.0
    if len(nodes) <= 4096:
        total_hops = 0.0
        total_pairs = 0
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                total_hops += shortest_path_hops(src, dst, topology_config)
                total_pairs += 1
        return total_hops / total_pairs

    topology_type = str(topology_config.get("type", "fully_connected")).lower()
    if topology_type == "fully_connected":
        return 1.0
    if topology_type == "ring":
        return float(node_count(topology_config)) / 4.0

    dims = _dims_for_topology(topology_config)
    if topology_type in {"2d_mesh", "3d_mesh"}:
        return sum(((dim * dim) - 1) / (3 * dim) for dim in dims if dim > 0)
    if topology_type == "torus":
        return sum(dim / 4.0 for dim in dims)

    raise ValueError(f"Unsupported topology type '{topology_type}'.")


def _dims_for_topology(topology_config: Mapping[str, Any]) -> tuple[int, ...]:
    dims = topology_config.get("dims")
    if dims is None:
        return ()
    return tuple(int(dim) for dim in dims)


def _enumerate_nodes(topology_config: Mapping[str, Any]) -> list[Any]:
    topology_type = str(topology_config.get("type", "fully_connected")).lower()
    if topology_type in {"2d_mesh", "3d_mesh", "torus"}:
        dims = _dims_for_topology(topology_config)
        return list(itertools.product(*(range(dim) for dim in dims)))
    return list(range(node_count(topology_config)))


def _node_to_linear_index(node: Any, topology_config: Mapping[str, Any]) -> int:
    coords = topology_config.get("node_coordinates", {})
    if isinstance(node, str) and node in coords:
        node = coords[node]

    if isinstance(node, int):
        return node
    if isinstance(node, str) and node.isdigit():
        return int(node)

    dims = _dims_for_topology(topology_config)
    coord_tuple = _node_to_coords(node, dims, topology_config)
    linear_index = 0
    stride = 1
    for coord, dim in zip(reversed(coord_tuple), reversed(dims)):
        linear_index += coord * stride
        stride *= dim
    return linear_index


def _node_to_coords(node: Any, dims: Sequence[int], topology_config: Mapping[str, Any]) -> tuple[int, ...]:
    coords = topology_config.get("node_coordinates", {})
    if isinstance(node, str) and node in coords:
        node = coords[node]

    if isinstance(node, Mapping):
        if "coords" in node:
            node = node["coords"]
        elif "coordinate" in node:
            node = node["coordinate"]

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        tuple_node = tuple(int(value) for value in node)
        if len(tuple_node) != len(dims):
            raise ValueError(f"Expected node coordinates with {len(dims)} dimensions, got {tuple_node}.")
        return tuple_node

    if isinstance(node, str) and "," in node:
        tuple_node = tuple(int(fragment.strip()) for fragment in node.split(","))
        if len(tuple_node) != len(dims):
            raise ValueError(f"Expected node coordinates with {len(dims)} dimensions, got {tuple_node}.")
        return tuple_node

    linear_index = int(node)
    coords_out = []
    remaining = linear_index
    for dim in reversed(dims):
        coords_out.append(remaining % dim)
        remaining //= dim
    return tuple(reversed(coords_out))
