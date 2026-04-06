"""Network topology modeling for distributed accelerator systems."""

from network_topology.topology import (
    Topology, Ring, Mesh2D, Mesh3D, Torus3D, Custom,
    _make_ring_adj, _make_mesh_adj, _make_torus_adj,
)
from network_topology.cost_model import (
    NetworkCostResult, NetworkTransfer, CollectiveType, compute_network_cost,
)
from network_topology.tpu_v4 import make_tpu_v4_topology
