import numpy as np
import pytest

from network_topology import compute_network_cost, make_tpu_v4_topology
from network_topology.cost_model import CollectiveType, NetworkTransfer
from network_topology.topology import Custom, Mesh3D, Ring, Torus3D


def test_make_tpu_v4_topology_selects_expected_topology_class():
    assert isinstance(make_tpu_v4_topology((4, 4, 4)), Torus3D)
    assert isinstance(make_tpu_v4_topology((2, 2, 2)), Mesh3D)
    assert isinstance(make_tpu_v4_topology((4, 4, 4), force_mesh=True), Mesh3D)


def test_compute_network_cost_sums_independent_transfer_latencies():
    topo = Custom(
        adj_matrix=np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        ),
        link_bandwidth=10.0,
        energy_per_bit_per_hop=1.0,
        per_hop_latency=0.0,
    )
    transfers = [
        NetworkTransfer("left", 10.0, CollectiveType.POINT_TO_POINT, src_chip=0, dst_chips=[1]),
        NetworkTransfer("right", 10.0, CollectiveType.POINT_TO_POINT, src_chip=1, dst_chips=[2]),
    ]

    result = compute_network_cost(topo, transfers)

    assert result.total_energy == 160.0
    assert result.total_latency == 2.0
    assert [t["latency"] for t in result.per_transfer] == [1.0, 1.0]
    assert [t["max_link_load"] for t in result.per_transfer] == [10.0, 10.0]
    assert result.total_network_bytes == 20.0
    assert result.energy_per_network_access == 8.0
    assert result.latency_per_network_access == 0.1


def test_reduce_scatter_and_allgather_are_half_of_allreduce_on_ring():
    topo = Ring(
        num_chips=4,
        link_bandwidth=1.0,
        energy_per_bit_per_hop=1.0,
        per_hop_latency=0.0,
    )

    allreduce_energy, allreduce_latency = topo.allreduce_cost(8.0)
    reduce_scatter_energy, reduce_scatter_latency = topo.reduce_scatter_cost(8.0)
    allgather_energy, allgather_latency = topo.allgather_cost(8.0)

    assert reduce_scatter_energy == allreduce_energy / 2
    assert reduce_scatter_latency == allreduce_latency / 2
    assert allgather_energy == reduce_scatter_energy
    assert allgather_latency == reduce_scatter_latency


def test_broadcast_tree_edges_carry_one_copy_each():
    topo = Custom(
        adj_matrix=np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        ),
        link_bandwidth=10.0,
        energy_per_bit_per_hop=1.0,
        per_hop_latency=0.0,
    )

    loads = topo._broadcast_link_loads(10.0, src=0, dst_chips=[1, 2])

    assert loads == {(0, 1): 10.0, (1, 2): 10.0}


def test_optimized_topologies_fall_back_for_partial_collectives():
    topo = Ring(
        num_chips=4,
        link_bandwidth=1.0,
        energy_per_bit_per_hop=1.0,
        per_hop_latency=0.0,
    )

    allreduce_loads = topo._allreduce_link_loads(8.0, participating_chips=[0, 1])
    broadcast_loads = topo._broadcast_link_loads(8.0, src=0, dst_chips=[1])

    assert allreduce_loads == {(0, 1): 8.0, (1, 0): 8.0}
    assert broadcast_loads == {(0, 1): 8.0}


def test_compute_network_cost_rejects_chips_outside_topology():
    topo = Ring(
        num_chips=4,
        link_bandwidth=1.0,
        energy_per_bit_per_hop=1.0,
        per_hop_latency=0.0,
    )

    with pytest.raises(ValueError, match="outside topology range"):
        compute_network_cost(
            topo,
            [
                NetworkTransfer(
                    "bad",
                    8.0,
                    CollectiveType.ALLREDUCE,
                    participating_chips=[0, 4],
                )
            ],
        )
