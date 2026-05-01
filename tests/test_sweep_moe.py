import pytest

from network_topology.topology import Ring
from network_topology.tpu_v4 import ICI_ENERGY_PER_BIT_PER_HOP, ICI_LINK_BW_UNIDIR
from sweep_moe import (
    MoeWorkload,
    build_moe_phases,
    evaluate_moe_workload,
    expert_source_bytes,
    make_batch_scaling_workloads,
    place_experts,
    placement_hop_objective,
)


def _ring(num_chips):
    return Ring(
        num_chips=num_chips,
        link_bandwidth=ICI_LINK_BW_UNIDIR,
        energy_per_bit_per_hop=ICI_ENERGY_PER_BIT_PER_HOP,
        per_hop_latency=0.0,
    )


def test_moe_expected_expert_source_bytes_match_logical_dispatch_bytes():
    workload = MoeWorkload(
        name="tiny",
        num_chips=4,
        num_experts=2,
        tokens_per_chip=10,
        hidden_size=8,
        bytes_per_value=2,
        top_k=2,
        local_fraction=0.5,
    )

    expert_sources = expert_source_bytes(workload)
    total_bytes = sum(
        data_bytes
        for source_bytes in expert_sources.values()
        for data_bytes in source_bytes.values()
    )

    assert total_bytes == pytest.approx(workload.logical_dispatch_bytes)


def test_moe_topology_aware_placement_reduces_locality_objective_on_ring():
    workload = MoeWorkload(
        name="local",
        num_chips=8,
        num_experts=4,
        tokens_per_chip=1,
        hidden_size=1,
        bytes_per_value=1,
        top_k=1,
        local_fraction=1.0,
    )
    topology = _ring(8)
    expert_sources = expert_source_bytes(workload)

    clustered = place_experts(workload, topology, "clustered", expert_sources)
    topology_aware = place_experts(
        workload, topology, "topology_aware", expert_sources
    )

    assert placement_hop_objective(
        topology, expert_sources, topology_aware
    ) < placement_hop_objective(topology, expert_sources, clustered)


def test_moe_phases_skip_local_expert_traffic():
    workload = MoeWorkload(
        name="all-local",
        num_chips=4,
        num_experts=4,
        tokens_per_chip=1,
        hidden_size=1,
        bytes_per_value=1,
        top_k=1,
        local_fraction=1.0,
    )
    expert_sources = expert_source_bytes(workload)
    placement = {expert: expert for expert in range(workload.num_experts)}

    phases, traffic = build_moe_phases(workload, expert_sources, placement)

    assert traffic["logical_dispatch_bytes"] == 4.0
    assert traffic["local_dispatch_bytes"] == 4.0
    assert traffic["remote_dispatch_bytes"] == 0.0
    assert phases[0].transfers == []
    assert phases[1].transfers == []


def test_moe_evaluation_accounts_for_dispatch_and_combine_bytes():
    workload = MoeWorkload(
        name="remote",
        num_chips=4,
        num_experts=1,
        tokens_per_chip=1,
        hidden_size=1,
        bytes_per_value=1,
        top_k=1,
        local_fraction=1.0,
    )
    topology = _ring(4)

    result = evaluate_moe_workload(workload, "Ring 4", topology, "clustered")
    traffic = result["traffic_summary"]

    assert traffic["logical_dispatch_bytes"] == 4.0
    assert traffic["local_dispatch_bytes"] == 1.0
    assert traffic["remote_dispatch_bytes"] == 3.0
    assert traffic["remote_combine_bytes"] == 3.0
    assert result["total_network_bytes"] == 6.0
    assert [phase["phase"] for phase in result["per_phase"]] == ["dispatch", "combine"]
    assert result["payload_throughput_bytes_per_s"] > 0
    assert result["payload_link_efficiency"] > 0


def test_batch_scaling_workloads_are_full_chip_and_monotonic():
    workloads = make_batch_scaling_workloads()

    assert [workload.tokens_per_chip for workload in workloads] == sorted(
        workload.tokens_per_chip for workload in workloads
    )
    assert all(workload.num_chips == 64 for workload in workloads)
    assert all(workload.experiment == "batch_scaling" for workload in workloads)
