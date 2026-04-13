from types import SimpleNamespace

from network_topology.cost_model import CollectiveType

from sweep_matmuls import (
    _annotate_collective_decisions,
    _damp_network_proxies,
    _infer_matmul_collectives,
    _initial_network_proxies,
    _updated_network_proxies,
)


def _tensor_info(name, ranks, sharded, is_output=False):
    return {
        "name": name,
        "tensor_ranks": set(ranks),
        "chip_sharded_ranks": set(sharded),
        "is_output": is_output,
    }


def test_local_matmul_needs_no_collective():
    decisions = _infer_matmul_collectives(
        "Matmul0",
        {
            "A": _tensor_info("A", {"m", "k"}, {"m"}),
            "B": _tensor_info("B", {"k", "n"}, set()),
            "C": _tensor_info("C", {"m", "n"}, {"m"}, is_output=True),
        },
        {"A": 128.0, "B": 128.0, "C": 0.0},
        {"A": 0.0, "B": 0.0, "C": 128.0},
        {"A": 64.0, "B": 64.0, "C": 64.0},
    )

    assert decisions[0]["collective_type"] is None
    assert decisions[0]["proxy_action"] is None


def test_one_contracting_shard_uses_allgather():
    decisions = _infer_matmul_collectives(
        "Matmul0",
        {
            "A": _tensor_info("A", {"m", "k"}, {"k"}),
            "B": _tensor_info("B", {"k", "n"}, set()),
            "C": _tensor_info("C", {"m", "n"}, set(), is_output=True),
        },
        {"A": 128.0, "B": 128.0, "C": 0.0},
        {"A": 0.0, "B": 0.0, "C": 128.0},
        {"A": 64.0, "B": 64.0, "C": 64.0},
    )

    assert decisions[0]["collective_type"] == CollectiveType.ALLGATHER.name
    assert decisions[0]["tensor_name"] == "A"
    assert decisions[0]["proxy_action"] == "read"


def test_shared_contracting_shard_uses_allreduce():
    decisions = _infer_matmul_collectives(
        "Matmul0",
        {
            "A": _tensor_info("A", {"m", "k"}, {"k"}),
            "B": _tensor_info("B", {"k", "n"}, {"k"}),
            "C": _tensor_info("C", {"m", "n"}, set(), is_output=True),
        },
        {"A": 128.0, "B": 128.0, "C": 0.0},
        {"A": 0.0, "B": 0.0, "C": 128.0},
        {"A": 64.0, "B": 64.0, "C": 64.0},
    )

    assert decisions[0]["collective_type"] == CollectiveType.ALLREDUCE.name
    assert decisions[0]["tensor_name"] == "C"
    assert decisions[0]["proxy_action"] == "write"


def test_shared_contracting_shard_uses_reduce_scatter_when_output_is_sharded():
    decisions = _infer_matmul_collectives(
        "Matmul0",
        {
            "A": _tensor_info("A", {"m", "k"}, {"k"}),
            "B": _tensor_info("B", {"k", "n"}, {"k"}),
            "C": _tensor_info("C", {"m", "n"}, {"m"}, is_output=True),
        },
        {"A": 128.0, "B": 128.0, "C": 0.0},
        {"A": 0.0, "B": 0.0, "C": 128.0},
        {"A": 64.0, "B": 64.0, "C": 64.0},
    )

    assert decisions[0]["collective_type"] == CollectiveType.REDUCE_SCATTER.name
    assert decisions[0]["proxy_action"] == "write"


def test_conflicting_output_shards_allgathers_smaller_input():
    decisions = _infer_matmul_collectives(
        "Matmul0",
        {
            "A": _tensor_info("A", {"m", "k"}, {"m"}),
            "B": _tensor_info("B", {"k", "n"}, {"n"}),
            "C": _tensor_info("C", {"m", "n"}, {"m"}, is_output=True),
        },
        {"A": 512.0, "B": 128.0, "C": 0.0},
        {"A": 0.0, "B": 0.0, "C": 128.0},
        {"A": 64.0, "B": 64.0, "C": 64.0},
    )

    assert decisions[0]["collective_type"] == CollectiveType.ALLGATHER.name
    assert decisions[0]["tensor_name"] == "B"
    assert decisions[0]["proxy_action"] == "read"


def test_proxy_update_reduces_per_transfer_estimates_to_read_write_scalars():
    decisions = [
        {
            "einsum": "Matmul0",
            "tensor_name": "A",
            "collective_type": CollectiveType.ALLGATHER.name,
            "proxy_action": "read",
            "data_bytes": 100.0,
        },
        {
            "einsum": "Matmul0",
            "tensor_name": "C",
            "collective_type": CollectiveType.ALLREDUCE.name,
            "proxy_action": "write",
            "data_bytes": 50.0,
        },
    ]
    network_result = SimpleNamespace(
        total_latency=30.0,
        per_transfer=[
            {"tensor": "Matmul0:A", "energy": 200.0, "latency": 10.0},
            {"tensor": "Matmul0:C", "energy": 100.0, "latency": 20.0},
        ],
    )

    annotated = _annotate_collective_decisions(decisions, network_result)
    updated = _updated_network_proxies(_initial_network_proxies(), annotated, network_result)

    assert annotated[0]["estimated_network_latency"] == 15.0
    assert annotated[1]["estimated_network_latency"] == 30.0
    assert updated["NETWORK_READ_ENERGY"] == 2.0
    assert updated["NETWORK_READ_LATENCY"] == 0.15
    assert updated["NETWORK_WRITE_ENERGY"] == 2.0
    assert updated["NETWORK_WRITE_LATENCY"] == 0.6


def test_damp_network_proxies_is_identity_when_disabled():
    old = {
        "NETWORK_READ_ENERGY": 1.0,
        "NETWORK_WRITE_ENERGY": 2.0,
        "NETWORK_READ_LATENCY": 3.0,
        "NETWORK_WRITE_LATENCY": 4.0,
    }
    new = {
        "NETWORK_READ_ENERGY": 10.0,
        "NETWORK_WRITE_ENERGY": 20.0,
        "NETWORK_READ_LATENCY": 30.0,
        "NETWORK_WRITE_LATENCY": 40.0,
    }

    assert _damp_network_proxies(old, new, use_damping=False) == new
