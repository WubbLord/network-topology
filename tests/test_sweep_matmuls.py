from types import SimpleNamespace

from network_topology.cost_model import CollectiveType

from sweep_matmuls import (
    DECOMPOSED_GPT3_MODEL,
    EVAL_CHIPS,
    GPT3_EINSUM_SPECS,
    SCALE,
    _build_milestone2_topology_summary,
    _annotate_collective_decisions,
    _chip_sharded_rank_vars,
    _damp_network_proxies,
    _decomposed_gpt3_config,
    _gpt3_einsum_workload_yaml,
    _is_decomposed_gpt3_workload,
    _estimate_actual_system_cost,
    _fallback_network_access_decisions,
    _infer_matmul_collectives,
    _initial_network_proxies,
    _network_transfer_from_decision,
    _summarize_mapping_costs,
    _summarize_actual_network,
    _updated_network_proxies,
)


class _FakeMapping:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_nodes_of_type(self, _node_type):
        return self.nodes


def _tensor_info(name, ranks, sharded, is_output=False):
    return {
        "name": name,
        "tensor_ranks": set(ranks),
        "chip_sharded_ranks": set(sharded),
        "is_output": is_output,
    }


def test_chip_sharded_rank_vars_reads_fused_spatial_annotations():
    from accelforge.frontend.mapping.mapping import Spatial

    spatial = Spatial(name="Chip", component="ChipArray", rank_variable={"m", "n"})
    spatial._einsum_to_rank_variables = {"E0": {"m"}, "E1": {"n"}}
    local_spatial = Spatial(name="Z", component="LocalBuffer", rank_variable="m")
    plain_spatial = Spatial(name="Chip", component="ChipArray", rank_variable="k")

    mapping = _FakeMapping([spatial, local_spatial, plain_spatial])

    assert _chip_sharded_rank_vars("E0", mapping, {"m", "k"}) == {"m", "k"}
    assert _chip_sharded_rank_vars("E1", mapping, {"m", "k"}) == {"k"}


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

    transfer = _network_transfer_from_decision(decisions[0])
    assert transfer.participating_chips == list(range(EVAL_CHIPS))


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

    assert annotated[0]["estimated_network_latency"] == 10.0
    assert annotated[1]["estimated_network_latency"] == 20.0
    assert updated["NETWORK_READ_ENERGY"] == 2.0
    assert updated["NETWORK_READ_LATENCY"] == 0.1
    assert updated["NETWORK_WRITE_ENERGY"] == 2.0
    assert updated["NETWORK_WRITE_LATENCY"] == 0.4


def test_fallback_network_access_decisions_use_already_scaled_bytes():
    tensor_infos = {
        "A": _tensor_info("A", {"m", "k"}, set()),
        "C": _tensor_info("C", {"m", "n"}, set(), is_output=True),
    }

    decisions = _fallback_network_access_decisions(
        "Matmul0",
        tensor_infos,
        {"A": 64.0, "C": 0.0},
        {"A": 0.0, "C": 32.0},
    )

    assert [decision["collective_type"] for decision in decisions] == [
        CollectiveType.ALLGATHER.name,
        CollectiveType.ALLREDUCE.name,
    ]
    assert [decision["data_bytes"] for decision in decisions] == [64.0, 32.0]
    assert [decision["proxy_action"] for decision in decisions] == ["read", "write"]

    raw_decisions = _fallback_network_access_decisions(
        "Matmul0",
        tensor_infos,
        {"A": 64.0, "C": 0.0},
        {"A": 0.0, "C": 0.0},
        read_collective_type=CollectiveType.BROADCAST,
    )
    assert raw_decisions[0]["collective_type"] == CollectiveType.BROADCAST.name
    assert raw_decisions[0]["data_bytes"] == 64.0


def test_decomposed_gpt3_workload_specs_are_independent_single_einsums():
    params = {
        "__decomposed_model": DECOMPOSED_GPT3_MODEL,
        "BATCH_SIZE": 1,
        "N_TOKENS": 8,
        "N_LAYERS": 3,
    }

    config = _decomposed_gpt3_config(params)
    names = [spec["name"] for spec in GPT3_EINSUM_SPECS]
    yaml_text = _gpt3_einsum_workload_yaml(GPT3_EINSUM_SPECS[0])

    assert _is_decomposed_gpt3_workload(params)
    assert config["hidden_dim"] == 96 * 128
    assert config["activation_bytes"] == 8 * 96 * 128
    assert config["n_layers"] == 3
    assert names == ["I", "V", "K", "Q", "QK", "QK_softmax", "AV", "Z", "FFA", "FFB"]
    assert all("output" in spec["renames"] for spec in GPT3_EINSUM_SPECS)
    assert 'einsum: "I[b, m, d] = I_in[b, m, d]"' in yaml_text
    assert "renames: {input: I_in, output: I}" in yaml_text


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


def test_mapping_cost_summary_scales_energy_and_separates_networkmemory():
    summary = _summarize_mapping_costs(
        7.0,
        {
            ("Matmul0", "MAC", "None", "compute"): 1.0,
            ("Matmul0", "NetworkMemory", "T1", "write"): 2.0,
            ("Matmul1", "GlobalBuffer", "T0", "read"): 3.0,
        },
        {
            ("Matmul0", "MAC"): 4.0,
            ("Matmul0", "NetworkMemory"): 5.0,
            ("Matmul1", "GlobalBuffer"): 6.0,
        },
    )

    assert summary["proxy_total_energy_scaled"] == 6.0 * SCALE
    assert summary["proxy_non_network_energy_scaled"] == 4.0 * SCALE
    assert summary["proxy_network_energy_scaled"] == 2.0 * SCALE
    assert summary["per_einsum"]["Matmul0"]["non_network_bottleneck_latency"] == 4.0
    assert summary["per_einsum"]["Matmul0"]["proxy_network_latency"] == 5.0
    assert summary["per_einsum"]["Matmul1"]["proxy_total_latency"] == 6.0


def test_actual_system_cost_replaces_proxy_network_costs():
    mapping_summary = {
        "proxy_non_network_energy_scaled": 100.0,
        "per_einsum": {
            "Matmul0": {
                "non_network_energy_scaled": 60.0,
                "proxy_network_energy_scaled": 10.0,
                "non_network_bottleneck_latency": 3.0,
                "proxy_network_latency": 7.0,
                "proxy_total_latency": 7.0,
            },
            "Matmul1": {
                "non_network_energy_scaled": 40.0,
                "proxy_network_energy_scaled": 20.0,
                "non_network_bottleneck_latency": 11.0,
                "proxy_network_latency": 2.0,
                "proxy_total_latency": 11.0,
            },
        },
    }
    network_summary = {
        "total_energy": 55.0,
        "total_latency": 13.0,
        "total_bytes": 200.0,
        "per_einsum": {
            "Matmul0": {
                "estimated_network_energy": 30.0,
                "assigned_network_latency": 5.0,
            },
            "Matmul1": {
                "estimated_network_energy": 25.0,
                "assigned_network_latency": 4.0,
            },
        },
    }

    summary = _estimate_actual_system_cost(mapping_summary, network_summary)

    assert summary["estimated_total_energy_scaled"] == 155.0
    assert summary["estimated_total_latency"] == 16.0
    assert summary["per_einsum"]["Matmul0"]["estimated_actual_total_latency"] == 5.0
    assert summary["per_einsum"]["Matmul1"]["estimated_actual_total_latency"] == 11.0


def test_milestone2_topology_summary_reports_required_deltas():
    decisions = [
        {
            "einsum": "Matmul0",
            "tensor_name": "T1",
            "collective_type": CollectiveType.ALLREDUCE.name,
            "proxy_action": "write",
            "data_bytes": 16.0,
            "chip_sharded_ranks": ["n0"],
        }
    ]
    network_result = SimpleNamespace(
        total_energy=12.0,
        total_latency=6.0,
        total_network_bytes=16.0,
        energy_per_network_access=0.75,
        latency_per_network_access=0.375,
        per_transfer=[
            {
                "tensor": "Matmul0:T1",
                "collective": "ALLREDUCE",
                "energy": 12.0,
                "latency": 6.0,
                "data_bytes": 16.0,
            }
        ],
    )
    annotated = _annotate_collective_decisions(decisions, network_result)
    mapping_summary = _summarize_mapping_costs(
        4.0,
        {
            ("Matmul0", "MAC", "None", "compute"): 1.0,
            ("Matmul0", "NetworkMemory", "T1", "write"): 1.0,
        },
        {
            ("Matmul0", "MAC"): 2.0,
            ("Matmul0", "NetworkMemory"): 4.0,
        },
    )
    network_summary = _summarize_actual_network(annotated, network_result)
    estimated_actual = _estimate_actual_system_cost(mapping_summary, network_summary)

    topology_result = {
        "feedback_converged": True,
        "final_network_proxies": {"NETWORK_WRITE_ENERGY": 0.75},
        "feedback_iterations": [
            {
                "iteration": 1,
                "input_network_proxies": {"NETWORK_WRITE_ENERGY": 0.1},
                "updated_network_proxies": {"NETWORK_WRITE_ENERGY": 0.75},
                "raw_relative_change": 1.0,
                "applied_relative_change": 1.0,
                "allgather_pct": 0.0,
                "mapping": {
                    "mapping_yaml_path": "mappings/workload/topology/iter_001.yaml",
                    "mapping_sha256": "aaa",
                    "cost_summary": mapping_summary,
                },
                "network": {"actual_summary": network_summary},
                "estimated_actual_system": estimated_actual,
                "collective_decisions": annotated,
            },
            {
                "iteration": 2,
                "input_network_proxies": {"NETWORK_WRITE_ENERGY": 0.75},
                "updated_network_proxies": {"NETWORK_WRITE_ENERGY": 0.75},
                "raw_relative_change": 0.0,
                "applied_relative_change": 0.0,
                "allgather_pct": 0.0,
                "mapping": {
                    "mapping_yaml_path": "mappings/workload/topology/iter_002.yaml",
                    "mapping_sha256": "bbb",
                    "cost_summary": mapping_summary,
                },
                "network": {"actual_summary": network_summary},
                "estimated_actual_system": estimated_actual,
                "collective_decisions": annotated,
            },
        ],
    }

    summary = _build_milestone2_topology_summary(topology_result)

    assert summary["baseline_without_network_model"]["iteration"] == 1
    assert summary["final_stabilized_mapping"]["iteration"] == 2
    assert summary["changes"]["mapping_hash_changed_first_to_final"] is True
    assert "energy" in summary["changes"]["without_network_proxy_to_first_actual"]
    assert len(summary["convergence_trace"]) == 2
