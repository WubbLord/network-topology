from __future__ import annotations

import tempfile
import unittest

from network_topology import clear_mapping_runner, register_mapping_runner, run_comm_fixed_point_loop
from network_topology.yaml_utils import load_yaml, save_yaml


class FixedPointLoopTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_mapping_runner()

    def test_loop_converges_with_deterministic_mock_mapping(self) -> None:
        def fake_runner(arch_yaml_path: str, workload_yaml_path: str) -> dict[str, object]:
            arch = load_yaml(arch_yaml_path)
            logical = arch["architecture"]["subtrees"][0]
            assumed_cost = logical["attributes"]["energy_per_byte"]
            return {
                "mapping": {"tile_order": ["B", "K", "N"], "assumed_cost": assumed_cost},
                "total_energy": 42.0,
                "total_latency": 7.5,
                "component_actions": {
                    "ici_link": {
                        "bytes": 4096,
                        "count": 16,
                    }
                },
            }

        register_mapping_runner(fake_runner)

        arch = {
            "architecture": {
                "name": "root",
                "subtrees": [
                    {
                        "name": "logical_interconnect",
                        "attributes": {"energy_per_byte": 0.0, "latency_per_byte": 0.0},
                    }
                ],
            }
        }
        workload = {"problem": {"name": "dummy"}}
        topology = {
            "type": "fully_connected",
            "num_nodes": 4,
            "link_bandwidth_bytes_per_s": 1_000_000_000.0,
            "per_hop_latency_s": 1e-9,
            "energy_per_byte_hop_j": 2e-12,
            "effective_cost_mode": "per_byte",
            "patch": {"target_name": "logical_interconnect", "style": "toll_component"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            arch_path = f"{tmpdir}/arch.yaml"
            workload_path = f"{tmpdir}/workload.yaml"
            save_yaml(arch, arch_path)
            save_yaml(workload, workload_path)

            result = run_comm_fixed_point_loop(
                arch_yaml_path=arch_path,
                workload_yaml_path=workload_path,
                topology_config=topology,
                max_iters=5,
            )

        self.assertTrue(result["converged"])
        self.assertGreaterEqual(len(result["iterations"]), 2)
        self.assertAlmostEqual(result["effective_cost"]["energy_per_byte"], 2e-12)
        self.assertAlmostEqual(result["comm_estimate"]["total_remote_bytes"], 4096.0)


if __name__ == "__main__":
    unittest.main()
