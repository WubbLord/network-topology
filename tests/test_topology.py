from __future__ import annotations

import unittest

from network_topology.topology import compute_average_hops, shortest_path_hops


class TopologyTests(unittest.TestCase):
    def test_fully_connected_average_hops(self) -> None:
        config = {"type": "fully_connected", "num_nodes": 8}
        self.assertEqual(compute_average_hops(config), 1.0)

    def test_ring_shortest_path_uses_wraparound(self) -> None:
        config = {"type": "ring", "num_nodes": 8}
        self.assertEqual(shortest_path_hops(0, 7, config), 1.0)
        self.assertEqual(shortest_path_hops(0, 4, config), 4.0)

    def test_mesh_average_hops_respects_src_dst_weights(self) -> None:
        config = {"type": "2d_mesh", "dims": [2, 2]}
        pairs = [{"src": (0, 0), "dst": (1, 1), "bytes": 8.0}]
        self.assertEqual(compute_average_hops(config, pairs), 2.0)


if __name__ == "__main__":
    unittest.main()
