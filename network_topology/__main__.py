from __future__ import annotations

import argparse
import json
import logging

from .fixed_point import run_comm_fixed_point_loop
from .yaml_utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AccelForge communication fixed-point loop.")
    parser.add_argument("--arch", required=True, help="Path to the architecture YAML file.")
    parser.add_argument("--workload", required=True, help="Path to the workload YAML file.")
    parser.add_argument(
        "--topology-config",
        required=True,
        help="Path to a YAML/JSON topology configuration file.",
    )
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    parser.add_argument("--abs-tol", type=float, default=1e-9)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    topology_config = load_yaml(args.topology_config)
    result = run_comm_fixed_point_loop(
        arch_yaml_path=args.arch,
        workload_yaml_path=args.workload,
        topology_config=topology_config,
        max_iters=args.max_iters,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )
    print(json.dumps(result, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
