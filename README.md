## Overview

This repository implements the outer communication fixed-point loop described in [PROMPT.md](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/PROMPT.md). The code treats AccelForge as the local mapper and memory/computation model, then wraps it with an external communication model that estimates inter-device traffic, computes network cost, writes back an updated logical communication cost, and repeats until the assumed cost and observed cost agree.

The implementation is intentionally a research prototype:

- It is modular and type-annotated.
- It uses defensive extraction because mapper outputs often vary across versions and experiments.
- It leaves the actual AccelForge invocation pluggable, because this repo does not ship the AccelForge package itself.
- It keeps the original architecture YAML untouched and patches a temporary copy per iteration.

## Environment

This repo is set up to use the local virtual environment already present at `.venv`.

Install dependencies into that environment with:

```bash
./.venv/bin/pip install -r requirements.txt
```

Run tests with:

```bash
./.venv/bin/python -m unittest discover -s tests
```

## Code Layout

The implementation lives in the `network_topology/` package.

- [network_topology/__init__.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/network_topology/__init__.py)
  Re-exports the public API so downstream scripts can import the main driver and helper functions from one place.
- [network_topology/__main__.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/network_topology/__main__.py)
  Small CLI wrapper. It loads a topology config file, runs the fixed-point loop, and prints JSON.
- [network_topology/fixed_point.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/network_topology/fixed_point.py)
  The main orchestration layer. It owns the per-iteration loop, convergence checks, damping, temporary architecture patch files, logging, and the final return payload.
- [network_topology/accelforge_adapter.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/network_topology/accelforge_adapter.py)
  Adapter boundary for running the mapper. You can either register a Python callback with `register_mapping_runner(...)` or point `ACCELFORGE_MAPPER_CMD` at an external command template.
- [network_topology/mapping.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/network_topology/mapping.py)
  Defensive mapper-result normalization. It extracts total energy, total latency, component activity, tensor metadata, the mapping structure, and a stable hash-like mapping signature.
- [network_topology/comms.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/network_topology/comms.py)
  Communication inference and cost modeling. It infers remote traffic from mapping output, computes average hop counts from the selected topology, estimates byte-hops, energy, latency, and derives effective per-byte or per-access writeback cost.
- [network_topology/topology.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/network_topology/topology.py)
  Shortest-path and average-hop calculations for `fully_connected`, `ring`, `2d_mesh`, `3d_mesh`, and `torus`.
- [network_topology/yaml_utils.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/network_topology/yaml_utils.py)
  YAML load/save helpers, recursive node lookup, nested field patching, and logical communication cost writeback.
- [network_topology/common.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/network_topology/common.py)
  Shared utilities for recursive object-to-dict conversion, key normalization, numeric coercion, deep key search, and stable signatures.

The test coverage lives under `tests/`.

- [tests/test_topology.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/tests/test_topology.py)
  Validates hop calculations.
- [tests/test_fixed_point.py](/Users/zhangbr/Documents/MIT/6.5931 Hardware for DL/network-topology/tests/test_fixed_point.py)
  Runs the loop end-to-end with a deterministic fake mapper to verify convergence and writeback behavior.

## How The Loop Works

The central function is:

```python
run_comm_fixed_point_loop(
    arch_yaml_path: str,
    workload_yaml_path: str,
    topology_config: dict,
    max_iters: int = 10,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-9,
) -> dict
```

Its control flow is:

1. Load the original architecture YAML.
2. Resolve the initial assumed communication cost.
3. Patch a temporary copy of the architecture with that assumed cost.
4. Run the mapper on the patched architecture and the workload.
5. Extract mapping-level stats from the mapper result.
6. Infer inter-device traffic from the mapping stats.
7. Compute topology-aware network energy and latency.
8. Convert that network estimate into an effective logical communication cost.
9. Optionally apply damping.
10. Check convergence against the previous iteration and repeat if needed.

The loop returns:

- `final_mapping`
- `final_mapping_stats`
- `iterations`
- `effective_cost`
- `comm_estimate`
- `final_patched_architecture`
- `converged`

## Mapper Boundary

This repo does not assume a hard-coded AccelForge Python API. Instead, the mapper boundary is explicit.

### Option 1: Register a Python runner

Use this when you already have an in-process wrapper around AccelForge:

```python
from network_topology import register_mapping_runner, run_comm_fixed_point_loop


def my_runner(arch_yaml_path: str, workload_yaml_path: str):
    # Replace this with your AccelForge call.
    return {
        "mapping": {"tile_order": ["B", "K", "N"]},
        "total_energy": 1.23,
        "total_latency": 4.56,
        "component_actions": {
            "ici_link": {"bytes": 8192, "count": 32},
        },
    }


register_mapping_runner(my_runner)
result = run_comm_fixed_point_loop("arch.yaml", "workload.yaml", topology_config={...})
```

### Option 2: Use an external command

Set `ACCELFORGE_MAPPER_CMD` to a shell template. The template must contain `{arch}` and `{workload}` placeholders. The command can either:

- print JSON to stdout, or
- print a path to a JSON/YAML result file

Example:

```bash
export ACCELFORGE_MAPPER_CMD='python run_accelforge.py --arch {arch} --workload {workload}'
```

## Expected Mapper Output Shape

The extraction code is defensive and does not require one exact schema, but it works best when the mapper returns some subset of these fields:

```python
{
    "mapping": {...},
    "total_energy": 0.0,
    "total_latency": 0.0,
    "component_actions": {
        "ici_link": {
            "bytes": 12345,
            "count": 67,
            "src": 0,
            "dst": 3,
        }
    },
    "tensors": {
        "weights": {
            "size_bytes": 4096,
            "accesses": 8,
            "shard_count": 4,
        }
    }
}
```

The code first tries direct signals such as `remote_bytes`, `byte_hops`, or explicit `src`/`dst` pairs. If those are missing, it falls back to component-level heuristics, and then tensor partitioning heuristics.

## Topology Model

Supported topology types:

- `fully_connected`
- `ring`
- `2d_mesh`
- `3d_mesh`
- `torus`

Relevant fields in `topology_config`:

```yaml
type: 3d_mesh
dims: [4, 4, 4]
link_bandwidth_bytes_per_s: 2.5e11
per_hop_latency_s: 5.0e-9
energy_per_byte_hop_j: 1.0e-12
default_message_size_bytes: 262144
effective_cost_mode: per_byte
damping: 1.0
patch:
  target_name: logical_interconnect
  style: toll_component
  energy_path: attributes.energy_per_byte
  latency_path: attributes.latency_per_byte
collective:
  type: allreduce
  algorithm: ring
  participants: 64
```

How those fields are used:

- `type` and `dims` or `num_nodes` determine the graph used for average-hop calculations.
- `link_bandwidth_bytes_per_s` determines the serialization term in network latency.
- `per_hop_latency_s` determines the propagation term in network latency.
- `energy_per_byte_hop_j` converts byte-hops into network energy.
- `default_message_size_bytes` is only used when the mapper does not provide a message count.
- `effective_cost_mode` selects per-byte versus per-access writeback.
- `damping` mixes old and new costs to reduce oscillation.
- `collective` enables a lightweight collective-specific adjustment. Right now the specialized path is implemented for ring allreduce.

## Architecture Patching

`patch_logical_comm_cost(...)` patches one named node inside the architecture YAML.

The patch process is:

1. Find the named component with `find_node_by_name(...)`.
2. Resolve the energy and latency field paths.
3. Write the current effective cost into those fields.

The default target name is `logical_interconnect`, but you should usually set this explicitly in `topology_config["patch"]`.

Two styles are recognized:

- `logical_shared_memory`
  Defaults to `attributes.energy_per_byte` and `attributes.latency_per_byte` for per-byte mode, or the analogous per-access fields.
- `toll_component`
  Defaults to per-byte fields for per-byte mode and `attributes.energy_per_traversal` / `attributes.latency_per_traversal` for per-access mode.

If your AccelForge schema uses different field names, set `energy_path` and `latency_path` directly.

## Mapping Extraction Details

`extract_mapping_stats(...)` takes arbitrary mapper output and normalizes it into a dictionary with these fields:

- `total_energy`
- `total_latency`
- `per_component_actions`
- `per_tensor_stats`
- `mapping_structure`
- `mapping_signature`
- `raw`
- `notes`

Important behavior:

- It converts dataclasses, objects, lists, and mappings into plain Python structures.
- It searches for common aliases such as `best_result`, `best_mapping`, `component_actions`, and `tensor_stats`.
- It computes a stable `mapping_signature` by hashing the normalized mapping structure.
- It preserves the raw normalized payload so later stages can still do deeper searches.

The signature is used by the fixed-point loop to avoid calling a cost sequence converged if the mapper is still changing schedules between iterations.

## Communication Inference Details

`infer_remote_traffic(...)` follows a simple layered strategy:

1. Direct inference:
   Look for explicit `remote_bytes`, `byte_hops`, `message_count`, or traffic pair data.
2. Component-action fallback:
   Treat actions on interconnect-like components such as `ici`, `network`, `link`, `logical`, or `remote` as communication.
3. Tensor fallback:
   If tensor metadata shows sharding, replication, or explicit remote fractions, estimate remote bytes from that information.

This intentionally favors usefulness over strict schema assumptions. The returned payload includes a `notes` list that explains which inference path was used.

## Effective Cost Writeback

`compute_effective_comm_cost(...)` supports:

- `per_byte`
- `per_access`

Formulas:

- Per-byte:
  `energy_per_byte = total_network_energy / total_remote_bytes`
  `latency_per_byte = total_network_latency / total_remote_bytes`
- Per-access:
  `energy_per_access = total_network_energy / num_accesses`
  `latency_per_access = total_network_latency / num_accesses`

Zero denominators are clamped to zero and recorded in the returned `notes`.

## Logging And Iteration Records

The loop uses the Python `logging` module, not `print`. Each iteration records:

- iteration number
- assumed effective cost
- mapping signature
- mapper energy and latency
- remote bytes
- byte-hops
- network energy
- network latency
- computed next effective cost
- absolute and relative deltas
- convergence flags
- notes from extraction and inference

That per-iteration history is returned in `result["iterations"]`, which makes it easy to dump the full fixed-point trace for analysis.

## Example CLI Usage

If you have a mapper command wired up, you can run the package directly:

```bash
./.venv/bin/python -m network_topology \
  --arch path/to/arch.yaml \
  --workload path/to/workload.yaml \
  --topology-config path/to/topology.yaml
```

## Assumptions And Limitations

- The communication model is external and approximate by design.
- The mapper adapter is intentionally generic because the actual AccelForge API is not bundled here.
- The latency model reports aggregate network cost, not a full cycle-accurate overlap model.
- The fallback traffic inference heuristics are only as good as the metadata present in the mapper output.
- Only ring allreduce has a specialized collective adjustment today; other collective settings are accepted but use the generic path.

## Summary

This codebase gives you a clean prototype scaffold for the class project:

- a fixed-point driver loop,
- defensive mapping extraction,
- topology-aware network estimation,
- architecture writeback,
- test coverage for the core mechanics,
- and a pluggable mapper boundary so you can connect it to your actual AccelForge flow.
