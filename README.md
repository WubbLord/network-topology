# Network Topology

`network-topology` models communication cost for distributed accelerator workloads, with an emphasis on how the physical interconnect shape changes latency and energy once link congestion is taken into account.

At a high level, the repo does two things:

1. It defines a small topology/cost-model library for rings, meshes, tori, and custom graphs.
2. It plugs that library into an AccelForge mapping flow, so a mapped workload can be re-evaluated on multiple physical topologies.

The core idea is simple: route every transfer onto physical links, merge the byte load from all concurrent transfers, and let the most loaded link determine latency.

## What Lives Where

| Path | Purpose |
| --- | --- |
| `network_topology/topology.py` | Topology definitions, routing logic, and congestion-aware latency/energy calculation |
| `network_topology/cost_model.py` | Transfer dataclasses and the public `compute_network_cost()` entrypoint |
| `network_topology/tpu_v4.py` | TPU v4 constants and a convenience factory for TPU-like topologies |
| `accelforge_configs/tpu_v4_distributed_1d.yaml` | AccelForge architecture spec used to map workloads quickly on a simplified 1D chip array |
| `sweep_gpt3.py` | End-to-end script that maps workloads with AccelForge, extracts network traffic, and evaluates multiple topologies |
| `tests/` | Local pytest coverage for the topology and cost-model behavior |
| `EXPERIMENTS.md` | Example results and higher-level conclusions |

## How The Repo Works

There are two layers in the workflow.

### 1. Mapping and traffic extraction

The sweep script uses AccelForge to map a workload onto a distributed TPU v4-like architecture:

- It loads the local architecture spec from `accelforge_configs/tpu_v4_distributed_1d.yaml`.
- It loads workload YAMLs from an AccelForge checkout.
- It maps on `MAP_CHIPS = 2` chips first, because that search is much faster than mapping directly on 64 chips.
- It reads AccelForge's per-component, per-tensor action counts and filters out only `NetworkMemory` traffic.

In `sweep_gpt3.py`, the extracted traffic is converted into this repo's transfer model as follows:

- `NetworkMemory read` becomes a `BROADCAST`.
- `NetworkMemory write` becomes an `ALLREDUCE`.
- The transferred bytes are multiplied by `SCALE = EVAL_CHIPS / MAP_CHIPS`, so traffic found on the 2-chip mapping is projected to the 64-chip evaluation size.

This means the sweep is not trying to re-map the workload for every topology. It maps once with AccelForge, turns the mapped network actions into abstract transfers, and then replays those transfers on each topology model.

### 2. Topology-aware network replay

Once the script has a list of `NetworkTransfer` objects, `compute_network_cost()` in `network_topology/cost_model.py` takes over:

1. Each transfer is routed onto physical links.
2. The per-link byte loads from all transfers are merged.
3. Energy is computed from total bit-hops.
4. Latency is computed from the bottleneck link load plus a hop-dependent term.

The routing rules depend on topology:

- `Ring` uses explicit ring formulas for broadcast and all-reduce.
- `Mesh3D` uses dimension-ordered behavior.
- `Torus3D` spreads traffic across per-dimension ring collectives.
- `Custom` falls back to shortest-path routing from the base `Topology` class.

The base cost function is:

- `energy = sum(bytes_on_link) * 8 * energy_per_bit_per_hop`
- `latency = max(bytes_on_link) / link_bandwidth + diameter * per_hop_latency`

The important modeling choice is that energy depends on total routed traffic, while latency depends on the busiest link after all concurrent transfers are merged.

## The Iterative Convergence/Search Loop

This repo does not contain its own iterative numerical solver. The iterative search loop lives in AccelForge's mapper, and this repo consumes the mapper output.

That distinction matters:

- `network_topology/*` is a deterministic post-mapping network model.
- `sweep_gpt3.py` is a driver that calls the external mapper once per workload.
- AccelForge's Fast and Fusiest Mapper (FFM) is the part that performs the expensive iterative search.

From this repo's point of view, the end-to-end loop is:

1. Build an AccelForge `Spec` from the local arch YAML plus a workload YAML.
2. Call `map_workload_to_arch(spec)`.
3. Let AccelForge search for Pareto-optimal mappings.
4. Extract `NetworkMemory` reads and writes from the chosen mapping.
5. Convert those actions into `NetworkTransfer` objects.
6. Replay the same transfer list on each candidate topology.
7. Compare congested network latency across topologies.

If you mean the actual iterative search inside `map_workload_to_arch()`, AccelForge's documented FFM flow is:

1. Generate partial-mapping templates ("pmappings") for each Einsum.
2. Fill those templates with tile shapes and loop bounds.
3. Prune non-Pareto-optimal candidates as the search proceeds.
4. Join compatible partial mappings across Einsums into full mappings.

So the "convergence" behavior here is really a repeated search-and-prune loop over mapping candidates, not a fixed-point iteration. This repo treats that loop as an upstream black box and focuses on what happens after the mapping has already decided how much network traffic exists.

## TPU v4 Modeling Assumptions

`network_topology/tpu_v4.py` encodes the constants used throughout the repo:

- 45 GB/s unidirectional ICI link bandwidth
- about 4 pJ/bit/hop network energy
- about 500 ns per hop latency
- 6 links per chip

`make_tpu_v4_topology()` auto-selects:

- `Torus3D` when all dimensions are at least 4
- `Mesh3D` otherwise

The sweep script also manually instantiates some non-cubic torus shapes to compare aspect ratios.

## Running The Code

### 1. Install local dependencies

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

This installs:

- the local `network-topology` package in editable mode
- `numpy`
- `scipy`
- `pytest`

### 2. Run a local smoke test

This exercises the library without needing AccelForge:

```bash
.venv/bin/python - <<'PY'
from network_topology import make_tpu_v4_topology
from network_topology.cost_model import NetworkTransfer, CollectiveType, compute_network_cost

topo = make_tpu_v4_topology((4, 4, 4))
transfers = [
    NetworkTransfer("weights", 1024.0, CollectiveType.BROADCAST, src_chip=0),
    NetworkTransfer("grads", 2048.0, CollectiveType.ALLREDUCE),
]
result = compute_network_cost(topo, transfers)
print(topo.summary())
print(result.summary())
PY
```

### 3. Run the topology sweep

The sweep requires a local AccelForge checkout because it loads workload YAMLs from that repo and imports `accelforge` directly.

```bash
ACCELFORGE_ROOT=/path/to/accelforge .venv/bin/python sweep_gpt3.py
```

Notes:

- The script expects workload YAMLs under `$ACCELFORGE_ROOT/examples/workloads`.
- The architecture YAML is local to this repo: `accelforge_configs/tpu_v4_distributed_1d.yaml`.
- The current sweep maps on 2 chips and scales traffic to 64 chips before replaying it on the candidate topologies.
- If `ACCELFORGE_ROOT` is not set and there is no sibling `../accelforge` checkout, the script exits with a clear error.

The script prints:

- a topology summary table
- one mapping pass per workload
- a results table with per-topology latency in milliseconds
- a short insight summary

## Running Tests

Run the local test suite with:

```bash
.venv/bin/python -m pytest -q
```

The current tests cover:

- TPU v4 topology factory selection (`Torus3D` vs `Mesh3D`)
- concurrent link-load merging in `compute_network_cost()`
- the relationship between all-reduce, reduce-scatter, and all-gather on a ring

## Important Caveats

- The repo's network model is analytical. It does not simulate packet scheduling, queueing, or compute/network overlap.
- The end-to-end sweep only converts AccelForge `NetworkMemory` actions into `BROADCAST` and `ALLREDUCE`. If the upstream mapping emits richer communication patterns, the adapter would need to be extended.
- The 2-chip-to-64-chip scaling in `sweep_gpt3.py` is a modeling shortcut, not a proof that the mapping itself would stay optimal at 64 chips.
- `compute_l` and `compute_e` are extracted in the sweep script, but the printed comparison table is focused on network latency.

## Related Notes

- See `EXPERIMENTS.md` for example results and the qualitative conclusions the repo was used to study.
