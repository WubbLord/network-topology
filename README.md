# Network Topology

`network-topology` models communication cost for distributed accelerator workloads, with an emphasis on how the physical interconnect shape changes latency and energy once link congestion is taken into account.

At a high level, the repo does two things:

1. It defines a small topology/cost-model library for rings, meshes, tori, and custom graphs.
2. It plugs that library into an AccelForge mapping flow, so a mapped workload can be re-evaluated on multiple physical topologies.

The core idea is simple: route each collective onto physical links, compute that collective's bottleneck-link latency, and sum the independently costed collectives in workload order.

## What Lives Where

| Path | Purpose |
| --- | --- |
| `network_topology/topology.py` | Topology definitions, routing logic, and link-load latency/energy calculation |
| `network_topology/cost_model.py` | Transfer dataclasses and the public `compute_network_cost()` entrypoint |
| `network_topology/tpu_v4.py` | TPU v4 constants and a convenience factory for TPU-like topologies |
| `accelforge_configs/tpu_v4_distributed_1d.yaml` | AccelForge architecture spec used to map workloads quickly on a simplified 1D chip array |
| `sweep_matmuls.py` | End-to-end script that maps workloads with AccelForge, extracts network traffic, and evaluates multiple topologies |
| `sweep_moe.py` | Synthetic MoE token-dispatch/combine sweep with expert-placement strategies |
| `tests/` | Local pytest coverage for the topology and cost-model behavior |
| `EXPERIMENTS.md` | Example results and higher-level conclusions |

## How The Repo Works

There are two layers in the workflow.

### 1. Mapping and traffic extraction

The sweep script uses AccelForge to map a workload onto a distributed TPU v4-like architecture:

- It loads the local architecture spec from `accelforge_configs/tpu_v4_distributed_1d.yaml`.
- It loads workload YAMLs from an AccelForge checkout.
- It maps on `MAP_CHIPS = 8` chips first, because that search is much faster than mapping directly on 64 chips.
- It reads AccelForge's per-component, per-tensor action counts and filters out only `NetworkMemory` traffic.

In `sweep_matmuls.py`, the extracted traffic is converted into this repo's transfer model as follows:

- sharded matmul reads become `ALLGATHER` when a needed contracting or output dimension must be gathered.
- sharded matmul writes become `ALLREDUCE` or `REDUCE_SCATTER` when partial products must be reduced.
- fallback `NetworkMemory` reads become `BROADCAST` or `ALLGATHER` depending on the fallback path, and fallback writes become `ALLREDUCE`.
- The transferred bytes are multiplied by `SCALE = EVAL_CHIPS / MAP_CHIPS`, so traffic found on the 8-chip mapping is projected to the 64-chip evaluation size.

This means the sweep is not trying to re-map the workload for every topology. It maps once with AccelForge, turns the mapped network actions into abstract transfers, and then replays those transfers on each topology model.

### 2. Topology-aware network replay

Once the script has a list of `NetworkTransfer` objects, `compute_network_cost()` in `network_topology/cost_model.py` takes over:

1. Each transfer is routed onto physical links.
2. Each transfer's per-link byte loads are costed independently.
3. Energy is computed from total bit-hops.
4. Latency is computed from each transfer's bottleneck link load plus a hop-dependent term, then summed across transfers.

The routing rules depend on topology:

- `Ring` uses explicit ring formulas for broadcast and all-reduce.
- `Mesh3D` uses dimension-ordered behavior.
- `Torus3D` spreads traffic across per-dimension ring collectives.
- `Custom` falls back to shortest-path routing from the base `Topology` class.

The base cost function is:

- `energy = sum(bytes_on_link) * 8 * energy_per_bit_per_hop`
- `latency = max(bytes_on_link) / link_bandwidth + diameter * per_hop_latency`

The important modeling choice is that different collective operations are treated as sequential at this layer: one collective's link loads do not increase another collective's bottleneck load. Inside a single collective, link-level congestion still matters because the latency is set by that collective's most loaded link.

## The Iterative Convergence/Search Loop

This repo does not contain its own iterative numerical solver. The iterative search loop lives in AccelForge's mapper, and this repo consumes the mapper output.

That distinction matters:

- `network_topology/*` is a deterministic post-mapping network model.
- `sweep_matmuls.py` is a driver that calls the external mapper once per workload/topology feedback iteration.
- AccelForge's Fast and Fusiest Mapper (FFM) is the part that performs the expensive iterative search.

From this repo's point of view, the end-to-end loop is:

1. Build an AccelForge `Spec` from the local arch YAML plus a workload YAML.
2. Call `map_workload_to_arch(spec)`.
3. Let AccelForge search for Pareto-optimal mappings.
4. Extract `NetworkMemory` reads and writes from the chosen mapping.
5. Convert those actions into `NetworkTransfer` objects.
6. Replay the same transfer list on each candidate topology.
7. Compare topology-dependent network latency across topologies.

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

## How To Run The Code

### Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

For tests, use `pip install -e '.[dev]'`. For the full sweep, use `pip install -e '.[accelforge]'` and point `ACCELFORGE_ROOT` at a local AccelForge checkout that contains `examples/workloads`.

### Smoke Test

This smoke test exercises the topology model without requiring AccelForge:

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

### Topology Sweep

```bash
ACCELFORGE_ROOT=/path/to/accelforge .venv/bin/python sweep_matmuls.py
```

This uses the local architecture config in `accelforge_configs/tpu_v4_distributed_1d.yaml` and workload templates from `$ACCELFORGE_ROOT/examples/workloads`. If `ACCELFORGE_ROOT` is unset, the script falls back to a sibling checkout at `../accelforge`. The full run is also saved to `logs/<timestamp>/results.json`.

### MoE Sweep

The MoE sweep is synthetic and does not require AccelForge. It models token
dispatch to expert devices and the output combine step as two phase-synchronous
point-to-point traffic phases.

The default workloads use 64 chips and vary:

- expert count: 8, 16, or 64 experts
- routing fanout: top-1 or top-2
- locality bias: uniform routing or mostly local expert routing
- token shape: `tokens_per_chip * hidden_size * bytes_per_value`

For each workload, the sweep compares three expert placements:

- `clustered`: experts on low-numbered chips
- `spread`: experts evenly spread across the chip-id range
- `topology_aware`: starts from clustered, spread, and hop-greedy seeds, then
  runs a small exact-latency local search using the topology cost model

The synthetic MoE path evaluates explicit 64-chip topology graphs directly. It
does not use the AccelForge `MAP_CHIPS=8` shortcut from the matmul/GPT-3 flow.
To study batch effects, use `--experiment batch_scaling` or `--experiment all`;
the batch-scaling cases sweep `tokens_per_chip` from 64 through 8192 and report
remote-payload throughput plus payload link efficiency.

```bash
.venv/bin/python sweep_moe.py
```

Run the full baseline plus batch-scaling sweep over all topologies with parallel
workers:

```bash
.venv/bin/python sweep_moe.py --experiment all --all-topologies --jobs 32
```

On MIT ORCD/Engaging, submit the CPU Slurm wrapper:

```bash
sbatch --chdir=/home/nvemuri/projects/network-topology/slurm/logs \
  /home/nvemuri/projects/network-topology/slurm/sweep_moe.sbatch
```

The wrapper defaults to `mit_normal`, but the request can be strengthened from
the submit command:

```bash
sbatch --partition=mit_quicktest --cpus-per-task=32 --mem=64G --time=00:15:00 \
  --export=ALL,PROJECT_DIR=/home/nvemuri/projects/network-topology \
  --chdir=/home/nvemuri/projects/network-topology/slurm/logs \
  /home/nvemuri/projects/network-topology/slurm/sweep_moe.sbatch \
  --experiment all --all-topologies --jobs 32
```

Plot a completed MoE run with:

```bash
.venv/bin/python plot_moe_results.py logs/slurm-moe-12790884
```

See `MOE_EXPERIMENTS.md` for the full-64-chip MoE run notes and report-ready
tables.

## Running Tests

Run the local test suite with:

```bash
.venv/bin/python -m pytest -q
```

The current tests cover:

- TPU v4 topology factory selection (`Torus3D` vs `Mesh3D`)
- independent per-collective latency summation in `compute_network_cost()`
- the relationship between all-reduce, reduce-scatter, and all-gather on a ring

## Important Caveats

- The repo's network model is analytical. It does not simulate packet scheduling, queueing, or compute/network overlap.
- The end-to-end sweep infers collectives from AccelForge chip sharding, with a fallback from raw `NetworkMemory` actions to collective traffic. If the upstream mapping emits richer communication patterns, the adapter would need to be extended.
- The 8-chip-to-64-chip scaling in `sweep_matmuls.py` is a modeling shortcut, not a proof that the mapping itself would stay optimal at 64 chips.
- `compute_l` and `compute_e` are extracted in the sweep script, but the printed comparison table is focused on network latency.

## Related Notes

- See `EXPERIMENTS.md` for example results and the qualitative conclusions the repo was used to study.
