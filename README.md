# Network Topology

`network-topology` couples AccelForge mapping with an analytical inter-chip
network model for 64-chip TPU-v4-like systems. The current cleaned tree keeps
the core topology/cost pipeline, the workload drivers used for the active
figure sets, and the Slurm wrappers needed to reproduce them on the cluster.

Do not run full experiments on the login node. Use the Slurm submit scripts in
`slurm/`; local commands are only for static checks, plotting completed JSON, or
unit tests that do not launch mapper sweeps.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `network_topology/` | Graph topologies, TPU-v4 constants, and network cost model. |
| `sweep_matmuls.py` | AccelForge-backed matmul, square-attention, GPT-3, and MoE expert-FFN sweep driver. |
| `sweep_moe.py` | Synthetic MoE token dispatch/combine replay driver. |
| `aggregate_results.py` | Merge Slurm array partial JSON into one `results.json`. |
| `plot_results.py` | Shared numbered figure helpers. |
| `plot_*_results.py` | Workload-specific figure generators. |
| `workloads/` | YAML workload templates for local matmul and MoE expert-FFN sweeps. |
| `accelforge_configs/` | TPU-v4-like AccelForge architecture config. |
| `slurm/` | Submit, array-task, aggregate, and plot jobs for retained experiments. |
| `figures/` | Current generated figure sets. |
| `report/` | Paper source and the figures included in the paper. |
| `tests/` | Pytest coverage for topology/cost behavior and MoE helpers. |
| `SLURM.md` | Job inventory, commands, output paths, and successful run log. |
| `EXPERIMENTS.md` | Current experiment index and figure provenance. |
| `MOE_EXPERIMENTS.md` | MoE-specific notes. |

## Model Summary

The workflow has two stages.

1. AccelForge maps each workload against a logical `NetworkMemory` level.
2. This repo extracts `NetworkMemory` traffic, infers collectives, and replays
   those collectives on physical topologies.

Each collective is costed independently in program order. That means bytes from
different collectives are not merged into one global link-load table. Congestion
is still modeled inside each collective: routed bytes are accumulated per
physical link, and latency is set by the busiest link in that collective.

For a transfer with routed link byte loads `B_l`, the model uses:

```text
energy  = 8 * energy_per_bit_per_hop * sum_l(B_l)
latency = max_l(B_l) / link_bandwidth + diameter * per_hop_latency
```

The TPU-v4-like constants are:

| Constant | Value |
| --- | ---: |
| ICI link bandwidth | 45 GB/s, unidirectional |
| Network energy | 4 pJ/bit/hop |
| First-byte latency | 500 ns/hop |
| Link budget | 6 links/chip |

The main paper compares four 64-chip topologies: `Ring 64`, `Mesh 4x4x4`,
`Torus 4x4x4`, and `Circulant {1,5,17}`.

## AccelForge Feedback Loop

Mapper-backed sweeps run a proxy-feedback loop:

1. Map the workload with the current logical network proxy.
2. Extract AccelForge `NetworkMemory` reads and writes.
3. Infer broadcast, all-gather, reduce-scatter, or all-reduce collectives from
   tensor sharding and network actions.
4. Replay those collectives on the requested topology.
5. Convert final latency/energy back to per-byte proxy costs.
6. Repeat until the proxy changes by less than 5% or the iteration cap is hit.

The active small, tall, square, and MoE expert-FFN result sets all converged in
two proxy iterations.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,accelforge]'
```

Set `ACCELFORGE_ROOT` if AccelForge is not available as a sibling checkout:

```bash
export ACCELFORGE_ROOT=/path/to/accelforge
```

The Slurm scripts source `.venv/bin/activate` through `slurm/setup_env.sh`.

## Running Experiments

Submit experiments through Slurm. The wrappers resolve the project root from the
script path, so they can be submitted from any working directory.

```bash
PROJECT_DIR=/home/zhangbr/network-topology

sbatch --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/submit_small_batched_matmuls.sbatch"

sbatch --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/submit_tall_matmuls.sbatch"

sbatch --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/submit_square_matmuls.sbatch"

sbatch --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/submit_moe_accelforge.sbatch"
```

See `SLURM.md` for the full script matrix, output directories, successful job
IDs, and GPT-3/synthetic-MoE notes.

## Active Figure Sets

| Figure directory | Source script |
| --- | --- |
| `figures/current_small_batched_matmuls/` | `slurm/submit_small_batched_matmuls.sbatch` |
| `figures/current_tall_matmuls/map64/` | `slurm/submit_tall_matmuls.sbatch` |
| `figures/current_square_matmuls/` | `slurm/submit_square_matmuls.sbatch` |
| `figures/current_moe_expert_ffn/` | `slurm/submit_moe_accelforge.sbatch` plus `plot_moe_expert_ffn_results.py` |
| `figures/current_moe_slurm-moe-12965825/` | `slurm/sweep_moe.sbatch` synthetic MoE run |
| `figures/topology_schematics.png` | `plot_topologies.py` |
| `report/figs/` | Paper figure copies used by `report/main.tex` |

## Static Verification

These checks do not launch experiment sweeps:

```bash
.venv/bin/python -m compileall network_topology sweep_matmuls.py sweep_moe.py \
  aggregate_results.py plot_results.py plot_small_batched_matmul_results.py \
  plot_tall_matmul_results.py plot_square_matmul_results.py \
  plot_moe_results.py plot_moe_expert_ffn_results.py plot_topologies.py tests

.venv/bin/python -m pytest -q
```

If the cluster module stack does not provide LaTeX, the report source can still
be reviewed, but `pdflatex` will not run locally.

## Caveats

- The model is analytical; it does not simulate packet scheduling, queueing, or
  compute/network overlap.
- The AccelForge adapter infers collectives from mapped tensor sharding and
  `NetworkMemory` actions. Richer communication patterns would need a richer
  adapter.
- `MAP_CHIPS=64` is used for the current small, tall, and square paper runs.
  Older `MAP_CHIPS=8` scaling artifacts were removed from the tracked figure
  set during cleanup.
- GPT-3 is retained as a decomposed-einsum study, but it is not one of the main
  paper figure sources.
