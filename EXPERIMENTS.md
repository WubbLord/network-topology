# Experiment Index

This file summarizes the experiment set. Use `SLURM.md` for exact commands, job
IDs, and output paths.

## Topologies

The paper and AccelForge-backed figure sets compare four 64-chip topologies
under the same TPU-v4-like link model:

| Topology | Degree | Diameter | Role |
| --- | ---: | ---: | --- |
| `Ring 64` | 2 | 32 | Low-degree baseline. |
| `Mesh 4x4x4` | 3-6 | 9 | Mesh baseline. |
| `Torus 4x4x4` | 6 | 6 | TPU-v4-style baseline. |
| `Circulant {1,5,17}` | 6 | 4 | Proposed six-link topology. |

## Workload Groups

| Group | Shapes | Driver | Figures |
| --- | --- | --- | --- |
| Small batched matmuls | `2Kx8K`, `4Kx16K`; `B=1,16,256` | `slurm/submit_small_batched_matmuls.sbatch` | `figures/current_small_batched_matmuls/` |
| Tall matmuls | `Nx2K`, `Nx4K`; `N=4K,16K,64K,256K`; `B=1,16,256` | `slurm/submit_tall_matmuls.sbatch` | `figures/current_tall_matmuls/map64/` |
| Square matmuls | `(N,N)x(N,T)`; `N=4K,16K,64K,256K`; `T=2K,4K` | `slurm/submit_square_matmuls.sbatch` | `figures/current_square_matmuls/` |
| MoE expert-FFN | 16 experts, hidden 12288, FFN 49152, `T=64,256,1024,4096` | `slurm/submit_moe_accelforge.sbatch` | `figures/current_moe_expert_ffn/` |
| Synthetic MoE | Sparse token dispatch and combine phases | `slurm/sweep_moe.sbatch` | `figures/current_moe_slurm-moe-12965825/` |
| GPT-3 175B | 10 decomposed single-einsum workloads, repeated over 96 layers | `slurm/submit_gpt3_175b.sbatch` | `logs/slurm-gpt3-175b-12632099/results.json` |

All paper matmul workloads use `MAP_CHIPS=64` and `EVAL_CHIPS=64`.

## Main Findings

Across the small, tall, square, and MoE expert-FFN result sets, the circulant
topology wins every AccelForge-backed workload. The gain is driven primarily by
lower bottleneck-link serialization, not by lower first-byte hop latency.
Serialization is more than 99% of the modeled network latency in the small,
tall, and square result sets.

Mapping sensitivity is low: successful small, tall, square, and MoE expert-FFN
tasks converge in two proxy-feedback iterations, and the inferred collective
plans remain stable between the first and final iterations.

## Figure Provenance

| Paper figure | File | Source |
| --- | --- | --- |
| Topology schematic | `report/figs/topology_schematics.png` | `plot_topologies.py` |
| Square latency | `report/figs/square_latency_comparison.png` | `figures/current_square_matmuls/3_latency_comparison.png` |
| Small latency | `report/figs/small_latency_comparison_a.png`, `report/figs/small_latency_comparison_b.png` | `figures/current_small_batched_matmuls/3_latency_comparison.png` split for report layout |
| Tall latency breakdown | `report/figs/tall_map64_b256_latency_breakdown_b.png` | `figures/current_tall_matmuls/map64/5_latency_breakdown_B256.png` split for report layout |
| Small energy breakdown | `report/figs/small_energy_breakdown.png` | `figures/current_small_batched_matmuls/4_energy_breakdown.png` |
| Tall network energy | `report/figs/tall_map64_b256_energy_comparison_b.png` | `figures/current_tall_matmuls/map64/13_network_energy_comparison_B256.png` split for report layout |
| MoE expert-FFN latency | `report/figs/moe_expert_ffn_latency_comparison.png` | `figures/current_moe_expert_ffn/moe_expert_ffn_latency_comparison.png` |

## MoE Experiment Details

There are two MoE paths.

1. `sweep_moe.py` is a synthetic replay of sparse token dispatch and combine.
2. `workloads/moe_expert_ffn.yaml` plus `sweep_matmuls.py` is an
   AccelForge-backed expert-FFN workload that uses the same mapper/replay loop
   as the matmul experiments.

### Synthetic Dispatch/Combine Model

Each MoE layer is modeled as two phase-synchronous sparse point-to-point
phases:

1. `dispatch`: the token-owning chip sends token activations to selected expert
   chips.
2. `combine`: the expert chip sends output activations back to the token-owning
   chip.

All transfers inside one phase are merged into concurrent link loads before
costing. Dispatch and combine phase latencies are then summed. Local
token-to-expert traffic on the same chip is not sent through the network.

The default batch-scaling experiment uses 64 chips, 16 experts, top-2 routing,
hidden size 12288, two-byte values, and 75% locality bias. It sweeps
`tokens_per_chip` from 64 to 8192.

Expert placement strategies:

| Strategy | Meaning |
| --- | --- |
| `clustered` | Experts are placed on low-numbered chips. |
| `spread` | Experts are evenly spread across the chip-id range. |
| `topology_aware` | Starts from clustered, spread, and hop-greedy seeds, then runs exact-latency local search. |

Synthetic MoE run:

| Field | Value |
| --- | --- |
| Job | `12965825` |
| Results | `logs/slurm-moe-12965825/moe_results.json` |
| Figures | `figures/current_moe_slurm-moe-12965825/` |
| Records | 288 result records |

### AccelForge Expert-FFN Workload

`workloads/moe_expert_ffn.yaml` approximates the dense expert compute portion
of a MoE layer as two expert-indexed FFN matmuls:

- `ExpertUp`: `X[e,t,h] * Wup[e,h,f] -> U[e,t,f]`
- `ExpertDown`: `U[e,t,f] * Wdown[e,f,h] -> Y[e,t,h]`

This path does not model sparse token routing. It evaluates the expert compute
communication that follows routing, using the same AccelForge mapping,
collective inference, topology replay, congestion model, and feedback loop as
the matmul experiments.

The wrapper submits `MoE ExpertFFN E16 T{64,256,1024,4096}` across the four core
topologies. Results are written to:

```text
logs/slurm-moe-accelforge-<submit_job>/results.json
```

The figure set is:

```text
figures/current_moe_expert_ffn/
```

Figures are generated from completed `results.json` files with:

```bash
.venv/bin/python plot_moe_expert_ffn_results.py \
  logs/slurm-moe-accelforge-<submit_job> \
  --mirror-dir figures/current_moe_expert_ffn
```

## Reproducing

Submit through Slurm only:

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
sbatch --partition=mit_quicktest --cpus-per-task=32 --mem=64G \
  --time=00:15:00 --export=ALL,PROJECT_DIR="$PROJECT_DIR" \
  --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/sweep_moe.sbatch" \
  --experiment all --all-topologies --jobs 32
```

Use `slurm/update_tall_matmul_figures.sbatch` to regenerate tall figures from
an existing completed tall run without resubmitting the array.
