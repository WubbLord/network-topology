# Current Experiment Index

This file summarizes the cleaned experiment set. Use `SLURM.md` for exact
commands, job IDs, and output paths.

## Topologies

The paper and retained AccelForge-backed figure sets compare four 64-chip
topologies under the same TPU-v4-like link model:

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
| GPT-3 175B | 10 decomposed single-einsum workloads, repeated over 96 layers | `slurm/submit_gpt3_175b.sbatch` | JSON retained in `logs/slurm-gpt3-175b-12632099/` |

All current paper matmul workloads use `MAP_CHIPS=64` and `EVAL_CHIPS=64`.

## Main Findings

Across the current small, tall, square, and MoE expert-FFN result sets, the
circulant topology wins every AccelForge-backed workload. The gain is driven
primarily by lower bottleneck-link serialization, not by lower first-byte hop
latency. Serialization is more than 99% of the modeled network latency in the
small, tall, and square result sets.

Mapping sensitivity is low for the retained runs: successful small, tall,
square, and MoE expert-FFN tasks converge in two proxy-feedback iterations, and
the inferred collective plans remain stable between the first and final
iterations.

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
```

Use `slurm/update_tall_matmul_figures.sbatch` to regenerate tall figures from
an existing completed tall run without resubmitting the array.
