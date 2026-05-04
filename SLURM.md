# Slurm Runs

This file records the retained Slurm workflow and the successful jobs whose
results are still useful. JSON latency fields are seconds and energy fields are
joules; console summaries usually print latency in milliseconds.

Do not run mapper experiments on the login node. Submit the wrappers below with
`sbatch`.

## Submit Commands

```bash
PROJECT_DIR=/home/zhangbr/network-topology
```

Small batched matmuls, `MAP_CHIPS=64`:

```bash
sbatch --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/submit_small_batched_matmuls.sbatch"
```

Tall matmuls, `MAP_CHIPS=64`:

```bash
sbatch --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/submit_tall_matmuls.sbatch"
```

Square attention-like matmuls, `MAP_CHIPS=64`:

```bash
sbatch --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/submit_square_matmuls.sbatch"
```

Mapper-backed MoE expert-FFN:

```bash
sbatch --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/submit_moe_accelforge.sbatch"
```

Synthetic MoE dispatch/combine:

```bash
sbatch --partition=mit_quicktest --cpus-per-task=32 --mem=64G \
  --time=00:15:00 --export=ALL,PROJECT_DIR="$PROJECT_DIR" \
  --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/sweep_moe.sbatch" \
  --experiment all --all-topologies --jobs 32
```

Decomposed GPT-3 175B:

```bash
sbatch --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/submit_gpt3_175b.sbatch"
```

## Active Scripts

| Script | Purpose | Output |
| --- | --- | --- |
| `slurm/setup_env.sh` | Loads cluster Python module and activates `.venv`. | Shared by array/plot jobs. |
| `slurm/aggregate.sbatch` | Merges one array run into `results.json`. | `RUN_DIR/results.json`; often also `milestone2_analysis.json`. |
| `slurm/submit_small_batched_matmuls.sbatch` | Submits small `2Kx8K` and `4Kx16K` workloads at `B=1,16,256`. | `logs/slurm-small-batched-matmuls-<job>/results.json`; mirrors figures to `figures/current_small_batched_matmuls/`. |
| `slurm/sweep_small_batched_matmuls.sbatch` | Runs one small workload/topology array task. | Partial JSON in the run directory. |
| `slurm/plot_small_batched_matmuls.sbatch` | Generates numbered small-workload figures. | `figures/current_small_batched_matmuls/`. |
| `slurm/submit_tall_matmuls.sbatch` | Submits tall `Nx2K` and `Nx4K` workloads for `N=4K,16K,64K,256K` and `B=1,16,256`. | `logs/slurm-tall-matmuls-<job>/map64/results.json`; figures under `figures/current_tall_matmuls/map64/`. |
| `slurm/sweep_tall_matmuls.sbatch` | Runs one tall workload/topology array task. | Partial JSON in `map64/`. |
| `slurm/plot_tall_matmuls.sbatch` | Generates numbered tall-workload figures. | `figures/current_tall_matmuls/map64/`. |
| `slurm/update_tall_matmul_figures.sbatch` | Regenerates tall figures from an existing `RUN_ROOT`. | Updated tall figure directory. |
| `slurm/submit_square_matmuls.sbatch` | Submits square attention-like `(N,N)x(N,T)` workloads for `N=4K,16K,64K,256K` and `T=2K,4K`. | `logs/slurm-square-matmuls-<job>/results.json`; mirrors figures to `figures/current_square_matmuls/`. |
| `slurm/sweep_square_matmuls.sbatch` | Runs one square workload/topology array task. | Partial JSON in the run directory. |
| `slurm/plot_square_matmuls.sbatch` | Generates numbered square-workload figures. | `figures/current_square_matmuls/`. |
| `slurm/sweep_moe.sbatch` | Runs synthetic MoE dispatch/combine sweeps. | `logs/slurm-moe-<job>/moe_results.json`. |
| `slurm/submit_moe_accelforge.sbatch` | Submits AccelForge-backed MoE expert-FFN workloads for the four core topologies. | `logs/slurm-moe-accelforge-<job>/results.json`. |
| `slurm/submit_moe_accelforge_map64.sbatch` | Same expert-FFN workload with explicit map/eval chip exports. | `logs/slurm-moe-accelforge-map64-<job>/results.json`. |
| `slurm/sweep_moe_accelforge.sbatch` | Runs one expert-FFN workload/topology array task. | Partial JSON in the run directory. |
| `slurm/plot_moe_accelforge.sbatch` | Generates MoE expert-FFN figures and summaries after aggregation. | `figures/current_moe_expert_ffn/`. |
| `slurm/submit_gpt3_175b.sbatch` | Submits decomposed GPT-3 175B einsum workload across the four core topologies. | `logs/slurm-gpt3-175b-<job>/results.json`. |
| `slurm/sweep_gpt3_175b.sbatch` | Runs one GPT-3 topology array task. | Partial JSON in the run directory. |

All submit wrappers resolve the repository root from their own script path, so
they are independent of the directory where `sbatch` is invoked.

## Active Workloads

| Group | Workloads | Topologies | Figure directory |
| --- | --- | --- | --- |
| Small batched matmuls | `2Kx8K`, `4Kx16K`, with `B=1,16,256` | Ring, mesh, torus, circulant | `figures/current_small_batched_matmuls/` |
| Tall matmuls | `Nx2K`, `Nx4K`; `N=4K,16K,64K,256K`; `B=1,16,256` | Ring, mesh, torus, circulant | `figures/current_tall_matmuls/map64/` |
| Square matmuls | `(N,N)x(N,T)`; `N=4K,16K,64K,256K`; `T=2K,4K` | Ring, mesh, torus, circulant | `figures/current_square_matmuls/` |
| MoE expert-FFN | 16 experts, hidden 12288, FFN 49152, `T=64,256,1024,4096` | Ring, mesh, torus, circulant | `figures/current_moe_expert_ffn/` |
| Synthetic MoE | Token dispatch/combine cases from `sweep_moe.py` | Configurable; default core or `--all-topologies` | `figures/current_moe_slurm-moe-12965825/` |
| GPT-3 175B | 10 decomposed single-einsum workloads, repeated across 96 layers | Ring, mesh, torus, circulant | JSON result only in retained docs |

## Successful Current Runs

| Submit job | Array job | Aggregate job | Workloads | Status | Results |
| --- | --- | --- | --- | --- | --- |
| `12973543` | `12973566` | `12973567` | Small batched matmuls at `B=1,16,256`, `MAP_CHIPS=64`. | Successful; figures mirrored to `figures/current_small_batched_matmuls/`. | `logs/slurm-small-batched-matmuls-12973543/results.json` |
| `12961497` | `12961515` | `12961516` | Tall matmuls at `B=1,16,256`, `MAP_CHIPS=64`. | Successful; map64 figures mirrored to `figures/current_tall_matmuls/map64/`. | `logs/slurm-tall-matmuls-12961497/map64/results.json` |
| `12976708` | `12976838` | `12977601` | Square attention-like matmuls, `MAP_CHIPS=64`. | Successful; figures mirrored to `figures/current_square_matmuls/`. | `logs/slurm-square-matmuls-12976708/results.json` |
| unknown | unknown | unknown | MoE expert-FFN AccelForge run used for `figures/current_moe_expert_ffn/`. | Successful; retained result summary and paper figure. | `figures/current_moe_expert_ffn/moe_expert_ffn_summary.csv` |
| `12965825` | n/a | n/a | Synthetic MoE dispatch/combine run. | Successful; 288 result records and 5 figures. | `logs/slurm-moe-12965825/moe_results.json` |
| `12632099` | `12632101` | `12632102` | Decomposed GPT-3 175B. | Successful; all four topology tasks converged in two proxy iterations. | `logs/slurm-gpt3-175b-12632099/results.json` |

## Historical Or Obsolete Runs

These logs were useful during debugging, but their old wrappers or old figures
were removed from the cleaned tracked tree.

| Job | What happened | Useful output |
| --- | --- | --- |
| `12524857` | First successful seven-workload small matmul run. Superseded by the current small batched `MAP_CHIPS=64` run. | `logs/slurm-12524857/results.json` |
| `12634419` | Exact rerun of the seven unbatched `12524857` workloads. Superseded by current small/tall/square splits. | `logs/slurm-12634419/results.json` |
| `12634425` | Batched versions of the seven `12524857` workloads at `B=256,512,1024`. Superseded by current small/tall/square splits. | `logs/slurm-batched-matmuls-12634415/results.json` |
| `12553483`, `12553224`, `12553012` | Direct synthetic GPT-3 all-reduce models. Mechanically successful but not comparable to mapper-backed decomposed GPT-3. | `logs/slurm-gpt3-175b-*`. |
| `12524891`, `12530236` | Full AccelForge GPT-3 attempts before decomposition. Failed inside AccelForge with `AssertionError: BUG`. | Per-task failure JSON/stdout in `logs/slurm-gpt3-175b-*`. |
| `12524129` | Early small sweep; later tasks were cancelled and no aggregate survived. | Partial JSONs in `logs/slurm-12524129/`. |

## Notes

- The current cost model computes each collective independently. It does not
  merge link loads across different collectives.
- The current paper matmul runs use `MAP_CHIPS=64` and `EVAL_CHIPS=64`.
- Decomposed GPT-3 maps each single-einsum workload independently, avoiding the
  full multi-einsum AccelForge assertion failure but not modeling cross-einsum
  fusion or tensor residency.
- `slurm/logs/` contains scheduler stdout/stderr. The durable result artifacts
  are under `logs/` and the mirrored figure directories under `figures/`.
