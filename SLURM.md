# Slurm Runs

This file records the Slurm sweep jobs that have been run from this repo and what
their results mean. JSON latency fields are in seconds and energy fields are in
joules; console summaries usually print latency in milliseconds.

## Current Submit Commands

Small/core matmul sweep:

```bash
sbatch \
  --chdir=/home/zhangbr/network-topology/slurm/logs \
  /home/zhangbr/network-topology/slurm/submit_small.sbatch
```

GPT-3 175B decomposed-einsum sweep:

```bash
sbatch \
  --chdir=/home/zhangbr/network-topology/slurm/logs \
  /home/zhangbr/network-topology/slurm/submit_gpt3_175b.sbatch
```

Batched matmul sweep for the 7 `12524857` workloads at batch sizes 256, 512,
and 1024:

```bash
sbatch \
  --chdir=/home/zhangbr/network-topology/slurm/logs \
  /home/zhangbr/network-topology/slurm/submit_batched_matmuls.sbatch
```

Exact rerun of the 7 unbatched `12524857` workloads:

```bash
sbatch \
  --chdir=/home/zhangbr/network-topology/slurm/logs \
  /home/zhangbr/network-topology/slurm/submit_12524857_rerun.sbatch
```

Large matmul sweep with the mapper using all 64 chips:

```bash
sbatch \
  --chdir=/home/zhangbr/network-topology/slurm/logs \
  /home/zhangbr/network-topology/slurm/submit_large_map64.sbatch
```

Synthetic MoE token dispatch/combine sweep:

```bash
sbatch \
  --chdir=/home/nvemuri/projects/network-topology/slurm/logs \
  /home/nvemuri/projects/network-topology/slurm/sweep_moe.sbatch
```

AccelForge-backed MoE expert-FFN approximation:

```bash
sbatch \
  --chdir=/home/nvemuri/projects/network-topology/slurm/logs \
  /home/nvemuri/projects/network-topology/slurm/submit_moe_accelforge.sbatch
```

## Scripts

| Script | Purpose | Resources | Output location |
| --- | --- | --- | --- |
| `slurm/submit_small.sbatch` | Slurm wrapper for `bash slurm/submit.sh --small`. Resolves the repo root and submits the array plus aggregate job. | Submit wrapper: 1 CPU, 1 GB, 5 min. | `logs/slurm-<array_job>/results.json` |
| `slurm/submit.sh --small` | Submits the first 7 matmul workloads across the 4 core topologies. | Uses `slurm/sweep_array.sbatch`. | `logs/slurm-<array_job>/` |
| `slurm/sweep_array.sbatch` | Runs one workload/topology pair per array task. | 8 CPUs, 64 GB, 6 hours per task. | One partial JSON per workload/topology. |
| `slurm/submit_12524857_rerun.sbatch` | Wrapper for rerunning the exact first 7 matmul workloads from job `12524857` across the 4 core topologies. | Submit wrapper: 1 CPU, 1 GB, 5 min. Array: 8 CPUs, 64 GB, 6 hours per task, `0-27%2`. | `logs/slurm-<array_job>/results.json` |
| `slurm/submit_batched_matmuls.sbatch` | Wrapper for the batched versions of the 7 `12524857` workloads at batch sizes 256, 512, and 1024. | Submit wrapper: 1 CPU, 1 GB, 5 min. Array: 8 CPUs, 64 GB, 6 hours per task, `0-83%2`. | `logs/slurm-batched-matmuls-<submit_job>/results.json` |
| `slurm/sweep_batched_matmuls.sbatch` | Runs one batched workload/topology pair per array task. | 8 CPUs, 64 GB, 6 hours per task. | One partial JSON per workload/topology. |
| `slurm/submit_large_map64.sbatch` | Wrapper for the four large matmul workloads with `MAP_CHIPS=64` and `EVAL_CHIPS=64`. Uses workload array indices `28-43`, corresponding to `Square 128Kx128K`, `Tall 256Kx64K`, `LargeSquare 256Kx256K`, and `VeryTall 256Kx8K` across the 4 core topologies. | Submit wrapper: 1 CPU, 1 GB, 5 min. Array: 16 CPUs, 128 GB, 12 hours per task, `28-43%1`. | `logs/slurm-large-map64-<submit_job>/results.json` |
| `slurm/submit_gpt3_175b.sbatch` | Slurm wrapper for the GPT-3 175B sweep. Submits array tasks for the 4 core topologies and an aggregate dependency. | Submit wrapper: 1 CPU, 1 GB, 5 min. Array override: 4 CPUs, 8 GB, 2 hours. |
| `slurm/sweep_gpt3_175b.sbatch` | Runs GPT-3 175B on one topology per array task. | Default headers: 4 CPUs, 8 GB, 2 hours. | `logs/slurm-gpt3-175b-<submit_job>/` |
| `slurm/sweep_moe.sbatch` | Runs the synthetic MoE sweep on the CPU partition. | 1 CPU, 4 GB, 15 minutes on `mit_normal`. | `logs/slurm-moe-<job>/moe_results.json` |
| `slurm/submit_moe_accelforge.sbatch` | Wrapper for AccelForge-backed MoE expert-FFN approximation workloads. | Submit wrapper: 1 CPU, 1 GB, 5 min. Array: 8 CPUs, 64 GB, 6 hours per task, `0-15%1`. | `logs/slurm-moe-accelforge-<submit_job>/results.json` |
| `slurm/sweep_moe_accelforge.sbatch` | Runs one MoE expert-FFN workload/topology pair per array task through `sweep_matmuls.py`. | 8 CPUs, 64 GB, 6 hours per task. | One partial JSON per workload/topology. |
| `slurm/aggregate.sbatch` | Merges partial JSON files after an array job succeeds. | 1 CPU, 4 GB, 5 min. | Writes `results.json` and usually `milestone2_analysis.json`. |

## Workloads

Available in `sweep_matmuls.py`:

| Workload | Meaning |
| --- | --- |
| `GPT3 175B` | Decomposed GPT-3 175B forward-pass workload. It maps 10 independent single-einsum workloads: `I`, `V`, `K`, `Q`, `QK`, `QK_softmax`, `AV`, `Z`, `FFA`, `FFB`; then replicates their collectives across `N_LAYERS=96`. |
| `Small 2Kx8K` | Single matmul with `M=2048`, `KN=8192`. |
| `Small 4Kx16K` | Single matmul with `M=4096`, `KN=16384`. |
| `Medium 8Kx32K` | Single matmul with `M=8192`, `KN=32768`. |
| `Medium 16Kx32K` | Single matmul with `M=16384`, `KN=32768`. |
| `Wide 8Kx256K` | Giant, weight-dominated matmul. |
| `VeryWide 2Kx256K` | Giant, very wide matmul. |
| `Rect 64Kx128K` | Giant rectangular matmul. |
| `Square 128Kx128K` | Giant square matmul. |
| `Tall 256Kx64K` | Giant activation-heavy matmul. |
| `LargeSquare 256Kx256K` | Giant large square matmul. |
| `VeryTall 256Kx8K` | Giant very tall matmul. |

`submit.sh --small` currently runs the first 7 matmul workloads in that list:
`Small 2Kx8K`, `Small 4Kx16K`, `Medium 8Kx32K`, `Medium 16Kx32K`,
`Wide 8Kx256K`, `VeryWide 2Kx256K`, and `Rect 64Kx128K`.

The batched matmul sweep uses the same 7 base workloads, with a leading batch
rank `BATCH_SIZE` in `{256, 512, 1024}`. The workload names are of the form
`Batched B256 Small 2Kx8K`, `Batched B512 Small 2Kx8K`, and so on.

`sweep_moe.py` includes synthetic MoE workloads that place tokens and experts on
different chips, then compare `clustered`, `spread`, and `topology_aware`
expert placement. Dispatch and combine are costed as two concurrent
point-to-point phases.

Unlike the AccelForge matmul/GPT-3 sweeps, this synthetic MoE sweep evaluates
explicit 64-chip topologies directly; there is no 8-chip-to-64-chip scaling
shortcut. For the full baseline plus batch-scaling experiment, request a larger
CPU allocation and let the script use the Slurm CPU count:

```bash
sbatch --partition=mit_quicktest --cpus-per-task=32 --mem=64G --time=00:15:00 \
  --export=ALL,PROJECT_DIR=/home/nvemuri/projects/network-topology \
  --chdir=/home/nvemuri/projects/network-topology/slurm/logs \
  /home/nvemuri/projects/network-topology/slurm/sweep_moe.sbatch \
  --experiment all --all-topologies --jobs 32
```

MoE figures can be generated from a completed run with:

```bash
.venv/bin/python plot_moe_results.py logs/slurm-moe-12790884
```

## Topologies

The 4 core topologies used by the small and GPT-3 sweeps are:

| Topology | Notes |
| --- | --- |
| `Torus 4x4x4` | Baseline 3D torus, 64 chips. |
| `Mesh 4x4x4` | 3D mesh, 64 chips. |
| `Ring 64` | 64-chip ring. |
| `Circulant {1,5,17}` | Proposed 6-link/chip circulant topology. |

Additional topologies available in `sweep_matmuls.py` include `Torus 8x2x4`,
`4D Torus 4x4x2x2`, `5D Torus 4x2x2x2x2`, and `6D Hypercube`.

## Successful Runs

| Submit job | Array job | Aggregate job | Workloads | Topologies | Status | Results |
| --- | --- | --- | --- | --- | --- | --- |
| `12634415` | `12634425` | `12634426` | Batched versions of the 7 `12524857` matmul workloads at batch sizes 256, 512, and 1024. | 4 core topologies. | Successful. All 84 array tasks completed and aggregate succeeded. Figures copied to `figures/current_batched_matmul_slurm_12634425/`. | `logs/slurm-batched-matmuls-12634415/results.json` |
| `12634416` | `12634419` | `12634424` | Exact rerun of the 7 unbatched `12524857` matmul workloads. | 4 core topologies. | Successful. All 28 array tasks completed and aggregate succeeded. Figures copied to `figures/current_matmul_slurm_12634419/`. | `logs/slurm-12634419/results.json` |
| `12632099` | `12632101` | `12632102` | `GPT3 175B` decomposed into 10 single-einsum workloads, `N_TOKENS=8192`, `N_LAYERS=96`. | 4 core topologies. | Successful. All 4 tasks converged in 2 proxy iterations. | `logs/slurm-gpt3-175b-12632099/results.json` |
| `12524833` | `12524857` | `12524858` | 7 matmul workloads from `--small`: small, medium, wide, very wide, and rectangular. | 4 core topologies. | Successful. All 28 tasks completed and converged in 2 proxy iterations. | `logs/slurm-12524857/results.json` |
| `12553483` | `12553487` | `12553488` | `GPT3 175B` direct synthetic allreduce model. | 4 core topologies. | Successful mechanically, but obsolete. This bypassed AccelForge and is not comparable to the decomposed GPT-3 run. | `logs/slurm-gpt3-175b-12553483/results.json` |
| `12553224` | `12553229` | `12553230` | `GPT3 175B` direct synthetic allreduce model. | 4 core topologies. | Successful mechanically, but obsolete for the same reason as `12553483`. | `logs/slurm-gpt3-175b-12553224/results.json` |
| unknown wrapper | `12553012` | `12553013` | `GPT3 175B` direct synthetic allreduce quicktest. | 4 core topologies. | Successful mechanically, but obsolete. | `logs/slurm-gpt3-175b-quicktest-20260426-003612/results.json` |

## Key Results

### Decomposed GPT-3 175B, job `12632101`

Output: `logs/slurm-gpt3-175b-12632099/results.json`

| Topology | Network latency | Wall time | Notes |
| --- | ---: | ---: | --- |
| `Circulant {1,5,17}` | 2,418,069.3 ms | 384 s | Best latency. |
| `Torus 4x4x4` | 4,343,075.0 ms | 354 s | 1.80x slower than circulant. |
| `Ring 64` | 5,412,539.3 ms | 442 s | 2.24x slower than circulant. |
| `Mesh 4x4x4` | 7,325,502.3 ms | 444 s | 3.03x slower than circulant. |

Other useful facts:

- Total modeled network traffic: `1.66e14` bytes.
- Transfer count: 1344 per topology.
- Collective byte mix on the circulant run: `ALLREDUCE 48%`, `BROADCAST 25%`,
  `ALLGATHER 24%`, `REDUCE_SCATTER 3%`, local near zero.
- Hottest per-layer transfers are the attention score tensors: `QK`,
  `QK_softmax`, and `AV:QK_softmax`.
- The aggregate generated both `results.json` and `milestone2_analysis.json`.

### Matmul Small Sweep, job `12524857`

Output: `logs/slurm-12524857/results.json`

All 7 workloads converged in 2 proxy iterations on all 4 core topologies.

| Workload | Best topology | Best latency | Worst topology | Worst latency | Speedup |
| --- | --- | ---: | --- | ---: | ---: |
| `Small 2Kx8K` | `Circulant {1,5,17}` | 7.8 ms | `Mesh 4x4x4` | 23.9 ms | 3.05x |
| `Small 4Kx16K` | `Circulant {1,5,17}` | 31.3 ms | `Mesh 4x4x4` | 95.4 ms | 3.05x |
| `Medium 8Kx32K` | `Circulant {1,5,17}` | 125.3 ms | `Mesh 4x4x4` | 381.8 ms | 3.05x |
| `Medium 16Kx32K` | `Circulant {1,5,17}` | 250.5 ms | `Mesh 4x4x4` | 763.6 ms | 3.05x |
| `Wide 8Kx256K` | `Circulant {1,5,17}` | 2004.3 ms | `Mesh 4x4x4` | 6108.4 ms | 3.05x |
| `VeryWide 2Kx256K` | `Circulant {1,5,17}` | 250.5 ms | `Mesh 4x4x4` | 763.6 ms | 3.05x |
| `Rect 64Kx128K` | `Circulant {1,5,17}` | 8017.3 ms | `Mesh 4x4x4` | 24433.6 ms | 3.05x |

The matmul runs were reduction-only in these results: either `REDUCE_SCATTER`
or `ALLREDUCE`, depending on the selected mapping.

## Partial Or Obsolete Runs

| Job | What happened | Useful output |
| --- | --- | --- |
| `12524129` | Earlier `--small` array. 21 partial workload/topology JSONs completed, but later tasks were cancelled and there is no aggregate `results.json`. Some early tasks hit stale file handle failures. | `logs/slurm-12524129/*.json` |
| `12552512` | Direct synthetic GPT-3 run with only `Torus 4x4x4` written. Obsolete. | `logs/slurm-gpt3-175b-12552501/GPT3_175B__Torus_4x4x4.json` |
| `12524891` | Full AccelForge GPT-3 attempt before decomposition. The per-task stdout reported `FAILED: BUG`; no successful result records. | `logs/slurm-gpt3-175b-12524859/*.json` |
| `12530236` | Full AccelForge GPT-3 attempt with failure handling enabled. All 4 topology tasks failed with `AssertionError: BUG` inside AccelForge mapping. | `logs/slurm-gpt3-175b-12530151/*.json` |

## Notes

- Current cost model computes each collective independently. It does not merge
  link loads across different collectives.
- The mapper runs on `MAP_CHIPS=8`; network transfers are evaluated on
  `EVAL_CHIPS=64` with `SCALE=8`.
- The decomposed GPT-3 path maps each single-einsum workload independently, so
  it avoids the AccelForge multi-einsum GPT-3 assertion failure but does not
  model cross-einsum fusion or tensor residency.
