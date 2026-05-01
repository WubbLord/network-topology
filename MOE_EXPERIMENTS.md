# MoE Experiment Notes

Synthetic MoE experiments are run by `sweep_moe.py`. Unlike the AccelForge
matmul/GPT-3 path, this path evaluates explicit 64-chip topologies directly and
does not use the `MAP_CHIPS=8` scaling shortcut.

## Model

Each MoE layer is modeled as two phase-synchronous sparse point-to-point phases:

1. `dispatch`: the token-owning chip sends token activations to selected expert
   chips.
2. `combine`: the expert chip sends output activations back to the token-owning
   chip.

All transfers inside one phase are merged into concurrent link loads before
costing. Dispatch and combine phase latencies are then summed. Local token-to-
expert traffic on the same chip is not sent through the network.

The default 64-chip batch-scaling experiment uses 16 experts, top-2 routing,
hidden size 12288, BF16-like two-byte values, and 75% locality bias. It sweeps
`tokens_per_chip` from 64 to 8192.

Expert placement strategies:

- `clustered`: experts on low-numbered chips.
- `spread`: experts evenly spread across the chip-id range.
- `topology_aware`: starts from clustered, spread, and hop-greedy seeds, then
  runs exact-latency local search under a one-expert-per-chip capacity for the
  16-expert cases.

## Full 64-Chip Slurm Run

Command:

```bash
sbatch --partition=mit_quicktest --cpus-per-task=32 --mem=64G --time=00:15:00 \
  --export=ALL,PROJECT_DIR=/home/nvemuri/projects/network-topology \
  --chdir=/home/nvemuri/projects/network-topology/slurm/logs \
  /home/nvemuri/projects/network-topology/slurm/sweep_moe.sbatch \
  --experiment all --all-topologies --jobs 32
```

Output:

- Results JSON: `logs/slurm-moe-12965825/moe_results.json`
- Figures: `figures/current_moe_slurm-moe-12965825/`

The job used 32 CPUs, completed successfully in 36 seconds, and produced 288
result records: 12 workloads, 8 topologies, and 3 placement strategies.

## Key Results

With topology-aware placement, the circulant graph beats the 3D torus on every
MoE case in this run. It is the best topology for 11 of the 12 topology-aware
cases. The exception is the fully uniform 16-expert top-2 workload, where the
6D hypercube has the lowest latency; the circulant still remains 1.37x faster
than the 3D torus on that case.

| Workload | Best topology | Best latency | Torus / circulant |
| --- | --- | ---: | ---: |
| MoE local E16 top2 | Circulant `{1,5,17}` | 4.32 ms | 1.18x |
| MoE local E64 top2 | Circulant `{1,5,17}` | 0.46 ms | 1.85x |
| MoE uniform E16 top2 | 6D Hypercube | 4.48 ms | 1.37x |
| MoE heavy E8 top1 | Circulant `{1,5,17}` | 5.38 ms | 1.11x |

Batch scaling with the 16-expert local top-2 workload shows the circulant's
payload link efficiency rising with batch size as fixed hop latency is
amortized:

| Tokens/chip | Circulant latency | Torus latency | Torus / circulant | Circulant efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.274 ms | 0.325 ms | 1.19x | 7.08% |
| 128 | 0.544 ms | 0.644 ms | 1.19x | 7.13% |
| 256 | 1.083 ms | 1.282 ms | 1.18x | 7.16% |
| 512 | 2.162 ms | 2.558 ms | 1.18x | 7.17% |
| 1024 | 4.321 ms | 5.109 ms | 1.18x | 7.18% |
| 2048 | 8.637 ms | 10.212 ms | 1.18x | 7.18% |
| 4096 | 17.271 ms | 20.418 ms | 1.18x | 7.18% |
| 8192 | 34.537 ms | 40.831 ms | 1.18x | 7.19% |

The corresponding 3D torus efficiency rises from 5.80% to 5.91%, so the
circulant maintains a higher useful payload efficiency across the whole batch
sweep.
