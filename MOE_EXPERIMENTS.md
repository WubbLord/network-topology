# MoE Experiment Notes

There are two MoE paths in the cleaned repo.

1. `sweep_moe.py` is a synthetic replay of sparse token dispatch and combine.
2. `workloads/moe_expert_ffn.yaml` plus `sweep_matmuls.py` is an
   AccelForge-backed expert-FFN workload that uses the same mapper/replay loop
   as the matmul experiments.

## Synthetic Dispatch/Combine Model

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

Submit the full synthetic run through Slurm:

```bash
PROJECT_DIR=/home/zhangbr/network-topology

sbatch --partition=mit_quicktest --cpus-per-task=32 --mem=64G \
  --time=00:15:00 --export=ALL,PROJECT_DIR="$PROJECT_DIR" \
  --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/sweep_moe.sbatch" \
  --experiment all --all-topologies --jobs 32
```

Retained successful run:

| Field | Value |
| --- | --- |
| Job | `12965825` |
| Results | `logs/slurm-moe-12965825/moe_results.json` |
| Figures | `figures/current_moe_slurm-moe-12965825/` |
| Records | 288 result records |

## AccelForge Expert-FFN Workload

`workloads/moe_expert_ffn.yaml` approximates the dense expert compute portion
of a MoE layer as two expert-indexed FFN matmuls:

- `ExpertUp`: `X[e,t,h] * Wup[e,h,f] -> U[e,t,f]`
- `ExpertDown`: `U[e,t,f] * Wdown[e,f,h] -> Y[e,t,h]`

This path does not model sparse token routing. It evaluates the expert compute
communication that follows routing, using the same AccelForge mapping,
collective inference, topology replay, congestion model, and feedback loop as
the matmul experiments.

Submit the expert-FFN sweep:

```bash
PROJECT_DIR=/home/zhangbr/network-topology

sbatch --chdir="$PROJECT_DIR/slurm/logs" \
  "$PROJECT_DIR/slurm/submit_moe_accelforge.sbatch"
```

The wrapper submits `MoE ExpertFFN E16 T{64,256,1024,4096}` across the four core
topologies. Results are written to:

```text
logs/slurm-moe-accelforge-<submit_job>/results.json
```

The retained figure set is:

```text
figures/current_moe_expert_ffn/
```

Figures are generated from completed `results.json` files with:

```bash
.venv/bin/python plot_moe_expert_ffn_results.py \
  logs/slurm-moe-accelforge-<submit_job> \
  --mirror-dir figures/current_moe_expert_ffn
```

The paper uses:

```text
report/figs/moe_expert_ffn_latency_comparison.png
```
