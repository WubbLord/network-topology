# Report-Relevant Figures, 2026-05-01

This folder collects the figures most relevant to the current report draft.
It is intentionally grouped by provenance so full AccelForge MAP64 runs are not
mixed with synthetic/reference plots.

## Full AccelForge MAP64 Runs

- `map64_accelforge_moe/`
  - Source: `logs/slurm-moe-map64-rerun-20260501-160133`
  - Setup: `MAP_CHIPS=64`, `EVAL_CHIPS=64`
  - Workloads: ExpertFFN-style MoE runs over token counts 64, 256, 1024, 4096.
  - Use these for the main full-chip AccelForge MoE result.

- `map64_accelforge_gpt3/`
  - Source: `logs/slurm-gpt3-map64-12971957`
  - Setup: `MAP_CHIPS=64`, `EVAL_CHIPS=64`
  - Workload: decomposed GPT-3 175B architecture workload.
  - Use these for the full-chip GPT-3 architecture result.

- `map64_accelforge_large_arch/`
  - Source: `logs/slurm-large-map64-12971958`
  - Setup: `MAP_CHIPS=64`, `EVAL_CHIPS=64`
  - Workloads: large square/tall architecture stress cases.
  - Use these for additional full-chip architecture evidence.

## Brian Tall-Matmul MAP64 Figures

- `brian_tall_matmuls_map64/`
  - Source: remote branch `brian-new` at commit `4c817d15`.
  - Setup reported by that branch: `MAP_CHIPS=64`, `EVAL_CHIPS=64`.
  - Includes tall-matmul summaries, batch-scaling plots, max-link plots, and
    the supporting CSV.

- `brian_tall_matmuls_map8_reference/`
  - Source: same remote branch/commit.
  - Setup: `MAP_CHIPS=8`, `EVAL_CHIPS=64`.
  - Keep as a reference/comparison only.

## Reference Figures

- `synthetic_moe_reference/`
  - Source: existing synthetic MoE sweep.
  - Useful for placement, phase breakdown, and batch-efficiency story.
  - Do not present this as a full AccelForge MAP64 result.

- `topology_model_reference/`
  - Source: existing topology-model plots.
  - Useful for explaining why the circulant helps AllReduce and bottleneck load.

- `existing_unbatched_matmul/` and `existing_batched_matmul/`
  - Source: existing AccelForge-backed matmul/batched sweeps.
  - Useful as older supporting evidence, but Brian's MAP64 tall-matmul figures
    are the better current source for tall/batched MAP64 claims.

- `brian_report_figs/`
  - Source: report-ready figures from Brian's remote branch.
  - These are already named for direct use in the LaTeX draft.
