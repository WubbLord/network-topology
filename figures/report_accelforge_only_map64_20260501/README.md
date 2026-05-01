# AccelForge-Only MAP64 Report Figures, 2026-05-01

Use this folder for report figures if the report should only cite experiments
that went through the AccelForge mapper with `MAP_CHIPS=64` and
`EVAL_CHIPS=64`.

Excluded on purpose:

- Synthetic sparse-MoE dispatch/combine figures.
- Direct transformer stress-test figures.
- Topology-model-only explanatory figures.
- MAP8 reference plots.
- Brian's report-ready MoE figure, because that one comes from the synthetic
  sparse-MoE path.

## Included Subfolders

- `map64_accelforge_moe/`
  - Source: `logs/slurm-moe-map64-rerun-20260501-160133`
  - Full AccelForge MoE ExpertFFN approximation with `MAP_CHIPS=64`,
    `EVAL_CHIPS=64`.
  - Main figure: `moe_map64_latency_comparison.png`.

- `map64_accelforge_gpt3/`
  - Source: `logs/slurm-gpt3-map64-12971957`
  - Full AccelForge GPT-3 175B decomposition with `MAP_CHIPS=64`,
    `EVAL_CHIPS=64`.
  - Main figure: `gpt3_map64_latency_comparison.png`.

- `map64_accelforge_large_arch/`
  - Source: `logs/slurm-large-map64-12971958`
  - Full AccelForge large architecture workload sweep with `MAP_CHIPS=64`,
    `EVAL_CHIPS=64`.
  - Main figure: `large_arch_map64_latency_comparison.png`.

- `brian_tall_matmuls_map64/`
  - Source: remote branch `brian-new` at commit `4c817d15`.
  - Reported setup: `MAP_CHIPS=64`, `EVAL_CHIPS=64`.
  - Main figures: `3_latency_comparison_B*.png`,
    `10_max_link_serialization.png`, and
    `11_batch_scaling_max_link_serialization.png`.

## MoE Note

For MoE claims in an AccelForge-only report, use only
`map64_accelforge_moe/`. The synthetic sparse-MoE figures in
`figures/report_relevant_map64_20260501/synthetic_moe_reference/` are useful
for design exploration, but they should not be cited as AccelForge mapper
results.
