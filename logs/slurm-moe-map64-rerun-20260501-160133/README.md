# Curated MoE MAP64 AccelForge Logs

This directory is the MoE experiment bundle to cite for the report. It contains
the full `MAP_CHIPS=64`, `eval_chips=64` AccelForge mapping + topology feedback
runs for `MoE ExpertFFN E16` at token counts `T64`, `T256`, `T1024`, and
`T4096`.

## Why This Bundle

- These runs use the AccelForge mapper with 64 simulation chips, not the earlier
  MAP8 scaling shortcut.
- Each workload/topology pair was run independently through iterative feedback.
- Each final topology result converged in two feedback iterations.
- The communication extracted from the mapper is a single `REDUCE_SCATTER`
  transfer on tensor `ExpertDown:Y`, which is why the topology ratios are
  stable across token counts.

## Files

- `results.json`: aggregate result file used for report tables/figures.
- `milestone2_analysis.json`: aggregate plotting/analysis artifact.
- `MoE_ExpertFFN_E16_T*__*.json`: per-workload/per-topology raw result files.
- `mappings/`: AccelForge mapping YAMLs for each feedback iteration.
- Corresponding Slurm stdout/stderr files are committed under `slurm/logs/`:
  - `moe-af-sweep-12972407-{0,1}.*`
  - `moe-af-sweep-12972699-{2,3}.*`
  - `moe-af-sweep-12972788-{4..15}.*`

## Report-Relevant Results

| Workload | Bytes | Circulant | Torus | Mesh | Ring | Torus/Circ | Mesh/Circ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MoE ExpertFFN E16 T64` | 100,663,296 | 0.736 ms | 1.681 ms | 2.241 ms | 2.218 ms | 2.28x | 3.05x |
| `MoE ExpertFFN E16 T256` | 402,653,184 | 2.938 ms | 6.714 ms | 8.952 ms | 8.824 ms | 2.29x | 3.05x |
| `MoE ExpertFFN E16 T1024` | 1,610,612,736 | 11.746 ms | 26.847 ms | 35.796 ms | 35.248 ms | 2.29x | 3.05x |
| `MoE ExpertFFN E16 T4096` | 6,442,450,944 | 46.978 ms | 107.377 ms | 143.170 ms | 140.945 ms | 2.29x | 3.05x |

## Interpretation

The batch/token scaling is close to linear because the extracted communication
phase stays the same and serialization dominates the modeled first-byte hop
latency. The circulant advantage is strongest here because the MoE trace maps to
`REDUCE_SCATTER`, where the `C(64,{1,5,17})` Hamiltonian-cycle decomposition
splits load over three independent rings while the `4x4x4` torus dimension-wise
schedule bottlenecks on shorter dimension rings.

