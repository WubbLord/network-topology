# Higher-TPU Realistic Large-Slice Circulant Notes

This branch now includes a theoretical extension of the circulant idea to
realistic TPU slices larger than 64 chips. This is the result to use when the
claim is "circulants can scale beyond our 64-chip AccelForge setup."

## What Was Run

Run:

```bash
.venv/bin/python analyze_higher_tpu_circulants.py \
  --scenario realistic-large-slices \
  --out-dir logs/higher_tpu_realistic_slices_20260501 \
  --samples 128
```

Generated figures and CSVs are in
`logs/higher_tpu_realistic_slices_20260501/`. The important figures are:

- `gpt3_slice_speedups.png`: torus/circulant latency ratio across supported
  larger TPU slices for the GPT-3 MAP64 trace replay.
- `moe_slice_speedups.png`: the same ratio for the MoE ExpertFFN T1024 MAP64
  trace replay.
- `gpt3_higher_tpu_replay.png` and `moe_higher_tpu_replay.png`: absolute
  serialized network latencies on a log scale.

The replay still uses existing AccelForge-derived MAP64 communication traces.
So this is not a claim that AccelForge remapped the workload at 2048 or 6144
chips. It is a topology-theory plus congestion-model extension over realistic
slice sizes.

## Realistic Slices Included

The run uses public supported slices:

- v5e/v6e: `8x16` 128-chip slices and `16x16` 256-chip full-pod slices.
- v4: `4x4x8`, `4x8x8`, `8x8x8`, `8x8x16`, and `8x16x16`.
- v5p: `4x4x8`, `4x8x8`, `8x8x8`, `8x8x16`, `8x16x16`,
  `16x16x16`, and `16x16x24`.
- TPU7x: `4x4x8`, `4x8x8`, `8x8x8`, `8x8x16`, and `8x16x16`.

## Searched Circulant Families

The extension keeps the same hardware budget as the baseline slice: degree 4
for 2D-torus generations and degree 6 for 3D-torus generations. Every generator
is coprime to the chip count, so each generator forms a Hamiltonian cycle and
can be used as an independent collective ring.

| Slice size | Baseline shape | Searched circulant | Diameter | L1 lower bound |
|---:|---|---|---:|---:|
| 128 | `8x16` degree-4 | `C(128,{3,13})` | 8 | 8 |
| 256 | `16x16` degree-4 | `C(256,{15,29})` | 12 | 11 |
| 128 | `4x4x8` degree-6 | `C(128,{3,5,51})` | 5 | 4 |
| 256 | `4x8x8` degree-6 | `C(256,{7,19,41})` | 7 | 6 |
| 512 | `8x8x8` degree-6 | `C(512,{45,169,223})` | 8 | 7 |
| 1024 | `8x8x16` degree-6 | `C(1024,{113,245,269})` | 10 | 9 |
| 2048 | `8x16x16` degree-6 | `C(2048,{405,505,739})` | 13 | 12 |
| 4096 | `16x16x16` degree-6 | `C(4096,{421,653,1783})` | 17 | 15 |
| 6144 | `16x16x24` degree-6 | `C(6144,{1,23,373})` | 19 | 17 |

For all 3D slices up to 2048 chips, the searched circulant is within one hop of
the L1 lattice lower bound. The 4096 and 6144-chip v5p slices are within two
hops in this heuristic search.

## Main Result

The 3D large-slice circulants preserve or slightly improve the relative latency
advantage over torus:

| Slice family | GPT-3 MAP64 replay | MoE T1024 MAP64 replay |
|---|---:|---:|
| v4 128-512 chip slices | 2.01-2.02x | 2.63-2.65x |
| v4 1024-2048 chip slices | 2.12-2.13x | 2.81x |
| v5p 128-512 chip slices | 2.01-2.02x | 2.63-2.64x |
| v5p 1024-6144 chip slices | 2.12-2.16x | 2.81-2.87x |
| TPU7x 128-512 chip slices | 2.01-2.02x | 2.63-2.64x |
| TPU7x 1024-2048 chip slices | 2.12x | 2.81x |

The 2D v5e/v6e degree-4 case is still positive but weaker: about `1.63x` on
GPT-3 replay and `1.88-1.89x` on MoE replay. That is expected because there are
only two Hamiltonian cycles instead of three.

## Interpretation

The result supports a clean extension story: the circulant design is not just a
64-chip trick. For supported 3D TPU slices larger than 64 chips, degree-6
circulants can be searched with the same Hamiltonian-cycle constraint, remain
close to a graph-theoretic diameter lower bound, and keep the congestion-model
latency advantage on the same AccelForge-derived communication traces.

This should be presented as theoretical evidence plus trace replay, not as a
large-N AccelForge remapping result.

