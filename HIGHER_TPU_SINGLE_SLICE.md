# Higher-TPU Single-Slice Circulant Notes

This branch keeps the higher-TPU what-if experiments on supported 64-chip
single slices, not full pod footprints. That matches the MAP64 AccelForge traces
used elsewhere in the repo.

## Why v4 Is Both 64 And 4096 Chips

Google uses "TPU v4 Pod" for the full system, which is 4096 chips. Our project
simulates a TPU v4-like 64-chip slice: the `v4-128`/`4x4x4` large topology in
the Cloud TPU v4 configuration table. So the paper should say "64-chip v4
slice" or "v4-like 4x4x4 slice" when describing our simulations, and reserve
"v4 Pod" for the full 4096-chip system.

## Single-Slice Systems Modeled

The single-slice replay uses these public configurations:

| Generation | Modeled slice | Baseline topology | Link model |
|---|---:|---|---:|
| TPU v5e | 64 chips | 2D torus `8x8` | 100 GB/s/link |
| TPU v6e | 64 chips | 2D torus `8x8` | 200 GB/s/link |
| TPU v4 | 64 chips | 3D torus `4x4x4` | 45 GB/s/link |
| TPU v5p | 64 chips | 3D torus `4x4x4` | 200 GB/s/link |
| TPU7x Ironwood | 64 chips | 3D torus `4x4x4` | 200 GB/s/link |

For v5e/v6e, the 64-chip slice is the public `8x8` 2D slice. For v4/v5p/TPU7x,
the 64-chip slice is one `4x4x4` cube. The v5p and TPU7x full pods are much
larger, but those full-pod chip counts are not used by the default run.

## Results

Run:

```bash
.venv/bin/python analyze_higher_tpu_circulants.py \
  --scenario single-slice64 \
  --out-dir logs/higher_tpu_single_slice_circulants_20260501 \
  --samples 128
```

Generated files are in `logs/higher_tpu_single_slice_circulants_20260501/`.
Because `logs/` is ignored, this markdown file records the report-relevant
numbers.

| System | Torus/Circulant on GPT-3 MAP64 | Torus/Circulant on MoE T1024 MAP64 |
|---|---:|---:|
| TPU v5e 64-chip slice | 1.55x | 1.78x |
| TPU v6e 64-chip slice | 1.55x | 1.78x |
| TPU v4 one-cube slice | 1.79x | 2.29x |
| TPU v5p one-cube slice | 1.79x | 2.29x |
| TPU7x Ironwood one-cube slice | 1.79x | 2.29x |

The searched degree-4 circulant for 2D 64-chip slices is `C(64,{3,7})`, with
diameter 6 versus the `8x8` torus diameter 8. For 3D 64-chip slices, the
branch keeps the report design `C(64,{1,5,17})`, which ties the best searched
diameter and average-hop score: diameter 4 versus the `4x4x4` torus diameter 6.

## Source Notes

- Cloud TPU v4 docs: full v4 Pod is 4096 chips; `v4-128` is a 64-chip
  `4x4x4` topology.
- Cloud TPU v5e docs: v5e has 4 ICI ports/chip, 400 GB/s bidirectional ICI per
  chip, 2D torus topology, and an `8x8` 64-chip slice.
- Cloud TPU v6e docs: v6e has 4 ICI ports/chip, 800 GB/s bidirectional ICI per
  chip, 2D torus topology, and an `8x8` 64-chip slice.
- Cloud TPU v5p docs: full v5p Pod is 8960 chips, but `4x4x4` is a supported
  64-chip one-cube slice with full 3D torus connectivity.
- Cloud TPU7x docs: full TPU7x pod is 9216 chips, but `4x4x4` is a supported
  64-chip slice; TPU7x uses 3D torus ICI with 1200 GB/s bidirectional ICI per
  chip and 200 GB/s per axis.

Official docs used: <https://docs.cloud.google.com/tpu/docs/v4>,
<https://docs.cloud.google.com/tpu/docs/v5e>,
<https://docs.cloud.google.com/tpu/docs/v6e>,
<https://docs.cloud.google.com/tpu/docs/v5p>, and
<https://docs.cloud.google.com/tpu/docs/tpu7x>.
