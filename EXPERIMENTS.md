# Experiments: Impact of Network Topology on Distributed Inference

## Overview

We study how network topology affects energy and latency for distributed inference across 64 TPU v4 chips. The system uses AccelForge for intra-chip mapping optimization and a custom network topology model with link-level congestion for inter-chip communication cost.

**Accelerator**: TPU v4 (275 TFLOPS BF16, 32 GiB HBM2, 6 ICI links/chip at 45 GB/s each)

**Topologies tested**:
| Topology | Degree | Diameter | Bisection BW |
|---|---|---|---|
| Torus 4x4x4 | 6 | 6 | 2.07 TB/s |
| Mesh 4x4x4 | 3-6 | 9 | 2.16 TB/s |
| Torus 8x2x4 | 6 | 7 | 1.17 TB/s |
| Torus 16x2x2 | 6 | 10 | 0.63 TB/s |
| Ring 64 | 2 | 32 | 0.18 TB/s |

## Congestion Model

The network model routes each collective operation (broadcast, all-reduce) onto physical links, computing per-link byte loads. When multiple transfers happen concurrently, link loads are merged and the bottleneck link (most loaded) determines overall latency.

This is critical for realistic results. Without congestion, a ring (degree 2) appears optimal for all-reduce. With congestion, its 2 links become the bottleneck when broadcast and all-reduce traffic share them. The torus (degree 6) distributes load across 3 independent dimensions.

## Experiment 1: Multi-Workload Comparison (64 chips)

17 workloads ranging from tiny matmuls to multi-layer chains and decode-like matvecs.

| Workload | Net Bytes | Bcast% | Torus 4x4x4 | Mesh 4x4x4 | Ring 64 | Speedup |
|---|---|---|---|---|---|---|
| Tiny 256x256 | 5.03e+07 | 67% | 1.3ms | 2.8ms | 1.5ms | 2.14x |
| Medium 4Kx4K | 1.29e+10 | 67% | 334ms | 716ms | 379ms | 2.14x |
| Wide 4Kx16K (FFN) | 1.03e+11 | 83% | 2482ms | 6299ms | 2661ms | 2.54x |
| Tall 16Kx4K | 3.87e+10 | 56% | 1050ms | 2004ms | 1229ms | 1.91x |
| 3-layer chain 4K | 2.15e+10 | 80% | 525ms | 1289ms | 570ms | 2.45x |
| 5-layer chain 4K | 3.01e+10 | 86% | 716ms | 1861ms | 761ms | 2.60x |
| Decode 1x4096 | 4.30e+09 | 100% | 96ms | 286ms | 96ms | 3.00x |
| Decode 1x16384 | 6.87e+10 | 100% | 1527ms | 4582ms | 1527ms | 3.00x |
| Batch 64tok x 4K | 4.43e+09 | 98% | 99ms | 293ms | 100ms | 2.95x |

**Key finding**: Torus 4x4x4 wins all 17 workloads. Mesh is 1.9-3.0x slower.

## Experiment 2: Scaling Study (8 to 512 chips)

Same 4096x4096 matmul, increasing chip count.

| Chips | Torus | Mesh | Ring | Torus/Mesh |
|---|---|---|---|---|
| 8 | 35.8ms | 35.8ms | 44.7ms | 1.00x |
| 64 | 334ms | 716ms | 379ms | 0.47x |
| 128 | 716ms | 3007ms | 761ms | 0.24x |
| 512 | 2863ms | 12026ms | 3051ms | 0.24x |

**Key finding**: Torus advantage grows with scale. At 512 chips, torus is 4.2x faster than mesh.

## Experiment 3: Torus Aspect Ratio (64 chips)

Same chip count, different 3D arrangements.

| Shape | Diameter | Latency | vs Best |
|---|---|---|---|
| 4x4x4 (cube) | 6 | 334ms | 1.00x |
| 2x4x8 | 7 | 358ms | 1.07x |
| 2x2x16 | 10 | 370ms | 1.11x |
| 64x1x1 (ring) | 32 | 379ms | 1.13x |

**Key finding**: Balanced cube is optimal. Elongated shapes are 7-13% worse.

## Experiment 4: Workload Shape (same FLOPs, different tensor shapes)

| Shape | Torus Latency | Mesh Latency | Torus/Mesh |
|---|---|---|---|
| Square 4Kx4K | 334ms | 716ms | 0.47x |
| Wide 1Kx16K | 1766ms | 5011ms | 0.35x |
| Tall 16Kx4K | 1050ms | 2004ms | 0.52x |

**Key finding**: Wide matmuls (large weight broadcast) show the biggest topology impact. Torus is 2.8x faster than mesh for wide (FFN-like) workloads.

## Experiment 5: GPT-3 6.7B Transformer Layer

Full layer (attention + FFN) decomposed into components on Torus 4x4x4 vs Mesh 4x4x4.

| Component | Torus Latency | Mesh Latency |
|---|---|---|
| Q/K/V projections (3x) | 1002ms | 2148ms |
| Attention QK | 0.3ms | 0.7ms |
| Attention AV | 0.3ms | 0.7ms |
| Z output projection | 103ms | 300ms |
| FFN up (4K->16K) | 103ms | 300ms |
| FFN down (16K->4K) | 103ms | 300ms |
| **Total** | **1312ms** | **3048ms** |

- Torus saves **57% latency** and **41% energy** vs mesh
- Q/K/V projections dominate (76% of network latency) — weight broadcast is the bottleneck
- Attention (QK, AV) is negligible (<0.1% of latency)
- Network is **108x compute energy** (torus) / **184x** (mesh)

## Key Takeaways

1. **Topology choice gives 2-3x latency difference** on the same workload and chip count
2. **The 3D torus (4x4x4) consistently wins** — its 6 links/chip distribute broadcast + all-reduce load across 3 independent dimensions
3. **Broadcast-heavy workloads** (decode, small batch, deep networks) show the largest topology impact
4. **The advantage grows with scale** — at 512 chips, torus is 4.2x faster than mesh
5. **Ring is competitive for all-reduce but bottlenecks on broadcast** — its degree-2 links can't handle concurrent traffic
6. **Balanced torus dimensions are optimal** — 4x4x4 beats elongated shapes like 2x2x16

## How to Reproduce

```bash
# Requires AccelForge installed in its venv
/Users/nikhil/Documents/StudioProjects/accelforge/.venv/bin/python sweep_gpt3.py
```
