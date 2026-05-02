# MoE Expert-FFN Sweep Summary

Completed topology results: 16
Complete four-topology workload blocks: 4

| Workload | Best topology | Circulant latency | Torus/Circ | Mesh/Circ | Ring/Circ |
|---|---|---:|---:|---:|---:|
| T64 | Circulant {1,5,17} | 0.736 ms | 2.28x | 3.05x | 3.01x |
| T256 | Circulant {1,5,17} | 2.938 ms | 2.29x | 3.05x | 3.00x |
| T1024 | Circulant {1,5,17} | 11.746 ms | 2.29x | 3.05x | 3.00x |
| T4096 | Circulant {1,5,17} | 46.978 ms | 2.29x | 3.05x | 3.00x |

All completed rows converge in two feedback iterations. The inferred network phase is one `REDUCE_SCATTER` on `ExpertDown:Y`; `ExpertUp` is local in the mapper output.
