# Small Batched Matmul Summary

Completed topology results: 24
Complete four-topology workload blocks: 6

| Workload | Best topology | Latency | Max-link term | Iterations | Worst/best |
|---|---|---:|---:|---:|---:|
| 2Kx8K | Circulant {1,5,17} | 16.9ms | 16.9ms | 2 | 3.01x |
| 4Kx16K | Circulant {1,5,17} | 3.9ms | 3.9ms | 2 | 3.05x |
| B16 2Kx8K | Circulant {1,5,17} | 3.9ms | 3.9ms | 2 | 3.05x |
| B16 4Kx16K | Circulant {1,5,17} | 62.6ms | 62.6ms | 2 | 3.05x |
| B256 2Kx8K | Circulant {1,5,17} | 1.3s | 1.3s | 2 | 3.02x |
| B256 4Kx16K | Circulant {1,5,17} | 1s | 1s | 2 | 3.05x |
