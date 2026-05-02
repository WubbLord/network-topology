# Square Matmul Sweep Summary

Completed topology results: 32
Complete four-topology workload blocks: 8

| Workload | Best topology | Latency | Max-link term | Iterations | Worst/best |
|---|---|---:|---:|---:|---:|
| Square 4Kx2K | Circulant {1,5,17} | 491.3us | 489.3us | 2 | 3.04x |
| Square 4Kx4K | Circulant {1,5,17} | 7.9ms | 7.9ms | 2 | 3.01x |
| Square 16Kx2K | Circulant {1,5,17} | 2.0ms | 2.0ms | 2 | 3.05x |
| Square 16Kx4K | Circulant {1,5,17} | 3.9ms | 3.9ms | 2 | 3.05x |
| Square 64Kx2K | Circulant {1,5,17} | 7.8ms | 7.8ms | 2 | 3.05x |
| Square 64Kx4K | Circulant {1,5,17} | 15.7ms | 15.7ms | 2 | 3.05x |
| Square 256Kx2K | Circulant {1,5,17} | 31.3ms | 31.3ms | 2 | 3.05x |
| Square 256Kx4K | Circulant {1,5,17} | 62.6ms | 62.6ms | 2 | 3.05x |
