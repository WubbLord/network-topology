# Tall Matmul Sweep Summary

Completed topology results: 192
Observed workload/MAP cases: 48
Complete four-topology blocks: 48

| MAP | Workload | Best topology | Latency | Max-link term | Worst/best |
|---:|---|---|---:|---:|---:|
| 8 | 4Kx2K | Circulant {1,5,17} | 2.0ms | 2.0ms | 3.05x |
| 8 | 4Kx4K | Circulant {1,5,17} | 7.8ms | 7.8ms | 3.05x |
| 8 | 16Kx2K | Circulant {1,5,17} | 85.0ms | 85.0ms | 3.02x |
| 8 | 16Kx4K | Circulant {1,5,17} | 31.3ms | 31.3ms | 3.05x |
| 8 | 64Kx2K | Circulant {1,5,17} | 322.1ms | 322.1ms | 3.02x |
| 8 | 64Kx4K | Circulant {1,5,17} | 656.2ms | 656.2ms | 3.02x |
| 8 | 256Kx2K | Circulant {1,5,17} | 1.3s | 1.3s | 3.02x |
| 8 | 256Kx4K | Circulant {1,5,17} | 2.6s | 2.6s | 3.02x |
| 8 | B16 4Kx2K | Circulant {1,5,17} | 322.1ms | 322.1ms | 3.02x |
| 8 | B16 4Kx4K | Circulant {1,5,17} | 656.2ms | 656.2ms | 3.02x |
| 8 | B16 16Kx2K | Circulant {1,5,17} | 1.3s | 1.3s | 3.02x |
| 8 | B16 16Kx4K | Circulant {1,5,17} | 2.6s | 2.6s | 3.02x |
| 8 | B16 64Kx2K | Circulant {1,5,17} | 5.1s | 5.1s | 3.02x |
| 8 | B16 64Kx4K | Circulant {1,5,17} | 10s | 10s | 3.02x |
| 8 | B16 256Kx2K | Circulant {1,5,17} | 20s | 20s | 3.02x |
| 8 | B16 256Kx4K | Circulant {1,5,17} | 40s | 40s | 3.02x |
| 8 | B256 4Kx2K | Circulant {1,5,17} | 5.1s | 5.1s | 3.02x |
| 8 | B256 4Kx4K | Circulant {1,5,17} | 10s | 10s | 3.02x |
| 8 | B256 16Kx2K | Circulant {1,5,17} | 20s | 20s | 3.02x |
| 8 | B256 16Kx4K | Circulant {1,5,17} | 40s | 40s | 3.02x |
| 8 | B256 64Kx2K | Circulant {1,5,17} | 1.3m | 1.3m | 3.02x |
| 8 | B256 64Kx4K | Circulant {1,5,17} | 2.7m | 2.7m | 3.02x |
| 8 | B256 256Kx2K | Circulant {1,5,17} | 5.4m | 5.4m | 3.02x |
| 8 | B256 256Kx4K | Circulant {1,5,17} | 11m | 11m | 3.02x |
| 64 | 4Kx2K | Circulant {1,5,17} | 246.7us | 244.7us | 3.04x |
| 64 | 4Kx4K | Circulant {1,5,17} | 7.9ms | 7.9ms | 3.01x |
| 64 | 16Kx2K | Circulant {1,5,17} | 246.7us | 244.7us | 3.04x |
| 64 | 16Kx4K | Circulant {1,5,17} | 980.7us | 978.7us | 3.05x |
| 64 | 64Kx2K | Circulant {1,5,17} | 40.3ms | 40.3ms | 3.02x |
| 64 | 64Kx4K | Circulant {1,5,17} | 980.7us | 978.7us | 3.05x |
| 64 | 256Kx2K | Circulant {1,5,17} | 158.8ms | 158.8ms | 3.02x |
| 64 | 256Kx4K | Circulant {1,5,17} | 62.6ms | 62.6ms | 3.05x |
| 64 | B16 4Kx2K | Circulant {1,5,17} | 246.7us | 244.7us | 3.04x |
| 64 | B16 4Kx4K | Circulant {1,5,17} | 980.7us | 978.7us | 3.05x |
| 64 | B16 16Kx2K | Circulant {1,5,17} | 158.8ms | 158.8ms | 3.02x |
| 64 | B16 16Kx4K | Circulant {1,5,17} | 319.1ms | 319.1ms | 3.02x |
| 64 | B16 64Kx2K | Circulant {1,5,17} | 633.1ms | 633.1ms | 3.02x |
| 64 | B16 64Kx4K | Circulant {1,5,17} | 1.3s | 1.3s | 3.02x |
| 64 | B16 256Kx2K | Circulant {1,5,17} | 2.5s | 2.5s | 3.02x |
| 64 | B16 256Kx4K | Circulant {1,5,17} | 5.1s | 5.1s | 3.02x |
| 64 | B256 4Kx2K | Circulant {1,5,17} | 633.1ms | 633.1ms | 3.02x |
| 64 | B256 4Kx4K | Circulant {1,5,17} | 1.3s | 1.3s | 3.02x |
| 64 | B256 16Kx2K | Circulant {1,5,17} | 2.5s | 2.5s | 3.02x |
| 64 | B256 16Kx4K | Circulant {1,5,17} | 5.1s | 5.1s | 3.02x |
| 64 | B256 64Kx2K | Circulant {1,5,17} | 10s | 10s | 3.02x |
| 64 | B256 64Kx4K | Circulant {1,5,17} | 20s | 20s | 3.02x |
| 64 | B256 256Kx2K | Circulant {1,5,17} | 40s | 40s | 3.02x |
| 64 | B256 256Kx4K | Circulant {1,5,17} | 1.3m | 1.3m | 3.02x |
