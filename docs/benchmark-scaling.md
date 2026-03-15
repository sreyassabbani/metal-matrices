# Matrix Scaling Benchmarks

- Generated: 2026-03-15 13:23:28 UTC
- Bench suite: `matrix_scaling`
- Metric: Criterion mean runtime (plots show mean lines; 95% CI is in the table)

![Runtime vs n](plots/matrix-scaling-time.png)

![Speedup vs CPU](plots/matrix-scaling-speedup.png)
## n = 128

| backend | μ ± 95% CI | σ | speedup vs cpu |
| --- | --- | ---: | ---: |
| cpu | 876.347 µs ± 6.867 µs | 19.539 µs | 1.000x |
| gpu warm | 335.186 µs ± 58.260 µs | 165.778 µs | 2.615x |
| ndarray | 45.685 µs ± 3.010 µs | 8.847 µs | 19.183x |

## n = 192

| backend | μ ± 95% CI | σ | speedup vs cpu |
| --- | --- | ---: | ---: |
| cpu | 3.810 ms ± 213.533 µs | 608.201 µs | 1.000x |
| gpu warm | 691.847 µs ± 34.712 µs | 98.914 µs | 5.507x |
| ndarray | 138.960 µs ± 1.350 µs | 3.850 µs | 27.419x |

## n = 256

| backend | μ ± 95% CI | σ | speedup vs cpu |
| --- | --- | ---: | ---: |
| cpu | 10.232 ms ± 507.947 µs | 1.460 ms | 1.000x |
| gpu warm | 895.051 µs ± 50.205 µs | 158.250 µs | 11.431x |
| ndarray | 311.320 µs ± 518.331 ns | 1.477 µs | 32.865x |

## n = 320

| backend | μ ± 95% CI | σ | speedup vs cpu |
| --- | --- | ---: | ---: |
| cpu | 15.771 ms ± 172.006 µs | 508.678 µs | 1.000x |
| gpu warm | 836.209 µs ± 159.583 µs | 453.590 µs | 18.860x |
| ndarray | 606.408 µs ± 3.428 µs | 9.990 µs | 26.008x |

## n = 384

| backend | μ ± 95% CI | σ | speedup vs cpu |
| --- | --- | ---: | ---: |
| cpu | 29.516 ms ± 28.019 µs | 83.369 µs | 1.000x |
| gpu warm | 2.970 ms ± 37.254 µs | 105.711 µs | 9.938x |
| ndarray | 1.036 ms ± 769.338 ns | 2.198 µs | 28.503x |

## n = 512

| backend | μ ± 95% CI | σ | speedup vs cpu |
| --- | --- | ---: | ---: |
| cpu | 79.853 ms ± 400.901 µs | 1.157 ms | 1.000x |
| gpu warm | 2.535 ms ± 585.891 µs | 1.673 ms | 31.498x |
| ndarray | 2.457 ms ± 17.506 µs | 50.158 µs | 32.495x |

## n = 768

| backend | μ ± 95% CI | σ | speedup vs cpu |
| --- | --- | ---: | ---: |
| cpu | 265.089 ms ± 1.400 ms | 4.032 ms | 1.000x |
| gpu warm | 10.423 ms ± 804.539 µs | 2.288 ms | 25.433x |
| ndarray | 8.337 ms ± 65.296 µs | 186.982 µs | 31.798x |

## n = 1024

| backend | μ ± 95% CI | σ | speedup vs cpu |
| --- | --- | ---: | ---: |
| cpu | 915.681 ms ± 31.078 ms | 88.799 ms | 1.000x |
| gpu warm | 14.219 ms ± 1.537 ms | 4.382 ms | 64.400x |
| ndarray | 20.394 ms ± 333.540 µs | 949.426 µs | 44.900x |

_Lower runtime is faster._
