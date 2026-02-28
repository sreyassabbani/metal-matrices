# Matrix Scaling Benchmarks

- Generated: 2026-02-28 19:49:17 UTC
- Bench suite: `matrix_scaling`
- Metric: Criterion mean runtime (plots show mean lines; 95% CI is in the table)

![Runtime vs n](plots/matrix-scaling-time.png)

![Speedup vs CPU](plots/matrix-scaling-speedup.png)

| n | backend | μ ± 95% CI | σ | speedup vs cpu |
| ---: | --- | --- | ---: | ---: |
| 128 | cpu | 844.324 µs ± 826.263 ns | 2.352 µs | 1.000x |
| 128 | gpu | 3.350 ms ± 160.770 µs | 454.624 µs | 0.252x |
| 128 | ndarray | 42.894 µs ± 439.552 ns | 1.260 µs | 19.684x |
| 256 | cpu | 9.406 ms ± 87.247 µs | 249.487 µs | 1.000x |
| 256 | gpu | 6.959 ms ± 411.260 µs | 1.170 ms | 1.352x |
| 256 | ndarray | 359.238 µs ± 36.008 µs | 102.867 µs | 26.182x |
| 512 | cpu | 79.019 ms ± 669.608 µs | 1.929 ms | 1.000x |
| 512 | gpu | 9.493 ms ± 752.020 µs | 2.144 ms | 8.324x |
| 512 | ndarray | 2.515 ms ± 45.427 µs | 129.964 µs | 31.422x |
| 1024 | cpu | 860.202 ms ± 14.558 ms | 42.894 ms | 1.000x |
| 1024 | gpu | 11.094 ms ± 355.336 µs | 1.019 ms | 77.538x |
| 1024 | ndarray | 19.481 ms ± 159.503 µs | 454.108 µs | 44.155x |

_Lower runtime is faster._
