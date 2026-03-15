# Benchmark Comparison

- Generated: 2026-03-15 13:19:39 UTC
- Criterion group: `matrix multiplication`
- Baseline: `cpu matrix multiply`
- Significance: permutation test on per-sample ns/op (`alpha=0.05`, `reps=20000`)

| Benchmark | samples | μ ± 95% CI | σ | Speed vs baseline | p-value vs baseline | Significant | Cliff's delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| cpu matrix multiply | 100 | 11.543 ms ± 436.503 µs | 2.243 ms | 1.000x | - | - | - |
| ndarray multiply | 100 | 323.089 µs ± 4.589 µs | 24.640 µs | 35.727x | 0.0000 | yes | +1.000 (large) |
| gpu warm matrix multiply | 100 | 865.912 µs ± 52.582 µs | 269.641 µs | 13.330x | 0.0000 | yes | +1.000 (large) |
| gpu cold init + multiply | 100 | 1.616 ms ± 70.550 µs | 362.486 µs | 7.144x | 0.0000 | yes | +1.000 (large) |

_Lower mean time is faster._
