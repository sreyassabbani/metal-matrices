# Benchmark Comparison

- Generated: 2026-03-15 16:01:43 UTC
- Criterion group: `matrix multiplication`
- Baseline: `cpu matrix multiply`
- Significance: permutation test on per-sample ns/op (`alpha=0.05`, `reps=20000`)

| Benchmark | samples | μ ± 95% CI | σ | Speed vs baseline | p-value vs baseline | Significant | Cliff's delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| cpu matrix multiply | 100 | 9.530 ms ± 104.859 µs | 560.660 µs | 1.000x | - | - | - |
| ndarray multiply | 100 | 322.518 µs ± 3.484 µs | 18.249 µs | 29.549x | 0.0000 | yes | +1.000 (large) |
| gpu warm matrix multiply | 100 | 419.162 µs ± 22.853 µs | 117.048 µs | 22.736x | 0.0000 | yes | +1.000 (large) |
| gpu cold init + multiply | 100 | 1.003 ms ± 37.221 µs | 192.053 µs | 9.498x | 0.0000 | yes | +1.000 (large) |

_Lower mean time is faster._
