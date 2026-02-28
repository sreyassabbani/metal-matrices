# Benchmark Comparison

- Generated: 2026-02-28 19:16:18 UTC
- Criterion group: `matrix multiplication`
- Baseline: `cpu matrix multiply`
- Significance: permutation test on per-sample ns/op (`alpha=0.05`, `reps=20000`)

| Benchmark | n | μ ± 95% CI | σ | Speed vs baseline | p-value vs baseline | Significant | Cliff's delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| cpu matrix multiply | 100 | 9.809 ms ± 263.835 µs | 1.382 ms | 1.000x | - | - | - |
| ndarray multiply | 100 | 411.337 µs ± 20.025 µs | 102.513 µs | 23.846x | 0.0000 | yes | +1.000 (large) |
| gpu matrix multiply | 100 | 2.894 ms ± 160.269 µs | 821.096 µs | 3.390x | 0.0000 | yes | +1.000 (large) |

_Lower mean time is faster._
