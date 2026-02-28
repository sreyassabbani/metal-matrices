# Benchmark Comparison

- Generated: 2026-02-28 17:58:30 UTC
- Criterion group: `matrix multiplication`
- Baseline: `cpu matrix multiply`
- Significance: permutation test on per-sample ns/op (`alpha=0.05`, `reps=20000`)

| Benchmark | n | Mean (95% CI) | Std dev | Speed vs baseline | p-value vs baseline | Significant | Cliff's delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| cpu matrix multiply | 100 | 9.292 ms (9.274 ms to 9.323 ms) | 134.924 us | 1.000x | - | - | - |
| ndarray multiply | 100 | 312.351 us (310.500 us to 314.865 us) | 11.392 us | 29.750x | 0.0000 | yes | +1.000 (large) |
| gpu matrix multiply | 100 | 4.421 ms (4.149 ms to 4.696 ms) | 1.408 ms | 2.102x | 0.0000 | yes | +0.980 (large) |

_Lower mean time is faster._
