# Selected-bin preprocessing benchmark

Baseline: `aed4b19c42e26f84b83ef2c1201feeae30ea31c6`.

Median of warmed, alternating-order rounds; NumPy calls are synchronous. Setup is separate. No data I/O, reference projection, or sampling is timed.

| Workload | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|
| Coordinates, N=8192 | 0.448 | 0.015 | 29.64x |
| Coordinates, N=32768 | 1.630 | 0.033 | 48.81x |
| Hann, N=8192, sparse bins, complete | 1.591 | 1.087 | 1.46x |
| Hann, N=8192, all bins, complete | 3.503 | 3.074 | 1.14x |
| Hann, N=8192, sparse bins, gapped | 1.398 | 0.948 | 1.47x |
| Hann, N=32768, sparse bins, complete | 5.668 | 4.013 | 1.41x |
| Hann, N=32768, all bins, complete | 14.105 | 12.254 | 1.15x |
| Hann, N=32768, sparse bins, gapped | 5.338 | 3.697 | 1.44x |
| LISA transform | 812.155 | 284.370 | 2.86x |
| LISA plan setup | 27.382 | 9.448 | 2.90x |

- Core coefficients, schedule coordinates, and Hann observation arrays match the baseline bit for bit.
- Replayed all 69 accepted LISA cycles (70,587 coefficients) from the hash-checked original data context.
- LISA computes 1,023 coefficients per cycle instead of 7,680; original time/frequency pairs and normalization are preserved.
- Only requested moving-window updates and outputs are computed. The shared first-window DFT retains its original full grid: pruning that FFT gave no measurable speed benefit and increased roundoff in weak coefficients.
- Coefficient relative L2 difference: 7.93e-14; maximum relative difference: 9.82e-10.
- Maximum absolute log-likelihood change across four parameter states: 1.19e-09.
- Maximum absolute parameter-gradient change: 5.14e-10 (all 112 field parameters checked).
- LISA equality is numerical, not bitwise: pruning changes FFT roundoff. Coefficients were checked with relative tolerance 1e-8 and a scale-dependent absolute floor of 1e-12 times their RMS; likelihoods and gradients with rtol=1e-9, atol=1e-7.

These are local preprocessing timings, not posterior-sampling speedups. Full measurements, source hashes, versions, and individual timing rounds are in `benchmark.json`.

Reproduce from the package root:

```bash
.venv/bin/python -m experiments.dw_wdm_small.benchmark_selected_bins --baseline-ref aed4b19c42e26f84b83ef2c1201feeae30ea31c6 --lisa
```
