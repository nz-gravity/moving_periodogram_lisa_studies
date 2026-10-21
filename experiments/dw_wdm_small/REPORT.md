# Small Dynamic Whittle versus WDM benchmark

**Finding after frequency reuse:** The large runtime difference was mostly an
implementation effect. With the same model, prior and NUTS settings, Dynamic
Whittle now takes **29.3 s** versus WDM's **26.3 s** per record: a **1.12×**
ratio, down from 1.77×. Dynamic Whittle is 39% faster than its original
implementation. There is still no resolved accuracy winner on this control.

## Same statistical experiment

Ten paired Gaussian records, seeds 6101–6110, N=8192 and dt=1. The existing
LS2 simulator supplies white innovations, multiplied by a smooth envelope
with `log v(u)=0.7 sin(2πu)+0.2 cos(4πu)`; the interior one-sided PSD is
`2v(u)`. Both methods use the same cubic 12-by-6 tensor spline, knots,
trace-normalized derivative penalties, HalfNormal roughness scales and null
precision. NUTS remains noncentered, dense mass, two sequential chains,
600 warmup and 1000 retained draws per chain, acceptance 0.95 and depth 10.
All raw-record, model, prior/configuration and scoring-grid checks match the
original experiment. [Audit and design](README.md).

Dynamic Whittle retains the original Tang selection, m=32 and thin=2,
without pooling. WDM retains the original Meyer grid, nt=256 and nf=32,
with DC/Nyquist and the end time rows excluded. Unit conversions, likelihood
counts and normalization are unchanged. The only production change is exact
frequency reuse in the scattered spline evaluator; the likelihood formula,
prior and sampler are unchanged.

For Dynamic Whittle the frequency is assigned to each selected time:
`log S(t_p,f(t_p)) = B_time(t_p)^T W B_freq(f(t_p))`.
We compute `W B_freq(f_j)` once for each of the 32 distinct frequencies and
look up the correct projection for every time. All 4064 paired coordinates
remain intact. The coefficient prior penalizes time and frequency roughness
of the underlying 2D field; it does not acquire a separate term at each
observation. Full posterior values, gradients and initialization agree with
the original evaluator. The frequency-to-time mapping is checked for every
retained site, including both gap patterns.

## Ten-seed results

Mean ± seed standard deviation; every primary result is retained.

| Method | Integrated log-PSD squared error | 90% coverage | Median 90% width | Preprocessing (ms) | Inference (s) | Observations |
|---|---:|---:|---:|---:|---:|---:|
| Dynamic Whittle, frequency reuse | 0.001470 ± 0.000429 | 90.8% ± 5.8% | 0.484 ± 0.025 | 0.96 ± 0.61 | 29.3 ± 0.9 | 4064 complex |
| WDM | 0.001324 ± 0.000425 | 94.1% ± 4.8% | 0.495 ± 0.026 | 2.33 ± 0.99 | 26.3 ± 0.6 | 7874 real |

Error integrates squared natural-log posterior-median error over the common
interior grid, u=0.05…0.95 and f=0.025…0.475 Hz. Divide by area 0.405 for
mean squared error. Coverage uses the same quadrature weights; width is the
spatial median of q95(S)−q05(S), in data-unit squared/Hz. Spatial cells are
correlated; there are ten independent calibration records.

The paired normalized-error difference, DW minus WDM, is 0.000359, with an
exploratory 95% t interval **−0.000039 to 0.000758**. WDM has lower error on
seven seeds. DW's average inclusion is closer to 90%, with slightly narrower
intervals. This does not establish a general accuracy or calibration winner.
WDM posterior surfaces are identical to the originals; DW's median log-PSD
changes have average RMS 0.00204, consistent with a numerically equivalent
target producing different finite NUTS trajectories.

![True PSD and posterior estimates after frequency reuse](results/reuse_rerun/comparison.png)

[Per-seed table](results/reuse_rerun/metrics.csv),
[aggregates](results/reuse_rerun/aggregate.json),
[PDF](results/reuse_rerun/comparison.pdf) and
[original results](results/metrics.csv) retain the evidence.

## Runtime and sampler checks

The 20 primary fits total **555.9 s**. Inference includes compilation,
warmup and sampling; model preparation is separate, averaging 17.4 / 2.8 ms
for DW/WDM. Retained draws average 31.91 / 31.02 leapfrog steps. Warm
likelihood-and-weight gradients take 60.4 µs with the original DW evaluator,
33.7 µs with frequency reuse and 23.7 µs for WDM's grid. These gradient
timings omit priors and sampler overhead and cannot allocate the entire
wall-clock difference. [Controlled timing](results/reuse_rerun/runtime_check.json).

DW passes all ten primary sampler screens. WDM passes eight: one divergence
on 6103 and Rhat 1.010685 on 6110. Both methods were rerun on both flagged
seeds with the original follow-up settings, 1000 warmup, 1500 draws and
acceptance 0.99. All four checks pass; median log-PSD changes have RMS below
0.00215. The primary table is unchanged. Charging only required follow-ups
gives **293.2 s for DW versus 337.5 s for WDM**, so DW costs about 13% less
under this particular screening policy. This is conditional on two flags
in ten records, not a universal runtime ranking.
[Longer checks](results/reuse_rerun/repair_summary.json).

**29 production checks and 20 experiment checks pass.** Independent
quadrature verifies all twenty primary metrics, and source snapshots and
hashes retain the before/after implementation.
[Validation](results/reuse_rerun/validation.json) and
[provenance](results/reuse_rerun/provenance.json).

## Limits and next study

The existing normalization and dependence audits still apply: effective time
widths and frequency grids are close rather than identical, and the
likelihoods retain their explicit approximations. This is one smooth
broadband-amplitude control. The next bounded study should add changing-shape
LS2 spectra and repeated fixed-gap records, preserving the current evaluator
reuse, model and screening policy. A large campaign or default-method change
is premature. In the one central-gap record, DW's gap-region PSD overshoot
is larger: 13.1% versus WDM's 6.6%, and its log-PSD MSE is 3.7 times larger.
That negative result merits a repeated gap study before claiming robust
calibration. [Updated whole-column gap pilot](GAPS.md).
