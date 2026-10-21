# Frequency reuse and whole-column WDM gap pilot

**Follow-up:** The [ten-record taper study](taper_study/REPORT.md) now compares
Hann windows, matched rectangular windows, the original moving periodogram
and these guarded WDM outputs. It shows that the single-record ordering below
does not describe the repeated gap comparison. Hann improves coloured-noise
expected powers, but does not improve posterior accuracy over its matched
rectangular control in the broadband gap study.

## The time-frequency pairing

Dynamic Whittle observes paired sites `(t_p, f_p)`, where `f_p=f(t_p)` is the
frequency selected at that time. Its predictor is
`eta_p = B_time(t_p)^T W B_freq(f(t_p))`.
Frequency reuse computes a projection for each distinct frequency and then
uses the projection indexed by `f(t_p)` at every time. It neither replaces
these sites with a Cartesian grid nor averages their times or frequencies.
The existing tensor coefficient prior and its time/frequency roughness
penalties remain unchanged. Full posterior density, gradients, initialization
and the exact frequency-to-time mapping are verified against the original
implementation.

The ten-seed primary comparison is now **29.3 / 26.3 seconds** for DW/WDM,
versus the original 48.4 / 27.3 seconds. Warm PSD-plus-likelihood gradients
are **60.4 / 33.7 / 23.7 microseconds** for original DW, frequency-reusing DW
and WDM grid respectively. The remaining difference is modest and depends
on the evaluators and sampler protocol.
[Primary report](REPORT.md) and
[timing evidence](results/reuse_rerun/runtime_check.json).

## Gap treatment

The original raw missing-sample masks and sample coordinates are preserved.
For Dynamic Whittle, keep the original thinned selection and retain an
ordinate only when all 65 samples in its window are observed. Retained powers
and counts are exactly unchanged from the ungapped record and are invariant
to hidden missing values. No samples are concatenated across gaps.

WDM now uses **whole time columns**: retain a column only if every interior
frequency coefficient has at least 99.99% of its squared analysis-kernel
energy on observed samples. Retained powers still use the same per-coefficient
retained-energy normalization. Excluded columns have zero power/count at
every frequency on the original grid. The prior and sampler are unchanged.

The central gap leaves **209 complete columns, 45 empty columns and zero
partial columns**, giving **6479 coefficients**. The former per-coefficient
rule retained 6508: removing the two partial columns costs 29 coefficients.
Dots in the figure mean coefficients used in the fit. The gray band marks
the raw outage; the wider blank WDM area is the additional kernel guard.
[Mask audit](gaps/column_gated/audit.json).

| Approximately 10% raw data missing | DW retained observations | WDM retained coefficients |
|---|---:|---:|
| One central 819-sample gap | 3623 / 4064 = 89.1% | 6479 / 7874 = 82.3% |
| Sixteen 51-sample gaps | 3144 / 4064 = 77.4% | 403 / 7874 = 5.1% |

The fragmented mask retains 13 complete WDM columns. This severe loss is
strongly guard-dependent: allowing 0.1% or 1% lost energy instead keeps
2542 or 4495 coefficients. It is not an intrinsic WDM information-loss
claim. Meyer time tails make center-only gating unsafe: that naive rule
leaves expected-power suppression up to 43.5% / 48.4% for these masks.
The strict guard and normalization leave maximum pointwise mean error below
0.034% on this known Gaussian amplitude control. This remains an approximate
guarded likelihood, not a general covariance-aware treatment of gapped
or coloured data. Mean correction alone does not establish calibration.

## One paired central-gap rerun

Seed 6101 uses the same model, prior and primary sampler settings as the
ungapped rerun. Both fits pass, with no divergences/depth hits, Rhat below
1.006 and bulk/tail ESS above 892/1043.

| Method | Log-PSD MSE | Spatial truth inclusion in 90% intervals | Median width | Inference (s) |
|---|---:|---:|---:|---:|
| Dynamic Whittle, frequency reuse | 0.004814 | 91.2% | 0.615 | 29.1 |
| WDM, whole-column gate | 0.002269 | 98.7% | 0.649 | 28.2 |

Transform/mask preprocessing takes 3.0 / 3.7 ms. WDM's separately measured
kernel preparation takes 91.0 ms, including the known-truth expectation
audit; it is excluded from those transform times and must be charged once
for this mask. Model preparation is also recorded separately in the metrics.

Inside the raw gap, widths increase **1.58× / 2.25×** relative to the same
ungapped record. Truth lies inside **71.9% / 100%** of the gap-region
intervals. These are single-realization spatial inclusion rates, not
repeated-sampling coverage estimates. The PSD inside the gap is interpolated
through the prior and observations outside it. The fragmented case remains
a preprocessing audit only.

The positive estimation error is larger for DW in this record. Averaged over
the gap region, its posterior-median PSD is **13.1% above truth**, versus
**6.6% for WDM**. Relative to each method's ungapped fit, the mean log-PSD
shift is +0.0685 for DW versus +0.00241 for WDM. At u=0.5 and f=0.25 Hz,
truth is 2.443; DW estimates 2.781, with a 90% interval [2.456, 3.172],
and WDM estimates 2.625, with [2.195, 3.213]. DW's gap-region log-PSD MSE is
3.7 times WDM's. This feature was present before frequency reuse. The
finite-window expected-power discrepancy for retained DW coefficients is
below 0.016%, far too small to explain this overshoot by itself.
This is evidence of a worse gap interpolation on this realization; systematic
estimator bias still requires repeated gapped records.
[Signed-error calculations](gaps/column_gated/signed_error.json).

![Central-gap posterior estimates and complete WDM time-column gate](gaps/column_gated/gap_comparison.png)

[Regional metrics](gaps/column_gated/regional_metrics.json),
[PDF](gaps/column_gated/gap_comparison.pdf) and
[original per-coefficient results](gaps/regional_metrics.json).

## Next step and reproduction

Repeated paired gap seeds should score observed regions, edges and interiors
separately. Add coloured LS2 spectra before generalizing. Retaining partially
observed analysis windows would require mask-aware expected power and
covariance, rather than just sample-count normalization. Related treatments
include [Guillaumin et al.](https://arxiv.org/abs/1907.02447) and
[Burke et al.](https://arxiv.org/abs/2609.31550); those results do not by
themselves validate this time-varying tensor model.

From the repository root, after the main rerun:

```bash
export MPLCONFIGDIR=/tmp/psd-benchmark-mpl
export OPENBLAS_NUM_THREADS=1
export JAX_ENABLE_X64=true
.venv/bin/python -m experiments.dw_wdm_small.runtime_check
.venv/bin/python -m experiments.dw_wdm_small.gap_smoke audit
.venv/bin/python -m experiments.dw_wdm_small.gap_smoke fit --baseline experiments/dw_wdm_small/results/reuse_rerun
.venv/bin/python -m experiments.dw_wdm_small.gap_smoke summarize --baseline experiments/dw_wdm_small/results/reuse_rerun
```
