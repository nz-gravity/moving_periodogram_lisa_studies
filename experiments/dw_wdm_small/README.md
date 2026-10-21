# Small Dynamic Whittle versus WDM benchmark

The [gap/taper follow-up](taper_study/REPORT.md) contains ten paired central-gap
records, a Hann-versus-matched-rectangular control, guarded WDM comparisons,
and an exact expected-power audit on coloured LS2 noise.

This study uses existing scalar preprocessing, tensor P-splines, priors,
NUTS and result containers. Production code is unchanged.

## Audit and implementation plan

Audited `nz-gravity/LogPSplinePSD`, checkout `ffce954`, initially clean.

| Existing component | Location and reuse |
|---|---|
| Gaussian simulation and analytic truth | `src/log_psplines/example_datasets/ls2_data.py`: univariate, time-varying MA(1); its `amplitude=0` special case supplies Gaussian white noise here. `varma_data.py` and `tvvar_data.py` supply additional stationary and time-varying multichannel examples, outside this experiment. |
| Dynamic Whittle selection | `preprocessing/moving_periodogram.py`: Tang zig-zag frequencies, complete interior cycles, `2*m+1` windows, whole-cycle thinning; scattered and rectangular interfaces. |
| Compression | `bin_tang_ordinates` and `preprocessing/power_partition.py`: sum raw powers and component counts, approximate a varying spectrum at block centres; mask and gap controls. No pooling is used in this primary comparison. |
| WDM integration | `preprocessing/wdm.py`: installed optional `wdm-transform`; real interior coefficients squared, count one; DC/Nyquist and one time row per end removed. No legacy `pywavelet` dependency is needed. |
| Common likelihood | `likelihoods/whittle.py`: `-0.5*sum(counts*log(S) + power/S)`; count two for complex Tang ordinates and one for real WDM coefficients. |
| Shared 2D model and prior | `models/spectrum.py`, `basis/splines.py`, `basis/penalty.py`, `inference/power.py`: tensor log P-splines with trace-normalized integrated derivative penalties, additive marginal precision and HalfNormal roughness scales. ANOVA and parametric models also exist, but are not needed. |
| Common inference | Public `fit`, `PowerConfig`, `prepare_power_model`, existing NUTS and scalar VI; all posterior weights and metadata can be saved as `PSDResult`. |
| Tests and previous results | `tests/time_varying/test_time_varying.py` has transform, pooling, prior and WDM/LS2 recovery checks. Existing ignored archives `runs/dynamic-whittle-stages-0-3/report.md` and `runs/window-expected-power-20261002/report.md` contain fixed-window and observation-mean investigations. The first documents VI/NUTS uncertainty differences and residual dependence; the second shows why expected window power must be distinguished from latent PSD. Neither is a matched ten-seed DW/WDM study. |
| Existing direct comparison | `docs/examples/timevarying-example.ipynb` contains LS2 WDM/moving-periodogram fits, saved curves, and a one-seed size sweep from 1024 to 16384 samples. Its summaries estimate a units offset using truth, score each native grid and include posterior dispersion in its MSE. At N=8192, its saved 300-draw, one-chain results report WDM/moving coverage 82%/66%, log MSE 0.046/0.051 and fit wall times about 5.2 seconds each. These are archived notebook outputs, not freshly validated timing or calibration evidence; the truth-fitted scaling, unmatched support and thin=1 sweep prevent treating them as this benchmark's baseline. |
| Prior compression smoke study | The separate local worktree `dw-adaptive-smoke/LogPSplinePSD/experiments/dw_multiscale/SMOKE_REPORT.md` reports an engineering improvement from simple local pooling on two records. It did not establish general calibration or a WDM likelihood advantage. |

**Missing pieces are small:** neither scalar preprocessing adapter returns common
physical PSD units automatically, and no common-unit paired-seed benchmark/report
was present in this checkout. This experiment supplies only those conversions,
paired orchestration and metrics. Repository whitening diagnostics are still
placeholders; signed coefficient moments provide a descriptive diagnostic here.

1. Check physical normalization with 128 white-noise records; calculate exact
   Gaussian expected powers and selected-neighbour correlations for the
   generating process.
2. Fit one debugging record using a shared cubic tensor model/prior
   and two-chain NUTS. Check cost and sampler quality before ten new seeds.
3. Evaluate both coefficient posteriors on an identical interior grid, save
   complete draws, phase-separated timing, errors, coverage and widths.
4. Produce a common-scale PSD figure, seed table and short findings report.

## Frozen study choices

- `N=8192`, `dt=1`, one channel. Simulation:
  `x_j=sqrt(v(j/N))*epsilon_j`, with independent standard Gaussian innovations
  and `log(v(u))=0.7*sin(2*pi*u)+0.2*cos(4*pi*u)`.
- True one-sided PSD: `S(u,f)=2*dt*v(u)` for interior positive frequencies.
  This is smooth broadband amplitude evolution, with no narrow spectral
  features. It is deliberately an easy normalization/calibration control.
- Tang: `m=32`, window 65 samples, `thin=2`, exact selected coordinates;
  `power *= 4*pi*dt`, counts remain two. Subtract `1/N` from returned
  one-based times to align both methods with the simulator's zero-based `u`.
- WDM: `nt=256`, `nf=32`, installed default Meyer kernel;
  `power *= 2*dt/N`, counts remain one. Drop DC/Nyquist and one time row at
  each end. Frequency spacing 1/64 Hz versus Tang 1/65 Hz. WDM has 32-sample
  time spacing versus Tang's 64-sample sweep spacing; squared-kernel effective
  window widths are approximately 38–51 versus 65 samples. Resolution is
  close but not identical; no exact equivalence is claimed.
- Shared cubic spline knots on full domains `u=[0,1]`, `f=[0,0.5]`, eight
  interior time knots and two interior frequency knots; 12-by-6 coefficients.
  Evaluation rows change for WDM, while knots and penalties stay fixed.
  HalfNormal roughness scale 10, null precision `1e-4`, ridge `1e-6`, as in
  the existing tensor defaults. Non-centered parameterization, dense mass NUTS,
  target acceptance 0.95, depth cap 10, two sequential chains; float64 enabled.
- Debug seed 6100; proposed independent comparison seeds 6101–6110. No truth
  initializes inference. Inference seed is data seed plus 10000, shared
  across methods. Alternate method order across seed parity.
- Initial cost pilot: 8-by-6 basis, centered, 400 warmup and 600 draws per
  chain, target 0.9 and depth 9; failed sampler screens and retained in
  `results/pilot_initial`. The revised pilot and comparison use the above
  12-by-6 basis and 600 warmup/1000 retained draws per chain. Revised pilot
  inference took about 79 seconds per pair and passed both screens; the
  ten-seed launch estimate was about 13 minutes. Data-independent dense-grid
  log-truth projection RMSE fell from 0.02204 to 0.000847 when increasing
  the time knots. Only the debugging seed informed these choices.
- Interior scoring grid: 61 times in `[0.05,0.95]`, 57 frequencies in
  `[0.025,0.475]`. Natural-log squared error and coverage use normalized
  trapezoidal area weights. Point estimate is the posterior median PSD.
  Width is median pointwise `q95(S)-q05(S)`; log width is also retained.
- Sampler screens use latent coefficients/roughness and a sparse functional
  log-PSD grid: zero divergences/depth hits, unrounded rank Rhat at most 1.01,
  bulk/tail ESS at least 200. These are small-study screens, not guarantees.
- Both are approximate diagonal/composite likelihoods. Whole-cycle thinning
  is used without likelihood reweighting; it does not imply independence.
- Main execution completed all ten paired seeds. Two flagged WDM fits were
  followed by paired longer checks on seeds 6103 and 6110: 1000 warmup,
  1500 retained draws per chain and acceptance 0.99. All four checks pass,
  while the primary 600/1000 runs remain the table's timing baseline.

## Reproduce

From the repository root, using its existing virtual environment:

```bash
export MPLCONFIGDIR=/tmp/psd-benchmark-mpl
export OPENBLAS_NUM_THREADS=1
.venv/bin/python experiments/dw_wdm_small/benchmark.py audit
.venv/bin/python experiments/dw_wdm_small/benchmark.py fit --seeds 6100
# Run the following only after assessing pilot cost and diagnostics.
.venv/bin/python experiments/dw_wdm_small/benchmark.py fit --seeds 6101 6102 6103 6104 6105 6106 6107 6108 6109 6110
.venv/bin/python experiments/dw_wdm_small/benchmark.py figure --seeds 6101
.venv/bin/python experiments/dw_wdm_small/benchmark.py summarize
.venv/bin/python experiments/dw_wdm_small/benchmark.py fit --seeds 6103 6110 --warmup 1000 --samples 1500 --target-accept 0.99 --output experiments/dw_wdm_small/results/repairs
```

`results/` contains full coefficient NetCDF posteriors, per-seed metrics,
common-grid posterior quantiles, exact expected-power checks, environment
versions and the PNG/PDF figure. `metrics.csv` and `summary.json` retain
individual failures instead of discarding them. Generated results are
explicitly retained by this experiment's `.gitignore` exceptions.

Method reference: [Tang et al., published article](https://doi.org/10.1080/01621459.2025.2594191)
and [author preprint](https://arxiv.org/abs/2303.11561). Published search text
and current source confirm the thinned product; this checkout retains complete
interior cycles rather than introducing padded boundary ordinates.

## Frequency-reuse rerun and column gating

The current scattered evaluator reuses exact repeated frequencies while
preserving each original paired site, coefficient prior and initialization.
`results/` retains the original comparison; the new ten-seed results are in
`results/reuse_rerun/`. Full original/optimized evaluator snapshots and source
hashes are retained there. `gaps/column_gated/` holds the new whole-column
WDM pilot; `gaps/` retains the old per-coefficient pilot.

```bash
export MPLCONFIGDIR=/tmp/psd-benchmark-mpl
export OPENBLAS_NUM_THREADS=1
export JAX_ENABLE_X64=true
.venv/bin/python -m experiments.dw_wdm_small.benchmark audit --output experiments/dw_wdm_small/results/reuse_rerun
.venv/bin/python -m experiments.dw_wdm_small.benchmark fit --seeds 6101 6102 6103 6104 6105 6106 6107 6108 6109 6110 --output experiments/dw_wdm_small/results/reuse_rerun
.venv/bin/python -m experiments.dw_wdm_small.benchmark figure --seeds 6101 --output experiments/dw_wdm_small/results/reuse_rerun
.venv/bin/python -m experiments.dw_wdm_small.rerun_summary
.venv/bin/python -m experiments.dw_wdm_small.benchmark fit --seeds 6103 6110 --warmup 1000 --samples 1500 --target-accept 0.99 --output experiments/dw_wdm_small/results/reuse_rerun/repairs
```

[Current results](REPORT.md) and [whole-column gap comparison](GAPS.md).
