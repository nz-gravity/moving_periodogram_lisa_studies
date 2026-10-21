# Moving-periodogram LISA studies

Moving-periodogram, Dynamic Whittle, Hann-window and gap studies, with matched
WDM comparisons. The reusable transforms, spline models, priors and samplers
remain in [LogPSplinePSD](https://github.com/nz-gravity/LogPSplinePSD).

## Layout

- `experiments/dw_wdm_small/`: current runners, validation tests and reports.
  The historical module name is retained so existing commands still work.
- `archive/logpsplinepsd/runs/`: earlier fixed-window, expected-power and
  multivariate moving-periodogram runs, preserved locally.
- `migration/library-study-move.json`: original paths and SHA-256 checksums.

The migration preserved 2798 files/links (5.03 GB),
including ignored results and the latest selected-bin benchmark. One absolute
internal result symlink was converted to a relative link. Historical JSON,
posteriors and source snapshots retain their original contents. Readers resolve
old product paths in memory; source ledgers locate the installed library.

## Environment

Keep this checkout beside `LogPSplinePSD` with its updated `main`
(`1059dec` or later, including the Tang coordinate-only helper), then create
its own environment:

```bash
uv sync --extra dev
export JAX_ENABLE_X64=true MPLBACKEND=Agg MPLCONFIGDIR=/tmp/moving-periodogram-mpl
.venv/bin/python -m pytest -q
```

While using the existing local environment, the equivalent test command is:

```bash
../LogPSplinePSD/.venv/bin/python -m pytest -q
```

## Entry points

Run these from this repository's root:

```bash
.venv/bin/python -m experiments.dw_wdm_small.benchmark --help
.venv/bin/python -m experiments.dw_wdm_small.hann_paper --help
.venv/bin/python -m experiments.dw_wdm_small.hann_paper_all_bins --help
.venv/bin/python -m experiments.dw_wdm_small.lisa_hann_one_case --help
.venv/bin/python -m experiments.dw_wdm_small.benchmark_selected_bins --baseline-ref aed4b19c42e26f84b83ef2c1201feeae30ea31c6 --baseline-repo ../LogPSplinePSD --lisa
```

The LISA replay reads existing raw data from the sibling `wdm_psd` checkout and
its saved prepared products; raw LISA data are not copied into this repository.
Manuscript figures and scripts remain in `moving_periodogram_pspline`.

Generated result directories and the archive are ignored by Git. Source,
study tests and top-level study documentation are the reviewable migration;
the compact [selected-bin benchmark](experiments/dw_wdm_small/selected_bin_benchmark/REPORT.md)
and its timing/parity JSON are included. Six small frozen regression fixtures
(about 120 KB total) include three mask arrays, the original source comparator
and two diagnostic summaries, so tests work in a fresh clone. Ignore exceptions
name each fixture explicitly. Individual versioned files stay below 1 MiB;
raw data, posterior draws and bulk arrays remain local.

## Migration verification

- 87 study tests, 189 library tests and 5 manuscript checks passed.
- All five primary command-line entry points passed their help checks.
- The selected-bin LISA benchmark replay passed from this checkout against
  the original library Git revision, checking all 70,587 retained coefficients.
- `migration/verification.json` records the validation and the small source
  adaptations made after the byte-preserving move.

Verify the versioned migration files in a fresh clone:

```bash
.venv/bin/python migration/verify_move.py --source-only
```

For the complete local migration, omit `--source-only`; this checks all 2798
files, including the ignored results that are not available in a fresh clone.

Historical stationary, MorphZ, p=1 and other compact reports are now preserved
in [psd_studies_archive](https://github.com/nz-gravity/psd_studies_archive).
Their heavy generated files were removed from the library checkout.

## Publication checks

The source was checked against library `main`: 88 study tests and five CLI
help checks passed, along with Ruff lint and formatting checks. The local LISA
replay rechecked all 70,587 coefficients and the likelihood/gradient tolerances.
The paired-site adapter is also checked against the library's rectangular
ANOVA prior and likelihood; it does not require private helpers from an
unmerged feature branch. These checks are recorded separately from the original
migration validation in `migration/verification.json`.

The selected-bin report and JSON preserve the original benchmark snapshot and
its source hashes. Fresh-clone tests run without the ignored data; replaying
full historical LISA runs still requires the local data and prepared products.
