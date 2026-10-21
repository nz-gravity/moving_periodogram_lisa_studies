"""Benchmark selected-bin preprocessing against a supplied, unchanged Git ref.

Run from the package root with .venv/bin/python -m
experiments.dw_wdm_small.benchmark_selected_bins --baseline-ref REVISION.
The optional --lisa flag replays every retained cycle in the saved LISA case.
It reads existing data/products and writes only this benchmark's report.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
from functools import partial
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from log_psplines import PowerData
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood
from log_psplines.preprocessing.moving_periodogram import (
    tang_moving_periodogram,
    tang_moving_periodogram_coordinates,
)
from scipy.interpolate import BSpline
from scipy.signal import ZoomFFT

from experiments.dw_wdm_small import hann_paper as study
from experiments.dw_wdm_small import lisa_hann_one_case as lisa
from experiments.dw_wdm_small.paths import LIBRARY_REPO, source_path

REPO = Path(__file__).resolve().parents[2]
CORE = "src/log_psplines/preprocessing/moving_periodogram.py"
HANN = "experiments/dw_wdm_small/hann_paper.py"
LISA = "experiments/dw_wdm_small/lisa_hann_one_case.py"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def baseline_definitions(source: str, names: set[str], **namespace) -> dict:
    """Load only the old definitions being measured, without running a study."""
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body if getattr(node, "name", None) in names
    ]
    assert {node.name for node in nodes} == names
    # Postponed annotations let the original scientific functions run without
    # importing unrelated samplers or writing their usual study outputs.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<baseline>", "exec"),
        namespace,
    )
    return namespace


def measure_pair(before, after, *, rounds: int, batch: int = 1) -> dict:
    """Alternate order after warmup, keeping setup and I/O outside the timer."""
    for function in (before, after):
        function()
        function()
    samples = [[], []]
    for repeat in range(rounds):
        for which in (0, 1) if repeat % 2 == 0 else (1, 0):
            function = (before, after)[which]
            started = perf_counter()
            for _ in range(batch):
                function()
            samples[which].append(1000 * (perf_counter() - started) / batch)
    old, new = [float(np.median(sample)) for sample in samples]
    return dict(
        before_ms=old,
        after_ms=new,
        speedup=old / new,
        before_samples_ms=samples[0],
        after_samples_ms=samples[1],
        rounds=rounds,
        calls_per_round=batch,
    )


def coefficient_errors(before: np.ndarray, after: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(abs(before) ** 2)))
    # Absolute tolerances must follow the signal scale: LISA coefficients
    # are tiny, so a fixed tolerance such as 1e-12 would be meaningless.
    np.testing.assert_allclose(after, before, rtol=1e-8, atol=1e-12 * rms)
    return dict(
        relative_l2=float(
            np.linalg.norm(after - before) / np.linalg.norm(before)
        ),
        maximum_relative=float(
            np.max(abs(after - before) / np.maximum(abs(before), rms * 1e-12))
        ),
        max_absolute=float(np.max(abs(after - before))),
        rms_before=rms,
    )


def likelihood_parity(old_power, new_power, bt, bf) -> dict:
    """Compare the paired likelihood and every field-parameter derivative."""
    jax.config.update("jax_enable_x64", True)
    bt, bf = jnp.asarray(bt), jnp.asarray(bf)
    n_time, n_freq = bt.shape[1], bf.shape[1]
    index = jnp.tile(jnp.arange(len(bf)), old_power.shape[0])

    def objective(parameters, power):
        g = parameters[:n_freq]
        eta = parameters[n_freq:].reshape(n_time, n_freq)
        log_s = (bf @ g)[index] + jnp.sum(bt * (bf @ eta.T)[index], axis=1)
        return power_whittle_log_likelihood(power.ravel(), 2.0, log_s)

    value_grad = jax.jit(jax.value_and_grad(objective))
    rng = np.random.default_rng(20261022)
    old_power, new_power = jnp.asarray(old_power), jnp.asarray(new_power)
    checks = []
    for scale in (0.0, 0.03, 0.1, 0.3):
        parameters = jnp.asarray(rng.normal(0, scale, n_freq * (n_time + 1)))
        old_value, old_grad = value_grad(parameters, old_power)
        new_value, new_grad = value_grad(parameters, new_power)
        np.testing.assert_allclose(new_value, old_value, rtol=1e-9, atol=1e-7)
        np.testing.assert_allclose(new_grad, old_grad, rtol=1e-9, atol=1e-7)
        checks.append(
            dict(
                parameter_scale=scale,
                before_log_likelihood=float(old_value),
                log_likelihood_abs_change=float(abs(new_value - old_value)),
                gradient_max_abs_change=float(
                    np.max(abs(new_grad - old_grad))
                ),
            )
        )
    return dict(parameters=n_freq * (n_time + 1), states=checks)


def replay_lisa(old_class, rounds: int) -> dict:
    """Read the original context, preserving its rotation and retained sites."""
    import h5py

    sys.path.insert(0, str(lisa.LEGACY))
    from tv_pspline_psd.lisa_aet import xyz_to_aet_series

    prepared = lisa.HERE / "lisa_hann_one_case/noncentered_eta/prepared.npz"
    saved = dict(np.load(prepared))
    m, dt = lisa.M, lisa.DT
    full_frequency = np.arange(1, m + 1) / ((2 * m + 1) * dt)
    indices = np.flatnonzero(
        (np.arange(1, m + 1) % 3 == 0)
        & (full_frequency >= lisa.FMIN)
        & (full_frequency <= lisa.FMAX)
    )
    np.testing.assert_array_equal(full_frequency[indices], saved["frequency"])
    starts = saved["sample_index"][:, 0] - m - indices[0]
    np.testing.assert_array_equal(
        starts[:, None] + m + indices, saved["sample_index"]
    )
    np.testing.assert_array_equal(
        saved["sample_index"] / lisa.N, saved["time"]
    )
    start = ((int(lisa.START_DAY * 86400 / dt) - 4 * m) // (2 * m)) * (2 * m)
    stop = int(lisa.STOP_DAY * 86400 / dt) + 4 * m
    with h5py.File(lisa.LISA / "data/lisa.h5", "r") as handle:
        x = xyz_to_aet_series(handle["tdi/total"][:, start:stop])[0]
    metadata = json.loads(
        (lisa.HERE / "lisa_hann_one_case/preparation.json").read_text()
    )
    assert (
        digest(np.ascontiguousarray(x).tobytes())
        == metadata["raw_context_A_sha256"]
    )
    blocks = [x[int(s) - start : int(s) - start + 3 * m] for s in starts]
    old, new = old_class(m), lisa.SelectedHann(m, indices=indices)

    def before():
        return np.array([old.block(block)[indices] for block in blocks])

    def after():
        return np.array([new.block(block) for block in blocks])

    old_coeff, new_coeff = before(), after()
    errors = coefficient_errors(old_coeff, new_coeff)
    # The production path uses sum(h**2), so retain that exact normalization.
    old_power, new_power = [
        4 * dt * abs(c) ** 2 / np.sum(old.h**2) for c in (old_coeff, new_coeff)
    ]
    np.testing.assert_allclose(old_power, saved["power"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(new_power, saved["power"], rtol=3e-8, atol=0)
    basis = dict(np.load(prepared.with_name("model_basis.npz")))
    # Reuse the saved centring and eigenvectors. Recentring on retained times
    # would change the model and invalidate the before/after comparison.
    bt = BSpline.design_matrix(
        saved["time"].ravel(), basis["time_knots"], 3
    ).toarray()
    bt = bt @ basis["time_transform"] @ basis["U_time"]
    bf = (
        BSpline.design_matrix(
            saved["frequency"], basis["frequency_knots"], 3
        ).toarray()
        @ basis["U_freq"]
    )
    likelihood = likelihood_parity(
        old_power / saved["reference"], new_power / saved["reference"], bt, bf
    )
    return dict(
        cycles=len(blocks),
        retained_coefficients=int(old_coeff.size),
        computed_before_per_cycle=m,
        computed_after_per_cycle=len(indices),
        prepared_sha256=digest(prepared.read_bytes()),
        raw_context_sha256=metadata["raw_context_A_sha256"],
        coefficient_errors=errors,
        saved_power_max_relative_change=float(
            np.max(abs(new_power / saved["power"] - 1))
        ),
        likelihood=likelihood,
        transform=measure_pair(before, after, rounds=rounds),
        plan_setup=measure_pair(
            lambda: old_class(m),
            lambda: lisa.SelectedHann(m, indices=indices),
            rounds=rounds,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument(
        "--baseline-repo",
        type=Path,
        default=LIBRARY_REPO,
        help="Original LogPSplinePSD checkout containing the baseline Git revision.",
    )
    parser.add_argument("--lisa", action="store_true")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=lisa.HERE / "selected_bin_benchmark"
    )
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("Use at least three timing rounds.")
    revision = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref],
        cwd=args.baseline_repo,
        text=True,
    ).strip()
    sources = {
        path: subprocess.check_output(
            ["git", "show", f"{revision}:{path}"],
            cwd=args.baseline_repo,
            text=True,
        )
        for path in (CORE, HANN, LISA)
    }
    old_core = baseline_definitions(
        sources[CORE], {"tang_moving_periodogram"}, np=np
    )["tang_moving_periodogram"]
    old_hann = baseline_definitions(
        sources[HANN],
        {"hann_observations"},
        np=np,
        b=SimpleNamespace(tang_moving_periodogram=old_core),
        taper=study.taper,
        PowerData=PowerData,
    )["hann_observations"]
    old_lisa = baseline_definitions(
        sources[LISA],
        {"OffsetFourier", "SelectedHann"},
        np=np,
        ZoomFFT=ZoomFFT,
    )["SelectedHann"]
    report = dict(
        baseline_revision=revision,
        baseline_repository=str(args.baseline_repo.resolve()),
        baseline_source_sha256={
            path: digest(source.encode()) for path, source in sources.items()
        },
        current_source_sha256={
            path: digest(source_path(path).read_bytes())
            for path in (
                CORE,
                HANN,
                LISA,
                str(Path(__file__).relative_to(REPO)),
            )
        },
        environment=dict(
            platform=platform.platform(),
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            jax=jax.__version__,
        ),
        method="Median of warmed, alternating-order rounds; NumPy calls are synchronous. Setup is separate. No data I/O, reference projection, or sampling is timed.",
        coordinates=[],
        hann_preparation=[],
    )
    for n in (8192, 32768):
        x = np.random.default_rng(n).normal(size=n)
        old_raw, new_raw = (
            old_core(x, m=32, thin=2),
            tang_moving_periodogram(x, m=32, thin=2),
        )
        for key in old_raw:
            np.testing.assert_array_equal(old_raw[key], new_raw[key])
        schedule = tang_moving_periodogram_coordinates(n, m=32, thin=2)
        for key in schedule:
            np.testing.assert_array_equal(schedule[key], old_raw[key])
        report["coordinates"].append(
            dict(
                n=n,
                exact_core_coefficient_and_coordinate_parity=True,
                timing=measure_pair(
                    lambda n=n: old_core(np.zeros(n), m=32, thin=2),
                    partial(
                        tang_moving_periodogram_coordinates, n, m=32, thin=2
                    ),
                    rounds=args.rounds,
                    batch=20,
                ),
            )
        )
        for complete, rungs in (
            (True, None),
            (True, tuple(range(1, 33))),
            (False, None),
        ):
            observed = np.ones(n, bool)
            if not complete:
                observed[n // 3 : n // 3 + n // 10] = False
            before = partial(old_hann, x, observed, 0.1, rungs=rungs)
            after = partial(
                study.hann_observations, x, observed, 0.1, rungs=rungs
            )
            old_data, new_data = before(), after()
            for key in ("power", "counts", "time", "frequency"):
                np.testing.assert_array_equal(
                    getattr(old_data, key), getattr(new_data, key)
                )
            report["hann_preparation"].append(
                dict(
                    n=n,
                    complete=complete,
                    all_rungs=rungs is not None,
                    exact_observation_parity=True,
                    timing=measure_pair(
                        before, after, rounds=args.rounds, batch=10
                    ),
                )
            )
    if args.lisa:
        report["lisa"] = replay_lisa(old_lisa, args.rounds)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    rows = [
        "# Selected-bin preprocessing benchmark",
        "",
        f"Baseline: `{revision}`.",
        "",
        report["method"],
        "",
        "| Workload | Before (ms) | After (ms) | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for item in report["coordinates"]:
        t = item["timing"]
        rows.append(
            f"| Coordinates, N={item['n']} | {t['before_ms']:.3f} | {t['after_ms']:.3f} | {t['speedup']:.2f}x |"
        )
    for item in report["hann_preparation"]:
        t = item["timing"]
        label = f"Hann, N={item['n']}, {'all bins' if item['all_rungs'] else 'sparse bins'}, {'complete' if item['complete'] else 'gapped'}"
        rows.append(
            f"| {label} | {t['before_ms']:.3f} | {t['after_ms']:.3f} | {t['speedup']:.2f}x |"
        )
    rows.extend(
        [
            "",
            "- Core coefficients, schedule coordinates, and Hann observation arrays match the baseline bit for bit.",
        ]
    )
    if "lisa" in report:
        result = report["lisa"]
        for name in ("transform", "plan_setup"):
            t = result[name]
            rows.insert(
                -2,
                f"| LISA {name.replace('_', ' ')} | {t['before_ms']:.3f} | {t['after_ms']:.3f} | {t['speedup']:.2f}x |",
            )
        checks = result["likelihood"]["states"]
        rows.extend(
            [
                f"- Replayed all {result['cycles']} accepted LISA cycles ({result['retained_coefficients']:,} coefficients) from the hash-checked original data context.",
                f"- LISA computes {result['computed_after_per_cycle']:,} coefficients per cycle instead of {result['computed_before_per_cycle']:,}; original time/frequency pairs and normalization are preserved.",
                "- Only requested moving-window updates and outputs are computed. The shared first-window DFT retains its original full grid: pruning that FFT gave no measurable speed benefit and increased roundoff in weak coefficients.",
                f"- Coefficient relative L2 difference: {result['coefficient_errors']['relative_l2']:.3g}; maximum relative difference: {result['coefficient_errors']['maximum_relative']:.3g}.",
                f"- Maximum absolute log-likelihood change across four parameter states: {max(c['log_likelihood_abs_change'] for c in checks):.3g}.",
                f"- Maximum absolute parameter-gradient change: {max(c['gradient_max_abs_change'] for c in checks):.3g} (all {result['likelihood']['parameters']} field parameters checked).",
                "- LISA equality is numerical, not bitwise: pruning changes FFT roundoff. Coefficients were checked with relative tolerance 1e-8 and a scale-dependent absolute floor of 1e-12 times their RMS; likelihoods and gradients with rtol=1e-9, atol=1e-7.",
            ]
        )
    rows.extend(
        [
            "",
            "These are local preprocessing timings, not posterior-sampling speedups. Full measurements, source hashes, versions, and individual timing rounds are in `benchmark.json`.",
            "",
            "Reproduce from the package root:",
            "",
            "```bash",
            f".venv/bin/python -m experiments.dw_wdm_small.benchmark_selected_bins --baseline-ref {revision} --baseline-repo {args.baseline_repo.resolve()}"
            + (" --lisa" if args.lisa else ""),
            "```",
            "",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(rows))
    print("\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
