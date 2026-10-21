"""Direct transforms and independent paired-predictor/gradient checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from log_psplines import ANOVALogPSpline, PowerConfig, PowerData, SplineBasis
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood

from experiments.dw_wdm_small.lisa_hann_model import prepare_paired_model
from experiments.dw_wdm_small.lisa_hann_one_case import (
    OffsetFourier,
    SelectedHann,
)


def test_fast_hann_selected_transform_agrees_with_every_direct_window():
    m = 64
    rng = np.random.default_rng(20261015)
    x = rng.normal(size=3 * m)
    engine = SelectedHann(m)
    direct = np.array(
        [
            x[j : j + 2 * m + 1]
            @ (
                engine.h
                * np.exp(
                    -2j * np.pi * (j + 1) * np.arange(2 * m + 1) / (2 * m + 1)
                )
            )
            for j in range(m)
        ]
    )
    np.testing.assert_allclose(engine.block(x), direct, atol=2e-11)


@pytest.mark.parametrize(
    "indices",
    [
        np.array([0]),
        np.array([66]),
        np.arange(67),
        np.arange(2, 50, 3),
        np.arange(11, 41),
    ],
)
def test_pruned_hann_keeps_the_original_window_and_frequency(indices):
    # The non-power-of-two cycle exercises both partial leaves and cross-half
    # prefix sums; the last site needs all preceding entering/leaving samples.
    m = 67
    x = np.random.default_rng(27).normal(size=3 * m)
    full = SelectedHann(m)
    pruned = SelectedHann(m, indices=indices)
    actual = pruned.block(x)
    direct = np.array(
        [
            x[j : j + 2 * m + 1]
            @ (
                full.h
                * np.exp(
                    -2j * np.pi * (j + 1) * np.arange(2 * m + 1) / (2 * m + 1)
                )
            )
            for j in indices
        ]
    )
    assert actual.shape == indices.shape
    np.testing.assert_allclose(actual, full.block(x)[indices], atol=2e-11)
    np.testing.assert_allclose(actual, direct, atol=2e-11)


@pytest.mark.parametrize("indices", [[], [-1], [64], [2, 1], [1, 3, 6], [1.5]])
def test_pruned_transform_rejects_invalid_frequency_selections(indices):
    with pytest.raises(ValueError):
        OffsetFourier(64, indices=np.asarray(indices))


def test_pruning_preserves_weak_coefficients_next_to_a_strong_tone():
    # White noise alone misses roundoff leakage from a loud, narrow line.
    # Keep this scale-separated case as a guard on future FFT changes.
    m = 7680
    length = 2 * m + 1
    time = np.arange(3 * m)
    x = np.cos(2 * np.pi * 2500 * time / length)
    x += 1e-6 * np.random.default_rng(29).normal(size=len(time))
    indices = np.arange(5, 3072, 3)
    full, pruned = SelectedHann(m), SelectedHann(m, indices=indices)
    expected = full.block(x)[indices]
    np.testing.assert_allclose(
        pruned.block(x), expected, rtol=1e-8, atol=1e-10
    )


def test_pruned_powers_preserve_paired_likelihood_and_parameter_gradients():
    jax.config.update("jax_enable_x64", True)
    m, dt = 64, 2.0
    indices = np.arange(2, 60, 3)
    rng = np.random.default_rng(28)
    blocks = rng.normal(size=(3, 3 * m))
    full, pruned = SelectedHann(m), SelectedHann(m, indices=indices)
    before = np.array([full.block(x)[indices] for x in blocks])
    after = np.array([pruned.block(x) for x in blocks])
    frequency = (indices + 1) / ((2 * m + 1) * dt)
    time = (2 * m * np.arange(3)[:, None] + m + indices) / (8 * m)
    cycle_time = time.mean(axis=1)
    power_before, power_after = [
        (4 * dt * abs(c) ** 2 / (full.h @ full.h)).ravel()
        for c in (before, after)
    ]
    data = PowerData(power_before, 2, np.tile(frequency, 3), time.ravel())
    spline = ANOVALogPSpline(
        SplineBasis.from_grid(frequency, 2),
        SplineBasis.from_grid(np.linspace(0, 1, 30), 2),
    )
    _, pair, _, evaluate = prepare_paired_model(
        data, spline, PowerConfig(centered=True), cycle_time
    )
    g = jnp.asarray(rng.normal(0, 0.1, len(pair["lam_freq"])))
    eta = jnp.asarray(rng.normal(0, 0.1, pair["eta_scale"].shape))

    def objective(g, eta, power):
        return power_whittle_log_likelihood(
            power, data.counts, evaluate(g, eta)
        )

    value_grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
    old_value, old_grad = value_grad(g, eta, jnp.asarray(power_before))
    new_value, new_grad = value_grad(g, eta, jnp.asarray(power_after))
    np.testing.assert_allclose(new_value, old_value, rtol=1e-12, atol=1e-10)
    for new, old in zip(new_grad, old_grad, strict=True):
        np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-10)


def test_paired_reference_predictor_and_gradient_preserve_exact_coordinates():
    jax.config.update("jax_enable_x64", True)
    frequency = np.linspace(0.01, 0.1, 8)
    cycle = np.array([0.2, 0.5, 0.8])
    time = (cycle[:, None] + np.linspace(-0.01, 0.01, len(frequency))).ravel()
    f = np.tile(frequency, len(cycle))
    data = PowerData(np.linspace(0.1, 4, len(time)), 2, f, time)
    spline = ANOVALogPSpline(
        SplineBasis.from_grid(frequency, 2),
        SplineBasis.from_grid(np.linspace(0, 1, 30), 2),
    )
    _, pair, _, evaluate = prepare_paired_model(
        data, spline, PowerConfig(centered=True), cycle
    )
    bt = np.asarray(spline.time.design_at(time)) @ spline.time_transform
    bf = np.asarray(spline.frequency.design_at(f))
    bt, bf = jnp.asarray(bt @ pair["U_time"]), jnp.asarray(bf @ pair["U_freq"])
    rng = np.random.default_rng(4)
    g = jnp.asarray(rng.normal(size=bf.shape[1]))
    eta = jnp.asarray(rng.normal(size=(bt.shape[1], bf.shape[1])))
    expected = bf @ g + jnp.einsum("qi,ij,qj->q", bt, eta, bf)
    np.testing.assert_allclose(evaluate(g, eta), expected, atol=1e-13)

    def objective(value):
        return power_whittle_log_likelihood(
            data.power, data.counts, evaluate(g, value)
        )

    def reference(value):
        return power_whittle_log_likelihood(
            data.power,
            data.counts,
            bf @ g + jnp.einsum("qi,ij,qj->q", bt, value, bf),
        )

    np.testing.assert_allclose(
        jax.grad(objective)(eta), jax.grad(reference)(eta), atol=1e-12
    )


def test_noncentered_adapter_has_identical_physical_prior_and_likelihood():
    from numpyro.infer.util import log_density

    jax.config.update("jax_enable_x64", True)
    f = np.linspace(0.01, 0.1, 8)
    t = np.array([0.2, 0.5, 0.8])
    data = PowerData(
        np.linspace(0.1, 4, 24), 2, np.tile(f, 3), np.repeat(t, 8)
    )
    spline = ANOVALogPSpline(
        SplineBasis.from_grid(f, 2),
        SplineBasis.from_grid(np.linspace(0, 1, 30), 2),
    )
    config = PowerConfig(centered=True)
    centered, pair, init, _ = prepare_paired_model(data, spline, config, t)
    transformed, _, init_z, _ = prepare_paired_model(
        data, spline, config, t, noncentered_eta=True
    )
    log_centered = log_density(centered, (), {}, init)[0]
    log_z = log_density(transformed, (), {}, init_z)[0]
    jacobian = np.sum(np.log(init["sigma_eta"] * pair["eta_scale"]))
    np.testing.assert_allclose(log_z, log_centered + jacobian, atol=2e-10)


def test_paired_adapter_matches_library_rectangular_prior_and_likelihood():
    """Independent library model fixes the physical prior and normalization."""
    from log_psplines.inference.anova_power import prepare_anova_power_model
    from numpyro.infer.util import log_density

    frequency = np.linspace(0.01, 0.1, 8)
    time = np.array([0.2, 0.5, 0.8])
    power = np.linspace(0.1, 4.0, 24).reshape(3, 8)
    grid = PowerData(power, 2, frequency, time)
    paired = PowerData(
        power.ravel(), 2, np.tile(frequency, 3), np.repeat(time, 8)
    )
    spline = ANOVALogPSpline(
        SplineBasis.from_grid(frequency, 2),
        SplineBasis.from_grid(np.linspace(0, 1, 30), 2),
    )
    config = PowerConfig(centered=True)
    reference, _, initial = prepare_anova_power_model(grid, spline, config)
    adapter, _, _, _ = prepare_paired_model(paired, spline, config, time)
    # Use the same nonzero physical states in both models; agreement includes
    # the mean null modes, interaction scales, priors and Whittle factor.
    for factor in (0.7, 1.0, 1.3):
        state = {name: value * factor for name, value in initial.items()}
        expected, _ = log_density(reference, (), {}, state)
        actual, _ = log_density(adapter, (), {}, state)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-10)
