"""Small paired-site adapter for the existing scalar ANOVA prior/likelihood.

Only the predictor geometry differs from rectangular GridTV. Initialization
uses a per-cycle grid; the likelihood uses every ordinate's exact coordinate.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from log_psplines import ANOVALogPSpline, PowerConfig, PowerData
from log_psplines.basis.penalty import whiten_penalty_pair
from log_psplines.inference.anova_power import initialize_anova
from log_psplines.inference.power import sample_eigen_coefficients
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood


def anova_design(
    spline: ANOVALogPSpline, time: np.ndarray, frequency: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the fixed centered basis at new observation or plot sites."""
    # Reuse the full-record centering transform. Re-centering on selected
    # observations would change the meaning of the mean and deviation fields.
    bt = np.asarray(spline.time.design_at(time)) @ spline.time_transform
    bf = np.asarray(spline.frequency.design_at(frequency))
    return bt, bf


def prepare_paired_model(
    data: PowerData,
    spline: ANOVALogPSpline,
    config: PowerConfig,
    cycle_time: np.ndarray,
    *,
    noncentered_eta: bool = False,
):
    """Return the unchanged ANOVA prior, paired predictor, and data-only start."""
    assert not data.is_grid and config.centered
    bt, _ = anova_design(spline, data.time, data.frequency)
    frequency, first, inverse = np.unique(
        data.frequency, return_index=True, return_inverse=True
    )
    bf = np.asarray(spline.frequency.design_at(frequency))
    # Use the same fixed penalties and null-mode scales as the library's
    # rectangular ANOVA model. Keep the paired-site adapter self-contained;
    # helper functions from an unmerged library feature branch are not needed.
    pair = whiten_penalty_pair(spline.time_penalty, spline.frequency.penalty)
    lam_t, lam_f = pair["lam_time"], pair["lam_freq"]
    null_f = jnp.asarray(lam_f <= 1e-10 * max(lam_f.max(), 1.0))
    pair["eta_scale"] = np.where(
        pair["joint_null"],
        1.0,
        (lam_t[:, None] + lam_f[None, :] + config.ridge_eps) ** -0.5,
    )
    bt_eigen = jnp.asarray(bt @ pair["U_time"])
    bf_eigen = jnp.asarray(bf @ pair["U_freq"])
    frequency_index = jnp.asarray(inverse)
    power, counts = jnp.asarray(data.power), jnp.asarray(data.counts)

    def evaluate(g, eta):
        mean = bf_eigen @ g
        projection = bf_eigen @ eta.T
        return mean[frequency_index] + jnp.sum(
            bt_eigen * projection[frequency_index], axis=1
        )

    def model():
        sigma_g = numpyro.sample(
            "sigma_g", dist.HalfNormal(config.roughness_scale)
        )
        mean_scale = jnp.where(
            null_f,
            config.null_precision**-0.5,
            sigma_g
            / jnp.sqrt(jnp.asarray(lam_f) + config.ridge_eps * sigma_g**2),
        )
        g = sample_eigen_coefficients("g", mean_scale, (len(lam_f),), config)
        sigma_eta = numpyro.sample(
            "sigma_eta", dist.HalfNormal(spline.sigma_eta_prior)
        )
        scale = sigma_eta * jnp.asarray(pair["eta_scale"])
        # Changing sampling coordinates must preserve the physical deviation
        # field and its likelihood. The regression test checks the Jacobian.
        if noncentered_eta:
            physical = sample_eigen_coefficients(
                "z_eta",
                scale.reshape(-1),
                (scale.size,),
                replace(config, centered=False),
            )
            eta = numpyro.deterministic("eta", physical).reshape(scale.shape)
        else:
            eta = sample_eigen_coefficients(
                "eta", scale.reshape(-1), (scale.size,), config
            ).reshape(scale.shape)
        likelihood = power_whittle_log_likelihood(
            power, counts, evaluate(g, eta)
        )
        numpyro.deterministic("log_likelihood", likelihood)
        numpyro.factor("whittle", likelihood)

    shape = (len(cycle_time), len(frequency))
    assert len(data.power) == np.prod(shape)
    grid = PowerData(
        np.asarray(data.power).reshape(shape),
        np.asarray(data.counts).reshape(shape),
        frequency,
        cycle_time,
    )
    bt_init, bf_init = anova_design(spline, cycle_time, frequency)
    init = initialize_anova(
        grid,
        bt_init,
        bf_init,
        spline.time_penalty,
        np.asarray(spline.frequency.penalty),
        pair,
        config,
    )
    if noncentered_eta:
        init["z_eta"] = init.pop("eta") / (
            init["sigma_eta"] * pair["eta_scale"].reshape(-1)
        )
    return model, pair, init, evaluate
