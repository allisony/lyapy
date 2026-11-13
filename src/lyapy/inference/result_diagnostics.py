"""Noise-aware fit diagnostics and plots for portable inference results."""

from dataclasses import dataclass

import numpy as np

from lyapy.inference.reconstruction import (
    DEFAULT_PERCENTILE_PROBABILITIES,
    posterior_percentile_bands,
)


def _spectrum_name(result, spectrum):
    if spectrum in result.observations:
        return spectrum
    available = ", ".join(repr(name) for name in result.observations)
    raise KeyError(
        f"unknown spectrum {spectrum!r}; available spectra: {available}"
    )


def _gaussian_settings():
    return {
        "distribution": "Gaussian",
        "ln_error_scale_index": None,
        "log10_jitter_index": None,
        "fixed_error_scale": 1.0,
        "fixed_jitter": 0.0,
    }


def _settings_from_callable(callable_name):
    if str(callable_name).endswith(".gaussian_log_likelihood"):
        return _gaussian_settings()
    raise ValueError(
        "posterior-predictive diagnostics require GaussianNoiseModel, "
        "StudentTNoiseModel, or gaussian_log_likelihood metadata"
    )


def _likelihood_settings(result, spectrum):
    hypothesis = result.problem_specification["hypothesis"]
    specification = hypothesis.get("likelihood_specification")
    if specification is None:
        return _settings_from_callable(hypothesis["likelihood_function"])
    if specification.get("distribution") != "joint":
        return specification
    try:
        spectrum_specification = specification["spectra"][spectrum]
    except KeyError as error:
        raise ValueError(
            f"likelihood metadata does not contain spectrum {spectrum!r}"
        ) from error
    settings = spectrum_specification.get("settings")
    if settings is not None:
        return settings
    return _settings_from_callable(spectrum_specification["callable"])


def _full_parameter_samples(result):
    hypothesis = result.problem_specification["hypothesis"]
    base = np.asarray(hypothesis["base_parameters"], dtype=np.float64)
    full = np.broadcast_to(base, (result.draws, base.size)).copy()
    specifications = {
        parameter["name"]: parameter
        for parameter in hypothesis["free_parameters"]
    }
    for column, name in enumerate(result.parameter_names):
        try:
            index = specifications[name]["index"]
        except KeyError as error:
            raise ValueError(
                f"result metadata does not define free parameter {name!r}"
            ) from error
        full[:, index] = result.samples[:, column]
    return full


def effective_error_samples(result, spectrum="spectrum"):
    """Return the likelihood scale for every posterior draw and detector bin."""
    spectrum = _spectrum_name(result, spectrum)
    observation = result.observations[spectrum]
    settings = _likelihood_settings(result, spectrum)
    distribution = settings.get("distribution")
    if distribution not in {"Gaussian", "StudentT"}:
        raise ValueError(
            f"unsupported likelihood distribution {distribution!r}"
        )
    parameters = _full_parameter_samples(result)
    scale_index = settings.get("ln_error_scale_index")
    jitter_index = settings.get("log10_jitter_index")
    if scale_index is None:
        scale = np.full(result.draws, settings.get("fixed_error_scale", 1.0))
    else:
        scale = np.exp(parameters[:, scale_index])
    if jitter_index is None:
        jitter = np.full(result.draws, settings.get("fixed_jitter", 0.0))
    else:
        jitter = 10.0 ** parameters[:, jitter_index]
    return np.sqrt(
        (scale[:, None] * observation.error[None, :]) ** 2
        + jitter[:, None] ** 2
    )


def posterior_predictive_samples(result, spectrum="spectrum", random_seed=0):
    """Draw noisy replicated spectra from every stored posterior sample."""
    if not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    spectrum = _spectrum_name(result, spectrum)
    settings = _likelihood_settings(result, spectrum)
    effective_error = effective_error_samples(result, spectrum)
    random = np.random.default_rng(random_seed)
    distribution = settings.get("distribution")
    if distribution == "Gaussian":
        standardized_noise = random.normal(size=effective_error.shape)
    elif distribution == "StudentT":
        degrees_of_freedom = float(settings["degrees_of_freedom"])
        standardized_noise = random.standard_t(
            degrees_of_freedom,
            size=effective_error.shape,
        )
    else:
        raise ValueError(
            f"unsupported likelihood distribution {distribution!r}"
        )
    return result.predictions[spectrum] + effective_error * standardized_noise


def posterior_predictive_bands(
    result,
    spectrum="spectrum",
    probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    random_seed=0,
):
    """Summarize replicated data, including model and measurement uncertainty."""
    samples = posterior_predictive_samples(
        result,
        spectrum=spectrum,
        random_seed=random_seed,
    )
    return posterior_percentile_bands(
        samples,
        probabilities=probabilities,
        weights=result.weights,
    )


@dataclass(frozen=True)
class SpectrumFitDiagnostics:
    """One spectrum's central residuals and uncertainty-coverage summaries."""

    spectrum: str
    likelihood_distribution: str
    likelihood_scale_description: str
    wavelength: np.ndarray
    central_prediction: np.ndarray
    central_effective_error: np.ndarray
    scaled_residuals: np.ndarray
    latent_bands: np.ndarray
    predictive_bands: np.ndarray
    central_chi_squared: float | None
    degrees_of_freedom: int | None
    reduced_chi_squared: float | None
    residual_mean: float
    residual_standard_deviation: float
    residual_root_mean_square: float
    maximum_absolute_residual: float
    minimum_effective_to_reported_error_ratio: float
    median_effective_to_reported_error_ratio: float
    maximum_effective_to_reported_error_ratio: float
    latent_68_percent_coverage: float
    latent_95_percent_coverage: float
    predictive_68_percent_coverage: float
    predictive_95_percent_coverage: float
    large_residual_indices: tuple[int, ...]
    large_residual_threshold: float

    def as_dict(self):
        """Return scalar diagnostics while excluding wavelength-sized arrays."""
        return {
            "spectrum": self.spectrum,
            "likelihood_distribution": self.likelihood_distribution,
            "likelihood_scale_description": self.likelihood_scale_description,
            "pixels": int(self.wavelength.size),
            "central_chi_squared": self.central_chi_squared,
            "degrees_of_freedom": self.degrees_of_freedom,
            "reduced_chi_squared": self.reduced_chi_squared,
            "residual_mean": self.residual_mean,
            "residual_standard_deviation": (
                self.residual_standard_deviation
            ),
            "residual_root_mean_square": self.residual_root_mean_square,
            "maximum_absolute_residual": self.maximum_absolute_residual,
            "minimum_effective_to_reported_error_ratio": (
                self.minimum_effective_to_reported_error_ratio
            ),
            "median_effective_to_reported_error_ratio": (
                self.median_effective_to_reported_error_ratio
            ),
            "maximum_effective_to_reported_error_ratio": (
                self.maximum_effective_to_reported_error_ratio
            ),
            "latent_68_percent_coverage": self.latent_68_percent_coverage,
            "latent_95_percent_coverage": self.latent_95_percent_coverage,
            "predictive_68_percent_coverage": (
                self.predictive_68_percent_coverage
            ),
            "predictive_95_percent_coverage": (
                self.predictive_95_percent_coverage
            ),
            "large_residual_threshold": self.large_residual_threshold,
            "large_residual_indices": list(self.large_residual_indices),
            "large_residual_wavelengths": self.wavelength[
                list(self.large_residual_indices)
            ].tolist(),
        }


def _coverage(observed, lower, upper):
    return float(np.mean((observed >= lower) & (observed <= upper)))


def evaluate_fit_diagnostics(
    result,
    spectrum="spectrum",
    *,
    random_seed=0,
    large_residual_threshold=3.0,
):
    """Evaluate latent, predictive, and noise-scaled fit diagnostics."""
    if large_residual_threshold <= 0.0:
        raise ValueError("large_residual_threshold must be positive")
    spectrum = _spectrum_name(result, spectrum)
    observation = result.observations[spectrum]
    settings = _likelihood_settings(result, spectrum)
    distribution = settings["distribution"]
    latent_bands = result.prediction_bands(spectrum)
    predictive_bands = posterior_predictive_bands(
        result,
        spectrum=spectrum,
        random_seed=random_seed,
    )
    central_prediction = latent_bands[2]
    effective_error = effective_error_samples(result, spectrum)
    central_effective_error = posterior_percentile_bands(
        effective_error,
        probabilities=(0.5,),
        weights=result.weights,
    )[0]
    scaled_residuals = (
        observation.flux - central_prediction
    ) / central_effective_error
    if distribution == "Gaussian":
        chi_squared = float(np.sum(scaled_residuals**2))
        if len(result.observations) == 1:
            degrees_of_freedom = observation.wavelength.size - len(
                result.parameter_names
            )
            if degrees_of_freedom <= 0:
                degrees_of_freedom = None
        else:
            degrees_of_freedom = None
        reduced_chi_squared = (
            None
            if degrees_of_freedom is None
            else chi_squared / degrees_of_freedom
        )
        scale_description = "Gaussian standard deviation"
    else:
        chi_squared = None
        degrees_of_freedom = None
        reduced_chi_squared = None
        scale_description = "Student-t scale (not a Gaussian standard deviation)"
    large_indices = tuple(
        int(index)
        for index in np.flatnonzero(
            np.abs(scaled_residuals) >= large_residual_threshold
        )
    )
    effective_to_reported = central_effective_error / observation.error
    return SpectrumFitDiagnostics(
        spectrum=spectrum,
        likelihood_distribution=distribution,
        likelihood_scale_description=scale_description,
        wavelength=observation.wavelength,
        central_prediction=central_prediction,
        central_effective_error=central_effective_error,
        scaled_residuals=scaled_residuals,
        latent_bands=latent_bands,
        predictive_bands=predictive_bands,
        central_chi_squared=chi_squared,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_squared=reduced_chi_squared,
        residual_mean=float(np.mean(scaled_residuals)),
        residual_standard_deviation=float(np.std(scaled_residuals)),
        residual_root_mean_square=float(
            np.sqrt(np.mean(scaled_residuals**2))
        ),
        maximum_absolute_residual=float(np.max(np.abs(scaled_residuals))),
        minimum_effective_to_reported_error_ratio=float(
            np.min(effective_to_reported)
        ),
        median_effective_to_reported_error_ratio=float(
            np.median(effective_to_reported)
        ),
        maximum_effective_to_reported_error_ratio=float(
            np.max(effective_to_reported)
        ),
        latent_68_percent_coverage=_coverage(
            observation.flux,
            latent_bands[1],
            latent_bands[3],
        ),
        latent_95_percent_coverage=_coverage(
            observation.flux,
            latent_bands[0],
            latent_bands[4],
        ),
        predictive_68_percent_coverage=_coverage(
            observation.flux,
            predictive_bands[1],
            predictive_bands[3],
        ),
        predictive_95_percent_coverage=_coverage(
            observation.flux,
            predictive_bands[0],
            predictive_bands[4],
        ),
        large_residual_indices=large_indices,
        large_residual_threshold=float(large_residual_threshold),
    )


def _flux_display_scale(*values):
    maximum = max(float(np.max(np.abs(value))) for value in values)
    if not np.isfinite(maximum) or maximum == 0.0:
        return 1.0, 0
    exponent = int(np.floor(np.log10(maximum)))
    return 10.0**exponent, exponent


def plot_inference_result(
    result,
    spectrum="spectrum",
    *,
    random_seed=0,
    show_posterior_predictive=True,
):
    """Plot data, latent/predictive bands, residuals, and intrinsic profile."""
    import matplotlib.pyplot as plt

    diagnostics = evaluate_fit_diagnostics(
        result,
        spectrum=spectrum,
        random_seed=random_seed,
    )
    observation = result.observations[diagnostics.spectrum]
    reconstruction = result.reconstructions.get(diagnostics.spectrum)
    rows = 3 if reconstruction is not None else 2
    height_ratios = (3, 1, 3) if reconstruction is not None else (3, 1)
    figure, axes = plt.subplots(
        rows,
        1,
        figsize=(9.5, 9.0 if reconstruction is not None else 6.5),
        sharex=False,
        gridspec_kw={"height_ratios": height_ratios},
        layout="constrained",
    )
    profile_axis, residual_axis = axes[:2]
    scale, exponent = _flux_display_scale(
        observation.flux,
        diagnostics.central_prediction,
    )
    profile_axis.errorbar(
        observation.wavelength,
        observation.flux / scale,
        yerr=observation.error / scale,
        color="0.45",
        linewidth=0.8,
        label="data (reported errors)",
    )
    if show_posterior_predictive:
        profile_axis.fill_between(
            observation.wavelength,
            diagnostics.predictive_bands[0] / scale,
            diagnostics.predictive_bands[4] / scale,
            color="0.65",
            alpha=0.18,
            label="95% posterior-predictive band",
        )
    profile_axis.fill_between(
        observation.wavelength,
        diagnostics.latent_bands[0] / scale,
        diagnostics.latent_bands[4] / scale,
        color="tab:blue",
        alpha=0.15,
        label="95% latent-model band",
    )
    profile_axis.fill_between(
        observation.wavelength,
        diagnostics.latent_bands[1] / scale,
        diagnostics.latent_bands[3] / scale,
        color="tab:blue",
        alpha=0.3,
        label="68% latent-model band",
    )
    profile_axis.plot(
        observation.wavelength,
        diagnostics.central_prediction / scale,
        color="tab:blue",
        label="latent posterior median",
    )
    profile_axis.set_ylabel(rf"flux density / $10^{{{exponent}}}$")
    profile_axis.legend(frameon=False, fontsize="small")

    residual_axis.axhline(0.0, color="0.4", linewidth=1.0)
    residual_axis.plot(
        observation.wavelength,
        diagnostics.scaled_residuals,
        color="tab:blue",
    )
    residual_axis.axhline(3.0, color="0.7", linestyle=":", linewidth=1.0)
    residual_axis.axhline(-3.0, color="0.7", linestyle=":", linewidth=1.0)
    residual_axis.set_ylabel("residual / likelihood scale")
    residual_axis.set_xlabel(r"wavelength ($\AA$)")

    if reconstruction is not None:
        intrinsic_axis = axes[2]
        intrinsic_bands = posterior_percentile_bands(
            reconstruction.intrinsic,
            weights=result.weights,
        )
        intrinsic_scale, intrinsic_exponent = _flux_display_scale(
            intrinsic_bands[2]
        )
        intrinsic_axis.fill_between(
            reconstruction.model_wavelength,
            intrinsic_bands[0] / intrinsic_scale,
            intrinsic_bands[4] / intrinsic_scale,
            color="tab:orange",
            alpha=0.25,
            label="95% intrinsic-profile band",
        )
        intrinsic_axis.plot(
            reconstruction.model_wavelength,
            intrinsic_bands[2] / intrinsic_scale,
            color="tab:orange",
            label="intrinsic posterior median",
        )
        intrinsic_axis.set_xlabel(r"wavelength ($\AA$)")
        intrinsic_axis.set_ylabel(
            rf"intrinsic flux density / $10^{{{intrinsic_exponent}}}$"
        )
        intrinsic_axis.legend(frameon=False, fontsize="small")

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"{result.name}: {diagnostics.spectrum}")
    return figure


def plot_inference_comparison(
    comparison,
    spectrum="spectrum",
    *,
    random_seed=0,
    show_68_percent_bands=True,
):
    """Overlay central latent curves and scaled residuals for several models."""
    import matplotlib.pyplot as plt

    first_result = comparison.runs[0].result
    spectrum = _spectrum_name(first_result, spectrum)
    observation = first_result.observations[spectrum]
    figure, (profile_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
        layout="constrained",
    )
    scale, exponent = _flux_display_scale(observation.flux)
    profile_axis.errorbar(
        observation.wavelength,
        observation.flux / scale,
        yerr=observation.error / scale,
        color="0.7",
        linewidth=0.8,
        label="data (reported errors)",
    )
    residual_axis.axhline(0.0, color="0.4", linewidth=1.0)
    colors = plt.get_cmap("tab10").colors
    line_styles = ("-", "--", "-.", ":")
    for index, run in enumerate(comparison.runs):
        result = run.result
        diagnostics = evaluate_fit_diagnostics(
            result,
            spectrum=spectrum,
            random_seed=random_seed + index,
        )
        color = colors[index % len(colors)]
        line_style = line_styles[index % len(line_styles)]
        if show_68_percent_bands:
            profile_axis.fill_between(
                observation.wavelength,
                diagnostics.latent_bands[1] / scale,
                diagnostics.latent_bands[3] / scale,
                color=color,
                alpha=0.12,
            )
        profile_axis.plot(
            observation.wavelength,
            diagnostics.central_prediction / scale,
            color=color,
            linestyle=line_style,
            linewidth=2.0,
            label=result.name,
        )
        residual_axis.plot(
            observation.wavelength,
            diagnostics.scaled_residuals,
            color=color,
            linestyle=line_style,
            label=result.name,
        )
    profile_axis.set_ylabel(rf"flux density / $10^{{{exponent}}}$")
    profile_axis.legend(frameon=False, ncol=2, fontsize="small")
    residual_axis.set_xlabel(r"wavelength ($\AA$)")
    residual_axis.set_ylabel("residual / model scale")
    residual_axis.legend(frameon=False, ncol=2, fontsize="small")
    for axis in (profile_axis, residual_axis):
        axis.grid(alpha=0.2)
    figure.suptitle(f"Model comparison: best = {comparison.best_model}")
    return figure
