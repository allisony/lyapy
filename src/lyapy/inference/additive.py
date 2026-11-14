"""Configuration helpers for fitted background and airglow components."""

from dataclasses import replace
import math

from lyapy.inference.common import (
    InferenceConfiguration,
    TruncatedNormalParameter,
    UniformParameter,
)
from lyapy.jax_additive import (
    AffineBackgroundComponent,
    AffineContinuumComponent,
    CompositeAdditiveModel,
    GaussianAirglowComponent,
)


def _bounds_contain(name, value, bounds):
    lower, upper = (float(item) for item in bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} bounds must be finite and increasing")
    if not lower <= value <= upper:
        raise ValueError(f"base {name} must lie within its bounds")
    return lower, upper


def augment_configuration_with_background_and_airglow(
    configuration,
    forward_model,
    *,
    airglow_center,
    baseline_flux_density_scale,
    airglow_flux_scale,
    base_airglow_fwhm,
    airglow_fwhm_bounds,
    shared_airglow_fwhm_name=None,
    prefix="",
    baseline_kind="background",
    baseline_reference=None,
    baseline_wavelength_scale=1.0,
    base_baseline_intercept=0.0,
    baseline_intercept_bounds=(-5.0, 5.0),
    include_baseline_slope=True,
    base_baseline_slope=0.0,
    baseline_slope_bounds=(-5.0, 5.0),
    base_airglow_scaled_total_flux=1.0,
    airglow_scaled_total_flux_bounds=(-10.0, 10.0),
    base_airglow_velocity_offset=0.0,
    airglow_velocity_offset_bounds=(-30.0, 30.0),
    airglow_velocity_offset_standard_deviation=10.0,
):
    """Append one exposure's fitted background and airglow parameters.

    The baseline can be either an additive extraction ``"background"`` or an
    astrophysical ``"stellar_continuum"``. A stellar continuum is placed
    before ISM absorption and the point-source LSF; a background is placed
    after both. Airglow is always additive and detector-bin integrated after
    the point-source model. Its FWHM is the complete observed profile in km/s,
    so it is deliberately not reconvolved with the point-source LSF.

    The two required scales keep the sampled background flux density and
    signed integrated airglow flux numerically near unity without mixing their
    physical units. Scientific prior bounds remain explicit. Use a unique
    ``prefix`` for every exposure in a joint fit. Pass one common
    ``shared_airglow_fwhm_name`` to reuse a width across exposures with the
    same grating and aperture; all other nuisance parameters remain
    exposure-specific.
    """
    if not isinstance(configuration, InferenceConfiguration):
        raise TypeError("configuration must be an InferenceConfiguration")
    if not hasattr(forward_model, "observation_operator"):
        raise TypeError("forward_model must define observation_operator")
    if getattr(forward_model, "additive_model", None) is not None:
        raise ValueError("forward_model already contains an additive model")
    if getattr(forward_model, "continuum_model", None) is not None:
        raise ValueError("forward_model already contains a continuum model")
    if baseline_kind not in {"background", "stellar_continuum"}:
        raise ValueError(
            "baseline_kind must be 'background' or 'stellar_continuum'"
        )
    airglow_center = float(airglow_center)
    baseline_flux_density_scale = float(baseline_flux_density_scale)
    airglow_flux_scale = float(airglow_flux_scale)
    base_airglow_fwhm = float(base_airglow_fwhm)
    baseline_wavelength_scale = float(baseline_wavelength_scale)
    if not math.isfinite(airglow_center) or airglow_center <= 0.0:
        raise ValueError("airglow_center must be positive and finite")
    if (
        not math.isfinite(baseline_flux_density_scale)
        or baseline_flux_density_scale <= 0.0
    ):
        raise ValueError(
            "baseline_flux_density_scale must be positive and finite"
        )
    if not math.isfinite(airglow_flux_scale) or airglow_flux_scale <= 0.0:
        raise ValueError("airglow_flux_scale must be positive and finite")
    if (
        not math.isfinite(baseline_wavelength_scale)
        or baseline_wavelength_scale <= 0.0
    ):
        raise ValueError(
            "baseline_wavelength_scale must be positive and finite"
        )
    if (
        not math.isfinite(airglow_velocity_offset_standard_deviation)
        or airglow_velocity_offset_standard_deviation <= 0.0
    ):
        raise ValueError(
            "airglow_velocity_offset_standard_deviation must be positive"
        )

    prefix = str(prefix).strip()
    stem = f"{prefix}_" if prefix else ""
    names = list(configuration.parameter_names)
    base = list(configuration.base_parameters)
    free = list(configuration.free_parameters)

    def append_uniform(short_name, initial, bounds, *, full_name=None):
        name = f"{stem}{short_name}" if full_name is None else full_name
        if name in names:
            raise ValueError(f"configuration already contains {name!r}")
        initial = float(initial)
        lower, upper = _bounds_contain(name, initial, bounds)
        index = len(names)
        names.append(name)
        base.append(initial)
        free.append(UniformParameter(name, index, lower, upper))
        return index

    baseline_name = (
        "background" if baseline_kind == "background" else "continuum"
    )
    intercept_index = append_uniform(
        f"{baseline_name}_intercept",
        base_baseline_intercept,
        baseline_intercept_bounds,
    )
    slope_index = None
    if include_baseline_slope:
        slope_index = append_uniform(
            f"{baseline_name}_slope",
            base_baseline_slope,
            baseline_slope_bounds,
        )
    airglow_flux_index = append_uniform(
        "airglow_scaled_total_flux",
        base_airglow_scaled_total_flux,
        airglow_scaled_total_flux_bounds,
    )

    velocity_name = f"{stem}airglow_velocity_offset"
    if velocity_name in names:
        raise ValueError(
            f"configuration already contains {velocity_name!r}"
        )
    base_airglow_velocity_offset = float(base_airglow_velocity_offset)
    velocity_bounds = _bounds_contain(
        velocity_name,
        base_airglow_velocity_offset,
        airglow_velocity_offset_bounds,
    )
    if not velocity_bounds[0] <= 0.0 <= velocity_bounds[1]:
        raise ValueError(
            "airglow_velocity_offset_bounds must contain the zero-centered "
            "prior mean"
        )
    velocity_index = len(names)
    names.append(velocity_name)
    base.append(base_airglow_velocity_offset)
    free.append(
        TruncatedNormalParameter(
            velocity_name,
            velocity_index,
            *velocity_bounds,
            mean=0.0,
            standard_deviation=float(
                airglow_velocity_offset_standard_deviation
            ),
        )
    )
    fwhm_bounds = tuple(float(value) for value in airglow_fwhm_bounds)
    if len(fwhm_bounds) != 2 or fwhm_bounds[0] <= 0.0:
        raise ValueError("airglow_fwhm_bounds must be strictly positive")
    if shared_airglow_fwhm_name is None:
        fwhm_index = append_uniform(
            "airglow_fwhm",
            base_airglow_fwhm,
            fwhm_bounds,
        )
    else:
        shared_airglow_fwhm_name = str(shared_airglow_fwhm_name).strip()
        if not shared_airglow_fwhm_name:
            raise ValueError("shared_airglow_fwhm_name must not be empty")
        if shared_airglow_fwhm_name in names:
            fwhm_index = names.index(shared_airglow_fwhm_name)
            existing = next(
                (
                    parameter
                    for parameter in free
                    if parameter.index == fwhm_index
                ),
                None,
            )
            if not isinstance(existing, UniformParameter):
                raise ValueError(
                    "shared airglow FWHM must be an existing free uniform "
                    "parameter"
                )
            if (
                base[fwhm_index] != base_airglow_fwhm
                or (existing.lower, existing.upper) != fwhm_bounds
            ):
                raise ValueError(
                    "shared airglow FWHM initial value and bounds must match"
                )
        else:
            fwhm_index = append_uniform(
                "airglow_fwhm",
                base_airglow_fwhm,
                fwhm_bounds,
                full_name=shared_airglow_fwhm_name,
            )

    operator = forward_model.observation_operator
    if baseline_reference is None:
        baseline_reference = float(
            0.5 * (operator.detector_edges[0] + operator.detector_edges[-1])
        )
    baseline_reference = float(baseline_reference)
    if not math.isfinite(baseline_reference) or baseline_reference <= 0.0:
        raise ValueError("baseline_reference must be positive and finite")

    baseline_class = (
        AffineBackgroundComponent
        if baseline_kind == "background"
        else AffineContinuumComponent
    )
    baseline_model = CompositeAdditiveModel(
        (
            baseline_class(
                baseline_reference,
                intercept_index,
                slope_index,
                flux_scale=baseline_flux_density_scale,
                wavelength_scale=baseline_wavelength_scale,
            ),
        )
    )
    airglow_model = CompositeAdditiveModel(
        (
            GaussianAirglowComponent(
                airglow_center,
                airglow_flux_index,
                fwhm_index,
                velocity_index,
                flux_scale=airglow_flux_scale,
            ),
        )
    )
    augmented = InferenceConfiguration(
        base_parameters=tuple(base),
        free_parameters=tuple(free),
        parameter_names=tuple(names),
    )
    if baseline_kind == "background":
        additive_model = CompositeAdditiveModel(
            (*baseline_model.components, *airglow_model.components)
        )
        return augmented, replace(
            forward_model,
            additive_model=additive_model,
        )
    return augmented, replace(
        forward_model,
        additive_model=airglow_model,
        continuum_model=baseline_model,
    )
