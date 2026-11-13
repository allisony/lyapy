"""Curated UV line groups and explicit multiplet constructors."""

from dataclasses import dataclass
from types import MappingProxyType

from lyapy.atomic_data import ATOMIC_TRANSITIONS, AtomicTransition
from lyapy.jax_components import (
    CompositeEmissionModel,
    GaussianEmissionComponent,
    VoigtEmissionComponent,
)


@dataclass(frozen=True)
class SpectralLineGroup:
    """A singlet or multiplet whose components may share kinematics."""

    key: str
    label: str
    transitions: tuple[AtomicTransition, ...]
    optically_thin_relative_strengths: tuple[float, ...] | None = None

    def __post_init__(self):
        transitions = tuple(self.transitions)
        if not self.key or not self.label:
            raise ValueError("line-group key and label must not be empty")
        if not transitions:
            raise ValueError("a line group must contain at least one transition")
        if not all(
            isinstance(transition, AtomicTransition)
            for transition in transitions
        ):
            raise TypeError("transitions must contain AtomicTransition values")
        keys = [transition.key for transition in transitions]
        if len(keys) != len(set(keys)):
            raise ValueError("transition keys must be unique within a line group")
        if self.optically_thin_relative_strengths is not None:
            strengths = tuple(self.optically_thin_relative_strengths)
            if len(strengths) != len(transitions):
                raise ValueError(
                    "optically thin strengths must match the transitions"
                )
            if any(strength <= 0.0 for strength in strengths):
                raise ValueError("optically thin strengths must be positive")
            object.__setattr__(
                self,
                "optically_thin_relative_strengths",
                strengths,
            )
        object.__setattr__(self, "transitions", transitions)

    def as_dict(self):
        return {
            "key": self.key,
            "label": self.label,
            "transitions": [
                transition.as_dict() for transition in self.transitions
            ],
            "optically_thin_relative_strengths": (
                None
                if self.optically_thin_relative_strengths is None
                else list(self.optically_thin_relative_strengths)
            ),
        }


_LINE_GROUPS = (
    SpectralLineGroup(
        "lya",
        "H I Ly-alpha",
        (ATOMIC_TRANSITIONS["h_i_1215"],),
        (1.0,),
    ),
    SpectralLineGroup(
        "si_iii_1206",
        "Si III 1206",
        (ATOMIC_TRANSITIONS["si_iii_1206"],),
        (1.0,),
    ),
    SpectralLineGroup(
        "n_v_1238_1242",
        "N V resonance doublet",
        (
            ATOMIC_TRANSITIONS["n_v_1238"],
            ATOMIC_TRANSITIONS["n_v_1242"],
        ),
        (2.0, 1.0),
    ),
    SpectralLineGroup(
        "o_v_1218",
        "O V 1218",
        (ATOMIC_TRANSITIONS["o_v_1218"],),
        (1.0,),
    ),
    SpectralLineGroup(
        "c_ii_1334_1335",
        "C II 1334/1335 multiplet",
        (
            ATOMIC_TRANSITIONS["c_ii_1334"],
            ATOMIC_TRANSITIONS["c_ii_1335a"],
            ATOMIC_TRANSITIONS["c_ii_1335b"],
        ),
    ),
    SpectralLineGroup(
        "mg_ii_hk",
        "Mg II h and k",
        (
            ATOMIC_TRANSITIONS["mg_ii_k"],
            ATOMIC_TRANSITIONS["mg_ii_h"],
        ),
        (2.0, 1.0),
    ),
)

SPECTRAL_LINE_GROUPS = MappingProxyType(
    {group.key: group for group in _LINE_GROUPS}
)


def get_spectral_line_group(group):
    """Resolve a stable catalog key or validate a user-defined line group."""
    if isinstance(group, SpectralLineGroup):
        return group
    try:
        return SPECTRAL_LINE_GROUPS[group]
    except (KeyError, TypeError) as error:
        choices = ", ".join(SPECTRAL_LINE_GROUPS)
        raise ValueError(
            f"unknown spectral line group {group!r}; choices: {choices}"
        ) from error


def _component_weights(
    group,
    relative_weights,
    log_relative_weight_indices,
):
    number = len(group.transitions)
    if relative_weights is not None and log_relative_weight_indices is not None:
        raise ValueError(
            "provide relative_weights or log_relative_weight_indices, not both"
        )
    if log_relative_weight_indices is not None:
        indices = tuple(log_relative_weight_indices)
        if len(indices) != number - 1:
            raise ValueError(
                "one log relative-weight index is required after the reference line"
            )
        if any(index < 0 for index in indices):
            raise ValueError("log relative-weight indices must be non-negative")
        return tuple(
            (None, 1.0) if index is None else (index, 1.0)
            for index in (None, *indices)
        )
    if relative_weights is None:
        if number > 1:
            raise ValueError(
                "multiplet ratios must be explicit; pass relative_weights or "
                "log_relative_weight_indices"
            )
        relative_weights = (1.0,)
    weights = tuple(relative_weights)
    if len(weights) != number or any(weight <= 0.0 for weight in weights):
        raise ValueError(
            "relative_weights must contain one positive value per transition"
        )
    return tuple((None, weight) for weight in weights)


def gaussian_components_for_line_group(
    group,
    velocity_index,
    fwhm_index,
    *,
    relative_weights=None,
    log_relative_weight_indices=None,
):
    """Create Gaussian components tied to one velocity and one FWHM."""
    group = get_spectral_line_group(group)
    weights = _component_weights(
        group,
        relative_weights,
        log_relative_weight_indices,
    )
    return tuple(
        GaussianEmissionComponent(
            transition.rest_wavelength,
            velocity_index,
            fwhm_index,
            log_weight_index=log_index,
            fixed_weight=fixed_weight,
        )
        for transition, (log_index, fixed_weight) in zip(
            group.transitions,
            weights,
        )
    )


def voigt_components_for_line_group(
    group,
    velocity_index,
    gaussian_fwhm_index,
    lorentz_fwhm_index,
    *,
    relative_weights=None,
    log_relative_weight_indices=None,
):
    """Create Voigt components tied to one velocity and shared widths."""
    group = get_spectral_line_group(group)
    weights = _component_weights(
        group,
        relative_weights,
        log_relative_weight_indices,
    )
    return tuple(
        VoigtEmissionComponent(
            transition.rest_wavelength,
            velocity_index,
            gaussian_fwhm_index,
            lorentz_fwhm_index,
            log_weight_index=log_index,
            fixed_weight=fixed_weight,
        )
        for transition, (log_index, fixed_weight) in zip(
            group.transitions,
            weights,
        )
    )


def gaussian_emission_model_for_line_group(
    group,
    velocity_index,
    fwhm_index,
    log_total_flux_index,
    *,
    relative_weights=None,
    log_relative_weight_indices=None,
    reversal=None,
):
    """Build a total-flux Gaussian singlet or tied multiplet model."""
    return CompositeEmissionModel(
        gaussian_components_for_line_group(
            group,
            velocity_index,
            fwhm_index,
            relative_weights=relative_weights,
            log_relative_weight_indices=log_relative_weight_indices,
        ),
        log_total_flux_index,
        reversal=reversal,
    )


def voigt_emission_model_for_line_group(
    group,
    velocity_index,
    gaussian_fwhm_index,
    lorentz_fwhm_index,
    log_total_flux_index,
    *,
    relative_weights=None,
    log_relative_weight_indices=None,
    reversal=None,
):
    """Build a total-flux Voigt singlet or tied multiplet model."""
    return CompositeEmissionModel(
        voigt_components_for_line_group(
            group,
            velocity_index,
            gaussian_fwhm_index,
            lorentz_fwhm_index,
            relative_weights=relative_weights,
            log_relative_weight_indices=log_relative_weight_indices,
        ),
        log_total_flux_index,
        reversal=reversal,
    )
