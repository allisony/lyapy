"""Curated atomic transition data used by emission and absorption models."""

import math
from dataclasses import asdict, dataclass
from types import MappingProxyType


MORTON_2003_TABLE_2 = (
    "Morton, D. C. 2003, ApJS, 149, 205, Table 2; "
    "doi:10.1086/377639"
)


@dataclass(frozen=True)
class AtomicTransition:
    """Atomic metadata for one transition.

    Wavelengths are vacuum Angstroms, oscillator strengths are dimensionless,
    and damping rates are total radiative decay rates in s^-1.  A transition
    may omit the absorption data when it is intended only as a user-defined
    phenomenological emission component.
    """

    key: str
    species: str
    label: str
    rest_wavelength: float
    oscillator_strength: float | None = None
    damping_rate: float | None = None
    reference: str | None = None

    def __post_init__(self):
        if not self.key or not self.species or not self.label:
            raise ValueError("transition key, species, and label must not be empty")
        if not math.isfinite(self.rest_wavelength) or self.rest_wavelength <= 0.0:
            raise ValueError("rest_wavelength must be finite and positive")
        supplied = (
            self.oscillator_strength is not None,
            self.damping_rate is not None,
        )
        if supplied[0] != supplied[1]:
            raise ValueError(
                "oscillator_strength and damping_rate must be provided together"
            )
        if self.oscillator_strength is not None:
            if (
                not math.isfinite(self.oscillator_strength)
                or self.oscillator_strength <= 0.0
            ):
                raise ValueError("oscillator_strength must be finite and positive")
            if not math.isfinite(self.damping_rate) or self.damping_rate <= 0.0:
                raise ValueError("damping_rate must be finite and positive")
            if not self.reference:
                raise ValueError("absorption data require a reference")

    @property
    def has_absorption_data(self):
        """Whether this transition can parameterize a Voigt optical depth."""
        return self.oscillator_strength is not None

    def absorption_parameters(self):
        """Return ``(wavelength, f, Gamma)`` for an absorption calculation."""
        if not self.has_absorption_data:
            raise ValueError(f"transition {self.key!r} has no absorption data")
        return (
            self.rest_wavelength,
            self.oscillator_strength,
            self.damping_rate,
        )

    def as_dict(self):
        return asdict(self)


_TRANSITIONS = (
    AtomicTransition(
        "h_i_1215",
        "H I",
        "Ly-alpha",
        1215.6700,
        4.164e-1,
        6.265e8,
        MORTON_2003_TABLE_2,
    ),
    AtomicTransition(
        "si_iii_1206",
        "Si III",
        "Si III 1206",
        1206.500,
        1.63,
        2.48e9,
        MORTON_2003_TABLE_2,
    ),
    AtomicTransition(
        "n_v_1238",
        "N V",
        "N V 1238",
        1238.821,
        1.560e-1,
        3.391e8,
        MORTON_2003_TABLE_2,
    ),
    AtomicTransition(
        "n_v_1242",
        "N V",
        "N V 1242",
        1242.804,
        7.770e-2,
        3.356e8,
        MORTON_2003_TABLE_2,
    ),
    AtomicTransition(
        "o_v_1218",
        "O V",
        "O V 1218",
        1218.344,
        1.46e-6,
        2.18e3,
        MORTON_2003_TABLE_2,
    ),
    AtomicTransition(
        "c_ii_1334",
        "C II",
        "C II 1334",
        1334.5323,
        1.28e-1,
        2.88e8,
        MORTON_2003_TABLE_2,
    ),
    AtomicTransition(
        "c_ii_1335a",
        "C II*",
        "C II* 1335a",
        1335.6627,
        1.28e-2,
        2.88e8,
        MORTON_2003_TABLE_2,
    ),
    AtomicTransition(
        "c_ii_1335b",
        "C II*",
        "C II* 1335b",
        1335.7077,
        1.15e-1,
        2.88e8,
        MORTON_2003_TABLE_2,
    ),
    AtomicTransition(
        "mg_ii_k",
        "Mg II",
        "Mg II k",
        2796.3543,
        6.155e-1,
        2.625e8,
        MORTON_2003_TABLE_2,
    ),
    AtomicTransition(
        "mg_ii_h",
        "Mg II",
        "Mg II h",
        2803.5315,
        3.058e-1,
        2.595e8,
        MORTON_2003_TABLE_2,
    ),
)

ATOMIC_TRANSITIONS = MappingProxyType(
    {transition.key: transition for transition in _TRANSITIONS}
)


def get_atomic_transition(transition):
    """Resolve a catalog key or validate a user-defined transition."""
    if isinstance(transition, AtomicTransition):
        return transition
    try:
        return ATOMIC_TRANSITIONS[transition]
    except (KeyError, TypeError) as error:
        choices = ", ".join(ATOMIC_TRANSITIONS)
        raise ValueError(
            f"unknown atomic transition {transition!r}; choices: {choices}"
        ) from error
