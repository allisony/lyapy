"""Spectrum loading and deterministic STIS preprocessing."""

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
from scipy.io import readsav

from .instrument import interpolate_stis_lsf


GJ411_SPECTRUM_RESOURCE = "HD-95735-E140M-odn905010_1.xl_stis"
STIS_E140M_LSF_RESOURCE = "STIS_E140M_LSF.dat"
GJ411_CONTINUUM_WINDOWS = ((1212.0, 1213.0), (1217.0, 1217.9))
GJ411_FIT_WINDOW = (1214.5, 1216.55)


def _as_vector(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class ObservedSpectrum:
    """One observed spectrum before continuum subtraction or cropping."""

    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    name: str = ""

    def __post_init__(self):
        wavelength = _as_vector(self.wavelength, "wavelength")
        flux = _as_vector(self.flux, "flux")
        error = _as_vector(self.error, "error")
        if flux.shape != wavelength.shape or error.shape != wavelength.shape:
            raise ValueError("wavelength, flux, and error must have matching shapes")
        if wavelength.size < 2 or np.any(np.diff(wavelength) <= 0.0):
            raise ValueError("wavelength must be strictly increasing")
        if np.any(error < 0.0):
            raise ValueError("error values must be non-negative")
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "error", error)


@dataclass(frozen=True)
class TabulatedLsf:
    """A tabulated line-spread function before interpolation to the data grid."""

    offset: np.ndarray
    profile: np.ndarray

    def __post_init__(self):
        offset = _as_vector(self.offset, "LSF offset")
        profile = _as_vector(self.profile, "LSF profile")
        if offset.shape != profile.shape:
            raise ValueError("LSF offset and profile must have matching shapes")
        if offset.size < 2 or np.any(np.diff(offset) <= 0.0):
            raise ValueError("LSF offsets must be strictly increasing")
        if np.any(profile < 0.0) or not np.any(profile > 0.0):
            raise ValueError("LSF profile must be non-negative with a positive sum")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "profile", profile)


@dataclass(frozen=True)
class PreparedObservation:
    """Continuum-subtracted fit data and its normalized convolution kernel."""

    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    convolution_kernel: np.ndarray
    continuum_coefficients: np.ndarray
    name: str = ""

    def __post_init__(self):
        wavelength = _as_vector(self.wavelength, "wavelength")
        flux = _as_vector(self.flux, "flux")
        error = _as_vector(self.error, "error")
        kernel = _as_vector(self.convolution_kernel, "convolution kernel")
        coefficients = _as_vector(
            self.continuum_coefficients,
            "continuum coefficients",
        )
        if flux.shape != wavelength.shape or error.shape != wavelength.shape:
            raise ValueError("wavelength, flux, and error must have matching shapes")
        if np.any(error <= 0.0):
            raise ValueError("prepared error values must be positive")
        if np.any(kernel < 0.0) or not np.any(kernel > 0.0):
            raise ValueError("convolution kernel must be non-negative and nonzero")
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "convolution_kernel", kernel)
        object.__setattr__(self, "continuum_coefficients", coefficients)


def load_idl_spectrum(
    path,
    *,
    wavelength_key="wave",
    flux_key="flux",
    error_key="photerr",
    name=None,
):
    """Load a legacy IDL save file into an :class:`ObservedSpectrum`."""
    path = Path(path)
    contents = readsav(path)
    missing = {
        key
        for key in (wavelength_key, flux_key, error_key)
        if key not in contents
    }
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"spectrum is missing required fields: {names}")
    return ObservedSpectrum(
        contents[wavelength_key],
        contents[flux_key],
        contents[error_key],
        name=path.stem if name is None else name,
    )


def load_tabulated_lsf(
    path,
    *,
    wavelength_column=0,
    profile_column=3,
    skiprows=2,
):
    """Load two columns of a text line-spread-function table."""
    table = np.loadtxt(path, skiprows=skiprows)
    if table.ndim != 2:
        raise ValueError("LSF table must be two-dimensional")
    try:
        return TabulatedLsf(
            table[:, wavelength_column],
            table[:, profile_column],
        )
    except IndexError as error:
        raise ValueError("requested LSF column is not present") from error


def fit_continuum(spectrum, windows, degree=1):
    """Fit a polynomial continuum within open wavelength windows."""
    if degree < 0:
        raise ValueError("continuum degree must be non-negative")
    mask = np.zeros(spectrum.wavelength.shape, dtype=bool)
    for lower, upper in windows:
        if lower >= upper:
            raise ValueError("continuum-window lower bounds must precede upper bounds")
        mask |= (spectrum.wavelength > lower) & (spectrum.wavelength < upper)
    if np.count_nonzero(mask) <= degree:
        raise ValueError("continuum windows do not contain enough samples")
    coefficients = np.polyfit(
        spectrum.wavelength[mask],
        spectrum.flux[mask],
        deg=degree,
    )
    return coefficients, np.polyval(coefficients, spectrum.wavelength)


def prepare_stis_observation(
    spectrum,
    lsf,
    *,
    continuum_windows,
    fit_window,
    continuum_degree=1,
    grating_dispersion=None,
):
    """Continuum-subtract, crop, and prepare an STIS LSF for one spectrum."""
    coefficients, continuum = fit_continuum(
        spectrum,
        continuum_windows,
        degree=continuum_degree,
    )
    lower, upper = fit_window
    if lower >= upper:
        raise ValueError("fit-window lower bound must precede its upper bound")
    fit_mask = (spectrum.wavelength >= lower) & (spectrum.wavelength <= upper)
    if np.count_nonzero(fit_mask) < 2:
        raise ValueError("fit window must contain at least two samples")
    wavelength = spectrum.wavelength[fit_mask]
    error = spectrum.error[fit_mask]
    if np.any(error <= 0.0):
        raise ValueError("fit window contains non-positive errors")
    if grating_dispersion is None:
        grating_dispersion = wavelength[1] - wavelength[0]
    kernel = interpolate_stis_lsf(
        lsf.offset,
        lsf.profile,
        grating_dispersion,
        wavelength,
    )
    return PreparedObservation(
        wavelength=wavelength,
        flux=spectrum.flux[fit_mask] - continuum[fit_mask],
        error=error,
        convolution_kernel=kernel,
        continuum_coefficients=coefficients,
        name=spectrum.name,
    )


def load_gj411_observation():
    """Load and reproduce the reviewed GJ 411 E140M preprocessing recipe."""
    package = resources.files("lyapy")
    with resources.as_file(package.joinpath(GJ411_SPECTRUM_RESOURCE)) as path:
        spectrum = load_idl_spectrum(path, name="GJ 411")
    with resources.as_file(package.joinpath(STIS_E140M_LSF_RESOURCE)) as path:
        lsf = load_tabulated_lsf(path)
    return prepare_stis_observation(
        spectrum,
        lsf,
        continuum_windows=GJ411_CONTINUUM_WINDOWS,
        fit_window=GJ411_FIT_WINDOW,
    )
