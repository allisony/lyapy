"""Spectrum loading and deterministic STIS preprocessing."""

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import MappingProxyType

import numpy as np
from astropy.io import fits
from scipy.io import readsav

from .instrument import interpolate_stis_lsf


GJ411_SPECTRUM_RESOURCE = "HD-95735-E140M-odn905010_1.xl_stis"
GJ729_SPECTRUM_RESOURCE = "GJ729_E140M-1425_020X020_58228.xl_stis"
STIS_E140M_LSF_RESOURCE = "data/stis_lsf/LSF_E140M_1200.txt"
STIS_G140M_LSF_RESOURCE = "data/stis_lsf/LSF_G140M_1200.txt"
STIS_LSF_SOURCE_URL = (
    "https://www.stsci.edu/hst/instrumentation/stis/performance/"
    "spectral-resolution"
)
GJ411_CONTINUUM_WINDOWS = ((1212.0, 1213.0), (1217.0, 1217.9))
GJ411_FIT_WINDOW = (1214.5, 1216.55)
AYRES_PHOTERR_SCALE = float(np.sqrt(2.0))
SPEED_OF_LIGHT_KM_S = 299792.458

# STIS Data Handbook, Table 2.9. These are bit values rather than mutually
# exclusive integer codes; a pixel can carry several conditions at once.
STIS_DQ_FLAGS = MappingProxyType(
    {
        1: "Reed-Solomon decoding error",
        2: "Lost data replaced by fill values",
        4: "Bad detector pixel",
        8: "Data masked by occulting bar",
        16: "Dark rate more than 5 sigma above the median",
        32: "Large blemish below 40% flat-field response (repeller wire)",
        64: "Vignetted pixel",
        128: "Overscan-region pixel",
        256: "Saturated or locally nonlinear pixel",
        512: "Bad pixel in a reference file",
        1024: "Small blemish between 40% and 70% flat-field response",
        2048: "More than 30% of background pixels rejected",
        4096: "Extracted flux affected by bad input data",
        8192: "Input pixel rejected during image combination",
        16384: "Extracted flux not CTI-corrected because gross counts are non-positive",
    }
)


def decode_stis_dq(value):
    """Return the documented STIS conditions present in one DQ value."""
    value = int(value)
    if value < 0:
        raise ValueError("STIS DQ values must be non-negative")
    return tuple(
        (flag, description)
        for flag, description in STIS_DQ_FLAGS.items()
        if value & flag
    )


def stis_airglow_wavelength(
    heliocentric_velocity_km_s,
    *,
    rest_wavelength=1215.67,
):
    """Locate geocentric airglow on a heliocentric-corrected STIS grid.

    ``V_HELIO`` is the velocity correction applied by CALSTIS. A terrestrial
    feature therefore appears with the opposite apparent velocity after the
    correction. The first-order Doppler expression is adequate at HST orbital
    and barycentric velocities.
    """
    velocity = float(heliocentric_velocity_km_s)
    rest_wavelength = float(rest_wavelength)
    if not np.isfinite(velocity):
        raise ValueError("heliocentric velocity must be finite")
    if not np.isfinite(rest_wavelength) or rest_wavelength <= 0.0:
        raise ValueError("rest wavelength must be positive and finite")
    return rest_wavelength * (1.0 - velocity / SPEED_OF_LIGHT_KM_S)


def _normalized_grating(value):
    normalized = str(value).strip().upper()
    for separator in ("-", "/", " "):
        normalized = normalized.split(separator, 1)[0]
    return normalized


def _normalized_aperture(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("×", "x")
        .replace('"', "")
        .replace(" ", "")
    )


@dataclass(frozen=True)
class StisLsfCalibration:
    """One tabulated STIS LSF at a reference wavelength."""

    reference_wavelength: float
    resource: str

    def __post_init__(self):
        if (
            not np.isfinite(self.reference_wavelength)
            or self.reference_wavelength <= 0.0
        ):
            raise ValueError("reference_wavelength must be positive and finite")
        if not self.resource:
            raise ValueError("resource must not be empty")

    def as_dict(self):
        return {
            "reference_wavelength_angstrom": self.reference_wavelength,
            "resource": self.resource,
        }


@dataclass(frozen=True)
class StisInstrumentConfiguration:
    """One STIS grating and its aperture-dependent LSF calibrations."""

    grating: str
    calibrations: tuple[StisLsfCalibration, ...]
    aperture_profile_columns: tuple[tuple[str, int], ...]
    source: str = "STScI STIS spectral-resolution reference tables"
    source_url: str = STIS_LSF_SOURCE_URL

    def __post_init__(self):
        grating = _normalized_grating(self.grating)
        if not grating:
            raise ValueError("grating must not be empty")
        calibrations = tuple(self.calibrations)
        if not calibrations:
            raise ValueError("calibrations must not be empty")
        if not all(
            isinstance(calibration, StisLsfCalibration)
            for calibration in calibrations
        ):
            raise TypeError("calibrations must contain StisLsfCalibration values")
        wavelengths = [
            calibration.reference_wavelength for calibration in calibrations
        ]
        if len(wavelengths) != len(set(wavelengths)):
            raise ValueError("calibration reference wavelengths must be unique")
        columns = tuple(
            (_normalized_aperture(aperture), int(column))
            for aperture, column in self.aperture_profile_columns
        )
        names = [aperture for aperture, _ in columns]
        if len(names) != len(set(names)):
            raise ValueError("aperture names must be unique")
        if any(column < 1 for _, column in columns):
            raise ValueError("LSF profile columns must be positive")
        object.__setattr__(self, "grating", grating)
        object.__setattr__(
            self,
            "calibrations",
            tuple(sorted(calibrations, key=lambda item: item.reference_wavelength)),
        )
        object.__setattr__(self, "aperture_profile_columns", columns)

    @property
    def reference_wavelength(self):
        """Lowest tabulated wavelength, retained for API compatibility."""
        return self.calibrations[0].reference_wavelength

    @property
    def lsf_resource(self):
        """Lowest-wavelength resource, retained for API compatibility."""
        return self.calibrations[0].resource

    @property
    def apertures(self):
        return tuple(aperture for aperture, _ in self.aperture_profile_columns)

    def profile_column(self, aperture):
        normalized = _normalized_aperture(aperture)
        for name, column in self.aperture_profile_columns:
            if name == normalized:
                return column
        choices = ", ".join(self.apertures) or "none bundled"
        raise ValueError(
            f"no bundled {self.grating} LSF for aperture {aperture!r}; "
            f"available apertures: {choices}"
        )

    def select_calibration(self, wavelength=None):
        """Return the tabulation nearest the requested wavelength."""
        if wavelength is None:
            return self.calibrations[0]
        wavelength = float(wavelength)
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("wavelength must be positive and finite")
        return min(
            self.calibrations,
            key=lambda item: abs(item.reference_wavelength - wavelength),
        )

    def as_dict(self):
        return {
            "grating": self.grating,
            "calibrations": [
                calibration.as_dict() for calibration in self.calibrations
            ],
            "aperture_profile_columns": dict(
                self.aperture_profile_columns
            ),
            "source": self.source,
            "source_url": self.source_url,
        }


def _calibrations(grating, *reference_wavelengths):
    return tuple(
        StisLsfCalibration(
            float(wavelength),
            f"data/stis_lsf/LSF_{grating}_{int(wavelength)}.txt",
        )
        for wavelength in reference_wavelengths
    )


_FIRST_ORDER_APERTURES = (
    ("52x0.1", 1),
    ("52x0.2", 2),
    ("52x0.5", 3),
    ("52x2.0", 4),
)
_MEDIUM_ECHELLE_APERTURES = (
    ("0.1x0.03", 1),
    ("0.2x0.06", 2),
    ("0.2x0.2", 3),
    ("6x0.2", 4),
)


_STIS_INSTRUMENT_CONFIGURATIONS = (
    StisInstrumentConfiguration(
        "E140M",
        _calibrations("E140M", 1200, 1500),
        _MEDIUM_ECHELLE_APERTURES,
    ),
    StisInstrumentConfiguration(
        "G140M",
        _calibrations("G140M", 1200, 1500),
        _FIRST_ORDER_APERTURES,
    ),
    StisInstrumentConfiguration(
        "G140L",
        _calibrations("G140L", 1200, 1500),
        _FIRST_ORDER_APERTURES,
    ),
    StisInstrumentConfiguration(
        "E140H",
        _calibrations("E140H", 1200, 1500),
        (
            ("0.1x0.03", 1),
            ("0.2x0.09", 2),
            ("0.2x0.2", 3),
            ("6x0.2", 4),
        ),
    ),
    StisInstrumentConfiguration(
        "E230M",
        _calibrations("E230M", 1700, 2400),
        _MEDIUM_ECHELLE_APERTURES,
    ),
    StisInstrumentConfiguration(
        "E230H",
        _calibrations("E230H", 1700, 2400),
        (
            ("0.1x0.03", 1),
            ("0.1x0.09", 2),
            ("0.1x0.2", 3),
            ("6x0.2", 4),
        ),
    ),
    StisInstrumentConfiguration(
        "G230L",
        _calibrations("G230L", 1700, 2400),
        _FIRST_ORDER_APERTURES,
    ),
    StisInstrumentConfiguration(
        "G230M",
        _calibrations("G230M", 1700, 2400),
        _FIRST_ORDER_APERTURES,
    ),
)

STIS_INSTRUMENT_CONFIGURATIONS = MappingProxyType(
    {configuration.grating: configuration
     for configuration in _STIS_INSTRUMENT_CONFIGURATIONS}
)


def get_stis_instrument_configuration(grating):
    """Resolve a STIS grating, accepting central-wavelength suffixes."""
    key = _normalized_grating(grating)
    try:
        return STIS_INSTRUMENT_CONFIGURATIONS[key]
    except KeyError as error:
        choices = ", ".join(STIS_INSTRUMENT_CONFIGURATIONS)
        raise ValueError(
            f"unsupported STIS grating {grating!r}; choices: {choices}"
        ) from error


@dataclass(frozen=True)
class StisTargetSpecification:
    """Reviewed preprocessing choices for one packaged STIS spectrum."""

    name: str
    spectrum_resource: str
    grating: str
    aperture: str
    wavelength_frame: str
    stellar_radial_velocity: float
    continuum_windows: tuple[tuple[float, float], ...]
    fit_window: tuple[float, float]
    uncertainty_scale: float = 1.0
    lsf_profile_column: int | None = None
    excluded_windows: tuple[tuple[float, float], ...] = ()
    program: str = ""
    citation: str = ""
    citation_url: str = ""
    reduction_source: str = ""
    coadded: bool | None = None
    visits: int | None = None
    airglow_subtracted: bool | None = None
    notes: str = ""

    def __post_init__(self):
        for attribute in (
            "name",
            "spectrum_resource",
            "grating",
            "aperture",
            "wavelength_frame",
        ):
            if not getattr(self, attribute):
                raise ValueError(f"{attribute} must not be empty")
        if not np.isfinite(self.stellar_radial_velocity):
            raise ValueError("stellar_radial_velocity must be finite")
        if not np.isfinite(self.uncertainty_scale) or self.uncertainty_scale <= 0.0:
            raise ValueError("uncertainty_scale must be positive and finite")
        if self.lsf_profile_column is not None and self.lsf_profile_column < 1:
            raise ValueError("lsf_profile_column must select a profile column")
        if self.visits is not None and self.visits < 1:
            raise ValueError("visits must be positive or None")
        for name, windows in (
            ("continuum_windows", self.continuum_windows),
            ("excluded_windows", self.excluded_windows),
        ):
            for lower, upper in windows:
                if lower >= upper:
                    raise ValueError(f"{name} lower bounds must precede upper bounds")
        if self.fit_window[0] >= self.fit_window[1]:
            raise ValueError("fit_window lower bound must precede upper bound")

    def as_dict(self):
        instrument = get_stis_instrument_configuration(self.grating)
        calibration = instrument.select_calibration(
            0.5 * (self.fit_window[0] + self.fit_window[1])
        )
        profile_column = (
            instrument.profile_column(self.aperture)
            if self.lsf_profile_column is None
            else self.lsf_profile_column
        )
        return {
            "name": self.name,
            "spectrum_resource": self.spectrum_resource,
            "grating": self.grating,
            "aperture": self.aperture,
            "wavelength_frame": self.wavelength_frame,
            "stellar_radial_velocity_km_s": self.stellar_radial_velocity,
            "continuum_windows": [list(window) for window in self.continuum_windows],
            "fit_window": list(self.fit_window),
            "uncertainty_scale": self.uncertainty_scale,
            "uncertainty_scale_provenance": (
                "Tom Ayres reduction recommendation"
                if np.isclose(self.uncertainty_scale, AYRES_PHOTERR_SCALE)
                else "target specification"
            ),
            "lsf_profile_column": profile_column,
            "lsf_resource": calibration.resource,
            "lsf_reference_wavelength_angstrom": (
                calibration.reference_wavelength
            ),
            "excluded_windows": [list(window) for window in self.excluded_windows],
            "program": self.program,
            "citation": self.citation,
            "citation_url": self.citation_url,
            "reduction_source": self.reduction_source,
            "coadded": self.coadded,
            "visits": self.visits,
            "airglow_subtracted": self.airglow_subtracted,
            "notes": self.notes,
        }


GJ411_E140M_TARGET = StisTargetSpecification(
    name="GJ 411",
    spectrum_resource=GJ411_SPECTRUM_RESOURCE,
    grating="E140M-1425",
    aperture="0.2x0.2",
    wavelength_frame="heliocentric",
    stellar_radial_velocity=-87.0,
    continuum_windows=GJ411_CONTINUUM_WINDOWS,
    fit_window=GJ411_FIT_WINDOW,
    uncertainty_scale=AYRES_PHOTERR_SCALE,
    lsf_profile_column=3,
    excluded_windows=((1215.5475, 1215.6325),),
    reduction_source="Tom Ayres custom STIS reduction",
    notes=(
        "Tom Ayres STIS reduction. The exclusion spans two Gaussian sigma "
        "around the historical 1215.59 Angstrom, 0.05 Angstrom FWHM "
        "airglow model."
    ),
)


GJ729_E140M_TARGET = StisTargetSpecification(
    name="GJ 729",
    spectrum_resource=GJ729_SPECTRUM_RESOURCE,
    grating="E140M-1425",
    aperture="0.2x0.2",
    wavelength_frame="heliocentric",
    stellar_radial_velocity=-11.0,
    continuum_windows=((1190.0, 1200.0), (1250.0, 1260.0)),
    fit_window=(1214.8, 1216.3),
    uncertainty_scale=AYRES_PHOTERR_SCALE,
    lsf_profile_column=3,
    excluded_windows=((1215.699, 1215.801),),
    program="HST GO 15071",
    citation="Wilson et al. 2025, ApJ, 978, 85",
    citation_url="https://scixplorer.org/abs/2025ApJ...978...85W/abstract",
    reduction_source="Tom Ayres custom STIS reduction",
    coadded=True,
    visits=2,
    airglow_subtracted=False,
    notes=(
        "Tom Ayres STIS reduction; multiple orbits across two visits were "
        "coadded; no airglow subtraction was applied. The exclusion spans "
        "two Gaussian sigma around the historical 1215.75 Angstrom, "
        "0.06 Angstrom FWHM airglow model. The undefined epsilon array is "
        "retained in the source file but is not used as a quality mask."
    ),
)


E140M_TARGETS = {
    target.name: target
    for target in (GJ411_E140M_TARGET, GJ729_E140M_TARGET)
}


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
    quality_mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    data_quality: np.ndarray | None = None

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
        quality_mask = (
            np.ones(wavelength.shape, dtype=bool)
            if self.quality_mask is None
            else np.asarray(self.quality_mask, dtype=bool)
        )
        if quality_mask.shape != wavelength.shape:
            raise ValueError("quality_mask must match the wavelength shape")
        if not np.any(quality_mask):
            raise ValueError("quality_mask must include at least one pixel")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        data_quality = (
            np.zeros(wavelength.shape, dtype=np.int64)
            if self.data_quality is None
            else np.asarray(self.data_quality, dtype=np.int64)
        )
        if data_quality.shape != wavelength.shape:
            raise ValueError("data_quality must match the wavelength shape")
        if np.any(data_quality < 0):
            raise ValueError("data_quality values must be non-negative")
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "quality_mask", quality_mask)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "data_quality", data_quality)


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
    likelihood_mask: np.ndarray | None = None
    uncertainty_scale: float = 1.0
    metadata: dict = field(default_factory=dict)

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
        likelihood_mask = (
            np.ones(wavelength.shape, dtype=bool)
            if self.likelihood_mask is None
            else np.asarray(self.likelihood_mask, dtype=bool)
        )
        if likelihood_mask.shape != wavelength.shape:
            raise ValueError("likelihood_mask must match the wavelength shape")
        if not np.any(likelihood_mask):
            raise ValueError("likelihood_mask must include at least one pixel")
        if not np.isfinite(self.uncertainty_scale) or self.uncertainty_scale <= 0.0:
            raise ValueError("uncertainty_scale must be positive and finite")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "convolution_kernel", kernel)
        object.__setattr__(self, "continuum_coefficients", coefficients)
        object.__setattr__(self, "likelihood_mask", likelihood_mask)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def raw_error(self):
        return self.error / self.uncertainty_scale


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


def _stis_header_metadata(primary_header, science_header, path, spectral_order):
    heliocentric_correction = primary_header.get("HELCORR", "").strip().upper()
    return {
        "source_format": "STIS x1d",
        "filename": Path(path).name,
        "rootname": primary_header.get("ROOTNAME"),
        "target_name": primary_header.get("TARGNAME"),
        "program": primary_header.get("PROPOSID"),
        "instrument": primary_header.get("INSTRUME"),
        "detector": primary_header.get("DETECTOR"),
        "grating": primary_header.get("OPT_ELEM"),
        "central_wavelength_angstrom": primary_header.get("CENWAVE"),
        "aperture": primary_header.get("APERTURE"),
        "wavelength_frame": (
            "heliocentric"
            if heliocentric_correction == "COMPLETE"
            else "pipeline"
        ),
        "heliocentric_correction": heliocentric_correction or None,
        "heliocentric_velocity_km_s": science_header.get("V_HELIO"),
        "date_obs": science_header.get("DATE-OBS"),
        "time_obs": science_header.get("TIME-OBS"),
        "exposure_start_mjd": science_header.get("EXPSTART"),
        "exposure_end_mjd": science_header.get("EXPEND"),
        "exposure_time_seconds": science_header.get("EXPTIME"),
        "association_id": primary_header.get("ASN_ID"),
        "observation_set": primary_header.get("OBSET_ID"),
        "calibration_version": primary_header.get("CAL_VER"),
        "spectral_order": spectral_order,
        "coadded": False,
    }


def load_stis_x1d(
    path,
    *,
    row=None,
    wavelength=None,
    name=None,
    dq_bad_bits=None,
    metadata_overrides=None,
):
    """Load one calibrated STIS x1d/sx1 spectral row.

    A multi-order echelle product requires either ``row`` or a wavelength used
    to select the unique order containing it. By default, DQ bits listed in the
    SCI extension's ``SDQFLAGS`` value and pixels with non-positive errors are
    excluded through ``ObservedSpectrum.quality_mask``; pixels remain on the
    wavelength grid for later convolution and bin integration.
    """
    path = Path(path)
    with fits.open(path, memmap=False) as hdus:
        primary_header = hdus[0].header
        if str(primary_header.get("INSTRUME", "")).strip().upper() != "STIS":
            raise ValueError("FITS file is not a STIS product")
        try:
            science = hdus["SCI"]
        except KeyError as error:
            raise ValueError("STIS FITS file has no SCI extension") from error
        if science.data is None or len(science.data) == 0:
            raise ValueError("STIS SCI extension contains no spectral rows")
        required = {"NELEM", "WAVELENGTH", "FLUX", "ERROR", "DQ"}
        columns = set(science.columns.names or ())
        missing = required - columns
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"STIS SCI table is missing columns: {names}")

        if row is not None and wavelength is not None:
            raise ValueError("pass either row or wavelength, not both")
        if row is None and wavelength is not None:
            matches = []
            for index, candidate in enumerate(science.data):
                count = int(candidate["NELEM"])
                candidate_wavelength = np.asarray(
                    candidate["WAVELENGTH"][:count],
                    dtype=np.float64,
                )
                if (
                    candidate_wavelength.size
                    and candidate_wavelength[0]
                    <= wavelength
                    <= candidate_wavelength[-1]
                ):
                    matches.append(index)
            if len(matches) != 1:
                raise ValueError(
                    "wavelength must select exactly one STIS spectral row"
                )
            row = matches[0]
        elif row is None:
            if len(science.data) != 1:
                raise ValueError(
                    "multi-order STIS product requires row or wavelength"
                )
            row = 0
        if row < 0 or row >= len(science.data):
            raise IndexError("STIS spectral row is outside the SCI table")

        record = science.data[row]
        count = int(record["NELEM"])
        if count < 2:
            raise ValueError("STIS spectral row must contain at least two pixels")
        wavelength_values = np.asarray(
            record["WAVELENGTH"][:count],
            dtype=np.float64,
        )
        flux = np.asarray(record["FLUX"][:count], dtype=np.float64)
        error = np.asarray(record["ERROR"][:count], dtype=np.float64)
        dq = np.asarray(record["DQ"][:count], dtype=np.int64)
        if dq_bad_bits is None:
            dq_bad_bits = int(science.header.get("SDQFLAGS", 0))
        if dq_bad_bits < 0:
            raise ValueError("dq_bad_bits must be non-negative")
        quality_mask = (error > 0.0) & ((dq & int(dq_bad_bits)) == 0)
        spectral_order = (
            int(record["SPORDER"])
            if "SPORDER" in columns
            else None
        )
        metadata = _stis_header_metadata(
            primary_header,
            science.header,
            path,
            spectral_order,
        )
        metadata.update(
            {
                "science_row": int(row),
                "pixels": int(count),
                "dq_bad_bits": int(dq_bad_bits),
                "extraction_center_a2_pixels": (
                    float(record["A2CENTER"])
                    if "A2CENTER" in columns
                    else None
                ),
                "extraction_box_height_pixels": (
                    float(record["EXTRSIZE"])
                    if "EXTRSIZE" in columns
                    else None
                ),
                "extraction_max_search_pixels": (
                    int(record["MAXSRCH"])
                    if "MAXSRCH" in columns
                    else None
                ),
                "extraction_offset_pixels": (
                    float(record["OFFSET"])
                    if "OFFSET" in columns
                    else None
                ),
                "excluded_quality_pixels": int(
                    np.count_nonzero(~quality_mask)
                ),
                "dq_flag_counts": {
                    str(flag): int(np.count_nonzero(dq & flag))
                    for flag in STIS_DQ_FLAGS
                    if np.any(dq & flag)
                },
                "flux_unit": science.columns["FLUX"].unit,
                "error_unit": science.columns["ERROR"].unit,
                "wavelength_unit": science.columns["WAVELENGTH"].unit,
            }
        )
        if metadata_overrides is not None:
            if not isinstance(metadata_overrides, dict):
                raise TypeError("metadata_overrides must be a dictionary")
            metadata.update(metadata_overrides)

    return ObservedSpectrum(
        wavelength_values,
        flux,
        error,
        name=(
            str(metadata.get("target_name") or path.stem)
            if name is None
            else name
        ),
        quality_mask=quality_mask,
        metadata=metadata,
        data_quality=dq,
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


def load_stis_lsf(
    grating,
    aperture,
    *,
    wavelength=None,
    profile_column=None,
    path=None,
):
    """Load the nearest bundled or an explicit aperture-specific STIS LSF.

    Table offsets are retained in detector-pixel units. When ``path`` is not
    supplied, ``wavelength`` selects the nearest tabulated STScI reference
    wavelength. A user-supplied table overrides the bundled calibration.
    """
    configuration = get_stis_instrument_configuration(grating)
    if profile_column is None:
        profile_column = configuration.profile_column(aperture)
    if path is not None:
        return load_tabulated_lsf(path, profile_column=profile_column)
    calibration = configuration.select_calibration(wavelength)
    package = resources.files("lyapy")
    with resources.as_file(
        package.joinpath(calibration.resource)
    ) as resource_path:
        return load_tabulated_lsf(
            resource_path,
            profile_column=profile_column,
        )


def fit_continuum(spectrum, windows, degree=1):
    """Fit a polynomial continuum within open wavelength windows."""
    if degree < 0:
        raise ValueError("continuum degree must be non-negative")
    mask = np.zeros(spectrum.wavelength.shape, dtype=bool)
    for lower, upper in windows:
        if lower >= upper:
            raise ValueError("continuum-window lower bounds must precede upper bounds")
        mask |= (spectrum.wavelength > lower) & (spectrum.wavelength < upper)
    mask &= spectrum.quality_mask
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
    uncertainty_scale=1.0,
    excluded_windows=(),
    metadata=None,
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
    error = spectrum.error[fit_mask] * uncertainty_scale
    likelihood_mask = spectrum.quality_mask[fit_mask].copy()
    likelihood_mask &= error > 0.0
    if not np.any(likelihood_mask):
        raise ValueError("fit window contains no usable positive-error pixels")
    if np.any(error <= 0.0):
        error = error.copy()
        error[error <= 0.0] = float(np.median(error[likelihood_mask]))
    if grating_dispersion is None:
        grating_dispersion = wavelength[1] - wavelength[0]
    kernel = interpolate_stis_lsf(
        lsf.offset,
        lsf.profile,
        grating_dispersion,
        wavelength,
    )
    for excluded_lower, excluded_upper in excluded_windows:
        if excluded_lower >= excluded_upper:
            raise ValueError(
                "excluded-window lower bounds must precede upper bounds"
            )
        likelihood_mask &= ~(
            (wavelength >= excluded_lower) & (wavelength <= excluded_upper)
        )
    return PreparedObservation(
        wavelength=wavelength,
        flux=spectrum.flux[fit_mask] - continuum[fit_mask],
        error=error,
        convolution_kernel=kernel,
        continuum_coefficients=coefficients,
        name=spectrum.name,
        likelihood_mask=likelihood_mask,
        uncertainty_scale=float(uncertainty_scale),
        metadata={
            **spectrum.metadata,
            **({} if metadata is None else metadata),
        },
    )


def load_stis_target_observation(target):
    """Load a packaged spectrum using one reviewed target specification."""
    if not isinstance(target, StisTargetSpecification):
        raise TypeError("target must be a StisTargetSpecification")
    package = resources.files("lyapy")
    with resources.as_file(package.joinpath(target.spectrum_resource)) as path:
        spectrum = load_idl_spectrum(path, name=target.name)
    lsf = load_stis_lsf(
        target.grating,
        target.aperture,
        wavelength=0.5 * (target.fit_window[0] + target.fit_window[1]),
        profile_column=target.lsf_profile_column,
    )
    return prepare_stis_observation(
        spectrum,
        lsf,
        continuum_windows=target.continuum_windows,
        fit_window=target.fit_window,
        uncertainty_scale=target.uncertainty_scale,
        excluded_windows=target.excluded_windows,
        metadata=target.as_dict(),
    )


def load_gj411_observation():
    """Load GJ 411 with the reviewed Ayres uncertainty and airglow mask."""
    return load_stis_target_observation(GJ411_E140M_TARGET)


def load_gj729_observation():
    """Load GJ 729 using its reviewed E140M preprocessing specification."""
    return load_stis_target_observation(GJ729_E140M_TARGET)
